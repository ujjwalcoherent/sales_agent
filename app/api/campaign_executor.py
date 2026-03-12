"""Campaign executor — processes companies through enrich -> contacts -> outreach.

Supports 3 campaign types:
  - company_first: Companies provided upfront
  - industry_first: Discover companies via web search, then company_first flow
  - report_driven: Extract company names from report text via LLM, then company_first flow

Entry point: execute_campaign(campaign_id, event_queue)
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_RE_DIGITS = re.compile(r"\d+")

_ENRICH_TIMEOUT = 90.0
_CONTACTS_TIMEOUT = 120.0
_OUTREACH_TIMEOUT = 300.0  # 5 min — campaign_mode person_intel is fast (10s vs 25s)
_DISCOVERY_TIMEOUT = 60.0
_MAX_CONCURRENT_COMPANIES = 2
_UPDATE_LOCK = asyncio.Lock()


def _company_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()[:12]


async def execute_campaign(campaign_id: str, event_queue: asyncio.Queue) -> None:
    """Run a full campaign: discover (if needed) -> enrich -> contacts -> outreach.

    Loads campaign from DB, processes each company with bounded concurrency,
    pushes SSE events for real-time progress, and updates DB throughout.
    """
    from app.database import get_database
    from app.schemas.campaign import CampaignType, CampaignCompanyInput, CampaignConfig

    db = get_database()
    campaign = db.get_campaign(campaign_id)
    if not campaign:
        await event_queue.put({"event": "campaign_error", "campaign_id": campaign_id, "error": "Campaign not found"})
        return

    config = CampaignConfig(**campaign.get("config", {}))
    campaign_type = campaign.get("campaign_type", "company_first")

    try:
        db.update_campaign(campaign_id, {"status": "running"})

        # Phase 1: resolve company list based on campaign type
        if campaign_type == CampaignType.INDUSTRY_FIRST:
            industry = campaign.get("config", {}).get("industry", "")
            if not industry:
                raise ValueError("industry_first campaign requires 'industry' in config")
            company_inputs = await asyncio.wait_for(
                _discover_industry_companies(industry, config), timeout=_DISCOVERY_TIMEOUT,
            )
        elif campaign_type == CampaignType.REPORT_DRIVEN:
            report_text = campaign.get("config", {}).get("report_text", "")
            if not report_text:
                raise ValueError("report_driven campaign requires 'report_text' in config")
            company_inputs = await asyncio.wait_for(
                _extract_report_companies(report_text, config), timeout=_DISCOVERY_TIMEOUT,
            )
        else:
            raw_companies = campaign.get("companies", [])
            company_inputs = [
                CampaignCompanyInput(**c) if isinstance(c, dict) else c
                for c in raw_companies
            ]

        if not company_inputs:
            raise ValueError("No companies to process")

        total = len(company_inputs)
        companies_status = [
            {"company_name": ci.company_name, "domain": ci.domain, "industry": ci.industry, "status": "pending"}
            for ci in company_inputs
        ]
        db.update_campaign(campaign_id, {"companies": companies_status, "total_companies": total})
        await event_queue.put({"event": "campaign_start", "campaign_id": campaign_id, "total": total})

        # Phase 2: process each company (bounded concurrency, per-campaign semaphore)
        total_contacts = 0
        total_outreach = 0
        completed = 0
        company_sem = asyncio.Semaphore(_MAX_CONCURRENT_COMPANIES)

        async def _run_one(idx: int, ci: CampaignCompanyInput) -> tuple:
            async with company_sem:
                return await _process_one_company(ci, campaign_id, idx, total, config, event_queue)

        results = await asyncio.gather(
            *[_run_one(i, ci) for i, ci in enumerate(company_inputs)],
            return_exceptions=True,
        )
        for i, res in enumerate(results):
            if isinstance(res, tuple):
                total_contacts += res[0]
                total_outreach += res[1]
            elif isinstance(res, Exception):
                logger.warning(f"Company task {i} raised: {res}")
            completed += 1

        # Phase 3: finalize
        db.update_campaign(campaign_id, {
            "status": "completed", "completed_companies": completed,
            "total_contacts": total_contacts, "total_outreach": total_outreach,
            "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })
        await event_queue.put({
            "event": "campaign_done", "campaign_id": campaign_id,
            "total_contacts": total_contacts, "total_outreach": total_outreach,
        })
    except Exception as e:
        logger.exception(f"Campaign {campaign_id} failed: {e}")
        db.update_campaign(campaign_id, {"status": "failed", "error": str(e)[:500]})
        await event_queue.put({"event": "campaign_error", "campaign_id": campaign_id, "error": str(e)[:500]})


# ── Discovery helpers ────────────────────────────────────────────

async def _discover_industry_companies(industry: str, config) -> list:
    from app.tools.web.web_intel import search_industry_companies
    from app.schemas.campaign import CampaignCompanyInput

    # Combine broad industry with narrow keyword if provided
    search_query = industry
    if getattr(config, "narrow_keyword", ""):
        search_query = f"{industry} {config.narrow_keyword}"

    profiles = await search_industry_companies(search_query, max_companies=config.max_companies)

    # Filter by company size if requested
    size_filter = getattr(config, "company_size_filter", "all")
    _SIZE_RANGES = {
        "smb": (0, 200),
        "mid_market": (200, 1000),
        "enterprise": (1000, 10**9),
    }

    def _size_matches(profile) -> bool:
        if size_filter == "all":
            return True
        lo, hi = _SIZE_RANGES.get(size_filter, (0, 10**9))
        ec = getattr(profile, "employee_count", "") or ""
        if isinstance(ec, str):
            m = _RE_DIGITS.search(ec)
            n = int(m.group()) if m else None
        else:
            n = ec
        if n is None:
            return True  # unknown size — include by default
        return lo <= n < hi

    return [
        CampaignCompanyInput(
            company_name=p.company_name, domain=p.domain or "",
            industry=p.industry or industry,
            context=f"Discovered via {search_query!r} industry search",
        )
        for p in profiles if p.company_name and _size_matches(p)
    ]


async def _extract_report_companies(report_text: str, config) -> list:
    from app.tools.llm.llm_service import LLMService
    from app.schemas.campaign import CampaignCompanyInput

    llm = LLMService(lite=True)
    prompt = (
        "Extract company names mentioned in this report. For each company, "
        "provide the name and a one-sentence context about why it was mentioned.\n\n"
        "Return as JSON array: [{\"name\": \"...\", \"context\": \"...\"}]\n\n"
        f"Report:\n{report_text[:8000]}"
    )
    raw = await asyncio.wait_for(llm.generate(prompt, temperature=0.1, max_tokens=1000), timeout=15.0)
    if not raw:
        return []

    try:
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            entries = json.loads(text[start:end + 1])
        else:
            return []
    except (json.JSONDecodeError, IndexError):
        logger.warning("Failed to parse report company extraction")
        return []

    results = []
    for entry in entries[:config.max_companies]:
        name = entry.get("name", "").strip()
        if name and len(name) >= 2:
            results.append(CampaignCompanyInput(
                company_name=name, context=entry.get("context", "Mentioned in report"),
            ))
    return results


# ── Single-company processor ─────────────────────────────────────

async def _process_one_company(company_input, campaign_id, index, total, config, event_queue):
    from app.database import get_database
    from app.tools.company_enricher import enrich
    from app.agents.workers.contact_agent import ContactFinder
    from app.agents.workers.email_agent import EmailGenerator
    from app.schemas.sales import AgentState, CompanyData
    from app.agents.deps import AgentDeps

    db = get_database()
    name = company_input.company_name
    contacts_found = outreach_generated = 0
    await event_queue.put({"event": "company_start", "company": name, "index": index, "total": total})

    try:
        # Step 1: Enrich
        await _update_company_status(db, campaign_id, name, "enriching")
        enriched = await asyncio.wait_for(
            enrich(name, domain=company_input.domain, skip_validation=False,
                   background_deep=config.background_deep),
            timeout=_ENRICH_TIMEOUT,
        )
        if not enriched:
            await _update_company_status(db, campaign_id, name, "failed", error="Entity validation failed")
            await event_queue.put({"event": "company_error", "company": name, "error": "Entity validation failed"})
            return 0, 0

        domain = enriched.domain or ""
        industry = enriched.industry or company_input.industry or ""
        await event_queue.put({"event": "company_enriched", "company": name, "domain": domain, "industry": industry})
        await _update_company_status(db, campaign_id, name, "enriched",
                               domain=domain, industry=industry, description=enriched.description or "")

        # Step 2: Find contacts
        await _update_company_status(db, campaign_id, name, "contacts")
        company_data = CompanyData(
            id=_company_id(name), company_name=enriched.company_name,
            industry=industry, domain=domain, website=enriched.website or "",
            reason_relevant=company_input.context or f"Campaign target: {name}",
        )
        # Use global mock mode if enabled
        from app.api.settings_router import get_global_mock_mode
        _mock = get_global_mock_mode()

        # Pre-set target_roles on company_data so contact agent uses them directly
        if config.target_roles:
            company_data.target_roles = list(config.target_roles)

        deps = AgentDeps(mock_mode=_mock)
        state = AgentState(companies=[company_data])
        finder = ContactFinder(mock_mode=_mock, deps=deps)
        state = await asyncio.wait_for(finder.find_contacts(state), timeout=_CONTACTS_TIMEOUT)
        contacts = state.contacts or []

        # Apply seniority filter if requested
        seniority = getattr(config, "seniority_filter", "both")
        if seniority != "both" and contacts:
            contacts = [c for c in contacts if getattr(c, "seniority_tier", "influencer") == seniority]
            # Fallback: if filtering removed all contacts, keep originals
            if not contacts:
                contacts = state.contacts or []

        contacts_found = len(contacts)
        # Serialize contact data for campaign storage
        contact_dicts = []
        for c in contacts:
            contact_dicts.append({
                "full_name": getattr(c, "person_name", "") or getattr(c, "full_name", ""),
                "role": getattr(c, "role", ""),
                "email": getattr(c, "email", ""),
                "linkedin_url": getattr(c, "linkedin_url", ""),
                "seniority": getattr(c, "seniority_tier", "") or getattr(c, "seniority", ""),
                "email_confidence": float(getattr(c, "email_confidence", 0) or 0),
            })
        await event_queue.put({"event": "company_contacts", "company": name, "contacts_found": contacts_found})
        await _update_company_status(db, campaign_id, name, "contacts_done",
                                     contacts_found=contacts_found, contacts=contact_dicts)

        # Step 3: Generate outreach
        email_dicts = []
        if config.generate_outreach and contacts:
            await _update_company_status(db, campaign_id, name, "outreach")
            generator = EmailGenerator(mock_mode=_mock, deps=deps, campaign_mode=True)
            # Inject product context into company reason_relevant for personalization
            if getattr(config, "product_context", ""):
                company_data.reason_relevant = (
                    f"{company_data.reason_relevant} | Pitching: {config.product_context}"
                ).strip(" |")
            state = AgentState(companies=[company_data], contacts=contacts)
            state = await asyncio.wait_for(generator.process_emails(state), timeout=_OUTREACH_TIMEOUT)
            outreach_emails = state.outreach_emails or []
            outreach_generated = len(outreach_emails)
            # Serialize email data for campaign storage
            email_dicts = []
            for e in outreach_emails:
                email_dicts.append({
                    "recipient_name": getattr(e, "person_name", "") or getattr(e, "recipient_name", ""),
                    "recipient_role": getattr(e, "role", "") or getattr(e, "recipient_role", ""),
                    "subject": getattr(e, "subject", ""),
                    "body": getattr(e, "body", ""),
                })
            await event_queue.put({"event": "company_outreach", "company": name, "outreach_generated": outreach_generated})

        # Step 4: Mark done (final totals set in Phase 3 of execute_campaign)
        await _update_company_status(db, campaign_id, name, "done",
                                     contacts_found=contacts_found, outreach_generated=outreach_generated,
                                     contacts=contact_dicts, emails=email_dicts if config.generate_outreach else None)

        # Step 5: Cross-save contacts to saved_companies table
        # This ensures contacts are visible on the company detail page too
        if contact_dicts:
            try:
                company_hash = hashlib.md5(name.lower().encode()).hexdigest()[:12]
                # Build PersonResponse-compatible dicts for save_company_contacts
                person_dicts = []
                for cd in contact_dicts:
                    pd: dict = {
                        "person_name": cd.get("full_name", ""),
                        "role": cd.get("role", ""),
                        "email": cd.get("email", ""),
                        "linkedin_url": cd.get("linkedin_url", ""),
                        "seniority_tier": cd.get("seniority", ""),
                        "email_confidence": cd.get("email_confidence", 0),
                        "verified": False,
                        "reach_score": 0,
                    }
                    # Match outreach email if available
                    for ed in email_dicts:
                        if (ed.get("recipient_name", "").lower() == cd.get("full_name", "").lower()):
                            pd["outreach_subject"] = ed.get("subject", "")
                            pd["outreach_body"] = ed.get("body", "")
                            break
                    person_dicts.append(pd)
                db.save_company_contacts(
                    company_hash, person_dicts,
                    reasoning=f"Campaign {campaign_id} — {contacts_found} contacts, {outreach_generated} emails",
                )
            except Exception as e:
                logger.debug(f"Cross-save contacts to saved_companies failed for '{name}': {e}")

        await event_queue.put({"event": "company_done", "company": name, "index": index})
        return contacts_found, outreach_generated

    except asyncio.TimeoutError:
        logger.warning(f"Company '{name}' timed out in campaign {campaign_id}")
        await _update_company_status(db, campaign_id, name, "failed", error="Processing timed out")
        await event_queue.put({"event": "company_error", "company": name, "error": "Processing timed out"})
        return 0, 0
    except Exception as e:
        error_msg = str(e)[:300]
        logger.warning(f"Company '{name}' failed in campaign {campaign_id}: {e}")
        await _update_company_status(db, campaign_id, name, "failed", error=error_msg)
        await event_queue.put({"event": "company_error", "company": name, "error": error_msg})
        return 0, 0


# ── DB helper ────────────────────────────────────────────────────

async def _update_company_status(db, campaign_id: str, company_name: str, status: str, *,
                                 domain: str = "", industry: str = "", description: str = "",
                                 contacts_found: int = 0, outreach_generated: int = 0,
                                 contacts: list | None = None, emails: list | None = None,
                                 error: str = "") -> None:
    """Thread-safe company status update — serialized via _UPDATE_LOCK."""
    async with _UPDATE_LOCK:
        campaign = db.get_campaign(campaign_id)
        if not campaign:
            return
        companies = campaign.get("companies", [])
        updated = False
        for entry in companies:
            if entry.get("company_name") == company_name:
                entry["status"] = status
                if domain: entry["domain"] = domain
                if industry: entry["industry"] = industry
                if description: entry["description"] = description
                if contacts_found: entry["contacts_found"] = contacts_found
                if outreach_generated: entry["outreach_generated"] = outreach_generated
                if contacts is not None: entry["contacts"] = contacts
                if emails is not None: entry["emails"] = emails
                if error: entry["error"] = error
                updated = True
                break
        if not updated:
            companies.append({
                "company_name": company_name, "status": status, "domain": domain,
                "industry": industry, "description": description,
                "contacts_found": contacts_found, "outreach_generated": outreach_generated,
                "contacts": contacts or [], "emails": emails or [],
                "error": error,
            })
        db.update_campaign(campaign_id, {"companies": companies})
