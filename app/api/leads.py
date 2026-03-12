"""Leads API router -- returns crystallized lead sheets from pipeline runs.

Data source priority:
  1. In-memory RunManager (active/recent runs with full state)
  2. Database (persisted call sheets from completed runs)
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.api.pipeline import run_manager
from app.api.schemas import LeadListResponse, LeadResponse, PersonResponse
from app.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter()


def _leads_from_run(run) -> list:
    """Extract LeadResponse list from a PipelineRun (in-memory).

    Supports two data paths:
      1. Real runs: deps._lead_sheets (pydantic objects) in final_state
      2. Replay runs: run.result["leads"] (dicts from JSON recordings)
    """
    if run is None:
        return []

    # Path 1: replay result (dicts from recorded JSON)
    if run.result and isinstance(run.result, dict) and run.result.get("replay"):
        sheets = run.result.get("leads", [])
        # Build enrichment indexes from recording data
        _company_by_name: dict[str, dict] = {}
        for c in run.result.get("companies", []):
            _company_by_name[c.get("company_name", "")] = c
        _contacts_by_company: dict[str, list] = {}
        for ct in run.result.get("contacts", []):
            cname = ct.get("company_name", "")
            _contacts_by_company.setdefault(cname, []).append(ct)
        _outreach_by_company: dict[str, dict] = {}
        for em in run.result.get("outreach", []):
            cname = em.get("company_name", "")
            if cname not in _outreach_by_company:
                _outreach_by_company[cname] = em
        _profiles_by_company: dict[str, list] = {}
        for p in run.result.get("people", []):
            cname = p.get("company_name", "")
            _profiles_by_company.setdefault(cname, []).append(p)

        results = []
        for i, s in enumerate(sheets):
            cname = s.get("company_name", "")
            company = _company_by_name.get(cname, {})
            contacts = _contacts_by_company.get(cname, [])
            outreach = _outreach_by_company.get(cname, {})
            contact = next((c for c in contacts if c.get("email")), contacts[0] if contacts else {})
            results.append(LeadResponse(
                id=i,
                company_name=cname,
                company_cin=s.get("company_cin", ""),
                company_state=s.get("company_state", ""),
                company_city=s.get("company_city", ""),
                company_size_band=s.get("company_size_band", ""),
                company_website=company.get("website", ""),
                company_domain=company.get("domain", ""),
                reason_relevant=company.get("reason_relevant", ""),
                hop=s.get("hop", 1),
                lead_type=s.get("lead_type", ""),
                trend_title=s.get("trend_title", ""),
                event_type=s.get("event_type", ""),
                contact_name=contact.get("person_name", ""),
                contact_role=contact.get("role", "") or s.get("contact_role", ""),
                contact_email=contact.get("email", ""),
                contact_linkedin=contact.get("linkedin_url", ""),
                email_confidence=int(contact.get("email_confidence", 0)),
                email_subject=outreach.get("subject", ""),
                email_body=outreach.get("body", ""),
                trigger_event=s.get("trigger_event", ""),
                pain_point=s.get("pain_point", ""),
                service_pitch=s.get("service_pitch", ""),
                opening_line=s.get("opening_line", ""),
                urgency_weeks=s.get("urgency_weeks", 4),
                confidence=s.get("confidence", 0.0),
                oss_score=s.get("oss_score", 0.0),
                data_sources=s.get("data_sources", []),
                company_news=s.get("company_news", []),
                people=[
                    PersonResponse(
                        person_name=p.get("person_name", ""),
                        role=p.get("role", ""),
                        seniority_tier=p.get("seniority_tier", "influencer"),
                        linkedin_url=p.get("linkedin_url", ""),
                        email=p.get("email", ""),
                        email_confidence=int(p.get("email_confidence", 0)),
                        verified=p.get("verified", False),
                        reach_score=int(p.get("reach_score", 0)),
                        outreach_tone=p.get("outreach_tone", "consultative"),
                        outreach_subject=p.get("outreach_subject", ""),
                        outreach_body=p.get("outreach_body", ""),
                    )
                    for p in _profiles_by_company.get(cname, [])
                ],
            ))
        return results

    # Path 2: real run with deps._lead_sheets (pydantic objects)
    if run.final_state is None:
        return []
    deps = run.final_state.get("deps")
    if deps is None:
        return []

    # Build enrichment indexes (contacts, companies, emails from lead_gen)
    _company_by_name = {}
    for c in getattr(deps, "_companies", []):
        _company_by_name[getattr(c, "company_name", "")] = c
    _contacts_by_company = {}
    for ct in getattr(deps, "_contacts", []):
        cname = getattr(ct, "company_name", "")
        _contacts_by_company.setdefault(cname, []).append(ct)
    _outreach_by_company = {}
    for em in getattr(deps, "_outreach", []):
        cname = getattr(em, "company_name", "")
        if cname not in _outreach_by_company:
            _outreach_by_company[cname] = em
    _profiles_by_company = {}
    for p in getattr(deps, "_person_profiles", []):
        cname = getattr(p, "company_name", "")
        _profiles_by_company.setdefault(cname, []).append(p)

    sheets = getattr(deps, "_lead_sheets", [])
    results = []
    for i, sheet in enumerate(sheets):
        cname = getattr(sheet, "company_name", "")
        company = _company_by_name.get(cname)
        contacts = _contacts_by_company.get(cname, [])
        outreach = _outreach_by_company.get(cname)
        contact = next((c for c in contacts if getattr(c, "email", "")), contacts[0] if contacts else None)
        results.append(LeadResponse(
            id=i,
            company_name=cname,
            company_cin=getattr(sheet, "company_cin", ""),
            company_state=getattr(sheet, "company_state", ""),
            company_city=getattr(sheet, "company_city", ""),
            company_size_band=getattr(sheet, "company_size_band", ""),
            company_website=getattr(company, "website", "") if company else "",
            company_domain=getattr(company, "domain", "") if company else "",
            reason_relevant=getattr(company, "reason_relevant", "") if company else "",
            hop=getattr(sheet, "hop", 1),
            lead_type=getattr(sheet, "lead_type", ""),
            trend_title=getattr(sheet, "trend_title", ""),
            event_type=getattr(sheet, "event_type", ""),
            contact_name=getattr(contact, "person_name", "") if contact else "",
            contact_role=getattr(contact, "role", "") if contact else getattr(sheet, "contact_role", ""),
            contact_email=getattr(contact, "email", "") if contact else "",
            contact_linkedin=getattr(contact, "linkedin_url", "") if contact else "",
            email_confidence=getattr(contact, "email_confidence", 0) if contact else 0,
            email_subject=getattr(outreach, "subject", "") if outreach else "",
            email_body=getattr(outreach, "body", "") if outreach else "",
            trigger_event=getattr(sheet, "trigger_event", ""),
            pain_point=getattr(sheet, "pain_point", ""),
            service_pitch=getattr(sheet, "service_pitch", ""),
            opening_line=getattr(sheet, "opening_line", ""),
            urgency_weeks=getattr(sheet, "urgency_weeks", 4),
            confidence=getattr(sheet, "confidence", 0.0),
            oss_score=getattr(sheet, "oss_score", 0.0),
            data_sources=getattr(sheet, "data_sources", []),
            company_news=getattr(sheet, "company_news", []),
            people=[
                PersonResponse(
                    person_name=getattr(p, "person_name", ""),
                    role=getattr(p, "role", ""),
                    seniority_tier=getattr(p, "seniority_tier", "influencer"),
                    linkedin_url=getattr(p, "linkedin_url", ""),
                    email=getattr(p, "email", ""),
                    email_confidence=getattr(p, "email_confidence", 0),
                    verified=getattr(p, "verified", False),
                    reach_score=getattr(p, "reach_score", 0),
                    outreach_tone=getattr(p, "outreach_tone", "consultative"),
                    outreach_subject=getattr(p, "outreach_subject", ""),
                    outreach_body=getattr(p, "outreach_body", ""),
                )
                for p in _profiles_by_company.get(cname, [])
            ],
        ))
    return results


def _leads_from_db(
    run_id: Optional[str] = None,
    hop: Optional[int] = None,
    lead_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 100,
) -> list:
    """Query call sheets from DB (durable across restarts)."""
    try:
        db = get_database()
        rows = db.get_call_sheets(
            run_id=run_id,
            hop=hop,
            lead_type=lead_type,
            min_confidence=min_confidence,
            limit=limit,
        )
        return [LeadResponse(**row) for row in rows]
    except Exception as e:
        logger.warning(f"DB lead query failed: {e}")
        return []


def _enrich_leads_from_db(leads: list) -> None:
    """Merge DB enrichment (people, contacts, emails) into leads that lack it.

    Two-tier approach:
    1. Load call_sheets for primary contact/email backfill
    2. Directly query lead_contacts table for people[] (catches enriched contacts)
    """
    needs_enrich = [l for l in leads if not l.people]
    if not needs_enrich:
        return

    try:
        db = get_database()

        # Tier 1: Call sheets for primary contact data
        db_rows = db.get_call_sheets(limit=300)
        db_by_company: dict[str, dict] = {}
        for row in db_rows:
            cname = row.get("company_name", "")
            existing = db_by_company.get(cname)
            if not existing or (row.get("people") and not existing.get("people")):
                db_by_company[cname] = row

        # Tier 2: Direct query lead_contacts for ALL companies (including enriched)
        contacts_by_company: dict[str, list[dict]] = {}
        try:
            from app.database import LeadContactModel
            with db.get_session() as session:
                all_contacts = session.query(LeadContactModel).all()
                for c in all_contacts:
                    cname = c.company_name
                    if cname not in contacts_by_company:
                        contacts_by_company[cname] = []
                    contacts_by_company[cname].append({
                        "person_name": c.person_name,
                        "role": c.role,
                        "seniority_tier": c.seniority_tier or "influencer",
                        "linkedin_url": c.linkedin_url or "",
                        "email": c.email or "",
                        "email_confidence": int(c.email_confidence or 0),
                        "verified": bool(c.verified),
                        "reach_score": int(c.reach_score or 0),
                        "outreach_tone": c.outreach_tone or "consultative",
                        "outreach_subject": c.outreach_subject or "",
                        "outreach_body": c.outreach_body or "",
                    })
        except Exception as e:
            logger.debug(f"Direct contact query failed: {e}")

        for lead in needs_enrich:
            # Try direct contacts first (catches enriched contacts)
            direct_contacts = contacts_by_company.get(lead.company_name, [])
            if direct_contacts and not lead.people:
                lead.people = [
                    PersonResponse(**p) for p in direct_contacts
                ]
                # Also set primary contact from first person
                if not lead.contact_name and direct_contacts[0].get("person_name"):
                    lead.contact_name = direct_contacts[0]["person_name"]
                if not lead.contact_email and direct_contacts[0].get("email"):
                    lead.contact_email = direct_contacts[0]["email"]
                if not lead.contact_role and direct_contacts[0].get("role"):
                    lead.contact_role = direct_contacts[0]["role"]
                if not lead.email_subject and direct_contacts[0].get("outreach_subject"):
                    lead.email_subject = direct_contacts[0]["outreach_subject"]
                if not lead.email_body and direct_contacts[0].get("outreach_body"):
                    lead.email_body = direct_contacts[0]["outreach_body"]
                continue

            # Fall back to call sheets
            db_row = db_by_company.get(lead.company_name)
            if not db_row:
                continue
            if db_row.get("people") and not lead.people:
                lead.people = [
                    PersonResponse(
                        person_name=p.get("person_name", ""),
                        role=p.get("role", ""),
                        seniority_tier=p.get("seniority_tier", "influencer"),
                        linkedin_url=p.get("linkedin_url", ""),
                        email=p.get("email", ""),
                        email_confidence=int(p.get("email_confidence", 0)),
                        verified=p.get("verified", False),
                        reach_score=int(p.get("reach_score", 0)),
                        outreach_tone=p.get("outreach_tone", "consultative"),
                        outreach_subject=p.get("outreach_subject", ""),
                        outreach_body=p.get("outreach_body", ""),
                    )
                    for p in db_row["people"]
                ]
            if not lead.contact_name and db_row.get("contact_name"):
                lead.contact_name = db_row["contact_name"]
            if not lead.contact_email and db_row.get("contact_email"):
                lead.contact_email = db_row["contact_email"]
            if not lead.contact_role and db_row.get("contact_role"):
                lead.contact_role = db_row["contact_role"]
            if not lead.contact_linkedin and db_row.get("contact_linkedin"):
                lead.contact_linkedin = db_row["contact_linkedin"]
            if not lead.email_subject and db_row.get("email_subject"):
                lead.email_subject = db_row["email_subject"]
            if not lead.email_body and db_row.get("email_body"):
                lead.email_body = db_row["email_body"]
            if not lead.email_confidence and db_row.get("email_confidence"):
                lead.email_confidence = db_row["email_confidence"]
    except Exception as e:
        logger.debug(f"DB enrichment merge skipped: {e}")


def _merge_enriched_db_leads(leads: list) -> None:
    """Add DB leads with enrichment data that aren't already in the list.

    When the latest run lacks contacts but older runs have rich data,
    append those enriched leads so the user sees ALL available intelligence.
    """
    try:
        existing_names = {l.company_name.lower() for l in leads}
        db_leads = _leads_from_db(limit=100)
        added = 0
        for dl in db_leads:
            if dl.company_name.lower() in existing_names:
                continue
            # Only add if it has some enrichment
            if dl.contact_name or dl.people or dl.email_subject:
                dl.id = len(leads)  # Assign sequential ID
                leads.append(dl)
                existing_names.add(dl.company_name.lower())
                added += 1
        if added:
            logger.info(f"Merged {added} enriched leads from DB")
    except Exception as e:
        logger.debug(f"DB lead merge skipped: {e}")


@router.get("", response_model=LeadListResponse)
async def list_leads(
    run_id: str = Query(None, description="Filter by pipeline run"),
    hop: int = Query(None, ge=1, le=3, description="Filter by hop level"),
    lead_type: str = Query(None, description="pain|opportunity|risk|intelligence"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    """List leads — tries in-memory first, falls back to recordings, then DB."""
    # Try in-memory RunManager first (faster, has active runs)
    leads = []
    if run_id:
        run = run_manager.get_run(run_id)
        if run and (run.final_state or run.result):
            leads = _leads_from_run(run)
        # Fallback: load from recordings if not in memory
        if not leads and run_id:
            from app.api.pipeline import _load_run_from_recording
            from app.tools.run_recorder import get_recording
            recording_dir = get_recording(run_id)
            if recording_dir and (recording_dir / "manifest.json").exists():
                run = _load_run_from_recording(run_id, recording_dir)
                leads = _leads_from_run(run)
    else:
        run = run_manager.get_latest_run()
        if run and run.status == "completed":
            leads = _leads_from_run(run)
        # Fallback: load latest recording
        if not leads:
            from app.api.pipeline import _load_run_from_recording
            from app.tools.run_recorder import get_latest_recording
            recording_dir = get_latest_recording()
            if recording_dir and (recording_dir / "manifest.json").exists():
                run = _load_run_from_recording(recording_dir.name, recording_dir)
                leads = _leads_from_run(run)

    # Fall back to DB if in-memory has no data
    if not leads:
        leads = _leads_from_db(
            run_id=run_id,
            hop=hop,
            lead_type=lead_type,
            min_confidence=min_confidence if min_confidence > 0 else None,
            limit=limit + offset,
        )
    else:
        # Apply in-memory filters
        if hop is not None:
            leads = [l for l in leads if l.hop == hop]
        if lead_type:
            leads = [l for l in leads if l.lead_type == lead_type]
        if min_confidence > 0:
            leads = [l for l in leads if l.confidence >= min_confidence]
        leads.sort(key=lambda l: l.confidence, reverse=True)

    # Enrich leads with DB people data when replay/in-memory lacks enrichment
    _enrich_leads_from_db(leads)

    # Merge enriched DB leads that aren't in the current set
    # (shows leads from older runs that have contact/email data)
    if not run_id:
        _merge_enriched_db_leads(leads)

    total = len(leads)
    leads = leads[offset:offset + limit]

    return LeadListResponse(total=total, leads=leads)


@router.get("/latest", response_model=LeadListResponse)
async def latest_leads(limit: int = Query(50)):
    """Get leads from most recent completed run."""
    # Try in-memory first
    run = run_manager.get_latest_run()
    if run and run.status == "completed":
        leads = _leads_from_run(run)
    else:
        leads = _leads_from_db(limit=limit)

    # Enrich from DB (contacts, people, emails) + merge older enriched leads
    _enrich_leads_from_db(leads)
    _merge_enriched_db_leads(leads)
    leads.sort(key=lambda l: l.confidence, reverse=True)
    return LeadListResponse(total=len(leads), leads=leads[:limit])


@router.get("/{lead_id}")
async def get_lead_detail(lead_id: int):
    """Single lead by ID — tries in-memory, then DB."""
    # Try in-memory
    run = run_manager.get_latest_run()
    leads = _leads_from_run(run)
    if not leads:
        # Try latest recording
        from app.api.pipeline import _load_run_from_recording
        from app.tools.run_recorder import get_latest_recording
        recording_dir = get_latest_recording()
        if recording_dir and (recording_dir / "manifest.json").exists():
            run = _load_run_from_recording(recording_dir.name, recording_dir)
            leads = _leads_from_run(run)

    lead = next((l for l in leads if l.id == lead_id), None)
    if lead:
        _enrich_leads_from_db([lead])
        return lead

    # Try DB
    try:
        db = get_database()
        all_sheets = db.get_call_sheets(limit=200)
        for sheet in all_sheets:
            if sheet["id"] == lead_id:
                return LeadResponse(**sheet)
    except Exception as e:
        logger.error(f"DB error fetching lead {lead_id}: {e}")
        raise HTTPException(500, "Database error")

    raise HTTPException(404, "Lead not found")


# ── Enrich lead (find contacts + generate outreach) ─────────────────


class EnrichLeadResponse(BaseModel):
    success: bool
    contacts_found: int = 0
    outreach_generated: int = 0
    people: list = []
    error: str = ""


@router.post("/{lead_id}/enrich", response_model=EnrichLeadResponse)
async def enrich_lead(lead_id: int):
    """Find contacts + generate outreach for a single lead.

    Uses Hunter/Apollo to find people, then LLM to generate personalized emails.
    This allows re-enriching a lead without re-running the full pipeline.
    """
    # Get lead data
    run = run_manager.get_latest_run()
    leads = _leads_from_run(run)
    if not leads:
        from app.api.pipeline import _load_run_from_recording
        from app.tools.run_recorder import get_latest_recording
        recording_dir = get_latest_recording()
        if recording_dir and (recording_dir / "manifest.json").exists():
            run = _load_run_from_recording(recording_dir.name, recording_dir)
            leads = _leads_from_run(run)
    _enrich_leads_from_db(leads)

    lead = next((l for l in leads if l.id == lead_id), None)
    if not lead:
        raise HTTPException(404, "Lead not found")

    company_name = lead.company_name
    domain = lead.company_domain
    trend_title = lead.trend_title

    try:
        # Step 1: Find domain if missing
        if not domain and lead.company_website:
            from urllib.parse import urlparse
            parsed = urlparse(lead.company_website)
            domain = parsed.netloc.lower().removeprefix("www.")

        if not domain:
            # Try Tavily search for company domain
            try:
                from app.tools.web.tavily_tool import TavilyTool
                tavily = TavilyTool()
                results = await asyncio.wait_for(
                    tavily.search(f"{company_name} official website", max_results=3),
                    timeout=10.0,
                )
                from urllib.parse import urlparse as _up
                skip = {"wikipedia", "linkedin", "crunchbase", "zoominfo",
                        "ambitionbox", "glassdoor", "moneycontrol", "tofler"}
                for r in results or []:
                    url = r.get("url", "") or r.get("href", "")
                    if url:
                        host = _up(url).netloc.lower().removeprefix("www.")
                        if host and not any(s in host for s in skip):
                            domain = host
                            break
            except Exception as e:
                logger.warning(f"Tavily domain search failed for {company_name}: {e}")

        if not domain:
            # Fallback: DDG search
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(f"{company_name} official website", max_results=3))
                    from urllib.parse import urlparse as _up2
                    for r in results:
                        url = r.get("href", "")
                        if url:
                            host = _up2(url).netloc.lower().removeprefix("www.")
                            if host and not any(s in host for s in [
                                "wikipedia", "linkedin", "crunchbase", "zoominfo",
                            ]):
                                domain = host
                                break
            except Exception as e:
                logger.warning(f"DDG domain search failed for {company_name}: {e}")

        if not domain:
            return EnrichLeadResponse(
                success=False, error=f"Could not resolve domain for {company_name}"
            )
        logger.info(f"Enriching {company_name} with domain={domain}")

        # Step 2: Find contacts via Hunter + Apollo
        people_found = []

        # Try Hunter domain search
        try:
            from app.tools.crm.hunter_tool import HunterTool
            hunter = HunterTool()
            hunter_results = await asyncio.wait_for(
                hunter.domain_search(domain, limit=5),
                timeout=15.0,
            )
            for h in hunter_results or []:
                people_found.append({
                    "person_name": f"{h.get('first_name', '')} {h.get('last_name', '')}".strip(),
                    "role": h.get("position", ""),
                    "email": h.get("email", ""),
                    "email_confidence": h.get("confidence", 0),
                    "linkedin_url": h.get("linkedin", ""),
                    "seniority_tier": h.get("seniority", "influencer"),
                })
        except Exception as e:
            logger.debug(f"Hunter enrichment failed: {e}")

        # Try Apollo if Hunter finds nothing
        if not people_found:
            try:
                from app.tools.crm.apollo_tool import ApolloTool
                apollo = ApolloTool()
                apollo_result = await asyncio.wait_for(
                    apollo.search_people_at_company(domain, limit=5),
                    timeout=15.0,
                )
                for p in (apollo_result or {}).get("people", []):
                    people_found.append({
                        "person_name": p.get("name", ""),
                        "role": p.get("title", ""),
                        "email": p.get("email", ""),
                        "email_confidence": 70 if p.get("email") else 0,
                        "linkedin_url": p.get("linkedin_url", ""),
                        "seniority_tier": p.get("seniority", "influencer"),
                    })
            except Exception as e:
                logger.debug(f"Apollo enrichment failed: {e}")

        if not people_found:
            return EnrichLeadResponse(
                success=False, error="No contacts found via Hunter or Apollo"
            )

        # Step 3: Generate outreach emails via LLM for top contacts
        outreach_count = 0
        try:
            from app.tools.llm.llm_service import LLMService
            llm = LLMService(lite=True)
            for p in people_found[:3]:  # Top 3 contacts
                if not p.get("email"):
                    continue
                from app.config import get_settings
                _s = get_settings()
                sender_org = _s.brevo_sender_name or "Coherent Market Insights"
                prompt = (
                    f"Write a short, personalized sales outreach email from {sender_org}.\n"
                    f"Sender: {sender_org} — a global market research and consulting firm.\n"
                    f"To: {p['person_name']}, {p['role']} at {company_name}\n"
                    f"Context: {trend_title[:200] if trend_title else 'Industry opportunity'}\n"
                    f"Pain point: {lead.pain_point[:200] if lead.pain_point else 'N/A'}\n"
                    f"Our pitch: {lead.service_pitch[:200] if lead.service_pitch else 'Market intelligence and strategic advisory'}\n\n"
                    f"RULES:\n"
                    f"- Sign off as: Best regards,\\n{sender_org} Team\n"
                    f"- Do NOT use placeholders like [Your Name] or [Your Contact Information]\n"
                    f"- Do NOT use generic openers like 'Hope this finds you well'\n"
                    f"- Keep it under 150 words\n\n"
                    f"Reply with ONLY:\nSubject: <subject line>\n\n<email body>"
                )
                result = await asyncio.wait_for(
                    llm.generate(prompt, temperature=0.7, max_tokens=400),
                    timeout=15.0,
                )
                if result:
                    lines = result.strip().split("\n", 2)
                    subject = ""
                    body = result
                    if lines[0].lower().startswith("subject:"):
                        subject = lines[0].split(":", 1)[1].strip()
                        body = "\n".join(lines[1:]).strip()
                    p["outreach_subject"] = subject
                    p["outreach_body"] = body
                    p["outreach_tone"] = "consultative"
                    outreach_count += 1
        except Exception as e:
            logger.debug(f"Outreach generation failed: {e}")

        # Step 4: Save to DB for persistence using proper database method
        try:
            db = get_database()
            from types import SimpleNamespace
            profiles = [
                SimpleNamespace(
                    company_name=company_name,
                    person_name=p.get("person_name", ""),
                    role=p.get("role", ""),
                    seniority_tier=p.get("seniority_tier", "influencer"),
                    linkedin_url=p.get("linkedin_url", ""),
                    email=p.get("email", ""),
                    email_confidence=float(p.get("email_confidence", 0)),
                    email_source="hunter",
                    verified=False,
                    reach_score=0,
                    outreach_subject=p.get("outreach_subject", ""),
                    outreach_body=p.get("outreach_body", ""),
                    outreach_tone=p.get("outreach_tone", "consultative"),
                )
                for p in people_found
            ]
            saved = db.save_lead_contacts("enriched", profiles)
            logger.info(f"Saved {saved} contacts to DB for {company_name}")
        except Exception as e:
            logger.warning(f"DB save failed for {company_name}: {e}")

        person_responses = [
            PersonResponse(
                person_name=p.get("person_name") or "",
                role=p.get("role") or "",
                seniority_tier=p.get("seniority_tier") or "influencer",
                linkedin_url=p.get("linkedin_url") or "",
                email=p.get("email") or "",
                email_confidence=int(p.get("email_confidence") or 0),
                outreach_subject=p.get("outreach_subject") or "",
                outreach_body=p.get("outreach_body") or "",
                outreach_tone=p.get("outreach_tone") or "consultative",
            )
            for p in people_found
        ]

        return EnrichLeadResponse(
            success=True,
            contacts_found=len(people_found),
            outreach_generated=outreach_count,
            people=[pr.model_dump() for pr in person_responses],
        )

    except Exception as e:
        logger.error(f"Lead enrichment failed: {e}")
        return EnrichLeadResponse(success=False, error=str(e))


# ── Email sending ────────────────────────────────────────────────────


class SendEmailRequest(BaseModel):
    person_index: int = 0  # Which person in the people[] array (0 = primary contact)


class SendEmailResponse(BaseModel):
    success: bool
    message_id: str = ""
    recipient: str = ""
    subject: str = ""
    error: str = ""
    test_mode: bool = False
    sent_at: str = ""


class EmailSettingsResponse(BaseModel):
    sending_enabled: bool
    test_mode: bool
    test_recipient: str
    brevo_configured: bool
    sender_email: str
    sender_name: str


@router.get("/email/settings", response_model=EmailSettingsResponse)
async def get_email_settings():
    """Get current email sending configuration (safe to expose — no secrets)."""
    from app.config import get_settings
    s = get_settings()
    return EmailSettingsResponse(
        sending_enabled=s.email_sending_enabled,
        test_mode=s.email_test_mode,
        test_recipient=s.email_test_recipient,
        brevo_configured=bool(s.brevo_api_key),
        sender_email=s.brevo_sender_email,
        sender_name=s.brevo_sender_name,
    )


@router.post("/{lead_id}/send-email", response_model=SendEmailResponse)
async def send_lead_email(lead_id: int, req: SendEmailRequest):
    """Send outreach email for a specific lead.

    Safety:
    - EMAIL_SENDING_ENABLED must be True (global kill switch)
    - EMAIL_TEST_MODE=True redirects to test recipient from settings
    """
    # Find the lead — use same pipeline as GET /leads for consistent indexing
    run = run_manager.get_latest_run()
    leads = []
    if run and run.status == "completed":
        leads = _leads_from_run(run)
    if not leads:
        # Try latest recording
        from app.api.pipeline import _load_run_from_recording
        from app.tools.run_recorder import get_latest_recording
        recording_dir = get_latest_recording()
        if recording_dir and (recording_dir / "manifest.json").exists():
            run = _load_run_from_recording(recording_dir.name, recording_dir)
            leads = _leads_from_run(run)
    if not leads:
        leads = _leads_from_db(limit=200)
    _enrich_leads_from_db(leads)
    _merge_enriched_db_leads(leads)
    lead = next((l for l in leads if l.id == lead_id), None)

    if lead is None:
        raise HTTPException(404, "Lead not found")

    # Determine email content — from people[] or primary contact
    to_email = ""
    to_name = ""
    subject = ""
    body = ""

    tone = "consultative"
    if lead.people and 0 <= req.person_index < len(lead.people):
        person = lead.people[req.person_index]
        to_email = person.email
        to_name = person.person_name
        subject = person.outreach_subject
        body = person.outreach_body
        tone = person.outreach_tone or "consultative"
    else:
        # Fallback to primary contact
        to_email = lead.contact_email
        to_name = lead.contact_name
        subject = lead.email_subject
        body = lead.email_body

    # Fallback: if contact has no email, use test recipient from settings
    from app.config import get_settings
    _settings = get_settings()
    if not to_email:
        if _settings.email_test_mode and _settings.email_test_recipient:
            to_email = _settings.email_test_recipient
            logger.info(f"No contact email — using test recipient: {to_email}")
        else:
            raise HTTPException(400, "No email address available for this contact")
    if not subject or not body:
        raise HTTPException(400, "No outreach email generated for this contact")

    # Send via Brevo with branded template
    from app.tools.crm.brevo_tool import BrevoTool
    brevo = BrevoTool()
    result = brevo.send_email(
        to_email=to_email,
        to_name=to_name,
        subject=subject,
        body=body,
        dry_run=False,
        tone=tone,
        trend_title=lead.trend_title,
        company_name=lead.company_name,
        lead_type=lead.lead_type,
    )

    return SendEmailResponse(**result.to_dict())
