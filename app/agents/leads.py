"""
Leads — crystallization + enrichment pipeline.

Two stages:
  1. crystallize_leads()  — converts causal chain hops into concrete call sheets (LeadSheet)
  2. run_lead_gen()       — enriches lead sheets with contacts, emails, person profiles

Previously split across lead_crystallizer.py and lead_gen.py.
"""
import asyncio
import hashlib
import logging
from typing import Any, List, Literal

from pydantic import BaseModel, Field

from app.agents.deps import AgentDeps

logger = logging.getLogger(__name__)


# ── Lead sheet schema ─────────────────────────────────────────────────────────

class LeadSheet(BaseModel):
    """One concrete lead — everything a salesperson needs to make the call."""

    # Company identity
    company_name: str               # "Kama Jewellery Pvt Ltd"
    company_cin: str = ""           # "U36912GJ2001PTC039001" (from MCA)
    company_state: str = ""         # "Gujarat"
    company_city: str = ""          # "Rajkot"
    company_size_band: str = ""     # "sme" | "mid"
    company_category: str = ""      # "Company limited by shares"
    lei_id: str = ""                # Global LEI (future: GLEIF data)

    # Lead context
    hop: int                        # 1 = direct, 2 = buyer/supplier, 3 = downstream
    lead_type: Literal["pain", "opportunity", "risk", "intelligence"]
    trend_title: str                # "Steel import duty raised 15%"
    event_type: str = ""            # "trade_policy", "supply_chain", etc.

    # Sales content (what the rep says)
    contact_role: str               # "VP Procurement / Head of Sourcing"
    trigger_event: str              # "Steel import duty raised 15% on 2026-02-24"
    pain_point: str                 # "Input costs will rise ~15% in 4-6 weeks"
    service_pitch: str              # "Should-cost analysis + alternative sourcing"
    opening_line: str               # Ready-to-use first sentence for the call/email
    urgency_weeks: int = 4          # How quickly does this impact materialize?
    confidence: float = 0.7

    # Chain-of-thought carried from CausalCouncil reasoning
    reasoning: str = ""

    # Company-specific recent news
    company_news: list[dict] = Field(default_factory=list, description="Recent news about this company")

    # Self-learning signals
    data_sources: list[str] = []    # ["mca_kb", "bm25_article", "causal_inference"]
    oss_score: float = 0.0          # OSS of source synthesis (for source bandit)


# ── Contact role and service pitch mappings ───────────────────────────────────

_CONTACT_ROLES: dict[str, str] = {
    "supply_chain":  "VP Procurement / Head of Sourcing / CSCO",
    "price_change":  "CFO / VP Finance / Head of Treasury",
    "regulation":    "Compliance Head / VP Legal / Regulatory Affairs Director",
    "trade_policy":  "CEO / MD / Head of International Trade",
    "m_and_a":       "CEO / CFO / VP Strategy",
    "technology":    "CTO / VP Engineering / Chief Digital Officer",
    "labor":         "CHRO / VP HR / Head of Industrial Relations",
    "infrastructure": "COO / VP Operations / Head of Projects",
    "general":       "CEO / Managing Director",
}

_SERVICES: dict[tuple[str, str], str] = {
    ("pain", "supply_chain"):     "Should-cost analysis + alternative sourcing strategy",
    ("pain", "price_change"):     "Commodity price risk assessment + hedging strategy",
    ("pain", "regulation"):       "Regulatory compliance roadmap + gap analysis",
    ("pain", "trade_policy"):     "Trade impact assessment + mitigation playbook",
    ("pain", "infrastructure"):   "Project delay risk analysis + critical path review",
    ("opportunity", "supply_chain"): "Market share expansion intelligence + supplier mapping",
    ("opportunity", "price_change"): "Procurement timing + forward-buying strategy",
    ("opportunity", "trade_policy"): "Export opportunity intelligence + market entry support",
    ("risk", "trade_policy"):     "Trade risk monitoring + scenario planning",
    ("risk", "regulation"):       "Regulatory risk exposure report + remediation roadmap",
    ("intelligence", "m_and_a"):  "M&A target intelligence + due diligence support",
    ("intelligence", "technology"): "Technology landscape intelligence + competitive benchmarking",
}


# ── Stage 1: Crystallize ──────────────────────────────────────────────────────

async def crystallize_leads(
    causal_result,                  # CausalChainResult from run_causal_council()
    trend_title: str,
    trend_summary: str,
    event_type: str,
    oss_score: float = 0.0,
) -> list[LeadSheet]:
    """
    Convert causal chain hops into concrete call-sheet leads.

    For each hop (above confidence threshold):
    1. Find real companies from hop.companies_found (already from KB)
       or fall back to KB search on hop.segment
    2. Build a LeadSheet with opening line, service pitch, contact role
    3. Enrich with company details (CIN, state, city) from KB

    Returns leads sorted by (confidence DESC, urgency_weeks ASC).
    """
    leads: list[LeadSheet] = []

    if not causal_result or not causal_result.hops:
        logger.warning("Lead crystallizer: no causal hops — returning empty")
        return leads

    for hop in causal_result.hops:
        if hop.confidence < 0.35:
            logger.debug(f"Skip hop {hop.hop}: confidence {hop.confidence:.2f} < 0.35")
            continue

        companies: list[str] = list(hop.companies_found) if hop.companies_found else []
        if not companies:
            companies = [f"[{hop.segment}]"]

        contact_role = _CONTACT_ROLES.get(event_type, _CONTACT_ROLES["general"])
        service_key = (hop.lead_type, event_type)
        service_pitch = _SERVICES.get(service_key, _default_service(hop.lead_type))

        for company_name in companies[:3]:   # Max 3 companies per hop (no spam)
            is_placeholder = company_name.startswith("[")

            lead = LeadSheet(
                company_name=company_name if not is_placeholder else hop.segment,
                company_state=_first_geo(hop.geo_hint),
                company_size_band=hop.employee_band,
                company_category="",
                hop=hop.hop,
                lead_type=hop.lead_type,
                trend_title=trend_title,
                event_type=event_type,
                contact_role=contact_role,
                trigger_event=f"{trend_title} — {causal_result.event_summary[:80]}",
                pain_point=hop.mechanism,
                service_pitch=service_pitch,
                opening_line=_opening(
                    company_name if not is_placeholder else "Your business",
                    trend_title, hop.mechanism, hop.lead_type, hop.urgency_weeks,
                ),
                urgency_weeks=hop.urgency_weeks,
                confidence=hop.confidence * (0.8 if is_placeholder else 1.0),
                reasoning=causal_result.reasoning,
                data_sources=["causal_inference"],
                oss_score=oss_score,
            )
            leads.append(lead)

    leads.sort(key=lambda l: (-l.confidence, l.urgency_weeks))
    logger.info(f"Crystallized {len(leads)} leads from {len(causal_result.hops)} hops")
    return leads


def _opening(company: str, trend: str, mechanism: str, lead_type: str, weeks: int) -> str:
    if lead_type == "pain":
        return (
            f"{company} faces a concrete challenge: {mechanism}. "
            f"This will likely materialize within {weeks} weeks. "
            f"We help companies navigate exactly this kind of pressure."
        )
    elif lead_type == "opportunity":
        return (
            f"The recent {trend.lower()} creates a specific opening for {company}. "
            f"{mechanism}. "
            f"Companies that move in the next {weeks} weeks capture the most value."
        )
    elif lead_type == "risk":
        return (
            f"{company} carries elevated risk right now: {mechanism}. "
            f"Our risk intelligence platform can help quantify and mitigate this exposure."
        )
    else:
        return (
            f"Given the recent {trend.lower()}, {company} stands to benefit from "
            f"sharper competitive intelligence. {mechanism}."
        )


def _first_geo(geo_hint: str) -> str:
    if not geo_hint:
        return ""
    return geo_hint.split(",")[0].strip()


def _default_service(lead_type: str) -> str:
    if lead_type == "pain":
        return "Market intelligence + strategic advisory"
    elif lead_type == "opportunity":
        return "Opportunity intelligence + execution support"
    else:
        return "Business intelligence + risk monitoring"


# ── Stage 2: Lead Gen (contact + email enrichment) ────────────────────────────

class LeadGenResult(BaseModel):
    """Structured output from the Lead Gen Agent."""
    companies_found: int = 0
    contacts_found: int = 0
    emails_generated: int = 0
    outreach_generated: int = 0
    low_relevance_filtered: int = 0
    reasoning: str = ""


async def _resolve_domain(company_name: str, mock_mode: bool = False) -> str:
    """Find a company's domain via web search + validation."""
    if mock_mode:
        from app.tools.domain_utils import extract_domain_from_company_name
        return extract_domain_from_company_name(company_name) or ""
    try:
        from app.tools.web.web_intel import search
        from app.tools.domain_utils import extract_clean_domain, is_valid_company_domain

        results = await search(f"{company_name} official website", max_results=5)
        for r in results:
            d = extract_clean_domain(r.url)
            if d and is_valid_company_domain(d):
                return d
    except Exception as e:
        logger.debug(f"Domain resolution for '{company_name}' failed: {e}")
    return ""


async def _build_companies_from_leads(
    deps: AgentDeps,
    lead_sheets: list,
) -> list:
    """Build CompanyData objects from unique company names in lead sheets."""
    from app.schemas.sales import CompanyData

    seen: dict[str, Any] = {}
    sem = asyncio.Semaphore(5)

    async def _build_one(name: str, lead):
        async with sem:
            if name in seen:
                return
            try:
                from app.database import get_database
                db = get_database()
                cached = db.get_or_enrich_company(name, max_age_days=7)
                if cached and (cached.get("description") or cached.get("headquarters") or cached.get("wikidata_id")):
                    cid = cached["id"]
                    seen[name] = CompanyData(
                        id=cid,
                        company_name=name,
                        industry=cached.get("industry") or lead.event_type or "general",
                        domain=cached.get("domain", ""),
                        website=cached.get("website", ""),
                        description=cached.get("description", ""),
                        headquarters=cached.get("headquarters", ""),
                        employee_count=cached.get("employee_count", ""),
                        stock_ticker=cached.get("stock_ticker", ""),
                        ceo=cached.get("ceo", ""),
                        founded_year=cached.get("founded_year"),
                        wikidata_id=cached.get("wikidata_id", ""),
                        reason_relevant=lead.pain_point or lead.trend_title,
                        trend_id=str(lead.trend_title),
                    )
                    logger.info(f"  KB hit for {name} — skipping enrichment")
                    return
            except Exception:
                pass

            domain = await _resolve_domain(name, mock_mode=deps.mock_mode)
            cid = hashlib.md5(name.lower().encode()).hexdigest()[:12]
            try:
                seen[name] = CompanyData(
                    id=cid,
                    company_name=name,
                    industry=lead.event_type or "general",
                    domain=domain,
                    website=f"https://{domain}" if domain else "",
                    reason_relevant=lead.pain_point or lead.trend_title,
                    trend_id=str(lead.trend_title),
                )
            except Exception as e:
                logger.debug(f"CompanyData validation failed for '{name}': {e}")
                return
            if domain:
                logger.info(f"  Resolved domain: {name} → {domain}")
            else:
                logger.debug(f"  No domain found for: {name}")

    tasks = []
    for lead in lead_sheets:
        name = lead.company_name
        # Skip placeholders and segment descriptions (not real company names)
        if not name or name.startswith("[") or len(name) > 100:
            continue
        if name not in seen:
            tasks.append(_build_one(name, lead))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    companies = list(seen.values())
    to_enrich = [c for c in companies if not c.description and not getattr(c, "wikidata_id", "")]
    if to_enrich:
        try:
            from app.tools.company_enricher import enrich

            enrich_sem = asyncio.Semaphore(5)

            async def _do_enrich(c: "CompanyData"):
                async with enrich_sem:
                    try:
                        enriched = await asyncio.wait_for(
                            enrich(c.company_name, domain=c.domain, skip_validation=True),
                            timeout=8.0,
                        )
                        if enriched:
                            c.description = enriched.description or c.description
                            c.headquarters = enriched.headquarters or getattr(c, "headquarters", "")
                            c.employee_count = enriched.employee_count or getattr(c, "employee_count", "")
                            c.stock_ticker = enriched.stock_ticker or getattr(c, "stock_ticker", "")
                            c.ceo = enriched.ceo or getattr(c, "ceo", "")
                            c.founded_year = enriched.founded_year or getattr(c, "founded_year", None)
                            c.wikidata_id = enriched.wikidata_id or getattr(c, "wikidata_id", "")
                            c.industry = enriched.industry or c.industry
                            if enriched.domain and not c.domain:
                                c.domain = enriched.domain
                                c.website = f"https://{enriched.domain}"
                    except Exception as e:
                        logger.debug(f"Enrichment failed for {c.company_name}: {e}")

            results = await asyncio.gather(*[_do_enrich(c) for c in to_enrich], return_exceptions=True)
            err_count = sum(1 for r in results if isinstance(r, Exception))
            if err_count:
                logger.debug(f"Enrichment: {err_count}/{len(to_enrich)} companies had errors")
            enriched_count = sum(1 for c in to_enrich if c.description)
            logger.info(f"  Enriched {enriched_count}/{len(to_enrich)} companies via company_enricher")
        except Exception as e:
            logger.debug(f"Batch enrichment failed: {e}")

    return companies


def _build_person_profiles(
    contacts: list,
    outreach_emails: list,
    impacts: list,
    companies: list,
) -> list:
    """Build PersonProfile objects from contacts + outreach, with reach scores."""
    from app.schemas.sales import PersonProfile
    from app.agents.workers.contact_agent import ContactFinder

    outreach_by_contact = {}
    for em in outreach_emails:
        cid = getattr(em, "contact_id", "")
        if cid:
            outreach_by_contact[cid] = em

    impact_roles: dict[str, set] = {}
    for imp in impacts:
        tid = getattr(imp, "trend_id", "")
        roles = {r.lower() for r in getattr(imp, "target_roles", [])}
        impact_roles[tid] = roles

    company_trend: dict[str, str] = {}
    for c in companies:
        company_trend[getattr(c, "company_name", "")] = getattr(c, "trend_id", "")

    profiles = []
    for ct in contacts:
        try:
            tier = ContactFinder.classify_tier(ct.role)
            tone = ContactFinder.get_outreach_tone(tier)

            trend_id = company_trend.get(ct.company_name, "")
            target_roles = impact_roles.get(trend_id, set())
            role_relevance = 0.7 if ct.role.lower() in target_roles else 0.3

            reach = ContactFinder.compute_reach_score(
                email=ct.email,
                email_confidence=ct.email_confidence,
                verified=ct.verified,
                linkedin_url=ct.linkedin_url,
                seniority_tier=tier,
                role_relevance=role_relevance,
            )

            outreach = outreach_by_contact.get(ct.id)

            profiles.append(PersonProfile(
                id=ct.id,
                company_id=ct.company_id,
                company_name=ct.company_name,
                person_name=ct.person_name,
                role=ct.role,
                seniority_tier=tier,
                linkedin_url=ct.linkedin_url,
                email=ct.email,
                email_confidence=ct.email_confidence,
                email_source=ct.email_source,
                verified=ct.verified,
                reach_score=reach,
                outreach_tone=tone,
                outreach_subject=getattr(outreach, "subject", "") if outreach else "",
                outreach_body=getattr(outreach, "body", "") if outreach else "",
            ))
        except Exception as e:
            logger.warning(f"Failed to build profile for {getattr(ct, 'person_name', '?')}: {e}")
            continue

    tier_order = {"decision_maker": 0, "influencer": 1, "gatekeeper": 2}
    profiles.sort(key=lambda p: (tier_order.get(p.seniority_tier, 1), -p.reach_score))
    return profiles


async def run_lead_gen(deps: AgentDeps) -> tuple:
    """Enrich pre-resolved lead sheets with contacts and emails.

    Returns (companies, contacts, outreach, result).
    """
    lead_sheets = getattr(deps, "_lead_sheets", [])
    if not lead_sheets:
        logger.warning("Lead gen: no lead sheets from crystallizer — skipping enrichment")
        return [], [], [], LeadGenResult(reasoning="No lead sheets to enrich")

    logger.info(f"Lead gen: enriching {len(lead_sheets)} lead sheets")
    agent_reasoning_parts = []

    try:
        companies = await asyncio.wait_for(
            _build_companies_from_leads(deps, lead_sheets),
            timeout=300.0,
        )
        deps._companies = companies
        with_domain = sum(1 for c in companies if c.domain)
        agent_reasoning_parts.append(
            f"Resolved {len(companies)} companies ({with_domain} with domain)"
        )
        logger.info(f"Lead gen: {len(companies)} companies, {with_domain} with domain")
    except asyncio.TimeoutError:
        logger.warning("Lead gen: domain resolution timed out")
        deps._companies = []
        agent_reasoning_parts.append("Domain resolution timed out")
    except Exception as e:
        logger.error(f"Lead gen: company build failed: {e}")
        deps._companies = []
        agent_reasoning_parts.append(f"Company build error: {e}")

    if deps._companies:
        try:
            from app.agents.workers.contact_agent import ContactFinder
            from app.schemas import AgentState

            finder = ContactFinder(mock_mode=deps.mock_mode, deps=deps)
            state = AgentState(
                trends=deps._trend_data,
                impacts=deps._impacts,
                companies=deps._companies,
            )
            result = await asyncio.wait_for(
                finder.find_contacts(state), timeout=180.0,
            )
            deps._contacts = result.contacts or []
            with_email = sum(1 for c in deps._contacts if getattr(c, "email", ""))
            agent_reasoning_parts.append(
                f"Found {len(deps._contacts)} contacts ({with_email} with email)"
            )
        except asyncio.TimeoutError:
            logger.warning("Lead gen: contact search timed out")
            agent_reasoning_parts.append("Contact search timed out")
        except Exception as e:
            logger.error(f"Lead gen: contact search failed: {e}")
            agent_reasoning_parts.append(f"Contact error: {e}")

    if deps._contacts:
        try:
            from app.agents.workers.email_agent import EmailGenerator
            from app.schemas import AgentState

            generator = EmailGenerator(mock_mode=deps.mock_mode, deps=deps)
            state = AgentState(
                trends=deps._trend_data,
                companies=deps._companies,
                contacts=deps._contacts,
            )
            result = await asyncio.wait_for(
                generator.process_emails(state), timeout=180.0,
            )
            deps._contacts = result.contacts or deps._contacts
            deps._outreach = result.outreach_emails or []
            agent_reasoning_parts.append(
                f"Generated {len(deps._outreach)} outreach emails"
            )
        except asyncio.TimeoutError:
            logger.warning("Lead gen: email generation timed out")
            agent_reasoning_parts.append("Email generation timed out")
        except Exception as e:
            logger.error(f"Lead gen: email generation failed: {e}")
            agent_reasoning_parts.append(f"Email error: {e}")

    logger.info(
        f"Lead gen step 4: building profiles from "
        f"{len(deps._contacts)} contacts, {len(deps._outreach)} outreach, "
        f"{len(deps._impacts)} impacts, {len(deps._companies)} companies"
    )
    try:
        profiles = _build_person_profiles(
            contacts=deps._contacts,
            outreach_emails=deps._outreach,
            impacts=deps._impacts,
            companies=deps._companies,
        )
        deps._person_profiles = profiles
        dm_count = sum(1 for p in profiles if p.seniority_tier == "decision_maker")
        inf_count = sum(1 for p in profiles if p.seniority_tier == "influencer")
        logger.info(f"Lead gen step 4: built {len(profiles)} profiles ({dm_count} DMs, {inf_count} inf)")
        agent_reasoning_parts.append(
            f"Built {len(profiles)} person profiles ({dm_count} DMs, {inf_count} influencers)"
        )
    except Exception as e:
        logger.error(f"Lead gen: person profile build failed: {e}", exc_info=True)
        deps._person_profiles = []
        agent_reasoning_parts.append(f"Profile build error: {e}")

    agent_result = LeadGenResult(
        companies_found=len(deps._companies),
        contacts_found=len(deps._contacts),
        emails_generated=len(deps._outreach),
        outreach_generated=len(deps._outreach),
        reasoning=" | ".join(agent_reasoning_parts) or "No data produced",
    )

    logger.info(
        f"Lead gen complete: {agent_result.companies_found} companies, "
        f"{agent_result.contacts_found} contacts, "
        f"{agent_result.outreach_generated} outreach"
    )

    return deps._companies, deps._contacts, deps._outreach, agent_result
