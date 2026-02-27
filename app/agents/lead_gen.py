"""
Lead Gen Agent — enriches pre-resolved lead sheets with contacts and emails.

The crystallizer (step 3.8) already resolves real company names via SearXNG +
LLM extraction.  This agent's job is narrower and faster:

1. Read lead sheets from deps._lead_sheets (populated by crystallize step)
2. For each unique company: find domain via quick SearXNG search
3. Build CompanyData objects (needed by ContactFinder / EmailGenerator)
4. Run ContactFinder per company (Apollo primary, SearXNG fallback)
5. Run EmailGenerator per contact (Apollo → Hunter → pattern)

Expected time: 60-120s (down from 490s when discovering companies from scratch).
"""

import asyncio
import hashlib
import logging
from typing import Any, List

from pydantic import BaseModel

from app.agents.deps import AgentDeps

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────

class LeadGenResult(BaseModel):
    """Structured output from the Lead Gen Agent."""
    companies_found: int = 0
    contacts_found: int = 0
    emails_generated: int = 0
    outreach_generated: int = 0
    low_relevance_filtered: int = 0
    reasoning: str = ""


# ── Domain resolution ────────────────────────────────────────────────

async def _resolve_domain(search_manager, company_name: str) -> str:
    """Find a company's domain via SearXNG search."""
    try:
        data = await search_manager.web_search(
            f"{company_name} official website India", max_results=3,
        )
        from app.tools.domain_utils import extract_clean_domain
        for r in data.get("results", []):
            url = r.get("url", "")
            if url:
                domain = extract_clean_domain(url)
                if domain:
                    return domain
    except Exception as e:
        logger.debug(f"Domain resolution for '{company_name}' failed: {e}")
    return ""


async def _build_companies_from_leads(
    deps: AgentDeps,
    lead_sheets: list,
) -> list:
    """Build CompanyData objects from unique company names in lead sheets.

    Resolves domains via SearXNG for each unique company.
    """
    from app.schemas.sales import CompanyData

    seen: dict[str, Any] = {}  # company_name -> CompanyData
    sem = asyncio.Semaphore(5)

    async def _build_one(name: str, lead):
        if name in seen:
            return
        async with sem:
            domain = await _resolve_domain(deps.search_manager, name)
            cid = hashlib.md5(name.encode()).hexdigest()[:12]
            seen[name] = CompanyData(
                id=cid,
                company_name=name,
                industry=lead.event_type or "general",
                domain=domain,
                website=f"https://{domain}" if domain else "",
                reason_relevant=lead.pain_point or lead.trend_title,
                trend_id=str(lead.trend_title),
            )
            if domain:
                logger.info(f"  Resolved domain: {name} → {domain}")
            else:
                logger.debug(f"  No domain found for: {name}")

    tasks = []
    for lead in lead_sheets:
        name = lead.company_name
        if name and not name.startswith("[") and name not in seen:
            tasks.append(_build_one(name, lead))

    if tasks:
        await asyncio.gather(*tasks)

    return list(seen.values())


# ── Person profile builder ────────────────────────────────────────────

def _build_person_profiles(
    contacts: list,
    outreach_emails: list,
    impacts: list,
    companies: list,
) -> list:
    """Build PersonProfile objects from contacts + outreach, with reach scores.

    Each contact becomes a PersonProfile with:
    - seniority_tier (from role classification)
    - reach_score (0-100 composite)
    - outreach_tone (executive/consultative/professional)
    - outreach_subject/body (if email was generated for this contact)
    """
    from app.schemas.sales import PersonProfile
    from app.agents.workers.contact_agent import ContactFinder

    # Build outreach index: contact_id -> OutreachEmail
    outreach_by_contact = {}
    for em in outreach_emails:
        cid = getattr(em, "contact_id", "")
        if cid:
            outreach_by_contact[cid] = em

    # Build impact index: trend_id -> target_roles set (lowercased)
    impact_roles: dict[str, set] = {}
    for imp in impacts:
        tid = getattr(imp, "trend_id", "")
        roles = {r.lower() for r in getattr(imp, "target_roles", [])}
        impact_roles[tid] = roles

    # Build company -> trend_id map
    company_trend: dict[str, str] = {}
    for c in companies:
        company_trend[getattr(c, "company_name", "")] = getattr(c, "trend_id", "")

    profiles = []
    for ct in contacts:
        try:
            tier = ContactFinder.classify_tier(ct.role)
            tone = ContactFinder.get_outreach_tone(tier)

            # Role relevance: does this role match impact target_roles?
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

    # Sort: decision_makers first, then by reach_score descending
    tier_order = {"decision_maker": 0, "influencer": 1, "gatekeeper": 2}
    profiles.sort(key=lambda p: (tier_order.get(p.seniority_tier, 1), -p.reach_score))

    return profiles


# ── Public runner ─────────────────────────────────────────────────────

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

    # Step 1: Build CompanyData from lead sheets (with domain resolution)
    try:
        companies = await asyncio.wait_for(
            _build_companies_from_leads(deps, lead_sheets),
            timeout=120.0,
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

    # Step 2: Find contacts (Apollo primary, SearXNG fallback)
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

    # Step 3: Find emails and generate outreach
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

    # Step 4: Build person profiles with reach scores and outreach tones
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
        agent_reasoning_parts.append(
            f"Built {len(profiles)} person profiles ({dm_count} DMs, {inf_count} influencers)"
        )
    except Exception as e:
        logger.error(f"Lead gen: person profile build failed: {e}")
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
