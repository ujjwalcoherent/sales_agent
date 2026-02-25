"""
Lead Gen Agent — autonomous company discovery, contact finding, and pitch generation.

Combines the previous CompanyDiscovery + ContactFinder + EmailGenerator
into a single pydantic-ai agent that autonomously decides:
- Search strategy per trend (NER-based vs industry-based)
- How many companies to pursue per trend
- Which roles to target based on trend type
- Whether a company-trend fit is strong enough

Tools:
  - find_companies_for_trends: Discover companies affected by trends
  - find_contacts: Find decision-makers at target companies
  - generate_outreach: Create personalized pitch emails
  - assess_company_relevance: Score company-trend fit via bandit
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.agent_deps import AgentDeps

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


# ── System prompt ─────────────────────────────────────────────────────

LEAD_GEN_PROMPT = """\
You are the Lead Gen Agent for a sales intelligence pipeline. Your job is to \
find companies, contacts, and generate personalized outreach for each trend.

WORKFLOW:
1. Find companies affected by each high-confidence trend using the company finder.
2. Assess company-trend relevance — skip weak fits to save API calls.
3. Find decision-maker contacts at promising companies.
4. Generate personalized outreach emails connecting the trend to the company's needs.

QUALITY RULES:
- Only pursue companies where the trend creates a genuine pain point.
- Target the right role: tech trends → CTO/VP Engineering, regulatory → GC/CCO, \
  market shifts → CEO/CSO.
- Personalize emails with specific trend details — no generic templates.
- Quality over quantity: 5 great leads > 20 mediocre ones.
"""


# ── Agent definition ──────────────────────────────────────────────────

lead_gen_agent = Agent(
    'test',  # Placeholder — overridden at runtime via deps.get_model()
    deps_type=AgentDeps,
    output_type=LeadGenResult,
    system_prompt=LEAD_GEN_PROMPT,
    retries=2,
)


# ── Tools ─────────────────────────────────────────────────────────────

@lead_gen_agent.tool
async def find_companies_for_trends(ctx: RunContext[AgentDeps]) -> str:
    """Find companies affected by each analyzed trend.

    Uses the existing CompanyDiscovery pipeline with NER-based
    hallucination guard and Wikipedia verification.
    """
    from app.agents.company_agent import CompanyDiscovery
    from app.schemas import AgentState

    trends = ctx.deps._trend_data
    impacts = ctx.deps._impacts
    if not trends:
        return "ERROR: No trends available."

    discovery = CompanyDiscovery(
        mock_mode=ctx.deps.mock_mode,
        deps=ctx.deps,
        log_callback=ctx.deps.log_callback,
    )

    state = AgentState(trends=trends, impacts=impacts)
    result = await discovery.find_companies(state)

    ctx.deps._companies = result.companies or []
    verified = sum(1 for c in ctx.deps._companies if getattr(c, 'ner_verified', False))

    return (
        f"Found {len(ctx.deps._companies)} companies across {len(trends)} trends. "
        f"NER-verified: {verified}. Errors: {len(result.errors)}"
    )


@lead_gen_agent.tool
async def find_contacts_for_companies(ctx: RunContext[AgentDeps]) -> str:
    """Find decision-maker contacts at target companies.

    Uses Apollo (primary) + web search (fallback) to find
    name, role, LinkedIn, and email.
    """
    from app.agents.contact_agent import ContactFinder
    from app.schemas import AgentState

    companies = ctx.deps._companies
    trends = ctx.deps._trend_data
    impacts = ctx.deps._impacts
    if not companies:
        return "No companies to find contacts for."

    finder = ContactFinder(mock_mode=ctx.deps.mock_mode, deps=ctx.deps)
    state = AgentState(
        trends=trends,
        impacts=impacts,
        companies=companies,
    )
    result = await finder.find_contacts(state)

    ctx.deps._contacts = result.contacts or []
    with_email = sum(1 for c in ctx.deps._contacts if getattr(c, 'email', ''))

    return (
        f"Found {len(ctx.deps._contacts)} contacts "
        f"({with_email} with email). "
        f"Errors: {len(result.errors)}"
    )


@lead_gen_agent.tool
async def generate_outreach_emails(ctx: RunContext[AgentDeps]) -> str:
    """Generate personalized outreach emails for all contacts.

    Creates pitch emails connecting the specific trend to the
    company's pain points and CMI's relevant services.
    """
    from app.agents.email_agent import EmailGenerator
    from app.schemas import AgentState

    contacts = ctx.deps._contacts
    companies = ctx.deps._companies
    trends = ctx.deps._trend_data
    if not contacts:
        return "No contacts to generate outreach for."

    generator = EmailGenerator(mock_mode=ctx.deps.mock_mode, deps=ctx.deps)
    state = AgentState(
        trends=trends,
        companies=companies,
        contacts=contacts,
    )
    result = await generator.process_emails(state)

    ctx.deps._contacts = result.contacts or contacts
    ctx.deps._outreach = result.outreach_emails or []

    return (
        f"Generated {len(ctx.deps._outreach)} outreach emails. "
        f"Errors: {len(result.errors)}"
    )


@lead_gen_agent.tool
async def assess_company_relevance(
    ctx: RunContext[AgentDeps],
    company_size: str,
    event_type: str,
    industry_match: float = 0.5,
    trend_severity: str = "medium",
) -> str:
    """Assess how relevant a company-trend pairing is using the bandit.

    Uses Thompson Sampling contextual bandit for company relevance scoring.
    Higher scores = better fit between company profile and trend characteristics.

    Args:
        company_size: Company size category ("startup", "mid", "large", "enterprise").
        event_type: Trend event type ("regulation", "funding", "technology", etc.).
        industry_match: Overlap between company industry and trend sectors (0.0-1.0).
        trend_severity: Trend severity ("high", "medium", "low").
    """
    bandit = ctx.deps.company_bandit
    score = bandit.compute_relevance(
        company_size=company_size,
        event_type=event_type,
        industry_match=industry_match,
        trend_severity=trend_severity,
    )
    return (
        f"Company relevance ({company_size}, {event_type}): {score:.3f} "
        f"(industry_match={industry_match:.2f}, severity={trend_severity})"
    )


# ── Public runner ─────────────────────────────────────────────────────

async def run_lead_gen(deps: AgentDeps) -> tuple:
    """Run the Lead Gen Agent. Returns (companies, contacts, outreach, result)."""
    trends = deps._trend_data
    impacts = deps._impacts

    prompt = (
        f"Find companies and contacts for {len(trends)} trends "
        f"({len(impacts)} with impact analysis). "
        f"Generate personalized outreach emails."
    )

    try:
        model = deps.get_model()
        result = await lead_gen_agent.run(prompt, deps=deps, model=model)

        companies = deps._companies
        contacts = deps._contacts
        outreach = deps._outreach

        logger.info(
            f"Lead Gen Agent: {len(companies)} companies, "
            f"{len(contacts)} contacts, {len(outreach)} outreach"
        )
        return companies, contacts, outreach, result.output

    except Exception as e:
        logger.error(f"Lead Gen Agent failed: {e}")
        # Fallback: run each stage directly
        from app.agents.company_agent import run_company_agent
        from app.agents.contact_agent import run_contact_agent
        from app.agents.email_agent import run_email_agent
        from app.schemas import AgentState

        state = AgentState(trends=trends, impacts=impacts)
        state = await run_company_agent(state, deps=deps)
        state = await run_contact_agent(state, deps=deps)
        state = await run_email_agent(state, deps=deps)

        fallback = LeadGenResult(reasoning=f"Fallback: {e}")
        return state.companies, state.contacts, state.outreach_emails, fallback
