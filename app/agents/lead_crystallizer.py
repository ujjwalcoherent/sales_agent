"""
Lead Crystallizer — converts causal chain hops into concrete call sheets.

Input:  CausalChainResult (from causal_council.py)
Output: List[LeadSheet] — each a complete sales call sheet with company name,
        contact role, opening line, pain point, service pitch, and urgency.

This is the "last mile" of sales intelligence — what actually gets handed
to the sales team to make calls. No more "which segment might be affected" —
instead: "Call Rohan at Precision Parts Pvt Ltd, Pune, by next Tuesday."

Self-learning integration:
  - oss_score from source trend is carried on each LeadSheet
  - leads_with_companies vs leads_generated ratio = kb_hit_rate signal
  - These feed back into LearningSignal → source bandit → weight learner
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

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

    # Company-specific recent news (from SearXNG news search)
    company_news: list[dict] = Field(default_factory=list, description="Recent news about this company")

    # Self-learning signals
    data_sources: list[str] = []    # ["mca_kb", "bm25_article", "causal_inference"]
    oss_score: float = 0.0          # OSS of source synthesis (for source bandit)


# ── Contact role and service pitch mappings ────────────────────────────────────

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


# ── Main entry point ──────────────────────────────────────────────────────────

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

        # ── Get company list ──────────────────────────────────────────────────
        companies: list[str] = list(hop.companies_found) if hop.companies_found else []

        # Placeholder: generate segment-level lead even without a named company
        if not companies:
            companies = [f"[{hop.segment}]"]

        # ── Build leads for this hop ──────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _opening(company: str, trend: str, mechanism: str, lead_type: str, weeks: int) -> str:
    """Build a specific, event-driven opening line for the call or email."""
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
    """Extract first city/state from comma-separated geo hint."""
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
