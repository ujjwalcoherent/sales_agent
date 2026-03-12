"""
Inner Pydantic models for LLM structured output.

These models define ONLY what the LLM produces — fields set programmatically
(trend_id, trend_title, etc.) are excluded. Used with LLMService.run_structured()
to get typed, validated output with automatic retry on validation failure.

Convention: Suffix with "LLM" to distinguish from the full output schemas.
"""

from typing import Annotated, List, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic.functional_validators import BeforeValidator


def _coerce_to_str(v):
    if isinstance(v, dict):
        # Try common keys in order of preference
        for key in ("fact", "citation", "company_type", "problem", "project", "opportunity", "text", "value"):
            if key in v and v[key]:
                return str(v[key])
        # Fallback: join all string values
        vals = [str(x) for x in v.values() if x and isinstance(x, (str, int, float))]
        return " — ".join(vals) if vals else str(v)
    return str(v) if v is not None else ""

StrFromDict = Annotated[str, BeforeValidator(_coerce_to_str)]


def _coerce_to_str_list(v):
    """Coerce common LLM output variants to List[str].

    Handles: None→[], str→[str], dict→[str(dict)], non-list→[str(val)].
    Filters empty items from existing lists.
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(item).strip() for item in v if item and str(item).strip()]
    return [str(v)] if v else []


StrList = Annotated[List[str], BeforeValidator(_coerce_to_str_list)]


class ImpactAnalysisLLM(BaseModel):
    """LLM output for impact analysis (per trend)."""
    direct_impact: StrList = Field(default_factory=list)
    direct_impact_reasoning: str = ""
    indirect_impact: StrList = Field(default_factory=list)
    indirect_impact_reasoning: str = ""
    additional_verticals: StrList = Field(default_factory=list)
    additional_verticals_reasoning: str = ""
    midsize_pain_points: StrList = Field(default_factory=list)
    consulting_projects: StrList = Field(default_factory=list)
    positive_sectors: StrList = Field(default_factory=list)
    negative_sectors: StrList = Field(default_factory=list)
    relevant_services: StrList = Field(default_factory=list)
    target_roles: StrList = Field(default_factory=list)
    pitch_angle: str = ""

    @model_validator(mode='before')
    @classmethod
    def coerce_str_fields(cls, values):
        """Coerce string fields from non-str types (None, int, list)."""
        if not isinstance(values, dict):
            return values
        for key in ("direct_impact_reasoning", "indirect_impact_reasoning",
                     "additional_verticals_reasoning", "pitch_angle"):
            val = values.get(key)
            if val is not None and not isinstance(val, str):
                values[key] = str(val)
            elif val is None:
                values[key] = ""
        # Truncate pitch_angle
        pitch = values.get("pitch_angle", "")
        if isinstance(pitch, str) and len(pitch) > 150:
            values["pitch_angle"] = pitch[:147] + "..."
        return values


class ServiceRecommendationLLM(BaseModel):
    """Single CMI service recommendation from LLM."""
    service: str = ""
    offering: str = ""
    justification: str = ""
    urgency: str = "medium"


class UnifiedImpactAnalysisLLM(BaseModel):
    """LLM output for unified multi-perspective impact analysis.

    Replaces the 13-call council (4 agents x 3 rounds + moderator) with a
    single structured call. Research backing (ICML 2024, ICLR 2025):
    - Multi-agent debate does not reliably outperform single-agent baselines
    - Embedding all perspectives in one prompt achieves comparable quality
    - Adding a reasoning field before answer fields improves accuracy by 60%
    """
    # Chain-of-thought FIRST — research shows this preserves reasoning quality
    # in structured output mode (OpenAI Structured Outputs docs, 2024)
    reasoning: str = Field(
        default="",
        description="Step-by-step multi-perspective analysis: industry structure, "
        "strategy implications, risks, and evidence quality"
    )

    # ── FIRST-ORDER: companies/segments DIRECTLY named in the source articles ──
    # Rule: only include if the company type or segment is explicitly mentioned
    # in the source data. These are the IMMEDIATE casualties/beneficiaries.
    first_order_companies: List[StrFromDict] = Field(
        default_factory=list,
        description="Company types DIRECTLY named or explicitly mentioned in source articles. "
        "Rule: only include if you can cite the specific article. "
        "Example: 'Silver jewellery exporters in Rajkot (named in ET article)' — NOT 'all manufacturers'"
    )
    first_order_mechanism: str = Field(
        default="",
        description="The direct causal mechanism: HOW the event hits first-order companies. "
        "Be specific: 'Silver import duty increased from 7.5% to 12.5% → direct cost increase for silver importers'"
    )

    # ── SECOND-ORDER: companies indirectly affected through supply chain/market ──
    # Rule: must name the CHAIN — how first-order effects propagate downstream.
    second_order_companies: List[StrFromDict] = Field(
        default_factory=list,
        description="Company types INDIRECTLY affected through supply chain, customer base, "
        "or market dynamics. Must state the chain: 'Silver jewellers raise prices → "
        "wedding venue coordinators face lower budgets → event management firms impacted'. "
        "These are NOT in the source articles — they are inferred from business logic."
    )
    second_order_mechanism: str = Field(
        default="",
        description="The indirect causal chain: 'First-order effect X propagates to second-order "
        "because [mechanism]. The transmission channel is [supply chain / pricing / demand / regulation].'"
    )

    # Industry analyst perspective (all affected — union of first + second order)
    affected_company_types: List[StrFromDict] = Field(default_factory=list)
    affected_sectors: List[StrFromDict] = Field(default_factory=list)

    # Strategy consultant perspective
    pain_points: List[StrFromDict] = Field(default_factory=list)
    consulting_projects: List[StrFromDict] = Field(default_factory=list)
    business_opportunities: List[StrFromDict] = Field(default_factory=list)

    # Risk analyst perspective
    detailed_reasoning: str = Field(
        default="",
        description="5-paragraph reasoning: what happened, who is affected (first-order), "
        "who is indirectly affected (second-order), decisions they face, urgency"
    )

    # Market researcher perspective
    evidence_citations: List[StrFromDict] = Field(default_factory=list)

    # Service mapping
    service_recommendations: List[ServiceRecommendationLLM] = Field(default_factory=list)

    # Actionable output
    pitch_angle: str = ""
    target_roles: List[StrFromDict] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class OutreachDraftLLM(BaseModel):
    """LLM output for email outreach generation."""
    subject: str = ""
    body: str = ""


class PersonOutreachInsightsLLM(BaseModel):
    """LLM output for person outreach intelligence synthesis."""
    background_summary: str = ""
    recent_focus: str = ""
    notable_achievements: StrList = Field(default_factory=list)
    shared_interests: StrList = Field(default_factory=list)
    talking_points: StrList = Field(default_factory=list)


class ContentThemesLLM(BaseModel):
    """LLM output for person content theme extraction.

    Handles both patterns:
      - Array:  ["theme1", "theme2"]            (LLM returns raw list)
      - Object: {"themes": ["theme1", ...]}     (wrapped)
    """
    themes: StrList = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, list):
            return {"themes": v}
        return v


class ContactRolesLLM(BaseModel):
    """LLM output for contact role inference.

    Handles both patterns:
      - Array:  ["CEO", "VP Engineering"]       (LLM returns raw list)
      - Object: {"roles": ["CEO", ...]}         (wrapped)
    """
    roles: StrList = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, list):
            return {"roles": v}
        return v


class ResolvedCompanyLLM(BaseModel):
    """Single company resolved from segment search results."""
    name: str = ""
    city: str = ""
    state: str = ""
    size_band: str = "sme"


class SegmentResolutionLLM(BaseModel):
    """Segment-level company resolution result."""
    index: int = 0
    companies: List[ResolvedCompanyLLM] = Field(default_factory=list)


class SegmentResolutionListLLM(BaseModel):
    """LLM output for batch company resolution from segment search results.

    Handles both patterns:
      - Array:  [{index, companies: [...]}, ...]   (LLM returns raw list)
      - Object: {"segments": [{...}]}              (wrapped)
    """
    segments: List[SegmentResolutionLLM] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, list):
            return {"segments": v}
        return v


class ProductServicesLLM(BaseModel):
    """LLM output for company products/services extraction.

    Handles both patterns:
      - Array:  ["Product A", "Service B"]      (LLM returns raw list)
      - Object: {"products": ["Product A",...]} (wrapped)
    """
    products: StrList = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, list):
            return {"products": v}
        # Also try common dict keys
        if isinstance(v, dict):
            for key in ("products", "services", "solutions", "offerings", "items"):
                if key in v:
                    return {"products": v[key]}
        return v


class ReportEntitiesLLM(BaseModel):
    """LLM output for report entity extraction (report-driven pipeline).

    Extracts company names, industries, and topics from analyst report text.
    Coercion via StrList handles common LLM quirks (None→[], str→[str]).
    """
    companies: StrList = Field(default_factory=list)
    industries: StrList = Field(default_factory=list)
    topics: StrList = Field(default_factory=list)


class CompanyFieldsLLM(BaseModel):
    """LLM output for structured company field extraction.

    All fields optional — the LLM fills whatever it can find in the source text.
    StrList coercion handles common LLM quirks (comma-separated strings, None, etc.).
    """
    industry: str = ""
    headquarters: str = ""
    ceo: str = ""
    employee_count: str = ""
    founded_year: Optional[int] = None
    products_services: StrList = Field(default_factory=list)
    competitors: StrList = Field(default_factory=list)
    sub_industries: StrList = Field(default_factory=list)
    tech_stack: StrList = Field(default_factory=list)
    investors: StrList = Field(default_factory=list)
    funding_stage: str = ""
    revenue: str = ""
    stock_ticker: str = ""

    @model_validator(mode='before')
    @classmethod
    def coerce_str_fields(cls, values):
        """Coerce None/int/list to str for string fields."""
        if not isinstance(values, dict):
            return values
        for key in ("industry", "headquarters", "ceo", "employee_count",
                     "funding_stage", "revenue", "stock_ticker"):
            val = values.get(key)
            if val is not None and not isinstance(val, str):
                values[key] = str(val)
            elif val is None:
                values[key] = ""
        return values


class HiringSignalsLLM(BaseModel):
    """LLM output for company hiring signal extraction."""
    hiring_signals: StrList = Field(default_factory=list)


class TechIpLLM(BaseModel):
    """LLM output for company tech stack and IP intelligence."""
    tech_stack: StrList = Field(default_factory=list)
    patents: StrList = Field(default_factory=list)
    partnerships: StrList = Field(default_factory=list)
