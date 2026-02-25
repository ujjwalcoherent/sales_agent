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


class TrendValidationLLM(BaseModel):
    """LLM output for trend validation (Stage A council)."""
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    validated_depth: str = Field(default="SUB", description="MAJOR, SUB, MICRO, or NOISE")
    reasoning: str = Field(default="", description="2-3 sentences explaining classification")
    cmi_relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    relevant_services: List[str] = Field(default_factory=list)
    should_subcluster: bool = False
    subcluster_reason: str = ""
    validated_event_type: str = "general"
    event_type_reasoning: str = ""


class LeadValidationLLM(BaseModel):
    """LLM output for lead quality validation (Stage C council)."""
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    is_relevant: bool = True
    reasoning: str = ""
    improved_pitch: str = ""
    recommended_service: str = ""
    recommended_offering: str = ""


class ImpactAnalysisLLM(BaseModel):
    """LLM output for impact analysis (per trend)."""
    direct_impact: List[str] = Field(default_factory=list)
    direct_impact_reasoning: str = ""
    indirect_impact: List[str] = Field(default_factory=list)
    indirect_impact_reasoning: str = ""
    additional_verticals: List[str] = Field(default_factory=list)
    additional_verticals_reasoning: str = ""
    midsize_pain_points: List[str] = Field(default_factory=list)
    consulting_projects: List[str] = Field(default_factory=list)
    positive_sectors: List[str] = Field(default_factory=list)
    negative_sectors: List[str] = Field(default_factory=list)
    relevant_services: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    pitch_angle: str = ""


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


# ── Stage 0: Article Triage (pre-clustering noise filter) ─────────────

class ArticleTriageItemLLM(BaseModel):
    """LLM judgment for a single article in a batch triage call."""
    id: int = Field(description="Article number from the batch (1-indexed)")
    is_business: bool = Field(
        default=True,
        description="True if article discusses business, finance, companies, markets, "
        "economy, technology adoption, or corporate/government activity"
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    noise_category: str = Field(
        default="none",
        description="If not business: entertainment, sports, lifestyle, opinion, "
        "astrology, or other. Use 'none' if business-relevant."
    )
    reasoning: str = Field(default="", description="1 sentence explaining the decision")


class ArticleTriageBatchLLM(BaseModel):
    """LLM output for batch article triage (10-15 articles per call).

    Reasoning field comes FIRST to preserve chain-of-thought quality
    (OpenAI Structured Outputs research, 2024).
    """
    reasoning: str = Field(
        default="",
        description="Brief overall assessment of this batch before individual judgments"
    )
    articles: List[ArticleTriageItemLLM] = Field(default_factory=list)


class CompanyExtractionLLM(BaseModel):
    """Single company extracted by LLM from search results."""
    company_name: str = ""
    industry: str = ""
    website: str = ""
    reason_relevant: str = ""
    company_size: str = "mid"
    intent_signal: str = ""
    description: str = ""


class CompanyListLLM(BaseModel):
    """LLM output for company extraction (list of companies).

    Handles two LLM output patterns:
      - Wrapped:  {"companies": [{...}, ...]}        (expected)
      - Flat list: [{...}, ...]                       (LLM shortcut)
    """
    companies: List[CompanyExtractionLLM] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, list):
            return {"companies": v}
        return v


class OutreachDraftLLM(BaseModel):
    """LLM output for email outreach generation."""
    subject: str = ""
    body: str = ""


class TrendSynthesisLLM(BaseModel):
    """LLM output for cluster synthesis (Phase 8)."""
    trend_title: str = ""
    trend_summary: str = ""
    trend_type: str = "general"
    severity: str = "medium"
    lifecycle_stage: str = "emerging"
    primary_sectors: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    affected_companies: List[str] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    actionable_insight: str = ""
    event_5w1h: dict = Field(default_factory=dict)
    causal_chain: List[str] = Field(default_factory=list)
    buying_intent: dict = Field(default_factory=dict)


# ── Stage D: Causal Council (multi-agent causal reasoning) ────────────

# ── Layer 2.75: LLM Cluster Validation ────────────────────────────────

class ClusterValidationLLM(BaseModel):
    """LLM output for cluster coherence validation.

    Reasoning-first pattern (OpenAI Structured Outputs, 2024).
    Used in curriculum-learning cascade: deterministic checks first,
    LLM only for borderline clusters. Cost: ~$0.0003/cluster.

    REF: NewsCatcher rejects 80% of clusters via LLM validation.
    """
    reasoning: str = Field(
        default="",
        description="Step-by-step analysis: (1) Do these articles describe the SAME "
        "real-world event or closely related developments? (2) What specific "
        "evidence links them? (3) Are there any outlier articles that don't "
        "belong? (4) Could this cluster be split into tighter sub-topics?"
    )
    is_coherent: bool = Field(
        default=True,
        description="True if all articles discuss the same event/development. "
        "False if the cluster is a grab-bag of loosely related topics."
    )
    coherence_score: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="0.0=completely incoherent grab-bag, 1.0=perfect single-event cluster"
    )
    suggested_label: str = Field(
        default="",
        description="A concise label for this cluster (e.g., 'Peak XV $1.3B Fund Raise', "
        "'Trump Tariff Market Impact')"
    )
    outlier_indices: List[int] = Field(
        default_factory=list,
        description="0-indexed positions of articles that don't belong in this cluster"
    )
    should_split: bool = Field(
        default=False,
        description="True if the cluster contains 2+ distinct sub-topics that "
        "should be separated into their own clusters"
    )
    split_reason: str = Field(
        default="",
        description="If should_split=True, explain what sub-topics exist"
    )


class CausalEdgeLLM(BaseModel):
    """LLM output for pairwise causal evaluation between two trends.

    The reasoning field comes FIRST to preserve chain-of-thought quality
    (OpenAI Structured Outputs research, 2024).
    """
    reasoning: str = Field(
        default="",
        description="Step-by-step analysis: (1) What evidence links these trends? "
        "(2) Is the connection causal or coincidental? "
        "(3) What is the mechanism? (4) What direction?"
    )
    is_causal: bool = Field(
        default=False,
        description="True if evidence supports a causal link, not just co-occurrence"
    )
    relationship_type: str = Field(
        default="co-occurs",
        description="causes | amplifies | mitigates | co-occurs"
    )
    causal_mechanism: str = Field(
        default="",
        description="HOW Trend A affects Trend B — the specific transmission channel "
        "(e.g., 'RBI rate hike → higher NBFC borrowing costs → reduced lending')"
    )
    direction: str = Field(
        default="a_to_b",
        description="a_to_b | b_to_a | bidirectional"
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.3)
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description="Direct quotes or facts from articles that support this causal link"
    )
    business_implication: str = Field(
        default="",
        description="What this causal link means for affected companies RIGHT NOW"
    )


class CascadeNarrativeLLM(BaseModel):
    """LLM output for multi-hop cascade narrative (3+ linked trends).

    Produces a coherent chain-reaction story showing how Trend A → B → C
    with specific evidence at each hop.
    """
    reasoning: str = Field(
        default="",
        description="Step-by-step analysis of the full cascade chain"
    )
    cascade_narrative: str = Field(
        default="",
        description="Plain-English paragraph explaining the full chain reaction "
        "(e.g., 'Silver price surge → jewellery manufacturers face cost pressure "
        "→ downstream retail pricing disruption → consumer demand shift to gold')"
    )
    hop_mechanisms: List[str] = Field(
        default_factory=list,
        description="One mechanism per hop in the cascade "
        "(e.g., ['Silver price surge drives raw material costs up 12%', "
        "'Cost pressure forces jewellery manufacturers to raise retail prices', ...])"
    )
    cascade_confidence: float = Field(ge=0.0, le=1.0, default=0.3)
    weakest_link: str = Field(
        default="",
        description="Which hop in the chain has the weakest evidence?"
    )
    compound_business_impact: str = Field(
        default="",
        description="The COMBINED business effect of the full cascade — "
        "what decision must companies make knowing the full chain?"
    )
