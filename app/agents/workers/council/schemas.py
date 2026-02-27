"""
Pydantic schemas for AI Council input/output.

Every council decision carries structured reasoning so the pipeline
can explain WHY each value exists.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Stage A: Trend Validation ──────────────────────────────────────────

class TrendValidation(BaseModel):
    """AI-validated trend classification (replaces volume-based hierarchy)."""
    trend_id: str
    importance_score: float = Field(ge=0.0, le=1.0, description="How significant for business consulting")
    validated_depth: str = Field(description="MAJOR, SUB, MICRO, or NOISE")
    reasoning: str = Field(description="2-3 sentences explaining the classification")
    cmi_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How relevant to CMI's 9 services")
    relevant_services: List[str] = Field(default_factory=list, description="CMI services applicable to this trend")
    should_subcluster: bool = Field(default=False, description="Whether this trend should be sub-clustered")
    subcluster_reason: str = ""
    validated_event_type: str = ""
    event_type_reasoning: str = ""


# ── Stage B: Impact Analysis Council ───────────────────────────────────

class CouncilPerspective(BaseModel):
    """Single specialist agent's analysis of a trend."""
    agent_role: str = Field(description="industry_analyst, strategy_consultant, risk_analyst, or market_researcher")
    analysis: str = Field(description="Multi-paragraph analysis from this perspective")
    key_findings: List[str] = Field(default_factory=list)
    affected_company_types: List[str] = Field(default_factory=list)
    recommended_services: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_citations: List[str] = Field(default_factory=list)


class ServiceRecommendation(BaseModel):
    """Specific CMI service recommendation with justification."""
    service_name: str
    offering: str
    justification: str
    urgency: str = "medium"  # low, medium, high, critical


class ImpactCouncilResult(BaseModel):
    """Combined output from the 4-agent impact council + moderator."""
    perspectives: List[CouncilPerspective] = Field(default_factory=list)

    # Moderator synthesis
    consensus_reasoning: str = Field(default="", description="Multi-paragraph synthesis of all perspectives")
    debate_summary: str = Field(default="", description="Key disagreements and how they were resolved")

    # Actionable output
    detailed_reasoning: str = ""
    pitch_angle: str = ""
    service_recommendations: List[ServiceRecommendation] = Field(default_factory=list)
    evidence_citations: List[str] = Field(default_factory=list)

    # Scores
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # Impact breakdown
    affected_sectors: List[str] = Field(default_factory=list)
    affected_company_types: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    business_opportunities: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)


# ── Stage C: Lead Quality Validation ───────────────────────────────────

class LeadValidation(BaseModel):
    """AI-validated company-trend fit."""
    company_name: str
    trend_title: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    is_relevant: bool = True
    reasoning: str = ""
    improved_pitch: str = ""
    recommended_service: str = ""
    recommended_offering: str = ""


# ── Stage D: Causal Council ───────────────────────────────────────────

class CausalEdgeResult(BaseModel):
    """Validated causal edge between two trends (output of Causal Council).

    Each edge represents a real cause-effect relationship backed by article
    evidence, not just statistical co-occurrence.
    """
    source_node_id: str = ""        # TrendNode UUID (the cause)
    target_node_id: str = ""        # TrendNode UUID (the effect)
    source_title: str = ""          # Human-readable
    target_title: str = ""
    relationship_type: str = "co-occurs"   # causes | amplifies | mitigates | co-occurs
    causal_mechanism: str = ""      # HOW A affects B
    strength: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_quotes: List[str] = Field(default_factory=list)
    business_implication: str = ""
    detection_method: str = ""      # pre_filter | llm_causal | cascade_hop
    shared_entities: List[str] = Field(default_factory=list)


class CascadeNarrative(BaseModel):
    """A multi-hop chain reaction narrative linking 3+ trends.

    Represents a chain like: Policy change → Industry disruption → Market shift
    with evidence and mechanisms at each hop.
    """
    cascade_id: str = ""            # Unique identifier
    node_ids: List[str] = Field(default_factory=list)  # Ordered UUIDs in chain
    node_titles: List[str] = Field(default_factory=list)
    narrative: str = ""             # Full chain-reaction story
    hop_mechanisms: List[str] = Field(default_factory=list)  # One per hop
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    weakest_link: str = ""
    compound_business_impact: str = ""
    total_hops: int = 0


class CausalCouncilResult(BaseModel):
    """Complete output of the multi-agent Causal Council (Stage D).

    Contains validated causal edges, cascade narratives, and summary metrics.
    """
    causal_edges: List[CausalEdgeResult] = Field(default_factory=list)
    cascade_narratives: List[CascadeNarrative] = Field(default_factory=list)
    # Metrics
    pairs_evaluated: int = 0        # How many pairs the LLM analyzed
    pairs_pre_filtered: int = 0     # How many passed statistical pre-filter
    edges_confirmed: int = 0        # How many the LLM confirmed as causal
    cascades_found: int = 0
    llm_calls_made: int = 0
    total_seconds: float = 0.0
