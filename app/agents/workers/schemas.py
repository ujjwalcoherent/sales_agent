"""
Pydantic schemas for AI Council input/output.

Every council decision carries structured reasoning so the pipeline
can explain WHY each value exists.
"""

from typing import List
from pydantic import BaseModel, Field


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


