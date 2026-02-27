"""
AI Council — Multi-agent validation and analysis at every pipeline step.

Four stages of council validation:
  Stage A: Trend Validation — validates clusters, hierarchy, event types
  Stage B: Impact Analysis — 4 specialist agents + moderator debate
  Stage C: Lead Quality — validates company-trend fit and pitch angle
  Stage D: Causal Council — multi-agent cross-trend causal reasoning

Each stage produces structured output with reasoning, so every decision
in the pipeline can be explained and defended.
"""

from .schemas import (
    CouncilPerspective,
    TrendValidation,
    ImpactCouncilResult,
    LeadValidation,
    CausalEdgeResult,
    CascadeNarrative,
    CausalCouncilResult,
)
from .trend_validator import validate_trends
from .impact_council import run_impact_council
from .lead_validator import validate_lead
from .causal_council import run_causal_council, apply_causal_results
from .article_triage import triage_articles, select_triage_candidates

__all__ = [
    "CouncilPerspective",
    "TrendValidation",
    "ImpactCouncilResult",
    "LeadValidation",
    # Stage D: Causal Council
    "CausalEdgeResult",
    "CascadeNarrative",
    "CausalCouncilResult",
    "run_causal_council",
    "apply_causal_results",
    # Pipeline stages
    "validate_trends",
    "run_impact_council",
    "validate_lead",
    "triage_articles",
    "select_triage_candidates",
]
