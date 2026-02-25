"""
LearningSignal — captures per-step quality metrics for autonomous self-learning.

Passed through all pipeline agents and logged at end of run.
Used by the OSS auto-learning system — no human feedback required.
The OSS score (Objective Specificity Score) measures TEXT properties:
entity count, numbers, geography mentions — NOT the scoring algorithm itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HopSignal:
    """Quality signal for a single causal hop."""
    hop: int
    segment: str
    lead_type: str
    confidence: float
    companies_found: int          # How many real companies KB returned
    tool_calls: int               # How many tool calls the LLM agent made
    mechanism_specificity: float  # OSS-like score for mechanism text (0-1)


@dataclass
class LearningSignal:
    """
    Per-run learning signal captured from all pipeline agents.

    Feeds into:
    - Weight auto-learner: uses oss_score as reward signal
    - Source bandit (Thompson Sampling): maps article sources → quality via oss
    - Trend memory: tracks oss improvement per semantic centroid across runs

    Each field is deliberately measurable without LLM grading.
    """
    trend_title: str
    event_type: str

    # Synthesis quality (set from app/trends/specificity.py OSS computation)
    oss_score: float = 0.0
    synthesis_retries: int = 0

    # Causal chain quality
    hops_generated: int = 0
    hop_signals: list[HopSignal] = field(default_factory=list)
    causal_tool_calls: int = 0     # Total LLM tool calls across all hops
    kb_hit_rate: float = 0.0       # Fraction of hops that found real KB companies

    # Lead crystallization quality
    leads_generated: int = 0
    leads_with_companies: int = 0  # Leads with a real company from KB (not placeholder)
    avg_lead_confidence: float = 0.0

    # Source tracking (for source bandit feedback loop)
    source_article_ids: list[str] = field(default_factory=list)

    # Run metadata
    run_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_title": self.trend_title,
            "event_type": self.event_type,
            "oss_score": self.oss_score,
            "hops_generated": self.hops_generated,
            "causal_tool_calls": self.causal_tool_calls,
            "kb_hit_rate": self.kb_hit_rate,
            "leads_generated": self.leads_generated,
            "leads_with_companies": self.leads_with_companies,
            "avg_lead_confidence": self.avg_lead_confidence,
        }
