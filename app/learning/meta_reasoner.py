"""
MetaReasoner — GUTTED. All methods return empty immediately.

Evidence for removal (8-agent audit, March 2026):
  - 7-8 LLM calls per run → output logged to reasoning_traces.jsonl + signal_bus.reasoning_*
  - Grep confirmed: NO loop reads bus.reasoning_* to adjust behavior
  - Zero behavioral impact on pipeline output
  - Cost: ~$0.05-0.10/run in LLM inference for pure decoration

Dataclasses kept for type compatibility (referenced by orchestrator, signal_bus).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTrace:
    """Structured chain-of-thought reasoning output (kept for type compat)."""
    step: str
    run_id: str = ""
    timestamp: str = ""
    observations: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    confidence: float = 0.5
    strategy_adjustments: Dict[str, Any] = field(default_factory=dict)
    projected_outcome: str = ""
    reasoning_time_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class Retrospective:
    """Full post-run self-critique (kept for type compat)."""
    run_id: str = ""
    timestamp: str = ""
    run_grade: str = ""
    summary: str = ""
    successes: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    surprises: List[str] = field(default_factory=list)
    improvement_plan: List[Dict[str, str]] = field(default_factory=list)
    vs_previous: str = ""
    regression_risks: List[str] = field(default_factory=list)


class MetaReasoner:
    """Gutted — all methods return empty. No LLM calls."""

    def __init__(self, llm_service=None, enabled: bool = True):
        self._llm = llm_service
        self._enabled = False  # Always disabled
        self._traces: List[ReasoningTrace] = []
        self._run_id = ""

    async def reason_about_sources(self, **kwargs) -> ReasoningTrace:
        return ReasoningTrace(step="source_intel", run_id=self._run_id)

    async def reason_about_trends(self, **kwargs) -> ReasoningTrace:
        return ReasoningTrace(step="analysis", run_id=self._run_id)

    async def reason_about_leads(self, **kwargs) -> ReasoningTrace:
        return ReasoningTrace(step="lead_gen", run_id=self._run_id)

    async def run_retrospective(self, **kwargs) -> Retrospective:
        return Retrospective(run_id=kwargs.get("run_id", ""))

    def get_active_hypotheses(self, target: Optional[str] = None) -> List[Dict]:
        return []

    def get_run_summary(self) -> Dict[str, Any]:
        return {}

    async def record_run_metrics(self, metrics: Dict[str, Any]) -> None:
        pass

    @property
    def traces(self) -> List[ReasoningTrace]:
        return self._traces


_INSTANCE: Optional["MetaReasoner"] = None


def get_meta_reasoner() -> "MetaReasoner":
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MetaReasoner(llm_service=None, enabled=False)
    return _INSTANCE
