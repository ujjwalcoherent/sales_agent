"""Pipeline run manager -- tracks active and completed runs in-memory.

Provides an asyncio.Queue per run for SSE streaming to the Next.js frontend.
Runs are identified by timestamp-based IDs (e.g., "20260226_143022").
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# LangGraph step → progress percentage mapping
STEP_PROGRESS: Dict[str, int] = {
    "init": 0,
    "source_intel_complete": 15,
    "analysis_complete": 30,
    "impact_complete": 45,
    "quality_complete": 55,
    "quality_retry_analysis": 50,
    "causal_council_complete": 70,
    "lead_crystallize_complete": 80,
    "lead_gen_complete": 90,
    "learning_update_complete": 100,
}


@dataclass
class PipelineRun:
    """State for a single pipeline execution."""
    run_id: str
    status: str = "started"  # started | running | completed | failed
    current_step: str = "init"
    progress_pct: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    final_state: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    trends_count: int = 0
    companies_count: int = 0
    leads_count: int = 0


class RunManager:
    """Singleton that tracks pipeline runs across API requests."""

    def __init__(self):
        self._runs: Dict[str, PipelineRun] = {}

    def create_run(self, run_id: str) -> PipelineRun:
        run = PipelineRun(run_id=run_id)
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        return self._runs.get(run_id)

    def get_latest_run(self) -> Optional[PipelineRun]:
        if not self._runs:
            return None
        return max(self._runs.values(), key=lambda r: r.started_at)

    def list_runs(self, limit: int = 20) -> List[PipelineRun]:
        runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    @property
    def is_running(self) -> bool:
        return any(r.status in ("started", "running") for r in self._runs.values())


# Module-level singleton
run_manager = RunManager()
