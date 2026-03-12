"""
ExperimentTracker — AutoResearch-inspired keep/discard loop.

Each pipeline run = one experiment. The tracker:
  1. Records run metrics (like AutoResearch's results.tsv)
  2. Detects regressions (keep vs discard decision)
  3. Monitors loop health (which learning loops are hurting quality?)
  4. Snapshots/restores learning state for regression rollback

Persistence: data/experiments.jsonl (append-only log of all runs)
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EXPERIMENTS_LOG = Path("data/experiments.jsonl")
_SNAPSHOT_DIR = Path("data/_learning_snapshot")


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentRecord(BaseModel):
    """One pipeline run = one experiment (AutoResearch pattern)."""
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # What was different this run (diff from baseline)
    params_snapshot: Dict[str, Any] = Field(default_factory=dict)
    hypothesis: str = ""

    # Quality metrics (the "val_bpb" equivalents)
    mean_oss: float = 0.0
    mean_coherence: float = 0.0
    noise_rate: float = 0.0
    actionable_rate: float = 0.0    # trends with OSS > 0.4
    article_count: int = 0
    cluster_count: int = 0

    # Decision
    status: Literal["keep", "discard", "crash"] = "keep"
    reason: str = ""

    # Which loops contributed
    learning_updates: Dict[str, str] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT / RESTORE (for rollback)
# ══════════════════════════════════════════════════════════════════════════════

_SNAPSHOT_FILES = [
    "data/adaptive_thresholds.json",
    "data/filter_hypothesis.json",   # hypothesis drift must roll back with other learning state
    "data/nli_baseline.json",        # stale baseline would trigger false retraining after rollback
]


def snapshot_learning_state() -> bool:
    """Save copies of learned params before loop updates (pre-experiment)."""
    try:
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for src_path in _SNAPSHOT_FILES:
            src = Path(src_path)
            if src.exists():
                shutil.copy2(src, _SNAPSHOT_DIR / src.name)
        logger.debug("Learning state snapshot saved")
        return True
    except Exception as e:
        logger.warning(f"Failed to snapshot learning state: {e}")
        return False


def restore_learning_state() -> bool:
    """Restore learned params from snapshot (discard experiment)."""
    try:
        restored = 0
        for src_path in _SNAPSHOT_FILES:
            snap = _SNAPSHOT_DIR / Path(src_path).name
            if snap.exists():
                shutil.copy2(snap, src_path)
                restored += 1
        if restored:
            logger.info(f"ROLLBACK: restored {restored} learning files from snapshot")
        return restored > 0
    except Exception as e:
        logger.warning(f"Failed to restore learning state: {e}")
        return False


def cleanup_snapshot() -> None:
    """Remove snapshot after successful keep decision."""
    try:
        if _SNAPSHOT_DIR.exists():
            shutil.rmtree(_SNAPSHOT_DIR)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentTracker:
    """Log, compare, and decide keep/discard per pipeline run.

    Implements the AutoResearch pattern:
      - Each run is an experiment
      - Compare against rolling best
      - Detect regressions → trigger rollback
      - Track which learning loops helped or hurt
    """

    def __init__(self, log_path: Path = EXPERIMENTS_LOG):
        self.log_path = log_path

    def record(self, record: ExperimentRecord) -> None:
        """Append experiment to log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        logger.info(
            f"Experiment {record.run_id}: status={record.status} "
            f"oss={record.mean_oss:.3f} actionable={record.actionable_rate:.1%}"
        )

    def recent_runs(self, n: int = 10) -> List[ExperimentRecord]:
        """Load last N experiments from log."""
        if not self.log_path.exists():
            return []
        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(ExperimentRecord.model_validate_json(line))
        except Exception as e:
            logger.warning(f"Failed to load experiments: {e}")
        return records[-n:]

    def rolling_baseline(self, window: int = 5) -> Optional[Dict[str, float]]:
        """Compute rolling average of last N 'keep' runs as baseline."""
        kept = [r for r in self.recent_runs(20) if r.status == "keep"]
        if len(kept) < 2:
            return None
        recent = kept[-window:]
        return {
            "mean_oss": sum(r.mean_oss for r in recent) / len(recent),
            "actionable_rate": sum(r.actionable_rate for r in recent) / len(recent),
            "mean_coherence": sum(r.mean_coherence for r in recent) / len(recent),
            "noise_rate": sum(r.noise_rate for r in recent) / len(recent),
        }

    def is_regression(self, current: ExperimentRecord) -> bool:
        """Detect regression: BOTH mean_oss AND actionable_rate must drop significantly.

        Uses rolling baseline (not single best run) to avoid noise-driven rollbacks.
        Requires ≥3 prior runs for comparison — cold starts always return False.
        """
        baseline = self.rolling_baseline()
        if baseline is None:
            return False  # Not enough data yet — keep everything

        oss_drop = (baseline["mean_oss"] - current.mean_oss) / max(baseline["mean_oss"], 0.01)
        act_drop = (baseline["actionable_rate"] - current.actionable_rate) / max(baseline["actionable_rate"], 0.01)

        # Both must drop >10% to trigger rollback (avoids false positives from noise)
        is_regressing = oss_drop > 0.10 and act_drop > 0.15

        if is_regressing:
            logger.warning(
                f"Regression detected: oss dropped {oss_drop:.1%} "
                f"(baseline={baseline['mean_oss']:.3f} → current={current.mean_oss:.3f}), "
                f"actionable dropped {act_drop:.1%}"
            )

        return is_regressing

    def trend(self, metric: str = "mean_oss", window: int = 5) -> str:
        """Return 'improving', 'stable', or 'degrading' over last N runs."""
        kept = [r for r in self.recent_runs(20) if r.status == "keep"]
        if len(kept) < 3:
            return "stable"

        recent = kept[-window:]
        values = [getattr(r, metric, 0.0) for r in recent]

        if len(values) < 2:
            return "stable"

        # Simple linear slope check
        first_half = sum(values[:len(values)//2]) / max(len(values)//2, 1)
        second_half = sum(values[len(values)//2:]) / max(len(values) - len(values)//2, 1)

        diff = second_half - first_half
        threshold = 0.02  # 2% change threshold

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "degrading"
        return "stable"

    def loop_health(self, loop_name: str, window: int = 10) -> float:
        """Score a loop's contribution over last N runs (0-1, <0.3 = failing)."""
        runs = self.recent_runs(window)
        if len(runs) < 3:
            return 0.5  # Not enough data

        active_oss = []
        inactive_oss = []
        for r in runs:
            status = r.learning_updates.get(loop_name, "skipped")
            if status == "updated":
                active_oss.append(r.mean_oss)
            else:
                inactive_oss.append(r.mean_oss)

        if not active_oss or not inactive_oss:
            return 0.5  # Can't compare

        active_mean = sum(active_oss) / len(active_oss)
        inactive_mean = sum(inactive_oss) / len(inactive_oss)

        # Score: 0.5 = neutral, >0.5 = loop helps, <0.5 = loop hurts
        delta = active_mean - inactive_mean
        return max(0.0, min(1.0, 0.5 + delta * 5))  # Scale delta to 0-1 range

    def should_dampen(self, loop_name: str) -> bool:
        """True if loop has hurt quality in >60% of last 10 runs."""
        return self.loop_health(loop_name) < 0.3


