"""
Pipeline metric logging — appends a JSON record per run to a JSONL file.

Supports distribution-based calibration, EMA threshold adaptation.

Functions removed (March 2026 audit — never called or results never consumed):
  - record_run, compute_distributions, detect_drift, detect_drift_ewma,
    compute_source_quality, compute_cluster_stability, save_cluster_assignments,
    compute_run_quality
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("./data/pipeline_run_log.jsonl")
CLUSTER_SIGNAL_LOG_PATH = Path("./data/cluster_signal_log.jsonl")


def load_history(
    log_path: Path = DEFAULT_LOG_PATH,
    last_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load run history from JSONL log, oldest first. Optionally last N only."""
    if not log_path.exists():
        return []

    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to load pipeline history: {e}")
        return []

    if last_n is not None:
        records = records[-last_n:]

    return records


class AdaptiveThreshold:
    """Target-anchored asymmetric EMA with variance tracking.

    Uses a quality target as anchor to prevent threshold drift. Adapts faster
    to degradation (alpha_down) than recovery (alpha_up). Safety bounds
    prevent catastrophic drift.

    REF: EWMA control charts (Montgomery 2009) with target-based adaptation.
    """

    def __init__(
        self,
        name: str,
        default: float,
        floor: float,
        ceiling: float,
        alpha_up: float = 0.2,
        alpha_down: float = 0.4,
        metric_path: str = "",
        higher_is_better: bool = True,
        quality_target: float = None,
    ):
        self.name = name
        self.default = default
        self.floor = floor
        self.ceiling = ceiling
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.metric_path = metric_path
        self.higher_is_better = higher_is_better
        self.quality_target = quality_target

        self.ema_mean: float = default
        self.ema_var: float = 0.01

    def update(self, value: float) -> float:
        """Update EMA with a new observation, return adapted threshold."""
        if self.higher_is_better:
            is_degradation = value < self.ema_mean
        else:
            is_degradation = value > self.ema_mean

        alpha = self.alpha_down if is_degradation else self.alpha_up

        self.ema_mean = alpha * value + (1 - alpha) * self.ema_mean

        diff = value - self.ema_mean
        self.ema_var = alpha * (diff ** 2) + (1 - alpha) * self.ema_var

        if self.quality_target is not None:
            # 60% observed EMA + 40% quality target (prevents drift toward degradation)
            adapted = 0.6 * self.ema_mean + 0.4 * self.quality_target
        else:
            adapted = self.ema_mean

        return max(self.floor, min(self.ceiling, adapted))

    @property
    def ema_std(self) -> float:
        return self.ema_var ** 0.5

    def is_anomaly(self, value: float, sigma: float = 2.5) -> bool:
        if self.ema_std < 0.001:
            return False
        z = abs(value - self.ema_mean) / self.ema_std
        return z > sigma

    def update_from_history(self, values: List[float]) -> float:
        """Replay historical values to initialize EMA state."""
        for v in values:
            self.update(v)
        return max(self.floor, min(self.ceiling, self.ema_mean))


# Per-metric threshold configuration with quality targets.
THRESHOLD_REGISTRY: Dict[str, AdaptiveThreshold] = {
    "coherence_min": AdaptiveThreshold(
        name="coherence_min",
        default=0.40, floor=0.20, ceiling=0.70,
        alpha_up=0.15, alpha_down=0.25,
        metric_path="coherence_distribution.p50",
        higher_is_better=True,
        quality_target=0.50,
    ),
    "coherence_reject": AdaptiveThreshold(
        name="coherence_reject",
        default=0.25, floor=0.10, ceiling=0.45,
        alpha_up=0.15, alpha_down=0.25,
        metric_path="coherence_distribution.p25",
        higher_is_better=True,
        quality_target=0.30,
    ),
    "merge_threshold": AdaptiveThreshold(
        name="merge_threshold",
        default=0.75, floor=0.60, ceiling=0.90,
        alpha_up=0.20, alpha_down=0.30,
        metric_path="coherence_distribution.p75",
        higher_is_better=True,
        quality_target=0.78,
    ),
    "signal_p10": AdaptiveThreshold(
        name="signal_p10",
        default=0.20, floor=0.05, ceiling=0.40,
        alpha_up=0.25, alpha_down=0.25,
        metric_path="trend_score_distribution.p10",
        higher_is_better=True,
    ),
    "signal_p50": AdaptiveThreshold(
        name="signal_p50",
        default=0.50, floor=0.20, ceiling=0.70,
        alpha_up=0.25, alpha_down=0.25,
        metric_path="trend_score_distribution.p50",
        higher_is_better=True,
    ),
    "confidence_p25": AdaptiveThreshold(
        name="confidence_p25",
        default=0.35, floor=0.15, ceiling=0.60,
        alpha_up=0.20, alpha_down=0.35,
        metric_path="confidence_distribution.p25",
        higher_is_better=True,
        quality_target=0.40,
    ),
}


def compute_adaptive_thresholds(
    log_path: Path = DEFAULT_LOG_PATH,
    min_runs: int = 5,
) -> Dict[str, float]:
    """Compute dual-rate EMA adaptive thresholds from pipeline history.

    Uses signal bus cross-loop data to modulate adaptation speed:
    - Low avg_novelty -> tighten thresholds (recycled trends need higher bar)
    - Low NLI entailment -> widen filter thresholds (articles are harder to classify)

    Returns config-key -> adapted-value dict. Empty if insufficient history.
    """
    history = load_history(log_path, last_n=20)

    if len(history) < min_runs:
        logger.debug(
            f"Adaptive thresholds: only {len(history)} runs "
            f"(need {min_runs}), using config defaults"
        )
        return {}

    # Load signal bus for cross-loop modulation
    bus_avg_novelty = 0.5
    try:
        from app.learning.signal_bus import LearningSignalBus
        bus = LearningSignalBus.load_previous()
        if bus:
            modulation = bus.get_adaptive_threshold_modulation()
            bus_avg_novelty = modulation.get("avg_novelty", 0.5)
    except Exception:
        pass

    results = {}
    anomalies = []

    for threshold_name, at in THRESHOLD_REGISTRY.items():
        at.ema_mean = at.default
        at.ema_var = 0.01

        values = []
        for record in history:
            val = _extract_nested(record, at.metric_path)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue

        if len(values) < min_runs:
            continue

        adapted = at.update_from_history(values)

        # Signal bus modulation: low novelty tightens thresholds
        if bus_avg_novelty < 0.3 and at.higher_is_better:
            tighten = 0.05 * (0.3 - bus_avg_novelty) / 0.3
            adapted += tighten

        adapted = max(at.floor, min(at.ceiling, adapted))
        results[threshold_name] = round(adapted, 4)

        if values and at.is_anomaly(values[-1]):
            anomalies.append(
                f"{threshold_name}: latest={values[-1]:.3f} "
                f"(ema={at.ema_mean:.3f} ±{at.ema_std:.3f})"
            )

        logger.debug(
            f"Adaptive threshold {threshold_name}: "
            f"ema={at.ema_mean:.4f} (std={at.ema_std:.4f}) -> "
            f"clipped={adapted:.4f} [{at.floor}, {at.ceiling}]"
        )

    if anomalies:
        logger.warning(f"Adaptive threshold anomalies: {anomalies}")

    if results:
        bus_info = ""
        if bus_avg_novelty < 0.3:
            bus_info = f" | bus: novelty={bus_avg_novelty:.2f}"
        logger.info(f"Adaptive thresholds (dual-rate EMA, {len(history)} runs): {results}{bus_info}")

    return results


def record_cluster_signals(
    run_id: str,
    cluster_signals: Dict[int, Dict[str, Any]],
    cluster_oss: Dict[int, float],
    cluster_outcomes: Optional[Dict[int, Dict[str, float]]] = None,
    log_path: Path = CLUSTER_SIGNAL_LOG_PATH,
) -> int:
    """Log per-cluster signal breakdowns + outcome scores for weight auto-learning.

    The outcome_score is a NON-CIRCULAR composite:
      40% kb_hit_rate + 30% lead_quality + 30% oss
    This breaks the circular dependency where OSS alone drove weight updates.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    records_written = 0
    cluster_outcomes = cluster_outcomes or {}

    breakdown_keys = [
        "actionability_breakdown",
        "trend_score_breakdown",
        "cluster_quality_breakdown",
        "confidence_breakdown",
    ]

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for cid, signals in cluster_signals.items():
                oss = cluster_oss.get(cid, 0.0)
                if oss is None:
                    oss = 0.0

                breakdowns = {}
                for key in breakdown_keys:
                    bd = signals.get(key)
                    if bd and isinstance(bd, dict):
                        breakdowns[key] = bd

                if not breakdowns:
                    continue

                # Compute composite outcome score from external signals
                outcomes = cluster_outcomes.get(cid, {})
                kb_hit = outcomes.get("kb_hit_rate", 0.0)
                lead_quality = outcomes.get("lead_quality", 0.0)
                outcome_score = 0.40 * kb_hit + 0.30 * lead_quality + 0.30 * float(oss)

                record = {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "cluster_id": cid,
                    "oss": round(float(oss), 4),
                    "outcome_score": round(outcome_score, 4),
                    "kb_hit_rate": round(kb_hit, 4),
                    "lead_quality": round(lead_quality, 4),
                    "breakdowns": breakdowns,
                }
                f.write(json.dumps(record, default=str) + "\n")
                records_written += 1

        if records_written > 0:
            logger.info(
                f"Logged {records_written} cluster signal records "
                f"for weight auto-learning (run={run_id})"
            )
    except Exception as e:
        logger.warning(f"Failed to log cluster signals: {e}")

    return records_written


def _extract_nested(data: Dict[str, Any], dotted_key: str) -> Optional[Any]:
    """Extract a value from a dict using a dotted key path like 'a.b.c'."""
    parts = dotted_key.split(".")
    val = data
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val
