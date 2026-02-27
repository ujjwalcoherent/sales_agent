"""
Pipeline metric logging — appends a JSON record per run to a JSONL file.

Supports distribution-based calibration, EMA threshold adaptation,
and drift detection. Passive layer: never modifies pipeline behavior.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("./data/pipeline_run_log.jsonl")
CLUSTER_SIGNAL_LOG_PATH = Path("./data/cluster_signal_log.jsonl")


def record_run(
    metrics: Dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
) -> Dict[str, Any]:
    """Append a pipeline run record to the JSONL log and return it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": str(uuid.uuid4())[:8],
        **_flatten_metrics(metrics),
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(
            f"Pipeline metrics logged: {record.get('article_counts', {}).get('input', '?')} articles, "
            f"{record.get('n_clusters', '?')} clusters, "
            f"run_id={record['run_id']}"
        )
    except Exception as e:
        logger.warning(f"Failed to log pipeline metrics: {e}")

    return record


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


def compute_distributions(
    similarities: Optional[np.ndarray] = None,
    coherences: Optional[np.ndarray] = None,
    trend_scores: Optional[np.ndarray] = None,
    confidence_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute percentile distributions for threshold calibration logging."""
    result = {}

    percentile_levels = [10, 25, 50, 75, 90, 95]

    def _percentiles(arr, name):
        if arr is not None and len(arr) > 0:
            pcts = np.percentile(arr, percentile_levels)
            result[f"{name}_distribution"] = {
                f"p{level}": round(float(val), 4)
                for level, val in zip(percentile_levels, pcts)
            }

    _percentiles(similarities, "similarity")
    _percentiles(coherences, "coherence")
    _percentiles(trend_scores, "trend_score")
    _percentiles(confidence_scores, "confidence")

    return result


def detect_drift(
    current_metrics: Dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
    window: int = 10,
    sigma_threshold: float = 2.5,
) -> List[str]:
    """Detect metric drift via z-score against recent history. Returns alert strings."""
    history = load_history(log_path, last_n=window)
    if len(history) < 5:
        return []

    alerts = []
    monitor_keys = ["noise_count", "n_clusters"]
    flat_current = _flatten_metrics(current_metrics)

    for key in monitor_keys:
        if key not in flat_current:
            continue

        historical_values = []
        for h in history:
            flat_h = _flatten_metrics(h) if "article_counts" in h else h
            if key in flat_h:
                try:
                    historical_values.append(float(flat_h[key]))
                except (ValueError, TypeError):
                    continue

        if len(historical_values) < 3:
            continue

        mean = np.mean(historical_values)
        std = np.std(historical_values)
        if std < 0.001:
            continue

        current_val = float(flat_current[key])
        z_score = abs(current_val - mean) / std

        if z_score > sigma_threshold:
            alerts.append(
                f"DRIFT: {key}={current_val:.3f} "
                f"(expected {mean:.3f} +/- {sigma_threshold * std:.3f}, "
                f"z={z_score:.1f})"
            )

    if alerts:
        logger.warning(f"Pipeline drift detected: {alerts}")

    return alerts


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

    Returns config-key -> adapted-value dict. Empty if insufficient history.
    """
    history = load_history(log_path, last_n=20)

    if len(history) < min_runs:
        logger.debug(
            f"Adaptive thresholds: only {len(history)} runs "
            f"(need {min_runs}), using config defaults"
        )
        return {}

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
        logger.info(f"Adaptive thresholds (dual-rate EMA, {len(history)} runs): {results}")

    return results


def detect_drift_ewma(
    current_metrics: Dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
    lambda_param: float = 0.2,
    L: float = 3.0,
    window: int = 20,
) -> List[str]:
    """EWMA control chart drift detection for slow persistent degradation.

    REF: Montgomery 2009, Ch. 9 — EWMA control charts.
    """
    history = load_history(log_path, last_n=window)
    if len(history) < 5:
        return []

    alerts = []
    flat_current = _flatten_metrics(current_metrics)

    monitor_defs = {
        "noise_count": None,
        "n_clusters": None,
        "coherence_distribution.p50": None,
        "general_ratio": None,
    }

    for metric_key in monitor_defs:
        historical_values = []
        for h in history:
            val = _extract_nested(h, metric_key)
            if val is not None:
                try:
                    historical_values.append(float(val))
                except (ValueError, TypeError):
                    continue

        if len(historical_values) < 5:
            continue

        current_val = _extract_nested(flat_current, metric_key)
        if current_val is None:
            current_val = _extract_nested(current_metrics, metric_key)
        if current_val is None:
            continue
        current_val = float(current_val)

        mu = np.mean(historical_values)
        sigma = np.std(historical_values)
        if sigma < 0.001:
            continue

        all_values = historical_values + [current_val]
        z = mu  # Initialize EWMA at process mean
        for i, x in enumerate(all_values, 1):
            z = lambda_param * x + (1 - lambda_param) * z

            factor = np.sqrt(
                lambda_param / (2 - lambda_param)
                * (1 - (1 - lambda_param) ** (2 * i))
            )
            ucl = mu + L * sigma * factor
            lcl = mu - L * sigma * factor

        if z > ucl or z < lcl:
            direction = "above UCL" if z > ucl else "below LCL"
            alerts.append(
                f"EWMA_DRIFT: {metric_key} EWMA={z:.3f} {direction} "
                f"(mu={mu:.3f}, UCL={ucl:.3f}, LCL={lcl:.3f})"
            )

    if alerts:
        logger.warning(f"EWMA drift detected: {alerts}")

    return alerts


def compute_source_quality(
    articles: list,
    labels: "np.ndarray",
    cluster_quality_scores: Dict[int, float],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-source quality stats (noise rate, avg cluster quality)."""
    from collections import defaultdict

    source_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"articles": 0, "clustered": 0, "noise": 0, "quality_sum": 0.0}
    )

    for article, label in zip(articles, labels):
        src = getattr(article, 'source_id', '') or ''
        if not src:
            continue
        label = int(label)
        source_stats[src]["articles"] += 1
        if label >= 0:
            source_stats[src]["clustered"] += 1
            source_stats[src]["quality_sum"] += cluster_quality_scores.get(label, 0.5)
        else:
            source_stats[src]["noise"] += 1

    result = {}
    for src, stats in source_stats.items():
        n = stats["articles"]
        clustered = stats["clustered"]
        result[src] = {
            "articles": n,
            "clustered": clustered,
            "noise": stats["noise"],
            "noise_rate": round(stats["noise"] / max(n, 1), 3),
            "avg_quality": round(
                stats["quality_sum"] / max(clustered, 1), 3
            ),
        }

    return result


def compute_cluster_stability(
    current_articles: list,
    current_labels: "np.ndarray",
    log_path: Path = DEFAULT_LOG_PATH,
) -> float:
    """Cluster Stability Index: co-clustering consistency across consecutive runs.

    Returns float in [0, 1] (1.0 = stable), or -1.0 if insufficient data.
    REF: von Luxburg 2010, "Clustering Stability: An Overview"
    """
    import hashlib

    history = load_history(log_path, last_n=1)
    if not history:
        return -1.0

    prev_assignments = history[-1].get("article_cluster_assignments", {})
    if not prev_assignments:
        return -1.0

    current_map = {}
    for article, label in zip(current_articles, current_labels):
        title = getattr(article, 'title', '') or ''
        if title:
            key = hashlib.md5(title.encode()).hexdigest()[:12]
            current_map[key] = int(label)

    overlap_keys = set(current_map.keys()) & set(prev_assignments.keys())
    if len(overlap_keys) < 5:
        return -1.0

    # Skip CSI if <50% overlap (meaningless comparison)
    max_set_size = max(len(current_map), len(prev_assignments))
    overlap_fraction = len(overlap_keys) / max_set_size if max_set_size > 0 else 0
    if overlap_fraction < 0.5:
        logger.info(
            f"Cluster stability: skipping CSI (overlap={overlap_fraction:.1%}, "
            f"{len(overlap_keys)}/{max_set_size} articles shared)"
        )
        return -1.0

    overlap_list = list(overlap_keys)
    pairs_checked = 0
    pairs_stable = 0

    # Sample pairs to keep O(N) not O(N^2)
    rng = np.random.RandomState(42)
    for i, key_a in enumerate(overlap_list):
        # Compare with up to 5 random other overlapping articles
        others = [k for k in overlap_list if k != key_a]
        if not others:
            continue
        sample_size = min(5, len(others))
        for key_b in rng.choice(others, size=sample_size, replace=False):
            prev_same = (prev_assignments[key_a] == prev_assignments[key_b])
            curr_same = (current_map[key_a] == current_map[key_b])
            pairs_checked += 1
            if prev_same == curr_same:
                pairs_stable += 1

    if pairs_checked == 0:
        return -1.0

    stability = pairs_stable / pairs_checked
    logger.info(
        f"Cluster stability index: {stability:.3f} "
        f"({pairs_stable}/{pairs_checked} pairs stable, "
        f"{len(overlap_keys)} overlapping articles)"
    )
    return round(stability, 4)


def save_cluster_assignments(
    articles: list,
    labels: "np.ndarray",
    metrics: Dict[str, Any],
) -> None:
    """Save article->cluster assignments into metrics for next-run stability check."""
    import hashlib

    assignments = {}
    for article, label in zip(articles, labels):
        title = getattr(article, 'title', '') or ''
        if title:
            key = hashlib.md5(title.encode()).hexdigest()[:12]
            assignments[key] = int(label)

    metrics["article_cluster_assignments"] = assignments


def compute_run_quality(
    syntheses: Dict[int, Dict[str, Any]],
    cluster_cmi_scores: Optional[Dict[int, float]] = None,
    original_cluster_count: int = 0,
    merged_cluster_count: int = 0,
    source_count: int = 0,
    contributing_sources: int = 0,
    causal_edges: int = 0,
    previous_mean_oss: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute per-run quality fingerprint from synthesis OSS scores."""
    oss_scores = []
    for cid, synth in syntheses.items():
        if synth:
            oss = synth.get("_oss", 0.0)
            oss_scores.append(float(oss) if oss is not None else 0.0)

    total_trends = len(oss_scores)

    if total_trends == 0:
        return {
            "mean_oss": 0.0,
            "oss_above_04": 0,
            "oss_below_02": 0,
            "actionable_rate": 0.0,
            "total_trends": 0,
        }

    oss_arr = np.array(oss_scores)
    mean_oss = float(oss_arr.mean())

    result = {
        "mean_oss": round(mean_oss, 4),
        "median_oss": round(float(np.median(oss_arr)), 4),
        "oss_above_04": int((oss_arr >= 0.4).sum()),
        "oss_below_02": int((oss_arr < 0.2).sum()),
        "actionable_rate": round(float((oss_arr >= 0.4).sum()) / total_trends, 4),
        "total_trends": total_trends,
    }

    if cluster_cmi_scores:
        irrelevant = sum(1 for s in cluster_cmi_scores.values() if s < 0.30)
        result["irrelevant_rate"] = round(irrelevant / max(len(cluster_cmi_scores), 1), 4)

    if original_cluster_count > 0:
        merged = original_cluster_count - merged_cluster_count
        result["duplicate_rate"] = round(merged / max(original_cluster_count, 1), 4)

    if source_count > 0:
        result["source_utilization"] = round(contributing_sources / source_count, 4)

    result["cross_trend_edges"] = causal_edges

    if previous_mean_oss is not None:
        result["oss_improvement"] = round(mean_oss - previous_mean_oss, 4)

    logger.info(
        f"Run quality: mean_oss={result['mean_oss']:.3f}, "
        f"actionable={result['oss_above_04']}/{total_trends} "
        f"({result['actionable_rate']:.0%}), "
        f"garbage={result['oss_below_02']}/{total_trends}"
    )

    return result


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


def load_cluster_signal_history(
    log_path: Path = CLUSTER_SIGNAL_LOG_PATH,
    min_runs: int = 5,
) -> List[Dict[str, Any]]:
    """Load cluster signal history for auto-learning. Empty if < min_runs."""
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
        logger.warning(f"Failed to load cluster signal history: {e}")
        return []

    run_ids = set(r.get("run_id", "") for r in records)
    if len(run_ids) < min_runs:
        return []

    return records


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


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metrics dict for consistent logging."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict) and k in ("phase_times", "article_counts"):
            for sub_k, sub_v in v.items():
                flat[f"{k}.{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat
