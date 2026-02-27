"""
Feedback-driven weight learning -- dual-path adaptation.

Priority order:
  1. Human feedback (50+ records) -- highest authority, 3x learning rate
  2. Outcome-based auto-learning (5+ runs) -- uses KB hit rate + lead quality
     as non-circular reward signals (NOT OSS, which is circular)
  3. Default weights -- cold start fallback

Safety: weights clamped to [0.02, 0.40], always normalized to sum 1.0,
learning rate decays with data, outcome variance check prevents learning
from undiscriminating scores.
"""

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PERSIST_PATH = Path("./data/learned_weights.json")


def _load_persisted_weights() -> Dict[str, Dict[str, float]]:
    """Load previously computed weights from disk (warm start for next run)."""
    if _PERSIST_PATH.exists():
        try:
            with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_persisted_weights(all_weights: Dict[str, Dict[str, float]]) -> None:
    """Persist all weight types to disk so next run starts from learned values."""
    try:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(all_weights, f, indent=2)
    except Exception as e:
        logger.warning(f"Weight learner: failed to persist weights: {e}")


# Warm-start: load previously persisted weights at import time
_persisted_weights: Dict[str, Dict[str, float]] = _load_persisted_weights()


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


# Per-run cache (avoids re-reading files repeatedly per pipeline run)
_feedback_cache: Dict[str, Any] = {"data": None, "ts": 0.0}
_oss_cache: Dict[str, Any] = {"data": None, "ts": 0.0}
_weight_cache: Dict[str, Any] = {}  # {weight_type: {"weights": {...}, "ts": float}}
_CACHE_TTL = 300.0  # 5 minutes
_cache_lock = threading.Lock()
_logged_types: set = set()


def compute_learned_weights(
    weight_type: str,
    default_weights: Dict[str, float],
    min_feedback: int = 50,
    learning_rate: float = 0.08,
    weight_floor: float = 0.02,
    weight_ceiling: float = 0.40,
) -> Dict[str, float]:
    """Compute adapted weights using the best available learning signal.

    Tries human feedback first, then OSS auto-learning, then defaults.
    """
    with _cache_lock:
        now = time.time()
        cached = _weight_cache.get(weight_type)
        if cached and (now - cached["ts"]) < _CACHE_TTL:
            return dict(cached["weights"])

    human_weights = _learn_from_human_feedback(
        weight_type, default_weights, min_feedback,
        learning_rate * 3.0,  # 3x LR for human feedback
        weight_floor, weight_ceiling,
    )
    if human_weights is not None:
        _cache_result(weight_type, human_weights)
        return human_weights

    auto_weights = _learn_from_outcomes(
        weight_type, default_weights,
        learning_rate, weight_floor, weight_ceiling,
    )
    if auto_weights is not None:
        _cache_result(weight_type, auto_weights)
        return auto_weights

    # Warm start: use previously persisted weights from last run where we had enough data
    if weight_type in _persisted_weights:
        logger.debug(f"Weight learner ({weight_type}): warm start from persisted weights")
        return dict(_persisted_weights[weight_type])

    _cache_result(weight_type, default_weights)
    return default_weights


def _learn_from_human_feedback(
    weight_type: str,
    default_weights: Dict[str, float],
    min_feedback: int,
    learning_rate: float,
    weight_floor: float,
    weight_ceiling: float,
) -> Optional[Dict[str, float]]:
    """Learn weights from human feedback (excludes auto-generated feedback).

    Returns adapted weights, or None if insufficient data.
    """
    feedback = _load_cached_feedback()
    if feedback is None:
        return None

    human_feedback = [
        f for f in feedback
        if not f.get("metadata", {}).get("auto", False)
    ]
    auto_count = len(feedback) - len(human_feedback)

    if len(human_feedback) < min_feedback:
        if weight_type not in _logged_types:
            _logged_types.add(weight_type)
            if auto_count > 0:
                logger.debug(
                    f"Weight learner ({weight_type}): {len(human_feedback)} human records "
                    f"(need {min_feedback}), {auto_count} auto records excluded. "
                    f"Trying OSS auto-learning instead."
                )
        return None

    good_trends = [f for f in human_feedback if f.get("rating") == "good_trend"]
    bad_trends = [f for f in human_feedback if f.get("rating") == "bad_trend"]

    if len(good_trends) < 5 or len(bad_trends) < 5:
        return None

    signal_key = _get_signal_key(weight_type)
    good_signals = _extract_signal_averages(good_trends, signal_key, default_weights)
    bad_signals = _extract_signal_averages(bad_trends, signal_key, default_weights)

    if not good_signals or not bad_signals:
        return None

    weights = _apply_preference_learning(
        default_weights, good_signals, bad_signals,
        len(human_feedback), learning_rate, weight_floor, weight_ceiling,
    )

    if weight_type not in _logged_types:
        _logged_types.add(weight_type)
        drift = sum(abs(weights[k] - default_weights[k]) for k in weights)
        logger.info(
            f"Weight learner ({weight_type}): HUMAN feedback path — "
            f"{len(good_trends)} good / {len(bad_trends)} bad records, "
            f"drift={drift:.4f}"
        )

    return weights


def _learn_from_outcomes(
    weight_type: str,
    default_weights: Dict[str, float],
    learning_rate: float,
    weight_floor: float,
    weight_ceiling: float,
    min_runs: int = 3,
    min_clusters: int = 10,
) -> Optional[Dict[str, float]]:
    """Learn weights from OUTCOME-BASED quality scores across pipeline runs.

    Uses a composite quality metric that is NON-CIRCULAR:
      - kb_hit_rate (40%): Did companies match in external KB? (external validation)
      - lead_quality (30%): Fraction of leads with real companies (external)
      - oss_score (30%): Text specificity (weakly circular but downweighted)

    This replaces the old _learn_from_oss which was fully circular
    (OSS → weights → trend scoring → synthesis → OSS).

    Splits clusters into high/low outcome groups and nudges weights toward
    factors that correlate with better real-world outcomes.
    """
    records = _load_cached_oss_data(min_runs)
    if not records:
        return None

    signal_key = _get_signal_key(weight_type)
    relevant = [
        r for r in records
        if r.get("breakdowns", {}).get(signal_key)
    ]

    if len(relevant) < min_clusters:
        if weight_type not in _logged_types:
            logger.debug(
                f"Weight learner ({weight_type}): outcome auto-learning — "
                f"only {len(relevant)} clusters with {signal_key} data "
                f"(need {min_clusters})"
            )
        return None

    # Compute composite outcome score per cluster
    # Uses outcome_score if available (non-circular), falls back to oss
    outcome_values = []
    for r in relevant:
        outcome = r.get("outcome_score")
        if outcome is not None:
            outcome_values.append(float(outcome))
        else:
            # Legacy records only have oss — use it but with discount
            outcome_values.append(float(r.get("oss", 0.0)) * 0.5)

    outcome_variance = _variance(outcome_values)
    if outcome_variance < 0.01:
        if weight_type not in _logged_types:
            logger.debug(
                f"Weight learner ({weight_type}): outcome variance={outcome_variance:.4f} "
                f"(< 0.01), no discrimination — skipping auto-learning"
            )
        return None

    median_outcome = _median(outcome_values)
    high_outcome = [r for r, v in zip(relevant, outcome_values) if v >= median_outcome]
    low_outcome = [r for r, v in zip(relevant, outcome_values) if v < median_outcome]

    if len(high_outcome) < 5 or len(low_outcome) < 5:
        return None

    high_mean = sum(v for v in outcome_values if v >= median_outcome) / max(len(high_outcome), 1)
    low_mean = sum(v for v in outcome_values if v < median_outcome) / max(len(low_outcome), 1)
    outcome_gap = high_mean - low_mean
    if outcome_gap < 0.08:
        if weight_type not in _logged_types:
            logger.debug(
                f"Weight learner ({weight_type}): outcome gap={outcome_gap:.3f} < 0.08 — "
                f"too small, skipping"
            )
        return None

    high_signals = _extract_oss_signal_averages(high_outcome, signal_key, default_weights)
    low_signals = _extract_oss_signal_averages(low_outcome, signal_key, default_weights)

    if not high_signals or not low_signals:
        return None

    n_runs = len(set(r.get("run_id", "") for r in relevant))
    weights = _apply_preference_learning(
        default_weights, high_signals, low_signals,
        n_runs * 10,
        learning_rate, weight_floor, weight_ceiling,
    )

    if weight_type not in _logged_types:
        _logged_types.add(weight_type)
        drift = sum(abs(weights[k] - default_weights[k]) for k in weights)
        adjustments = sum(
            1 for k in default_weights
            if abs(weights[k] - default_weights[k]) > 0.005
        )
        logger.info(
            f"Weight learner ({weight_type}): OUTCOME auto-learning — "
            f"{len(relevant)} clusters from {n_runs} runs, "
            f"median_outcome={median_outcome:.3f}, variance={outcome_variance:.4f}, "
            f"{adjustments} weights adjusted, drift={drift:.4f}"
        )
        if adjustments > 0:
            logger.debug(f"  Default: {default_weights}")
            logger.debug(f"  Learned: {weights}")

    return weights


def _apply_preference_learning(
    default_weights: Dict[str, float],
    good_signals: Dict[str, float],
    bad_signals: Dict[str, float],
    data_count: int,
    learning_rate: float,
    weight_floor: float,
    weight_ceiling: float,
) -> Dict[str, float]:
    """Nudge weights toward factors that distinguish good from bad outcomes.

    Adjusts by lr * delta for factors where |delta| > 0.10.
    Returns adapted weight dict (always sums to 1.0).
    """
    effective_lr = learning_rate * max(0.5, min(1.0, 50.0 / max(data_count, 1)))

    weights = dict(default_weights)

    for factor in weights:
        if factor not in good_signals or factor not in bad_signals:
            continue

        delta = good_signals[factor] - bad_signals[factor]

        if abs(delta) > 0.10:
            weights[factor] += effective_lr * delta

    for factor in weights:
        if not _is_finite(weights[factor]):
            logger.warning(f"NaN/Inf weight for {factor}, resetting to default")
            weights[factor] = default_weights.get(factor, 1.0 / len(weights))
        weights[factor] = max(weight_floor, min(weight_ceiling, weights[factor]))

    total = sum(weights.values())
    if total > 0 and _is_finite(total):
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    else:
        logger.warning("Weight sum is NaN/zero, falling back to defaults")
        return dict(default_weights)

    return weights


def _load_cached_feedback() -> Optional[list]:
    """Load feedback with caching."""
    global _feedback_cache
    now = time.time()
    if _feedback_cache["data"] is not None and (now - _feedback_cache["ts"]) < _CACHE_TTL:
        return _feedback_cache["data"]

    try:
        from app.tools.feedback import load_feedback
    except ImportError:
        return None

    data = load_feedback("trend")
    _feedback_cache = {"data": data, "ts": time.time()}
    return data


def _load_cached_oss_data(min_runs: int = 5) -> Optional[list]:
    """Load cluster signal history, enriched with outcome data from learning signals.

    Merges two data sources:
    - cluster_signal_log.jsonl: signal breakdowns per cluster (from trend engine)
    - learning_signals.jsonl: kb_hit_rate, lead quality per trend (from orchestrator)

    The merge produces outcome_score = 40% kb_hit + 30% lead_quality + 30% oss
    which is the non-circular reward signal for weight auto-learning.
    """
    global _oss_cache
    now = time.time()
    if _oss_cache["data"] is not None and (now - _oss_cache["ts"]) < _CACHE_TTL:
        return _oss_cache["data"]

    try:
        from app.learning.pipeline_metrics import load_cluster_signal_history
    except ImportError:
        return None

    data = load_cluster_signal_history(min_runs=min_runs)
    if not data:
        _oss_cache = {"data": None, "ts": time.time()}
        return None

    # Enrich with outcome data from learning_signals.jsonl
    outcome_map = _load_learning_signal_outcomes()
    if outcome_map:
        for record in data:
            run_id = record.get("run_id", "")
            if run_id in outcome_map and "outcome_score" not in record:
                outcomes = outcome_map[run_id]
                kb_hit = outcomes.get("avg_kb_hit_rate", 0.0)
                lead_q = outcomes.get("avg_lead_quality", 0.0)
                oss = record.get("oss", 0.0)
                record["outcome_score"] = 0.40 * kb_hit + 0.30 * lead_q + 0.30 * float(oss)
                record["kb_hit_rate"] = kb_hit
                record["lead_quality"] = lead_q

    _oss_cache = {"data": data, "ts": time.time()}
    return data


def _load_learning_signal_outcomes() -> Dict[str, Dict[str, float]]:
    """Load per-run outcome stats from learning_signals.jsonl.

    Returns {run_id: {avg_kb_hit_rate, avg_lead_quality}} for merging
    with cluster signal data.
    """
    signals_path = Path("./data/learning_signals.jsonl")
    if not signals_path.exists():
        return {}

    run_data: Dict[str, list] = {}
    try:
        with open(signals_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    rid = record.get("run_id", "")
                    if rid:
                        if rid not in run_data:
                            run_data[rid] = []
                        run_data[rid].append(record)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return {}

    result = {}
    for rid, records in run_data.items():
        kb_hits = [r.get("kb_hit_rate", 0.0) for r in records]
        lead_qualities = []
        for r in records:
            generated = r.get("leads_generated", 0)
            with_companies = r.get("leads_with_companies", 0)
            if generated > 0:
                lead_qualities.append(with_companies / generated)
            else:
                lead_qualities.append(0.0)

        result[rid] = {
            "avg_kb_hit_rate": sum(kb_hits) / max(len(kb_hits), 1),
            "avg_lead_quality": sum(lead_qualities) / max(len(lead_qualities), 1),
        }

    return result


def _cache_result(weight_type: str, weights: Dict[str, float]) -> None:
    with _cache_lock:
        _weight_cache[weight_type] = {"weights": dict(weights), "ts": time.time()}
        # Persist to disk so next pipeline run starts from learned values
        _persisted_weights[weight_type] = dict(weights)
        _save_persisted_weights(_persisted_weights)


def _get_signal_key(weight_type: str) -> str:
    return {
        "actionability": "actionability_breakdown",
        "trend_score": "trend_score_breakdown",
        "cluster_quality": "cluster_quality_breakdown",
        "confidence": "confidence_breakdown",
        "company_relevance": "company_relevance_breakdown",
    }.get(weight_type, "actionability_breakdown")


def _extract_signal_averages(
    feedback_records: List[Dict[str, Any]],
    signal_key: str,
    default_weights: Dict[str, float],
) -> Dict[str, float]:
    """Average raw signal values from human feedback records."""
    factor_sums: Dict[str, float] = {k: 0.0 for k in default_weights}
    factor_counts: Dict[str, int] = {k: 0 for k in default_weights}

    for record in feedback_records:
        signals = record.get("signals", {})
        breakdown = signals.get(signal_key, {})

        if breakdown:
            for factor in default_weights:
                info = breakdown.get(factor, {})
                raw = info.get("raw") if isinstance(info, dict) else None
                if raw is not None:
                    try:
                        val = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if not _is_finite(val):
                        continue
                    factor_sums[factor] += val
                    factor_counts[factor] += 1
        else:
            for factor in default_weights:
                val = signals.get(factor)
                if val is not None:
                    try:
                        fval = float(val)
                    except (ValueError, TypeError):
                        continue
                    if not _is_finite(fval):
                        continue
                    factor_sums[factor] += fval
                    factor_counts[factor] += 1

    result = {}
    for factor in default_weights:
        if factor_counts[factor] > 0:
            avg = factor_sums[factor] / factor_counts[factor]
            if _is_finite(avg):
                result[factor] = avg

    return result


def _extract_oss_signal_averages(
    oss_records: List[Dict[str, Any]],
    signal_key: str,
    default_weights: Dict[str, float],
) -> Dict[str, float]:
    """Average raw signal values from cluster signal log records."""
    factor_sums: Dict[str, float] = {k: 0.0 for k in default_weights}
    factor_counts: Dict[str, int] = {k: 0 for k in default_weights}

    for record in oss_records:
        breakdown = record.get("breakdowns", {}).get(signal_key, {})
        if not breakdown:
            continue

        for factor in default_weights:
            info = breakdown.get(factor, {})
            raw = info.get("raw") if isinstance(info, dict) else None
            if raw is not None:
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    continue
                if not _is_finite(val):
                    continue
                factor_sums[factor] += val
                factor_counts[factor] += 1

    result = {}
    for factor in default_weights:
        if factor_counts[factor] > 0:
            avg = factor_sums[factor] / factor_counts[factor]
            if _is_finite(avg):
                result[factor] = avg

    return result


def _variance(values: List[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return sorted_vals[n // 2]
