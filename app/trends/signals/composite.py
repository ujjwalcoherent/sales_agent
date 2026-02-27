"""
Composite signal computation — final scores that rank trends.

Three output scores:

1. actionability_score (0-1): How valuable is this trend for sales outreach?
   This is THE most important score. It determines which trends get emailed
   to prospects and which get filed away.

2. trend_score (0-1): How significant is this trend overall?
   Based on BERTrend + Reddit Hot algorithm research. Not sales-specific —
   measures raw importance regardless of commercial value.

3. cluster_quality_score (0-1): How well-formed is this cluster?
   Measures cluster coherence, source diversity, authority.

4. confidence_score (0-1): How confident are we this is a REAL, NEW trend?
   Combines cluster quality + temporal novelty + evidence specificity.

T3 ENHANCEMENT: All scores now include factor breakdowns stored in signals dict.
   Each breakdown shows {factor: {weight, raw, contribution}} so the UI can
   explain WHY a score is what it is. Every serious analytics product
   (Meltwater, Brandwatch, Salesforce Einstein) provides score explanations.

T2 ENHANCEMENT: classify_signal_strength accepts p10/p50 parameters
   (previously hardcoded). Engine computes actual percentiles in two-pass.

I2 ENHANCEMENT: compute_momentum_prediction uses velocity_history slope.

REF: Boutaleb et al. 2024 (BERTrend) — P10/P50 percentile classification.
     Reddit Hot ranking — log(score) + recency decay.
     6sense trigger event taxonomy (2024) — event urgency classification.
     Bombora intent signal categories — buying intent scoring.
     Meltwater Predictive Analytics — forecast spike growth/fade.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from app.schemas.trends import SignalStrength

logger = logging.getLogger(__name__)


def _load_weights() -> Dict[str, float]:
    """Load actionability weights from config, then overlay learned weights.

    Priority: learned weights (from feedback) > config > hardcoded defaults.
    Falls back to config/defaults if insufficient feedback.
    """
    try:
        from app.config import get_settings
        raw = get_settings().actionability_weights
        defaults = json.loads(raw)
    except Exception:
        defaults = {
            "recency": 0.12, "velocity": 0.07, "specificity": 0.12,
            "regulatory": 0.12, "trigger": 0.14, "diversity": 0.07,
            "authority": 0.13, "financial": 0.05, "person": 0.03,
            "event_focus": 0.05, "cmi_relevance": 0.10,
        }

    # Overlay learned weights from feedback (if sufficient data)
    try:
        from app.learning.weight_learner import compute_learned_weights
        return compute_learned_weights("actionability", defaults)
    except Exception:
        return defaults


# TTL-cached weights — refresh every 60s so mid-run weight learner updates take effect
_WEIGHTS_CACHE: Dict[str, Any] = {"weights": None, "ts": 0.0}
_WEIGHTS_TTL = 60.0


def _get_weights() -> Dict[str, float]:
    """Get actionability weights with TTL cache (refreshes every 60s)."""
    import time
    now = time.time()
    if _WEIGHTS_CACHE["weights"] is None or (now - _WEIGHTS_CACHE["ts"]) > _WEIGHTS_TTL:
        _WEIGHTS_CACHE["weights"] = _load_weights()
        _WEIGHTS_CACHE["ts"] = now
    return _WEIGHTS_CACHE["weights"]


def invalidate_weights_cache() -> None:
    """Force-invalidate the weights cache (called after weight learner updates)."""
    _WEIGHTS_CACHE["weights"] = None
    _WEIGHTS_CACHE["ts"] = 0.0


def compute_actionability_score(
    signals: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    THE most important score. Ranks trends by sales outreach value.

    T3: Now stores factor breakdown in signals["actionability_breakdown"]
    and top drivers in signals["actionability_top_drivers"].

    Args:
        signals: Flat dict of all computed signals. Also MUTATED to include breakdown.
        weights: Optional custom weights. Defaults to learned weights (TTL-cached).

    Returns:
        Float in [0, 1]. Higher = more actionable for sales outreach.
    """
    w = weights or _get_weights()

    # Extract and normalize each component
    factors = {
        "recency": signals.get("recency_score", 0.0),
        "velocity": min(1.0, signals.get("velocity", 0.0) / 10.0),
        "specificity": _compute_specificity(signals),
        "regulatory": float(signals.get("regulatory_flag", False)),
        "trigger": signals.get("trigger_urgency", 0.3),
        "diversity": signals.get("source_diversity", 0.0),
        "authority": signals.get("authority_weighted", 0.5),
        "financial": float(signals.get("financial_indicator", False)),
        "person": float(signals.get("key_person_flag", False)),
        "event_focus": signals.get("trigger_event_concentration", 0.0),
        "cmi_relevance": signals.get("cmi_relevance", 0.0),
    }

    # Compute weighted sum + breakdown
    score = 0.0
    breakdown = {}
    for factor_name, raw_value in factors.items():
        factor_weight = w.get(factor_name, 0.0)
        contribution = factor_weight * raw_value
        score += contribution
        breakdown[factor_name] = {
            "weight": round(factor_weight, 3),
            "raw": round(raw_value, 3),
            "contribution": round(contribution, 4),
        }

    # Store breakdown in signals dict (T3 explainability)
    signals["actionability_breakdown"] = breakdown

    # Top drivers sorted by contribution (highest first)
    top_drivers = sorted(
        breakdown.items(), key=lambda x: x[1]["contribution"], reverse=True
    )
    signals["actionability_top_drivers"] = [
        f"{name}: {info['contribution']:.3f}" for name, info in top_drivers[:5]
    ]

    return min(1.0, max(0.0, score))


def compute_trend_score(signals: Dict[str, Any]) -> float:
    """
    Overall trend importance (not sales-specific).

    Combines volume (log-scaled), momentum, diversity, and novelty.
    Based on BERTrend research + Reddit Hot algorithm principles.

    T3: Now stores breakdown in signals["trend_score_breakdown"].

    REF: Boutaleb et al. 2024 (BERTrend) — popularity-based trend scoring.
         Reddit "Hot" ranking — log(score) + sign(score) x seconds / 45000.
    """
    article_count = signals.get("article_count", 0)
    acceleration = signals.get("acceleration", 0.0)
    diversity = signals.get("source_diversity", 0.0)

    volume = min(1.0, math.log(1 + article_count) / 5.0)
    accel_norm = (min(1.0, max(-1.0, acceleration)) + 1.0) / 2.0

    # Load weights: learned (from feedback) > config > defaults
    from app.config import get_settings
    import json as _json
    _w = _json.loads(get_settings().trend_score_weights)
    try:
        from app.learning.weight_learner import compute_learned_weights
        _w = compute_learned_weights("trend_score", _w)
    except Exception:
        pass

    factors = {
        "volume": {"weight": _w.get("volume", 0.30), "raw": round(volume, 3)},
        "momentum": {"weight": _w.get("momentum", 0.45), "raw": round(accel_norm, 3)},
        "diversity": {"weight": _w.get("diversity", 0.25), "raw": round(diversity, 3)},
    }

    score = 0.0
    for info in factors.values():
        contribution = info["weight"] * info["raw"]
        info["contribution"] = round(contribution, 4)
        score += contribution

    signals["trend_score_breakdown"] = factors
    return min(1.0, max(0.0, score))


def classify_signal_strength(
    popularity_score: float,
    acceleration: float,
    p10: float = 0.2,
    p50: float = 0.5,
) -> SignalStrength:
    """
    Classify a trend's signal strength using BERTrend's P10/P50 method.

    T2: Now accepts p10/p50 as parameters instead of hardcoded constants.
    Engine computes actual percentiles from the distribution and passes them in.
    Falls back to 0.2/0.5 if engine doesn't provide them (backward compat).

    Args:
        popularity_score: The trend_score for this trend.
        acceleration: Growth rate of the trend (from temporal signals).
        p10: 10th percentile threshold (computed from actual distribution).
        p50: 50th percentile threshold (computed from actual distribution).

    Returns:
        SignalStrength.NOISE:  Too small to matter. Below P10.
        SignalStrength.WEAK:   Between P10 and P50 WITH positive growth = emerging.
        SignalStrength.STRONG: Above P50 = confirmed trend.

    REF: Boutaleb et al. 2024, arXiv:2411.05930 — "We classify each topic
         into noise, weak signal, or strong signal based on popularity percentiles
         and growth rate."
    """
    if popularity_score < p10:
        return SignalStrength.NOISE

    if popularity_score <= p50:
        if acceleration > 0:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NOISE

    return SignalStrength.STRONG


def compute_cluster_quality_score(signals: Dict[str, Any]) -> float:
    """How well-formed is this cluster? (Formerly compute_confidence_score.)

    Measures cluster coherence, source diversity, event agreement, and authority.
    This is a CLUSTER QUALITY metric, NOT trend confidence.

    Stores breakdown in signals["cluster_quality_breakdown"].
    """
    from app.config import get_settings
    import json as _json
    _cw = _json.loads(get_settings().cluster_quality_score_weights)
    try:
        from app.learning.weight_learner import compute_learned_weights
        _cw = compute_learned_weights("cluster_quality", _cw)
    except Exception:
        pass

    factors = {
        "coherence": {
            "weight": _cw.get("coherence", 0.28),
            "raw": round(signals.get("intra_cluster_cosine", 0.5), 3),
        },
        "source_diversity": {
            "weight": _cw.get("source_diversity", 0.25),
            "raw": round(signals.get("source_diversity", 0.0), 3),
        },
        "event_agreement": {
            "weight": _cw.get("event_agreement", 0.17),
            "raw": round(signals.get("trigger_event_concentration", 0.0), 3),
        },
        "evidence_volume": {
            "weight": _cw.get("evidence_volume", 0.12),
            "raw": round(min(1.0, math.log(1 + signals.get("article_count", 0)) / 3.0), 3),
        },
        "authority": {
            "weight": _cw.get("authority", 0.18),
            "raw": round(signals.get("authority_weighted", 0.5), 3),
        },
    }

    score = 0.0
    for info in factors.values():
        contribution = info["weight"] * info["raw"]
        info["contribution"] = round(contribution, 4)
        score += contribution

    signals["cluster_quality_breakdown"] = factors
    return min(1.0, max(0.0, score))


def compute_confidence_score(signals: Dict[str, Any]) -> float:
    """How confident are we this is a REAL, NEW, actionable trend?

    Combines cluster quality with temporal novelty and evidence specificity.
    This is the score that matters for downstream decisions.

    Returns:
        Float in [0, 1]. Higher = more confident.
        <0.40 = low (noise or stale recurring topic)
        0.40-0.60 = moderate (emerging, needs review)
        0.60-0.80 = high (confirmed, multi-source)
        0.80+ = very high (novel, well-evidenced, authoritative)
    """
    cluster_quality = signals.get("cluster_quality_score", 0.5)
    novelty = signals.get("novelty_score", 1.0)
    source_diversity = signals.get("source_diversity", 0.0)
    specificity = _compute_specificity(signals)

    _conf_defaults = {
        "temporal_novelty": 0.30,
        "cluster_quality": 0.25,
        "source_corroboration": 0.25,
        "evidence_specificity": 0.20,
    }
    try:
        from app.learning.weight_learner import compute_learned_weights
        _conf_w = compute_learned_weights("confidence", _conf_defaults)
    except Exception:
        _conf_w = _conf_defaults

    factors = {
        "temporal_novelty": {
            "weight": _conf_w.get("temporal_novelty", 0.30), "raw": round(novelty, 3),
        },
        "cluster_quality": {
            "weight": _conf_w.get("cluster_quality", 0.25), "raw": round(cluster_quality, 3),
        },
        "source_corroboration": {
            "weight": _conf_w.get("source_corroboration", 0.25), "raw": round(source_diversity, 3),
        },
        "evidence_specificity": {
            "weight": _conf_w.get("evidence_specificity", 0.20), "raw": round(specificity, 3),
        },
    }

    score = 0.0
    for info in factors.values():
        contribution = info["weight"] * info["raw"]
        info["contribution"] = round(contribution, 4)
        score += contribution

    signals["confidence_breakdown"] = factors
    return min(1.0, max(0.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM PREDICTION (I2 — Meltwater Predictive Analytics inspired)
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum_prediction(signals: Dict[str, Any]) -> str:
    """
    Predict whether a trend's spike will grow or fade.

    Uses velocity_history (from T1 temporal histogram) to compute
    recent slope and classify momentum trajectory.

    Returns:
        "likely_growing": Recent velocity increasing rapidly
        "likely_fading": Recent velocity decreasing
        "stable": No clear direction
        "insufficient_data": Not enough history to predict

    REF: Meltwater Predictive Analytics — "forecast whether a spike in
         mentions is likely to grow, or fade."
    """
    velocity_history = signals.get("velocity_history", [])

    if len(velocity_history) < 3:
        return "insufficient_data"

    # Use last 3 bins to compute recent slope
    recent = velocity_history[-3:]
    slope = (recent[-1] - recent[0]) / 2.0

    # Also check overall mean for context
    full_mean = sum(velocity_history) / len(velocity_history)

    # Threshold: slope needs to be meaningful relative to mean
    # A slope of 0.1 means nothing if mean velocity is 50
    threshold = max(0.5, full_mean * 0.3) if full_mean > 0 else 0.5

    if slope > threshold:
        prediction = "likely_growing"
    elif slope < -threshold:
        prediction = "likely_fading"
    else:
        prediction = "stable"

    logger.debug(
        f"Momentum prediction: {prediction} "
        f"(slope={slope:.2f}, threshold={threshold:.2f}, "
        f"recent={recent}, mean={full_mean:.2f})"
    )

    signals["momentum_prediction"] = prediction
    signals["momentum_slope"] = round(slope, 3)

    return prediction


# ══════════════════════════════════════════════════════════════════════════════
# PERCENTILE COMPUTATION (T2 — for two-pass classification)
# ══════════════════════════════════════════════════════════════════════════════

def compute_percentiles(
    scores: List[float], p10_pct: float = 10.0, p50_pct: float = 50.0
) -> Dict[str, float]:
    """
    Compute actual percentile thresholds from a distribution of scores.

    Used by engine._phase_signals two-pass: first compute all trend_scores,
    then derive actual P10/P50, then re-classify signal strengths.

    Edge cases:
    - <5 scores → fall back to default 0.2/0.5 (too few for meaningful percentiles)
    - All same score → p10=p50=that score (everything is "steady")
    - All zeros → p10=0, p50=0 (everything is noise)

    Args:
        scores: List of trend_score values from all clusters.
        p10_pct: Percentile for noise threshold (default 10th).
        p50_pct: Percentile for strong threshold (default 50th).

    Returns:
        {"p10": float, "p50": float, "min": float, "max": float, "count": int}
    """
    if len(scores) < 5:
        logger.debug(
            f"Percentile computation: only {len(scores)} scores, "
            f"using defaults (p10=0.2, p50=0.5)"
        )
        return {"p10": 0.2, "p50": 0.5, "min": 0.0, "max": 0.0, "count": len(scores)}

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def _percentile(pct: float) -> float:
        """Linear interpolation percentile (same as numpy default)."""
        idx = (pct / 100.0) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_scores[lower] * (1 - frac) + sorted_scores[upper] * frac

    p10 = _percentile(p10_pct)
    p50 = _percentile(p50_pct)

    logger.debug(
        f"Percentile computation: n={n}, p10={p10:.4f}, p50={p50:.4f}, "
        f"range=[{sorted_scores[0]:.4f}, {sorted_scores[-1]:.4f}]"
    )

    return {
        "p10": round(p10, 4),
        "p50": round(p50, 4),
        "min": round(sorted_scores[0], 4),
        "max": round(sorted_scores[-1], 4),
        "count": n,
    }


def _compute_specificity(signals: Dict[str, Any]) -> float:
    """
    How specific/concrete is this trend?

    Combines entity density, depth, and financial indicators.
    Specific trends are more actionable than vague ones.

    "RBI raised repo rate by 25bps affecting NBFCs" (specificity ~0.8)
    vs "Market conditions are changing" (specificity ~0.2)
    """
    entity_density = signals.get("entity_density", 0.0)
    depth = signals.get("depth_score", 0.5)
    financial = float(signals.get("financial_indicator", False))
    company_count = signals.get("company_count", 0)

    entity_norm = min(1.0, entity_density / 5.0)
    company_norm = min(1.0, company_count / 3.0)

    return (
        0.30 * entity_norm
        + 0.25 * depth
        + 0.25 * financial
        + 0.20 * company_norm
    )
