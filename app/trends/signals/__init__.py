"""
Signal computation modules for trend quality assessment.

Each module computes a family of signals independently. Adding a new signal
family = create a new file here and import it below. No other code changes needed.

EXTENSIBILITY: To add a new signal family:
1. Create app/trends/signals/your_signal.py
2. Define: compute_your_signals(articles) -> Dict[str, Any]
3. Import it below and add to compute_all_signals()
4. The composite module picks it up automatically

Modules:
- temporal.py: velocity, acceleration, burst_score, recency
- source.py: authority_weighted, tier_distribution, agreement
- content.py: depth, sentiment_distribution, controversy
- entity.py: key_person_flag, company_count, entity_density
- market.py: regulatory_flag, financial_indicator
- composite.py: actionability_score, trend_score, cluster_quality_score, confidence_score
"""

import logging
from typing import Any, Dict, List

from .temporal import compute_temporal_signals, compute_temporal_histogram
from .source import compute_source_signals
from .content import compute_content_signals
from .entity import compute_entity_signals
from .market import compute_market_signals
from .composite import (
    compute_actionability_score,
    compute_cluster_quality_score,
    compute_confidence_score,
    compute_trend_score,
    classify_signal_strength,
    compute_momentum_prediction,
    compute_percentiles,
)

logger = logging.getLogger(__name__)


def compute_all_signals(articles: list, all_articles: list = None) -> Dict[str, Any]:
    """
    Compute ALL signals for a group of articles (typically one cluster).

    Runs each signal module independently, merges into a flat dict,
    then computes the final composite scores on top.

    Args:
        articles: List of article objects belonging to a single cluster.
        all_articles: ALL articles in the pipeline (for cross-cluster metrics
                      like cross-citation and originality scoring). If None,
                      source signals use static credibility fallback.

    Returns:
        Flat dict of all signals including:
        - All temporal signals (velocity, acceleration, burst_score, recency_score, ...)
        - All source signals (authority_weighted, tier_distribution, ...)
        - All content signals (depth_score, sentiment_mean, ...)
        - All entity signals (key_person_flag, company_count, entity_density, ...)
        - All market signals (regulatory_flag, financial_indicator, ...)
        - Composite scores (actionability_score, trend_score, signal_strength)
    """
    if not articles:
        return {"actionability_score": 0.0, "trend_score": 0.0, "signal_strength": "noise"}

    # Compute each signal family independently
    signals: Dict[str, Any] = {}

    try:
        signals.update(compute_temporal_signals(articles))
    except Exception as e:
        logger.warning(f"Temporal signal computation failed: {e}")

    # Temporal histogram (BERTopic topics_over_time — sparkline data)
    try:
        hist_data = compute_temporal_histogram(articles)
        signals["temporal_histogram"] = hist_data.get("temporal_histogram", [])
        signals["velocity_history"] = hist_data.get("velocity_history", [])
        signals["first_seen_at"] = hist_data.get("first_seen_at")
        signals["last_seen_at"] = hist_data.get("last_seen_at")
        signals["momentum_label"] = hist_data.get("momentum_label", "")
    except Exception as e:
        logger.warning(f"Temporal histogram computation failed: {e}")

    try:
        signals.update(compute_source_signals(articles, all_articles=all_articles))
    except Exception as e:
        logger.warning(f"Source signal computation failed: {e}")

    try:
        signals.update(compute_content_signals(articles))
    except Exception as e:
        logger.warning(f"Content signal computation failed: {e}")

    try:
        signals.update(compute_entity_signals(articles))
    except Exception as e:
        logger.warning(f"Entity signal computation failed: {e}")

    try:
        signals.update(compute_market_signals(articles))
    except Exception as e:
        logger.warning(f"Market signal computation failed: {e}")

    # CMI service relevance (cluster-level average of per-article scores)
    # Stored on articles by engine._phase_cmi_relevance() as _cmi_relevance_score
    try:
        cmi_scores = [
            getattr(a, '_cmi_relevance_score', 0.0) for a in articles
        ]
        if cmi_scores:
            signals["cmi_relevance"] = sum(cmi_scores) / len(cmi_scores)
            signals["cmi_relevance_min"] = min(cmi_scores)
            signals["cmi_relevance_max"] = max(cmi_scores)
        else:
            signals["cmi_relevance"] = 0.0
    except Exception as e:
        logger.warning(f"CMI relevance signal computation failed: {e}")

    # Compute trigger event (buying intent) signals from pre-classified articles
    # These come from app/news/event_classifier.py which runs before clustering
    try:
        trigger_events = {}
        max_urgency = 0.0
        for a in articles:
            evt = getattr(a, '_trigger_event', 'general')
            urg = getattr(a, '_trigger_urgency', 0.3)
            trigger_events[evt] = trigger_events.get(evt, 0) + 1
            max_urgency = max(max_urgency, urg)
        # Dominant event type (most frequent non-general)
        non_general = {k: v for k, v in trigger_events.items() if k != 'general'}
        if non_general:
            dominant_event = max(non_general, key=non_general.get)
            event_concentration = non_general[dominant_event] / max(len(articles), 1)
        else:
            dominant_event = 'general'
            event_concentration = 0.0
        signals["trigger_event"] = dominant_event
        signals["trigger_urgency"] = max_urgency
        signals["trigger_event_concentration"] = event_concentration
        signals["trigger_event_distribution"] = trigger_events
    except Exception as e:
        logger.warning(f"Trigger event signal computation failed: {e}")

    # Compute composite scores on top of the raw signals
    # NOTE: These functions MUTATE signals dict to add score breakdowns (T3)
    signals["actionability_score"] = compute_actionability_score(signals)
    signals["trend_score"] = compute_trend_score(signals)
    signals["cluster_quality_score"] = compute_cluster_quality_score(signals)
    # Real confidence = cluster quality + temporal novelty + evidence specificity
    # (novelty_score is injected by _layer_temporalize before this is recomputed)
    signals["confidence_score"] = compute_confidence_score(signals)

    # Momentum prediction (I2 — Meltwater Predictive Analytics)
    try:
        compute_momentum_prediction(signals)
    except Exception as e:
        logger.warning(f"Momentum prediction failed: {e}")

    # Classify signal strength (BERTrend P10/P50 method)
    # NOTE: Uses default thresholds here. Engine re-classifies with actual
    # percentiles in two-pass (T2). This initial pass provides a baseline.
    strength = classify_signal_strength(
        popularity_score=signals["trend_score"],
        acceleration=signals.get("acceleration", 0.0),
    )
    signals["signal_strength"] = strength.value

    return signals
