"""
Temporal signal computation for news trend analysis.

Measures how a trend behaves over time: is it accelerating, bursting,
or fading? These are the strongest predictors of trend importance.

SIGNALS:
  velocity:       Articles per hour. >5/hr = breaking news.
  acceleration:   Slope of article counts over time. +ve = growing.
  burst_score:    Peak hour intensity vs average. 5.0 = 5x spike.
  recency_score:  How fresh. 1.0 = just now, 0.0 = 2+ days old.

HISTOGRAM (T1 — BERTopic topics_over_time approach):
  temporal_histogram:  Per-bin article counts + sentiment for sparkline.
  velocity_history:    Per-bin velocity deltas for trend curve.
  momentum_label:      Classified momentum from velocity curve.

REF: Kleinberg, "Bursty and Hierarchical Structure in Streams" (2003)
     — burst detection via hidden Markov model on event streams.
     BERTrend (Boutaleb et al. 2024) — exponential decay for trend freshness.
     BERTopic (Grootendorst 2022) — topics_over_time with global_tuning.
     Meltwater Predictive Analytics — forecast spike growth/fade.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_temporal_signals(articles: list) -> Dict[str, Any]:
    """
    Compute all temporal signals from a list of articles.

    Args:
        articles: List of article objects with .published_at (datetime) attribute.

    Returns:
        Dict with keys: velocity, acceleration, burst_score, recency_score,
        time_span_hours, article_count.
    """
    if not articles:
        return _empty_signals()

    # Extract timestamps, filtering None
    timestamps = []
    for a in articles:
        pub = getattr(a, 'published_at', None)
        if pub is not None:
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            timestamps.append(pub)

    if not timestamps:
        return _empty_signals()

    timestamps.sort()
    now = datetime.now(timezone.utc)

    return {
        "velocity": _compute_velocity(timestamps),
        "acceleration": _compute_acceleration(timestamps),
        "burst_score": _compute_burst_score(timestamps),
        "recency_score": _compute_recency(timestamps, now),
        "time_span_hours": _time_span_hours(timestamps),
        "article_count": len(timestamps),
    }


def _compute_velocity(timestamps: List[datetime]) -> float:
    """
    Articles per hour over the total time span.

    Interpretation:
      <0.5/hr: Slow-burn story (developing over days)
      0.5-2/hr: Normal news cycle
      2-5/hr: Hot story
      >5/hr: Breaking news

    REF: Google News uses similar velocity bucketing for story ranking.
    """
    span = _time_span_hours(timestamps)
    if span <= 0:
        # All articles published at same time — treat as a burst
        return float(len(timestamps))
    return len(timestamps) / span


def _compute_acceleration(timestamps: List[datetime]) -> float:
    """
    Trend momentum: is the story gaining or losing coverage?

    Method: Split the time span into 4 equal bins, count articles per bin,
    compute linear regression slope, normalize by mean count.

    Result:
      >0: Growing (more articles in recent bins)
      ~0: Steady
      <0: Declining (fewer articles recently)

    WHY 4 bins: Enough to detect a trend, few enough to be robust.
    With 500 articles / 5 clusters = ~100 articles per cluster, 4 bins
    gives ~25 articles per bin — statistically meaningful.
    """
    if len(timestamps) < 4:
        return 0.0

    earliest = timestamps[0]
    latest = timestamps[-1]
    span = (latest - earliest).total_seconds()
    if span <= 0:
        return 0.0

    # Split into 4 time bins
    num_bins = 4
    bin_counts = [0] * num_bins
    for ts in timestamps:
        elapsed = (ts - earliest).total_seconds()
        bin_idx = min(int(elapsed / span * num_bins), num_bins - 1)
        bin_counts[bin_idx] += 1

    # Linear regression: slope of bin_counts vs bin_index
    # Using least squares: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    n = num_bins
    x_mean = (n - 1) / 2.0
    y_mean = sum(bin_counts) / n

    numerator = sum((i - x_mean) * (bin_counts[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Normalize by mean count so acceleration is comparable across clusters
    # A slope of 5 means different things for a cluster of 10 vs 100 articles
    if y_mean > 0:
        return slope / y_mean
    return 0.0


def _compute_burst_score(timestamps: List[datetime]) -> float:
    """
    How "spiky" is the coverage? High burst = concentrated in one hour.

    Formula: max_hourly_count / mean_hourly_count

    Interpretation:
      ~1.0: Steady coverage (uniform distribution over time)
      2-3: Moderate concentration
      >5: Major spike (e.g., breaking announcement + reaction wave)

    REF: Kleinberg burst detection identifies hierarchical temporal spikes.
    This is a simplified version — sufficient for trend ranking without
    the full HMM machinery.
    """
    if len(timestamps) < 2:
        return 1.0

    # Count articles per hour
    hourly_counts: Dict[str, int] = {}
    for ts in timestamps:
        hour_key = ts.strftime("%Y-%m-%d-%H")
        hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

    if not hourly_counts:
        return 1.0

    counts = list(hourly_counts.values())
    max_count = max(counts)
    mean_count = sum(counts) / len(counts)

    if mean_count <= 0:
        return 1.0

    return max_count / mean_count


def _compute_recency(timestamps: List[datetime], now: datetime) -> float:
    """
    How fresh is the most recent article? Exponential decay (BERTrend).

    Formula: e^(-lambda * hours^2) where lambda from RECENCY_DECAY_LAMBDA env.

    Result (with default lambda=0.003):
      0 hours ago:   1.000 (just published)
      3 hours ago:   0.973
      6 hours ago:   0.898
      12 hours ago:  0.651 (half-day: still very relevant)
      24 hours ago:  0.180 (yesterday: fading fast)
      48 hours ago:  0.001 (2 days: essentially dead)

    WHY exponential over linear:
      Linear treats 1-hour-old and 12-hour-old trends similarly (both near 1.0).
      Exponential strongly favors the freshest trends — a breaking story from
      1 hour ago is MUCH more actionable for sales than one from 12 hours ago.

    REF: BERTrend (Boutaleb et al. 2024): p(t')=p(t'-1)*e^(-lambda*dt^2)
    """
    latest = max(timestamps)
    hours_since = (now - latest).total_seconds() / 3600
    lambda_decay = _get_recency_lambda()
    return math.exp(-lambda_decay * hours_since ** 2)


def _time_span_hours(timestamps: List[datetime]) -> float:
    """Total time span from first to last article, in hours."""
    if len(timestamps) < 2:
        return 0.0
    span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    return max(0.0, span)


def _empty_signals() -> Dict[str, Any]:
    """Return zero signals when no articles are available."""
    return {
        "velocity": 0.0,
        "acceleration": 0.0,
        "burst_score": 1.0,
        "recency_score": 0.0,
        "time_span_hours": 0.0,
        "article_count": 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL HISTOGRAM — BERTopic topics_over_time approach
# ══════════════════════════════════════════════════════════════════════════════

def _get_temporal_config() -> Dict[str, Any]:
    """Load temporal config from env-configurable settings. Falls back to defaults."""
    try:
        from app.config import get_settings
        s = get_settings()
        return {
            "num_bins": s.temporal_histogram_bins,
            "spike_multiplier": s.momentum_spike_multiplier,
            "momentum_window": s.momentum_window_bins,
        }
    except Exception:
        return {"num_bins": 8, "spike_multiplier": 3.0, "momentum_window": 3}


def _get_recency_lambda() -> float:
    """Load recency decay lambda from env. Falls back to 0.003."""
    try:
        from app.config import get_settings
        return get_settings().recency_decay_lambda
    except Exception:
        return 0.003


def _extract_sorted_timestamps(articles: list) -> Tuple[List[datetime], List[float]]:
    """
    Extract and sort timestamps from articles. Also returns per-article sentiment scores.

    Edge cases handled:
    - Articles without published_at → skipped with debug log
    - Naive datetimes → forced to UTC
    - Duplicate timestamps → kept (valid: multiple articles same time)
    - Future timestamps → clamped to now (clock skew / bad data)

    Returns:
        (sorted_timestamps, sentiment_scores) — parallel lists, same order.
    """
    now = datetime.now(timezone.utc)
    timestamps: List[datetime] = []
    sentiments: List[float] = []
    skipped = 0

    for a in articles:
        pub = getattr(a, 'published_at', None)
        if pub is None:
            skipped += 1
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        # Clamp future timestamps
        if pub > now:
            pub = now
        timestamps.append(pub)
        sentiments.append(getattr(a, 'sentiment_score', 0.0))

    if skipped > 0:
        logger.debug(f"Temporal histogram: skipped {skipped}/{len(articles)} articles without published_at")

    # Sort both lists together by timestamp
    if timestamps:
        paired = sorted(zip(timestamps, sentiments), key=lambda x: x[0])
        timestamps = [p[0] for p in paired]
        sentiments = [p[1] for p in paired]

    return timestamps, sentiments


def compute_temporal_histogram(articles: list) -> Dict[str, Any]:
    """
    Compute temporal histogram for sparkline visualization.

    BERTopic-inspired: bins articles into N time windows, computes per-bin
    counts and sentiment averages. Derives velocity_history for sparkline
    and momentum_label from recent velocity trajectory.

    Args:
        articles: List of article objects with .published_at and optional .sentiment_score.

    Returns:
        Dict with keys:
        - temporal_histogram: List[Dict] — [{period, count, sentiment_avg, velocity}, ...]
        - velocity_history: List[float] — per-bin velocity for sparkline
        - first_seen_at: ISO datetime string or None
        - last_seen_at: ISO datetime string or None
        - momentum_label: "accelerating"|"steady"|"decelerating"|"spiking"|""

    Edge cases:
    - Empty articles → empty histogram, no momentum
    - Single article → 1-bin histogram, "steady" momentum
    - All same timestamp → 1-bin histogram, "spiking" if many articles
    - <num_bins articles → fewer bins (min 1 per article group)
    """
    config = _get_temporal_config()
    num_bins = max(2, config["num_bins"])  # Floor at 2 bins for meaningful comparison
    spike_mult = config["spike_multiplier"]
    momentum_window = max(2, config["momentum_window"])  # Need ≥2 bins for slope

    timestamps, sentiments = _extract_sorted_timestamps(articles)

    if not timestamps:
        logger.debug("Temporal histogram: no valid timestamps, returning empty")
        return {
            "temporal_histogram": [],
            "velocity_history": [],
            "first_seen_at": None,
            "last_seen_at": None,
            "momentum_label": "",
        }

    first_seen = timestamps[0]
    last_seen = timestamps[-1]
    total_span = (last_seen - first_seen).total_seconds()

    # Edge case: all articles at same timestamp (or within 1 second)
    if total_span < 1.0:
        logger.debug(
            f"Temporal histogram: all {len(timestamps)} articles within <1s span, "
            f"single-bin histogram"
        )
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0
        histogram = [{
            "period": first_seen.isoformat(),
            "count": len(timestamps),
            "sentiment_avg": round(avg_sent, 3),
            "velocity": float(len(timestamps)),  # All at once = instantaneous burst
        }]
        return {
            "temporal_histogram": histogram,
            "velocity_history": [float(len(timestamps))],
            "first_seen_at": first_seen.isoformat(),
            "last_seen_at": last_seen.isoformat(),
            "momentum_label": "spiking" if len(timestamps) >= 3 else "steady",
        }

    # Adaptive bin count: don't create more bins than we have articles
    effective_bins = min(num_bins, len(timestamps))
    bin_duration = total_span / effective_bins
    bin_duration_hours = bin_duration / 3600.0

    logger.debug(
        f"Temporal histogram: {len(timestamps)} articles over {total_span/3600:.1f}h → "
        f"{effective_bins} bins of {bin_duration_hours:.1f}h each"
    )

    # Bin articles by time window
    bin_counts: List[int] = [0] * effective_bins
    bin_sentiments: List[List[float]] = [[] for _ in range(effective_bins)]
    bin_starts: List[datetime] = []

    for i in range(effective_bins):
        bin_starts.append(first_seen + timedelta(seconds=i * bin_duration))

    for ts, sent in zip(timestamps, sentiments):
        elapsed = (ts - first_seen).total_seconds()
        bin_idx = min(int(elapsed / bin_duration), effective_bins - 1)
        bin_counts[bin_idx] += 1
        bin_sentiments[bin_idx].append(sent)

    # Build histogram entries and velocity_history
    histogram: List[Dict[str, Any]] = []
    velocity_history: List[float] = []

    for i in range(effective_bins):
        count = bin_counts[i]
        # Velocity: articles per hour in this bin
        velocity = count / bin_duration_hours if bin_duration_hours > 0 else float(count)
        velocity_history.append(round(velocity, 3))

        # Sentiment average for this bin (0.0 if no articles or no sentiment data)
        sents = bin_sentiments[i]
        sent_avg = sum(sents) / len(sents) if sents else 0.0

        histogram.append({
            "period": bin_starts[i].isoformat(),
            "count": count,
            "sentiment_avg": round(sent_avg, 3),
            "velocity": round(velocity, 3),
        })

    # Derive momentum_label from velocity trajectory
    momentum_label = _classify_momentum(
        velocity_history, spike_mult, momentum_window
    )

    logger.debug(
        f"Temporal histogram complete: bins={effective_bins}, "
        f"momentum={momentum_label}, "
        f"velocity_range=[{min(velocity_history):.2f}, {max(velocity_history):.2f}]"
    )

    return {
        "temporal_histogram": histogram,
        "velocity_history": velocity_history,
        "first_seen_at": first_seen.isoformat(),
        "last_seen_at": last_seen.isoformat(),
        "momentum_label": momentum_label,
    }


def _classify_momentum(
    velocity_history: List[float],
    spike_multiplier: float = 3.0,
    window: int = 3,
) -> str:
    """
    Classify trend momentum from velocity history.

    Uses the last `window` bins to determine direction. Checks for spikes
    against the full history mean.

    Returns:
        "accelerating": Last window bins have positive slope (growing coverage)
        "decelerating": Last window bins have negative slope (fading)
        "spiking":      Any bin in last window exceeds spike_multiplier × mean
        "steady":       No clear direction

    Edge cases:
    - <2 bins → "steady" (insufficient data)
    - All zeros → "steady"
    - Single spike bin → "spiking" even if other bins are quiet

    REF: Meltwater predictive analytics — forecasts spike growth/fade direction.
    """
    if len(velocity_history) < 2:
        return "steady"

    # Check for spikes first (highest priority classification)
    full_mean = sum(velocity_history) / len(velocity_history)
    if full_mean > 0:
        # Check last `window` bins for spike
        recent = velocity_history[-window:]
        max_recent = max(recent)
        if max_recent > spike_multiplier * full_mean:
            logger.debug(
                f"Momentum: spiking (max_recent={max_recent:.2f} > "
                f"{spike_multiplier}×mean={spike_multiplier*full_mean:.2f})"
            )
            return "spiking"

    # Check direction from last window bins
    recent = velocity_history[-min(window, len(velocity_history)):]
    if len(recent) < 2:
        return "steady"

    # Simple slope: compare last bin to first bin in window
    # More robust than regression for small windows
    slope = recent[-1] - recent[0]

    # Threshold: need at least 20% change relative to mean to call directional
    threshold = full_mean * 0.2 if full_mean > 0 else 0.1

    if slope > threshold:
        return "accelerating"
    elif slope < -threshold:
        return "decelerating"
    return "steady"
