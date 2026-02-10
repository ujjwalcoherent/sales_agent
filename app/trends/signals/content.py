"""
Content quality signal computation for news trend analysis.

Measures the depth and sentiment characteristics of articles in a cluster.
Distinguishes between shallow wire reports and deep investigative coverage.

SIGNALS:
  depth_score:           Average article length (0-1 scale).
  sentiment_mean:        Overall sentiment direction (-1 to +1).
  sentiment_variance:    How much sentiment varies across articles.
  controversy_index:     High variance x high diversity = genuinely polarizing.

SENTIMENT METHOD:
  VADER (Valence Aware Dictionary and sEntiment Reasoner) — a rule-based
  model tuned for social media and news text. Much more accurate than
  keyword counting (~0.87 F1 vs ~0.60 for keyword lists).

  Falls back to keyword estimation if VADER is not installed.

REF: Hutto & Gilbert, "VADER: A Parsimonious Rule-based Model for Sentiment
     Analysis of Social Media Text" (ICWSM 2014)
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Try to load VADER; fall back to keyword-based if unavailable
_vader_analyzer = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()
    logger.debug("VADER sentiment analyzer loaded")
except ImportError:
    logger.debug("vaderSentiment not installed, using keyword fallback")

# Fallback keyword lexicon (only used if VADER is not installed)
_POSITIVE_WORDS = {
    "growth", "profit", "surge", "boom", "soar", "gain", "rise",
    "record", "strong", "success", "launch", "expand", "invest",
    "approve", "upgrade", "improve", "breakthrough", "innovation",
    "partnership", "agreement", "opportunity", "milestone", "revenue",
    "positive", "bullish", "optimistic", "recovery", "rally",
}

_NEGATIVE_WORDS = {
    "loss", "crash", "fall", "decline", "drop", "plunge", "slump",
    "crisis", "default", "bankrupt", "layoff", "shutdown", "ban",
    "penalty", "fraud", "scam", "scandal", "violation", "warning",
    "risk", "threat", "bearish", "pessimistic", "recession", "debt",
    "downturn", "failure", "concern", "volatility", "uncertainty",
}


def compute_content_signals(articles: list) -> Dict[str, Any]:
    """
    Compute all content quality signals from a list of articles.

    Args:
        articles: List of article objects with .title (str), .summary (str),
                  .word_count (int), .sentiment_score (float, optional).

    Returns:
        Dict with keys: depth_score, sentiment_mean, sentiment_variance,
        controversy_index.
    """
    if not articles:
        return _empty_signals()

    return {
        "depth_score": _compute_depth(articles),
        "sentiment_mean": _compute_sentiment_mean(articles),
        "sentiment_variance": _compute_sentiment_variance(articles),
        "controversy_index": _compute_controversy(articles),
    }


def _compute_depth(articles: list) -> float:
    """
    Average content depth: mean(word_count) / 500, clamped to [0, 1].

    Interpretation:
      <0.2: Wire headlines (50-100 words) — shallow, breaking
      0.3-0.5: Standard news articles (150-250 words)
      0.5-0.8: In-depth reporting (250-400 words)
      >0.8: Long-form analysis (400+ words) — deep, analytical
    """
    word_counts = []
    for a in articles:
        wc = getattr(a, 'word_count', 0)
        if wc == 0:
            summary = getattr(a, 'summary', '') or ''
            wc = len(summary.split())
        word_counts.append(wc)

    if not word_counts:
        return 0.5

    avg_words = sum(word_counts) / len(word_counts)
    return min(1.0, avg_words / 500.0)


def _compute_sentiment_mean(articles: list) -> float:
    """
    Average sentiment across all articles in the cluster.
    Range: -1.0 (strongly negative) to +1.0 (strongly positive).
    """
    sentiments = _get_sentiments(articles)
    if not sentiments:
        return 0.0
    return sum(sentiments) / len(sentiments)


def _compute_sentiment_variance(articles: list) -> float:
    """Variance of sentiment scores across articles."""
    sentiments = _get_sentiments(articles)
    if len(sentiments) < 2:
        return 0.0

    mean = sum(sentiments) / len(sentiments)
    variance = sum((s - mean) ** 2 for s in sentiments) / len(sentiments)
    return variance


def _compute_controversy(articles: list) -> float:
    """
    Controversy index: sentiment_variance x source_diversity.

    Genuinely controversial when BOTH different sentiment across sources
    AND those sources are independent (not echo chamber).
    """
    variance = _compute_sentiment_variance(articles)

    unique_sources = len({
        getattr(a, 'source_id', '') or getattr(a, 'source_name', '')
        for a in articles
    })
    diversity = unique_sources / max(len(articles), 1)

    return variance * diversity


def _get_sentiments(articles: list) -> List[float]:
    """Get sentiment scores for all articles (VADER > pre-computed > keyword fallback)."""
    sentiments = []
    for a in articles:
        score = getattr(a, 'sentiment_score', None)
        if score is not None:
            sentiments.append(score)
        else:
            sentiments.append(_analyze_sentiment(a))
    return sentiments


def analyze_article_sentiment(article) -> float:
    """
    Analyze sentiment for a single article using VADER > keyword fallback.

    Public API — used by engine Phase 2.7 for pre-computing per-article sentiment
    before clustering, so temporal histograms have real sentiment data.

    VADER understands negation ("not good"), intensifiers ("very bad"),
    punctuation ("great!!!"), and contextual valence shifts.

    Returns:
        Compound sentiment score in [-1.0, 1.0].
    """
    title = getattr(article, 'title', '') or ''
    summary = getattr(article, 'summary', '') or ''
    text = f"{title}. {summary}"

    if _vader_analyzer is not None:
        try:
            scores = _vader_analyzer.polarity_scores(text)
            return scores['compound']  # Already in [-1, 1] range
        except Exception:
            pass

    return _keyword_sentiment(text)


# Keep backward compat alias
_analyze_sentiment = analyze_article_sentiment


def _keyword_sentiment(text: str) -> float:
    """Simple keyword-based sentiment estimation (fallback only)."""
    words = set(re.findall(r'[a-z]+', text.lower()))
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)

    if pos_count + neg_count == 0:
        return 0.0

    return (pos_count - neg_count) / (pos_count + neg_count + 1)


def _empty_signals() -> Dict[str, Any]:
    """Return zero signals when no articles are available."""
    return {
        "depth_score": 0.5,
        "sentiment_mean": 0.0,
        "sentiment_variance": 0.0,
        "controversy_index": 0.0,
    }
