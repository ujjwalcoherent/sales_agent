"""
Source quality signal computation for news trend analysis.

Measures how trustworthy and diverse the coverage of a trend is.
A trend covered by 5 Tier-1 sources with agreeing facts is far more
reliable than one covered by 1 Tier-4 blog with speculation.

SIGNALS:
  authority_weighted:  Average source credibility (0.0-1.0).
  tier_distribution:   Count of articles per source tier.
  tier_1_ratio:        What fraction of articles are from Tier-1 sources.
  source_diversity:    Unique publishers / total articles.
  source_agreement:    Entity overlap between articles (do sources agree?).

REF: Source credibility tiers defined in app/config.py NEWS_SOURCES.
     Tier 1 (0.95+): PIB, RBI, ET, Mint, Moneycontrol
     Tier 2 (0.85-0.94): YourStory, Inc42, TechCrunch India
     Tier 3 (0.70-0.84): Google News aggregator
     Tier 4 (0.50-0.69): Social media, blogs
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


def compute_source_signals(articles: list) -> Dict[str, Any]:
    """
    Compute all source quality signals from a list of articles.

    Args:
        articles: List of article objects with .source_credibility (float),
                  .source_id (str), .source_name (str), .source_tier (str/enum),
                  .entity_names (list of str).

    Returns:
        Dict with keys: authority_weighted, tier_distribution, tier_1_ratio,
        source_diversity, source_agreement, unique_sources.
    """
    if not articles:
        return _empty_signals()

    return {
        "authority_weighted": _compute_authority(articles),
        "tier_distribution": _compute_tier_distribution(articles),
        "tier_1_ratio": _compute_tier_1_ratio(articles),
        "source_diversity": _compute_source_diversity(articles),
        "source_agreement": _compute_source_agreement(articles),
        "unique_sources": _count_unique_sources(articles),
    }


def _compute_authority(articles: list) -> float:
    """
    Average source credibility across all articles in this cluster.

    Weighted by nothing — each article counts equally. This prevents a
    single Tier-1 article from masking 20 Tier-4 articles.

    Result:
      0.90+: Primarily Tier-1 sources (high confidence)
      0.80-0.89: Mix of Tier-1 and Tier-2 (good confidence)
      0.70-0.79: Primarily Tier-2/3 (moderate confidence)
      <0.70: Low-quality sources (verify independently)
    """
    credibilities = [
        getattr(a, 'source_credibility', 0.5) for a in articles
    ]
    if not credibilities:
        return 0.5
    return sum(credibilities) / len(credibilities)


def _compute_tier_distribution(articles: list) -> Dict[str, int]:
    """
    Count articles per source tier.

    Returns: {"tier_1": 5, "tier_2": 3, "tier_3": 1, "tier_4": 0, "unknown": 0}
    """
    tier_counts = Counter()
    for a in articles:
        tier = getattr(a, 'source_tier', None)
        if tier is None:
            tier_counts["unknown"] += 1
        else:
            # Handle both enum and string
            tier_str = tier.value if hasattr(tier, 'value') else str(tier)
            tier_counts[tier_str] += 1

    return dict(tier_counts)


def _compute_tier_1_ratio(articles: list) -> float:
    """
    Fraction of articles from Tier-1 (most credible) sources.

    A trend with tier_1_ratio > 0.5 is well-established fact.
    A trend with tier_1_ratio = 0 is rumor/speculation until confirmed.
    """
    if not articles:
        return 0.0

    tier_1_count = 0
    for a in articles:
        tier = getattr(a, 'source_tier', None)
        if tier is not None:
            tier_str = tier.value if hasattr(tier, 'value') else str(tier)
            if tier_str == "tier_1":
                tier_1_count += 1

    return tier_1_count / len(articles)


def _compute_source_diversity(articles: list) -> float:
    """
    Unique publishers / total articles.

    High diversity (>0.5) means many independent sources covering the story.
    Low diversity (<0.2) means few sources — could be echo chamber.

    WHY this matters for sales: If 5 different publications cover a story,
    the prospect has likely seen it too → easier conversation starter.
    """
    if not articles:
        return 0.0

    unique = len({getattr(a, 'source_id', '') or getattr(a, 'source_name', '') for a in articles})
    return unique / len(articles) if articles else 0.0


def _compute_source_agreement(articles: list) -> float:
    """
    Average pairwise entity overlap between articles.

    High agreement (>0.3) means sources report the same entities — they
    agree on the key facts. Low agreement (<0.1) means different angles.

    WHY Jaccard not cosine: Entity sets are sparse and unordered.
    Jaccard is the natural similarity measure for sets.

    NOTE: This is O(K²) where K = articles in cluster. At K<100 this is
    instant. At K>200, consider sampling or MinHash approximation.
    """
    if len(articles) < 2:
        return 1.0  # Single article trivially agrees with itself

    # Extract entity sets per article
    entity_sets: List[Set[str]] = []
    for a in articles:
        names = getattr(a, 'entity_names', [])
        entity_sets.append({n.lower() for n in names} if names else set())

    # Compute pairwise Jaccard similarity
    total_sim = 0.0
    pair_count = 0
    for i in range(len(entity_sets)):
        for j in range(i + 1, len(entity_sets)):
            sim = _jaccard(entity_sets[i], entity_sets[j])
            total_sim += sim
            pair_count += 1

    return total_sim / pair_count if pair_count > 0 else 0.0


def _count_unique_sources(articles: list) -> int:
    """Count unique source publishers."""
    sources = set()
    for a in articles:
        src = getattr(a, 'source_id', '') or getattr(a, 'source_name', '')
        if src:
            sources.add(src)
    return len(sources)


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets. 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _empty_signals() -> Dict[str, Any]:
    """Return zero signals when no articles are available."""
    return {
        "authority_weighted": 0.5,
        "tier_distribution": {},
        "tier_1_ratio": 0.0,
        "source_diversity": 0.0,
        "source_agreement": 0.0,
        "unique_sources": 0,
    }
