"""
Source quality signal computation for news trend analysis.

Measures how trustworthy and diverse the coverage of a trend is.
A trend covered by 5 Tier-1 sources with agreeing facts is far more
reliable than one covered by 1 Tier-4 blog with speculation.

SIGNALS:
  authority_weighted:  Dynamic credibility combining base tier + cross-citation +
                       originality + factual consistency (0.0-1.0).
  tier_distribution:   Count of articles per source tier.
  tier_1_ratio:        What fraction of articles are from Tier-1 sources.
  source_diversity:    Unique publishers / total articles.
  source_agreement:    Entity overlap between articles (do sources agree?).
  cross_citation:      How often other sources corroborate this source.
  originality_score:   % of articles that are original (not republished).

CREDIBILITY SCORING (V2 — dynamic, not static):
  credibility = 0.40 * base_tier + 0.25 * cross_citation + 0.20 * originality + 0.15 * agreement
  Base tier is the static floor from NEWS_SOURCES config.
  Other components are computed per-run from actual article behavior.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


def compute_source_signals(articles: list, all_articles: list = None) -> Dict[str, Any]:
    """
    Compute all source quality signals from a list of articles.

    Args:
        articles: List of article objects for THIS cluster.
        all_articles: All articles in the pipeline (for cross-citation scoring).
                      If None, falls back to static credibility.

    Returns:
        Dict with keys: authority_weighted, tier_distribution, tier_1_ratio,
        source_diversity, source_agreement, unique_sources, cross_citation,
        originality_score, dynamic_credibility.
    """
    if not articles:
        return _empty_signals()

    # Core signals
    base_authority = _compute_authority(articles)
    diversity = _compute_source_diversity(articles)
    agreement = _compute_source_agreement(articles)

    # Dynamic credibility components
    cross_citation = _compute_cross_citation(articles, all_articles) if all_articles else 0.5
    originality = _compute_originality(articles) if all_articles else 0.5

    # V2: Dynamic credibility = weighted composite (overridable via .env)
    from app.config import get_settings
    import json as _json
    _sw = _json.loads(get_settings().source_credibility_weights)
    dynamic_credibility = (
        _sw.get("base_authority", 0.40) * base_authority
        + _sw.get("cross_citation", 0.25) * cross_citation
        + _sw.get("originality", 0.20) * originality
        + _sw.get("agreement", 0.15) * agreement
    )

    return {
        "authority_weighted": round(dynamic_credibility, 3),
        "base_authority": round(base_authority, 3),
        "tier_distribution": _compute_tier_distribution(articles),
        "tier_1_ratio": _compute_tier_1_ratio(articles),
        "source_diversity": round(diversity, 3),
        "source_agreement": round(agreement, 3),
        "unique_sources": _count_unique_sources(articles),
        "cross_citation": round(cross_citation, 3),
        "originality_score": round(originality, 3),
        "dynamic_credibility": round(dynamic_credibility, 3),
    }


def _compute_authority(articles: list) -> float:
    """
    Average source credibility across all articles in this cluster.

    Uses adaptive credibility from Source Bandit (Thompson Sampling) when
    available. Falls back to static credibility_score from config.

    Result:
      0.90+: Primarily Tier-1 sources (high confidence)
      0.80-0.89: Mix of Tier-1 and Tier-2 (good confidence)
      0.70-0.79: Primarily Tier-2/3 (moderate confidence)
      <0.70: Low-quality sources (verify independently)
    """
    # Try bandit-blended credibility first
    bandit = None
    try:
        from app.learning.source_bandit import SourceBandit
        bandit = SourceBandit()
    except Exception:
        pass

    credibilities = []
    for a in articles:
        src = getattr(a, 'source_id', '') or ''
        if bandit and src:
            credibilities.append(bandit.get_adaptive_credibility(src))
        else:
            credibilities.append(getattr(a, 'source_credibility', 0.5))

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


def _compute_cross_citation(cluster_articles: list, all_articles: list) -> float:
    """
    V2: How well is this cluster's content corroborated by multiple sources?

    If the same story appears from multiple independent sources, it's more credible.
    If only one source reports it, it could be unverified.

    Returns 0.0-1.0 where 1.0 = every article is corroborated by other sources.
    """
    if not cluster_articles or len(cluster_articles) < 2:
        return 0.5  # Can't measure cross-citation with 1 article

    # Count unique sources in this cluster
    cluster_sources = set()
    for a in cluster_articles:
        src = getattr(a, 'source_id', '') or getattr(a, 'source_name', '')
        if src:
            cluster_sources.add(src)

    if len(cluster_sources) <= 1:
        return 0.2  # Single source — uncorroborated

    # More sources = higher corroboration
    # 2 sources = 0.5, 3 = 0.7, 4+ = 0.85+
    ratio = min(1.0, len(cluster_sources) / max(len(cluster_articles), 1))
    return min(1.0, 0.3 + ratio * 0.7)


def _compute_originality(articles: list) -> float:
    """
    V2: What fraction of articles from each source are original vs republished?

    Sources that mostly contribute original content score higher.
    Uses title similarity as a proxy — if titles are very similar, likely republished.

    Returns 0.0-1.0 where 1.0 = all articles appear original.
    """
    if len(articles) < 2:
        return 1.0

    titles = [getattr(a, 'title', '').lower().strip() for a in articles]
    unique_titles = set()
    original_count = 0

    for title in titles:
        # Simple dedup: if title is >80% similar to any seen title, it's a republish
        is_original = True
        for seen in unique_titles:
            if _title_similarity(title, seen) > 0.8:
                is_original = False
                break
        if is_original:
            unique_titles.add(title)
            original_count += 1

    return original_count / len(articles) if articles else 0.5


def _title_similarity(a: str, b: str) -> float:
    """Quick word-overlap similarity between two titles."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _empty_signals() -> Dict[str, Any]:
    """Return zero signals when no articles are available."""
    return {
        "authority_weighted": 0.5,
        "base_authority": 0.5,
        "tier_distribution": {},
        "tier_1_ratio": 0.0,
        "source_diversity": 0.0,
        "source_agreement": 0.0,
        "unique_sources": 0,
        "cross_citation": 0.0,
        "originality_score": 0.0,
        "dynamic_credibility": 0.5,
    }
