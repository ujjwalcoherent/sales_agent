"""
7-check cluster validation with outlier ejection and B2B filtering.

Checks:
  1. min_articles — at least 2 articles
  2. multi_source — at least 2 distinct source domains
  3. coherence — mean pairwise cosine ≥ threshold
  4. entity_consistency — ≥60% articles share a top entity
  5. temporal_proximity — all articles within time window
  6. not_duplicate — centroid not too similar to existing clusters
  7. outlier_ejection — articles with centroid distance > 2σ ejected

NewsCatcher pattern: reject 60-80% of raw clusters → high-quality survivors.

All thresholds from ClusteringParams (config.py).

Standalone test:
    python -c "
    import numpy as np
    from app.intelligence.engine.validator import validate_cluster
    from app.intelligence.models import ClusterResult
    from app.intelligence.config import DEFAULT_PARAMS
    cluster = ClusterResult(article_indices=[0,1,2], article_count=3)
    embeddings = np.random.randn(10, 1536)
    result = validate_cluster(cluster, embeddings, params=DEFAULT_PARAMS)
    print(f'Passed: {result.passed}, Score: {result.composite_score:.2f}')
    "
"""

import logging
from collections import Counter
from datetime import timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import (
    ClusterResult, OutlierRecord, ValidationCheck, ValidationResult,
)

logger = logging.getLogger(__name__)

# Weights for composite score.
# Research basis: entity_consistency (semantic) outweighs coherence (embedding)
# for B2B signal detection — Salton & McGill (1983) term weighting shows entity
# co-occurrence is more discriminative than pure cosine similarity for domain-specific text.
CHECK_WEIGHTS = {
    "min_articles": 0.10,
    "multi_source": 0.15,
    "coherence": 0.20,           # was 0.25 — reduce embedding-similarity dominance
    "entity_consistency": 0.25,  # was 0.20 — entity co-occurrence is better B2B signal
    "temporal_proximity": 0.10,
    "not_duplicate": 0.10,
    "outlier_ejection": 0.10,
}

# Coherence threshold for discovery clusters (Leiden on ungrouped articles).
# Leiden discovers latent communities — lower threshold (0.35) appropriate since
# these clusters have no entity anchor. Entity-seeded clusters use val_coherence_min (0.40).
# Campello et al. (2013): soft membership naturally produces lower within-cluster similarity.
_DISCOVERY_COHERENCE_MIN: float = 0.35


def validate_cluster(
    cluster: ClusterResult,
    embeddings: Optional[np.ndarray],
    articles: Optional[List[Any]] = None,
    existing_centroids: Optional[List[np.ndarray]] = None,
    params: Optional[ClusteringParams] = None,
) -> ValidationResult:
    """Run 7 validation checks on a cluster.

    Args:
        cluster: ClusterResult to validate.
        embeddings: Full embedding matrix (all articles).
        articles: Full article list (for source/temporal/entity checks).
        existing_centroids: Centroids of already-passed clusters (for duplicate check).
        params: Clustering parameters with thresholds.

    Returns:
        ValidationResult with pass/fail, composite score, and outlier records.
    """
    if params is None:
        params = DEFAULT_PARAMS

    checks: Dict[str, bool] = {}
    scores: Dict[str, float] = {}
    reasons: List[str] = []
    outliers: List[OutlierRecord] = []
    ejected: List[int] = []

    indices = cluster.article_indices

    # Cluster type determines adaptive thresholds and hard vetoes.
    # Entity-seeded (HAC/HDBSCAN on entity groups): high prior — entity match IS relatedness.
    # Discovery (Leiden on ungrouped articles): no entity anchor — use adaptive coherence threshold.
    is_entity_seeded = getattr(cluster, "is_entity_seeded", False)

    # Adaptive coherence threshold per cluster type.
    # Discovery clusters naturally have lower cosine similarity (~0.35-0.42) because Leiden
    # groups articles by community structure, not entity co-occurrence.
    # Campello et al. (2013): soft cluster membership ↔ lower within-cluster similarity.
    coh_threshold = params.val_coherence_min if is_entity_seeded else min(
        _DISCOVERY_COHERENCE_MIN, params.val_coherence_min
    )

    # Check 1: Minimum articles
    checks["min_articles"], scores["min_articles"] = _check_min_articles(
        indices, params.val_min_articles,
    )
    if not checks["min_articles"]:
        reasons.append(f"Only {len(indices)} article(s), need {params.val_min_articles}")

    # Check 2: Multi-source
    checks["multi_source"], scores["multi_source"] = _check_multi_source(
        indices, articles, params.val_min_sources,
    )
    if not checks["multi_source"]:
        reasons.append(f"Single source cluster, need {params.val_min_sources}+ sources")

    # Check 3: Coherence (adaptive threshold)
    checks["coherence"], scores["coherence"], raw_coherence = _check_coherence(
        indices, embeddings, coh_threshold,
    )
    if not checks["coherence"]:
        reasons.append(f"Coherence {raw_coherence:.3f} < {coh_threshold:.2f}")

    # Check 4: Entity consistency
    checks["entity_consistency"], scores["entity_consistency"], raw_consistency = _check_entity_consistency(
        indices, articles, params.val_entity_consistency_min,
    )
    if not checks["entity_consistency"]:
        reasons.append(f"Entity consistency {raw_consistency:.2f} < {params.val_entity_consistency_min}")

    # Check 5: Temporal proximity
    checks["temporal_proximity"], scores["temporal_proximity"] = _check_temporal(
        indices, articles, params.val_temporal_window_hours,
    )
    if not checks["temporal_proximity"]:
        reasons.append(f"Temporal spread exceeds {params.val_temporal_window_hours}h window")

    # Check 6: Not duplicate of existing cluster
    checks["not_duplicate"], scores["not_duplicate"] = _check_not_duplicate(
        indices, embeddings, existing_centroids, params.val_duplicate_threshold,
    )
    if not checks["not_duplicate"]:
        reasons.append(f"Duplicate of existing cluster (centroid sim > {params.val_duplicate_threshold})")

    # Check 7: Outlier ejection
    checks["outlier_ejection"], scores["outlier_ejection"], outliers, ejected = _check_outliers(
        indices, embeddings, cluster.cluster_id,
    )
    if not checks["outlier_ejection"]:
        reasons.append(f"Ejected {len(ejected)} outlier article(s)")

    # Composite score
    composite = sum(
        scores.get(check, 0.0) * weight
        for check, weight in CHECK_WEIGHTS.items()
    )

    # Hard vetoes — must pass regardless of composite score.
    #
    # Entity-seeded clusters: min_articles + not_duplicate only.
    #   Entity match is the relatedness signal (Campello et al. 2013).
    #   Two articles about the same company in different events ARE related.
    #
    # Discovery clusters: min_articles + multi_source + not_duplicate.
    #   multi_source ensures we're not just clustering a single press release.
    #   coherence is NOT a hard veto for discovery clusters — it's weighted in
    #   composite score with the adaptive 0.35 threshold instead (see coh_threshold above).
    hard_vetoes = {"min_articles", "not_duplicate"}
    if not is_entity_seeded:
        hard_vetoes.add("multi_source")
    hard_pass = all(checks.get(v, False) for v in hard_vetoes)
    passed = hard_pass and composite >= params.val_composite_reject

    checks_list = [
        ValidationCheck(
            name=name,
            passed=checks.get(name, False),
            score=round(scores.get(name, 0.0), 4),
        )
        for name in CHECK_WEIGHTS
    ]

    return ValidationResult(
        cluster_id=cluster.cluster_id,
        passed=passed,
        composite_score=round(composite, 4),
        checks=checks_list,
        rejection_reasons=reasons,
        outliers=outliers,
        ejected_article_indices=ejected,
    )


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _check_min_articles(indices: List[int], min_count: int) -> tuple:
    """Check 1: At least min_count articles."""
    passed = len(indices) >= min_count
    score = min(len(indices) / max(min_count, 1), 1.0)
    return passed, score


def _check_multi_source(
    indices: List[int],
    articles: Optional[List[Any]],
    min_sources: int,
) -> tuple:
    """Check 2: At least min_sources distinct source domains."""
    if not articles:
        return True, 1.0  # Can't check without articles

    sources: Set[str] = set()
    for idx in indices:
        if idx < len(articles):
            art = articles[idx]
            source = getattr(art, "source_id", "") or getattr(art, "source_name", "")
            if source:
                sources.add(source.lower())

    n_sources = len(sources)
    passed = n_sources >= min_sources
    score = min(n_sources / max(min_sources, 1), 1.0)
    return passed, score


def _check_coherence(
    indices: List[int],
    embeddings: Optional[np.ndarray],
    min_coherence: float,
) -> tuple:
    """Check 3: Mean pairwise cosine similarity above threshold."""
    if len(indices) < 2:
        return True, 1.0, 1.0
    if embeddings is None:
        return True, 0.5, 0.5  # No embeddings → neutral pass

    cluster_emb = embeddings[indices]
    from app.intelligence.engine.similarity import _compute_semantic
    sim = _compute_semantic(cluster_emb)
    mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
    coherence = float(sim[mask].mean()) if mask.any() else 1.0

    passed = coherence >= min_coherence
    score = min(coherence / max(min_coherence, 0.01), 1.0)
    return passed, score, coherence  # Also return raw value for logging


def _check_entity_consistency(
    indices: List[int],
    articles: Optional[List[Any]],
    min_consistency: float,
) -> tuple:
    """Check 4: Fraction of articles sharing at least one top-3 entity."""
    if not articles or len(indices) < 2:
        return True, 1.0, 1.0

    # Collect entity sets per article
    entity_lists = []
    for idx in indices:
        if idx < len(articles):
            names = getattr(articles[idx], "entity_names", [])
            entity_lists.append(set(n.lower() for n in names) if names else set())
        else:
            entity_lists.append(set())

    if not any(entity_lists):
        return True, 0.5, 0.5  # No entity data → neutral

    # Find top-3 most common entities across cluster
    all_entities: Counter = Counter()
    for eset in entity_lists:
        for e in eset:
            all_entities[e] += 1

    if not all_entities:
        return True, 0.5, 0.5

    top_entities = set(e for e, _ in all_entities.most_common(3))

    # What fraction of articles contain at least one top entity?
    matching = sum(1 for eset in entity_lists if eset & top_entities)
    consistency = matching / len(entity_lists)

    passed = consistency >= min_consistency
    score = min(consistency / max(min_consistency, 0.01), 1.0)
    return passed, score, consistency  # Also return raw for logging


def _check_temporal(
    indices: List[int],
    articles: Optional[List[Any]],
    max_window_hours: float,
) -> tuple:
    """Check 5: All articles within temporal window."""
    if not articles or len(indices) < 2:
        return True, 1.0

    timestamps = []
    for idx in indices:
        if idx < len(articles):
            pub = getattr(articles[idx], "published_at", None)
            if pub:
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                timestamps.append(pub.timestamp())

    if len(timestamps) < 2:
        return True, 1.0

    spread_hours = (max(timestamps) - min(timestamps)) / 3600.0
    passed = spread_hours <= max_window_hours
    score = max(0.0, 1.0 - (spread_hours / max(max_window_hours, 1.0)))
    return passed, score


def _check_not_duplicate(
    indices: List[int],
    embeddings: Optional[np.ndarray],
    existing_centroids: Optional[List[np.ndarray]],
    threshold: float,
) -> tuple:
    """Check 6: Cluster centroid not too similar to existing clusters."""
    if not existing_centroids:
        return True, 1.0
    if embeddings is None:
        return True, 1.0  # No embeddings → can't compute centroid, skip check

    centroid = _compute_centroid(indices, embeddings)
    if centroid is None:
        return True, 1.0

    max_sim = 0.0
    for existing in existing_centroids:
        sim = _cosine_sim(centroid, existing)
        max_sim = max(max_sim, sim)

    passed = max_sim < threshold
    score = max(0.0, 1.0 - (max_sim / max(threshold, 0.01)))
    return passed, score


def _check_outliers(
    indices: List[int],
    embeddings: Optional[np.ndarray],
    cluster_id: str,
) -> tuple:
    """Check 7: Detect and eject outlier articles (distance > 2σ from centroid)."""
    if len(indices) < 3:
        return True, 1.0, [], []
    if embeddings is None:
        return True, 1.0, [], []  # No embeddings → skip outlier ejection

    cluster_emb = embeddings[indices]
    centroid = cluster_emb.mean(axis=0)

    # Compute distances to centroid
    distances = np.array([
        1.0 - _cosine_sim(cluster_emb[i], centroid)
        for i in range(len(cluster_emb))
    ])

    mean_dist = distances.mean()
    std_dist = distances.std()

    if std_dist < 1e-8:
        return True, 1.0, [], []

    # Outliers: distance > mean + 2σ
    threshold = mean_dist + 2 * std_dist
    outlier_mask = distances > threshold

    outliers = []
    ejected = []
    for local_idx in np.where(outlier_mask)[0]:
        global_idx = indices[local_idx]
        ejected.append(global_idx)
        outliers.append(OutlierRecord(
            item_type="article",
            item_id=str(global_idx),
            reason=f"distance_to_centroid={distances[local_idx]:.3f} > threshold={threshold:.3f}",
            confidence=min((distances[local_idx] - threshold) / max(std_dist, 1e-8), 1.0),
        ))

    n_outliers = len(ejected)
    passed = n_outliers <= max(1, len(indices) // 5)  # Allow up to 20% outliers
    score = max(0.0, 1.0 - (n_outliers / max(len(indices), 1)))

    return passed, score, outliers, ejected


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_centroid(indices: List[int], embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Compute centroid embedding for a cluster."""
    if not indices:
        return None
    valid = [i for i in indices if i < len(embeddings)]
    if not valid:
        return None
    return embeddings[valid].mean(axis=0)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Delegate to shared cosine_sim_pair in similarity.py."""
    from app.intelligence.engine.similarity import cosine_sim_pair
    return cosine_sim_pair(a, b)
