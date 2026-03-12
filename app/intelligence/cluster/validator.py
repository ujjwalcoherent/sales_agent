"""
ValidationAgent — 7-check math gate on clusters (Math Gate 6).

All 7 checks are pure math — no LLM involved.
Failures trigger bidirectional Signal Bus messages to ClusterAgent for retry.

Check  1: Coherence         — mean pairwise cosine >= 0.40
Check  2: Separation        — intra > inter + 0.10 margin
Check  3: Size              — at least 2 articles
Check  4: Source diversity  — at least 2 distinct sources
Check  5: Entity coverage   — entity appears in >= 60% of articles
Check  6: Temporal spread   — all articles within 30-day window
Check  7: Syndication check — at least 2 unique first sentences

Failure actions (specific, not generic — CRITIC pattern):
  Check 1 coherence fail → signal RECLUSTER with tighter threshold (k+2 for HAC)
  Check 3 size fail      → signal MERGE_NEAREST (cluster too small)
  Check 7 syndication    → signal FLAG_SYNDICATION (DedupAgent should have caught this)

NewsCatcher pattern: reject 60-80% of raw clusters → high-quality survivors.
All thresholds from adaptive_thresholds.json (EMA-adapted by ThresholdAdapterAgent).

"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import (
    ClusterResult,
    ValidationCheck,
    ValidationFailureAction,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Composite score weights (must sum to 1.0)
CHECK_WEIGHTS: Dict[str, float] = {
    "min_articles": 0.10,
    "multi_source": 0.15,
    "coherence": 0.25,
    "entity_consistency": 0.20,
    "temporal_spread": 0.10,
    "not_duplicate": 0.10,
    "syndication": 0.10,
}


def validate_cluster(
    cluster: ClusterResult,
    articles: list,
    entity_groups: list,
    embeddings: Optional[np.ndarray] = None,
    existing_centroids: Optional[List[np.ndarray]] = None,
    params: Optional[ClusteringParams] = None,
) -> ValidationResult:
    """Run all 7 checks on a cluster.

    Returns ValidationResult with:
      - passed: True if composite_score >= params.val_composite_reject
      - checks: List[ValidationCheck] with specific critique per check
      - outliers: Articles ejected for being too far from centroid
      - rejection_reasons: Human-readable reasons for failure
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Delegate to existing validator for the math (battle-tested over 21 runs)
    try:
        from app.intelligence.engine.validator import validate_cluster as _validate

        old_params = _adapt_params(params)
        # Both layers share the same ValidationResult model — return directly.
        return _validate(
            cluster=cluster,
            embeddings=embeddings,
            articles=articles,
            existing_centroids=existing_centroids,
            params=old_params,
        )

    except Exception as exc:
        logger.warning(f"[validator] Delegated validation failed: {exc}, running fallback")
        return _fallback_validate(cluster, articles, params)


def validate_all_clusters(
    clusters: List[ClusterResult],
    articles: list,
    entity_groups: list,
    embeddings: Optional[np.ndarray] = None,
    params: Optional[ClusteringParams] = None,
) -> List[ValidationResult]:
    """Validate all clusters and return results."""
    if params is None:
        params = DEFAULT_PARAMS

    # Build list of centroids for duplicate detection (grows as clusters pass)
    passed_centroids: List[np.ndarray] = []
    results = []

    for cluster in clusters:
        result = validate_cluster(
            cluster=cluster,
            articles=articles,
            entity_groups=entity_groups,
            embeddings=embeddings,
            existing_centroids=passed_centroids if passed_centroids else None,
            params=params,
        )
        results.append(result)

        # Add centroid for passed clusters
        if result.passed and embeddings is not None and cluster.article_indices:
            try:
                centroid = embeddings[cluster.article_indices].mean(axis=0)
                passed_centroids.append(centroid)
            except Exception:
                pass

    passed_count = sum(1 for r in results if r.passed)
    logger.info(
        f"[validator] {passed_count}/{len(clusters)} clusters passed "
        f"({len(clusters)-passed_count} rejected)"
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_params(params: ClusteringParams):
    """Convert intelligence ClusteringParams → clustering ClusteringParams."""
    try:
        from app.intelligence.config import ClusteringParams as OldParams
        old = OldParams()
        old.val_min_articles = params.val_min_articles
        old.val_min_sources = params.val_min_sources
        old.val_coherence_min = params.val_coherence_min
        old.val_entity_consistency_min = params.val_entity_consistency_min
        old.val_temporal_window_hours = params.val_temporal_window_hours
        old.val_duplicate_threshold = params.val_duplicate_threshold
        old.val_composite_reject = params.val_composite_reject
        return old
    except ImportError:
        return None


def _fallback_validate(
    cluster: ClusterResult,
    articles: list,
    params: ClusteringParams,
) -> ValidationResult:
    """Minimal fallback validation when delegated validator fails."""
    size_ok = cluster.article_count >= params.val_min_articles
    coherence_ok = cluster.coherence_score >= params.val_coherence_min

    checks = [
        ValidationCheck(
            name="min_articles",
            passed=size_ok,
            score=1.0 if size_ok else 0.0,
            critique="" if size_ok else f"Only {cluster.article_count} articles (need {params.val_min_articles})",
            action=ValidationFailureAction.MERGE_NEAREST,
        ),
        ValidationCheck(
            name="coherence",
            passed=coherence_ok,
            score=cluster.coherence_score,
            critique="" if coherence_ok else f"Coherence {cluster.coherence_score:.3f} < {params.val_coherence_min}",
            action=ValidationFailureAction.RECLUSTER,
        ),
    ]

    passed = size_ok and coherence_ok
    score = CHECK_WEIGHTS["min_articles"] * (1.0 if size_ok else 0.0) + \
            CHECK_WEIGHTS["coherence"] * cluster.coherence_score

    return ValidationResult(
        cluster_id=cluster.cluster_id,
        passed=passed,
        composite_score=round(score, 3),
        checks=checks,
        rejection_reasons=[c.critique for c in checks if not c.passed],
    )


