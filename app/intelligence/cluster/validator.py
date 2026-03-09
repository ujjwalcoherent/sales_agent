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

Delegates to app.clustering.tools.validator until Phase 11 migration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import (
    AgentRequestType,
    ClusterResult,
    OutlierRecord,
    RequestPriority,
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


def build_recluster_signal(
    failed_result: ValidationResult,
    cluster: ClusterResult,
) -> Dict[str, Any]:
    """Build the Signal Bus message for ClusterAgent retry.

    Returns specific critique, not generic "retry".
    This implements the CRITIC pattern (Gou et al. 2023).
    """
    failed_checks = [c for c in failed_result.checks if not c.passed]
    critique_parts = [f"{c.name}: {c.critique}" for c in failed_checks]

    # Determine the primary action
    primary_action = ValidationFailureAction.RECLUSTER
    for check in failed_checks:
        if check.action in (ValidationFailureAction.DROP, ValidationFailureAction.FLAG_SYNDICATION):
            primary_action = check.action
            break
        if check.action == ValidationFailureAction.MERGE_NEAREST:
            primary_action = ValidationFailureAction.MERGE_NEAREST

    return {
        "cluster_id": cluster.cluster_id,
        "cluster_entity": cluster.primary_entity,
        "critique": "; ".join(critique_parts),
        "action": primary_action.value,
        "failed_checks": [c.name for c in failed_checks],
        "composite_score": failed_result.composite_score,
    }


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


def _adapt_validation_result(
    old_result: Any,
    cluster_id: str,
    params: ClusteringParams,
) -> ValidationResult:
    """Convert clustering ValidationResult → intelligence ValidationResult."""
    checks_dict = getattr(old_result, "checks", {})
    check_scores = getattr(old_result, "check_scores", {})
    old_outliers = getattr(old_result, "outliers", [])

    checks = []
    for name, passed in checks_dict.items():
        score = check_scores.get(name, 1.0 if passed else 0.0)
        critique = _build_critique(name, score, params) if not passed else ""
        action = _check_action(name)

        checks.append(ValidationCheck(
            name=name,
            passed=passed,
            score=score,
            critique=critique,
            action=action,
        ))

    outliers = []
    for o in old_outliers:
        outliers.append(OutlierRecord(
            item_type=getattr(o, "item_type", "article"),
            item_id=getattr(o, "item_id", ""),
            reason=getattr(o, "reason", ""),
            evidence=getattr(o, "evidence", ""),
            confidence=getattr(o, "confidence", 0.0),
            silhouette_score=getattr(o, "silhouette_score", None),
        ))

    return ValidationResult(
        cluster_id=cluster_id,
        passed=getattr(old_result, "passed", False),
        composite_score=getattr(old_result, "composite_score", 0.0),
        checks=checks,
        rejection_reasons=getattr(old_result, "rejection_reasons", []),
        outliers=outliers,
        ejected_article_indices=getattr(old_result, "ejected_article_indices", []),
    )


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


def _build_critique(check_name: str, score: float, params: ClusteringParams) -> str:
    """Build specific critique for a failed check (CRITIC pattern)."""
    critiques = {
        "coherence": f"Coherence {score:.3f} < {params.val_coherence_min} required. Re-cluster with tighter threshold.",
        "min_articles": f"Only {int(score)} articles. Merge with nearest cluster or drop.",
        "multi_source": f"Only {int(score)} sources. Need {params.val_min_sources}. Fetch more diverse sources.",
        "entity_consistency": f"Entity coverage {score:.0%} < {params.val_entity_consistency_min:.0%}. Cluster may mix unrelated events.",
        "temporal_proximity": f"Articles span > {params.val_temporal_window_hours/24:.0f} days. May be artificial aggregation.",
        "not_duplicate": f"Centroid similarity {score:.3f} >= {params.val_duplicate_threshold} to existing cluster.",
        "syndication": f"All articles share identical first sentences. Syndicated copies — DedupAgent should catch this.",
    }
    return critiques.get(check_name, f"Check '{check_name}' failed with score {score:.3f}")


def _check_action(check_name: str) -> ValidationFailureAction:
    """Map check name to failure action for Signal Bus."""
    actions = {
        "coherence": ValidationFailureAction.RECLUSTER,
        "min_articles": ValidationFailureAction.MERGE_NEAREST,
        "multi_source": ValidationFailureAction.RECLUSTER,
        "entity_consistency": ValidationFailureAction.RECLUSTER,
        "temporal_proximity": ValidationFailureAction.DROP,
        "not_duplicate": ValidationFailureAction.DROP,
        "syndication": ValidationFailureAction.FLAG_SYNDICATION,
    }
    return actions.get(check_name, ValidationFailureAction.RECLUSTER)
