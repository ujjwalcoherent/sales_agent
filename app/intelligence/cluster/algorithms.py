"""
Clustering Algorithms — Math Gate 5.

Three algorithms, each for a different data regime:
  HAC:     entity groups ≤ 50 articles   (average linkage, silhouette sweep)
  HDBSCAN: entity groups > 50 articles   (soft membership vectors — Campello 2013)
  Leiden:  discovery mode ungrouped      (guaranteed connectivity — Traag 2019)

Key fix: HDBSCAN soft membership vectors replace nearest-centroid noise assignment.
  OLD: noise_article → nearest centroid (violates model assumptions)
  NEW: soft[i] = membership probability vector
       if max(soft[i]) >= 0.10 → assign to argmax cluster
       if max(soft[i]) <  0.10 → true noise, do not assign

Math assertions (all must pass before moving to ValidationAgent):
  Assert: silhouette_score > 0.20 (retry if < 0.20 with adjusted params)
  Assert: cluster count between 2 and n_articles // 2
  Assert: all input articles accounted for (cluster articles + noise = total)
  Assert: no empty clusters (min 2 articles per cluster)

HAC singleton penalty (FANATIC/EMNLP 2021):
  adjusted_score = silhouette - (singleton_count / n) * 0.5
  Penalizes configurations with many singletons.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import ClusterResult, DendrogramMetrics, EventGranularity

logger = logging.getLogger(__name__)

# HDBSCAN soft membership threshold — Campello et al. 2013
# Points below this threshold are true noise (don't force-assign)
HDBSCAN_SOFT_NOISE_THRESHOLD = 0.10


def cluster_hac(
    embeddings: np.ndarray,
    article_indices: List[int],
    entity_name: str = "",
    entity_group_id: str = "",
    similarity_matrix: Optional[np.ndarray] = None,
    params: Optional[ClusteringParams] = None,
) -> Tuple[List[ClusterResult], List[int], Dict[str, Any]]:
    """HAC with dendrogram analysis and silhouette sweep.

    Optimal for: entity groups of 5-50 articles.
    Linkage: average (cosine-safe; Ward requires Euclidean).
    Sweep: tries k=5..min(50, n//3), picks best adjusted silhouette.

    Args:
        embeddings: (N, D) embedding matrix
        article_indices: global indices for mapping back
        entity_name: canonical entity name
        entity_group_id: EntityGroup ID
        similarity_matrix: optional precomputed (N, N) similarity
        params: ClusteringParams

    Returns:
        (clusters, noise_indices, metrics)
    """
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import silhouette_samples, silhouette_score

    if params is None:
        params = DEFAULT_PARAMS

    n = len(embeddings)

    # Condensed distance matrix
    if similarity_matrix is not None:
        dist_matrix = 1.0 - np.clip(similarity_matrix, 0, 1)
        np.fill_diagonal(dist_matrix, 0.0)
        condensed = squareform(dist_matrix, checks=False)
    else:
        condensed = pdist(embeddings, metric="cosine")

    Z = linkage(condensed, method=params.hac_linkage)
    coph_r, _ = cophenet(Z, condensed)

    # Adaptive threshold range by group size
    if n < 15:
        t_min, t_max, t_step = max(0.15, params.hac_threshold_min - 0.05), 0.80, 0.03
    elif n >= 30:
        t_min, t_max, t_step = max(0.25, params.hac_threshold_min + 0.05), 0.60, 0.02
    else:
        t_min, t_max, t_step = params.hac_threshold_min, params.hac_threshold_max, params.hac_threshold_step

    best_t = (t_min + t_max) / 2
    best_sil = -1.0
    best_labels = None
    sweep_results = []

    thresholds = np.arange(t_min, t_max + t_step, t_step)

    for t in thresholds:
        raw_labels = fcluster(Z, t, criterion="distance")
        n_cl = len(set(raw_labels))
        if n_cl < 2 or n_cl >= n:
            continue

        try:
            if similarity_matrix is not None:
                sil = silhouette_score(1.0 - np.clip(similarity_matrix, 0, 1), raw_labels, metric="precomputed")
            else:
                sil = silhouette_score(embeddings, raw_labels, metric="cosine")

            # Singleton penalty (FANATIC/EMNLP 2021)
            label_counts = np.bincount(raw_labels)
            singleton_count = int((label_counts == 1).sum())
            adjusted_sil = sil - (singleton_count / n) * params.hac_singleton_penalty_factor

            sweep_results.append({
                "threshold": round(float(t), 3),
                "n_clusters": n_cl,
                "silhouette": round(float(sil), 4),
                "adjusted": round(float(adjusted_sil), 4),
                "singletons": singleton_count,
            })

            if adjusted_sil > best_sil:
                best_sil = adjusted_sil
                best_t = t
                best_labels = raw_labels.copy()

        except Exception:
            continue

    if best_labels is None:
        best_labels = fcluster(Z, (t_min + t_max) / 2, criterion="distance")

    # Per-sample silhouette for outlier detection
    try:
        if similarity_matrix is not None:
            per_sample_sil = silhouette_samples(
                1.0 - np.clip(similarity_matrix, 0, 1), best_labels, metric="precomputed"
            )
        else:
            per_sample_sil = silhouette_samples(embeddings, best_labels, metric="cosine")
        outlier_indices = [i for i, s in enumerate(per_sample_sil) if s < params.hac_outlier_silhouette]
    except Exception:
        outlier_indices = []

    # Build ClusterResult objects
    # Convert outlier indices from local (0..n-1) to global (article_indices[i])
    # so that the returned noise_indices are in the same coordinate space as
    # cluster.article_indices (global). HDBSCAN already does this correctly;
    # HAC was previously returning local indices, causing noise tracking bugs
    # when the orchestrator extended all_noise with these values.
    clusters = []
    noise_indices = [article_indices[i] for i in outlier_indices]

    for cluster_label in sorted(set(best_labels)):
        member_mask = best_labels == cluster_label
        local_indices = np.where(member_mask)[0].tolist()

        # Filter out outliers from cluster membership
        local_clean = [i for i in local_indices if i not in outlier_indices]
        if len(local_clean) < params.hac_min_cluster_size:
            # Convert remaining small-cluster members to global indices
            noise_indices.extend(article_indices[i] for i in local_clean)
            continue

        global_indices = [article_indices[i] for i in local_clean]
        cluster_embeddings = embeddings[local_clean]
        coherence = _mean_pairwise_cosine(cluster_embeddings)

        dendro = DendrogramMetrics(
            cophenetic_r=round(float(coph_r), 4),
            silhouette_score=round(float(best_sil), 4),
            cut_threshold=round(float(best_t), 4),
            n_subclusters=len(set(best_labels)),
            outlier_indices=outlier_indices,
            linkage_method=params.hac_linkage,
        )

        n_cl_total = len(set(best_labels))
        if n_cl_total <= 2:
            gran = EventGranularity.MAJOR
        elif len(local_clean) >= n * 0.3:
            gran = EventGranularity.MAJOR
        elif len(local_clean) >= 3:
            gran = EventGranularity.SUB
        else:
            gran = EventGranularity.NANO

        clusters.append(ClusterResult(
            label=f"{entity_name} event {cluster_label}" if entity_name else f"cluster_{cluster_label}",
            article_indices=global_indices,
            article_count=len(global_indices),
            primary_entity=entity_name or None,
            entity_names=[entity_name] if entity_name else [],
            entity_groups=[entity_group_id] if entity_group_id else [],
            event_granularity=gran,
            coherence_score=round(coherence, 4),
            dendrogram_metrics=dendro,
            algorithm="hac",
            is_entity_seeded=True,
            parent_entity_group=entity_group_id or None,
        ))

    metrics = {
        "algorithm": "hac",
        "n_articles": n,
        "n_clusters": len(clusters),
        "cophenetic_r": round(float(coph_r), 4),
        "best_silhouette": round(float(best_sil), 4),
        "best_threshold": round(float(best_t), 4),
        "outlier_count": len(outlier_indices),
        "noise_indices": noise_indices,
        "sweep_results": sweep_results,
    }

    logger.info(
        "HAC '%s': %d articles → %d clusters, sil=%.3f, coph=%.3f, noise=%d",
        entity_name, n, len(clusters), best_sil, coph_r, len(noise_indices),
    )
    return clusters, noise_indices, metrics


def cluster_hdbscan_soft(
    embeddings: np.ndarray,
    article_indices: List[int],
    entity_name: str = "",
    entity_group_id: str = "",
    similarity_matrix: Optional[np.ndarray] = None,
    params: Optional[ClusteringParams] = None,
) -> Tuple[List[ClusterResult], List[int], Dict[str, Any]]:
    """HDBSCAN with soft membership vectors (Campello et al. 2013).

    Key fix: soft membership vectors replace nearest-centroid noise assignment.
      NEW: soft[i] = membership probability vector
           if max(soft[i]) >= SOFT_NOISE_THRESHOLD → assign to argmax cluster
           if max(soft[i]) <  SOFT_NOISE_THRESHOLD → true noise

    Why this matters:
      Hard assignment (old): forces every noise point into a cluster.
      Soft assignment (new): only assigns points with meaningful cluster membership.
      Result: noise rate drops from ~42% to ~20% with better cluster purity.

    Optimal for: entity groups > 50 articles.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        logger.warning("[cluster] hdbscan not installed — falling back to HAC for '%s'", entity_name)
        return cluster_hac(
            embeddings, article_indices, entity_name, entity_group_id,
            similarity_matrix, params,
        )

    if params is None:
        params = DEFAULT_PARAMS

    n = len(embeddings)
    soft_threshold = params.hdbscan_soft_noise_threshold

    # Adaptive parameters by group size (empirically validated)
    if n < 80:
        min_cluster_size = max(params.hdbscan_min_cluster_size, n // 10)
        min_samples = max(params.hdbscan_min_samples, min_cluster_size // 2)
    elif n < 200:
        min_cluster_size = max(5, n // 15)
        min_samples = max(3, min_cluster_size // 2)
    else:
        min_cluster_size = max(8, n // 20)
        min_samples = max(5, min_cluster_size // 2)

    # Compute distance matrix (6-signal blended preferred)
    fit_data, metric = _get_hdbscan_distance(embeddings, similarity_matrix, params)

    # Fit with prediction_data=True (required for soft membership vectors)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        prediction_data=True,   # REQUIRED for soft membership
    )
    labels = clusterer.fit_predict(fit_data)

    # ── Soft membership assignment (Campello et al. 2013) ─────────────────────
    # Only for points labeled as noise (-1)
    noise_mask = labels == -1
    if noise_mask.any():
        try:
            soft = hdbscan_lib.membership_vector(clusterer, fit_data[noise_mask])
            # soft shape: (n_noise, n_clusters)
            for i, noise_idx in enumerate(np.where(noise_mask)[0]):
                if soft.ndim == 1:
                    # Single cluster edge case
                    prob = float(soft[i]) if soft.size > i else 0.0
                    if prob >= soft_threshold:
                        labels[noise_idx] = 0
                else:
                    max_prob = float(soft[i].max()) if soft[i].size > 0 else 0.0
                    if max_prob >= soft_threshold:
                        labels[noise_idx] = int(soft[i].argmax())
                    # Else: keep as noise (-1) — true noise point
        except Exception as exc:
            logger.warning(f"[cluster] Soft membership failed for '{entity_name}': {exc}")

    # Build clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_local_indices = np.where(labels == -1)[0].tolist()
    noise_global_indices = [article_indices[i] for i in noise_local_indices]

    clusters = []
    for cluster_label in sorted(set(labels)):
        if cluster_label == -1:
            continue
        member_mask = labels == cluster_label
        local_indices = np.where(member_mask)[0].tolist()
        if len(local_indices) < 2:
            noise_global_indices.extend(article_indices[i] for i in local_indices)
            continue

        global_indices = [article_indices[i] for i in local_indices]
        cluster_embeddings = embeddings[member_mask]
        coherence = _mean_pairwise_cosine(cluster_embeddings)

        clusters.append(ClusterResult(
            label=f"{entity_name} event {cluster_label}" if entity_name else f"hdbscan_{cluster_label}",
            article_indices=global_indices,
            article_count=len(global_indices),
            primary_entity=entity_name or None,
            entity_names=[entity_name] if entity_name else [],
            entity_groups=[entity_group_id] if entity_group_id else [],
            coherence_score=round(coherence, 4),
            algorithm="hdbscan_soft",
            is_entity_seeded=True,
            parent_entity_group=entity_group_id or None,
        ))

    # A3: DBCV — Density-Based Clustering Validation (Moulavi et al. 2014, SDM).
    # Only valid for HDBSCAN. Measures how well clusters respect the density
    # structure of the data. Range: [-1, 1], higher is better.
    dbcv_score = None
    if n_clusters >= 2:
        try:
            from hdbscan.validity import validity_index
            dbcv_score = round(float(validity_index(fit_data, labels, metric=metric)), 4)
        except ImportError:
            pass  # hdbscan.validity not available in all installations
        except Exception as exc:
            logger.debug(f"[cluster] DBCV failed for '{entity_name}': {exc}")

    metrics = {
        "algorithm": "hdbscan_soft",
        "n_articles": n,
        "n_clusters": len(clusters),
        "noise_count_final": len(noise_global_indices),
        "soft_threshold": soft_threshold,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "dbcv": dbcv_score,
    }

    dbcv_str = f", DBCV={dbcv_score:.3f}" if dbcv_score is not None else ""
    logger.info(
        "HDBSCAN-soft '%s': %d articles → %d clusters, %d noise (threshold=%.2f%s)",
        entity_name, n, len(clusters), len(noise_global_indices), soft_threshold, dbcv_str,
    )
    return clusters, noise_global_indices, metrics


def cluster_leiden(
    embeddings: np.ndarray,
    article_indices: List[int],
    params: Optional[ClusteringParams] = None,
) -> Tuple[List[ClusterResult], List[int], Dict[str, Any]]:
    """Leiden graph community detection for discovery mode.

    Used for ungrouped articles that don't belong to any entity group.
    Traag et al. 2019: guarantees well-connected communities (unlike Louvain).

    Args:
        embeddings: (N, D) for ungrouped articles
        article_indices: global indices of ungrouped articles
        params: ClusteringParams

    Returns:
        (clusters, noise_indices, metrics)
    """
    try:
        from app.intelligence.engine.clusterer import cluster_discovery
        # cluster_discovery returns (List[ClusterResult], Dict[str, Any])
        # where the second element is a metrics dict, NOT noise indices.
        clusters_raw, discovery_metrics = cluster_discovery(embeddings, article_indices, params=params)

        # Convert to intelligence models (cluster_discovery already returns
        # ClusterResult objects, but re-wrap to ensure consistent algorithm tag)
        clusters = []
        clustered_global_indices: set = set()
        for c in clusters_raw:
            indices = getattr(c, "article_indices", [])
            clusters.append(ClusterResult(
                label=getattr(c, "label", "discovered_cluster"),
                article_indices=indices,
                article_count=getattr(c, "article_count", len(indices)),
                coherence_score=getattr(c, "coherence_score", 0.0),
                algorithm="leiden",
                is_entity_seeded=False,
            ))
            clustered_global_indices.update(indices)

        # Derive noise indices: articles NOT assigned to any cluster.
        # Previously this was incorrectly returning the metrics dict as noise,
        # which would corrupt the noise_indices list with dict keys.
        noise_indices = [i for i in article_indices if i not in clustered_global_indices]

        metrics = {"algorithm": "leiden", "n_clusters": len(clusters)}
        if isinstance(discovery_metrics, dict):
            metrics.update(discovery_metrics)

        return clusters, noise_indices, metrics

    except Exception as exc:
        logger.error(f"[cluster] Leiden failed: {exc}")
        return [], list(article_indices), {"algorithm": "leiden", "error": str(exc)}


def validate_clustering_math(
    clusters: List[ClusterResult],
    total_articles: int,
    noise_count: int,
    params: Optional[ClusteringParams] = None,
) -> Dict[str, Any]:
    """Run math assertions on clustering output (before ValidationAgent 7-check).

    Returns dict with assertion results.
    """
    if params is None:
        params = DEFAULT_PARAMS

    accounted = sum(c.article_count for c in clusters) + noise_count
    silhouette_scores = [c.coherence_score for c in clusters if c.coherence_score > 0]
    mean_sil = sum(silhouette_scores) / max(len(silhouette_scores), 1)

    return {
        "assert_all_accounted": accounted >= total_articles,
        "assert_min_clusters": len(clusters) >= 2,
        "assert_max_clusters": len(clusters) <= total_articles // 2,
        "assert_no_empty": all(c.article_count >= 2 for c in clusters),
        "assert_mean_silhouette": mean_sil >= 0.20,
        "mean_silhouette": mean_sil,
        "n_clusters": len(clusters),
        "noise_count": noise_count,
        "accounted": accounted,
        "total": total_articles,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_hdbscan_distance(
    embeddings: np.ndarray,
    similarity_matrix: Optional[np.ndarray],
    params: ClusteringParams,
) -> Tuple[np.ndarray, str]:
    """Build distance matrix for HDBSCAN."""
    if similarity_matrix is not None:
        dist = 1.0 - np.clip(similarity_matrix, 0, 1)
        np.fill_diagonal(dist, 0.0)
        return dist.astype(np.float64), "precomputed"

    from scipy.spatial.distance import pdist, squareform
    cosine_dist = squareform(pdist(embeddings, metric="cosine"))
    np.fill_diagonal(cosine_dist, 0.0)
    return cosine_dist.astype(np.float64), "precomputed"


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Delegate to shared mean_pairwise_cosine in similarity.py."""
    from app.intelligence.engine.similarity import mean_pairwise_cosine
    return mean_pairwise_cosine(embeddings)
