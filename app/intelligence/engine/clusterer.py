"""
Clustering algorithms for entity-seeded and discovery modes.

Entity-seeded (within entity groups):
  - Small (<5 articles): Single cluster
  - Medium (5-50): HAC with dendrogram analysis + silhouette sweep
  - Large (50+): HDBSCAN (auto-detects density and cluster count)

Discovery (ungrouped articles):
  - Leiden community detection via existing cluster_leiden()

Research-backed parameters:
  - parkervg: entity-weighted TF-IDF + Ward HAC → F1=0.922
  - NewsCatcher: Leiden + cosine threshold=0.6 → production-proven
  - Cosine distance sweet spot for news: 0.30-0.45

All thresholds come from ClusteringParams (config.py).

Standalone test:
    python -c "
    import numpy as np
    from app.intelligence.engine.clusterer import cluster_entity_group, cluster_discovery
    embeddings = np.random.randn(20, 1536)
    indices = list(range(20))
    clusters, metrics = cluster_entity_group(embeddings, indices, entity_name='TestCo')
    print(f'Clusters: {len(clusters)}, Metrics: {list(metrics.keys())}')
    "
"""

import logging
import time
from collections import Counter as _Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import (
    ClusterResult, DendrogramMetrics, EventGranularity,
)

logger = logging.getLogger(__name__)


def cluster_entity_group(
    embeddings: np.ndarray,
    article_indices: List[int],
    entity_name: str = "",
    entity_group_id: str = "",
    similarity_matrix: Optional[np.ndarray] = None,
    articles: Optional[List[Any]] = None,
    params: Optional[ClusteringParams] = None,
) -> Tuple[List[ClusterResult], Dict[str, Any]]:
    """Cluster articles within an entity group.

    Routes to appropriate algorithm based on group size:
      - < min_articles: single cluster
      - min_articles to max_articles: HAC with dendrogram
      - > max_articles: HDBSCAN

    Args:
        embeddings: (N, D) embedding matrix for the entity group's articles.
        article_indices: Global article indices (for mapping back).
        entity_name: Canonical entity name for labeling.
        entity_group_id: EntityGroup ID for provenance.
        similarity_matrix: Optional precomputed (N, N) similarity matrix.
        articles: Optional full article list (for provenance).
        params: Clustering parameters.

    Returns:
        (clusters, metrics) where clusters are ClusterResult instances
        and metrics contains algorithm-specific diagnostics.
    """
    if params is None:
        params = DEFAULT_PARAMS

    n = len(embeddings)

    if n < params.hac_min_articles:
        # Too small to cluster → single cluster
        cluster = _build_single_cluster(
            embeddings, article_indices, entity_name, entity_group_id, articles,
        )
        return [cluster], {"algorithm": "single", "n_articles": n}

    if n <= params.hac_max_articles:
        clusters, metrics = _cluster_hac(
            embeddings, article_indices, entity_name, entity_group_id,
            similarity_matrix, articles, params,
        )
    else:
        clusters, metrics = _cluster_hdbscan(
            embeddings, article_indices, entity_name, entity_group_id,
            articles, params,
        )

    # Post-processing 1: merge small clusters into nearest larger neighbor
    # (FANATIC/EMNLP 2021, dynamicTreeCut/Bioinformatics 2008)
    if len(clusters) > 1:
        clusters, merge_count = _merge_small_clusters(
            clusters, embeddings, params.hac_min_cluster_size,
        )
        if merge_count > 0:
            metrics["small_clusters_merged"] = merge_count

    # Post-processing 2: enforce source diversity (merge single-source clusters)
    if params.enforce_source_diversity and articles is not None and len(clusters) > 1:
        clusters = enforce_source_diversity(
            clusters, articles, embeddings,
            min_sources=params.min_sources_per_cluster,
            min_merge_similarity=params.source_merge_min_similarity,
        )
        metrics["source_diversity_enforced"] = True

    return clusters, metrics


def _build_single_cluster(
    embeddings: np.ndarray,
    article_indices: List[int],
    entity_name: str,
    entity_group_id: str,
    articles: Optional[List[Any]],
) -> ClusterResult:
    """Create a single cluster from all articles in the group."""
    from app.intelligence.engine.similarity import _compute_semantic

    coherence = 0.0
    if len(embeddings) > 1:
        sim = _compute_semantic(embeddings)
        # Mean of upper triangle
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        coherence = float(sim[mask].mean()) if mask.any() else 0.0

    return ClusterResult(
        label=f"{entity_name} (all)" if entity_name else "single_cluster",
        article_indices=list(article_indices),
        article_count=len(article_indices),
        primary_entity=entity_name,
        entity_names=[entity_name] if entity_name else [],
        entity_groups=[entity_group_id] if entity_group_id else [],
        coherence_score=round(coherence, 4),
        algorithm="single",
        is_entity_seeded=True,
        parent_entity_group=entity_group_id,
    )


def _cluster_hac(
    embeddings: np.ndarray,
    article_indices: List[int],
    entity_name: str,
    entity_group_id: str,
    similarity_matrix: Optional[np.ndarray],
    articles: Optional[List[Any]],
    params: ClusteringParams,
) -> Tuple[List[ClusterResult], Dict[str, Any]]:
    """HAC with dendrogram analysis and silhouette sweep.

    Uses average linkage (cosine-compatible). Silhouette sweep finds
    optimal cut threshold. Cophenetic correlation validates dendrogram fidelity.
    Per-sample silhouette detects outliers.
    """
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import silhouette_samples, silhouette_score

    n = len(embeddings)

    # Compute condensed distance matrix
    if similarity_matrix is not None:
        dist_matrix = 1.0 - np.clip(similarity_matrix, 0, 1)
        np.fill_diagonal(dist_matrix, 0.0)
        condensed = squareform(dist_matrix, checks=False)
    else:
        condensed = pdist(embeddings, metric="cosine")

    # Linkage (average for cosine compatibility)
    Z = linkage(condensed, method=params.hac_linkage)

    # Cophenetic correlation (dendrogram fidelity)
    coph_r, _ = cophenet(Z, condensed)

    # Adaptive threshold range based on group size:
    # Small groups (5-15): wider range + finer step (higher variance, avoid fragmentation)
    # Medium groups (15-30): default range
    # Large groups (30-50): tighter range + finer step (enough data for precise cuts)
    if n < 15:
        t_min = max(0.15, params.hac_threshold_min - 0.05)
        t_max = min(0.80, params.hac_threshold_max + 0.10)
        t_step = 0.03
    elif n >= 30:
        t_min = max(0.25, params.hac_threshold_min + 0.05)
        t_max = min(0.60, params.hac_threshold_max - 0.10)
        t_step = 0.02
    else:
        t_min = params.hac_threshold_min
        t_max = params.hac_threshold_max
        t_step = params.hac_threshold_step

    # Silhouette sweep to find optimal threshold.
    # Key insight: raw silhouette rewards singletons (well-separated by definition)
    # but singletons are useless (fail min_articles validation). We penalize them.
    best_t = (t_min + t_max) / 2
    best_sil = -1.0
    sweep_results = []

    thresholds = np.arange(t_min, t_max + t_step, t_step)

    for t in thresholds:
        labels = fcluster(Z, t, criterion="distance")
        n_clusters = len(set(labels))
        if n_clusters < 2 or n_clusters >= n:
            continue
        try:
            if similarity_matrix is not None:
                sil = silhouette_score(1.0 - similarity_matrix, labels, metric="precomputed")
            else:
                sil = silhouette_score(embeddings, labels, metric="cosine")

            # Penalize singletons: fraction of 1-article clusters reduces score.
            # A threshold creating [3, 1, 1, 1, 1] (4/7 singletons) gets penalty 0.29.
            # A threshold creating [4, 3] (0 singletons) gets no penalty.
            # This steers HAC toward fewer, larger clusters.
            cluster_sizes = [int(np.sum(labels == lab)) for lab in set(labels)]
            singleton_count = sum(1 for s in cluster_sizes if s == 1)
            singleton_penalty = (singleton_count / n) * 0.5
            adjusted_sil = sil - singleton_penalty

            sweep_results.append({
                "threshold": round(t, 3),
                "n_clusters": n_clusters,
                "silhouette": round(sil, 4),
                "adjusted": round(adjusted_sil, 4),
                "singletons": singleton_count,
            })
            if adjusted_sil > best_sil:
                best_sil = adjusted_sil
                best_t = t
        except Exception:
            continue

    # Final cut with best threshold
    labels = fcluster(Z, best_t, criterion="distance")
    n_clusters = len(set(labels))

    # Per-sample silhouette for outlier detection
    outlier_indices = []
    sample_sils = np.zeros(n)
    if n_clusters >= 2:
        try:
            if similarity_matrix is not None:
                sample_sils = silhouette_samples(1.0 - similarity_matrix, labels, metric="precomputed")
            else:
                sample_sils = silhouette_samples(embeddings, labels, metric="cosine")
            outlier_indices = np.where(sample_sils < params.hac_outlier_silhouette)[0].tolist()
        except Exception:
            pass

    # Build dendrogram metrics
    dendro = DendrogramMetrics(
        cophenetic_r=round(float(coph_r), 4),
        cut_threshold=round(float(best_t), 4),
        n_subclusters=n_clusters,
        outlier_indices=outlier_indices,
        linkage_method=params.hac_linkage,
    )

    # Build ClusterResult for each sub-cluster
    clusters = []
    for cluster_label in sorted(set(labels)):
        member_mask = labels == cluster_label
        local_indices = np.where(member_mask)[0].tolist()
        global_indices = [article_indices[i] for i in local_indices]

        # Compute cluster coherence
        cluster_embeddings = embeddings[member_mask]
        coherence = _mean_pairwise_cosine(cluster_embeddings)

        # Determine event granularity based on cluster size relative to parent
        if n_clusters <= 2:
            granularity = EventGranularity.MAJOR
        elif len(local_indices) >= n * 0.3:
            granularity = EventGranularity.MAJOR
        elif len(local_indices) >= 3:
            granularity = EventGranularity.SUB
        else:
            granularity = EventGranularity.NANO

        cluster = ClusterResult(
            label=f"{entity_name} event {cluster_label}" if entity_name else f"cluster_{cluster_label}",
            article_indices=global_indices,
            article_count=len(global_indices),
            primary_entity=entity_name,
            entity_names=[entity_name] if entity_name else [],
            entity_groups=[entity_group_id] if entity_group_id else [],
            event_granularity=granularity,
            coherence_score=round(coherence, 4),
            dendrogram_metrics=dendro,
            algorithm="hac",
            is_entity_seeded=True,
            parent_entity_group=entity_group_id,
        )
        clusters.append(cluster)

    metrics = {
        "algorithm": "hac",
        "n_articles": n,
        "n_clusters": n_clusters,
        "cophenetic_r": round(float(coph_r), 4),
        "best_silhouette": round(float(best_sil), 4),
        "best_threshold": round(float(best_t), 4),
        "outlier_count": len(outlier_indices),
        "sweep_results": sweep_results,
        "linkage_matrix_shape": list(Z.shape),
    }

    logger.info(
        "HAC clustering for '%s': %d articles → %d clusters, sil=%.3f, coph=%.3f, outliers=%d",
        entity_name, n, n_clusters, best_sil, coph_r, len(outlier_indices),
    )
    return clusters, metrics


def _cluster_hdbscan(
    embeddings: np.ndarray,
    article_indices: List[int],
    entity_name: str,
    entity_group_id: str,
    articles: Optional[List[Any]],
    params: ClusteringParams,
) -> Tuple[List[ClusterResult], Dict[str, Any]]:
    """HDBSCAN for large entity groups (50+ articles).

    Improvements over naive HDBSCAN:
      - Adaptive min_cluster_size/min_samples based on group size
        (prevents mega-clusters from too-small defaults)
      - Uses blended 6-signal similarity matrix (not raw embeddings)
        so entity overlap, source penalty, temporal are all considered
    """
    try:
        import hdbscan
    except ImportError:
        logger.warning("hdbscan not installed, falling back to HAC for large group '%s'", entity_name)
        return _cluster_hac(
            embeddings, article_indices, entity_name, entity_group_id,
            None, articles, params,
        )

    n = len(embeddings)

    # Adaptive parameters: scale by group size to prevent mega-clusters
    # Small-ish groups (50-80): ~10% → min_cluster_size=5-8
    # Medium (80-200): ~7% → min_cluster_size=6-14
    # Large (200+): ~5% → min_cluster_size=10+
    if n < 80:
        min_cluster_size = max(params.hdbscan_min_cluster_size, n // 10)
        min_samples = max(params.hdbscan_min_samples, min_cluster_size // 2)
    elif n < 200:
        min_cluster_size = max(5, n // 15)
        min_samples = max(3, min_cluster_size // 2)
    else:
        min_cluster_size = max(8, n // 20)
        min_samples = max(5, min_cluster_size // 2)

    # Use blended similarity (all 6 signals) when available
    if params.hdbscan_use_blended_similarity and articles is not None:
        try:
            from app.intelligence.engine.similarity import (
                compute_decomposed_similarity,
                similarity_to_distance,
            )
            # Compute the article subset for this entity group
            group_articles = [articles[i] for i in article_indices] if articles else None
            sim_result = compute_decomposed_similarity(
                embeddings=embeddings,
                articles=group_articles,
                params=params,
            )
            dist_matrix = similarity_to_distance(sim_result["blended"])
            metric = "precomputed"
            fit_data = dist_matrix.astype(np.float64)  # HDBSCAN requires float64
            logger.debug(
                "HDBSCAN '%s': using blended 6-signal similarity (n=%d, mcs=%d, ms=%d)",
                entity_name, n, min_cluster_size, min_samples,
            )
        except Exception as e:
            logger.warning("HDBSCAN blended similarity failed for '%s': %s, falling back to cosine", entity_name, e)
            # HDBSCAN's BallTree/KDTree don't support cosine — precompute distance
            from scipy.spatial.distance import pdist, squareform
            cosine_dist = squareform(pdist(embeddings, metric="cosine"))
            np.fill_diagonal(cosine_dist, 0.0)
            fit_data = cosine_dist.astype(np.float64)
            metric = "precomputed"
    else:
        # HDBSCAN's BallTree/KDTree don't support cosine — precompute distance
        from scipy.spatial.distance import pdist, squareform
        cosine_dist = squareform(pdist(embeddings, metric="cosine"))
        np.fill_diagonal(cosine_dist, 0.0)
        fit_data = cosine_dist.astype(np.float64)
        metric = "precomputed"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(fit_data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())

    clusters = []
    for cluster_label in sorted(set(labels)):
        if cluster_label == -1:
            continue  # Skip noise
        member_mask = labels == cluster_label
        local_indices = np.where(member_mask)[0].tolist()
        global_indices = [article_indices[i] for i in local_indices]

        coherence = _mean_pairwise_cosine(embeddings[member_mask])

        cluster = ClusterResult(
            label=f"{entity_name} event {cluster_label}" if entity_name else f"hdbscan_{cluster_label}",
            article_indices=global_indices,
            article_count=len(global_indices),
            primary_entity=entity_name,
            entity_names=[entity_name] if entity_name else [],
            entity_groups=[entity_group_id] if entity_group_id else [],
            coherence_score=round(coherence, 4),
            algorithm="hdbscan",
            is_entity_seeded=True,
            parent_entity_group=entity_group_id,
        )
        clusters.append(cluster)

    metrics = {
        "algorithm": "hdbscan",
        "n_articles": n,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
    }

    logger.info(
        "HDBSCAN for '%s': %d articles → %d clusters, %d noise (mcs=%d, ms=%d, metric=%s)",
        entity_name, n, n_clusters, noise_count, min_cluster_size, min_samples, metric,
    )
    return clusters, metrics


def cluster_discovery(
    embeddings: np.ndarray,
    article_indices: List[int],
    articles: Optional[List[Any]] = None,
    params: Optional[ClusteringParams] = None,
    optimize: bool = False,
) -> Tuple[List[ClusterResult], Dict[str, Any]]:
    """Discovery clustering for ungrouped articles using Leiden.

    Uses fixed resolution=1.0 (params.leiden_resolution) — Optuna tuning removed
    because it ran 15 trials at ~2s each (~30s total), defeating the 5-10 min pipeline target.
    """
    if params is None:
        params = DEFAULT_PARAMS

    n = len(embeddings)
    if n < 4:
        logger.info("Discovery: only %d ungrouped articles, skipping", n)
        return [], {"algorithm": "leiden", "n_articles": n, "skipped": True}

    try:
        # Use local implementations (moved from app.trends.clustering in Phase 11)
        from app.intelligence.engine.clusterer import cluster_leiden
    except ImportError:
        logger.warning("cluster_leiden not available, skipping discovery")
        return [], {"algorithm": "leiden", "error": "import_failed"}

    try:
        labels, noise_count, leiden_metrics = cluster_leiden(
            embeddings,
            k=params.leiden_k,
            resolution=params.leiden_resolution,
        )
    except Exception as e:
        logger.warning("Leiden clustering failed: %s", e)
        return [], {"algorithm": "leiden", "error": str(e)}

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = []

    for cluster_label in sorted(set(labels)):
        if cluster_label == -1:
            continue
        member_mask = labels == cluster_label
        local_indices = np.where(member_mask)[0].tolist()
        global_indices = [article_indices[i] for i in local_indices]

        coherence = _mean_pairwise_cosine(embeddings[member_mask])

        cluster = ClusterResult(
            label=f"discovery_{cluster_label}",
            article_indices=global_indices,
            article_count=len(global_indices),
            coherence_score=round(coherence, 4),
            algorithm="leiden",
            is_entity_seeded=False,
        )
        clusters.append(cluster)

    metrics = {
        "algorithm": "leiden",
        "n_articles": n,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "leiden_metrics": leiden_metrics,
    }

    logger.info(
        "Discovery clustering: %d ungrouped articles → %d clusters, %d noise",
        n, n_clusters, noise_count,
    )
    return clusters, metrics


def enforce_source_diversity(
    clusters: List[ClusterResult],
    articles: List[Any],
    embeddings: np.ndarray,
    min_sources: int = 2,
    min_merge_similarity: float = 0.4,
) -> List[ClusterResult]:
    """Merge single-source clusters into nearest multi-source cluster.

    Post-processing step to prevent echo-chamber clusters. A cluster where all
    articles come from the same domain has low credibility — merge it into the
    nearest cluster that already has source diversity.

    Only merges if centroid cosine similarity exceeds min_merge_similarity.
    """
    if len(clusters) < 2 or not articles:
        return clusters

    # Classify clusters by source diversity
    single_source: List[ClusterResult] = []
    multi_source: List[ClusterResult] = []

    for c in clusters:
        sources = set()
        for idx in c.article_indices:
            if 0 <= idx < len(articles):
                src = (
                    getattr(articles[idx], "source_id", "")
                    or getattr(articles[idx], "source_name", "")
                )
                if src:
                    sources.add(src.lower())
        if len(sources) < min_sources:
            single_source.append(c)
        else:
            multi_source.append(c)

    if not single_source or not multi_source:
        return clusters  # Nothing to merge, or nothing to merge into

    merged_count = 0
    remaining_single = []

    for sc in single_source:
        # Compute centroid for the single-source cluster
        sc_idx = [i for i in sc.article_indices if 0 <= i < len(embeddings)]
        if not sc_idx:
            remaining_single.append(sc)
            continue
        sc_centroid = embeddings[sc_idx].mean(axis=0)
        sc_norm = np.linalg.norm(sc_centroid)
        if sc_norm < 1e-8:
            remaining_single.append(sc)
            continue

        # Find best merge target
        best_sim = -1.0
        best_target: Optional[ClusterResult] = None

        for mc in multi_source:
            mc_idx = [i for i in mc.article_indices if 0 <= i < len(embeddings)]
            if not mc_idx:
                continue
            mc_centroid = embeddings[mc_idx].mean(axis=0)
            mc_norm = np.linalg.norm(mc_centroid)
            if mc_norm < 1e-8:
                continue
            sim = float(np.dot(sc_centroid, mc_centroid) / (sc_norm * mc_norm))
            if sim > best_sim:
                best_sim = sim
                best_target = mc

        if best_target is not None and best_sim >= min_merge_similarity:
            # Merge: add single-source articles to the multi-source cluster
            best_target.article_indices.extend(sc.article_indices)
            best_target.article_count = len(best_target.article_indices)
            merged_count += 1
        else:
            remaining_single.append(sc)

    if merged_count > 0:
        logger.info(
            "Source diversity enforcement: merged %d single-source clusters into multi-source",
            merged_count,
        )

    return multi_source + remaining_single


def _merge_small_clusters(
    clusters: List[ClusterResult],
    embeddings: np.ndarray,
    min_size: int = 3,
) -> Tuple[List[ClusterResult], int]:
    """Merge clusters below min_size into their nearest larger neighbor.

    Research basis: FANATIC (Bloomberg, EMNLP 2021) uses DP-means which
    inherently prevents small fragments. dynamicTreeCut (Langfelder 2008)
    uses minClusterSize to aggressively merge small HAC branches.
    PeerJ Finance HAC paper (2021) confirms post-hoc small cluster merging
    improves both silhouette and Dunn Index by ~50%.

    Prevents the "66% size-2 clusters" problem caused by silhouette
    rewarding compact fragments over larger, coherent story clusters.
    """
    if not clusters or min_size < 2:
        return clusters, 0

    large = [c for c in clusters if c.article_count >= min_size]
    small = [c for c in clusters if c.article_count < min_size]

    if not small or not large:
        return clusters, 0

    # Compute centroids for large clusters
    large_centroids = []
    for c in large:
        valid_idx = [i for i in c.article_indices if 0 <= i < len(embeddings)]
        if valid_idx:
            centroid = embeddings[valid_idx].mean(axis=0)
            norm = np.linalg.norm(centroid)
            large_centroids.append(centroid / norm if norm > 1e-8 else centroid)
        else:
            large_centroids.append(np.zeros(embeddings.shape[1]))

    merge_count = 0
    for sc in small:
        sc_idx = [i for i in sc.article_indices if 0 <= i < len(embeddings)]
        if not sc_idx:
            large.append(sc)  # Can't compute centroid, keep as-is
            continue

        sc_centroid = embeddings[sc_idx].mean(axis=0)
        sc_norm = np.linalg.norm(sc_centroid)
        if sc_norm < 1e-8:
            large.append(sc)
            continue
        sc_centroid = sc_centroid / sc_norm

        # Find nearest large cluster by centroid cosine similarity
        best_sim = -1.0
        best_idx = -1
        for i, lc_centroid in enumerate(large_centroids):
            sim = float(np.dot(sc_centroid, lc_centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= 0.2:  # Very lenient threshold — merging is the goal
            # Merge into nearest large cluster
            target = large[best_idx]
            target.article_indices.extend(sc.article_indices)
            target.article_count = len(target.article_indices)
            # Recompute centroid for the enlarged cluster
            valid_idx = [i for i in target.article_indices if 0 <= i < len(embeddings)]
            if valid_idx:
                new_centroid = embeddings[valid_idx].mean(axis=0)
                new_norm = np.linalg.norm(new_centroid)
                large_centroids[best_idx] = new_centroid / new_norm if new_norm > 1e-8 else new_centroid
            merge_count += 1
        else:
            large.append(sc)  # No good target, keep as-is

    if merge_count > 0:
        logger.info(
            "Merged %d small clusters (<%d articles) into nearest neighbors",
            merge_count, min_size,
        )

    return large, merge_count


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity for a set of embeddings."""
    if len(embeddings) < 2:
        return 1.0
    try:
        from app.intelligence.engine.similarity import _compute_semantic
        sim = _compute_semantic(embeddings)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        return float(sim[mask].mean()) if mask.any() else 1.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LEIDEN COMMUNITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
# Moved from app/trends/clustering.py — canonical home for all clustering algos.


def _ensure_leiden_deps():
    """Import leidenalg + igraph, raising clear error if missing."""
    try:
        import igraph as ig
        import leidenalg
        return ig, leidenalg
    except ImportError as e:
        raise ImportError(
            "Leiden clustering requires: pip install leidenalg python-igraph\n"
            f"Original error: {e}"
        )


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 20,
    mutual: bool = True,
):
    """Build a k-NN graph from embeddings with cosine similarity edge weights."""
    ig, _ = _ensure_leiden_deps()
    from sklearn.neighbors import NearestNeighbors

    n_samples = embeddings.shape[0]
    effective_k = min(k, n_samples - 1)

    nn = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    edges = []
    weights = []
    seen = set()

    for i in range(n_samples):
        for j_idx in range(1, effective_k + 1):
            j = int(indices[i, j_idx])
            cos_dist = float(distances[i, j_idx])
            cos_sim = 1.0 - cos_dist

            if cos_sim <= 0:
                continue

            edge_key = (min(i, j), max(i, j))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            if mutual and i not in indices[j, 1:]:
                continue

            edges.append(edge_key)
            weights.append(cos_sim)

    g = ig.Graph(n=n_samples, edges=edges, directed=False)
    g.es["weight"] = weights
    return g


def build_knn_graph_from_similarity(
    sim_matrix: np.ndarray,
    k: int = 20,
    mutual: bool = True,
):
    """Build a k-NN graph from a precomputed similarity matrix."""
    ig, _ = _ensure_leiden_deps()
    n = sim_matrix.shape[0]
    effective_k = min(k, n - 1)

    edges = []
    weights = []
    seen = set()

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0
        top_k = np.argsort(sims)[-effective_k:][::-1]

        for j in top_k:
            j = int(j)
            sim = float(sim_matrix[i, j])
            if sim <= 0:
                continue
            edge_key = (min(i, j), max(i, j))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            if mutual:
                j_sims = sim_matrix[j].copy()
                j_sims[j] = -1.0
                j_top_k = set(np.argsort(j_sims)[-effective_k:])
                if i not in j_top_k:
                    continue

            edges.append(edge_key)
            weights.append(sim)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    return g


def cluster_leiden(
    embeddings: np.ndarray,
    k: int = 20,
    resolution: float = 1.0,
    seed: int = 42,
    mutual_knn: bool = True,
    min_community_size: int = 4,
    quality_function: str = "RBConfiguration",
    precomputed_sim: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Cluster embeddings using k-NN graph + Leiden community detection.

    Returns (labels, noise_count, metrics) where labels has -1 for noise.
    """
    _, leidenalg = _ensure_leiden_deps()

    n_samples = embeddings.shape[0]
    t_start = time.time()

    if n_samples < 3:
        return np.full(n_samples, -1, dtype=int), n_samples, {
            "n_clusters": 0, "noise_count": n_samples,
        }

    t_graph = time.time()
    if precomputed_sim is not None:
        graph = build_knn_graph_from_similarity(precomputed_sim, k=k, mutual=mutual_knn)
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normed = embeddings / norms
        graph = build_knn_graph(emb_normed, k=k, mutual=mutual_knn)
    graph_time = time.time() - t_graph

    t_leiden = time.time()

    partition_types = {
        "RBConfiguration": leidenalg.RBConfigurationVertexPartition,
        "Modularity": leidenalg.ModularityVertexPartition,
        "CPM": leidenalg.CPMVertexPartition,
        "Significance": leidenalg.SignificanceVertexPartition,
    }
    partition_type = partition_types.get(
        quality_function, leidenalg.RBConfigurationVertexPartition
    )

    leiden_kwargs = {"weights": "weight", "seed": seed, "n_iterations": -1}
    if quality_function in ("RBConfiguration", "CPM"):
        leiden_kwargs["resolution_parameter"] = resolution

    partition = leidenalg.find_partition(graph, partition_type, **leiden_kwargs)
    leiden_time = time.time() - t_leiden

    raw_labels = np.array(partition.membership)
    labels = raw_labels.copy()

    community_sizes = _Counter(raw_labels)
    noise_communities = {
        c for c, size in community_sizes.items() if size < min_community_size
    }
    for c in noise_communities:
        labels[raw_labels == c] = -1

    unique_labels = sorted(set(labels) - {-1})
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = -1
    labels = np.array([label_map[l] for l in labels])

    noise_count = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_sizes = sorted(
        [s for c, s in community_sizes.items() if c not in noise_communities],
        reverse=True,
    )
    metrics = {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_pct": noise_count / max(n_samples, 1),
        "modularity": float(partition.modularity),
        "quality": float(partition.quality()),
        "graph_edges": graph.ecount(),
        "graph_time_s": round(time.time() - t_start - leiden_time, 3),
        "leiden_time_s": round(leiden_time, 3),
        "resolution": resolution,
        "k": k,
        "cluster_sizes": cluster_sizes,
    }

    logger.info(
        f"Leiden: {n_samples} articles -> {n_clusters} clusters, "
        f"{noise_count} noise ({noise_count * 100 // max(n_samples, 1)}%), "
        f"modularity={partition.modularity:.3f}, "
        f"sizes={cluster_sizes[:5]}{'...' if len(cluster_sizes) > 5 else ''}, "
        f"graph={graph_time:.2f}s, leiden={leiden_time:.3f}s"
    )

    return labels, noise_count, metrics


