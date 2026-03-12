"""
Cluster Orchestrator — native implementation using intelligence/cluster/ algorithms.

Pipeline per entity group:
  HAC   → groups with 5-50 articles (average linkage, silhouette sweep)
  HDBSCAN → groups with 50+ articles (soft membership vectors)
  Leiden  → ungrouped / discovery mode articles

Embedding reuse: accepts precomputed embeddings from source_intel phase
(stored in deps._embeddings) to avoid redundant API calls. Falls back to
batch embedding via EmbeddingTool if not provided.

Similarity matrix caching: computes per-group cosine similarity matrix once
and passes it to HAC/HDBSCAN, avoiding redundant pairwise distance computations.

ValidationAgent can signal ClusterAgent to re-cluster (max 2 retries).
This is the bidirectional loop from the Blackboard pattern (Erman et al. 1980).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from app.intelligence.config import ClusteringParams
from app.intelligence.models import (
    AgentRequestType,
    ClusterResult,
    DiscoveryScope,
    PipelineState,
    ValidationResult,
)

logger = logging.getLogger(__name__)

_MAX_RECLUSTER_RETRIES = 2
_HDBSCAN_LARGE_GROUP_THRESHOLD = 50  # switch from HAC to HDBSCAN above this


async def cluster_and_validate(
    articles: list,
    entity_groups: list,
    scope: DiscoveryScope,
    params: ClusteringParams,
    state: PipelineState,
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[ClusterResult], List[str], List[str], List[ValidationResult], List[int]]:
    """Run similarity → cluster → validate loop.

    Args:
        articles: List of Article objects to cluster.
        entity_groups: Entity groups from NER/extraction step.
        scope: DiscoveryScope for the run.
        params: ClusteringParams (adaptive or default).
        state: PipelineState for thought logging and inter-agent communication.
        precomputed_embeddings: Optional (N, D) numpy array of embeddings from
            the source_intel phase (deps._embeddings). Avoids redundant embedding
            API calls when the source_intel agent already computed them.

    Returns:
        (clusters, passed_ids, rejected_ids, validation_results, noise_indices)

    If validation fails, signals ClusterAgent to re-cluster (max 2 retries).
    This implements the CRITIC pattern: math verifies output → critique → retry.
    """
    for attempt in range(_MAX_RECLUSTER_RETRIES + 1):
        state.add_thought(
            "cluster_orchestrator",
            f"Clustering attempt {attempt + 1}/{_MAX_RECLUSTER_RETRIES + 1}",
        )

        clusters, noise = await _run_clustering(
            articles, entity_groups, scope, params,
            precomputed_embeddings=precomputed_embeddings,
        )
        passed, rejected, val_results = await _run_validation(
            clusters, articles, entity_groups, params
        )

        # Check if re-cluster was requested by ValidationAgent
        recluster_requests = [
            r for r in state.pending_requests("cluster")
            if r.request_type == AgentRequestType.RETRY_CLUSTER and not r.resolved
        ]

        if not recluster_requests or attempt >= _MAX_RECLUSTER_RETRIES:
            for r in recluster_requests:
                r.resolved = True
            state.add_thought(
                "cluster_orchestrator",
                f"Clustering complete: {len(passed)} passed, {len(rejected)} rejected",
                observation=f"attempt={attempt+1}, no more retries needed",
            )
            return clusters, passed, rejected, val_results, noise

        # Process recluster requests — tighten parameters
        critiques = [r.details.get("critique", "") for r in recluster_requests]
        logger.info(f"[cluster] Re-clustering (attempt {attempt+1}): {critiques}")
        for r in recluster_requests:
            r.resolved = True

        # Tighten coherence requirement each retry
        params.val_coherence_min = min(params.val_coherence_min + 0.05, 0.70)

    # Should not reach here
    return clusters, passed, rejected, val_results, noise


async def _run_clustering(
    articles: list,
    entity_groups: list,
    scope: DiscoveryScope,
    params: ClusteringParams,
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[ClusterResult], List[int]]:
    """Run native HAC/HDBSCAN/Leiden clustering on entity groups.

    Args:
        precomputed_embeddings: Optional (N, D) numpy array from source_intel.
            When provided, skips the embedding computation step entirely.
            This saves one full batch embedding API call per pipeline run.
    """
    from app.intelligence.cluster.algorithms import (
        cluster_hac,
        cluster_hdbscan_soft as cluster_hdbscan,
        cluster_leiden,
    )

    if not articles:
        return [], []

    # Step 1: Reuse precomputed embeddings if available, else compute fresh.
    # The source_intel phase stores embeddings in deps._embeddings; the caller
    # can pass them here via precomputed_embeddings to avoid a redundant
    # batch embedding API call (saves ~2-5s and one API round-trip).
    if precomputed_embeddings is not None and len(precomputed_embeddings) == len(articles):
        embeddings = np.asarray(precomputed_embeddings, dtype=np.float32)
        logger.info("[cluster] Reusing %d precomputed embeddings from source_intel", len(embeddings))
    else:
        if precomputed_embeddings is not None:
            logger.warning(
                "[cluster] Precomputed embeddings size mismatch (%d vs %d articles), recomputing",
                len(precomputed_embeddings), len(articles),
            )
        embeddings = await _get_embeddings(articles)
    n = len(articles)

    all_clusters: List[ClusterResult] = []
    all_noise: List[int] = []
    grouped_article_indices: set = set()

    # Step 2: cluster each entity group independently
    for group in entity_groups:
        # Support both intelligence.models.EntityGroup and old clustering EntityGroup
        group_indices = _get_article_indices(group)
        if len(group_indices) < 2:
            all_noise.extend(group_indices)
            continue

        group_name = getattr(group, "canonical_name", getattr(group, "name", "unknown"))
        group_id = getattr(group, "id", group_name)

        # Extract per-group embeddings
        valid_indices = [i for i in group_indices if i < n]
        if len(valid_indices) < 2:
            all_noise.extend(valid_indices)
            continue

        group_embs = embeddings[valid_indices]

        # Compute per-group cosine similarity matrix ONCE and pass to both
        # HAC and HDBSCAN. This avoids redundant pairwise distance computation
        # inside each algorithm (O(n^2) work saved per retry).
        group_sim = _compute_cosine_similarity_matrix(group_embs)

        try:
            if len(valid_indices) >= _HDBSCAN_LARGE_GROUP_THRESHOLD:
                clusters, noise_local, _ = cluster_hdbscan(
                    embeddings=group_embs,
                    article_indices=valid_indices,
                    entity_name=group_name,
                    entity_group_id=group_id,
                    similarity_matrix=group_sim,
                    params=params,
                )
            else:
                clusters, noise_local, _ = cluster_hac(
                    embeddings=group_embs,
                    article_indices=valid_indices,
                    entity_name=group_name,
                    entity_group_id=group_id,
                    similarity_matrix=group_sim,
                    params=params,
                )

            # Tag clusters with entity provenance
            for c in clusters:
                c.primary_entity = group_name
                if not c.entity_names:
                    c.entity_names = [group_name]
                c.is_entity_seeded = True
                c.parent_entity_group = group_id

            all_clusters.extend(clusters)
            all_noise.extend(noise_local)
            grouped_article_indices.update(valid_indices)

        except Exception as exc:
            logger.warning(f"[cluster] Group '{group_name}' clustering failed: {exc}")
            all_noise.extend(valid_indices)

    # Step 3: Leiden on ungrouped articles (discovery mode)
    ungrouped = [i for i in range(n) if i not in grouped_article_indices]
    if len(ungrouped) >= 4:
        try:
            ungrouped_embs = embeddings[ungrouped]
            discovery_clusters, discovery_noise, _ = cluster_leiden(
                embeddings=ungrouped_embs,
                article_indices=ungrouped,
                params=params,
            )
            all_clusters.extend(discovery_clusters)
            all_noise.extend(discovery_noise)
        except Exception as exc:
            logger.warning(f"[cluster] Leiden discovery failed: {exc}")
            all_noise.extend(ungrouped)
    else:
        all_noise.extend(ungrouped)

    # Step 4: Soft-assign noise articles to nearest cluster (Campello et al. 2013)
    all_clusters, all_noise = _reassign_noise(all_clusters, all_noise, embeddings, params)

    logger.info(
        f"[cluster] {len(all_clusters)} clusters, {len(all_noise)} noise "
        f"from {n} articles across {len(entity_groups)} entity groups"
    )
    return all_clusters, all_noise


async def _run_validation(
    clusters: List[ClusterResult],
    articles: list,
    entity_groups: list,
    params: ClusteringParams,
) -> Tuple[List[str], List[str], List[ValidationResult]]:
    """Run 7-check validation on all clusters using native intelligence validator."""
    try:
        from app.intelligence.cluster.validator import validate_all_clusters

        val_results = validate_all_clusters(
            clusters=clusters,
            articles=articles,
            entity_groups=entity_groups,
            embeddings=None,  # validator uses article-level data, embeddings optional
            params=params,
        )

        passed_ids = [c.cluster_id for c, r in zip(clusters, val_results) if r.passed]
        rejected_ids = [c.cluster_id for c, r in zip(clusters, val_results) if not r.passed]
        return passed_ids, rejected_ids, val_results

    except Exception as exc:
        logger.error(f"[validation] Validation failed: {exc}", exc_info=True)
        # Fail-open: pass all clusters if validator crashes
        return [c.cluster_id for c in clusters], [], []


def _compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Delegate to shared cosine_sim_matrix in similarity.py."""
    from app.intelligence.engine.similarity import cosine_sim_matrix
    return cosine_sim_matrix(embeddings)


async def _get_embeddings(articles: list) -> np.ndarray:
    """Compute or retrieve embeddings for all articles.

    Checks article.embedding first (may be pre-populated by NER/filter step).
    Falls back to batch embedding via EmbeddingTool.

    Note: prefer passing precomputed_embeddings to _run_clustering() instead
    of relying on this function. The source_intel phase already computes
    embeddings and stores them in deps._embeddings — reusing those avoids
    a redundant API call.
    """
    # Check if articles already have embeddings (e.g. set by NER step)
    first = articles[0] if articles else None
    existing = getattr(first, "embedding", None) if first else None
    if existing and len(existing) > 0:
        embs = []
        for art in articles:
            e = getattr(art, "embedding", [])
            if e and len(e) > 0:
                embs.append(e)
            else:
                embs.append([0.0] * len(existing))
        return np.array(embs, dtype=np.float32)

    # Compute embeddings via batch embedding
    texts = []
    for art in articles:
        title = getattr(art, "title", "") or ""
        summary = getattr(art, "summary", "") or ""
        texts.append(f"{title}. {summary[:300]}" if summary else title)

    try:
        from app.tools.llm.embeddings import EmbeddingTool
        emb_tool = EmbeddingTool()
        embs = emb_tool.embed_batch(texts)
        if embs and len(embs) == len(articles):
            return np.array(embs, dtype=np.float32)
    except Exception as exc:
        logger.warning(f"[cluster] Embedding failed: {exc} — falling back to TF-IDF embeddings")

    # Fallback: TF-IDF embeddings (NOT zero vectors).
    # Zero vectors make cosine similarity undefined: 0/0 = NaN → clustering crashes.
    # TF-IDF produces valid non-zero vectors with real lexical similarity structure.
    # Research: Manning et al. (2008) IR Textbook — TF-IDF is correct for short-text similarity.
    logger.warning("[cluster] Using TF-IDF fallback embeddings — NLI embeddings unavailable")
    return _tfidf_fallback_embeddings(texts, dim=512)


def _tfidf_fallback_embeddings(texts: List[str], dim: int = 512) -> "np.ndarray":
    """TF-IDF embeddings as fallback when neural embeddings are unavailable.

    Produces L2-normalized TF-IDF vectors. Unlike zero vectors, these have
    non-zero cosine similarity structure that lets clustering algorithms
    find meaningful groupings based on shared vocabulary.

    Used only when LLMService.embed_batch() fails (API key issues, timeout, etc.).
    Clustering quality will be lower than with neural embeddings but won't crash.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            max_features=dim,
            sublinear_tf=True,
            min_df=1,
            ngram_range=(1, 2),  # bigrams for better phrase matching
        )
        matrix = vec.fit_transform(texts).toarray()
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, 1e-10)).astype(np.float32)
    except Exception as e:
        logger.error(f"[cluster] TF-IDF fallback also failed: {e} — using zeros (clustering degraded)")
        return np.zeros((len(texts), dim), dtype=np.float32)


def _reassign_noise(
    all_clusters: List[ClusterResult],
    all_noise: List[int],
    embeddings: np.ndarray,
    params: ClusteringParams,
) -> Tuple[List[ClusterResult], List[int]]:
    """Soft-assign noise articles to nearest cluster if similarity >= threshold.

    Research basis: Campello et al. (2013) soft membership principle extended
    to post-hoc reassignment. Articles that don't form tight clusters themselves
    may still carry useful signal when attached to an existing cluster.

    Only runs if params.noise_reassign_enabled and noise_reassign_min_similarity set.
    """
    if not params.noise_reassign_enabled or not all_clusters or not all_noise:
        return all_clusters, all_noise

    threshold = params.noise_reassign_min_similarity
    n_reassigned = 0
    remaining_noise = []

    # Compute centroid for each cluster (mean of member embeddings)
    cluster_centroids = []
    for cluster in all_clusters:
        member_indices = cluster.article_indices
        valid = [i for i in member_indices if i < len(embeddings)]
        if valid:
            centroid = embeddings[valid].mean(axis=0)
            # L2 normalize for cosine similarity
            norm = np.linalg.norm(centroid)
            centroid = centroid / max(norm, 1e-10)
            cluster_centroids.append(centroid)
        else:
            cluster_centroids.append(None)

    for noise_idx in all_noise:
        if noise_idx >= len(embeddings):
            remaining_noise.append(noise_idx)
            continue

        noise_emb = embeddings[noise_idx]
        noise_norm = np.linalg.norm(noise_emb)
        noise_emb_norm = noise_emb / max(noise_norm, 1e-10)

        # Find best cluster by cosine similarity to centroid
        best_sim = -1.0
        best_idx = -1
        for ci, centroid in enumerate(cluster_centroids):
            if centroid is None:
                continue
            sim = float(np.dot(noise_emb_norm, centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = ci

        if best_sim >= threshold and best_idx >= 0:
            # Add to cluster
            all_clusters[best_idx].article_indices.append(noise_idx)
            all_clusters[best_idx].article_count += 1
            # Update centroid (running mean) to prevent all noise piling into one cluster
            old_n = len(all_clusters[best_idx].article_indices) - 1
            cluster_centroids[best_idx] = (
                cluster_centroids[best_idx] * old_n + noise_emb_norm
            ) / max(old_n + 1, 1)
            norm2 = np.linalg.norm(cluster_centroids[best_idx])
            cluster_centroids[best_idx] /= max(norm2, 1e-10)
            n_reassigned += 1
        else:
            remaining_noise.append(noise_idx)

    if n_reassigned > 0:
        logger.info(
            f"[cluster] Noise reassignment: {n_reassigned}/{len(all_noise)} articles "
            f"soft-assigned (threshold={threshold:.2f}), "
            f"{len(remaining_noise)} remain as true noise"
        )

    return all_clusters, remaining_noise


def _get_article_indices(group) -> List[int]:
    """Extract article indices from either old or new EntityGroup format."""
    indices = getattr(group, "article_indices", None)
    if indices is not None:
        return list(indices)
    # Old format may use .articles (list of article objects or indices)
    articles_attr = getattr(group, "articles", [])
    if articles_attr and isinstance(articles_attr[0], int):
        return list(articles_attr)
    return []
