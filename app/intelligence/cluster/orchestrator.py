"""
Cluster Orchestrator — native implementation using intelligence/cluster/ algorithms.

Pipeline per entity group:
  HAC   → groups with 5-50 articles (average linkage, silhouette sweep)
  HDBSCAN → groups with 50+ articles (soft membership vectors)
  Leiden  → ungrouped / discovery mode articles

ValidationAgent can signal ClusterAgent to re-cluster (max 2 retries).
This is the bidirectional loop from the Blackboard pattern (Erman et al. 1980).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import (
    AgentRequestType,
    ClusterResult,
    DiscoveryScope,
    EntityGroup,
    PipelineState,
    RequestPriority,
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
) -> Tuple[List[ClusterResult], List[str], List[str], List[ValidationResult], List[int]]:
    """Run similarity → cluster → validate loop.

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

        clusters, noise = await _run_clustering(articles, entity_groups, scope, params)
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
) -> Tuple[List[ClusterResult], List[int]]:
    """Run native HAC/HDBSCAN/Leiden clustering on entity groups."""
    from app.intelligence.cluster.algorithms import (
        cluster_hac,
        cluster_hdbscan_soft as cluster_hdbscan,
        cluster_leiden,
    )

    if not articles:
        return [], []

    # Step 1: compute embeddings for all articles (needed by HAC/HDBSCAN)
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

        try:
            if len(valid_indices) >= _HDBSCAN_LARGE_GROUP_THRESHOLD:
                clusters, noise_local, _ = cluster_hdbscan(
                    embeddings=group_embs,
                    article_indices=valid_indices,
                    entity_name=group_name,
                    entity_group_id=group_id,
                    params=params,
                )
            else:
                clusters, noise_local, _ = cluster_hac(
                    embeddings=group_embs,
                    article_indices=valid_indices,
                    entity_name=group_name,
                    entity_group_id=group_id,
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


async def _get_embeddings(articles: list) -> np.ndarray:
    """Compute or retrieve embeddings for all articles.

    Checks article.embedding first (may be pre-populated).
    Falls back to batch embedding via LLMService.
    """
    # Check if articles already have embeddings
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
        from app.tools.llm.llm_service import LLMService
        llm = LLMService()
        embs = llm.embed_batch(texts)
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
