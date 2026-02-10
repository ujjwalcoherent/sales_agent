"""
Post-clustering coherence validator — ensures clusters are semantically tight.

THE PROBLEM THIS SOLVES:
  HDBSCAN groups articles by density in UMAP-reduced space (5-dim). But UMAP
  compresses 384 dimensions into 5, which can merge unrelated articles that
  happen to land near each other. Result: clusters with a mix of "RBI rate hike"
  and "RBI digital rupee" articles — same entity, different topics.

WHAT THIS MODULE DOES:
  1. VALIDATE: Check each cluster's coherence in ORIGINAL embedding space
     (not UMAP-reduced). If articles aren't truly similar → flag it.
  2. SPLIT: Clusters below coherence threshold get split via agglomerative
     clustering in original 384-dim space (where similarity is more accurate).
  3. MERGE: Clusters too similar to each other get merged (redundant splits).
  4. REJECT: Very low coherence clusters → articles demoted to noise.

WHY ORIGINAL SPACE, NOT UMAP?
  UMAP is great for density-based clustering (finding neighborhoods), but
  cosine similarity in 5-dim is noisy. The original 384-dim embeddings
  preserve the fine-grained semantic differences we need for validation.

PERFORMANCE: <0.5s for 20 clusters of ~300 articles (matrix ops, no LLM).

REF: BERTopic (Grootendorst 2022) — uses original embeddings for topic
     coherence scoring, not reduced space.
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from app.schemas.news import NewsArticle

logger = logging.getLogger(__name__)


def validate_and_refine_clusters(
    cluster_articles: Dict[int, List[NewsArticle]],
    embeddings: np.ndarray,
    articles: List[NewsArticle],
    labels: np.ndarray,
    min_coherence: float = 0.35,
    reject_threshold: float = 0.20,
    merge_threshold: float = 0.70,
    min_cluster_size: int = 3,
) -> Tuple[Dict[int, List[NewsArticle]], np.ndarray, int]:
    """
    Validate cluster quality and refine: split incoherent, merge redundant, reject noise.

    Operates in ORIGINAL embedding space (384-dim), not UMAP-reduced (5-dim).
    This catches clusters that look tight in UMAP but are actually diverse topics
    compressed together.

    Args:
        cluster_articles: HDBSCAN cluster assignments {cluster_id: [articles]}
        embeddings: ORIGINAL high-dim embeddings (N x 384), one per article
        articles: All articles in pipeline order (matching embeddings)
        labels: HDBSCAN labels array (one per article, -1 = noise)
        min_coherence: Clusters below this get split (mean pairwise cosine sim)
        reject_threshold: Clusters below this get rejected entirely
        merge_threshold: Clusters with centroids more similar than this get merged
        min_cluster_size: Minimum articles for a valid cluster after split

    Returns:
        (refined_cluster_articles, refined_labels, noise_change):
        - refined_cluster_articles: Updated cluster dict (may have new IDs)
        - refined_labels: Updated labels array
        - noise_change: How many articles changed to/from noise (positive = more noise)
    """
    if not cluster_articles:
        return cluster_articles, labels, 0

    # Build article index for fast lookup: id(article) → embedding index
    aid_to_idx = {id(a): i for i, a in enumerate(articles)}
    emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings

    # Normalize embeddings once for cosine similarity
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb_array / norms

    # Phase 1: Compute coherence for each cluster
    cluster_coherence: Dict[int, float] = {}
    cluster_centroids: Dict[int, np.ndarray] = {}

    for cid, arts in cluster_articles.items():
        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) < 2:
            cluster_coherence[cid] = 1.0  # Single article = trivially coherent
            if idxs:
                cluster_centroids[cid] = emb_norm[idxs[0]]
            continue

        cluster_embs = emb_norm[idxs]
        sim_matrix = np.dot(cluster_embs, cluster_embs.T)
        n = len(sim_matrix)
        # Mean pairwise (excluding diagonal)
        mean_sim = (sim_matrix.sum() - n) / max(n * (n - 1), 1)
        cluster_coherence[cid] = float(mean_sim)
        cluster_centroids[cid] = cluster_embs.mean(axis=0)

    # Log coherence distribution
    if cluster_coherence:
        coherences = list(cluster_coherence.values())
        logger.info(
            f"Cluster coherence: min={min(coherences):.3f}, "
            f"mean={sum(coherences)/len(coherences):.3f}, "
            f"max={max(coherences):.3f}"
        )

    # Phase 2: Split incoherent clusters
    new_labels = labels.copy()
    next_cluster_id = max(cluster_articles.keys()) + 1 if cluster_articles else 0
    splits_done = 0

    to_split = {
        cid: arts for cid, arts in cluster_articles.items()
        if cluster_coherence.get(cid, 1.0) < min_coherence
        and len(arts) >= min_cluster_size * 2  # Need enough to split
    }

    # NOTE: Very low coherence clusters are NOT rejected — they're kept in the tree
    # but flagged via intra_cluster_cosine signal so the confidence scorer penalizes them.
    # The user wants ALL data captured, just with quality differentiation.
    # Only truly unsplittable junk would get rejected, but HDBSCAN noise already handles that.
    low_coherence_cids = {
        cid for cid, arts in cluster_articles.items()
        if cluster_coherence.get(cid, 1.0) < reject_threshold
    }
    if low_coherence_cids:
        logger.debug(
            f"  Low coherence clusters (kept, flagged): {low_coherence_cids} "
            f"(coherences: {[round(cluster_coherence[c], 3) for c in low_coherence_cids]})"
        )

    for cid, arts in to_split.items():

        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) < min_cluster_size * 2:
            continue

        # Agglomerative clustering in original space to find coherent sub-groups
        cluster_embs = emb_norm[idxs]
        # Determine number of sub-clusters: aim for coherence > min_coherence
        # Start with 2, increase if needed (max 4 to avoid over-splitting)
        best_sub_labels = None
        best_coherence = cluster_coherence[cid]

        for n_sub in range(2, min(5, len(idxs) // min_cluster_size + 1)):
            try:
                agg = AgglomerativeClustering(
                    n_clusters=n_sub,
                    metric='cosine',
                    linkage='average',
                )
                sub_labels = agg.fit_predict(cluster_embs)

                # Check if ALL sub-clusters are coherent
                all_coherent = True
                min_sub_coherence = 1.0
                for sub_id in range(n_sub):
                    sub_mask = sub_labels == sub_id
                    if sub_mask.sum() < min_cluster_size:
                        all_coherent = False
                        break
                    sub_embs = cluster_embs[sub_mask]
                    sim = np.dot(sub_embs, sub_embs.T)
                    n = len(sim)
                    if n < 2:
                        continue
                    sub_coh = (sim.sum() - n) / max(n * (n - 1), 1)
                    min_sub_coherence = min(min_sub_coherence, sub_coh)
                    if sub_coh < min_coherence:
                        all_coherent = False

                if all_coherent and min_sub_coherence > best_coherence:
                    best_sub_labels = sub_labels
                    best_coherence = min_sub_coherence
            except Exception:
                continue

        if best_sub_labels is not None:
            # Apply the split
            for sub_id in range(best_sub_labels.max() + 1):
                sub_mask = best_sub_labels == sub_id
                sub_idxs = [idxs[i] for i, m in enumerate(sub_mask) if m]
                if len(sub_idxs) >= min_cluster_size:
                    for idx in sub_idxs:
                        new_labels[idx] = next_cluster_id
                    next_cluster_id += 1
                else:
                    # Too small → noise
                    for idx in sub_idxs:
                        new_labels[idx] = -1

            splits_done += 1
            logger.debug(
                f"  Split cluster {cid}: coherence {cluster_coherence[cid]:.3f} → "
                f"{best_sub_labels.max() + 1} sub-clusters (coherence ≥ {best_coherence:.3f})"
            )

    # Phase 3: Merge redundant clusters
    # Rebuild cluster articles from new labels
    refined_groups: Dict[int, List[NewsArticle]] = defaultdict(list)
    for article, label in zip(articles, new_labels):
        if label >= 0:
            refined_groups[int(label)].append(article)

    # Compute centroids for merge check
    refined_centroids: Dict[int, np.ndarray] = {}
    for cid, arts in refined_groups.items():
        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if idxs:
            refined_centroids[cid] = emb_norm[idxs].mean(axis=0)

    # Find pairs to merge (greedy: merge most similar first)
    merges_done = 0
    merge_map: Dict[int, int] = {}  # cid → merge_target_cid
    cids = sorted(refined_centroids.keys())

    if len(cids) > 1:
        centroid_matrix = np.array([refined_centroids[c] for c in cids])
        centroid_sims = np.dot(centroid_matrix, centroid_matrix.T)

        for i in range(len(cids)):
            if cids[i] in merge_map:
                continue
            for j in range(i + 1, len(cids)):
                if cids[j] in merge_map:
                    continue
                if centroid_sims[i, j] >= merge_threshold:
                    # Merge j into i (keep the larger cluster's ID)
                    merge_map[cids[j]] = cids[i]
                    merges_done += 1
                    logger.debug(
                        f"  Merging cluster {cids[j]} into {cids[i]}: "
                        f"similarity={centroid_sims[i, j]:.3f}"
                    )

    # Resolve transitive merge chains: C→B→A becomes C→A
    for cid in list(merge_map.keys()):
        target = merge_map[cid]
        while target in merge_map:
            target = merge_map[target]
        merge_map[cid] = target

    # Apply merges
    if merge_map:
        for idx in range(len(new_labels)):
            label = int(new_labels[idx])
            if label in merge_map:
                new_labels[idx] = merge_map[label]

    # Rebuild final cluster articles
    final_groups: Dict[int, List[NewsArticle]] = defaultdict(list)
    for article, label in zip(articles, new_labels):
        if label >= 0:
            final_groups[int(label)].append(article)

    # Remove clusters that became too small after operations
    noise_additions = 0
    final_clean: Dict[int, List[NewsArticle]] = {}
    for cid, arts in final_groups.items():
        if len(arts) >= min_cluster_size:
            final_clean[cid] = arts
        else:
            for a in arts:
                if id(a) in aid_to_idx:
                    new_labels[aid_to_idx[id(a)]] = -1
                    noise_additions += 1

    original_noise = int(np.sum(labels == -1))
    new_noise = int(np.sum(new_labels == -1))
    noise_change = new_noise - original_noise

    logger.info(
        f"Coherence validation: {splits_done} splits, {merges_done} merges, "
        f"noise {original_noise}→{new_noise} "
        f"({'+' if noise_change >= 0 else ''}{noise_change}), "
        f"{len(final_clean)} final clusters"
    )

    return final_clean, new_labels, noise_change
