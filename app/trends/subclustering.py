"""
Sub-clustering — finds sub-topics within large trend clusters.

STRATEGY (v2): Agglomerative clustering in original 1024-dim embedding space.
Replaces the old UMAP+HDBSCAN re-run which was unreliable on small article sets
(UMAP needs ~30+ points for stable manifolds; most clusters have 8-20 articles).

KEY CHANGES FROM v1:
- Agglomerative clustering with cosine distance (stable even on 8 articles)
- AI-gated: only sub-clusters trends where Phase 8.5 set should_subcluster=True
- Single level only (MAJOR→SUB). No recursive depth-3 garbage titles.
- Adaptive distance threshold from within-cluster distance distribution

KEPT FROM v1:
- Coherence validation in original embedding space
- Differentiation check (excludes child articles from parent centroid)
- Anti-hallucination validation on LLM titles
- Keyword fallback for small sub-clusters
- Computed confidence (not hardcoded)
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from app.schemas.news import NewsArticle
from app.schemas.trends import TrendDepth, TrendNode, TrendTree
from app.trends.signals import compute_all_signals
from app.trends.synthesis import synthesize_clusters
from app.trends.tree import create_trend_node

logger = logging.getLogger(__name__)

def _load_subcluster_settings():
    """Load subclustering thresholds from config (overridable via .env)."""
    from app.config import get_settings
    s = get_settings()
    global _SC_MAX_CHILDREN, _SC_MIN_LLM, _SC_MIN_COHERENCE, _SC_MIN_ARTICLES
    _SC_MAX_CHILDREN = s.max_children_per_parent      # default 6
    _SC_MIN_LLM = 3                                   # min articles for LLM synthesis
    _SC_MIN_COHERENCE = s.min_subcluster_coherence     # default 0.25
    _SC_MIN_ARTICLES = s.min_articles_for_subclustering  # default 6

# Initialize with defaults — must match _load_subcluster_settings values
_SC_MAX_CHILDREN = 6
_SC_MIN_LLM = 3       # was 5, inconsistent with loader; fixed to match
_SC_MIN_COHERENCE = 0.35
_SC_MIN_ARTICLES = 6


def _compute_coherence(embeddings: List[List[float]]) -> float:
    """Average pairwise cosine similarity within a cluster. 0-1 scale."""
    from app.tools.embeddings import mean_pairwise_cosine
    return mean_pairwise_cosine(np.array(embeddings))


def _compute_differentiation(
    parent_embeddings: List[List[float]],
    child_embeddings: List[List[float]],
) -> float:
    """Cosine distance between parent and child centroids. 0-2 scale."""
    if not parent_embeddings or not child_embeddings:
        return 0.0
    parent_arr = np.array(parent_embeddings) if not isinstance(parent_embeddings, np.ndarray) else parent_embeddings
    child_arr = np.array(child_embeddings) if not isinstance(child_embeddings, np.ndarray) else child_embeddings
    sim = cosine_similarity(
        [np.mean(parent_arr, axis=0)],
        [np.mean(child_arr, axis=0)]
    )[0, 0]
    return 1.0 - sim


def _passes_quality_gates(
    articles: List,
    parent_only_embeddings: List[List[float]],
    keywords: List[str],
    min_coherence: float = 0.25,
    min_differentiation: float = 0.05,
    min_articles: int = 3,
) -> tuple:
    """Check if a sub-cluster passes quality gates. Returns (passed, reason)."""
    if len(articles) < min_articles:
        return False, f"Too few articles ({len(articles)} < {min_articles})"

    child_embeddings = [
        getattr(a, 'title_embedding', None)
        for a in articles
        if getattr(a, 'title_embedding', None) is not None
    ]
    if len(child_embeddings) < 2:
        return False, "Not enough embeddings"

    coherence = _compute_coherence(child_embeddings)
    if coherence < min_coherence:
        return False, f"Low coherence ({coherence:.2f} < {min_coherence})"

    # Differentiation uses parent_only_embeddings (excludes child articles)
    if parent_only_embeddings:
        diff = _compute_differentiation(parent_only_embeddings, child_embeddings)
        if diff < min_differentiation:
            return False, f"Too similar to parent ({diff:.2f} < {min_differentiation})"
    else:
        diff = None

    meaningful_kw = [k for k in keywords if len(k) > 2 and k.lower() not in {'the', 'and', 'for', 'that', 'this', 'with'}]
    if len(meaningful_kw) < 2:
        return False, "No meaningful keywords"

    diff_str = f"{diff:.2f}" if diff is not None else "N/A"
    return True, f"Passed (coherence={coherence:.2f}, diff={diff_str})"


def _keyword_fallback_title(keywords: List[str]) -> str:
    """Generate a readable fallback title from keywords.

    Produces "Keyword1, Keyword2 and Keyword3 Developments" instead of
    the old "Keyword1 / Keyword2 / Keyword3" which looked like raw data.
    """
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'has',
        'had', 'are', 'was', 'were', 'been', 'being', 'not', 'but', 'its',
        'can', 'will', 'may', 'also', 'all', 'new', 'more', 'over', 'into',
        'out', 'than', 'just', 'about', 'after', 'before', 'between', 'such',
        'his', 'her', 'him', 'she', 'they', 'them', 'who', 'what', 'when',
        'where', 'how', 'which', 'there', 'here', 'very', 'much', 'many',
        'some', 'any', 'each', 'every', 'other', 'both', 'few', 'most',
    }
    meaningful = [k for k in keywords if len(k) >= 3 and k.lower() not in stopwords]
    if not meaningful:
        meaningful = [k for k in keywords if len(k) >= 2 and k.lower() not in stopwords]
    if not meaningful:
        return "Emerging Sub-trend"
    selected = [w.capitalize() for w in meaningful[:3]]
    if len(selected) == 1:
        return f"{selected[0]} Developments"
    elif len(selected) == 2:
        return f"{selected[0]} and {selected[1]} Developments"
    else:
        return f"{selected[0]}, {selected[1]} and {selected[2]} Developments"


def _validate_synthesis(summary: dict, keywords: List[str]) -> dict:
    """Validate LLM synthesis output. Light-touch — trust LLM titles.

    Only rejects:
    - Empty or >120 char titles
    - AI boilerplate ("as a language model" etc.)

    Does NOT reject on keyword mismatch — the LLM title is almost always
    better than keyword fallback, especially when extracted keywords are
    generic or noisy (e.g. "His", "Aqi", "Court").
    """
    title = summary.get("trend_title", "")
    kw_fallback = _keyword_fallback_title(keywords)

    if not title or len(title.strip()) < 5 or len(title) > 120:
        summary["trend_title"] = kw_fallback
        return summary

    # Check for AI boilerplate
    boilerplate = ["i'm an ai", "as a language model", "i cannot", "as an ai"]
    title_lower = title.lower()
    body_lower = (summary.get("summary", "") or "").lower()
    if any(bp in title_lower or bp in body_lower for bp in boilerplate):
        summary["trend_title"] = kw_fallback
        summary["summary"] = ""
        return summary

    return summary


def _compute_subcluster_confidence(signals: Dict[str, Any]) -> float:
    """Compute confidence from sub-cluster signals (not hardcoded 0.6)."""
    coherence = signals.get("intra_cluster_cosine", 0.5)
    source_div = signals.get("source_diversity", 0.3)
    article_count = signals.get("article_count", 3)
    # More articles + higher coherence + diverse sources = more confidence
    count_factor = min(1.0, article_count / 10.0) * 0.2
    confidence = coherence * 0.5 + source_div * 0.15 + count_factor + 0.15
    return round(min(1.0, max(0.1, confidence)), 2)


def _agglomerative_cluster(
    embeddings: np.ndarray,
    min_articles: int = 3,
) -> Optional[np.ndarray]:
    """Cluster articles using agglomerative clustering in original embedding space.

    Uses cosine distance + average linkage. The distance threshold is adaptive:
    split at 70% of the mean pairwise distance within the cluster.

    Returns cluster labels (0-indexed), or None if no meaningful split found.
    """
    n = len(embeddings)
    if n < min_articles * 2:
        return None

    # Cosine distance matrix in original 1024-dim space (no UMAP needed)
    dist_matrix = cosine_distances(embeddings)

    # Condensed distance matrix for scipy
    condensed = squareform(dist_matrix, checks=False)

    # Average linkage — robust for text embeddings (less sensitive to outliers
    # than single linkage, less prone to chaining than complete linkage)
    Z = linkage(condensed, method='average')

    # Adaptive threshold: 70% of mean pairwise distance
    # Tight clusters (low distance) → lower threshold → fewer splits
    # Spread clusters (high distance) → higher threshold → allow more splits
    mean_dist = float(np.mean(condensed))
    threshold = mean_dist * 0.7

    # Clamp threshold to reasonable range for cosine distances
    threshold = max(0.05, min(0.40, threshold))

    labels = fcluster(Z, t=threshold, criterion='distance')
    # fcluster labels are 1-indexed, convert to 0-indexed
    labels = labels - 1

    n_clusters = len(set(labels))
    if n_clusters <= 1:
        # Try a lower threshold for tighter split
        threshold_low = mean_dist * 0.5
        threshold_low = max(0.03, min(0.30, threshold_low))
        labels = fcluster(Z, t=threshold_low, criterion='distance') - 1
        n_clusters = len(set(labels))
        if n_clusters <= 1:
            return None

    # Reject if too many tiny fragments (sign of noise, not structure)
    from collections import Counter
    size_counts = Counter(labels)
    meaningful_clusters = sum(1 for c in size_counts.values() if c >= min_articles)
    if meaningful_clusters < 2:
        return None

    # Cap at _SC_MAX_CHILDREN clusters by merging smallest into nearest
    if n_clusters > _SC_MAX_CHILDREN:
        # Keep largest clusters, reassign small ones to nearest centroid
        sorted_clusters = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
        keep_labels = {c for c, _ in sorted_clusters[:_SC_MAX_CHILDREN]}
        centroids = {}
        for lbl in keep_labels:
            mask = labels == lbl
            centroids[lbl] = embeddings[mask].mean(axis=0)

        new_labels = labels.copy()
        for i in range(n):
            if labels[i] not in keep_labels:
                # Assign to nearest kept centroid
                dists = {lbl: float(cosine_distances([embeddings[i]], [c])[0, 0])
                         for lbl, c in centroids.items()}
                new_labels[i] = min(dists, key=dists.get)
        labels = new_labels

    logger.debug(
        f"  Agglomerative: {n} articles → {len(set(labels))} clusters "
        f"(threshold={threshold:.3f}, mean_dist={mean_dist:.3f})"
    )
    return labels


def _log_tree_structure(tree: TrendTree):
    """Log the final tree structure."""
    def _log_node(node_id, indent=0):
        node = tree.nodes.get(str(node_id))
        if not node:
            return
        prefix = "  " * indent
        depth_label = {1: "MAJOR", 2: "SUB", 3: "MICRO"}.get(node.depth, f"D{node.depth}")
        logger.debug(
            f"{prefix}[{depth_label}] {node.trend_title[:50]} "
            f"({node.article_count} articles, {len(node.children_ids)} children)"
        )
        for child_id in node.children_ids:
            _log_node(child_id, indent + 1)

    for root_id in tree.root_ids:
        _log_node(root_id)


async def _subcluster_node(engine, parent_node, parent_articles, tree):
    """Sub-cluster a single parent node into child nodes.

    Uses agglomerative clustering in original 1024-dim embedding space
    instead of re-running UMAP+HDBSCAN (which is unreliable on small sets).
    """
    n_articles = len(parent_articles)
    logger.info(f"Sub-clustering '{parent_node.trend_title}' ({n_articles} articles)")

    # Collect embeddings (already computed in Phase 3)
    embeddings = []
    valid_articles = []
    for a in parent_articles:
        emb = getattr(a, 'title_embedding', None)
        if emb is not None:
            embeddings.append(emb)
            valid_articles.append(a)
        else:
            body = a.content or a.summary or ""
            text = f"{a.title}. {body[:1000]}"
            emb = engine.embedding_tool.embed_batch([text])[0]
            embeddings.append(emb)
            valid_articles.append(a)

    emb_array = np.array(embeddings)

    # Agglomerative clustering in original space (no UMAP re-run)
    labels = _agglomerative_cluster(emb_array, min_articles=3)
    if labels is None:
        logger.info(f"  No sub-structure found for '{parent_node.trend_title[:40]}'")
        return []

    # Group articles by cluster label
    sub_cluster_articles: Dict[int, List[NewsArticle]] = {}
    for i, lbl in enumerate(labels):
        lbl = int(lbl)
        sub_cluster_articles.setdefault(lbl, []).append(valid_articles[i])

    if len(sub_cluster_articles) <= 1:
        logger.info(f"  Single cluster after grouping for '{parent_node.trend_title[:40]}'")
        return []

    logger.info(f"  Found {len(sub_cluster_articles)} sub-clusters for '{parent_node.trend_title[:40]}'")

    sub_keywords = engine.keyword_extractor.extract_from_articles(sub_cluster_articles)

    # Quality gates
    qualified = {}
    for cid, arts in sub_cluster_articles.items():
        # Exclude child articles from parent embeddings for differentiation
        child_ids_set = {id(a) for a in arts}
        parent_only_embeddings = [
            getattr(a, 'title_embedding', None) for a in parent_articles
            if getattr(a, 'title_embedding', None) is not None and id(a) not in child_ids_set
        ]

        passed, reason = _passes_quality_gates(
            arts, parent_only_embeddings, sub_keywords.get(cid, []),
            min_coherence=engine.subcluster_min_coherence,
            min_differentiation=engine.subcluster_min_differentiation,
            min_articles=3,
        )
        if passed:
            qualified[cid] = arts
            logger.debug(f"  + Sub-cluster {cid}: {len(arts)} articles, {reason}")
        else:
            logger.debug(f"  - Sub-cluster {cid}: {len(arts)} articles, REJECTED: {reason}")

    if not qualified:
        logger.info(f"  No sub-clusters passed quality gates")
        return []

    # Coherence check in original embedding space
    coherence_checked = {}
    for cid, arts in qualified.items():
        child_embs = [
            getattr(a, 'title_embedding', None) for a in arts
            if getattr(a, 'title_embedding', None) is not None
        ]
        if len(child_embs) >= 2:
            coh = _compute_coherence(child_embs)
            if coh < _SC_MIN_COHERENCE:
                logger.debug(f"  - Sub-cluster {cid}: coherence {coh:.2f} < {_SC_MIN_COHERENCE}, rejected")
                continue
        coherence_checked[cid] = arts
    qualified = coherence_checked

    if not qualified:
        logger.info(f"  No sub-clusters passed coherence check")
        return []

    # Cap max children
    if len(qualified) > _SC_MAX_CHILDREN:
        logger.info(f"  Capping {len(qualified)} sub-clusters to {_SC_MAX_CHILDREN}")
        qualified = dict(
            sorted(qualified.items(), key=lambda x: len(x[1]), reverse=True)[:_SC_MAX_CHILDREN]
        )

    # Compute signals for qualified clusters
    sub_signals = {cid: compute_all_signals(arts) for cid, arts in qualified.items()}

    # Split into LLM-worthy and keyword-only sub-clusters
    llm_qualified = {cid: arts for cid, arts in qualified.items() if len(arts) >= _SC_MIN_LLM}
    keyword_only = {cid: arts for cid, arts in qualified.items() if len(arts) < _SC_MIN_LLM}

    sub_summaries = {}
    if llm_qualified:
        llm_kw = {cid: sub_keywords.get(cid, []) for cid in llm_qualified}
        sub_summaries = await synthesize_clusters(llm_qualified, llm_kw, engine.llm_tool, engine.max_concurrent_llm)
        # Validate each LLM output (anti-hallucination)
        for cid in sub_summaries:
            sub_summaries[cid] = _validate_synthesis(sub_summaries[cid], sub_keywords.get(cid, []))

    # Keyword-only titles for small clusters (no LLM call)
    for cid in keyword_only:
        kw = sub_keywords.get(cid, [])
        sub_summaries[cid] = {
            "trend_title": _keyword_fallback_title(kw) if kw else f"Sub-topic {cid}",
            "summary": "",
        }

    # Create child TrendNodes
    child_ids = []
    for sub_cid in sorted(qualified.keys()):
        signals = sub_signals.get(sub_cid, {})
        confidence = _compute_subcluster_confidence(signals)

        node = create_trend_node(
            articles=qualified[sub_cid],
            keywords=sub_keywords.get(sub_cid, []),
            signals=signals,
            summary=sub_summaries.get(sub_cid, {}),
            depth=2,  # Always depth 2 (SUB) — single level only
            parent_id=parent_node.id,
            parent_tree_path=parent_node.tree_path,
            parent_sectors=parent_node.primary_sectors,
            confidence=confidence,
        )
        tree.nodes[str(node.id)] = node
        child_ids.append(node.id)

    return child_ids


async def recursive_subcluster(engine, tree, cluster_summaries):
    """
    Sub-cluster trend nodes: MAJOR → SUB (single level).

    AI-GATED: Only processes nodes where Phase 8.5 AI validator set
    should_subcluster=True. Falls back to article count threshold if
    no AI recommendation exists (e.g., mock mode).
    """
    _load_subcluster_settings()
    t = time.time()

    # Collect nodes that the AI recommends for sub-clustering
    nodes_to_process = []
    ai_gated_count = 0
    fallback_count = 0

    for rid in tree.root_ids:
        node = tree.nodes.get(str(rid))
        if not node:
            continue

        # Primary gate: AI recommendation from Phase 8.5
        if getattr(node, 'should_subcluster', False):
            if node.article_count >= _SC_MIN_ARTICLES:
                nodes_to_process.append(str(rid))
                ai_gated_count += 1
                continue

        # Fallback gate: large clusters without AI recommendation
        # (e.g., when running in mock mode or AI didn't set the flag)
        if node.article_count >= max(engine.min_subcluster_size, 8):
            nodes_to_process.append(str(rid))
            fallback_count += 1

    logger.info(
        f"Sub-clustering: {len(nodes_to_process)}/{len(tree.root_ids)} nodes "
        f"({ai_gated_count} AI-recommended, {fallback_count} size-fallback)"
    )

    if not nodes_to_process:
        logger.info("Sub-clustering: no nodes qualify")
        engine.metrics["recursive_subcluster_time"] = 0.0
        return tree

    total_created = 0
    article_map = {str(a.id): a for a in getattr(engine, '_all_articles', [])}

    for node_id_str in nodes_to_process:
        parent_node = tree.nodes.get(node_id_str)
        if not parent_node:
            continue

        parent_articles = [
            article_map[str(aid)] for aid in parent_node.source_articles
            if str(aid) in article_map
        ]
        if len(parent_articles) < _SC_MIN_ARTICLES:
            continue

        child_ids = await _subcluster_node(engine, parent_node, parent_articles, tree)

        if child_ids:
            parent_node.children_ids = child_ids
            total_created += len(child_ids)

    max_depth_seen = 2 if total_created > 0 else 1
    tree.max_depth_reached = max_depth_seen
    elapsed = time.time() - t
    engine.metrics["recursive_subcluster_time"] = round(elapsed, 2)

    _log_tree_structure(tree)
    logger.info(
        f"Sub-clustering: {total_created} sub-nodes created, "
        f"{len(tree.nodes)} total nodes in {elapsed:.1f}s"
    )
    return tree
