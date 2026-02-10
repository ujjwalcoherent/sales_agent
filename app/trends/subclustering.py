"""
Recursive sub-clustering — finds sub-topics within large trend clusters.

Extracted from engine.py for compaction. Takes an engine reference for
access to shared tools (reducer, embeddings, HDBSCAN, keywords, LLM).

STRATEGY: For each large cluster, re-run UMAP + HDBSCAN on just those
articles to find sub-topics. Quality gates filter out noise before LLM.

Depth hierarchy: MAJOR (1) → SUB (2) → MICRO (3)

ROBUSTNESS:
- Computed confidence (not hardcoded 0.6) from cluster signals
- Max 8 children per parent to prevent UI overload and LLM waste
- Coherence validation in original embedding space for sub-clusters
- LLM synthesis skipped for tiny sub-clusters (keyword-only titles)
- LLM output validated: titles must overlap cluster keywords (anti-hallucination)
- Differentiation check excludes child articles from parent centroid
"""

import logging
import math
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.news import NewsArticle
from app.schemas.trends import TrendDepth, TrendNode, TrendTree
from app.trends.signals import compute_all_signals
from app.trends.synthesis import synthesize_clusters
from app.trends.tree_builder import create_trend_node

logger = logging.getLogger(__name__)

# Max children per parent node — prevents UI overload and wasted LLM calls
MAX_CHILDREN_PER_PARENT = 8
# Min articles for LLM synthesis (below this, use keyword-only title)
MIN_ARTICLES_FOR_LLM = 5
# Min coherence in original embedding space for sub-clusters
MIN_SUBCLUSTER_COHERENCE = 0.25


def _compute_coherence(embeddings: List[List[float]]) -> float:
    """Average pairwise cosine similarity within a cluster. 0-1 scale."""
    if len(embeddings) < 2:
        return 1.0
    sim_matrix = cosine_similarity(np.array(embeddings))
    n = len(embeddings)
    upper = sim_matrix[np.triu_indices(n, k=1)]
    return float(upper.mean()) if len(upper) > 0 else 0.0


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
    min_coherence: float = 0.20,
    min_differentiation: float = 0.08,
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

    # R6: Differentiation uses parent_only_embeddings (excludes child articles)
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
    """Generate a readable fallback title from keywords (not raw comma join)."""
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'has',
        'had', 'are', 'was', 'were', 'been', 'being', 'not', 'but', 'its',
        'can', 'will', 'may', 'also', 'all', 'new', 'more', 'over', 'into',
        'out', 'than', 'just', 'about', 'after', 'before', 'between', 'such',
    }
    meaningful = [k for k in keywords if len(k) >= 3 and k.lower() not in stopwords]
    if not meaningful:
        meaningful = [k for k in keywords if len(k) >= 2 and k.lower() not in stopwords]
    if not meaningful:
        return "Emerging Sub-trend"
    selected = meaningful[:3]
    return " / ".join(w.capitalize() for w in selected)


def _validate_synthesis(summary: dict, keywords: List[str]) -> dict:
    """R5: Validate LLM synthesis output. Prevents hallucination.

    Checks:
    - title is non-empty and <=120 chars
    - title has at least 1 keyword overlap (LLM didn't invent a topic)
    - summary doesn't contain AI boilerplate
    Falls back to keyword-based title on failure.
    """
    title = summary.get("trend_title", "")
    kw_fallback = _keyword_fallback_title(keywords)

    if not title or len(title) > 120:
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

    # Check keyword overlap — at least 1 keyword word should appear in title
    if keywords:
        kw_words = set()
        for k in keywords:
            kw_words.update(w.lower() for w in k.split() if len(w) > 2)
        title_words = {w.lower() for w in title.split() if len(w) > 2}
        if not kw_words & title_words:
            logger.debug(f"  LLM title '{title[:40]}' has no keyword overlap → using keywords")
            summary["trend_title"] = kw_fallback

    return summary


def _compute_subcluster_confidence(signals: Dict[str, Any]) -> float:
    """R1: Compute confidence from sub-cluster signals (not hardcoded 0.6)."""
    coherence = signals.get("intra_cluster_cosine", 0.5)
    source_div = signals.get("source_diversity", 0.3)
    article_count = signals.get("article_count", 3)
    # More articles + higher coherence + diverse sources = more confidence
    count_factor = min(1.0, article_count / 10.0) * 0.2
    confidence = coherence * 0.5 + source_div * 0.15 + count_factor + 0.15
    return round(min(1.0, max(0.1, confidence)), 2)


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


async def _subcluster_node(engine, parent_node, parent_articles, current_depth, tree):
    """Sub-cluster a single parent node into child nodes."""
    n_articles = len(parent_articles)
    logger.info(f"Sub-clustering '{parent_node.trend_title}' (depth {current_depth}, {n_articles} articles)")

    # Get stored embeddings
    embeddings = []
    for a in parent_articles:
        emb = getattr(a, 'title_embedding', None)
        if emb is not None:
            embeddings.append(emb)
        else:
            body = a.content or a.summary or ""
            text = f"{a.title}. {body[:1000]}"
            embeddings.append(engine.embedding_tool.embed_batch([text])[0])

    # Auto-dynamic min_cluster_size for sub-clustering
    adaptive_min = max(2, min(10, int(math.log2(max(n_articles, 4)))))

    if len(embeddings) < adaptive_min * 2:
        logger.info(f"  Skipping: too few articles ({len(embeddings)}) for min_cluster={adaptive_min}")
        return []

    try:
        reduced = engine.reducer.reduce(np.array(embeddings))
        labels, noise_count = engine._cluster_hdbscan(
            reduced,
            min_cluster_size_override=adaptive_min,
            log_prefix="    "
        )
    except Exception as e:
        logger.warning(f"  Sub-clustering failed: {e}")
        return []

    sub_cluster_articles = engine._group_by_cluster(parent_articles, labels)
    if len(sub_cluster_articles) <= 1:
        logger.info(f"  No sub-structure found for '{parent_node.trend_title[:30]}'")
        return []

    logger.info(f"  Found {len(sub_cluster_articles)} sub-clusters for '{parent_node.trend_title[:30]}'")

    sub_keywords = engine.keyword_extractor.extract_from_articles(sub_cluster_articles)

    # Quality gates (relaxed for deeper levels)
    depth_coherence = max(0.15, engine.subcluster_min_coherence - (current_depth * 0.05))
    depth_diff = max(0.05, engine.subcluster_min_differentiation - (current_depth * 0.02))

    qualified = {}
    for cid, arts in sub_cluster_articles.items():
        # R6: Exclude child articles from parent embeddings for differentiation
        child_ids_set = {id(a) for a in arts}
        parent_only_embeddings = [
            getattr(a, 'title_embedding', None) for a in parent_articles
            if getattr(a, 'title_embedding', None) is not None and id(a) not in child_ids_set
        ]

        passed, reason = _passes_quality_gates(
            arts, parent_only_embeddings, sub_keywords.get(cid, []),
            min_coherence=depth_coherence,
            min_differentiation=depth_diff,
            min_articles=max(3, adaptive_min),
        )
        if passed:
            qualified[cid] = arts
            logger.debug(f"  + Sub-cluster {cid}: {len(arts)} articles, {reason}")

    if not qualified:
        logger.info(f"  No sub-clusters passed quality gates")
        return []

    # R4: Additional coherence check in original embedding space
    # Reject sub-clusters with low coherence even if UMAP-based gates passed
    coherence_checked = {}
    for cid, arts in qualified.items():
        child_embs = [
            getattr(a, 'title_embedding', None) for a in arts
            if getattr(a, 'title_embedding', None) is not None
        ]
        if len(child_embs) >= 2:
            coh = _compute_coherence(child_embs)
            if coh < MIN_SUBCLUSTER_COHERENCE:
                logger.debug(f"  - Sub-cluster {cid}: original-space coherence {coh:.2f} < {MIN_SUBCLUSTER_COHERENCE}, rejected")
                continue
        coherence_checked[cid] = arts
    qualified = coherence_checked

    if not qualified:
        logger.info(f"  No sub-clusters passed original-space coherence check")
        return []

    # R2: Cap max children per parent
    if len(qualified) > MAX_CHILDREN_PER_PARENT:
        logger.info(f"  Capping {len(qualified)} sub-clusters to {MAX_CHILDREN_PER_PARENT} (by article count)")
        qualified = dict(
            sorted(qualified.items(), key=lambda x: len(x[1]), reverse=True)[:MAX_CHILDREN_PER_PARENT]
        )

    # Compute signals for qualified clusters
    sub_signals = {cid: compute_all_signals(arts) for cid, arts in qualified.items()}

    # R3: Split into LLM-worthy and keyword-only sub-clusters
    llm_qualified = {cid: arts for cid, arts in qualified.items() if len(arts) >= MIN_ARTICLES_FOR_LLM}
    keyword_only = {cid: arts for cid, arts in qualified.items() if len(arts) < MIN_ARTICLES_FOR_LLM}

    sub_summaries = {}
    if llm_qualified:
        llm_kw = {cid: sub_keywords.get(cid, []) for cid in llm_qualified}
        sub_summaries = await synthesize_clusters(llm_qualified, llm_kw, engine.llm_tool, engine.max_concurrent_llm)
        # R5: Validate each LLM output
        for cid in sub_summaries:
            sub_summaries[cid] = _validate_synthesis(sub_summaries[cid], sub_keywords.get(cid, []))

    # Keyword-only titles for tiny clusters (no LLM call — resource efficient)
    for cid in keyword_only:
        kw = sub_keywords.get(cid, [])
        sub_summaries[cid] = {
            "trend_title": _keyword_fallback_title(kw) if kw else f"Sub-topic {cid}",
            "summary": "",
        }

    # Create child TrendNodes
    child_ids = []
    next_depth = current_depth + 1
    for sub_cid in sorted(qualified.keys()):
        # R1: Compute confidence from signals, not hardcoded
        signals = sub_signals.get(sub_cid, {})
        confidence = _compute_subcluster_confidence(signals)

        node = create_trend_node(
            articles=qualified[sub_cid],
            keywords=sub_keywords.get(sub_cid, []),
            signals=signals,
            summary=sub_summaries.get(sub_cid, {}),
            depth=next_depth,
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
    Recursively sub-cluster large trend nodes: MAJOR → SUB → MICRO.

    BFS traversal: processes each level before going deeper.
    Uses adaptive thresholds based on cluster size distribution.
    """
    import time
    t = time.time()
    max_depth_seen = 1

    # Adaptive threshold from cluster size distribution
    cluster_sizes = [
        tree.nodes[str(rid)].article_count for rid in tree.root_ids
        if tree.nodes.get(str(rid))
    ]
    if cluster_sizes:
        avg_size = sum(cluster_sizes) // len(cluster_sizes)
        adaptive_threshold = max(8, min(engine.min_subcluster_size, avg_size // 2))
    else:
        adaptive_threshold = engine.min_subcluster_size

    # Collect qualified nodes
    nodes_to_process = [
        (str(rid), 1) for rid in tree.root_ids
        if tree.nodes.get(str(rid)) and tree.nodes[str(rid)].article_count >= adaptive_threshold
    ]
    logger.info(f"Recursive sub-clustering: {len(nodes_to_process)}/{len(tree.root_ids)} nodes qualify (threshold={adaptive_threshold})")

    total_created = 0
    while nodes_to_process:
        current_batch = nodes_to_process
        nodes_to_process = []

        for node_id_str, current_depth in current_batch:
            if current_depth >= engine.max_depth:
                continue

            parent_node = tree.nodes.get(node_id_str)
            if not parent_node:
                continue

            # Get articles for this cluster
            article_map = {str(a.id): a for a in getattr(engine, '_all_articles', [])}
            parent_articles = [
                article_map[str(aid)] for aid in parent_node.source_articles
                if str(aid) in article_map
            ]
            if len(parent_articles) < adaptive_threshold:
                continue

            child_ids = await _subcluster_node(engine, parent_node, parent_articles, current_depth, tree)

            if child_ids:
                parent_node.children_ids = child_ids
                max_depth_seen = max(max_depth_seen, current_depth + 1)
                total_created += len(child_ids)

                # Queue children for further sub-clustering
                for cid in child_ids:
                    child = tree.nodes.get(str(cid))
                    if child and child.article_count >= engine.min_subcluster_size:
                        nodes_to_process.append((str(cid), current_depth + 1))

    tree.max_depth_reached = max_depth_seen
    elapsed = time.time() - t
    engine.metrics["recursive_subcluster_time"] = round(elapsed, 2)

    _log_tree_structure(tree)
    logger.info(
        f"Recursive sub-clustering: {total_created} sub-clusters, "
        f"max_depth={max_depth_seen}, {len(tree.nodes)} total nodes in {elapsed:.2f}s"
    )
    return tree
