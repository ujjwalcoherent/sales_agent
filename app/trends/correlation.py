"""
Cross-trend correlation — detects chain reactions between clusters.

4 methods layered (not standalone):
1. Entity bridges (IDF-weighted shared entities, filter generic ones)
2. Sector chains (configurable adjacency map)
3. Temporal lag (median timestamp comparison, NOT Granger)
4. LLM causal narrative (only for top pre-filtered pairs)

Called from engine.py Layer 3.
"""

import logging
from collections import Counter
from datetime import datetime
from itertools import combinations
from math import log
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.schemas.news import NewsArticle

logger = logging.getLogger(__name__)

# Indian business sector adjacency — configurable, not code
SECTOR_CHAINS = {
    "monetary_policy": ["banking", "nbfc", "housing", "consumer_credit"],
    "regulation": ["banking", "insurance", "fintech", "compliance"],
    "trade_policy": ["exports", "manufacturing", "logistics", "retail"],
    "technology": ["it_services", "startups", "digital_transformation"],
    "energy": ["manufacturing", "logistics", "agriculture", "chemicals"],
    "banking": ["nbfc", "housing", "consumer_credit", "fintech"],
    "infrastructure": ["real_estate", "construction", "logistics", "cement"],
}


def find_correlations(
    cluster_articles: Dict[int, List[NewsArticle]],
    cluster_signals: Dict[int, Dict[str, Any]],
    bridge_threshold: float = 0.10,
    max_correlations: int = 15,
    cluster_centroids: Optional[Dict[int, Any]] = None,
    centroid_threshold: float = 0.35,
) -> Tuple[List[Dict], List[List[int]], Dict[int, List[str]]]:
    """Find cross-trend correlations via entity bridges + semantic similarity.

    Args:
        cluster_articles: {cluster_id: [articles]}
        cluster_signals: {cluster_id: signals_dict} (has top_entities, dominant_event)
        bridge_threshold: Minimum IDF-weighted bridge score to consider
        max_correlations: Max number of correlations to return
        cluster_centroids: {cluster_id: np.ndarray} — mean embedding vectors
        centroid_threshold: Minimum centroid cosine similarity for semantic edges

    Returns:
        (edges, cascades, bridge_entities):
        - edges: List of correlation dicts with source, target, strength, lag, etc.
        - cascades: List of cluster chains [cid_a, cid_b, cid_c, ...]
        - bridge_entities: {(cid_a, cid_b): [shared_entity_names]}
    """
    if len(cluster_articles) < 2:
        return [], [], {}

    # Step 1: Extract entities per cluster (normalized for consistent matching)
    try:
        from app.news.entity_normalizer import normalize_entity
        _norm = lambda name: normalize_entity(name).lower().strip()
    except ImportError:
        _norm = lambda name: name.lower().strip()

    cluster_entities = {}
    for cid, arts in cluster_articles.items():
        entities = set()
        for a in arts:
            for name in getattr(a, "entity_names", []):
                entities.add(_norm(name))
        cluster_entities[cid] = entities

    # Step 2: Compute entity IDF across clusters (rarity = specificity)
    entity_cluster_count = Counter()
    for entities in cluster_entities.values():
        for e in entities:
            entity_cluster_count[e] += 1

    n_clusters = len(cluster_articles)
    # Filter only truly ubiquitous entities.
    # Adaptive: for small cluster sets (<=7), allow entities in up to 70% of clusters.
    # For larger sets, allow up to n_clusters-1 (filter only ALL-cluster entities).
    import math
    if n_clusters <= 7:
        max_freq = max(2, math.ceil(n_clusters * 0.7))  # e.g., 4 clusters → 3, 6 → 5
    else:
        max_freq = n_clusters - 1

    # Step 3: Find entity bridges between cluster pairs
    edges = []
    bridge_entities_map = {}
    cids = sorted(cluster_articles.keys())

    for cid_a, cid_b in combinations(cids, 2):
        ents_a = cluster_entities.get(cid_a, set())
        ents_b = cluster_entities.get(cid_b, set())
        if not ents_a or not ents_b:
            continue

        shared = ents_a & ents_b
        # Remove generic entities
        specific_shared = {e for e in shared if entity_cluster_count[e] <= max_freq}
        if not specific_shared:
            continue

        # IDF-weighted bridge score
        idf_sum = sum(
            log(n_clusters / entity_cluster_count[e])
            for e in specific_shared
            if entity_cluster_count[e] > 0
        )
        bridge_score = idf_sum / min(len(ents_a), len(ents_b))

        if bridge_score < bridge_threshold:
            continue

        # Step 4: Temporal lag detection
        lag_hours = _compute_temporal_lag(
            cluster_articles[cid_a], cluster_articles[cid_b]
        )

        # Determine relationship type (1h threshold — India news cycles are fast)
        if lag_hours is not None and abs(lag_hours) >= 1.0:
            if lag_hours > 0:
                relationship = "causes"
                source, target = cid_a, cid_b
            else:
                relationship = "caused_by"
                source, target = cid_b, cid_a
        else:
            relationship = "co-occurs"
            source, target = cid_a, cid_b
            lag_hours = 0.0

        # Step 5: Sector chain bonus
        sector_match = _check_sector_chain(
            cluster_signals.get(cid_a, {}),
            cluster_signals.get(cid_b, {}),
        )

        edge = {
            "source": source,
            "target": target,
            "relationship": relationship,
            "strength": round(bridge_score, 4),
            "lag_hours": round(abs(lag_hours or 0), 1),
            "bridge_entities": sorted(specific_shared)[:10],
            "sector_chain": sector_match,
        }
        edges.append(edge)
        bridge_entities_map[(cid_a, cid_b)] = sorted(specific_shared)

    # Step 5b: Semantic centroid similarity (finds topically related trends
    # that share NO entities — e.g., "RBI rate hike" and "credit squeeze")
    semantic_edges_added = 0
    if cluster_centroids:
        existing_pairs = {(e["source"], e["target"]) for e in edges}
        existing_pairs |= {(e["target"], e["source"]) for e in edges}

        for cid_a, cid_b in combinations(cids, 2):
            if (cid_a, cid_b) in existing_pairs:
                continue  # Already have an entity bridge edge
            cent_a = cluster_centroids.get(cid_a)
            cent_b = cluster_centroids.get(cid_b)
            if cent_a is None or cent_b is None:
                continue
            sim = _cosine_similarity(cent_a, cent_b)
            if sim >= centroid_threshold:
                lag_hours = _compute_temporal_lag(
                    cluster_articles[cid_a], cluster_articles[cid_b]
                )
                sector_match = _check_sector_chain(
                    cluster_signals.get(cid_a, {}),
                    cluster_signals.get(cid_b, {}),
                )
                edges.append({
                    "source": cid_a,
                    "target": cid_b,
                    "relationship": "topically_related",
                    "strength": round(sim, 4),
                    "lag_hours": round(abs(lag_hours or 0), 1),
                    "bridge_entities": [],
                    "sector_chain": sector_match,
                    "semantic_similarity": round(sim, 4),
                })
                semantic_edges_added += 1

    # Sort by strength, take top N
    edges.sort(key=lambda e: e["strength"], reverse=True)
    edges = edges[:max_correlations]

    # Step 6: Build cascades (chains of 3+ connected clusters)
    cascades = _build_cascades(edges)

    # Inject correlations into cluster signals
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        for cid in [src, tgt]:
            if cid in cluster_signals:
                if "correlations" not in cluster_signals[cid]:
                    cluster_signals[cid]["correlations"] = []
                cluster_signals[cid]["correlations"].append(edge)
                cluster_signals[cid]["in_cascade"] = True

    # Diagnostic: count how many pairs were evaluated vs passed
    n_pairs = len(list(combinations(cids, 2)))
    n_with_shared = sum(
        1 for a, b in combinations(cids, 2)
        if cluster_entities.get(a, set()) & cluster_entities.get(b, set())
    )
    logger.info(
        f"Correlation: {len(edges)} edges ({semantic_edges_added} semantic), "
        f"{len(cascades)} cascades from {len(cluster_articles)} clusters "
        f"(evaluated {n_pairs} pairs, {n_with_shared} with shared entities, "
        f"bridge_threshold={bridge_threshold})"
    )

    return edges, cascades, bridge_entities_map


def _compute_temporal_lag(
    articles_a: List[NewsArticle],
    articles_b: List[NewsArticle],
) -> Optional[float]:
    """Detect if cluster A's articles consistently precede cluster B's.

    Returns lag in hours (positive = A precedes B), or None if insufficient data.
    """
    ts_a = _extract_timestamps(articles_a)
    ts_b = _extract_timestamps(articles_b)

    if len(ts_a) < 2 or len(ts_b) < 2:
        return None

    median_a = np.median(ts_a)
    median_b = np.median(ts_b)
    lag = (median_b - median_a) / 3600  # seconds → hours

    # Return lag for any meaningful gap (>30 min).
    # Caller decides the causal/co-occurs threshold.
    if abs(lag) < 0.5:
        return None

    return lag


def _extract_timestamps(articles: List[NewsArticle]) -> List[float]:
    """Extract UNIX timestamps from articles."""
    timestamps = []
    for a in articles:
        pub = getattr(a, "published_at", None) or getattr(a, "published_date", None)
        if pub:
            if isinstance(pub, datetime):
                timestamps.append(pub.timestamp())
            elif isinstance(pub, str):
                try:
                    timestamps.append(datetime.fromisoformat(pub).timestamp())
                except (ValueError, TypeError):
                    pass
    return timestamps


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors (numpy or list)."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _check_sector_chain(
    signals_a: Dict[str, Any],
    signals_b: Dict[str, Any],
) -> Optional[str]:
    """Check if two clusters' sectors are adjacent in the sector chain map."""
    event_a = signals_a.get("dominant_event", "").lower().replace(" ", "_")
    event_b = signals_b.get("dominant_event", "").lower().replace(" ", "_")

    if not event_a or not event_b:
        return None

    for sector, adjacents in SECTOR_CHAINS.items():
        if sector in event_a and event_b in adjacents:
            return f"{sector} -> {event_b}"
        if sector in event_b and event_a in adjacents:
            return f"{sector} -> {event_a}"

    return None


def _build_cascades(edges: List[Dict]) -> List[List[int]]:
    """Build chains of 3+ connected clusters from edges."""
    if not edges:
        return []

    # Build adjacency from causal edges + strong co-occurs (strength > 0.3)
    adj = {}
    for edge in edges:
        if edge["relationship"] in ("causes", "caused_by"):
            src = edge["source"]
            tgt = edge["target"]
            adj.setdefault(src, []).append(tgt)
        elif edge["relationship"] in ("co-occurs", "topically_related") and edge.get("strength", 0) > 0.3:
            # Strong co-occurrence/semantic: add for cascade discovery
            src, tgt = edge["source"], edge["target"]
            adj.setdefault(src, []).append(tgt)

    # DFS to find chains
    cascades = []
    visited = set()

    def dfs(node, path):
        if len(path) >= 3:
            cascades.append(list(path))
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, path + [neighbor])
                visited.discard(neighbor)

    for start in adj:
        visited = {start}
        dfs(start, [start])

    # Deduplicate (keep longest chains)
    unique = []
    for chain in sorted(cascades, key=len, reverse=True):
        chain_set = set(chain)
        if not any(chain_set <= set(u) for u in unique):
            unique.append(chain)

    return unique[:8]  # Top 8 cascades


def find_subtopic_correlations(
    tree,
    article_map: Dict[str, Any],
    bridge_threshold: float = 0.10,
    max_correlations: int = 8,
) -> List[Dict]:
    """Find entity bridges between sub-trends across different parent clusters.

    Shows chain reactions at sub-trend level — e.g., "RBI rate hike" sub-trend
    in one cluster connects to "NBFC lending squeeze" sub-trend in another.

    Args:
        tree: TrendTree with sub-clustered nodes.
        article_map: {str(article.id): article} for entity lookup.
        bridge_threshold: Minimum IDF-weighted bridge score.
        max_correlations: Max edges to return.

    Returns:
        List of correlation dicts with source/target node IDs, bridge entities, etc.
    """
    # Collect depth-2 (SUB) nodes grouped by parent
    sub_nodes_by_parent: Dict[str, List] = {}
    for node_id_str, node in tree.nodes.items():
        if node.depth == 2 and node.parent_id:
            parent_key = str(node.parent_id)
            sub_nodes_by_parent.setdefault(parent_key, []).append(node)

    # Need at least 2 parents with sub-trends
    parents_with_subs = [p for p, subs in sub_nodes_by_parent.items() if len(subs) >= 1]
    if len(parents_with_subs) < 2:
        return []

    # Extract entities per sub-trend from their source articles
    node_entities: Dict[str, set] = {}
    for parent_key in parents_with_subs:
        for node in sub_nodes_by_parent[parent_key]:
            entities = set()
            for aid in node.source_articles:
                article = article_map.get(str(aid))
                if article:
                    for name in getattr(article, "entity_names", []):
                        entities.add(name.lower().strip())
            # Also include key_entities from synthesis
            for e in node.key_entities:
                entities.add(e.lower().strip())
            node_entities[str(node.id)] = entities

    # Compute entity IDF across all sub-trends
    entity_node_count = Counter()
    for entities in node_entities.values():
        for e in entities:
            entity_node_count[e] += 1

    n_sub = len(node_entities)
    max_freq = n_sub * 0.7  # filter entities in >70% of sub-trends (was 50%)

    # Find bridges between sub-trends from DIFFERENT parents
    edges = []
    for i, parent_a in enumerate(parents_with_subs):
        for parent_b in parents_with_subs[i + 1:]:
            for node_a in sub_nodes_by_parent[parent_a]:
                ents_a = node_entities.get(str(node_a.id), set())
                if not ents_a:
                    continue
                for node_b in sub_nodes_by_parent[parent_b]:
                    ents_b = node_entities.get(str(node_b.id), set())
                    if not ents_b:
                        continue

                    shared = ents_a & ents_b
                    specific = {e for e in shared if entity_node_count[e] <= max_freq}
                    if not specific:
                        continue

                    idf_sum = sum(
                        log(n_sub / entity_node_count[e])
                        for e in specific
                        if entity_node_count[e] > 0
                    )
                    score = idf_sum / min(len(ents_a), len(ents_b))

                    if score < bridge_threshold:
                        continue

                    edges.append({
                        "source_node_id": str(node_a.id),
                        "target_node_id": str(node_b.id),
                        "source_title": node_a.trend_title,
                        "target_title": node_b.trend_title,
                        "source_parent_id": parent_a,
                        "target_parent_id": parent_b,
                        "strength": round(score, 4),
                        "bridge_entities": sorted(specific)[:10],
                        "relationship": "cross-trend-bridge",
                    })

    edges.sort(key=lambda e: e["strength"], reverse=True)
    edges = edges[:max_correlations]

    if edges:
        logger.info(
            f"Sub-trend correlation: {len(edges)} cross-parent bridges "
            f"from {n_sub} sub-trends across {len(parents_with_subs)} parents"
        )

    return edges
