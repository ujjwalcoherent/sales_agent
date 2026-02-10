"""
Search interest signal — validates detected trends against Google Trends.

NOT just an API wrapper. This signal answers a crucial sales question:
  "Are people ACTIVELY SEARCHING for this topic right now?"

WHY THIS MATTERS FOR SALES:
  A trend about "RBI repo rate change" is 10x more actionable if thousands
  of people are googling it RIGHT NOW. Prospects are actively thinking about
  it, making them more receptive to outreach about related services.

HOW IT WORKS:
  1. Fetch today's trending searches from Google Trends RSS (free, unlimited)
  2. Compare using BOTH keyword overlap AND embedding similarity
  3. Compute a search_interest_score (0.0 - 1.0) based on best match
  4. This feeds into the composite actionability score

DATA SOURCE: Google Trends RSS (trends.google.com/trending/rss)
  - Free, no API key, no rate limits
  - Updates every ~15 minutes
  - Returns top 10-20 trending topics per country
  - Includes approximate traffic volume

MATCHING STRATEGY (v2 — hybrid keyword + semantic):
  - Keyword overlap catches exact term matches (fast, precise)
  - Embedding similarity catches semantic matches the keywords miss:
    "RBI monetary policy" matches Google Trend "repo rate" even without
    shared words, because the embedding model understands the connection.
  - Final score = max(keyword_score, semantic_score), boosted by traffic

PERFORMANCE: <1 sec (single HTTP request cached 15 min, embedding cached)
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Cache: avoid hitting Google Trends more than once per 15 min
_cache: Dict[str, Any] = {"data": None, "timestamp": 0}
_CACHE_TTL = 900  # 15 minutes

# Cache for trending topic embeddings (recomputed when RSS data changes)
_embedding_cache: Dict[str, Any] = {"embeddings": None, "texts": None, "timestamp": 0}


def _get_geo_code() -> str:
    """Get geo code from env-configurable settings. Falls back to 'IN'."""
    try:
        from app.config import get_settings
        return get_settings().country_code or "IN"
    except Exception:
        return "IN"


def _fetch_google_trending(geo_code: str = "") -> List[Dict[str, Any]]:
    """
    Fetch today's trending searches from Google Trends RSS.

    T7: Now uses configurable geo_code instead of hardcoded 'IN'.
    Cache key includes geo_code so different geos don't conflict.
    """
    if not geo_code:
        geo_code = _get_geo_code()

    now = time.time()
    cache_key = f"data_{geo_code}"
    if _cache.get(cache_key) and (now - _cache.get(f"ts_{geo_code}", 0)) < _CACHE_TTL:
        return _cache[cache_key]

    try:
        import httpx
        import feedparser

        url = f"https://trends.google.com/trending/rss?geo={geo_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"
        }

        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=10)
        if r.status_code != 200:
            logger.debug(f"Google Trends RSS: HTTP {r.status_code} (geo={geo_code})")
            return _cache.get(cache_key) or []

        feed = feedparser.parse(r.text)
        topics = []
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            traffic_str = entry.get("ht_approx_traffic", "0")
            # Parse traffic: "20,000+" → 20000
            traffic = int(traffic_str.replace(",", "").replace("+", "") or "0")
            news_title = entry.get("ht_news_item_title", "")

            topics.append({
                "topic": title.lower(),
                "topic_words": set(title.lower().split()),
                "traffic": traffic,
                "news": news_title,
                # Full text for embedding: combine topic + associated news title
                "embed_text": f"{title}. {news_title}" if news_title else title,
            })

        _cache[cache_key] = topics
        _cache[f"ts_{geo_code}"] = now
        # Invalidate embedding cache when RSS data changes
        _embedding_cache["embeddings"] = None
        logger.debug(f"Google Trends: {len(topics)} trending topics (geo={geo_code})")
        return topics

    except Exception as e:
        logger.debug(f"Google Trends fetch failed (geo={geo_code}): {e}")
        return _cache.get(cache_key) or []


def _get_trending_embeddings(trending: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Get (cached) embeddings for trending topics. Returns normalized embeddings."""
    if _embedding_cache["embeddings"] is not None:
        return _embedding_cache["embeddings"]

    try:
        from app.tools.embeddings import EmbeddingTool
        tool = EmbeddingTool()
        texts = [t["embed_text"] for t in trending]
        raw = np.array(tool.embed_batch(texts))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1
        _embedding_cache["embeddings"] = raw / norms
        _embedding_cache["texts"] = texts
        return _embedding_cache["embeddings"]
    except Exception as e:
        logger.debug(f"Trending topic embedding failed: {e}")
        return None


def compute_search_interest(
    cluster_keywords: List[str],
    cluster_title: str = "",
) -> Dict[str, Any]:
    """
    Compute search interest signal for a single cluster.

    Uses HYBRID matching: keyword overlap + embedding cosine similarity.
    This catches both exact matches ("RBI" in both) and semantic matches
    ("monetary policy tightening" ~ "repo rate hike").

    Args:
        cluster_keywords: Top keywords from c-TF-IDF for this cluster.
        cluster_title: LLM-generated trend title (if available).

    Returns:
        Dict with:
          - search_interest_score: 0.0-1.0
          - matching_trends: list of matching Google Trends topics
          - max_traffic: highest traffic from matching trends
    """
    trending = _fetch_google_trending()
    if not trending:
        return {
            "search_interest_score": 0.0,
            "matching_trends": [],
            "max_traffic": 0,
        }

    # ── Strategy 1: Keyword overlap (fast, catches exact matches) ──
    our_terms: Set[str] = set()
    for kw in cluster_keywords:
        for word in kw.lower().replace("_", " ").split():
            if len(word) >= 3:
                our_terms.add(word)
    for word in cluster_title.lower().split():
        if len(word) >= 3:
            our_terms.add(word)

    keyword_matches = []
    if our_terms:
        for topic in trending:
            overlap = our_terms & topic["topic_words"]
            if overlap:
                overlap_ratio = len(overlap) / max(len(topic["topic_words"]), 1)
                keyword_matches.append({
                    "topic": topic["topic"],
                    "overlap": list(overlap),
                    "score": overlap_ratio,
                    "traffic": topic["traffic"],
                    "method": "keyword",
                })

    # ── Strategy 2: Embedding similarity (catches semantic matches) ──
    semantic_matches = []
    trend_embs = _get_trending_embeddings(trending)
    if trend_embs is not None:
        try:
            from app.tools.embeddings import EmbeddingTool
            tool = EmbeddingTool()
            # Build cluster text: title + top keywords
            cluster_text = f"{cluster_title}. {', '.join(cluster_keywords[:8])}"
            cluster_emb = np.array(tool.embed_batch([cluster_text]))
            norm = np.linalg.norm(cluster_emb, axis=1, keepdims=True)
            norm[norm == 0] = 1
            cluster_emb_norm = cluster_emb / norm

            # Cosine similarity against all trending topics
            sims = np.dot(cluster_emb_norm, trend_embs.T).flatten()
            # Threshold: 0.35 for semantic match (lower than clustering threshold
            # because trending topics are short, ~2-5 words)
            for i, sim in enumerate(sims):
                if sim >= 0.35:
                    semantic_matches.append({
                        "topic": trending[i]["topic"],
                        "score": float(sim),
                        "traffic": trending[i]["traffic"],
                        "method": "semantic",
                    })
        except Exception as e:
            logger.debug(f"Semantic search interest matching failed: {e}")

    # ── Combine: take best score from either method ──
    all_matches = keyword_matches + semantic_matches
    if not all_matches:
        return {"search_interest_score": 0.0, "matching_trends": [], "max_traffic": 0}

    best_match = max(all_matches, key=lambda m: m["score"])
    max_traffic = max(m["traffic"] for m in all_matches)

    # Traffic boost: log-scaled (100 → 0.3, 10000 → 0.6, 100000 → 0.9)
    traffic_boost = min(0.9, math.log10(max(max_traffic, 1) + 1) / 6)

    # Final score: match quality + traffic volume
    score = min(1.0, best_match["score"] * 0.6 + traffic_boost * 0.4)

    # Deduplicate topic names for output
    seen = set()
    unique_topics = []
    for m in sorted(all_matches, key=lambda x: x["score"], reverse=True):
        if m["topic"] not in seen:
            seen.add(m["topic"])
            unique_topics.append(m["topic"])
        if len(unique_topics) >= 3:
            break

    return {
        "search_interest_score": round(score, 3),
        "matching_trends": unique_topics,
        "max_traffic": max_traffic,
    }


def compute_search_signals_batch(
    cluster_keywords_map: Dict[int, List[str]],
    cluster_titles: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute search interest for all clusters in one batch.

    EFFICIENCY: Embeds ALL cluster texts in ONE call (not one per cluster).
    Trending topic embeddings are computed once and cached across calls.
    """
    trending = _fetch_google_trending()
    if not trending:
        return {cid: {"search_interest_score": 0.0, "matching_trends": [], "max_traffic": 0}
                for cid in cluster_keywords_map}

    trend_embs = _get_trending_embeddings(trending)
    titles = cluster_titles or {}

    # Pre-compute ALL cluster embeddings in one batched call
    cids = list(cluster_keywords_map.keys())
    cluster_embs_norm = None
    if trend_embs is not None and cids:
        try:
            from app.tools.embeddings import EmbeddingTool
            tool = EmbeddingTool()
            cluster_texts = [
                f"{titles.get(cid, '')}. {', '.join(cluster_keywords_map[cid][:8])}"
                for cid in cids
            ]
            raw = np.array(tool.embed_batch(cluster_texts))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1
            cluster_embs_norm = raw / norms
            # Compute all similarities at once: (n_clusters, n_trends)
            all_sims = np.dot(cluster_embs_norm, trend_embs.T)
        except Exception as e:
            logger.debug(f"Batch cluster embedding failed: {e}")
            all_sims = None
    else:
        all_sims = None

    results = {}
    for idx, cid in enumerate(cids):
        keywords = cluster_keywords_map[cid]
        title = titles.get(cid, "")

        # ── Keyword matches ──
        our_terms: Set[str] = set()
        for kw in keywords:
            for word in kw.lower().replace("_", " ").split():
                if len(word) >= 3:
                    our_terms.add(word)
        for word in title.lower().split():
            if len(word) >= 3:
                our_terms.add(word)

        keyword_matches = []
        if our_terms:
            for topic in trending:
                overlap = our_terms & topic["topic_words"]
                if overlap:
                    overlap_ratio = len(overlap) / max(len(topic["topic_words"]), 1)
                    keyword_matches.append({
                        "topic": topic["topic"],
                        "score": overlap_ratio,
                        "traffic": topic["traffic"],
                    })

        # ── Semantic matches (from pre-computed batch) ──
        semantic_matches = []
        if all_sims is not None:
            sims = all_sims[idx]
            for j, sim in enumerate(sims):
                if sim >= 0.35:
                    semantic_matches.append({
                        "topic": trending[j]["topic"],
                        "score": float(sim),
                        "traffic": trending[j]["traffic"],
                    })

        # ── Combine ──
        all_matches = keyword_matches + semantic_matches
        if not all_matches:
            results[cid] = {"search_interest_score": 0.0, "matching_trends": [], "max_traffic": 0}
            continue

        best_match = max(all_matches, key=lambda m: m["score"])
        max_traffic = max(m["traffic"] for m in all_matches)
        traffic_boost = min(0.9, math.log10(max(max_traffic, 1) + 1) / 6)
        score = min(1.0, best_match["score"] * 0.6 + traffic_boost * 0.4)

        seen = set()
        unique_topics = []
        for m in sorted(all_matches, key=lambda x: x["score"], reverse=True):
            if m["topic"] not in seen:
                seen.add(m["topic"])
                unique_topics.append(m["topic"])
            if len(unique_topics) >= 3:
                break

        results[cid] = {
            "search_interest_score": round(score, 3),
            "matching_trends": unique_topics,
            "max_traffic": max_traffic,
        }

    return results
