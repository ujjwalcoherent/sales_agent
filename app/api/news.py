"""News feed router — exposes ArticleCache as a browsable news feed.

Data source: ChromaDB article cache (4-5 months of articles from RSS pipeline).
No new data collection — just exposing what we already have with pagination + filters.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Query

from app.api.schemas import NewsArticleResponse, NewsListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_article_cache():
    from app.tools.article_cache import ArticleCache
    return ArticleCache()


def _meta_to_response(meta: dict, article_id: str = "") -> NewsArticleResponse:
    """Convert ChromaDB metadata to NewsArticleResponse."""
    entity_names = []
    if "entity_names_json" in meta:
        try:
            entity_names = json.loads(meta["entity_names_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    keywords = []
    if "keywords_json" in meta:
        try:
            keywords = json.loads(meta["keywords_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    content = meta.get("content", "")
    content_preview = content[:300] if content else (meta.get("summary", "") or "")[:300]

    return NewsArticleResponse(
        id=article_id,
        title=meta.get("title", ""),
        summary=meta.get("summary", ""),
        url=meta.get("url", ""),
        source_name=meta.get("source_name", ""),
        source_type=meta.get("source_type", ""),
        source_credibility=meta.get("source_credibility", 0.0),
        published_at=meta.get("published_at", ""),
        sentiment_score=meta.get("sentiment_score", 0.0),
        entity_names=entity_names[:10],
        keywords=keywords[:10],
        content_preview=content_preview,
    )


@router.get("", response_model=NewsListResponse)
async def list_news(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    source: Optional[str] = Query(None, description="Filter by source name"),
    search: Optional[str] = Query(None, description="Semantic search query"),
    date_from: Optional[str] = Query(None, description="ISO date (e.g. 2026-01-01)"),
    date_to: Optional[str] = Query(None, description="ISO date (e.g. 2026-02-28)"),
):
    """List news articles from the article cache.

    - No search param: all articles sorted newest first
    - With search: semantic search via ChromaDB
    - source/date filters applied post-query
    """
    cache = _get_article_cache()
    count = cache.collection.count()

    if count == 0:
        return NewsListResponse(articles=[], total=0, page=page, per_page=per_page, sources=[])

    articles: list[tuple[dict, str]] = []  # (metadata, id)
    all_sources: set[str] = set()

    if search:
        # Semantic search
        results = cache.collection.query(
            query_texts=[search],
            n_results=min(count, 500),
            include=["metadatas"],
        )
        if results and results.get("metadatas"):
            ids = results.get("ids", [[]])[0]
            for i, meta in enumerate(results["metadatas"][0]):
                all_sources.add(meta.get("source_name", "unknown"))
                aid = ids[i] if i < len(ids) else ""
                articles.append((meta, aid))
    else:
        # Load all articles
        results = cache.collection.get(include=["metadatas"], limit=count)
        if results and results.get("metadatas"):
            ids = results.get("ids", [])
            for i, meta in enumerate(results["metadatas"]):
                all_sources.add(meta.get("source_name", "unknown"))
                aid = ids[i] if i < len(ids) else ""
                articles.append((meta, aid))

    # Apply filters
    filtered = []
    for meta, aid in articles:
        if source and meta.get("source_name", "") != source:
            continue
        pub = meta.get("published_at", "")
        if date_from and pub and pub < date_from:
            continue
        if date_to and pub and pub > date_to + "T23:59:59":
            continue
        filtered.append((meta, aid))

    # Sort newest first
    filtered.sort(key=lambda x: x[0].get("published_at", ""), reverse=True)

    total = len(filtered)
    start_idx = (page - 1) * per_page
    page_items = filtered[start_idx : start_idx + per_page]

    response_articles = [_meta_to_response(meta, aid) for meta, aid in page_items]

    # Auto-save mock data on first page load
    if page == 1 and not search and response_articles:
        try:
            from app.api.mock_loader import save_news_mock
            save_news_mock([a.model_dump() for a in response_articles[:100]], total)
        except Exception:
            pass

    return NewsListResponse(
        articles=response_articles,
        total=total,
        page=page,
        per_page=per_page,
        sources=sorted(all_sources),
    )


@router.get("/stats")
async def news_stats():
    """Article cache statistics — total count, sources, date range."""
    cache = _get_article_cache()
    stats = cache.get_stats()
    return {
        "total_articles": stats.get("count", 0),
        "sources": stats.get("sources", {}),
        "date_range": stats.get("date_range"),
    }
