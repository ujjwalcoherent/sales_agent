"""
Background news collector -- accumulates company-specific news in ChromaDB.

Runs:
1. After pipeline completion -- collects news for all discovered companies
2. On company detail page visit -- collects news for that specific company
3. On-demand via API endpoint -- manual trigger

Source strategy (two tiers):
  Fresh (last 7 days):  Google News RSS + Tavily (via web_intel.company_news)
  Historical (1-5 mo):  gnews library (proper date-range support)

All articles pass through an LLM-lite relevance filter before storage to ensure
they are genuinely about the target company (not just tangential mentions).
Content extraction uses trafilatura (inside web_intel.extract).
Storage: ChromaDB ``articles_news`` collection with company_name metadata.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Module-level semaphore to limit concurrent collection across all callers
_COLLECT_SEM = asyncio.Semaphore(3)


async def collect_company_news(
    company_name: str,
    months_back: int = 1,
    max_articles: int = 50,
    domain: str = "",
    industry: str = "",
) -> int:
    """Collect news for a single company and store in ChromaDB.

    Fetches fresh news via Google News RSS + Tavily, and historical news via
    the gnews library.  All articles are relevance-filtered by LLM before
    storage.

    Args:
        company_name: Company to collect news for.
        months_back: How far back to look (1-6 months).
        max_articles: Maximum total articles to fetch across all sources.
        domain: Company domain for relevance disambiguation.
        industry: Company industry for relevance disambiguation.

    Returns:
        Number of new articles stored (after dedup + relevance filter).
    """
    async with _COLLECT_SEM:
        try:
            return await _collect_impl(company_name, months_back, max_articles, domain, industry)
        except Exception as e:
            logger.debug(f"news_collector: {company_name} failed: {e}")
            return 0


_article_cache = None

def _get_article_cache():
    """Lazy singleton — avoids creating a new ChromaDB client per call."""
    global _article_cache
    if _article_cache is None:
        from app.tools.article_cache import ArticleCache
        _article_cache = ArticleCache()
    return _article_cache


async def _collect_impl(
    company_name: str,
    months_back: int,
    max_articles: int,
    domain: str = "",
    industry: str = "",
) -> int:
    """Internal implementation -- runs inside the semaphore.

    Two-source strategy:
      1. Fresh news (last 7 days) via web_intel.company_news (RSS + Tavily)
      2. Historical news (1-N months) via web_intel.company_news_gnews
    Then: relevance filter → content extraction → store in ChromaDB.
    """
    from app.tools.web.web_intel import (
        company_news,
        company_news_gnews,
        filter_relevant_articles,
        extract,
    )

    cache = _get_article_cache()

    # ── Fetch from both sources in parallel ──────────────────────
    fresh_per = max(5, max_articles // 3)
    hist_per_month = max(5, (max_articles * 2 // 3) // max(months_back, 1))

    fresh_task = asyncio.create_task(
        company_news(company_name, max_articles=fresh_per, extract_content=False)
    )
    hist_task = asyncio.create_task(
        company_news_gnews(company_name, months_back=months_back, max_per_month=hist_per_month)
    )

    fresh_articles, hist_articles = await asyncio.gather(
        fresh_task, hist_task, return_exceptions=True,
    )

    if isinstance(fresh_articles, Exception):
        logger.debug(f"news_collector: {company_name} fresh failed: {fresh_articles}")
        fresh_articles = []
    if isinstance(hist_articles, Exception):
        logger.debug(f"news_collector: {company_name} historical failed: {hist_articles}")
        hist_articles = []

    # ── Dedup by title across both sources ───────────────────────
    seen_titles: set[str] = set()
    combined = []
    for a in list(fresh_articles) + list(hist_articles):
        if not a.url and not a.title:
            continue
        title_key = (a.title or "").lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        combined.append(a)

    if not combined:
        return 0

    # ── LLM relevance filter ────────────────────────────────────
    relevant = await filter_relevant_articles(company_name, combined, domain=domain, industry=industry)

    # ── Content extraction for articles without content ─────────
    async def _extract_one(article):
        if not article.content and article.url:
            text = await extract(article.url, timeout=12.0)
            if text:
                article.content = text
                if not article.summary:
                    from app.tools.web.web_intel import _summarize_text
                    article.summary = _summarize_text(text, sentences=3)

    extract_tasks = [_extract_one(a) for a in relevant[:max_articles]]
    await asyncio.gather(*extract_tasks, return_exceptions=True)

    # ── Store in ChromaDB ───────────────────────────────────────
    stored = 0
    for article in relevant[:max_articles]:
        if not article.url:
            continue
        if cache.url_exists(article.url):
            # Also check by title for gnews URLs (opaque Google News redirects)
            continue
        ok = cache.store_news_article(
            title=article.title,
            url=article.url,
            source_name=article.source_name or "",
            published_at=(
                article.published_at.isoformat()
                if article.published_at
                else ""
            ),
            content=article.content or "",
            summary=article.summary or "",
            company_name=company_name,
        )
        if ok:
            stored += 1

    if stored:
        logger.info(f"news_collector: {company_name} -- {stored} new articles stored")
    return stored
