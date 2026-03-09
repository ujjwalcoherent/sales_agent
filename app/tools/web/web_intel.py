"""Unified web intelligence module — search, extract, news, deep scrape.

Every function is async-safe and uses bounded concurrency via module-level semaphores.

Source hierarchy:
  Search:     Tavily (primary, advanced) -> DuckDuckGo (free fallback)
  Extract:    trafilatura (fast, local, F1=0.958) -> Jina Reader (JS-heavy) -> empty
  News fresh: Google News RSS + Tavily news (parallel, merged, deduped)
  News hist:  gnews library (proper date-range support, 1-12 months back)
  Relevance:  LLM-lite batch classifier (auto-accept if name in title, LLM for ambiguous)
  Deep:       ScrapeGraphAI (SearchGraph — background only, 30-80s)

Usage:
    from app.tools.web.web_intel import (
        search, extract, company_news, company_news_gnews,
        filter_relevant_articles, deep_company_search,
    )

    results = await search("NVIDIA AI chips 2026", max_results=5)
    html_text = await extract("https://example.com/article")
    news = await company_news("Tesla", max_articles=10)           # last 7 days
    hist = await company_news_gnews("Tesla", months_back=5)       # 5 months
    relevant = await filter_relevant_articles("Tesla", hist)      # LLM filter
    profile = await deep_company_search("Infosys")
    companies = await search_industry_companies("fintech")
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Union
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Country → Google News region mapping ─────────────────────────

_COUNTRY_TO_REGION = {
    "India": "en-IN",
    "United States": "en-US",
    "United Kingdom": "en-GB",
    "Canada": "en-CA",
    "Australia": "en-AU",
    "Germany": "de-DE",
    "France": "fr-FR",
    "Japan": "ja-JP",
    "Singapore": "en-SG",
}

_COUNTRY_TO_CODE = {
    "India": "IN",
    "United States": "US",
    "United Kingdom": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "Germany": "DE",
    "France": "FR",
    "Japan": "JP",
    "Singapore": "SG",
}


def _get_news_region() -> tuple[str, str]:
    """Return (region, country_code) from settings, e.g. ('en-IN', 'IN')."""
    from app.config import get_settings
    country = get_settings().country
    region = _COUNTRY_TO_REGION.get(country, "en-US")
    code = _COUNTRY_TO_CODE.get(country, "US")
    return region, code


# ── Module-level semaphores (global concurrency control) ─────────

_SEARCH_SEM = asyncio.Semaphore(5)
_EXTRACT_SEM = asyncio.Semaphore(10)
_NEWS_SEM = asyncio.Semaphore(3)

# ── Content limits ───────────────────────────────────────────────

_MAX_CONTENT_CHARS = 50_000
_MIN_CONTENT_CHARS = 100


# ══════════════════════════════════════════════════════════════════
# Result Models
# ══════════════════════════════════════════════════════════════════


class SearchResult(BaseModel):
    """A single web search result with optional extracted content."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""  # "tavily", "ddgs"
    content: str = ""  # Full extracted text (when extract_content=True)
    content_length: int = 0


class NewsArticle(BaseModel):
    """A news article with optional extracted content and summary."""

    title: str = ""
    url: str = ""
    source_name: str = ""
    published_at: Optional[datetime] = None
    content: str = ""
    summary: str = ""


class CompanyProfile(BaseModel):
    """Full company profile from deep search. 20+ fields."""

    company_name: str = ""
    also_known_as: list[str] = Field(default_factory=list)
    industry: str = ""
    sub_industries: list[str] = Field(default_factory=list)
    description: str = ""
    founded_year: Optional[int] = None
    headquarters: str = ""
    ceo: str = ""
    key_people: list[str] = Field(default_factory=list)
    employee_count: Optional[str] = None
    revenue: str = ""
    market_cap: str = ""
    stock_ticker: str = ""
    website: str = ""
    domain: str = ""
    products_services: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    subsidiaries: list[str] = Field(default_factory=list)
    funding_stage: str = ""
    total_funding: str = ""
    investors: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# 1. search()
# ══════════════════════════════════════════════════════════════════


async def search(
    query: str,
    max_results: int = 10,
    extract_content: bool = False,
    news_mode: bool = False,
    region: str = "wt-wt",
) -> list[SearchResult]:
    """Web search with Tavily primary, DuckDuckGo free fallback.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        extract_content: If True, extract full text from each result URL in parallel.
        news_mode: If True, use news-specific search.
        region: Region code for DuckDuckGo fallback (e.g. "us-en", "wt-wt").

    Returns:
        List of SearchResult with optional extracted content.
    """
    # Primary: Tavily
    results = await _tavily_search(query, max_results=max_results, news_mode=news_mode)

    # Fallback: DuckDuckGo (free, no API key needed)
    if not results:
        results = await _ddgs_search(query, max_results=max_results, news_mode=news_mode, region=region)

    if extract_content and results:
        results = await _enrich_with_content(results)

    return results


async def _tavily_search(
    query: str,
    max_results: int = 10,
    news_mode: bool = False,
) -> list[SearchResult]:
    """Tavily primary search — returns SearchResult list for unified interface."""
    try:
        from .tavily_tool import TavilyTool

        tavily = TavilyTool()
        if not tavily.available:
            return []

        if news_mode:
            result = await tavily.news_search(
                query=query,
                max_results=max_results,
                include_domains=[],  # all domains for general search
            )
        else:
            result = await tavily.search(
                query=query,
                max_results=max_results,
                include_answer=False,
            )

        if result.get("error"):
            logger.debug(f"Tavily search error: {result['error']}")
            return []

        results: list[SearchResult] = []
        for r in result.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source="tavily",
            ))

        logger.info(f"Tavily returned {len(results)} results for '{query[:50]}'")
        return results

    except Exception as e:
        logger.warning(f"Tavily search failed for '{query[:50]}': {e}")
        return []


async def _ddgs_search(
    query: str,
    max_results: int = 10,
    news_mode: bool = False,
    region: str = "wt-wt",
) -> list[SearchResult]:
    """DuckDuckGo free fallback. Synchronous lib, wrapped in to_thread."""
    async with _SEARCH_SEM:
        try:
            from ddgs import DDGS

            def _do_search() -> list[dict]:
                with DDGS() as ddgs:
                    if news_mode:
                        return list(ddgs.news(query, region=region, max_results=max_results))
                    return list(ddgs.text(query, region=region, max_results=max_results))

            raw = await asyncio.to_thread(_do_search)
            if not raw:
                return []

            results: list[SearchResult] = []
            for r in raw:
                if news_mode:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("body", ""),
                        source="ddgs",
                    ))
                else:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="ddgs",
                    ))

            logger.info(f"DDGS returned {len(results)} results for '{query[:50]}'")
            return results

        except Exception as e:
            logger.warning(f"DDGS search failed for '{query[:50]}': {e}")
            return []


async def _enrich_with_content(results: list[SearchResult]) -> list[SearchResult]:
    """Extract full-text content for each result URL in parallel."""
    urls = [r.url for r in results if r.url]

    async def _extract_one(idx: int, url: str) -> None:
        text = await extract(url)
        if text:
            results[idx].content = text
            results[idx].content_length = len(text)

    tasks = [_extract_one(i, url) for i, url in enumerate(urls)]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


# ══════════════════════════════════════════════════════════════════
# 2. extract()
# ══════════════════════════════════════════════════════════════════


async def extract(url: str, timeout: float = 15.0) -> str:
    """Extract article text from a URL. Multi-layer: trafilatura -> Jina Reader -> empty.

    Args:
        url: The URL to extract content from.
        timeout: Max time per extraction attempt (seconds).

    Returns:
        Extracted text (truncated to 50k chars), or empty string on failure.
    """
    if not url:
        return ""

    # Layer 1: trafilatura (fast, local, high accuracy)
    text = await _extract_trafilatura(url, timeout=timeout)
    if text and len(text) >= _MIN_CONTENT_CHARS:
        return text[:_MAX_CONTENT_CHARS]

    # Layer 2: Jina Reader (handles JS-heavy pages)
    text = await _extract_jina(url, timeout=timeout)
    if text and len(text) >= _MIN_CONTENT_CHARS:
        return text[:_MAX_CONTENT_CHARS]

    # Layer 3: Graceful failure
    return ""


async def _extract_trafilatura(url: str, timeout: float = 15.0) -> str:
    """Extract text using trafilatura. Synchronous lib, wrapped in to_thread."""
    async with _EXTRACT_SEM:
        try:
            import trafilatura

            def _do_extract() -> str:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    return ""
                result = trafilatura.extract(
                    downloaded,
                    url=url,
                    include_tables=True,
                    include_links=False,
                    favor_precision=True,
                )
                return result or ""

            return await asyncio.wait_for(
                asyncio.to_thread(_do_extract),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(f"trafilatura timed out for {url[:80]}")
            return ""
        except Exception as e:
            logger.debug(f"trafilatura failed for {url[:80]}: {e}")
            return ""


async def _extract_jina(url: str, timeout: float = 15.0) -> str:
    """Extract text via Jina Reader API (handles JS-rendered pages)."""
    async with _EXTRACT_SEM:
        try:
            jina_url = f"https://r.jina.ai/{url}"
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(
                    jina_url,
                    headers={
                        "Accept": "text/plain",
                        "User-Agent": "SalesAgent/1.0",
                    },
                )
                if resp.status_code == 200:
                    text = resp.text.strip()
                    return text
                logger.debug(f"Jina Reader returned {resp.status_code} for {url[:80]}")
                return ""
        except Exception as e:
            logger.debug(f"Jina Reader failed for {url[:80]}: {e}")
            return ""


# ══════════════════════════════════════════════════════════════════
# 3. company_news()
# ══════════════════════════════════════════════════════════════════

# Noise domains + URL patterns to exclude from ALL news sources
_NOISE_DOMAINS = {
    "youtube.com", "youtu.be", "tiktok.com", "instagram.com",
    "facebook.com", "fb.com", "pinterest.com", "reddit.com",
    "twitter.com", "x.com", "linkedin.com",
    "stockanalysis.com", "tradingview.com", "seekingalpha.com",
    "glassdoor.com", "indeed.com", "ambitionbox.com",
}

_NOISE_TITLE_PATTERNS = re.compile(
    r"(stock price|share price|buy or sell|quarterly results|eps estimate|"
    r"stock forecast|stock analysis|dividend|target price|rating reiterated|"
    r"job opening|hiring|career|glassdoor|salary|interview question|"
    r"cookie policy|privacy policy|terms of service|subscribe now|"
    r"download app|scan qr|promo code|coupon|discount)", re.I,
)


def _prefilter_articles(articles: list) -> list:
    """Quick noise rejection before expensive LLM relevance filter.

    Removes: junk domains, stock/job noise, too-short titles, non-English.
    """
    clean = []
    for a in articles:
        url = (a.url or "").lower()
        title = (a.title or "").strip()

        # Skip articles with no title or very short titles
        if len(title) < 15:
            continue

        # Skip noise domains
        if any(d in url for d in _NOISE_DOMAINS):
            continue

        # Skip stock/job/cookie noise by title pattern
        if _NOISE_TITLE_PATTERNS.search(title):
            continue

        clean.append(a)
    return clean


async def company_news(
    company_name: str,
    max_articles: int = 20,
    time_range: str = "when:7d",
    region: str = "",
    country: str = "",
    extract_content: bool = True,
    domain: str = "",
    industry: str = "",
) -> list[NewsArticle]:
    """Fetch recent company news from Google News RSS + Tavily news in parallel.

    Merges and deduplicates results by URL, preferring Google News RSS
    (higher volume) with Tavily providing additional coverage.

    Args:
        company_name: Company to search for.
        max_articles: Max articles to return.
        time_range: Google News time filter. Supports "when:7d" (days) or
            "after:YYYY-MM-DD before:YYYY-MM-DD" (monthly windows).
        region: Language-region code (e.g. "en-US", "en-IN").
        country: Country code (e.g. "US", "IN").
        extract_content: If True, extract full article text via extract().

    Returns:
        List of NewsArticle with optional content and summary.
    """
    if not region or not country:
        default_region, default_code = _get_news_region()
        region = region or default_region
        country = country or default_code

    # Run both sources in parallel
    rss_task = asyncio.create_task(_google_news_rss(company_name, max_articles, time_range, region, country))
    tavily_task = asyncio.create_task(_tavily_news(company_name, max_articles=min(max_articles, 10)))

    rss_articles, tavily_articles = await asyncio.gather(
        rss_task, tavily_task, return_exceptions=True,
    )

    # Handle exceptions gracefully
    if isinstance(rss_articles, Exception):
        logger.debug(f"Google News RSS failed for '{company_name}': {rss_articles}")
        rss_articles = []
    if isinstance(tavily_articles, Exception):
        logger.debug(f"Tavily news failed for '{company_name}': {tavily_articles}")
        tavily_articles = []

    # Merge, deduplicate by URL, and pre-filter noise
    merged = _merge_dedup_articles(rss_articles, tavily_articles, max_articles * 2)
    merged = _prefilter_articles(merged)[:max_articles]

    if extract_content and merged:
        merged = await _enrich_articles(merged)

    return merged


async def _google_news_rss(
    company_name: str,
    max_articles: int = 20,
    time_range: str = "when:7d",
    region: str = "",
    country: str = "",
) -> list[NewsArticle]:
    """Google News RSS feed — high volume, free."""
    if not region or not country:
        default_region, default_code = _get_news_region()
        region = region or default_region
        country = country or default_code

    async with _NEWS_SEM:
        try:
            import feedparser

            lang = region.split("-")[0] if "-" in region else "en"
            # Quote multi-word names for exact phrase match in search
            # intitle: forces the name to appear in article headline (reduces noise)
            # B3: Add negative keywords to exclude stock/job noise at fetch time
            noise_exclude = "-jobs -hiring -salary -stock+price -share+price"
            if " " in company_name:
                name_query = f'intitle:"{company_name}" {noise_exclude}'
            else:
                name_query = f'"{company_name}" {noise_exclude}'
            encoded_query = quote_plus(f"{name_query} {time_range}")
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded_query}"
                f"&hl={region}&gl={country}&ceid={country}:{lang}"
            )

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(rss_url, headers={
                    "User-Agent": "SalesAgent/1.0 (news aggregator)",
                })
                if resp.status_code != 200:
                    logger.warning(f"Google News RSS returned {resp.status_code} for '{company_name}'")
                    return []

            def _parse_feed() -> list[dict]:
                feed = feedparser.parse(resp.text)
                entries = []
                for entry in feed.entries[:max_articles]:
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6])
                        except Exception:
                            pass
                    entries.append({
                        "title": getattr(entry, "title", ""),
                        "url": getattr(entry, "link", ""),
                        "source_name": _extract_source_from_title(getattr(entry, "title", "")),
                        "published_at": published,
                    })
                return entries

            parsed = await asyncio.to_thread(_parse_feed)
            return [
                NewsArticle(
                    title=e["title"],
                    url=e["url"],
                    source_name=e["source_name"],
                    published_at=e["published_at"],
                )
                for e in parsed
            ]

        except Exception as e:
            logger.warning(f"Google News RSS failed for '{company_name}': {e}")
            return []


async def _tavily_news(company_name: str, max_articles: int = 10) -> list[NewsArticle]:
    """Tavily news search — AI-curated, covers sources Google News may miss."""
    try:
        from .tavily_tool import TavilyTool

        tavily = TavilyTool()
        if not tavily.available:
            return []

        # Quote multi-word names for exact phrase match
        name_query = f'"{company_name}"' if " " in company_name else company_name
        result = await tavily.news_search(
            query=f"{name_query} company news",
            time_range="month",
            max_results=max_articles,
            include_domains=[],  # all domains
        )

        if result.get("error"):
            return []

        articles: list[NewsArticle] = []
        for r in result.get("results", []):
            published = None
            if r.get("published_date"):
                try:
                    published = datetime.fromisoformat(r["published_date"].replace("Z", "+00:00"))
                except Exception:
                    pass

            articles.append(NewsArticle(
                title=r.get("title", ""),
                url=r.get("url", ""),
                source_name="tavily",
                published_at=published,
                summary=r.get("content", "")[:300] if r.get("content") else "",
            ))

        return articles

    except Exception as e:
        logger.debug(f"Tavily news failed for '{company_name}': {e}")
        return []


def _normalize_url(url: str) -> str:
    """Strip tracking params for dedup. Keep domain + path."""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed = urlparse(url.rstrip("/").lower())
    # Strip common tracking params
    clean_params = {
        k: v for k, v in parse_qs(parsed.query).items()
        if not k.startswith(("utm_", "ref", "source", "campaign", "fbclid", "gclid"))
    }
    cleaned = parsed._replace(
        query=urlencode(clean_params, doseq=True),
        fragment="",
    )
    return urlunparse(cleaned)


def _normalize_title(title: str) -> str:
    """Normalize title for near-duplicate detection.

    Strips source suffix, parentheticals, years, quarters, punctuation.
    """
    t = title.lower().strip()
    # Google News RSS appends " - Source Name"
    if " - " in t:
        t = t.rsplit(" - ", 1)[0]
    # B5: Strip parenthetical content, years, quarters for better fuzzy dedup
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\b20\d{2}\b", "", t)       # Strip years 2000-2029
    t = re.sub(r"\bq[1-4]\b", "", t)         # Strip Q1-Q4
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:80]


def _merge_dedup_articles(
    primary: list[NewsArticle],
    secondary: list[NewsArticle],
    max_articles: int = 20,
) -> list[NewsArticle]:
    """Merge two article lists, deduplicate by URL + title similarity, cap at max_articles."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    merged: list[NewsArticle] = []

    def _is_new(article: NewsArticle) -> bool:
        url_key = _normalize_url(article.url)
        title_key = _normalize_title(article.title or "")
        if url_key in seen_urls:
            return False
        if title_key and title_key in seen_titles:
            return False
        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        return True

    # Primary first (Google News RSS — higher volume)
    for article in primary:
        if _is_new(article):
            merged.append(article)

    # Then secondary (Tavily — additional coverage)
    for article in secondary:
        if _is_new(article):
            merged.append(article)

    return merged[:max_articles]


def _extract_source_from_title(title: str) -> str:
    """Google News RSS titles end with ' - Source Name'. Extract it."""
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return ""


# ══════════════════════════════════════════════════════════════════
# 3b. company_news_gnews() — historical news via gnews library
# ══════════════════════════════════════════════════════════════════


_GNEWS_SEM = asyncio.Semaphore(2)  # gnews wraps Google News; be gentle


async def company_news_gnews(
    company_name: str,
    months_back: int = 5,
    max_per_month: int = 20,
) -> list[NewsArticle]:
    """Fetch historical company news using the gnews library.

    The gnews library properly supports date ranges via start_date/end_date,
    unlike Google News RSS which only reliably supports ``when:7d``.

    Fetches one window per month going back ``months_back`` months, returning
    up to ``max_per_month`` articles per window.

    Args:
        company_name: Company to search for.
        months_back: How many months of history to fetch (1-12).
        max_per_month: Maximum articles per monthly window.

    Returns:
        List of NewsArticle across all monthly windows, newest first.
    """
    async with _GNEWS_SEM:
        try:
            return await asyncio.to_thread(
                _gnews_historical_sync, company_name, months_back, max_per_month
            )
        except Exception as e:
            logger.warning(f"gnews historical failed for '{company_name}': {e}")
            return []


def _gnews_historical_sync(
    company_name: str,
    months_back: int,
    max_per_month: int,
) -> list[NewsArticle]:
    """Synchronous gnews fetch across monthly windows. Runs in thread."""
    from gnews import GNews

    now = datetime.now()
    all_articles: list[NewsArticle] = []
    seen_titles: set[str] = set()  # dedup by title (gnews URLs are opaque)

    region, country_code = _get_news_region()
    lang = region.split("-")[0] if "-" in region else "en"

    for i in range(months_back):
        window_end = now - timedelta(days=i * 30)
        window_start = now - timedelta(days=(i + 1) * 30)

        gn = GNews(
            language=lang,
            country=country_code,
            start_date=(window_start.year, window_start.month, window_start.day),
            end_date=(window_end.year, window_end.month, window_end.day),
            max_results=max_per_month,
            exclude_websites=[
                # Social media
                "youtube.com", "tiktok.com", "instagram.com",
                "facebook.com", "pinterest.com", "reddit.com",
                "twitter.com", "x.com",
                # Stock/trading noise
                "stockanalysis.com", "tradingview.com", "seekingalpha.com",
                "fool.com", "zacks.com", "tipranks.com",
                # Job boards
                "glassdoor.com", "indeed.com", "ambitionbox.com",
                "naukri.com",
                # Low quality
                "wikipedia.org",
            ],
        )

        try:
            results = gn.get_news(company_name)
        except Exception as e:
            logger.debug(f"gnews window {i} failed for '{company_name}': {e}")
            continue

        if not results:
            continue

        for r in results:
            title = r.get("title", "")
            title_key = title.lower().strip()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            published = None
            if r.get("published date"):
                try:
                    from email.utils import parsedate_to_datetime
                    published = parsedate_to_datetime(r["published date"])
                except Exception:
                    pass

            source_name = ""
            pub = r.get("publisher", {})
            if isinstance(pub, dict):
                source_name = pub.get("title", "")

            all_articles.append(NewsArticle(
                title=title,
                url=r.get("url", ""),
                source_name=source_name,
                published_at=published,
                summary=r.get("description", ""),
            ))

    # Pre-filter noise before returning
    all_articles = _prefilter_articles(all_articles)
    logger.info(
        f"gnews: {company_name} — {len(all_articles)} articles across {months_back} months"
    )
    return all_articles


async def filter_relevant_articles(
    company_name: str,
    articles: list[NewsArticle],
    batch_size: int = 15,
    domain: str = "",
    industry: str = "",
) -> list[NewsArticle]:
    """LLM-lite relevance filter — keeps only articles genuinely about the company.

    Multi-layer filtering:
      1. Full name match → auto-accept
      2. Multi-word: >=60% significant words (min 2) → auto-accept
      3. Single-word: whole-word match (not substring) → auto-accept
      4. Everything else → LLM classify (with company context for disambiguation)

    Args:
        company_name: The target company.
        articles: Candidate articles to filter.
        batch_size: How many articles to classify per LLM call.
        domain: Company domain for LLM disambiguation context.
        industry: Company industry for LLM disambiguation context.

    Returns:
        Subset of articles classified as relevant.
    """
    if not articles:
        return []

    name_lower = company_name.lower()
    name_parts = [p.lower() for p in company_name.split() if len(p) >= 3]

    auto_accepted: list[NewsArticle] = []
    need_llm: list[NewsArticle] = []

    for a in articles:
        title_lower = (a.title or "").lower()
        summary_lower = (a.summary or "").lower()
        combined = f"{title_lower} {summary_lower}"

        # Layer 1: Full name match (word-boundary aware) → always accept
        # B1: Using \b prevents "Apple" matching "pineapple", etc.
        if re.search(r'\b' + re.escape(name_lower) + r'\b', combined):
            auto_accepted.append(a)
            continue

        # Layer 2: Multi-word name (2+ words) — require >=60% significant words (min 2)
        if len(name_parts) >= 2:
            matched = sum(1 for p in name_parts if p in combined)
            threshold = max(2, int(len(name_parts) * 0.6))
            if matched >= threshold:
                auto_accepted.append(a)
                continue

        # Layer 3: Single-word name — whole-word match (not substring)
        if len(name_parts) == 1:
            if re.search(r'\b' + re.escape(name_parts[0]) + r'\b', combined):
                auto_accepted.append(a)
                continue

        # Layer 4: Everything else → LLM classification
        need_llm.append(a)

    if not need_llm:
        return auto_accepted

    # LLM classification for ambiguous articles — with company context
    try:
        from app.tools.llm.llm_service import LLMService
        llm = LLMService(lite=True)

        # Build company context for disambiguation
        context_line = ""
        if domain or industry:
            parts = []
            if industry:
                parts.append(f"in the {industry} industry")
            if domain:
                parts.append(f"with domain {domain}")
            context_line = f"Note: \"{company_name}\" is a company {' '.join(parts)}.\n"

        classified: list[NewsArticle] = []
        for start in range(0, len(need_llm), batch_size):
            batch = need_llm[start:start + batch_size]
            numbered = "\n".join(
                f"{i+1}. {a.title or '(no title)'} | {(a.summary or '')[:120]}"
                for i, a in enumerate(batch)
            )

            prompt = (
                f"You are classifying news articles for relevance to the company \"{company_name}\".\n"
                f"{context_line}\n"
                f"For each article below, reply with ONLY the number followed by Y (relevant) or N (not relevant).\n"
                f"An article is relevant if it is primarily about {company_name}, its products, employees, "
                f"financials, partnerships, or direct competitors mentioning {company_name}.\n"
                f"An article is NOT relevant if it merely mentions the company in passing, is about a "
                f"different entity with a similar name, or is a generic market/industry report.\n\n"
                f"Articles:\n{numbered}\n\n"
                f"Reply format (one per line): 1 Y\\n2 N\\n3 Y\\n..."
            )

            result = await asyncio.wait_for(
                llm.generate(prompt, temperature=0.0, max_tokens=200),
                timeout=12.0,
            )

            if result:
                for line in result.strip().split("\n"):
                    line = line.strip()
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            idx = int(parts[0]) - 1
                            verdict = parts[1].upper()
                            if verdict == "Y" and 0 <= idx < len(batch):
                                classified.append(batch[idx])
                        except (ValueError, IndexError):
                            continue

        total = auto_accepted + classified
        logger.info(
            f"relevance_filter: {company_name} — "
            f"{len(auto_accepted)} auto-accepted, {len(classified)}/{len(need_llm)} LLM-accepted, "
            f"{len(articles) - len(total)} rejected"
        )
        return total

    except Exception as e:
        logger.debug(f"relevance_filter LLM failed for {company_name}: {e}")
        # On LLM failure, use title-only heuristic as safety net:
        # accept if company name appears in the title (stricter than title+summary)
        title_rescued: list[NewsArticle] = []
        for a in need_llm:
            t = (a.title or "").lower()
            if name_lower in t:
                title_rescued.append(a)
            elif len(name_parts) >= 2:
                matched = sum(1 for p in name_parts if p in t)
                if matched >= max(2, int(len(name_parts) * 0.6)):
                    title_rescued.append(a)
        logger.info(
            f"relevance_filter: LLM failed, title-heuristic rescued "
            f"{len(title_rescued)}/{len(need_llm)} ambiguous articles"
        )
        return auto_accepted + title_rescued


async def _enrich_articles(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Extract content and generate summary for each article in parallel."""

    async def _process_one(article: NewsArticle) -> None:
        text = await extract(article.url, timeout=12.0)
        if text:
            article.content = text
            article.summary = _summarize_text(text, sentences=3)

    tasks = [_process_one(a) for a in articles]
    await asyncio.gather(*tasks, return_exceptions=True)
    return articles


def _summarize_text(text: str, sentences: int = 3) -> str:
    """Extract first N sentences as a simple summary."""
    if not text:
        return ""
    # Split on sentence boundaries (period followed by space and uppercase letter)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    selected = parts[:sentences]
    summary = " ".join(selected)
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


# ══════════════════════════════════════════════════════════════════
# 4. deep_company_search()
# ══════════════════════════════════════════════════════════════════

_DEFAULT_COMPANY_PROMPT = """Find comprehensive information about this company.
Return a JSON object with these exact fields:
- company_name: Official company name
- also_known_as: List of alternative names, abbreviations, former names
- industry: Primary industry
- sub_industries: List of specific sub-industries or sectors
- description: 2-3 sentence company description
- founded_year: Year founded (integer or null)
- headquarters: City, State/Country
- ceo: Current CEO name
- key_people: List of key executives (name - title format)
- employee_count: Approximate number of employees as string
- revenue: Latest annual revenue with currency
- market_cap: Current market capitalization with currency
- stock_ticker: Stock exchange ticker symbol
- website: Official website URL
- domain: Primary domain name
- products_services: List of main products or services
- competitors: List of main competitors
- subsidiaries: List of major subsidiaries
- funding_stage: Current funding stage (e.g., Series B, Public, Private)
- total_funding: Total funding raised with currency
- investors: List of notable investors
- tech_stack: List of known technologies used
- recent_events: List of recent news or events (last 6 months)
- sources: List of URLs used for information
"""


# Limit concurrent ScrapeGraphAI calls (each uses ~10K-50K tokens from OpenAI)
_scrapegraph_sem = asyncio.Semaphore(2)


async def deep_company_search(query: str, prompt: Optional[str] = None) -> Optional[CompanyProfile]:
    """Deep company search using ScrapeGraphAI SearchGraph.

    WARNING: Slow (30-80 seconds). Use for background enrichment only, not interactive requests.

    Args:
        query: Company name or search query.
        prompt: Custom extraction prompt. Defaults to comprehensive company profile extraction.

    Returns:
        CompanyProfile or None on failure.
    """
    async with _scrapegraph_sem:
      try:
        from scrapegraphai.graphs import SearchGraph

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Fall back to app settings (loaded from .env by pydantic)
            try:
                from app.config import get_settings
                api_key = get_settings().openai_api_key or ""
            except Exception:
                pass
        if not api_key:
            logger.warning("deep_company_search: OPENAI_API_KEY not set, skipping")
            return None

        config = {
            "llm": {
                "api_key": api_key,
                "model": "openai/gpt-4.1-mini",
                "temperature": 0.1,
            },
            "verbose": False,
        }

        effective_prompt = prompt or f"{_DEFAULT_COMPANY_PROMPT}\n\nCompany: {query}"

        def _run_search() -> str:
            graph = SearchGraph(
                prompt=effective_prompt,
                config=config,
            )
            return graph.run()

        raw = await asyncio.to_thread(_run_search)

        if not raw:
            return None

        return _parse_company_result(raw, query)

      except Exception as e:
        logger.warning(f"deep_company_search failed for '{query}': {e}")
        return None


def _parse_company_result(raw: Union[str, dict], query: str) -> Optional[CompanyProfile]:
    """Parse ScrapeGraphAI output into CompanyProfile. Handles both string and dict results."""
    import json

    data: dict = {}

    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse JSON from deep_company_search result for '{query}'")
                    return None
            else:
                return None

    if not data:
        return None

    def _get_list(key: str) -> list[str]:
        val = data.get(key, [])
        if isinstance(val, list):
            return [str(v) for v in val if v]
        if isinstance(val, str) and val:
            return [val]
        return []

    try:
        return CompanyProfile(
            company_name=data.get("company_name", "") or query,
            also_known_as=_get_list("also_known_as"),
            industry=data.get("industry", "") or "",
            sub_industries=_get_list("sub_industries"),
            description=data.get("description", "") or "",
            founded_year=_safe_int(data.get("founded_year")),
            headquarters=data.get("headquarters", "") or "",
            ceo=data.get("ceo", "") or "",
            key_people=_get_list("key_people"),
            employee_count=str(data.get("employee_count", "")) if data.get("employee_count") else None,
            revenue=data.get("revenue", "") or "",
            market_cap=data.get("market_cap", "") or "",
            stock_ticker=data.get("stock_ticker", "") or "",
            website=data.get("website", "") or "",
            domain=data.get("domain", "") or "",
            products_services=_get_list("products_services"),
            competitors=_get_list("competitors"),
            subsidiaries=_get_list("subsidiaries"),
            funding_stage=data.get("funding_stage", "") or "",
            total_funding=data.get("total_funding", "") or "",
            investors=_get_list("investors"),
            tech_stack=_get_list("tech_stack"),
            recent_events=_get_list("recent_events"),
            sources=_get_list("sources"),
        )
    except Exception as e:
        logger.debug(f"Failed to construct CompanyProfile from data: {e}")
        return None


def _safe_int(val) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════
# 5. search_industry_companies()
# ══════════════════════════════════════════════════════════════════


async def search_industry_companies(
    industry: str,
    max_companies: int = 10,
) -> list[CompanyProfile]:
    """Discover companies in an industry using Tavily advanced search + LLM extraction.

    Pipeline:
      1. Tavily advanced search for industry companies (AI answer + results)
      2. LLM: extract company names from answer + snippets
      3. Entity validation via company_enricher.validate_entity()
      4. Return validated CompanyProfile stubs

    Args:
        industry: Industry name (e.g., "electric vehicles", "fintech India").
        max_companies: Maximum number of companies to return.

    Returns:
        List of CompanyProfile stubs (basic info, not full enrichment).
    """
    # Step 1: Tavily search for industry companies
    try:
        from .tavily_tool import TavilyTool

        tavily = TavilyTool()
        current_year = datetime.now().year
        result = await tavily.search(
            query=f"top {industry} companies list {current_year}",
            search_depth="advanced",
            max_results=10,
            include_answer="advanced",
        )
    except Exception as e:
        logger.warning(f"Industry search failed for '{industry}': {e}")
        # Fallback to DDG
        current_year = datetime.now().year
        results = await _ddgs_search(f"top {industry} companies list {current_year}", max_results=10)
        result = {
            "answer": "",
            "results": [{"content": r.snippet} for r in results],
        }

    # Step 2: Extract company names using LLM
    names = await _extract_company_names_from_search(industry, result)
    if not names:
        return []

    # Step 3: Validate entities (filter out non-companies)
    validated: list[CompanyProfile] = []
    try:
        from app.tools.company_enricher import validate_entity

        for name in names[:max_companies + 5]:  # over-fetch to account for rejections
            try:
                vr = await asyncio.wait_for(validate_entity(name), timeout=8.0)
                if vr.is_valid_company:
                    validated.append(CompanyProfile(company_name=name))
                    if len(validated) >= max_companies:
                        break
            except (asyncio.TimeoutError, Exception):
                continue
    except Exception as e:
        logger.warning(f"Entity validation failed during industry search: {e}")
        # Return unvalidated names as fallback
        validated = [CompanyProfile(company_name=n) for n in names[:max_companies]]

    return validated


async def _extract_company_names_from_search(industry: str, search_result: dict) -> list[str]:
    """Use LLM to extract company names from Tavily search results."""
    answer = search_result.get("answer", "")
    snippets = "\n".join(
        r.get("content", "")[:300]
        for r in search_result.get("results", [])[:5]
        if r.get("content")
    )

    if not answer and not snippets:
        return []

    context = f"AI Answer: {answer}\n\nSearch Snippets:\n{snippets}" if answer else f"Search Snippets:\n{snippets}"

    try:
        from app.tools.llm.llm_service import LLMService

        llm = LLMService(lite=True)
        prompt = (
            f"From the following search results about {industry} companies, "
            f"extract ONLY the company names. Return one company name per line, "
            f"nothing else. No numbering, no descriptions, just clean company names.\n\n"
            f"{context}"
        )
        response = await asyncio.wait_for(
            llm.generate(prompt, temperature=0.1, max_tokens=500),
            timeout=10.0,
        )

        if not response:
            return []

        # Parse one name per line, clean up
        names = []
        for line in response.strip().split("\n"):
            name = line.strip().strip("-").strip("•").strip("*").strip()
            if name and len(name) >= 2 and len(name) <= 60:
                names.append(name)

        return names

    except Exception as e:
        logger.debug(f"LLM company name extraction failed: {e}")
        return []
