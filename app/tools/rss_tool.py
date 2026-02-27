"""
RSS Tool for fetching news from multiple Indian sources.
Supports 20+ free RSS feeds and API sources.
Focuses on TODAY's Indian business news - specific events, not generic trends.
"""

import html
import logging
import re
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import uuid4
import asyncio
import httpx
import feedparser

from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
DetectorFactory.seed = 0  # Deterministic language detection

from ..config import get_settings, NEWS_SOURCES, DEFAULT_ACTIVE_SOURCES
from ..schemas import NewsArticle, SourceType, SourceTier

logger = logging.getLogger(__name__)


class RSSTool:
    """
    Multi-source news fetcher supporting RSS feeds and free APIs.
    Fetches TODAY's Indian business news from 20+ sources.
    """

    # Source health tracking
    _source_health: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _is_target_language(text: str, target_lang: str = "en") -> bool:
        """Check if text is in the target language using langdetect.

        Uses Google's n-gram language detection (55+ languages).
        Scales globally — works for any language, no hardcoded Unicode ranges.
        Returns True if detected language matches target, or if text is too
        short for reliable detection (< 20 chars).
        """
        if not text or len(text.strip()) < 20:
            return True  # Too short to reliably detect
        try:
            detected = detect(text[:500])  # Cap input for speed
            return detected == target_lang
        except LangDetectException:
            return True  # Ambiguous — let through

    def __init__(self, mock_mode: bool = False):
        """Initialize RSS tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.active_sources = DEFAULT_ACTIVE_SOURCES
    
    async def fetch_all_sources(
        self,
        source_ids: Optional[List[str]] = None,
        max_per_source: int = None,
        hours_ago: int = None
    ) -> List[NewsArticle]:
        """
        Fetch news from ALL configured sources in parallel.

        Args:
            source_ids: List of source IDs to fetch (defaults to active sources)
            max_per_source: Max articles per source
            hours_ago: Only include news from last N hours

        Returns:
            List of NewsArticle objects with deduplication
        """
        # Default from config if not provided
        if max_per_source is None:
            max_per_source = self.settings.rss_max_per_source
        if hours_ago is None:
            hours_ago = self.settings.rss_hours_ago

        if self.mock_mode:
            return self._get_mock_articles()

        source_ids = source_ids or self.active_sources
        all_articles: List[NewsArticle] = []

        # Source bandit: allocate more articles to higher-quality sources
        source_caps = self._compute_source_caps(source_ids, max_per_source)

        # Fetch from all sources with concurrency limit (avoid overwhelming network)
        semaphore = asyncio.Semaphore(6)

        async def _fetch_limited(sid):
            async with semaphore:
                try:
                    # Add per-source timeout of 15 seconds (httpx has 30s, but this ensures overall progress)
                    return await asyncio.wait_for(
                        self._fetch_source(sid, source_caps.get(sid, max_per_source)),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[TIMEOUT] {NEWS_SOURCES.get(sid, {}).get('name', sid)}: Fetch timeout (15s), skipping")
                    return []

        tasks = [_fetch_limited(sid) for sid in source_ids if sid in NEWS_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Source fetch failed: {result}")
            elif isinstance(result, list):
                all_articles.extend(result)

        # Deduplicate articles
        all_articles = self._deduplicate_articles(all_articles)

        # Filter by time
        cutoff = datetime.utcnow() - timedelta(hours=hours_ago)
        all_articles = [a for a in all_articles if a.published_at >= cutoff]

        # Sort by recency
        all_articles.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(f"[RSS] Fetched {len(all_articles)} unique articles from {len(source_ids)} sources")
        return all_articles

    async def _fetch_source(self, source_id: str, max_results: int) -> List[NewsArticle]:
        """Fetch articles from a single source."""
        source = NEWS_SOURCES.get(source_id)
        if not source:
            return []

        try:
            source_type = source.get("source_type", "rss")

            if source_type == "rss":
                articles = await self._fetch_rss_source(source, max_results)
            elif source_type == "api":
                articles = await self._fetch_api_source(source, max_results)
            else:
                articles = []

            # Update health tracking
            self._source_health[source_id] = {
                "last_success": datetime.utcnow(),
                "consecutive_failures": 0,
                "articles_fetched": len(articles)
            }

            logger.info(f"[OK] {source['name']}: {len(articles)} articles")
            return articles

        except Exception as e:
            # Track failures
            health = self._source_health.get(source_id, {"consecutive_failures": 0})
            health["consecutive_failures"] = health.get("consecutive_failures", 0) + 1
            health["last_error"] = str(e)
            self._source_health[source_id] = health

            logger.warning(f"[FAIL] {source['name']}: {e}")
            return []

    def _compute_source_caps(
        self, source_ids: List[str], base_cap: int
    ) -> Dict[str, int]:
        """Allocate per-source article caps using Source Bandit posteriors.

        Higher-quality sources (learned from previous runs via Thompson Sampling)
        get more article slots. Lower-quality sources get fewer but never zero.

        Allocation: base_cap * quality_multiplier, where:
          - Top 25% sources: 1.5x base_cap
          - Middle 50%: 1.0x base_cap (default)
          - Bottom 25%: 0.6x base_cap (still fetched, just fewer)

        Falls back to equal allocation if no bandit data exists.
        """
        try:
            from app.learning.source_bandit import SourceBandit
            bandit = SourceBandit()
            estimates = bandit.get_quality_estimates()
        except Exception:
            return {sid: base_cap for sid in source_ids}

        if not estimates:
            return {sid: base_cap for sid in source_ids}

        # Get quality estimates for active sources (default 0.5 for unknown)
        qualities = {sid: estimates.get(sid, 0.5) for sid in source_ids}

        # Compute percentile thresholds
        vals = sorted(qualities.values())
        if len(vals) < 4:
            return {sid: base_cap for sid in source_ids}

        p25 = vals[len(vals) // 4]
        p75 = vals[(3 * len(vals)) // 4]

        caps = {}
        for sid in source_ids:
            q = qualities[sid]
            if q >= p75:
                caps[sid] = int(base_cap * 1.5)
            elif q <= p25:
                caps[sid] = max(3, int(base_cap * 0.6))
            else:
                caps[sid] = base_cap

        boosted = sum(1 for c in caps.values() if c > base_cap)
        reduced = sum(1 for c in caps.values() if c < base_cap)
        if boosted or reduced:
            logger.info(
                f"Source bandit caps: {boosted} boosted, {reduced} reduced "
                f"(base={base_cap}, p25={p25:.3f}, p75={p75:.3f})"
            )

        return caps

    # Browser-like User-Agent to avoid being blocked by some RSS feeds (e.g. Inc42)
    _USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

    async def _fetch_rss_source(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from an RSS feed source."""
        rss_url = source.get("rss_url")
        if not rss_url:
            return []

        headers = {"User-Agent": self._USER_AGENT}
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(rss_url, follow_redirects=True, headers=headers)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            articles = []

            for entry in feed.entries[:max_results]:
                article = self._parse_rss_entry(entry, source)
                if article:
                    articles.append(article)

            return articles

    # Dispatch table mapping source IDs to their fetch methods.
    # Populated in __init_subclass__ would be overkill; a simple property works.
    _API_DISPATCH = {
        "gnews": "_fetch_gnews",
        "newsdata": "_fetch_newsdata",
        "newsapi_org": "_fetch_newsapi_org",
        "rapidapi_realtime_news": "_fetch_rapidapi_news",
        "rapidapi_google_news": "_fetch_rapidapi_news",
        "rapidapi_google_trends_news": "_fetch_google_trends_news",
        "mediastack": "_fetch_mediastack",
        "thenewsapi": "_fetch_thenewsapi",
        "webz_news": "_fetch_webz",
        "gdelt_india": "_fetch_gdelt",
        "gdelt_india_business": "_fetch_gdelt",
    }

    async def _fetch_api_source(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from an API source (GNews, NewsData, RapidAPI, etc)."""
        source_id = source.get("id", "")
        method_name = self._API_DISPATCH.get(source_id)
        if method_name:
            method = getattr(self, method_name)
            return await method(source, max_results)
        return []

    async def _fetch_newsapi_org(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from NewsAPI.org (100 calls/day × 20 articles = 2000/day FREE)."""
        api_key = os.getenv("NEWSAPI_ORG_KEY", "")
        if not api_key:
            logger.debug("NEWSAPI_ORG_KEY not set, skipping")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "q": "India business OR India economy OR India startup",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 20)
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", "")[:500] if item.get("description") else "",
                    content=item.get("content"),
                    url=item.get("url", ""),
                    source_id=source["id"],
                    source_name=item.get("source", {}).get("name", "NewsAPI"),
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_1,
                    source_credibility=0.90,
                    published_at=self._parse_datetime(item.get("publishedAt", ""))
                )
                articles.append(article)

            return articles

    async def _fetch_rapidapi_news(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from RapidAPI News APIs (Real-Time News Data, Google News)."""
        api_key = os.getenv("RAPIDAPI_KEY", "")
        if not api_key:
            logger.debug("RAPIDAPI_KEY not set, skipping")
            return []

        rapidapi_host = source.get("rapidapi_host", "")
        url = source.get("api_endpoint", "")

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": rapidapi_host
        }

        params = {
            "query": "India business",
            "country": "IN",
            "lang": "en",
            "limit": str(min(max_results, 50))
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            # Handle different response formats
            items = data.get("data", []) or data.get("articles", []) or data.get("news", [])

            for item in items[:max_results]:
                article = NewsArticle(
                    title=item.get("title") or item.get("headline") or "",
                    summary=(item.get("snippet") or item.get("description") or "")[:500],
                    url=item.get("link") or item.get("url") or "",
                    source_id=source["id"],
                    source_name=item.get("source", {}).get("name", "") or item.get("publisher", "") or "RapidAPI",
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_2,
                    source_credibility=0.85,
                    published_at=self._parse_datetime(
                        item.get("published_datetime_utc", "") or
                        item.get("publishedAt", "") or
                        item.get("date", "")
                    )
                )
                articles.append(article)

            return articles

    async def _fetch_google_trends_news(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from Google Trends & News Insights API (RapidAPI).

        Calls up to 3 endpoints in parallel for maximum coverage:
        1. /news — keyword search for "India business"
        2. /top-headlines — trending headlines for India
        3. /topic-headlines — business topic headlines for India
        """
        api_key = os.getenv("RAPIDAPI_KEY", "")
        if not api_key:
            logger.debug("RAPIDAPI_KEY not set, skipping Google Trends News")
            return []

        host = source.get("rapidapi_host", "google-trends-news-insights-api.p.rapidapi.com")
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host,
        }
        base = f"https://{host}"
        per_endpoint = max(max_results // 3, 5)

        requests = [
            (f"{base}/news", {"q": "India business economy", "country": "in", "language": "en", "limit": str(per_endpoint)}),
            (f"{base}/top-headlines", {"country": "in", "language": "en", "limit": str(per_endpoint)}),
            (f"{base}/topic-headlines", {"country": "in", "language": "en", "topic": "Business", "limit": str(per_endpoint)}),
        ]

        async def _call(client: httpx.AsyncClient, url: str, params: Dict) -> List[Dict]:
            try:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                body = resp.json()
                # Response shape: {"data": {"articles": [...]}, "ok": true}
                if isinstance(body, dict):
                    inner = body.get("data", body)
                    if isinstance(inner, dict):
                        return inner.get("articles", [])
                    if isinstance(inner, list):
                        return inner
                return []
            except Exception as e:
                logger.debug(f"Google Trends endpoint {url} failed: {e}")
                return []

        all_items: List[Dict] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            results = await asyncio.gather(*[_call(client, url, p) for url, p in requests])
            for items in results:
                all_items.extend(items)

        # Deduplicate by title and convert to NewsArticle
        seen_titles: set = set()
        articles: List[NewsArticle] = []
        for item in all_items:
            title = item.get("title", "")
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())

            # Source can be a dict {"name": "...", "url": "..."} or a string
            src = item.get("source", "")
            source_name = src.get("name", "") if isinstance(src, dict) else str(src) or "Google Trends News"

            article = NewsArticle(
                title=title,
                summary=(item.get("description", "") or "")[:500],
                url=item.get("url", "") or item.get("link", ""),
                source_id=source["id"],
                source_name=source_name,
                source_type=SourceType.API,
                source_tier=SourceTier.TIER_2,
                source_credibility=0.87,
                published_at=self._parse_datetime(item.get("date", "")),
            )
            articles.append(article)

        logger.info(f"Google Trends News: {len(articles)} articles from 3 endpoints")
        return articles[:max_results]

    async def _fetch_mediastack(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from MediaStack (500 calls/month FREE)."""
        api_key = os.getenv("MEDIASTACK_API_KEY", "")
        if not api_key:
            logger.debug("MEDIASTACK_API_KEY not set, skipping")
            return []

        url = "http://api.mediastack.com/v1/news"
        params = {
            "access_key": api_key,
            "countries": "in",
            "languages": "en",
            "categories": "business",
            "limit": min(max_results, 25)
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("data", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", "")[:500] if item.get("description") else "",
                    url=item.get("url", ""),
                    source_id=source["id"],
                    source_name=item.get("source", "MediaStack"),
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_2,
                    source_credibility=0.85,
                    published_at=self._parse_datetime(item.get("published_at", ""))
                )
                articles.append(article)

            return articles

    async def _fetch_thenewsapi(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from TheNewsAPI (free tier)."""
        api_key = os.getenv("THENEWSAPI_KEY", "")
        if not api_key:
            logger.debug("THENEWSAPI_KEY not set, skipping")
            return []

        url = "https://api.thenewsapi.com/v1/news/all"
        params = {
            "api_token": api_key,
            "locale": "in",
            "language": "en",
            "categories": "business",
            "limit": min(max_results, 50)
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("data", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", "")[:500] if item.get("description") else "",
                    content=item.get("snippet"),
                    url=item.get("url", ""),
                    source_id=source["id"],
                    source_name=item.get("source", "TheNewsAPI"),
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_2,
                    source_credibility=0.85,
                    published_at=self._parse_datetime(item.get("published_at", ""))
                )
                articles.append(article)

            return articles

    async def _fetch_webz(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from Webz.io News API Lite (1000 calls/month × 10 articles FREE)."""
        api_key = os.getenv("WEBZ_API_KEY", "")
        if not api_key:
            logger.debug("WEBZ_API_KEY not set, skipping")
            return []

        url = "https://api.webz.io/newsApiLite"
        params = {
            "token": api_key,
            "q": "site_category:business language:english site_country:IN",
            "size": min(max_results, 10)
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("posts", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("text", "")[:500] if item.get("text") else "",
                    url=item.get("url", ""),
                    source_id=source["id"],
                    source_name=item.get("thread", {}).get("site", "Webz.io"),
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_2,
                    source_credibility=0.88,
                    published_at=self._parse_datetime(item.get("published", ""))
                )
                articles.append(article)

            return articles

    async def _fetch_gnews(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from GNews API (100 free/day)."""
        api_key = os.getenv("GNEWS_API_KEY", "")
        if not api_key:
            logger.debug("GNews API key not set, skipping")
            return []

        url = "https://gnews.io/api/v4/top-headlines"
        params = {
            "token": api_key,
            "country": "in",
            "category": "business",
            "max": min(max_results, 10),
            "lang": "en"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=(item.get("description") or "")[:500],
                    content=item.get("content"),
                    url=item.get("url", ""),
                    source_id=source["id"],
                    source_name=item.get("source", {}).get("name", "GNews"),
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_2,
                    source_credibility=0.85,
                    published_at=self._parse_datetime(item.get("publishedAt", ""))
                )
                articles.append(article)

            return articles

    async def _fetch_newsdata(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """Fetch from NewsData.io API (200 free/day)."""
        api_key = os.getenv("NEWSDATA_API_KEY", "")
        if not api_key:
            logger.debug("NewsData API key not set, skipping")
            return []

        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": api_key,
            "country": "in",
            "category": "business",
            "language": "en"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("results", [])[:max_results]:
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", "")[:500] if item.get("description") else "",
                    content=item.get("content"),
                    url=item.get("link", ""),
                    source_id=source["id"],
                    source_name=item.get("source_id", "NewsData"),
                    source_type=SourceType.API,
                    source_tier=SourceTier.TIER_2,
                    source_credibility=0.83,
                    published_at=self._parse_datetime(item.get("pubDate", ""))
                )
                articles.append(article)

            return articles

    async def _fetch_gdelt(self, source: Dict, max_results: int) -> List[NewsArticle]:
        """
        Fetch from GDELT DOC API (free, no key needed).

        GDELT indexes ~300k articles/day across 100+ languages.
        We use the DOC API v2 artlist mode for India-sourced articles.

        Two source variants:
          - gdelt_india: broad India news (sourcecountry:IN)
          - gdelt_india_business: business-focused (adds keyword filter)
        """
        from urllib.parse import quote

        source_id = source.get("id", "gdelt_india")
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

        # Build query based on source variant
        if source_id == "gdelt_india_business":
            query = "sourcecountry:IN sourcelang:english (business OR startup OR economy OR funding OR acquisition OR regulation)"
        else:
            query = "sourcecountry:IN sourcelang:english"

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max_results * 5, 250)),  # Request more, filter later
            "sort": "DateDesc",
            "timespan": "2880",  # Last 48 hours (in minutes)
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(base_url, params=params)

                # GDELT returns 429 on rate limit, HTML on error
                if response.status_code == 429:
                    logger.warning("GDELT rate limited (429), skipping")
                    return []
                if response.status_code != 200:
                    logger.debug(f"GDELT HTTP {response.status_code}")
                    return []

                # GDELT sometimes returns HTML error page instead of JSON
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    logger.debug("GDELT returned HTML (error page), skipping")
                    return []

                try:
                    data = response.json()
                except Exception:
                    # GDELT sometimes returns JSONP or malformed JSON
                    text = response.text.strip()
                    # Try stripping JSONP wrapper: callback({...})
                    if text.startswith("(") and text.endswith(")"):
                        text = text[1:-1]
                    try:
                        import json
                        data = json.loads(text)
                    except Exception:
                        logger.debug(f"GDELT returned unparseable response ({len(text)} chars)")
                        return []
                raw_articles = data.get("articles", [])

                articles = []
                seen_titles = set()

                for item in raw_articles:
                    title = (item.get("title") or "").strip()
                    url = (item.get("url") or "").strip()

                    if not title or not url or len(title) < 10:
                        continue

                    # Deduplicate within GDELT results
                    title_key = title.lower()[:60]
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)

                    # Parse GDELT date format: "20260205T143000Z"
                    seendate = item.get("seendate", "")
                    published = self._parse_gdelt_date(seendate)

                    domain = item.get("domain", "")

                    article = NewsArticle(
                        title=title,
                        summary=title,  # GDELT artlist doesn't return summaries
                        url=url,
                        source_id=source_id,
                        source_name=f"GDELT/{domain}" if domain else "GDELT",
                        source_type=SourceType.API,
                        source_tier=SourceTier(source.get("tier", "tier_1")),
                        source_credibility=source.get("credibility_score", 0.88),
                        published_at=published,
                    )
                    articles.append(article)

                    if len(articles) >= max_results:
                        break

                logger.info(f"GDELT ({source_id}): {len(articles)} articles from {len(raw_articles)} raw")
                return articles

        except httpx.TimeoutException:
            logger.warning("GDELT timeout (30s)")
            return []
        except Exception as e:
            logger.warning(f"GDELT fetch failed: {e}")
            return []

    def _parse_gdelt_date(self, date_str: str) -> datetime:
        """Parse GDELT date format: '20260205T143000Z' → datetime."""
        if not date_str:
            return datetime.utcnow()
        try:
            # Format: YYYYMMDDTHHMMSSZ
            return datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
        except ValueError:
            try:
                # Sometimes just YYYYMMDDHHMMSS
                return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            except ValueError:
                return datetime.utcnow()

    def _parse_rss_entry(self, entry: Dict, source: Dict) -> Optional[NewsArticle]:
        """Parse RSS entry to NewsArticle."""
        try:
            title = entry.get("title", "")
            # Remove source suffix from title (e.g., "News - Economic Times")
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]

            summary = entry.get("summary", "") or entry.get("description", "")
            summary = re.sub(r'<[^>]+>', '', summary)  # Strip HTML tags
            summary = html.unescape(summary)[:500]     # Decode &nbsp; &amp; etc.
            title = html.unescape(title)                # Decode entities in title too

            # Language filter: reject non-English articles (Hindi, Bengali, etc.)
            # Uses langdetect (Google n-gram detector) — scales to any language
            target_lang = source.get("language", "en")
            check_text = f"{title} {summary[:200]}"
            if not self._is_target_language(check_text, target_lang):
                logger.debug(f"Filtered non-{target_lang} article: {title[:60]}...")
                return None

            published = datetime.utcnow()
            if entry.get("published_parsed"):
                try:
                    published = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass

            return NewsArticle(
                title=title,
                summary=summary,
                url=entry.get("link", ""),
                source_id=source["id"],
                source_name=source["name"],
                source_type=SourceType(source.get("source_type", "rss")),
                source_tier=SourceTier(source.get("tier", "tier_2")),
                source_credibility=source.get("credibility_score", 0.8),
                published_at=published
            )

        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None

    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse various datetime formats."""
        if not date_str:
            return datetime.utcnow()

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.replace("+00:00", "Z"), fmt)
            except ValueError:
                continue

        return datetime.utcnow()

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on normalized title."""
        seen_hashes = set()
        unique = []

        for article in articles:
            title_norm = article.title.lower()[:60]
            content_hash = hashlib.md5(title_norm.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(article)
            else:
                article.is_duplicate = True

        logger.info(f"Deduplication: {len(articles)} -> {len(unique)} articles")
        return unique

    async def audit_sources(
        self,
        source_ids: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Dry-run audit of all configured sources. Checks reachability, response
        format, and article counts WITHOUT consuming API quotas aggressively.

        For RSS: GET the feed, parse XML, count entries.
        For API: check if API key env var is set. If set, make a minimal call
                 (limit=1 or smallest allowed) to validate response shape.
                 GDELT (no key needed) always gets tested.

        Returns:
            {source_id: {
                "status": "ok" | "broken" | "empty" | "no_key" | "slow" | "parse_error",
                "http_status": int | None,
                "response_time_ms": int,
                "article_count": int,   # entries found in response
                "error": str | None,
                "source_type": "rss" | "api",
                "name": str,
            }}
        """
        import time as _time

        source_ids = source_ids or list(NEWS_SOURCES.keys())
        results: Dict[str, Dict[str, Any]] = {}

        async def _audit_one(source_id: str) -> tuple:
            source = NEWS_SOURCES.get(source_id)
            if not source:
                return source_id, {"status": "broken", "error": "Not in NEWS_SOURCES", "name": source_id}

            source_type = source.get("source_type", "rss")
            name = source.get("name", source_id)
            result: Dict[str, Any] = {
                "name": name,
                "source_type": source_type,
                "http_status": None,
                "response_time_ms": 0,
                "article_count": 0,
                "error": None,
            }

            start = _time.time()
            try:
                if source_type == "rss":
                    result = await self._audit_rss(source, result, timeout)
                elif source_type == "api":
                    result = await self._audit_api(source, source_id, result, timeout)
            except Exception as e:
                result["status"] = "broken"
                result["error"] = str(e)
            result["response_time_ms"] = int((_time.time() - start) * 1000)

            # Classify slow sources
            if result.get("status") == "ok" and result["response_time_ms"] > 8000:
                result["status"] = "slow"

            return source_id, result

        # Run all audits in parallel
        audit_tasks = [_audit_one(sid) for sid in source_ids if sid in NEWS_SOURCES]
        audit_results = await asyncio.gather(*audit_tasks, return_exceptions=True)

        for item in audit_results:
            if isinstance(item, Exception):
                logger.warning(f"Audit task failed: {item}")
                continue
            sid, res = item
            results[sid] = res
            # Update health tracking
            self._source_health[sid] = res

        # Summary log
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        broken = sum(1 for r in results.values() if r.get("status") == "broken")
        no_key = sum(1 for r in results.values() if r.get("status") == "no_key")
        empty = sum(1 for r in results.values() if r.get("status") == "empty")
        logger.info(
            f"Source audit: {ok} ok, {broken} broken, {empty} empty, "
            f"{no_key} no API key, {len(results)} total"
        )
        return results

    async def _audit_rss(self, source: Dict, result: Dict, timeout: float) -> Dict:
        """Audit a single RSS source."""
        rss_url = source.get("rss_url")
        if not rss_url:
            result["status"] = "broken"
            result["error"] = "No rss_url configured"
            return result

        headers = {"User-Agent": self._USER_AGENT}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(rss_url, follow_redirects=True, headers=headers)
            result["http_status"] = response.status_code

            if response.status_code != 200:
                result["status"] = "broken"
                result["error"] = f"HTTP {response.status_code}"
                return result

            try:
                feed = feedparser.parse(response.text)
                if feed.bozo and not feed.entries:
                    result["status"] = "parse_error"
                    result["error"] = f"XML parse error: {feed.bozo_exception}"
                    return result
                result["article_count"] = len(feed.entries)
                result["status"] = "ok" if feed.entries else "empty"
            except Exception as e:
                result["status"] = "parse_error"
                result["error"] = str(e)

        return result

    async def _audit_api(self, source: Dict, source_id: str, result: Dict, timeout: float) -> Dict:
        """Audit a single API source. Checks env var first, then minimal call."""
        api_key_env = source.get("api_key_env", "")

        # GDELT needs no key — always test it
        is_gdelt = source_id.startswith("gdelt_")

        if not is_gdelt and api_key_env:
            key_val = os.getenv(api_key_env, "")
            if not key_val:
                result["status"] = "no_key"
                result["error"] = f"{api_key_env} not set"
                return result

        # Minimal fetch: try to get 1 article to validate endpoint
        try:
            articles = await self._fetch_source(source_id, max_results=1)
            result["http_status"] = 200
            result["article_count"] = len(articles) if isinstance(articles, list) else 0
            result["status"] = "ok" if result["article_count"] > 0 else "empty"
        except Exception as e:
            result["status"] = "broken"
            result["error"] = str(e)

        return result

    def get_healthy_sources(self, source_ids: Optional[List[str]] = None) -> List[str]:
        """Return source IDs that passed the last audit (status=ok or no audit yet)."""
        source_ids = source_ids or self.active_sources
        healthy = []
        for sid in source_ids:
            health = self._source_health.get(sid)
            if health is None:
                healthy.append(sid)  # No audit yet → assume ok
            elif health.get("status") in ("ok", "slow"):
                healthy.append(sid)
        return healthy

    def get_source_health(self) -> Dict[str, Dict]:
        """Get health status of all sources."""
        return self._source_health

    def _get_mock_articles(self) -> List[NewsArticle]:
        """Return mock NewsArticle objects for testing."""
        mock_data = self._get_mock_trends()
        articles = []
        for item in mock_data:
            articles.append(NewsArticle(
                title=item["title"],
                summary=item["summary"],
                url=item["link"],
                source_id="mock",
                source_name=item["source"],
                source_type=SourceType.RSS,
                source_tier=SourceTier.TIER_1,
                source_credibility=0.95,
                published_at=datetime.utcnow()
            ))
        return articles

    def _get_mock_trends(self) -> List[Dict]:
        """Return mock trends for testing - specific daily news events."""
        mock_news = [
            {
                "id": "news_001",
                "title": "RBI Mandates New KYC Norms for Digital Lending Apps - 90 Day Deadline",
                "summary": "The Reserve Bank of India today announced stricter KYC requirements for all digital lending platforms. Companies must comply within 90 days or face penalties. This affects over 500 fintech lenders including Lendingkart, Capital Float, and ZestMoney.",
                "link": "https://economictimes.com/news/rbi-kyc-mandate",
                "source": "Economic Times",
                "published": datetime.utcnow().isoformat(),
                "query": "RBI policy announcement"
            },
            {
                "id": "news_002",
                "title": "Swiggy Announces Layoffs of 400 Employees Ahead of IPO",
                "summary": "Food delivery giant Swiggy announced today it will lay off 400 employees as part of cost-cutting measures. The company is focusing on profitability ahead of its planned IPO in Q2 2026. This creates opportunities for HR tech and recruitment firms.",
                "link": "https://moneycontrol.com/news/swiggy-layoffs",
                "source": "Moneycontrol",
                "published": datetime.utcnow().isoformat(),
                "query": "India tech layoffs hiring"
            },
            {
                "id": "news_003",
                "title": "Zepto Raises $200M at $5B Valuation - Expansion to 50 Cities",
                "summary": "Quick commerce startup Zepto closed a $200 million funding round today, valuing the company at $5 billion. Funds will be used to expand dark store network to 50 new cities. This intensifies competition with Blinkit and Instamart.",
                "link": "https://inc42.com/news/zepto-funding",
                "source": "Inc42",
                "published": datetime.utcnow().isoformat(),
                "query": "Indian startup funding announced today"
            },
            {
                "id": "news_004",
                "title": "Cabinet Approves Rs 1.26 Lakh Crore for 3 New Semiconductor Fabs",
                "summary": "The Cabinet today approved setting up of 3 new semiconductor fabrication plants under the India Semiconductor Mission. Tata Electronics and Vedanta are key beneficiaries. Electronics manufacturing sector to see major boost.",
                "link": "https://businessstandard.com/news/semiconductor-approval",
                "source": "Business Standard",
                "published": datetime.utcnow().isoformat(),
                "query": "Indian government scheme launched"
            },
            {
                "id": "news_005",
                "title": "Reliance Jio and NVIDIA Announce AI Cloud Partnership",
                "summary": "Reliance Jio announced a strategic partnership with NVIDIA today to build AI cloud infrastructure in India. Enterprise AI services launching in Q1 2026. This positions Jio against AWS and Azure in the Indian enterprise market.",
                "link": "https://livemint.com/news/jio-nvidia",
                "source": "Mint",
                "published": datetime.utcnow().isoformat(),
                "query": "India business news today"
            }
        ]
        logger.info(f"[RSS] Returning {len(mock_news)} mock news items")
        return mock_news
