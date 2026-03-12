"""
Tavily Search Tool — primary search provider with advanced capabilities.

Key rotation: set TAVILY_API_KEYS (comma-separated) in .env.
Uses advanced search depth for company research, in-memory LRU cache,
Tavily Extract API for URL content extraction, and finance topic.
"""

import asyncio
import hashlib
import logging
import threading
import time
from typing import List, Dict, Optional, Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Country-specific business news domains (configured per deployment).
# Empty by default — callers should pass include_domains explicitly when needed.
_COUNTRY_BIZ_DOMAINS: list[str] = []

# Low-signal domains always excluded
_NOISE_DOMAINS = [
    "reddit.com", "quora.com", "wikipedia.org",
    "youtube.com", "facebook.com", "twitter.com",
]

# ── In-memory search cache (TTL-based) ────────────────────────────────────
# Two-tier cache: news/general searches expire quickly; company research lasts 24h.

_CACHE_TTL = 300          # 5 minutes — for news and general searches
_CACHE_TTL_COMPANY = 86400  # 24 hours — company facts rarely change intra-day
_CACHE_MAX = 200
_cache: Dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def _cache_key(query: str, **kwargs) -> str:
    """Hash query + params to create a stable cache key."""
    raw = f"{query}|{sorted(kwargs.items())}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str, ttl: int = _CACHE_TTL) -> Optional[dict]:
    """Return cached result if fresh (respects per-call TTL), else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < ttl:
            return entry[1]
        if entry:
            del _cache[key]  # expired
    return None


def _cache_put(key: str, value: dict) -> None:
    """Store result in cache, evicting oldest if full."""
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            # Evict oldest entry
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest_key]
        _cache[key] = (time.time(), value)


class TavilyTool:
    """
    Primary search provider — Tavily API wrapper with advanced capabilities.

    Returns raw Tavily results — or {"error": "...", "results": []} on failure.

    Methods:
      search()                      — general web (supports advanced depth, AI answers, raw content)
      news_search()                 — topic="news" + quality domain filter
      finance_search()              — topic="finance" for stock/funding/revenue data
      extract()                     — Tavily Extract API for clean URL content extraction
      deep_company_research()       — 3 parallel calls for maximum company intelligence
      enrich_trend()                — trend news enrichment
    """

    _key_index = 0
    _lock = threading.Lock()

    def __init__(self, mock_mode: bool = False):
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self._keys = self._load_keys()

    # ── Key management ────────────────────────────────────────────────────────

    def _load_keys(self) -> List[str]:
        """Load API keys from TAVILY_API_KEYS (comma-separated for rotation)."""
        raw = getattr(self.settings, "tavily_api_keys", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()] if raw else []
        if keys:
            logger.info(f"Tavily: {len(keys)} key(s) loaded for rotation")
        return keys

    def _next_key(self) -> str:
        """Round-robin key selection — thread-safe."""
        with self._lock:
            key = self._keys[TavilyTool._key_index % len(self._keys)]
            TavilyTool._key_index += 1
            return key

    @property
    def available(self) -> bool:
        return self.settings.tavily_enabled and bool(self._keys)

    # ── Core search ───────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_answer: bool | str = True,
        topic: str = "general",
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_raw_content: bool | str = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Call Tavily and return the raw result dict.

        On quota exhaustion or API error, returns {"error": "...", "results": []}.

        search_depth:       "basic" (1 credit) | "advanced" (2 credits, 5x more content)
        topic:              "general" | "news" | "finance"
        time_range:         "day" | "week" | "month" | "year"
        include_answer:     True/False | "basic" | "advanced" (multi-paragraph AI summary)
        include_raw_content: True/False | "markdown" (full page content in markdown)
        use_cache:          If True, check in-memory LRU cache before calling API
        """
        if self.mock_mode:
            return self._mock_result(query)
        if not self.available:
            return {"error": "Tavily disabled or no keys configured", "results": []}

        # Check cache — TTL depends on topic (company facts cached 24h, news 5min)
        ck = _cache_key(query, depth=search_depth, max=max_results, topic=topic,
                        time=time_range, answer=str(include_answer), raw=str(include_raw_content))
        cache_ttl = _CACHE_TTL_COMPANY if topic == "general" and not time_range else _CACHE_TTL
        if use_cache:
            cached = _cache_get(ck, ttl=cache_ttl)
            if cached:
                logger.debug(f"Tavily cache hit for '{query[:50]}'")
                return cached

        from tavily import AsyncTavilyClient
        from tavily.errors import UsageLimitExceededError, InvalidAPIKeyError

        backoff = 0.5
        for attempt in range(len(self._keys)):
            key = self._next_key()
            hint = f"...{key[-4:]}"
            try:
                client = AsyncTavilyClient(api_key=key)
                kwargs: Dict[str, Any] = dict(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_answer=include_answer,
                    topic=topic,
                    exclude_domains=exclude_domains or _NOISE_DOMAINS,
                )
                if time_range:
                    kwargs["time_range"] = time_range
                if include_domains:
                    kwargs["include_domains"] = include_domains
                if include_raw_content:
                    kwargs["include_raw_content"] = include_raw_content

                result = await client.search(**kwargs)
                n = len(result.get("results", []))
                logger.info(f"Tavily [{hint}] '{query[:50]}' → {n} results")

                # Cache successful results
                if use_cache and not result.get("error"):
                    _cache_put(ck, result)

                return result

            except (UsageLimitExceededError, InvalidAPIKeyError) as e:
                logger.warning(f"Tavily key {hint} quota/invalid: {e} — rotating (backoff {backoff}s)")
                if attempt < len(self._keys) - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 4.0)  # exponential: 0.5, 1, 2, 4
                continue
            except Exception as e:
                logger.error(f"Tavily [{hint}] error: {e}")
                return {"error": str(e), "results": []}

        logger.warning("All Tavily keys exhausted")
        return {"error": "all_keys_exhausted", "results": []}

    # ── Convenience wrappers ──────────────────────────────────────────────────

    async def news_search(
        self,
        query: str,
        time_range: str = "week",
        max_results: int = 8,
        include_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """News search — topic=news with optional domain filtering.

        Default: searches all domains (no country-specific filtering).
        Pass include_domains=[...] to restrict to specific domains.
        """
        domains = include_domains if include_domains is not None else _COUNTRY_BIZ_DOMAINS
        kwargs: Dict[str, Any] = dict(
            query=query,
            topic="news",
            time_range=time_range,
            max_results=max_results,
        )
        if domains:
            kwargs["include_domains"] = domains
        return await self.search(**kwargs)

    async def finance_search(
        self,
        query: str,
        max_results: int = 5,
        advanced: bool = False,
    ) -> Dict[str, Any]:
        """Financial data search — stock prices, funding rounds, revenue.

        Uses topic="finance" for specialized financial data extraction.
        advanced=True uses search_depth="advanced" (2 credits) for public companies
        with real-time stock data. Default (1 credit) is sufficient for private co's.
        """
        return await self.search(
            query=query,
            topic="finance",
            search_depth="advanced" if advanced else "basic",
            max_results=max_results,
            include_answer="advanced" if advanced else True,
        )

    async def deep_company_research(
        self,
        company_name: str,
        is_public: bool = False,
    ) -> Dict[str, Any]:
        """Full company dossier using 3 parallel Tavily calls for maximum data.

        Runs concurrently:
          1. Advanced company search (AI answer + raw content) — 2 credits
          2. News search (recent company news, all domains) — 1 credit
          3. Finance search — basic (1 credit) for private co, advanced (2 credits) for public

        Pass is_public=True for listed companies to get stock/market-cap data.
        Returns merged dict with keys: answer, results, news, finance.
        """
        async def _company_search():
            return await self.search(
                query=f"{company_name} company overview products services headquarters",
                search_depth="advanced",
                max_results=5,
                include_answer="advanced",
                include_raw_content=True,
            )

        async def _news():
            return await self.news_search(
                query=f"{company_name} company news",
                time_range="month",
                max_results=5,
                include_domains=[],  # all domains for company-specific news
            )

        async def _finance():
            # Private companies have no stock data — basic depth is sufficient
            return await self.finance_search(
                query=f"{company_name} {'stock price market cap' if is_public else 'funding revenue valuation'}",
                max_results=3,
                advanced=is_public,  # Only use 2 credits for public companies
            )

        company_result, news_result, finance_result = await asyncio.gather(
            _company_search(), _news(), _finance(),
            return_exceptions=True,
        )

        # Merge results, handling exceptions gracefully
        merged: Dict[str, Any] = {"company_name": company_name}

        if isinstance(company_result, dict):
            merged["answer"] = company_result.get("answer", "")
            merged["results"] = company_result.get("results", [])
        else:
            logger.debug(f"Company search failed for '{company_name}': {company_result}")
            merged["answer"] = ""
            merged["results"] = []

        if isinstance(news_result, dict):
            merged["news"] = news_result.get("results", [])
        else:
            logger.debug(f"News search failed for '{company_name}': {news_result}")
            merged["news"] = []

        if isinstance(finance_result, dict):
            merged["finance"] = {
                "answer": finance_result.get("answer", ""),
                "results": finance_result.get("results", []),
            }
        else:
            logger.debug(f"Finance search failed for '{company_name}': {finance_result}")
            merged["finance"] = {"answer": "", "results": []}

        return merged

    async def enrich_trend(self, trend_title: str, trend_summary: str) -> Dict[str, Any]:
        """Fetch recent news context for a trend (last week)."""
        country = self.settings.country
        result = await self.news_search(
            f"{trend_title} {country} market impact",
            time_range="week",
            max_results=5,
        )
        return {
            "trend_title": trend_title,
            "original_summary": trend_summary,
            "enriched_context": result.get("answer", ""),
            "sources": [
                {
                    "title": r.get("title", "") or "",
                    "url": r.get("url", "") or "",
                    "snippet": (r.get("content") or "")[:300],
                }
                for r in result.get("results", [])[:3]
            ],
        }

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_result(self, query: str) -> Dict[str, Any]:
        """Offline stub — set MOCK_MODE=false and provide keys for live results."""
        return {
            "answer": f"[MOCK] {query[:80]}",
            "results": [{"title": "Mock — Tavily disabled", "url": "", "content": "", "score": 0}],
        }
