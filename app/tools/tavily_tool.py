"""
Tavily Search Tool — pure Tavily API wrapper, no fallback logic.

The agent decides what to do when results are empty or an error is returned.
Fallback (DDG/BM25) is a separate tool that agents call explicitly via SearchManager.

Key rotation: set TAVILY_API_KEYS (comma-separated) in .env.
"""

import asyncio
import logging
import threading
from typing import List, Dict, Optional, Any

from ..config import get_settings

logger = logging.getLogger(__name__)

# High-quality Indian business news sources for news_search()
_INDIA_BIZ_DOMAINS = [
    "economictimes.indiatimes.com",
    "livemint.com",
    "businessstandard.com",
    "ndtvprofit.com",
    "moneycontrol.com",
    "financialexpress.com",
    "thehindubusinessline.com",
]

# Low-signal domains always excluded
_NOISE_DOMAINS = [
    "reddit.com", "quora.com", "wikipedia.org",
    "youtube.com", "facebook.com", "twitter.com",
]


class TavilyTool:
    """
    Pure Tavily API wrapper.

    Returns raw Tavily results — or {"error": "...", "results": []} on failure.
    The calling agent reads the result and decides its next action.

    Two search modes:
      search()                     — general web (company research, contacts)
      news_search()                — topic="news" + Indian business domains
      search_companies_concurrent() — parallel fan-out for multi-hop company lookup
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
        include_answer: bool = True,
        topic: str = "general",
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Call Tavily and return the raw result dict.

        On quota exhaustion or API error, returns {"error": "...", "results": []}.
        The agent inspects results and score to decide its next step.

        search_depth: "basic" (1 credit) | "advanced" (2 credits, more content)
        topic:        "general" | "news" | "finance"
        time_range:   "day" | "week" | "month" | "year"
        """
        if self.mock_mode:
            return self._mock_result(query)
        if not self.available:
            return {"error": "Tavily disabled or no keys configured", "results": []}

        from tavily import AsyncTavilyClient
        from tavily.errors import UsageLimitExceededError, InvalidAPIKeyError

        for _ in range(len(self._keys)):
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

                result = await client.search(**kwargs)
                n = len(result.get("results", []))
                logger.info(f"Tavily [{hint}] '{query[:50]}' → {n} results")
                return result

            except (UsageLimitExceededError, InvalidAPIKeyError) as e:
                logger.warning(f"Tavily key {hint} quota/invalid: {e} — rotating")
                continue
            except Exception as e:
                logger.error(f"Tavily [{hint}] error: {e}")
                return {"error": str(e), "results": []}

        logger.warning("All Tavily keys exhausted")
        return {"error": "all_keys_exhausted", "results": []}

    # ── Convenience wrappers ──────────────────────────────────────────────────

    async def news_search(self, query: str, time_range: str = "week", max_results: int = 8) -> Dict[str, Any]:
        """Fresh India business news — topic=news, domain-filtered to quality sources."""
        return await self.search(
            query=query,
            topic="news",
            time_range=time_range,
            max_results=max_results,
            include_domains=_INDIA_BIZ_DOMAINS,
        )

    async def search_companies_concurrent(self, queries: List[str], max_results_each: int = 5) -> List[Dict[str, Any]]:
        """
        Parallel company lookups — fires all queries at once via asyncio.gather().
        Used by causal_council for multi-hop company research without sequential waits.
        Each result may be a dict or an Exception; callers handle both.
        """
        tasks = [self.search(q, max_results=max_results_each, include_answer=False) for q in queries]
        return list(await asyncio.gather(*tasks, return_exceptions=True))

    async def enrich_trend(self, trend_title: str, trend_summary: str) -> Dict[str, Any]:
        """Fetch recent India news context for a trend (last week)."""
        result = await self.news_search(f"{trend_title} India market impact", time_range="week", max_results=5)
        return {
            "trend_title": trend_title,
            "original_summary": trend_summary,
            "enriched_context": result.get("answer", ""),
            "sources": [
                {"title": r.get("title", "") or "", "url": r.get("url", "") or "", "snippet": (r.get("content") or "")[:300]}
                for r in result.get("results", [])[:3]
            ],
        }

    async def find_companies(self, sector: str, company_size: str = "mid", limit: int = 5) -> List[Dict]:
        """Web search for Indian companies in a sector. Returns raw snippets for agent analysis."""
        size_map = {"startup": "early-stage startups", "mid": "mid-sized companies", "enterprise": "large enterprises"}
        result = await self.search(
            query=f"top {size_map.get(company_size, 'companies')} {sector} India",
            max_results=limit + 2,
        )
        return [
            {"title": r.get("title", "") or "", "url": r.get("url", "") or "", "snippet": r.get("content") or "", "score": r.get("score", 0)}
            for r in result.get("results", [])[:limit]
        ]

    async def find_contact(self, company_name: str, role: str) -> Dict[str, Any]:
        """Search for a decision-maker at a company (LinkedIn / news mentions)."""
        result = await self.search(query=f"{role} {company_name} India LinkedIn", max_results=3)
        return {
            "company_name": company_name,
            "target_role": role,
            "answer": result.get("answer", ""),
            "results": [
                {"title": r.get("title", "") or "", "url": r.get("url", "") or "", "snippet": (r.get("content") or "")[:200]}
                for r in result.get("results", [])
            ],
        }

    async def find_company_domain(self, company_name: str) -> Optional[str]:
        """Return the official website domain for an Indian company, or None."""
        from .domain_utils import extract_clean_domain, is_valid_company_domain

        result = await self.search(query=f"{company_name} India official website", max_results=3, include_answer=False)
        for item in result.get("results", []):
            domain = extract_clean_domain(item.get("url", ""))
            if domain and is_valid_company_domain(domain):
                if any(part in domain.lower() for part in company_name.lower().split()[:2]):
                    return domain
        return None

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_result(self, query: str) -> Dict[str, Any]:
        """Offline stub — set MOCK_MODE=false and provide keys for live results."""
        return {
            "answer": f"[MOCK] {query[:80]}",
            "results": [{"title": "Mock — Tavily disabled", "url": "", "content": "", "score": 0}],
        }
