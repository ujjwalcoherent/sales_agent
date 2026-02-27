"""
Unified search manager — orchestrates the fallback chain:
  BM25 article index → SearXNG → DuckDuckGo

Import this everywhere instead of calling tavily_tool.py directly.
Tavily is disabled (TAVILY_ENABLED=false in .env).
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .bm25_search import BM25Search

logger = logging.getLogger(__name__)


class SearchManager:
    """
    Single entry point for all search operations.

    Priority chain (configured via .env):
    1. BM25 article index  — offline, free, instant
    2. SearXNG             — self-hosted, free, Google+Bing+DDG
    3. DuckDuckGo          — free, fragile (blocks after ~20 req), last resort

    NOT included here (Tavily is disabled):
    - tavily_tool.py still exists for legacy code, but TAVILY_ENABLED=false routes
      its internal calls to DuckDuckGo automatically.

    Usage:
        mgr = SearchManager(articles=articles, searxng_url="http://localhost:8888", searxng_enabled=False)
        results = await mgr.search("gold jewellery suppliers Rajkot")
    """

    def __init__(
        self,
        articles: list[dict[str, Any]] | None = None,
        searxng_url: str | None = None,
        searxng_enabled: bool | None = None,
    ):
        # Auto-read from config when not explicitly passed
        if searxng_enabled is None or searxng_url is None:
            try:
                from app.config import get_settings
                s = get_settings()
                if searxng_enabled is None:
                    searxng_enabled = s.searxng_enabled
                if searxng_url is None:
                    searxng_url = s.searxng_url
            except Exception:
                searxng_enabled = searxng_enabled if searxng_enabled is not None else False
                searxng_url = searxng_url or "http://localhost:8888"
        self._searxng_url = searxng_url
        self._searxng_enabled = searxng_enabled

        # BM25 index built lazily from articles
        self._bm25: BM25Search | None = None
        if articles:
            self._init_bm25(articles)

    def _init_bm25(self, articles: list[dict[str, Any]]) -> None:
        from .bm25_search import BM25Search
        self._bm25 = BM25Search(articles)

    def update_articles(self, articles: list[dict[str, Any]]) -> None:
        """Rebuild BM25 index with new articles (call at start of each pipeline run)."""
        self._init_bm25(articles)

    async def search(
        self,
        query: str,
        max_results: int = 10,
        bm25_first: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full fallback chain. Returns {"results": [...], "answer": "", "source": "bm25"|"searxng"|"ddg"}.
        """
        # 1. BM25 (offline, instant)
        if bm25_first and self._bm25 and self._bm25.is_ready:
            bm25_hits = self._bm25.search(query, top_k=max_results)
            if bm25_hits:
                logger.info(f"BM25 returned {len(bm25_hits)} hits for '{query[:40]}'")
                return {"results": bm25_hits, "answer": "", "source": "bm25"}

        # 2. SearXNG (self-hosted, no rate limits)
        if self._searxng_enabled:
            from .searxng_search import SearXNGSearch
            searxng = SearXNGSearch(self._searxng_url)
            if await searxng.is_available():
                data = await searxng.search(query, max_results=max_results)
                if data.get("results"):
                    logger.info(f"SearXNG returned {len(data['results'])} results for '{query[:40]}'")
                    return {**data, "source": "searxng"}

        # 3. DuckDuckGo (last resort — fragile, no auth needed)
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            if raw:
                results = [
                    {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
                    for r in raw
                ]
                logger.info(f"DDG returned {len(results)} results for '{query[:40]}'")
                return {"results": results, "answer": "", "source": "ddg"}
        except Exception as e:
            logger.warning(f"DDG search failed: {e}")

        logger.warning(f"All search methods returned empty for: '{query[:50]}'")
        return {"results": [], "answer": "", "source": "none"}

    def set_bm25_index(self, bm25_index) -> None:
        """Attach a pre-built BM25 index (call after fetching articles each run)."""
        self._bm25 = bm25_index

    def search_articles(self, query: str, top_k: int = 15) -> list[dict[str, Any]]:
        """
        BM25 search over already-fetched articles. Synchronous — instant, free.
        Returns [] if no articles indexed yet.
        Used by CausalCouncil tool calls.
        """
        if not self._bm25 or not self._bm25.is_ready:
            return []
        return self._bm25.search(query, top_k=top_k)

    async def web_search(self, query: str, max_results: int = 10) -> dict[str, Any]:
        """
        Web search via SearXNG → DDG fallback. Async.
        Used by CausalCouncil tool calls when Company KB + BM25 insufficient.
        Returns {"results": [...], "answer": "", "source": "searxng"|"ddg"|"none"}
        """
        return await self.search(query, max_results=max_results, bm25_first=False)

    async def search_companies(
        self,
        segment: str,
        geo: str = "",
        size_band: str = "SME",
        max_results: int = 10,
    ) -> dict[str, Any]:
        """
        Company-specific search with SME focus.
        Builds query from segment + geo, excludes enterprise companies.
        """
        try:
            from app.config import get_settings
            blocklist = get_settings().enterprise_blocklist_set
        except Exception:
            blocklist = {"tata", "reliance", "infosys", "wipro", "hcl"}
        enterprise_terms = " ".join(f"NOT {t.capitalize()}" for t in list(blocklist)[:8])
        query = f"{segment} {geo} company {size_band} India supplier {enterprise_terms}".strip()
        return await self.search(query, max_results=max_results)

    async def company_news_search(
        self,
        company_name: str,
        months: int = 5,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for company-specific news via SearXNG news category.

        Returns list of {"title", "url", "content"} dicts.
        Falls back to empty list if SearXNG unavailable.
        """
        if not self._searxng_enabled:
            return []
        try:
            from .searxng_search import SearXNGSearch
            searxng = SearXNGSearch(self._searxng_url)
            time_range = "year" if months > 3 else "month"
            data = await searxng.search(
                f'"{company_name}" India news',
                max_results=max_results,
                categories="news",
                time_range=time_range,
            )
            return data.get("results", [])
        except Exception as e:
            logger.debug(f"Company news search failed for '{company_name}': {e}")
            return []
