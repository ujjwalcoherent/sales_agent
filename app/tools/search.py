"""
BM25 full-text search over already-fetched articles.
Zero API calls. Completely offline. Searches in milliseconds.
Requires: pip install rank-bm25
"""
from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "into", "about", "than", "this", "that", "these",
    "those", "it", "its", "has", "have", "had", "will", "would", "could",
    "should", "may", "might", "can", "do", "does", "did", "not", "no",
}


class BM25Search:
    """
    Searches already-fetched articles using BM25 ranking.

    Use this FIRST before any external search call — it's instant and free.
    Build it once per run from the article list, then reuse.

    Example:
        idx = BM25Search(articles)
        hits = idx.search("gold jewellery suppliers Rajkot", top_k=10)
    """

    def __init__(self, articles: list[dict[str, Any]]):
        self.articles = articles
        self._bm25 = None
        if articles:
            self._build(articles)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    def _build(self, articles: list[dict]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed. Run: pip install rank-bm25")
            return

        def _get(a, *keys):
            """Get value from either a dict or a Pydantic/object by attribute."""
            for k in keys:
                v = a.get(k) if isinstance(a, dict) else getattr(a, k, None)
                if v:
                    return v
            return ""

        corpus = [
            _get(a, "title") + " " + _get(a, "content", "body", "summary")[:500]
            for a in articles
        ]
        tokenized = [self._tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(articles)} articles")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k articles most relevant to query. Returns [] if index empty."""
        if not self._bm25 or not self.articles:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_idx:
            if scores[i] <= 0.1:
                continue
            a = self.articles[i]
            if isinstance(a, dict):
                results.append({**a, "_bm25_score": float(scores[i])})
            else:
                # Pydantic model — convert to dict for downstream consumers
                try:
                    d = a.model_dump()
                except Exception:
                    d = {k: getattr(a, k, None) for k in ("id", "title", "content", "summary", "url", "source_id", "source_name")}
                d["_bm25_score"] = float(scores[i])
                results.append(d)
        return results

    def search_companies(self, segment: str, geo: str = "", top_k: int = 20) -> list[dict]:
        """Specialized company search: segment + geo + company-related terms."""
        query = f"{segment} company {geo} supplier manufacturer exporter".strip()
        return self.search(query, top_k=top_k)

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None


# ── Search Manager (merged from search/manager.py) ───────────────────────────

"""
Unified search manager — orchestrates the fallback chain:
  BM25 article index → DuckDuckGo

Import this everywhere instead of calling tavily_tool.py directly.
Primary web search is handled by web_intel.py (Tavily + DDG fallback).
"""
import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .bm25_search import BM25Search

logger = logging.getLogger(__name__)


class SearchManager:
    """
    Single entry point for all search operations.

    Priority chain:
    1. BM25 article index  — offline, free, instant
    2. DuckDuckGo          — free, fragile (blocks after ~20 req), last resort

    Primary web search (Tavily + DDG fallback) is in web_intel.py.

    Usage:
        mgr = SearchManager(articles=articles)
        results = await mgr.search("gold jewellery suppliers Rajkot")
    """

    def __init__(
        self,
        articles: Optional[list[dict[str, Any]]] = None,
    ):
        # BM25 index built lazily from articles
        self._bm25: Optional["BM25Search"] = None
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
        Run the full fallback chain. Returns {"results": [...], "answer": "", "source": "bm25"|"ddg"}.
        """
        # 1. BM25 (offline, instant)
        if bm25_first and self._bm25 and self._bm25.is_ready:
            bm25_hits = self._bm25.search(query, top_k=max_results)
            if bm25_hits:
                logger.info(f"BM25 returned {len(bm25_hits)} hits for '{query[:40]}'")
                return {"results": bm25_hits, "answer": "", "source": "bm25"}

        # 2. DuckDuckGo (free fallback — fragile, no auth needed)
        try:
            from ddgs import DDGS
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
        Web search via DDG fallback. Async.
        Used by CausalCouncil tool calls when Company KB + BM25 insufficient.
        Returns {"results": [...], "answer": "", "source": "ddg"|"none"}
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
        country = get_settings().country
        query = f"{segment} {geo} company {size_band} {country} supplier {enterprise_terms}".strip()
        return await self.search(query, max_results=max_results)
