"""
BM25 full-text search over already-fetched articles.
Zero API calls. Completely offline. Searches in milliseconds.
Requires: pip install rank-bm25
"""
from __future__ import annotations

import asyncio as _asyncio
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RE_WORD_TOKENS = re.compile(r"\b[a-zA-Z]{3,}\b")

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
        tokens = _RE_WORD_TOKENS.findall(text.lower())
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

# Global DDG semaphore: DDG blocks after ~20 simultaneous requests from same IP.
# Limit to 2 concurrent DDG calls across ALL SearchManager instances.
_DDG_SEMAPHORE = _asyncio.Semaphore(2)


class SearchManager:
    """
    Single entry point for all search operations.

    Priority chain:
    1. BM25 article index  — offline, free, instant
    2. Tavily search       — primary web search (7 API keys, round-robin)
    3. DuckDuckGo          — free, fragile (blocks after ~20 req), last resort

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
        self._bm25 = BM25Search(articles)

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

        # 2. Tavily (primary web search — 7 API keys, round-robin)
        try:
            from app.tools.web.tavily_tool import TavilyTool
            tavily = TavilyTool()
            raw = await tavily.search(query, max_results=min(max_results, 5))
            tavily_hits = raw.get("results", []) if isinstance(raw, dict) else []
            if tavily_hits:
                results = [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                    for r in tavily_hits
                ]
                logger.info(f"Tavily returned {len(results)} results for '{query[:40]}'")
                return {"results": results, "answer": raw.get("answer", ""), "source": "tavily"}
        except Exception as e:
            logger.debug(f"Tavily search failed: {e}")

        # 3. DuckDuckGo (free fallback — rate-limited, max ~20 concurrent from one IP)
        try:
            from ddgs import DDGS
            async with _DDG_SEMAPHORE:
                # Run synchronous DDGS call in executor to avoid blocking event loop
                loop = _asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: list(DDGS().text(query, max_results=max_results)),
                )
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
        Web search via Tavily → DDG fallback chain. Async.
        Used by CausalCouncil tool calls when Company KB + BM25 insufficient.
        Returns {"results": [...], "answer": "", "source": "tavily"|"ddg"|"none"}
        """
        return await self.search(query, max_results=max_results, bm25_first=False)

    async def company_news_search(
        self, company_name: str, months: int = 5, max_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Fetch recent news about a specific company.

        Two-source strategy (BM25 instant + Tavily web):
          1. BM25 over this run's articles — free, instant, high relevance
          2. Tavily web search — broader coverage for news not in RSS feeds
        Used by lead_crystallize_node to attach company-specific context to leads.
        """
        results = []

        # BM25: check already-fetched articles for company mentions (instant, free)
        if self._bm25 and self._bm25.is_ready:
            bm25_hits = self._bm25.search(company_name, top_k=max_results)
            for hit in bm25_hits:
                results.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("url", ""),
                    "summary": (hit.get("summary") or hit.get("content", ""))[:300],
                    "source": "bm25",
                })

        # Tavily/DDG: web search for broader news if BM25 didn't fill quota
        if len(results) < max_results:
            try:
                web = await self.search(
                    f"{company_name} news",
                    max_results=max_results - len(results),
                    bm25_first=False,
                )
                for r in web.get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "summary": (r.get("content") or "")[:300],
                        "source": web.get("source", "web"),
                    })
            except Exception as e:
                logger.debug(f"Company news web search for '{company_name}' failed: {e}")

        return results[:max_results]

