"""
SearXNG REST client — self-hosted meta-search engine.
Deploy with Docker: docker run -d -p 8888:8080 searxng/searxng:latest
No API key. No rate limits. Aggregates Google+Bing+DDG+Brave simultaneously.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SearXNGSearch:
    """
    Query a local SearXNG instance.

    Setup (one-time):
        docker pull searxng/searxng
        docker run -d --name searxng --restart always -p 8888:8080 searxng/searxng:latest

    Then set in .env:
        SEARXNG_ENABLED=true
        SEARXNG_URL=http://localhost:8888
    """

    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url.rstrip("/")

    async def is_available(self) -> bool:
        """Ping SearXNG. Returns False if not running (used for graceful skip)."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/")
                return resp.status_code < 500
        except Exception:
            return False

    async def search(
        self,
        query: str,
        max_results: int = 10,
        time_range: str | None = None,   # "day", "week", "month", "year"
        categories: str = "general",      # "general", "news", "it"
        language: str = "en",
    ) -> dict[str, Any]:
        """
        Query SearXNG. Returns same shape as Tavily: {"results": [...], "answer": ""}.
        Each result: {"title", "url", "content", "engine", "score"}
        """
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "language": language,
            "categories": categories,
        }
        if time_range:
            params["time_range"] = time_range

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{self.base_url}/search", params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.warning(f"SearXNG timeout for query: {query[:50]}")
            return {"answer": "", "results": []}
        except Exception as e:
            logger.warning(f"SearXNG error: {e}")
            return {"answer": "", "results": []}

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "engine": r.get("engine", "searxng"),
                "score": r.get("score", 0.0),
            }
            for r in data.get("results", [])[:max_results]
        ]
        return {"answer": data.get("answer", ""), "results": results}

    async def news_search(self, query: str, max_results: int = 10) -> dict[str, Any]:
        """Search news category specifically."""
        return await self.search(query, max_results=max_results, categories="news", time_range="week")
