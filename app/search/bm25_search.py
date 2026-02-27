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
