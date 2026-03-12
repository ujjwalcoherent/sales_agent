"""
Persistent article + embedding cache using ChromaDB.

Stores fetched articles and their embeddings so we can:
1. Skip re-fetching + re-embedding on repeated runs (5-10 min saved)
2. Use cached articles as mock data for threshold tuning
3. Run calibrate.py repeatedly on the same dataset

Usage:
    cache = ArticleCache()
    cache.store_articles(articles, embeddings)   # after first fetch + embed
    articles, embeddings = cache.load_articles()  # instant on subsequent runs
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict
import chromadb


logger = logging.getLogger(__name__)

# Max metadata value size for ChromaDB (it has a limit)
_MAX_META_STR = 8000


class ArticleCache:
    """Persistent cache for articles + embeddings using ChromaDB."""

    def __init__(self, db_path: str = "./data/article_cache"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        # Pipeline articles with custom embeddings (NVIDIA/OpenAI)
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"},
        )
        # News articles auto-embedded by ChromaDB's default model.
        # Separate collection so the two embedding spaces don't mix.
        self.news_collection = self.client.get_or_create_collection(
            name="articles_news",
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        count = self.collection.count()
        if count == 0:
            return {"count": 0, "sources": {}, "date_range": None}

        result = self.collection.get(
            include=["metadatas"],
            limit=count,
        )

        sources: Dict[str, int] = {}
        dates = []
        for meta in result["metadatas"]:
            src = meta.get("source_name", "unknown")
            sources[src] = sources.get(src, 0) + 1
            if "published_at" in meta:
                dates.append(meta["published_at"])

        return {
            "count": count,
            "sources": sources,
            "date_range": (min(dates), max(dates)) if dates else None,
        }

    def clear(self):
        """Wipe all cached articles (both pipeline and news collections).

        NOTE: Not called by the pipeline automatically — available for manual
        cache resets during development/debugging.
        """
        self.client.delete_collection("articles")
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"},
        )
        try:
            self.client.delete_collection("articles_news")
        except ValueError:
            pass  # Collection doesn't exist yet
        self.news_collection = self.client.get_or_create_collection(
            name="articles_news",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ArticleCache: cleared")

    # ── Lightweight storage (no custom embeddings required) ──

    def url_exists(self, url: str) -> bool:
        """Check if a news article with this URL is already stored.

        Uses metadata filter on the news collection.
        Returns False on empty collection or any error.
        """
        try:
            if self.news_collection.count() == 0:
                return False
            result = self.news_collection.get(
                where={"url": url},
                limit=1,
                include=[],
            )
            return bool(result and result.get("ids"))
        except Exception as e:
            logger.debug(f"ArticleCache.url_exists check failed: {e}")
            return False

    def store_news_article(
        self,
        title: str,
        url: str,
        source_name: str = "",
        published_at: str = "",
        content: str = "",
        summary: str = "",
        company_name: str = "",
    ) -> bool:
        """Store a single news article using ChromaDB's default auto-embedding.

        Writes to ``news_collection`` (not the pipeline collection) so the
        auto-generated embeddings don't mix with custom pipeline embeddings.

        Returns True if stored, False if duplicate or on error.
        """
        if not url:
            return False

        # Validate title length (reject too short or too long)
        title = (title or "").strip()
        if len(title) < 10 or len(title) > 500:
            logger.debug(f"ArticleCache: rejected article with title length {len(title)}: {title[:50]}")
            return False

        # Validate date format if provided
        if published_at:
            try:
                datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                published_at = ""  # Clear invalid dates rather than storing garbage

        try:
            # B4: Normalize URL (strip tracking params) for better dedup
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            parsed = urlparse(url.rstrip("/").lower())
            clean_params = {
                k: v for k, v in parse_qs(parsed.query).items()
                if not k.startswith(("utm_", "ref", "source", "campaign", "fbclid", "gclid"))
            }
            norm_url = urlunparse(parsed._replace(query=urlencode(clean_params, doseq=True), fragment=""))
            doc_id = hashlib.sha256(norm_url.encode()).hexdigest()[:24]

            # Fast dedup
            existing = self.news_collection.get(ids=[doc_id], include=[])
            if existing and existing.get("ids"):
                return False

            doc_text = f"{title}\n{summary or (content[:500] if content else '')}"
            if not doc_text.strip():
                doc_text = url

            metadata: Dict = {
                "title": title[:_MAX_META_STR],
                "url": url,
                "source_name": source_name or "",
                "published_at": published_at or "",
                "summary": (summary or "")[:_MAX_META_STR],
                "company_name": company_name.lower() if company_name else "",
                "source_type": "news_collector",
            }
            if content:
                metadata["content"] = content[:_MAX_META_STR]

            self.news_collection.add(
                ids=[doc_id],
                documents=[doc_text[:_MAX_META_STR]],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.debug(f"ArticleCache.store_news_article failed for {url[:80]}: {e}")
            return False

