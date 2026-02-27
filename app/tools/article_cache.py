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

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import chromadb

from app.schemas.news import NewsArticle

logger = logging.getLogger(__name__)

# Max metadata value size for ChromaDB (it has a limit)
_MAX_META_STR = 8000


class ArticleCache:
    """Persistent cache for articles + embeddings using ChromaDB."""

    def __init__(self, db_path: str = "./data/article_cache"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"},
        )

    def store_articles(
        self,
        articles: List[NewsArticle],
        embeddings: List[List[float]],
    ) -> int:
        """Store articles + embeddings. Deduplicates by article ID.

        Returns number of new articles stored (skips existing).
        """
        if len(articles) != len(embeddings):
            raise ValueError(
                f"articles ({len(articles)}) and embeddings ({len(embeddings)}) must be same length"
            )

        # Get existing IDs to skip duplicates
        existing = set()
        if self.collection.count() > 0:
            result = self.collection.get(include=[])
            existing = set(result["ids"])

        ids = []
        documents = []
        embeds = []
        metadatas = []
        skipped = 0

        for article, embedding in zip(articles, embeddings):
            aid = str(article.id)
            if aid in existing:
                skipped += 1
                continue

            # Document: searchable text (title + summary)
            doc = f"{article.title}\n{article.summary}"

            # Metadata: serialized article fields for reconstruction
            meta = self._article_to_metadata(article)

            ids.append(aid)
            documents.append(doc[:_MAX_META_STR])
            embeds.append(embedding)
            metadatas.append(meta)

        if not ids:
            logger.info(f"ArticleCache: 0 new articles (all {skipped} already cached)")
            return 0

        # ChromaDB batch add (handles up to 41666 items)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                embeddings=embeds[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info(
            f"ArticleCache: stored {len(ids)} new articles "
            f"(skipped {skipped} existing). Total: {self.collection.count()}"
        )
        return len(ids)

    def load_articles(self) -> Tuple[List[NewsArticle], List[List[float]]]:
        """Load all cached articles + their embeddings.

        Returns (articles, embeddings) in the same order.
        """
        count = self.collection.count()
        if count == 0:
            logger.info("ArticleCache: empty cache")
            return [], []

        result = self.collection.get(
            include=["embeddings", "metadatas"],
            limit=count,
        )

        articles = []
        embeddings = []

        for meta, embedding in zip(result["metadatas"], result["embeddings"]):
            try:
                article = self._metadata_to_article(meta)
                articles.append(article)
                embeddings.append(embedding)
            except Exception as e:
                logger.debug(f"ArticleCache: skip corrupt entry: {e}")

        logger.info(f"ArticleCache: loaded {len(articles)} articles with embeddings")
        return articles, embeddings

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
        """Wipe all cached articles."""
        self.client.delete_collection("articles")
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ArticleCache: cleared")

    # ── Serialization helpers ──

    def _article_to_metadata(self, article: NewsArticle) -> Dict:
        """Convert NewsArticle to ChromaDB metadata dict.

        ChromaDB metadata values must be str, int, float, or bool.
        Complex fields are JSON-serialized.
        """
        meta = {
            "title": article.title[:_MAX_META_STR],
            "summary": article.summary[:_MAX_META_STR],
            "url": article.url,
            "source_id": article.source_id,
            "source_name": article.source_name,
            "source_type": article.source_type if isinstance(article.source_type, str) else article.source_type,
            "source_tier": article.source_tier if isinstance(article.source_tier, str) else article.source_tier,
            "source_credibility": article.source_credibility,
            "published_at": article.published_at.isoformat(),
            "sentiment_score": article.sentiment_score,
            "content_quality_score": article.content_quality_score,
            "word_count": article.word_count,
        }

        # Content (may be large — truncate)
        if article.content:
            meta["content"] = article.content[:_MAX_META_STR]

        # Entity names as JSON list
        if article.entity_names:
            meta["entity_names_json"] = json.dumps(article.entity_names[:50])

        # Keywords
        if article.keywords:
            meta["keywords_json"] = json.dumps(article.keywords[:30])

        # Event classification (stored as private attrs by classifier)
        for attr in ("_trigger_event", "_trigger_urgency", "_trigger_confidence"):
            val = getattr(article, attr, None)
            if val is not None:
                meta[attr.lstrip("_")] = val if isinstance(val, (str, int, float, bool)) else str(val)

        return meta

    def _metadata_to_article(self, meta: Dict) -> NewsArticle:
        """Reconstruct NewsArticle from ChromaDB metadata."""
        published = datetime.fromisoformat(meta["published_at"])
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)

        article = NewsArticle(
            title=meta["title"],
            summary=meta["summary"],
            content=meta.get("content"),
            url=meta["url"],
            source_id=meta["source_id"],
            source_name=meta["source_name"],
            source_type=meta.get("source_type", "rss"),
            source_tier=meta.get("source_tier", "tier_2"),
            source_credibility=meta.get("source_credibility", 0.8),
            published_at=published,
            sentiment_score=meta.get("sentiment_score", 0.0),
            content_quality_score=meta.get("content_quality_score", 0.5),
            word_count=meta.get("word_count", 0),
        )

        # Restore entity names
        if "entity_names_json" in meta:
            try:
                article.entity_names = json.loads(meta["entity_names_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Restore keywords
        if "keywords_json" in meta:
            try:
                article.keywords = json.loads(meta["keywords_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Restore event classification
        if "trigger_event" in meta:
            article._trigger_event = meta["trigger_event"]
        if "trigger_urgency" in meta:
            article._trigger_urgency = float(meta["trigger_urgency"])
        if "trigger_confidence" in meta:
            article._trigger_confidence = float(meta["trigger_confidence"])

        return article
