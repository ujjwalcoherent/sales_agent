"""
Layer 1: News ingestion and structuring.

Modules:
- fetcher (RSSTool): Multi-source RSS/API article fetching
- scraper: Full article content extraction (trafilatura)
- dedup: MinHash LSH near-duplicate removal
- entity_extractor: spaCy NER entity extraction
- event_classifier: Embedding-based event classification (semantic, no regex)
"""

from app.tools.rss_tool import RSSTool
from app.news.dedup import ArticleDeduplicator
from app.news.entity_extractor import EntityExtractor
from app.news.scraper import scrape_articles
from app.news.event_classifier import EmbeddingEventClassifier
