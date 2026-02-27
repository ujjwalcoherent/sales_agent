"""
News article and source data models.

These models represent the raw material of the pipeline: news articles
fetched from various sources, and the sources themselves.

Hierarchy: NewsSource → NewsArticle → (fed into clustering/trends)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4
import hashlib

from .base import (
    SourceType, SourceTier, Sector, TrendType, MoneyAmount,
)


class NewsSource(BaseModel):
    """News source configuration."""
    id: str
    name: str
    source_type: SourceType = SourceType.RSS
    tier: SourceTier = SourceTier.TIER_2
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.8)

    # Connection
    url: str
    rss_url: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Rate limiting
    rate_limit_per_day: Optional[int] = None

    # Content details
    categories: List[str] = Field(default_factory=list)
    language: str = "en"
    country: str = "IN"

    # Health tracking
    last_fetch_success: bool = True
    consecutive_failures: int = 0

    class Config:
        use_enum_values = True


class Entity(BaseModel):
    """
    Named entity extracted from article text via NER (spaCy).

    Entity types follow spaCy's convention:
    - PERSON: People (Narendra Modi, Elon Musk)
    - ORG: Organizations (RBI, Tata, Google)
    - GPE: Geopolitical entities (India, Mumbai, USA)
    - DATE: Dates (January 15, 2025)
    - EVENT: Named events (Union Budget, Olympics)
    - MONEY: Monetary values (Rs 1.26 lakh crore)
    - PRODUCT: Products (iPhone, Boeing 737)

    salience: How central this entity is to the article (0-1).
    Higher salience = more important to the article's main subject.
    """
    text: str
    type: str                    # PERSON, ORG, GPE, DATE, EVENT, MONEY, PRODUCT
    salience: float = 0.0       # 0.0-1.0: how central to the article


class NewsArticle(BaseModel):
    """
    Raw news article from any source.

    This is the atomic unit of the pipeline. Every article carries its source
    attribution, extracted entities, classification signals, and embeddings.
    """
    id: UUID = Field(default_factory=uuid4)

    # Core content
    title: str
    title_normalized: str = ""
    summary: str
    content: Optional[str] = None
    url: str

    # Source attribution
    source_id: str
    source_name: str
    source_type: SourceType = SourceType.RSS
    source_tier: SourceTier = SourceTier.TIER_2
    source_credibility: float = 0.8

    # Temporal
    published_at: datetime
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Extracted entities (populated by NER — app/news/entity_extractor.py)
    entities: List[Entity] = Field(default_factory=list)
    entity_names: List[str] = Field(default_factory=list)  # Flat list for quick overlap

    # Legacy entity fields (kept for backward compat with existing code)
    mentioned_companies: List[str] = Field(default_factory=list)
    mentioned_people: List[str] = Field(default_factory=list)
    mentioned_locations: List[str] = Field(default_factory=list)
    mentioned_amounts: List[MoneyAmount] = Field(default_factory=list)

    # Classification (populated by ML/LLM)
    detected_sectors: List[Sector] = Field(default_factory=list)
    detected_trend_types: List[TrendType] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    sentiment_score: float = 0.0  # -1.0 to 1.0

    # Embeddings for similarity (populated by embedding model)
    title_embedding: Optional[List[float]] = None
    content_embedding: Optional[List[float]] = None

    # Clustering
    similarity_to_cluster: float = 0.0

    # Quality signals
    is_duplicate: bool = False
    duplicate_of: Optional[UUID] = None
    content_quality_score: float = 0.5
    word_count: int = 0               # For depth_score signal
    novelty_score: float = 0.0        # Filled during dedup phase
    source_trust_score: float = 0.5   # From config source tiers

    @validator('title_normalized', pre=True, always=True)
    def normalize_title(cls, v, values):
        title = values.get('title', '')
        return title.lower().strip() if title else ""

    @validator('word_count', pre=True, always=True)
    def compute_word_count(cls, v, values):
        if v > 0:
            return v
        content = values.get('content') or values.get('summary', '')
        return len(content.split()) if content else 0

    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        content = f"{self.url}:{self.title[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    class Config:
        use_enum_values = True


class ArticleCluster(BaseModel):
    """Groups related articles before trend synthesis."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Clustering metadata
    algorithm_used: str = "dbscan"
    clustering_params: Dict[str, Any] = Field(default_factory=dict)

    # Cluster properties
    centroid_embedding: Optional[List[float]] = None
    coherence_score: float = 0.0

    # Member articles
    article_ids: List[UUID] = Field(default_factory=list)
    article_count: int = 0

    # Extracted common elements
    common_entities: List[str] = Field(default_factory=list)
    common_keywords: List[str] = Field(default_factory=list)
    dominant_sectors: List[Sector] = Field(default_factory=list)
    dominant_trend_type: Optional[TrendType] = None

    # Source diversity (higher = more reliable)
    unique_sources: int = 0
    source_diversity_score: float = 0.0

    # Temporal span
    earliest_article: Optional[datetime] = None
    latest_article: Optional[datetime] = None

    # Quality flags
    is_valid: bool = True
    validation_notes: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
