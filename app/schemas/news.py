"""
News article data models.

NewsArticle is the atomic unit of the pipeline — fetched from RSS, Tavily,
gnews, or Google News RSS, then filtered, embedded, and clustered.
"""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid import UUID, uuid4

from .base import (
    SourceType, SourceTier, Sector, TrendType,
)


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

    # Classification (populated by ML/LLM)
    detected_sectors: List[Sector] = Field(default_factory=list)
    detected_trend_types: List[TrendType] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    sentiment_score: float = 0.0  # -1.0 to 1.0

    # Embedding for similarity (populated by embedding model)
    embedding: Optional[List[float]] = None  # Combined embedding (title+content), set by source_intel

    # Clustering
    similarity_to_cluster: float = 0.0

    # Quality signals
    is_duplicate: bool = False
    duplicate_of: Optional[UUID] = None
    content_quality_score: float = 0.5
    word_count: int = 0               # For depth_score signal

    model_config = ConfigDict(use_enum_values=True)

    @field_validator('title_normalized', mode='before')
    @classmethod
    def normalize_title(cls, v, info):
        title = (info.data or {}).get('title', '')
        return title.lower().strip() if title else ""

    @field_validator('word_count', mode='before')
    @classmethod
    def compute_word_count(cls, v, info):
        if v and v > 0:
            return v
        data = info.data or {}
        content = data.get('content') or data.get('summary', '')
        return len(content.split()) if content else 0


