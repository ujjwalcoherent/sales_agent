"""
Pipeline state models for LangGraph agent orchestration.

These models track the state of each pipeline phase and the overall
agent state passed between nodes in the LangGraph state machine.
"""

from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field

from .news import NewsArticle, ArticleCluster
from .trends import MajorTrend


class NewsIngestionState(BaseModel):
    """State for news ingestion phase."""
    raw_articles: List[NewsArticle] = Field(default_factory=list)
    articles_fetched: int = 0
    articles_after_dedup: int = 0
    source_health: Dict[str, bool] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class ClusteringState(BaseModel):
    """State for clustering phase."""
    clusters: List[ArticleCluster] = Field(default_factory=list)
    clusters_created: int = 0
    noise_articles: int = 0
    avg_cluster_size: float = 0.0


class TrendSynthesisState(BaseModel):
    """State for trend synthesis phase."""
    major_trends: List[MajorTrend] = Field(default_factory=list)
    trends_synthesized: int = 0
    llm_tokens_used: int = 0
