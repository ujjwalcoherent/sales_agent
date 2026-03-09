"""
Schemas package — all data models for the Sales Agent.

Models are organized by domain in submodules:
  - base.py: Common enums and value objects
  - news.py: NewsArticle, NewsSource, Entity, ArticleCluster
  - trends.py: TrendNode, TrendTree, MajorTrend, SignalStrength, TrendDepth
  - sales.py: CompanyData, ContactData, OutreachEmail, AgentState
  - llm_outputs.py: Typed LLM response models with field coercers
  - campaign.py: Campaign configuration and company input models
"""

# base.py — enums and value objects
from app.schemas.base import (
    Severity, CompanySize, Sector, ServiceType, TrendType, ImpactType,
    SourceType, SourceTier, IntentLevel, LifecycleStage,
    BuyingIntentSignalType, BuyingUrgency,
    GeoLocation, MoneyAmount, ConfidenceScore,
)

# news.py — article models
from app.schemas.news import NewsSource, Entity, NewsArticle, ArticleCluster

# trends.py — trend models
from app.schemas.trends import (
    SignalStrength, TrendCorrelation, TrendDepth,
    TopicCluster, TrendEdge, TrendGraph, TrendNode, TrendTree, MajorTrend,
)

# sales.py — sales pipeline models
from app.schemas.sales import (
    TrendData, ImpactAnalysis,
    CompanyData, ContactData, PersonProfile, OutreachEmail,
    LeadRecord, PipelineResult, EmailFinderResult, AgentState,
)

__all__ = [
    # base
    "Severity", "CompanySize", "Sector", "ServiceType", "TrendType", "ImpactType",
    "SourceType", "SourceTier", "IntentLevel", "LifecycleStage",
    "BuyingIntentSignalType", "BuyingUrgency",
    "GeoLocation", "MoneyAmount", "ConfidenceScore",
    # news
    "NewsSource", "Entity", "NewsArticle", "ArticleCluster",
    # trends
    "SignalStrength", "TrendCorrelation", "TrendDepth",
    "TopicCluster", "TrendEdge", "TrendGraph", "TrendNode", "TrendTree", "MajorTrend",
    # sales
    "TrendData", "ImpactAnalysis",
    "CompanyData", "ContactData", "PersonProfile", "OutreachEmail",
    "LeadRecord", "PipelineResult", "EmailFinderResult", "AgentState",
]
