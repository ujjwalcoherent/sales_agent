"""
Schemas package — all data models for the India Trend Lead Agent.

Models are organized by domain in submodules:
  - base.py: Common enums and value objects
  - news.py: NewsArticle, NewsSource, Entity, ArticleCluster
  - trends.py: TrendNode, TrendTree, MajorTrend, SignalStrength, TrendDepth
  - pipeline.py: Pipeline state models (LangGraph)
  - sales.py: SectorImpact, CompanyData, ContactData, OutreachEmail, AgentState
  - validation.py: ValidationResult, ValidationVerdict, FieldGroundedness (V10)
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

# pipeline.py — pipeline state
from app.schemas.pipeline import NewsIngestionState, ClusteringState, TrendSynthesisState

# sales.py — sales pipeline models
from app.schemas.sales import (
    SectorImpact, TrendData, ImpactAnalysis,
    CompanyData, ContactData, SeniorityTier, PersonProfile, OutreachEmail,
    LeadRecord, PipelineResult, EmailFinderResult, AgentState,
)

# validation.py — validation models
from app.schemas.validation import (
    ValidationVerdict, FieldGroundedness, ValidationRound, ValidationResult,
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
    # pipeline
    "NewsIngestionState", "ClusteringState", "TrendSynthesisState",
    # sales
    "SectorImpact", "TrendData", "ImpactAnalysis",
    "CompanyData", "ContactData", "SeniorityTier", "PersonProfile", "OutreachEmail",
    "LeadRecord", "PipelineResult", "EmailFinderResult", "AgentState",
    # validation
    "ValidationVerdict", "FieldGroundedness", "ValidationRound", "ValidationResult",
]
