"""
Schemas package — all data models for the India Trend Lead Agent.

BACKWARD COMPATIBLE: `from app.schemas import NewsArticle` still works.
Models are organized by domain in submodules:
  - base.py: Common enums and value objects
  - news.py: NewsArticle, NewsSource, Entity, ArticleCluster
  - trends.py: TrendNode, TrendTree, MajorTrend, SignalStrength, TrendDepth
  - pipeline.py: Pipeline state models (LangGraph)
  - sales.py: SectorImpact, CompanyData, ContactData, OutreachEmail, AgentState
  - validation.py: ValidationResult, ValidationVerdict, FieldGroundedness (V10)
"""

# Re-export everything for backward compatibility
from app.schemas.base import *
from app.schemas.news import *
from app.schemas.trends import *
from app.schemas.pipeline import *
from app.schemas.sales import *
from app.schemas.validation import *
