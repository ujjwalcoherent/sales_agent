"""
Schemas package — all data models for the Sales Agent.

Models are organized by domain in submodules:
  - base.py: Common enums (Severity, CompanySize, Sector, TrendType, SourceType, SourceTier)
  - news.py: NewsArticle, Entity
  - sales.py: CompanyData, ContactData, OutreachEmail, AgentState
  - llm_outputs.py: Typed LLM response models with field coercers
  - campaign.py: Campaign configuration and company input models
"""

# base.py — enums
from app.schemas.base import (
    Severity, CompanySize, Sector, TrendType,
    SourceType, SourceTier,
)

# news.py — article models
from app.schemas.news import Entity, NewsArticle

# sales.py — sales pipeline models
from app.schemas.sales import (
    TrendData, ImpactAnalysis,
    CompanyData, ContactData, PersonProfile, OutreachEmail,
    PipelineResult, EmailFinderResult, AgentState,
)

__all__ = [
    # base
    "Severity", "CompanySize", "Sector", "TrendType",
    "SourceType", "SourceTier",
    # news
    "Entity", "NewsArticle",
    # sales
    "TrendData", "ImpactAnalysis",
    "CompanyData", "ContactData", "PersonProfile", "OutreachEmail",
    "PipelineResult", "EmailFinderResult", "AgentState",
]
