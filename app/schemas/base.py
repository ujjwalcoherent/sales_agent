"""
Common enums and value objects used across the entire application.

These are foundational types that don't belong to any specific domain layer.
They define the vocabulary of the system: severity levels, industry sectors,
service types, trend classifications, and reusable value objects.
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS - Classification Types
# ══════════════════════════════════════════════════════════════════════════════

class Severity(str, Enum):
    """Trend severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class CompanySize(str, Enum):
    """Company size categories."""
    STARTUP = "startup"           # < 50 employees
    SMB = "smb"                   # 50-200 employees
    MID = "mid"                   # 200-1000 employees
    MID_MARKET = "mid_market"     # 200-1000 employees
    ENTERPRISE = "enterprise"     # 1000-10000 employees
    LARGE_ENTERPRISE = "large_enterprise"  # > 10000 employees


class Sector(str, Enum):
    """12 Primary Industry Sectors (GICS-inspired)."""
    IT_TECHNOLOGY = "IT & Technology"
    BFSI = "Banking, Financial Services & Insurance"
    HEALTHCARE_PHARMA = "Healthcare & Pharmaceuticals"
    MANUFACTURING = "Manufacturing"
    ENERGY_UTILITIES = "Energy & Utilities"
    RETAIL_FMCG = "Retail & FMCG"
    REAL_ESTATE = "Real Estate & Construction"
    AGRICULTURE = "Agriculture & Agritech"
    TELECOM = "Telecommunications"
    AUTOMOTIVE = "Automotive"
    EDUCATION = "Education & EdTech"
    LOGISTICS = "Logistics & Supply Chain"


class ServiceType(str, Enum):
    """9 CMI Core Services."""
    PROCUREMENT_INTELLIGENCE = "Procurement Intelligence"
    MARKET_INTELLIGENCE = "Market Intelligence"
    COMPETITIVE_INTELLIGENCE = "Competitive Intelligence"
    MARKET_MONITORING = "Market Monitoring"
    INDUSTRY_ANALYSIS = "Industry Analysis"
    TECHNOLOGY_RESEARCH = "Technology Research"
    CROSS_BORDER_EXPANSION = "Cross Border Expansion"
    CONSUMER_INSIGHTS = "Consumer Insights"
    CONSULTING_ADVISORY = "Consulting and Advisory Services"


class TrendType(str, Enum):
    """Trend classification by business impact."""
    REGULATION = "regulation"
    POLICY = "policy"
    FUNDING = "funding"
    MARKET = "market"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    PARTNERSHIP = "partnership"
    EXPANSION = "expansion"
    LAYOFFS = "layoffs"
    HIRING = "hiring"
    PRODUCT_LAUNCH = "product_launch"
    IPO = "ipo"
    BANKRUPTCY = "bankruptcy"
    TECHNOLOGY = "technology"
    SUPPLY_CHAIN = "supply_chain"
    PRICE_CHANGE = "price_change"
    GENERAL = "general"
    EMERGING = "emerging"


class ImpactType(str, Enum):
    """Direction of impact."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    DISRUPTIVE = "disruptive"


class SourceType(str, Enum):
    """Data source type."""
    RSS = "rss"
    API = "api"
    SCRAPE = "scrape"
    GOVERNMENT = "government"
    SOCIAL = "social"
    MANUAL = "manual"


class SourceTier(str, Enum):
    """
    Source credibility tier.

    WHY these tiers: News sources vary wildly in editorial standards.
    Government sources (RBI, SEBI) and wire services (PTI, Reuters) are
    highest credibility. Industry blogs and aggregators are lower.
    These tiers directly feed the authority_weighted signal in trend scoring.

    Tier 1 (0.95-1.0): Government, wire services, major publications (ET, Mint)
    Tier 2 (0.85-0.94): Reputable business news (Inc42, VCCircle)
    Tier 3 (0.70-0.84): Industry blogs, niche sources
    Tier 4 (0.50-0.69): Social media, unverified aggregators
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    UNKNOWN = "unknown"


class IntentLevel(str, Enum):
    """Buying intent classification."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DORMANT = "dormant"


class LifecycleStage(str, Enum):
    """Trend lifecycle stage classification."""
    EMERGING = "emerging"
    GROWING = "growing"
    PEAK = "peak"
    DECLINING = "declining"


class BuyingIntentSignalType(str, Enum):
    """Buying intent signal types from synthesis."""
    COMPLIANCE_NEED = "compliance_need"
    GROWTH_OPPORTUNITY = "growth_opportunity"
    CRISIS_RESPONSE = "crisis_response"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    MARKET_ENTRY = "market_entry"
    RESTRUCTURING = "restructuring"
    PROCUREMENT_OPTIMIZATION = "procurement_optimization"
    COMPETITIVE_PRESSURE = "competitive_pressure"
    UNKNOWN = "unknown"


class BuyingUrgency(str, Enum):
    """Buying urgency levels from synthesis."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# VALUE OBJECTS - Immutable, Reusable
# ══════════════════════════════════════════════════════════════════════════════

class GeoLocation(BaseModel):
    """Geographic location."""
    country: str = "India"
    state: str | None = None
    city: str | None = None

    class Config:
        frozen = True


class MoneyAmount(BaseModel):
    """Financial amount with currency."""
    amount: float
    currency: str = "INR"
    is_estimated: bool = False

    class Config:
        frozen = True


class ConfidenceScore(BaseModel):
    """Confidence with explanation factors."""
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    factors: List[str] = Field(default_factory=list)

    @property
    def level(self) -> str:
        if self.score >= 0.9:
            return "very_high"
        if self.score >= 0.75:
            return "high"
        if self.score >= 0.5:
            return "medium"
        if self.score >= 0.25:
            return "low"
        return "very_low"
