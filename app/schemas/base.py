"""
Common enums and value objects used across the entire application.

These are foundational types that don't belong to any specific domain layer.
They define the vocabulary of the system: severity levels, industry sectors,
trend classifications, and source metadata.
"""

from enum import Enum


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
    EARNINGS = "earnings"
    MARKET_MOVEMENT = "market_movement"
    INFRASTRUCTURE = "infrastructure"
    GEOPOLITICAL = "geopolitical"
    CONSUMER_SHIFT = "consumer_shift"
    SUSTAINABILITY = "sustainability"
    CRISIS = "crisis"
    GENERAL = "general"
    EMERGING = "emerging"


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


