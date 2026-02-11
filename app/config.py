"""
Configuration management for India Trend Lead Agent.
Supports Ollama (local), Gemini (cloud), and Groq (cloud) LLM providers.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    # Provider priority: NVIDIA → Ollama → OpenRouter → Gemini
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_model: str = Field(default="moonshotai/kimi-k2.5", alias="NVIDIA_MODEL")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")

    use_ollama: bool = Field(default=True, alias="USE_OLLAMA")
    ollama_model: str = Field(default="mistral", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Groq Configuration (for 120B reasoning model)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")

    # OpenRouter Configuration (multi-model API)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="google/gemini-2.0-flash-001", alias="OPENROUTER_MODEL")

    # Gemini Configuration
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # Embedding Configuration
    huggingface_api_key: str = Field(default="", alias="HF_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", alias="LOCAL_EMBEDDING_MODEL")

    # ── Trend Engine Pipeline (RecursiveTrendEngine) ──
    # UMAP: Dimensionality reduction before HDBSCAN clustering
    umap_n_components: int = Field(default=5, alias="UMAP_N_COMPONENTS")
    umap_n_neighbors: int = Field(default=15, alias="UMAP_N_NEIGHBORS")
    umap_min_dist: float = Field(default=0.0, alias="UMAP_MIN_DIST")
    umap_metric: str = Field(default="cosine", alias="UMAP_METRIC")

    # HDBSCAN: Density-based clustering (no eps tuning needed)
    # NOTE: min_cluster_size is now ADAPTIVE in the engine based on article count.
    # This value is the FLOOR (minimum allowed). Engine calculates adaptive value.
    # For 500 articles, adaptive value will be ~15-20 to get 15-20 major clusters.
    hdbscan_min_cluster_size: int = Field(default=5, alias="HDBSCAN_MIN_CLUSTER_SIZE")
    hdbscan_min_samples: int = Field(default=2, alias="HDBSCAN_MIN_SAMPLES")  # Lowered from 3 for better sub-clustering
    hdbscan_cluster_selection: str = Field(default="eom", alias="HDBSCAN_CLUSTER_SELECTION")

    # Deduplication: MinHash LSH near-duplicate detection (lexical)
    # 0.25 = very aggressive, catches articles with ~25% shared word bigrams (RECOMMENDED)
    # 0.3 = aggressive, catches articles with ~30% shared word bigrams
    # 0.5 = moderate, catches near-identical with some variation
    # Combined with title-based, entity-based, and semantic dedup for comprehensive coverage
    dedup_threshold: float = Field(default=0.25, alias="DEDUP_THRESHOLD")
    dedup_num_perm: int = Field(default=128, alias="DEDUP_NUM_PERM")
    dedup_shingle_size: int = Field(default=2, alias="DEDUP_SHINGLE_SIZE")

    # Deduplication: Semantic (embedding-based) - catches cross-source duplicates
    # OBSERVED SIMILARITY RANGES (from debug logs):
    # - True duplicates (same story, different source): 0.75-0.85
    # - Related but different articles: 0.65-0.74
    # - Unrelated articles: 0.50-0.64
    # THRESHOLD SELECTION:
    # 0.78 = catches true duplicates, avoids most false positives (RECOMMENDED)
    # 0.70 = too aggressive, causes false positives on related articles
    # 0.85 = too conservative, misses cross-source duplicates
    semantic_dedup_threshold: float = Field(default=0.78, alias="SEMANTIC_DEDUP_THRESHOLD")

    # Entity extraction: spaCy NER model
    spacy_model: str = Field(default="en_core_web_sm", alias="SPACY_MODEL")

    # Engine: Pipeline-level settings
    engine_max_depth: int = Field(default=3, alias="ENGINE_MAX_DEPTH")
    engine_max_concurrent_llm: int = Field(default=6, alias="ENGINE_MAX_CONCURRENT_LLM")

    # Signal weights for actionability scoring (JSON string, override via env)
    # These determine how trends are ranked for sales outreach.
    # Weights should sum to ~1.0. Adjust to tune which signals matter most.
    actionability_weights: str = Field(
        default='{"recency":0.12,"velocity":0.08,"specificity":0.10,"regulatory":0.10,"trigger":0.12,"diversity":0.08,"authority":0.12,"financial":0.05,"person":0.03,"event_focus":0.05,"search_interest":0.15}',
        alias="ACTIONABILITY_WEIGHTS",
    )

    # ── Temporal Histogram (BERTopic topics_over_time approach) ──
    # Number of time bins for temporal histogram (sparkline data).
    # 8 bins balances granularity vs noise. Use 4-12 for different resolutions.
    temporal_histogram_bins: int = Field(default=8, alias="TEMPORAL_HISTOGRAM_BINS")
    # Recency decay lambda for BERTrend exponential decay: e^(-lambda * hours²)
    # 0.003 = 6hr→0.9, 24hr→0.18, 48hr→0.001
    recency_decay_lambda: float = Field(default=0.003, alias="RECENCY_DECAY_LAMBDA")
    # Momentum classification thresholds (applied to last N bins of velocity_history)
    # "spiking" if max bin velocity > spike_multiplier × mean velocity
    momentum_spike_multiplier: float = Field(default=3.0, alias="MOMENTUM_SPIKE_MULTIPLIER")
    # Number of trailing bins to evaluate for momentum direction
    momentum_window_bins: int = Field(default=3, alias="MOMENTUM_WINDOW_BINS")

    # ── LLM Synthesis Quality (T5) ──
    # Max articles to include in synthesis context (per cluster)
    synthesis_max_articles: int = Field(default=16, alias="SYNTHESIS_MAX_ARTICLES")
    # Max characters per article in synthesis context
    synthesis_article_char_limit: int = Field(default=1200, alias="SYNTHESIS_ARTICLE_CHAR_LIMIT")
    # Max retries on synthesis failure before returning empty
    synthesis_max_retries: int = Field(default=2, alias="SYNTHESIS_MAX_RETRIES")
    
    # ── LLM JSON Validation (V2) ──
    llm_json_max_retries: int = Field(default=2, alias="LLM_JSON_MAX_RETRIES")

    # ── Synthesis Validation (V3) ──
    synthesis_strict_mode: bool = Field(default=True, alias="SYNTHESIS_STRICT_MODE")

    # ── Event Classifier (V6) ──
    event_classifier_threshold: float = Field(default=0.35, alias="EVENT_CLASSIFIER_THRESHOLD")
    event_classifier_ambiguity_margin: float = Field(default=0.05, alias="EVENT_CLASSIFIER_AMBIGUITY_MARGIN")

    # ── Quality Gates (V9) ──
    min_synthesis_confidence: float = Field(default=0.3, alias="MIN_SYNTHESIS_CONFIDENCE")
    min_trend_confidence_for_agents: float = Field(default=0.25, alias="MIN_TREND_CONFIDENCE_FOR_AGENTS")

    # ── Cross-Validation (V10) ──
    # ValidatorAgent: scores LLM synthesis groundedness against source articles.
    # Enable/disable the validator (disable to save LLM calls during development)
    validator_enabled: bool = Field(default=True, alias="VALIDATOR_ENABLED")
    # Max back-and-forth rounds between synthesizer and validator (1 = validate only, 2+ = revise)
    validator_max_rounds: int = Field(default=2, alias="VALIDATOR_MAX_ROUNDS")
    # Overall groundedness score threshold to PASS (0.0-1.0). Below this = REVISE.
    validator_pass_threshold: float = Field(default=0.6, alias="VALIDATOR_PASS_THRESHOLD")
    # Overall groundedness score below which we REJECT outright (no revision attempt).
    validator_reject_threshold: float = Field(default=0.25, alias="VALIDATOR_REJECT_THRESHOLD")
    # Minimum entity overlap ratio (claimed entities found in sources) to pass entity check.
    validator_entity_overlap_min: float = Field(default=0.4, alias="VALIDATOR_ENTITY_OVERLAP_MIN")
    # Weight for NER entity overlap in overall score (0.0-1.0)
    validator_weight_entity: float = Field(default=0.35, alias="VALIDATOR_WEIGHT_ENTITY")
    # Weight for keyword overlap in overall score (0.0-1.0)
    validator_weight_keyword: float = Field(default=0.30, alias="VALIDATOR_WEIGHT_KEYWORD")
    # Weight for embedding similarity in overall score (0.0-1.0)
    validator_weight_embedding: float = Field(default=0.35, alias="VALIDATOR_WEIGHT_EMBEDDING")

    # ── Company Verification (V7) ──
    company_min_verification_confidence: float = Field(default=0.0, alias="COMPANY_MIN_VERIFICATION_CONFIDENCE")

    # Search APIs
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    
    # Email Finder APIs
    apollo_api_key: str = Field(default="", alias="APOLLO_API_KEY")
    hunter_api_key: str = Field(default="", alias="HUNTER_API_KEY")
    
    # Application Settings
    country: str = Field(default="India", alias="COUNTRY")
    country_code: str = Field(default="IN", alias="COUNTRY_CODE")
    max_trends: int = Field(default=3, alias="MAX_TRENDS")
    max_companies_per_trend: int = Field(default=3, alias="MAX_COMPANIES_PER_TREND")
    max_contacts_per_company: int = Field(default=2, alias="MAX_CONTACTS_PER_COMPANY")
    email_confidence_threshold: int = Field(default=70, alias="EMAIL_CONFIDENCE_THRESHOLD")
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./leads.db", 
        alias="DATABASE_URL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration based on settings.

        Priority: NVIDIA → Ollama → OpenRouter → Gemini
        """
        if self.nvidia_api_key:
            return {
                "provider": "nvidia",
                "api_key": self.nvidia_api_key,
                "model": self.nvidia_model,
                "base_url": self.nvidia_base_url
            }
        elif self.use_ollama:
            return {
                "provider": "ollama",
                "model": self.ollama_model,
                "base_url": self.ollama_base_url
            }
        elif self.openrouter_api_key:
            return {
                "provider": "openrouter",
                "api_key": self.openrouter_api_key,
                "model": self.openrouter_model
            }
        else:
            return {
                "provider": "gemini",
                "api_key": self.gemini_api_key,
                "model": self.gemini_model
            }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_domestic_source_ids(country_code: str = "") -> set:
    """Get source IDs configured for the target country.

    Checks NEWS_SOURCES[*]["country"] against the given ISO 3166-1 alpha-2 code.
    Dynamic — works for any country code (IN, BR, US, DE, etc.).
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    if not country_code:
        country_code = get_settings().country_code
    code_upper = country_code.upper()
    domestic = set()
    missing_country = []
    for src_id, cfg in NEWS_SOURCES.items():
        src_country = cfg.get("country", "")
        if not src_country:
            missing_country.append(src_id)
            continue
        if src_country.upper() == code_upper:
            domestic.add(src_id)
    if missing_country:
        _log.warning(f"Sources missing 'country' prop: {missing_country}")
    return domestic


# RSS Feed queries for DAILY Indian business news (specific events, not generic trends)
RSS_QUERIES = [
    # Breaking business news
    "India business news today",
    "Indian startup funding announced today",
    "India company acquisition merger",
    "RBI policy announcement",
    "Indian government scheme launched",
    "India regulatory change business",
    "Indian unicorn news",
    "India IPO listing news",
    # Sector-specific breaking news
    "India fintech regulation news",
    "India EV policy announcement",
    "India pharma approval news",
    "India tech layoffs hiring",
]

# Trend type to target role mapping for consulting services
TREND_ROLE_MAPPING = {
    "regulation": ["CEO", "Chief Strategy Officer", "VP Strategy", "Director of Business Development"],
    "policy": ["CEO", "COO", "Chief Strategy Officer", "Director Corporate Strategy"],
    "trade": ["VP Supply Chain", "Procurement Director", "Chief Procurement Officer", "Director Sourcing"],
    "market_shift": ["CMO", "VP Marketing", "Chief Strategy Officer", "Director Market Research"],
    "competition": ["CEO", "Chief Strategy Officer", "VP Business Development", "Director Strategy"],
    "technology": ["CTO", "VP Engineering", "Chief Digital Officer", "Director Innovation"],
    "expansion": ["CEO", "VP Business Development", "Chief Strategy Officer", "Director International"],
    "supply_chain": ["COO", "VP Operations", "Chief Procurement Officer", "Director Supply Chain"],
    "funding": ["CEO", "CFO", "Chief Strategy Officer", "VP Corporate Development"],
    "consumer": ["CMO", "VP Marketing", "Chief Customer Officer", "Director Consumer Insights"],
    "default": ["CEO", "Chief Strategy Officer", "VP Business Development", "Director Strategy"]
}

# Coherent Market Insights Service Catalog
CMI_SERVICES = {
    "procurement_intelligence": {
        "name": "Procurement Intelligence",
        "offerings": [
            "Supplier identification and profiling",
            "Cost structure and should-cost analysis",
            "Commodity and category market analysis",
            "Supply base risk assessment",
            "Procurement process optimization"
        ],
        "keywords": ["supply chain", "procurement", "supplier", "sourcing", "cost", "vendor", "raw material"]
    },
    "market_intelligence": {
        "name": "Market Intelligence",
        "offerings": [
            "Market sizing and segmentation",
            "Market trends and growth forecasts",
            "Regulatory and policy landscape assessment",
            "Market entry and expansion feasibility",
            "Trade Analysis (Export-import analysis)"
        ],
        "keywords": ["market", "growth", "expansion", "entry", "fta", "trade", "export", "import", "demand"]
    },
    "competitive_intelligence": {
        "name": "Competitive Intelligence",
        "offerings": [
            "Competitor profiling and benchmarking",
            "Analysis of competitor strategies",
            "Product and service comparisons",
            "Tracking competitor activities",
            "M&A and partnership tracking"
        ],
        "keywords": ["competitor", "competition", "merger", "acquisition", "market share", "benchmark"]
    },
    "market_monitoring": {
        "name": "Market Monitoring",
        "offerings": [
            "Real-time updates on regulatory changes",
            "Monitoring competitor and supplier activities",
            "Alerts on key market events",
            "Early warning systems for emerging risks"
        ],
        "keywords": ["regulation", "policy", "compliance", "disruption", "risk", "change", "update"]
    },
    "industry_analysis": {
        "name": "Industry Analysis",
        "offerings": [
            "Industry structure and value chain mapping",
            "Key industry drivers and challenges",
            "Regulatory and compliance environment review",
            "Demand and supply dynamics assessment"
        ],
        "keywords": ["industry", "sector", "manufacturing", "pharma", "automotive", "electronics", "chemical"]
    },
    "technology_research": {
        "name": "Technology Research",
        "offerings": [
            "Technology landscape and trends analysis",
            "Assessment of emerging technologies",
            "Technology adoption and impact studies",
            "Patent and intellectual property analysis"
        ],
        "keywords": ["technology", "AI", "automation", "digital", "innovation", "R&D", "tech", "software"]
    },
    "cross_border_expansion": {
        "name": "Cross Border Expansion",
        "offerings": [
            "Market entry strategy and feasibility studies",
            "Regulatory and compliance advisory",
            "Local partner and supplier identification",
            "Go-to-market planning and localization"
        ],
        "keywords": ["expansion", "international", "export", "import", "FTA", "global", "cross-border", "foreign"]
    },
    "consumer_insights": {
        "name": "Consumer Insights",
        "offerings": [
            "Consumer behavior and attitude analysis",
            "Segmentation and persona development",
            "Brand perception and loyalty studies",
            "Customer satisfaction tracking"
        ],
        "keywords": ["consumer", "customer", "brand", "retail", "FMCG", "D2C", "e-commerce"]
    },
    "consulting_advisory": {
        "name": "Consulting and Advisory Services",
        "offerings": [
            "Strategic planning and business transformation",
            "Operational efficiency and process optimization",
            "Technology and digital transformation advisory",
            "Market entry and growth strategy"
        ],
        "keywords": ["strategy", "transformation", "growth", "efficiency", "optimization", "advisory"]
    }
}

# Company size targeting for consulting
TARGET_COMPANY_SIZE = {
    "min_employees": 50,
    "max_employees": 300,
    "size_keywords": ["mid-size", "growing", "emerging", "scaling", "series B", "series C", "established"]
}

# Blacklisted domains (not company domains)
BLACKLISTED_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "google.com", "youtube.com", "wikipedia.org",
    "crunchbase.com", "bloomberg.com", "reuters.com",
    "economictimes.com", "moneycontrol.com", "livemint.com",
    "businesstoday.in", "yourstory.com", "inc42.com",
    "github.com", "medium.com", "quora.com"
}

# Company size keywords for classification
COMPANY_SIZE_KEYWORDS = {
    "startup": ["startup", "seed", "early-stage", "series a", "series b", "founded 202"],
    "mid": ["mid-size", "growing", "series c", "series d", "scale-up"],
    "enterprise": ["enterprise", "large", "multinational", "fortune", "listed", "ipo", "public"]
}


# ══════════════════════════════════════════════════════════════════════════════
# NEWS SOURCES - Free/Open-Source APIs & RSS Feeds
# ══════════════════════════════════════════════════════════════════════════════

NEWS_SOURCES = {
    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1: Major Business Publications (RSS - Unlimited, Free)
    # ─────────────────────────────────────────────────────────────────────────
    "economic_times": {
        "id": "economic_times",
        "name": "Economic Times",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://economictimes.indiatimes.com",
        "rss_url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        "categories": ["business", "economy", "markets"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None  # Unlimited
    },
    "et_industry": {
        "id": "et_industry",
        "name": "ET Industry",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://economictimes.indiatimes.com/industry",
        "rss_url": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
        "categories": ["industry", "manufacturing", "sectors"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "et_tech": {
        "id": "et_tech",
        "name": "ET Tech",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://economictimes.indiatimes.com/tech",
        "rss_url": "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
        "categories": ["technology", "startups", "IT"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "livemint": {
        "id": "livemint",
        "name": "Mint",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://www.livemint.com",
        "rss_url": "https://www.livemint.com/rss/news",
        "categories": ["business", "finance", "economy"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "mint_companies": {
        "id": "mint_companies",
        "name": "Mint Companies",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://www.livemint.com/companies",
        "rss_url": "https://www.livemint.com/rss/companies",
        "categories": ["companies", "corporate", "earnings"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "business_standard": {
        "id": "business_standard",
        "name": "Business Standard",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.business-standard.com",
        "rss_url": "https://www.business-standard.com/rss/home_page_top_stories.rss",
        "categories": ["business", "markets", "economy"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bs_companies": {
        "id": "bs_companies",
        "name": "BS Companies",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.business-standard.com/companies",
        "rss_url": "https://www.business-standard.com/rss/companies-101.rss",
        "categories": ["companies", "corporate"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "moneycontrol": {
        "id": "moneycontrol",
        "name": "Moneycontrol",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.92,
        "url": "https://www.moneycontrol.com",
        "rss_url": "https://www.moneycontrol.com/rss/latestnews.xml",
        "categories": ["markets", "finance", "business"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "financial_express": {
        "id": "financial_express",
        "name": "Financial Express",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.92,
        "url": "https://www.financialexpress.com",
        "rss_url": "https://www.financialexpress.com/feed/",
        "categories": ["finance", "economy", "industry"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2: Startup & Tech News (RSS - Unlimited, Free)
    # ─────────────────────────────────────────────────────────────────────────
    "yourstory": {
        "id": "yourstory",
        "name": "YourStory",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://yourstory.com",
        "rss_url": "https://yourstory.com/feed",
        "categories": ["startups", "funding", "entrepreneurship"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "inc42": {
        "id": "inc42",
        "name": "Inc42",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://inc42.com",
        "rss_url": "https://inc42.com/feed/",
        "categories": ["startups", "funding", "tech"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "vccircle": {
        "id": "vccircle",
        "name": "VCCircle",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.vccircle.com",
        "rss_url": "https://www.vccircle.com/feed/",
        "categories": ["funding", "PE", "VC", "M&A"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "entrackr": {
        "id": "entrackr",
        "name": "Entrackr",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://entrackr.com",
        "rss_url": "https://entrackr.com/feed/",
        "categories": ["startups", "funding", "tech"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1: Government & Regulatory (Official - Free, Unlimited)
    # ─────────────────────────────────────────────────────────────────────────
    "pib": {
        "id": "pib",
        "name": "Press Information Bureau",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://pib.gov.in",
        "rss_url": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
        "categories": ["government", "policy", "announcements"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "rbi_press": {
        "id": "rbi_press",
        "name": "RBI Press Releases",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://www.rbi.org.in",
        "rss_url": "https://www.rbi.org.in/pressreleases.rss",
        "categories": ["banking", "finance", "regulation"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "sebi": {
        "id": "sebi",
        "name": "SEBI Press Releases",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://www.sebi.gov.in",
        "rss_url": "https://www.sebi.gov.in/sebiweb/home/RSSFeed.jsp?cat=pr&type=p",
        "categories": ["markets", "regulation", "capital"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2: Industry-Specific (RSS - Free)
    # ─────────────────────────────────────────────────────────────────────────
    "hindu_business": {
        "id": "hindu_business",
        "name": "The Hindu Business Line",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://www.thehindubusinessline.com",
        "rss_url": "https://www.thehindubusinessline.com/feeder/default.rss",
        "categories": ["business", "economy", "industry"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "business_today": {
        "id": "business_today",
        "name": "Business Today",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://www.businesstoday.in",
        "rss_url": "https://www.businesstoday.in/rssfeeds/latest-news.xml",
        "categories": ["business", "companies", "economy"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "ndtv_profit": {
        "id": "ndtv_profit",
        "name": "NDTV Profit",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.ndtvprofit.com",
        "rss_url": "https://feeds.feedburner.com/ndtvprofit-latest",
        "categories": ["markets", "business", "economy"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "techcrunch_india": {
        "id": "techcrunch_india",
        "name": "TechCrunch (India tag)",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://techcrunch.com/tag/india/",
        "rss_url": "https://techcrunch.com/tag/india/feed/",
        "categories": ["tech", "startups", "funding"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # FREE APIs (Rate Limited but Free) - From RapidAPI & Direct
    # ─────────────────────────────────────────────────────────────────────────

    # NewsAPI.org - 100 calls/day × 20 articles = 2000 articles/day (BEST FREE)
    "newsapi_org": {
        "id": "newsapi_org",
        "name": "NewsAPI.org",
        "source_type": "api",
        "tier": "tier_1",
        "credibility_score": 0.90,
        "url": "https://newsapi.org",
        "api_endpoint": "https://newsapi.org/v2/everything",
        "api_key_env": "NEWSAPI_ORG_KEY",
        "categories": ["aggregator", "news", "business"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 100,  # 100 calls/day, 20 articles each
        "articles_per_call": 20
    },

    # Real-Time News Data (RapidAPI) - Powered by Google News
    "rapidapi_realtime_news": {
        "id": "rapidapi_realtime_news",
        "name": "Real-Time News Data (RapidAPI)",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-news-data",
        "api_endpoint": "https://real-time-news-data.p.rapidapi.com/search",
        "api_key_env": "RAPIDAPI_KEY",
        "rapidapi_host": "real-time-news-data.p.rapidapi.com",
        "categories": ["aggregator", "news", "google"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 500  # Free tier estimate
    },

    # Google News API (RapidAPI) - Real-time Google News
    "rapidapi_google_news": {
        "id": "rapidapi_google_news",
        "name": "Google News API (RapidAPI)",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://rapidapi.com/barvanet-barvanet-default/api/google-news-api-real-time-google-news-data",
        "api_endpoint": "https://google-news-api-real-time-google-news-data.p.rapidapi.com/",
        "api_key_env": "RAPIDAPI_KEY",
        "rapidapi_host": "google-news-api-real-time-google-news-data.p.rapidapi.com",
        "categories": ["aggregator", "news", "google"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 500  # Free tier estimate
    },

    # MediaStack - 500 calls/month free
    "mediastack": {
        "id": "mediastack",
        "name": "MediaStack",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://mediastack.com",
        "api_endpoint": "http://api.mediastack.com/v1/news",
        "api_key_env": "MEDIASTACK_API_KEY",
        "categories": ["aggregator", "news"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 17  # 500/month ÷ 30 days
    },

    # TheNewsAPI - Free tier
    "thenewsapi": {
        "id": "thenewsapi",
        "name": "TheNewsAPI",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://www.thenewsapi.com",
        "api_endpoint": "https://api.thenewsapi.com/v1/news/all",
        "api_key_env": "THENEWSAPI_KEY",
        "categories": ["aggregator", "news", "business"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 100  # Free tier
    },

    # Webz.io News API Lite - 1000 calls/month (10 articles each)
    "webz_news": {
        "id": "webz_news",
        "name": "Webz.io News API",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://webz.io",
        "api_endpoint": "https://api.webz.io/newsApiLite",
        "api_key_env": "WEBZ_API_KEY",
        "categories": ["aggregator", "news", "sentiment"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 33,  # 1000/month ÷ 30 days
        "articles_per_call": 10
    },

    # Google Trends & News Insights (RapidAPI) - Trending news + keyword search
    "rapidapi_google_trends_news": {
        "id": "rapidapi_google_trends_news",
        "name": "Google Trends News (RapidAPI)",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://rapidapi.com/environmentn1t21r5/api/google-trends-news-insights-api",
        "api_endpoint": "https://google-trends-news-insights-api.p.rapidapi.com/news",
        "api_key_env": "RAPIDAPI_KEY",
        "rapidapi_host": "google-trends-news-insights-api.p.rapidapi.com",
        "categories": ["trends", "news", "google"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 50  # Free tier estimate
    },

    # GNews - 100 calls/day free
    "gnews": {
        "id": "gnews",
        "name": "GNews API",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://gnews.io",
        "api_endpoint": "https://gnews.io/api/v4/search",
        "api_key_env": "GNEWS_API_KEY",
        "categories": ["aggregator", "news"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 100
    },

    # NewsData.io - 500 calls/month (actually higher than I thought)
    "newsdata": {
        "id": "newsdata",
        "name": "NewsData.io",
        "source_type": "api",
        "tier": "tier_2",
        "credibility_score": 0.83,
        "url": "https://newsdata.io",
        "api_endpoint": "https://newsdata.io/api/1/news",
        "api_key_env": "NEWSDATA_API_KEY",
        "categories": ["aggregator", "news"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 17,  # 500/month ÷ 30 days
        "articles_per_call": 10
    },

    # ─────────────────────────────────────────────────────────────────────────
    # UNOFFICIAL: Google News RSS (Free, works but unofficial)
    # ─────────────────────────────────────────────────────────────────────────
    "google_news_india_business": {
        "id": "google_news_india_business",
        "name": "Google News India Business",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,  # Aggregator, varies
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en",
        "categories": ["business", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "google_news_india_tech": {
        "id": "google_news_india_tech",
        "name": "Google News India Technology",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en",
        "categories": ["technology", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    # ── GDELT (Global Database of Events, Language, and Tone) ──────────
    # Free API, no key needed. Monitors 100+ languages across 250+ countries.
    # Indexes ~300,000 articles/day. Updates every 15 minutes.
    # REF: Leetaru & Schrodt 2013, "GDELT: Global Data on Events, Location and Tone"
    "gdelt_india": {
        "id": "gdelt_india",
        "name": "GDELT India",
        "source_type": "api",
        "tier": "tier_1",
        "credibility_score": 0.88,
        "url": "https://www.gdeltproject.org",
        "api_endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "categories": ["business", "economy", "geopolitical", "events"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 500,
    },
    "gdelt_india_business": {
        "id": "gdelt_india_business",
        "name": "GDELT India Business",
        "source_type": "api",
        "tier": "tier_1",
        "credibility_score": 0.88,
        "url": "https://www.gdeltproject.org",
        "api_endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "categories": ["business", "economy", "startup", "finance"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 500,
    },
}

# Quick access lists
RSS_SOURCES = [src for src in NEWS_SOURCES.values() if src["source_type"] == "rss"]
API_SOURCES = [src for src in NEWS_SOURCES.values() if src["source_type"] == "api"]
TIER_1_SOURCES = [src for src in NEWS_SOURCES.values() if src["tier"] == "tier_1"]
TIER_2_SOURCES = [src for src in NEWS_SOURCES.values() if src["tier"] == "tier_2"]

# Default sources to use (can be overridden via env)
# NOTE: Verified 2026-02-06 — tested all RSS feeds individually:
#   STILL BROKEN (do not re-add):
#     - financial_express (410 Gone — feed permanently removed)
#     - rbi_press (418 — anti-bot protection)
#     - business_today (404 Not Found)
#     - entrackr (404 Not Found)
#     - sebi (404 Not Found)
#     - vccircle (200 but malformed XML — parse error)
#   RECOVERED (added back):
#     - business_standard, bs_companies (working again, 10+35 articles)
DEFAULT_ACTIVE_SOURCES = [
    # Tier 1: Major Business Publications (RSS - Unlimited, Working)
    "economic_times", "et_industry", "et_tech",
    "livemint", "mint_companies",
    "moneycontrol",
    "business_standard", "bs_companies",  # Recovered — working again as of 2026-02
    # Tier 2: Startup & Tech (RSS - Unlimited)
    "yourstory", "inc42",
    # vccircle removed: returns 200 but malformed XML, feedparser fails
    # Government (RSS - Unlimited)
    "pib",
    # Other Publications (RSS - Unlimited)
    "hindu_business", "ndtv_profit", "techcrunch_india",
    # Google News (RSS - Unofficial but works)
    "google_news_india_business", "google_news_india_tech",
    # APIs (set env vars to activate)
    "newsapi_org",           # NEWSAPI_ORG_KEY - 100 calls/day (BEST)
    "rapidapi_realtime_news",         # RAPIDAPI_KEY - 500/day
    "rapidapi_google_trends_news",    # RAPIDAPI_KEY - trending news
    "gnews",                          # GNEWS_API_KEY - 100/day
    # GDELT (FREE, no API key needed — massive event coverage)
    "gdelt_india",
    "gdelt_india_business",
]
