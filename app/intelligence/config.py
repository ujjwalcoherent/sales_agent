"""
Intelligence Engine Configuration.

All parameters here are STARTING PRIORS — the self-learning system adapts them
over time via:
  - WeightLearnerAgent: signal weights (EWC + KL guardrail)
  - ThresholdAdapterAgent: validation thresholds (EMA α=0.1)
  - SourceBanditAgent: source quality scores (Thompson Sampling)
  - CompanyBanditAgent: company profile quality (LinTS)

No values are permanently hardcoded — they all live in data/*.json and are
updated after each run. Constants here are only used when no learned values exist.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ══════════════════════════════════════════════════════════════════════════════
# REGION CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Per-region source allowlists ─────────────────────────────────────────────
# Explicit allowlists prevent the "no-country" footgun:
# sources without a `country` field in NEWS_SOURCES would otherwise be included
# in ALL regions via the old `or not cfg.get("country")` fallback.
# Add new regions here — only sources in this list are fetched for that region.
# GLOBAL uses DEFAULT_ACTIVE_SOURCES (everything active).

REGION_SOURCES: Dict[str, Optional[List[str]]] = {
    "IN": [
        # ── Indian business publications (Tier 1) ────────────────────────────
        "economic_times", "et_industry", "et_tech", "et_markets", "et_startup",
        "livemint", "mint_companies", "mint_markets", "mint_economy", "mint_industry",
        "moneycontrol", "mc_topnews",
        "cnbctv18", "cnbctv18_market",
        "hindu_business",
        "et_bfsi", "et_cio", "et_infra",
        # ── Indian startup & tech ─────────────────────────────────────────────
        "yourstory", "inc42",
        "techcrunch_india",
        # ── Indian press releases ─────────────────────────────────────────────
        "prnewswire_india",
        "indiatoday_business", "ndtv_profit",
        # ── Indian government & regulatory ────────────────────────────────────
        "sebi_v2", "rbi_v2", "pib_finance", "pib_commerce",
        # ── India-focused Google News ─────────────────────────────────────────
        "google_news_india_startup", "google_news_india_fintech",
        "google_news_business", "google_news_tech", "google_news_economy",
        # ── Global B2B/enterprise tech (relevant to India tech sector) ────────
        "finextra", "techcrunch_fintech", "techcrunch_ai",
        "venturebeat", "siliconangle",
        "businesswire_tech", "businesswire_financial",
        "prnewswire_india", "globe_newswire",
        "reuters_business", "reuters_technology",
        # ── Asia-Pacific context ──────────────────────────────────────────────
        "techinasia", "channel_news_asia",
    ],
    "US": [
        "wsj_business", "wsj_tech", "wsj_markets",
        "nyt_business", "nyt_technology",
        "business_insider",
        "forbes", "fortune", "fast_company", "inc_magazine",
        "wired_business", "ars_technica",
        "techcrunch_main", "techcrunch_ai", "techcrunch_fintech",
        "venturebeat", "siliconangle", "zdnet",
        "cnbc_world", "cnbc_tech",
        "yahoo_finance", "marketwatch", "seeking_alpha",
        "pymnts", "banking_dive", "finextra",
        "businesswire_tech", "businesswire_financial",
        "prnewswire_india", "globe_newswire",
        "reuters_business", "reuters_technology",
        "google_news_us_business", "google_news_us_tech",
        "fed_reserve",
    ],
    "EU": [
        "bbc_business", "bbc_technology",
        "dw_business", "sky_business",
        "euractiv_economy", "euractiv_digital",
        "guardian_business",
        "finextra", "techcrunch_main",
        "venturebeat", "siliconangle",
        "businesswire_tech", "businesswire_financial",
        "google_news_uk_business",
    ],
    "SEA": [
        "channel_news_asia", "straits_times_biz", "asia_times", "techinasia",
        "finextra", "techcrunch_main",
        "businesswire_tech", "businesswire_financial", "prnewswire_india",
    ],
    "GLOBAL": None,  # None = use DEFAULT_ACTIVE_SOURCES (all active sources)
}


@dataclass
class RegionConfig:
    """Dynamic region configuration. No hardcoded countries in pipeline logic."""
    code: str        # "IN", "US", "EU", "SEA", "GLOBAL"
    name: str
    languages: List[str] = field(default_factory=lambda: ["en"])
    domestic_domains: List[str] = field(default_factory=list)

    @property
    def source_ids(self) -> List[str]:
        """Return source IDs for this region.

        Uses REGION_SOURCES allowlists — explicit is safer than fallback-based logic.
        GLOBAL returns DEFAULT_ACTIVE_SOURCES (all active sources).
        Unknown region codes also fall back to GLOBAL.
        """
        from app.config import NEWS_SOURCES, DEFAULT_ACTIVE_SOURCES
        region_list = REGION_SOURCES.get(self.code)
        if region_list is None:
            # GLOBAL or unknown: use full active source list
            return [sid for sid in DEFAULT_ACTIVE_SOURCES if sid in NEWS_SOURCES]
        return [sid for sid in region_list if sid in NEWS_SOURCES]


REGIONS: Dict[str, RegionConfig] = {
    "IN": RegionConfig(
        code="IN", name="India", languages=["en", "hi"],
        domestic_domains=[".in", "economictimes.com", "livemint.com", "moneycontrol.com"],
    ),
    "US": RegionConfig(
        code="US", name="United States", languages=["en"],
        domestic_domains=[".com", "reuters.com", "bloomberg.com", "wsj.com"],
    ),
    "EU": RegionConfig(
        code="EU", name="European Union", languages=["en", "de", "fr"],
        domestic_domains=[".eu", ".de", ".fr", ".uk"],
    ),
    "SEA": RegionConfig(
        code="SEA", name="Southeast Asia", languages=["en"],
        domestic_domains=[".sg", ".my", ".th", "channelnewsasia.com"],
    ),
    "GLOBAL": RegionConfig(code="GLOBAL", name="Global", languages=["en"]),
}


def get_region(code: str) -> RegionConfig:
    return REGIONS.get(code.upper(), REGIONS["GLOBAL"])


# ══════════════════════════════════════════════════════════════════════════════
# INDUSTRY TAXONOMY (GICS-inspired, B2B focus)
# ══════════════════════════════════════════════════════════════════════════════
#
# Structure per industry:
#   keywords:   search terms to use in Tavily/RSS queries
#   1st_order:  companies that ARE the industry (direct players)
#   2nd_order:  companies that SERVE the industry (adjacent players)
#   exclude:    sectors that are definitely NOT this industry
#
# Auto-expansion for unknown industries:
#   if industry not in INDUSTRY_TAXONOMY → LLM generates structure → cached for session

INDUSTRY_TAXONOMY: Dict[str, Dict] = {
    "Cybersecurity": {
        "keywords": ["breach", "ransomware", "threat", "zero-day", "SIEM", "SOC", "endpoint", "CVE"],
        "anchor_companies": ["CrowdStrike", "Palo Alto Networks", "Quick Heal", "Trellix", "Seqrite"],
        "1st_order": [
            "endpoint security", "network security", "identity management",
            "threat intelligence", "SIEM vendor", "cloud security", "zero trust",
            "vulnerability management", "security analytics",
        ],
        "2nd_order": [
            "enterprise software", "cloud provider", "managed IT services",
            "cyber insurance", "compliance software", "GRC platform",
        ],
        "exclude": ["gaming", "social media", "food", "retail", "pharma"],
        "target_roles": ["CISO", "VP Security", "Head of IT Security", "CTO"],
    },
    "Fintech": {
        "keywords": ["payment", "neobank", "digital banking", "insurtech", "wealthtech", "regtech", "UPI", "BNPL"],
        "anchor_companies": ["Razorpay", "PhonePe", "Paytm", "CRED", "BharatPe"],
        "1st_order": [
            "payment processor", "digital wallet", "lending platform",
            "crypto exchange", "robo-advisor", "insurtech", "neobank",
            "regtech", "open banking", "BNPL provider",
        ],
        "2nd_order": [
            "cloud infrastructure", "identity verification", "fraud detection",
            "banking CRM", "financial data provider", "KYC provider",
            "core banking software",
        ],
        "exclude": ["food delivery", "logistics", "real estate", "healthcare", "gaming"],
        "target_roles": ["CTO", "VP Engineering", "Head of Product", "CDO", "CFO"],
    },
    "Healthcare": {
        "keywords": ["drug", "clinical trial", "FDA", "biotech", "therapeutic", "pipeline", "pharma"],
        "anchor_companies": ["Sun Pharma", "Cipla", "Dr Reddy's", "Biocon", "Lupin"],
        "1st_order": [
            "pharmaceutical manufacturer", "biotech company",
            "drug developer", "CRO", "specialty pharma", "medical devices",
            "diagnostics", "hospital system",
        ],
        "2nd_order": [
            "health IT", "EHR vendor", "lab equipment maker",
            "cold chain logistics", "clinical data management", "health insurer",
        ],
        "exclude": ["consumer goods", "technology", "finance", "retail", "entertainment"],
        "target_roles": ["Chief Medical Officer", "VP Clinical", "Head of R&D", "CTO"],
    },
    "Pharmaceutical & Biotech": {
        "keywords": ["pharma", "biotech", "drug approval", "clinical trial", "FDA", "CDSCO", "API", "biosimilar", "generics"],
        "anchor_companies": ["Sun Pharma", "Cipla", "Dr Reddy's", "Biocon", "Lupin", "Divi's Laboratories"],
        "1st_order": [
            "pharmaceutical manufacturer", "biotech company", "API manufacturer",
            "drug developer", "biosimilar maker", "specialty pharma",
            "generic drug maker", "contract development and manufacturing",
        ],
        "2nd_order": [
            "CRO", "clinical data management", "cold chain logistics",
            "lab equipment maker", "pharma analytics software", "regulatory compliance",
        ],
        "exclude": ["consumer goods", "food", "finance", "retail", "entertainment", "IT services"],
        "target_roles": ["VP R&D", "Head of Regulatory", "Chief Medical Officer", "VP Manufacturing", "CTO"],
    },
    "Technology": {
        "keywords": ["SaaS", "cloud", "AI", "startup", "software", "semiconductor", "tech company", "data center"],
        "anchor_companies": ["Infosys", "TCS", "Wipro", "HCL Technologies", "Tech Mahindra"],
        "1st_order": [
            "cloud infrastructure", "enterprise SaaS", "AI/ML platform",
            "developer tools", "data analytics", "IT services", "semiconductors",
        ],
        "2nd_order": [
            "system integrators", "IT consulting", "resellers", "managed services",
        ],
        "exclude": ["consumer phones", "consumer electronics", "smartphones for consumers", "pharma", "food", "fashion", "entertainment", "cricket", "sports", "election", "politics", "government formation", "military", "war", "celebrity"],
        "target_roles": ["CTO", "VP Engineering", "Head of Infrastructure", "CDO"],
    },
    "Manufacturing": {
        "keywords": ["industrial", "automation", "supply chain", "ERP", "factory", "OEM", "precision"],
        "anchor_companies": ["Tata Motors", "Mahindra", "Bharat Forge", "L&T", "Godrej Industries"],
        "1st_order": [
            "industrial automation", "automotive OEM", "aerospace & defense",
            "chemicals", "precision engineering", "contract manufacturing",
        ],
        "2nd_order": [
            "industrial software", "supply chain software", "quality management",
            "MES vendor", "predictive maintenance",
        ],
        "exclude": ["fintech", "healthcare software", "gaming", "media"],
        "target_roles": ["COO", "VP Operations", "Head of Manufacturing", "CTO"],
    },
    "Energy": {
        "keywords": ["renewable", "solar", "EV", "battery", "grid", "oil", "gas", "power"],
        "anchor_companies": ["Reliance Industries", "ONGC", "Adani Green", "Tata Power", "NTPC"],
        "1st_order": [
            "oil & gas", "renewables", "solar developer", "wind developer",
            "EV manufacturer", "battery maker", "power grid operator",
        ],
        "2nd_order": [
            "energy management software", "grid analytics", "energy trading",
            "SCADA vendor", "smart meter maker",
        ],
        "exclude": ["fintech", "pharma", "gaming", "fashion"],
        "target_roles": ["CEO", "Head of Digital", "CTO", "VP Sustainability"],
    },
    "Retail & E-commerce": {
        "keywords": ["marketplace", "D2C", "quick commerce", "FMCG", "omnichannel", "loyalty"],
        "anchor_companies": ["Flipkart", "Meesho", "Nykaa", "Mamaearth", "Blinkit"],
        "1st_order": [
            "D2C brand", "marketplace", "quick commerce", "grocery chain",
            "fashion retailer", "luxury brand",
        ],
        "2nd_order": [
            "retail analytics", "loyalty platform", "POS vendor",
            "inventory management", "e-commerce platform",
        ],
        "exclude": ["fintech", "pharma", "defense", "industrial"],
        "target_roles": ["CMO", "VP Digital", "Head of E-commerce", "CTO"],
    },
    "Logistics & Supply Chain": {
        "keywords": ["freight", "warehouse", "3PL", "last-mile", "supply chain", "logistics"],
        "anchor_companies": ["Delhivery", "Ecom Express", "Mahindra Logistics", "TCI Express", "Gati"],
        "1st_order": [
            "freight broker", "3PL", "last-mile delivery", "warehousing",
            "ocean freight", "air cargo",
        ],
        "2nd_order": [
            "TMS vendor", "WMS vendor", "supply chain visibility platform",
            "cold chain software", "customs compliance",
        ],
        "exclude": ["healthcare", "fintech", "gaming"],
        "target_roles": ["COO", "VP Supply Chain", "Head of Logistics", "CTO"],
    },
    "Education": {
        "keywords": ["edtech", "e-learning", "LMS", "upskilling", "corporate training", "MOOCs"],
        "anchor_companies": ["BYJU's", "Unacademy", "upGrad", "Coursera India", "Great Learning"],
        "1st_order": [
            "edtech platform", "LMS vendor", "online university",
            "corporate training provider", "coding bootcamp",
        ],
        "2nd_order": [
            "education analytics", "student information systems",
            "content platform", "assessment software",
        ],
        "exclude": ["pharma", "defense", "fintech"],
        "target_roles": ["CHRO", "VP Learning", "Head of L&D", "CTO"],
    },
    "Real Estate": {
        "keywords": ["proptech", "commercial real estate", "REIT", "co-working", "smart building"],
        "anchor_companies": ["DLF", "Godrej Properties", "Prestige Group", "WeWork India", "Embassy REIT"],
        "1st_order": [
            "commercial real estate", "residential developer",
            "REIT", "co-working operator", "property management",
        ],
        "2nd_order": [
            "proptech platform", "smart building software", "facility management",
            "real estate analytics", "construction tech",
        ],
        "exclude": ["pharma", "fintech", "gaming"],
        "target_roles": ["COO", "Head of Technology", "VP Digital", "CTO"],
    },
    "BFSI": {
        "keywords": ["banking", "insurance", "NBFC", "lending", "mutual fund", "wealth management", "credit"],
        "anchor_companies": ["HDFC Bank", "ICICI Bank", "Axis Bank", "SBI", "Bajaj Finance"],
        "1st_order": [
            "commercial bank", "private bank", "insurance company", "NBFC",
            "mutual fund house", "wealth management firm", "credit bureau",
        ],
        "2nd_order": [
            "core banking software", "insurance tech", "risk analytics",
            "fraud detection platform", "KYC/AML vendor",
        ],
        "exclude": ["consumer goods", "pharma", "gaming", "entertainment"],
        "target_roles": ["CTO", "CDO", "VP Digital", "Head of Risk", "CFO"],
    },
    "Media & Entertainment": {
        "keywords": ["OTT", "streaming", "content", "digital media", "ad-tech", "broadcast"],
        "anchor_companies": ["Zee Entertainment", "Sony LIV", "JioCinema", "Times Group", "Inox Media"],
        "1st_order": [
            "OTT platform", "broadcast network", "digital media company",
            "content studio", "ad-tech platform", "gaming company",
        ],
        "2nd_order": [
            "content distribution", "ad serving platform", "audience analytics",
            "CDN provider", "media asset management",
        ],
        "exclude": ["pharma", "manufacturing", "logistics", "fintech"],
        "target_roles": ["CTO", "VP Product", "Head of Engineering", "CDO"],
    },
}

# Derived lookups
_KEYWORD_TO_INDUSTRY: Dict[str, str] = {}
for _ind, _cfg in INDUSTRY_TAXONOMY.items():
    for _kw in _cfg.get("keywords", []):
        _KEYWORD_TO_INDUSTRY[_kw.lower()] = _ind

# L1 → L2 sub-industry list (derived from rich taxonomy)
_L1_TO_L2: Dict[str, List[str]] = {
    ind: cfg.get("1st_order", []) for ind, cfg in INDUSTRY_TAXONOMY.items()
}

# L2 → L1 reverse lookup
_L2_TO_L1: Dict[str, str] = {}
for _l1, _l2_list in _L1_TO_L2.items():
    for _l2 in _l2_list:
        _L2_TO_L1[_l2.lower()] = _l1


def get_industry_keywords(industry: str) -> Set[str]:
    """Get all search keywords for an industry."""
    cfg = INDUSTRY_TAXONOMY.get(industry, {})
    return set(kw.lower() for kw in cfg.get("keywords", [industry.lower()]))


def get_industry_anchors(industry: str) -> List[str]:
    """Get known anchor companies for an industry (for targeted news fetching).

    Returns curated anchor companies from taxonomy if available.
    Falls back to empty list — caller should use Tavily to discover dynamically.
    """
    cfg = INDUSTRY_TAXONOMY.get(industry, {})
    return list(cfg.get("anchor_companies", []))


def parse_industry(industry_str: str) -> tuple:
    """Parse 'Healthcare > Pharmaceuticals' → ('Healthcare', 'Pharmaceuticals').

    Also handles single-level: 'Healthcare' → ('Healthcare', '').
    """
    if ">" in industry_str:
        parts = [p.strip() for p in industry_str.split(">", 1)]
        return parts[0], parts[1] if len(parts) > 1 else ""
    return industry_str.strip(), ""


def get_l2_industries(l1: str) -> List[str]:
    """Get all L2 sub-industries for an L1 parent."""
    return list(_L1_TO_L2.get(l1, []))


def get_l1_for_l2(l2: str) -> Optional[str]:
    """Reverse lookup: find L1 parent for an L2 industry."""
    return _L2_TO_L1.get(l2.lower())


def classify_industry_by_keyword(text: str) -> Optional[str]:
    """Quick keyword-based industry classification (no LLM).
    Returns None if ambiguous — use LLM for those.
    """
    text_lower = text.lower()
    matches = [ind for kw, ind in _KEYWORD_TO_INDUSTRY.items() if kw in text_lower]
    if len(set(matches)) == 1:
        return matches[0]
    return None


def get_target_roles(industry: str) -> List[str]:
    """Get the target buyer roles for an industry."""
    return INDUSTRY_TAXONOMY.get(industry, {}).get("target_roles", ["CEO", "CTO", "COO"])


# ══════════════════════════════════════════════════════════════════════════════
# B2B EVENT TYPES
# ══════════════════════════════════════════════════════════════════════════════

B2B_EVENT_TYPES: Set[str] = {
    "regulation", "policy", "funding", "market", "acquisition", "merger",
    "partnership", "expansion", "layoffs", "hiring", "product_launch",
    "ipo", "bankruptcy", "technology", "supply_chain", "price_change",
    "earnings", "market_movement", "infrastructure", "geopolitical",
    "sustainability", "crisis", "leadership_change", "restructuring",
}

NON_B2B_EVENT_TYPES: Set[str] = {
    "entertainment", "sports", "celebrity", "lifestyle", "food",
    "travel", "fashion", "gossip",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING PARAMETERS (all are starting priors — learned dynamically)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusteringParams:
    """Algorithm parameters. Adapted by ThresholdAdapterAgent via EMA (α=0.1).

    All threshold fields correspond to entries in data/adaptive_thresholds.json.
    On startup: load from JSON if exists, else use these defaults.
    """

    # ── Dedup (math gate 1) ──────────────────────────────────────────────────
    # Manber & Wu 1994: 0.85 is correct for title-only near-duplicate detection.
    # Our old value of 0.35 was for semantic similarity, not dedup.
    dedup_title_threshold: float = 0.85
    dedup_body_threshold: float = 0.70

    # ── Relevance Filter (math gate 2) ───────────────────────────────────────
    # NLI zero-shot classification thresholds (replaces Dunietz & Gillick salience)
    # Research: Yin et al. (2019) arXiv:1909.00161 — NLI for zero-shot text classification
    # Model: cross-encoder/nli-deberta-v3-small, entailment scores in [0,1]
    nli_auto_accept: float = 0.88       # entailment >= this → keep (no LLM)
    nli_auto_reject: float = 0.10       # entailment <= this → drop (no LLM)
    # 0.10-0.88 → LLM batch classify (ambiguous zone)
    # Raised from 0.55→0.75→0.88: real data shows political articles (PM Modi unveils ₹18k cr
    # Delhi metro, ED bail hearing, Centre-Bengal) score 0.75-0.85 — they bypass LLM at 0.75.
    # At 0.88, only very-high-confidence B2B articles auto-accept (Infosys deal 0.982, Ola IPO 0.990).
    # Lowered from 0.15→0.10: captures Sarvam open-source (0.092), Euler Motors PLI (0.076) for LLM.
    # Cost tradeoff: more LLM calls (~60% of articles), but eliminates government/political FPs.
    filter_gap4_days: int = 5           # Drop company if 0 articles in N days
    # Legacy salience threshold — adapted by threshold_adapter.py EMA
    filter_auto_accept: float = 0.30

    # ── NER / Entity extraction (math gate 3) ────────────────────────────────
    fuzzy_merge_threshold: float = 85.0      # rapidfuzz token_sort_ratio
    gliner_accept_threshold: float = 0.65
    gliner_reject_threshold: float = 0.30
    pos_propn_min_ratio: float = 0.50        # ≥50% tokens must be PROPN
    containment_overlap_min: float = 0.30    # For single-word suffix containment guard

    # ── Similarity matrix (math gate 4) ──────────────────────────────────────
    # Starting priors for signal weights — adapted by WeightLearnerAgent
    sim_weight_semantic: float = 0.35
    sim_weight_entity: float = 0.25
    sim_weight_lexical: float = 0.05
    sim_weight_temporal: float = 0.10
    sim_weight_source: float = 0.15
    # Temporal decay — dual-sigma Gaussian (max of short + long)
    temporal_sigma_short: float = 8.0       # σ=8h for breaking news
    temporal_sigma_long: float = 72.0       # σ=72h for ongoing coverage
    # Same-source penalty (strong: forces cross-source clustering)
    same_source_penalty: float = 0.3

    # ── HAC (math gate 5, entity groups ≤ 50 articles) ───────────────────────
    hac_linkage: str = "average"
    hac_k_min: int = 5
    hac_k_max_ratio: float = 0.33          # k_max = n_articles // 3
    hac_outlier_silhouette: float = -0.1
    hac_max_articles: int = 50
    hac_min_articles: int = 5
    hac_min_cluster_size: int = 3
    # Singleton penalty in silhouette sweep (FANATIC/EMNLP 2021)
    hac_singleton_penalty_factor: float = 0.5  # (singletons/n) * this
    # Distance threshold sweep range for dendrogram cut
    hac_threshold_min: float = 0.30        # Cosine distance min (lower = tighter clusters)
    hac_threshold_max: float = 0.65        # Cosine distance max (higher = looser clusters)
    hac_threshold_step: float = 0.05       # Step size for silhouette sweep in small groups

    # ── HDBSCAN soft (math gate 5, entity groups > 50 articles) ─────────────
    # Campello et al. 2013: soft membership vectors (NOT nearest-centroid)
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    hdbscan_soft_noise_threshold: float = 0.10  # max(soft[i]) < this → true noise
    hdbscan_use_blended_similarity: bool = True

    # ── Leiden (discovery mode — ungrouped articles) ──────────────────────────
    leiden_k: int = 20
    leiden_resolution: float = 1.0

    # ── Cluster validation (math gate 6) ─────────────────────────────────────
    val_min_articles: int = 2
    val_min_sources: int = 2
    val_coherence_min: float = 0.40         # Mean pairwise cosine
    val_separation_margin: float = 0.10     # intra > inter + this
    val_entity_consistency_min: float = 0.60
    val_temporal_window_hours: float = 720.0  # 30 days max
    val_duplicate_threshold: float = 0.85
    val_composite_reject: float = 0.50

    # ── Synthesis (math gate 7) ───────────────────────────────────────────────
    synthesis_max_retries: int = 3          # Reflexion retry limit
    synthesis_representative_k: int = 5    # Articles to include in LLM prompt
    synthesis_label_min_words: int = 3
    synthesis_label_max_words: int = 8

    # ── Match engine (math gate 8) ────────────────────────────────────────────
    match_keyword_weight: float = 0.50
    match_semantic_weight: float = 0.30
    match_industry_weight: float = 0.20

    # ── Noise reassignment ────────────────────────────────────────────────────
    noise_reassign_enabled: bool = True
    noise_reassign_min_similarity: float = 0.45

    # ── Post-clustering source diversity ──────────────────────────────────────
    enforce_source_diversity: bool = True
    min_sources_per_cluster: int = 2
    source_merge_min_similarity: float = 0.4

    # entity_weight_org/person/product/gpe — REMOVED (0 callers, March 2026 audit)

    # ── AutoResearch: query expansion + critic ─────────────────────────
    enable_query_expansion: bool = True  # LLM generates extra search queries
    enable_critic: bool = True           # LLM validates cluster quality


DEFAULT_PARAMS = ClusteringParams()


def load_adaptive_params() -> ClusteringParams:
    """Load params from data/adaptive_thresholds.json (EMA-adapted values).

    Falls back to DEFAULT_PARAMS if file doesn't exist.
    Called at pipeline start — ensures each run uses learned thresholds.
    """
    import json
    import os

    path = os.path.join("data", "adaptive_thresholds.json")
    if not os.path.exists(path):
        return DEFAULT_PARAMS

    try:
        with open(path) as f:
            stored = json.load(f)
        params = ClusteringParams()
        for key, value in stored.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params
    except Exception as exc:
        logging.getLogger(__name__).debug(f"Failed to load adaptive thresholds: {exc}")
        return DEFAULT_PARAMS
