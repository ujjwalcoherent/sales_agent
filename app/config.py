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
    # Provider priority (GCP-first): GeminiDirect → VertexDeepSeek → VertexLlama → NVIDIA → OpenRouter → Groq (last resort) → Ollama
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_model: str = Field(default="deepseek-ai/deepseek-v3.1", alias="NVIDIA_MODEL")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")

    use_ollama: bool = Field(default=True, alias="USE_OLLAMA")
    use_ddg_fallback: bool = Field(default=True, alias="USE_DDG_FALLBACK")
    offline_mode: bool = Field(default=False, alias="OFFLINE_MODE")
    ollama_model: str = Field(default="mistral", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Vertex AI Express Mode (free tier fallback — 10 RPM, 90 days)
    vertex_express_api_key: str = Field(default="", alias="VERTEX_EXPRESS_API_KEY")

    # Full Vertex AI (uses GCP credits — 60+ RPM)
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    gcp_service_account_file: str = Field(default="", alias="GCP_SERVICE_ACCOUNT_FILE")
    gcp_vertex_location: str = Field(default="us-central1", alias="GCP_VERTEX_LOCATION")

    # Vertex AI API Service — partner models via OpenAI-compatible endpoint
    # Same service account as above; access token is passed as Bearer auth.
    # DeepSeek V3.2: best reasoning + tools, ~$0.07/$0.14 per 1M in/out (cheapest reasoning model)
    # Llama 4 Scout: fastest tool-calling fallback, ~$0.04/$0.08 per 1M in/out
    # Set to "" to disable that provider (falls through to next in chain).
    vertex_deepseek_model: str = Field(default="deepseek/deepseek-v3-2", alias="VERTEX_DEEPSEEK_MODEL")
    vertex_llama_model: str = Field(default="meta/llama-4-scout-17b-16e-instruct-maas", alias="VERTEX_LLAMA_MODEL")

    # Groq Configuration (fast inference, 1K-14K free req/day)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="qwen-qwen3-32b", alias="GROQ_MODEL")
    # llama-3.3-70b-versatile is the confirmed tool-calling model on Groq
    # (replaces deprecated llama3-groq-70b-8192-tool-use-preview)
    groq_tool_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_TOOL_MODEL")

    # OpenRouter Configuration (multi-model API)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="google/gemini-2.5-flash", alias="OPENROUTER_MODEL")

    # OpenAI Configuration (GPT-4.1-mini for generation, GPT-4.1-nano for classification)
    # Tier 1: ~500 RPM, $0.40/$1.60 per 1M in/out (mini), $0.10/$0.40 (nano)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_lite_model: str = Field(default="gpt-4.1-nano", alias="OPENAI_LITE_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=1536, alias="OPENAI_EMBEDDING_DIMENSIONS")

    # Gemini Configuration (used for both direct Vertex Express and OpenRouter proxy)
    # Both default to gemini-2.5-flash-lite — cheapest GA model with tool calling + structured output.
    # Cost: $0.10/$0.40 per 1M in/out. At 200 runs/day × 140K tokens → ~$5/day → $300 lasts 90+ days.
    # Override GEMINI_MODEL=gemini-2.5-flash if complex synthesis quality needs a step up (~6x cost).
    gemini_model: str = Field(default="gemini-2.5-flash-lite", alias="GEMINI_MODEL")
    gemini_lite_model: str = Field(default="gemini-2.5-flash-lite", alias="GEMINI_LITE_MODEL")

    # Embedding Configuration
    # Primary: NVIDIA nv-embedqa-e5-v5 (1024-dim, best discrimination 0.238 gap)
    # Fallback: BAAI/bge-large-en-v1.5 (1024-dim, same dimensions, HF API or local)
    huggingface_api_key: str = Field(default="", alias="HF_API_KEY")
    embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5", alias="EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", alias="LOCAL_EMBEDDING_MODEL")
    # Priority: "nvidia" = NVIDIA NIM API (best quality), "api" = HF GPUs, "local" = local CPU/GPU
    embedding_provider: str = Field(default="nvidia", alias="EMBEDDING_PROVIDER")

    # ── Trend Engine Pipeline ──
    # Leiden: k-NN graph + community detection on raw 1024-dim embeddings.
    # k: number of nearest neighbors for graph construction (sqrt(N) is a good default)
    leiden_k: int = Field(default=20, alias="LEIDEN_K")
    # resolution: higher = more, smaller clusters. Auto-tuned if leiden_auto_resolution=True.
    leiden_resolution: float = Field(default=1.0, alias="LEIDEN_RESOLUTION")
    # Auto-resolve resolution to hit target cluster count range (sqrt(N)/3 to sqrt(N))
    leiden_auto_resolution: bool = Field(default=True, alias="LEIDEN_AUTO_RESOLUTION")
    # Minimum community size (smaller → noise)
    leiden_min_community_size: int = Field(default=3, alias="LEIDEN_MIN_COMMUNITY_SIZE")
    # Seed for deterministic results
    leiden_seed: int = Field(default=42, alias="LEIDEN_SEED")
    # Optuna: Bayesian multi-parameter optimization (k, resolution, min_community)
    # When True, replaces binary-search auto_resolution with Optuna TPE.
    # ~15s for 15 trials at N=1000. Warm-starts from last run's best params.
    leiden_optuna_enabled: bool = Field(default=True, alias="LEIDEN_OPTUNA_ENABLED")
    leiden_optuna_trials: int = Field(default=15, alias="LEIDEN_OPTUNA_TRIALS")
    leiden_optuna_timeout: int = Field(default=30, alias="LEIDEN_OPTUNA_TIMEOUT")

    # Deduplication: MinHash LSH near-duplicate detection (lexical)
    # 0.25 = very aggressive, catches articles with ~25% shared word bigrams (RECOMMENDED)
    # 0.3 = aggressive, catches articles with ~30% shared word bigrams
    # 0.5 = moderate, catches near-identical with some variation
    # Combined with title-based, entity-based, and semantic dedup for comprehensive coverage
    dedup_threshold: float = Field(default=0.25, alias="DEDUP_THRESHOLD")
    dedup_num_perm: int = Field(default=128, alias="DEDUP_NUM_PERM")
    dedup_shingle_size: int = Field(default=2, alias="DEDUP_SHINGLE_SIZE")

    # Deduplication: Semantic (embedding-based) - catches cross-source duplicates
    # OBSERVED SIMILARITY RANGES (nv-embedqa-e5-v5):
    # - True duplicates (same story, different source): 0.90-1.0
    # - Related but different articles: 0.65-0.72
    # - Unrelated articles: 0.40-0.50
    # THRESHOLD SELECTION:
    # 0.88 = catches true duplicates, comfortable margin above related (0.72)
    # 0.80 = too aggressive for nv-embedqa-e5-v5, catches related articles
    # 0.93 = too conservative, misses cross-source duplicates
    # Recalibrated for nv-embedqa-e5-v5 (sharper discrimination than bge-large)
    semantic_dedup_threshold: float = Field(default=0.88, alias="SEMANTIC_DEDUP_THRESHOLD")

    # Entity extraction: spaCy NER model
    spacy_model: str = Field(default="en_core_web_sm", alias="SPACY_MODEL")

    # Engine: Pipeline-level settings
    engine_max_depth: int = Field(default=3, alias="ENGINE_MAX_DEPTH")
    engine_max_concurrent_llm: int = Field(default=5, alias="ENGINE_MAX_CONCURRENT_LLM")

    # Signal weights for actionability scoring (JSON string, override via env)
    # These determine how trends are ranked for sales outreach.
    # Weights should sum to ~1.0. Adjust to tune which signals matter most.
    # Phase 6: Added cmi_relevance (10%) — clusters with low CMI service
    # alignment score lower for sales outreach (but articles NOT dropped).
    actionability_weights: str = Field(
        default='{"recency":0.12,"velocity":0.07,"specificity":0.12,"regulatory":0.12,"trigger":0.14,"diversity":0.07,"authority":0.13,"financial":0.05,"person":0.03,"event_focus":0.05,"cmi_relevance":0.10}',
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
    synthesis_max_articles: int = Field(default=15, alias="SYNTHESIS_MAX_ARTICLES")
    # Max characters per article in synthesis context (increased for richer context)
    synthesis_article_char_limit: int = Field(default=2000, alias="SYNTHESIS_ARTICLE_CHAR_LIMIT")
    # Max retries on synthesis failure before returning empty (3 = struct + specificity)
    synthesis_max_retries: int = Field(default=3, alias="SYNTHESIS_MAX_RETRIES")
    
    # ── LLM JSON Validation (V2) ──
    llm_json_max_retries: int = Field(default=2, alias="LLM_JSON_MAX_RETRIES")

    # ── Synthesis Validation (V3) ──
    synthesis_strict_mode: bool = Field(default=False, alias="SYNTHESIS_STRICT_MODE")

    # ── Event Classifier (V8) ──
    # 20 event types (was 14) — lower threshold since more categories to match.
    # Tier 2 LLM catches ambiguous cases above threshold.
    event_classifier_threshold: float = Field(default=0.35, alias="EVENT_CLASSIFIER_THRESHOLD")
    event_classifier_ambiguity_margin: float = Field(default=0.04, alias="EVENT_CLASSIFIER_AMBIGUITY_MARGIN")
    event_max_llm_calls: int = Field(default=80, alias="EVENT_MAX_LLM_CALLS")

    # ── Coherence Validation ──
    coherence_min: float = Field(default=0.48, alias="COHERENCE_MIN")
    coherence_reject: float = Field(default=0.35, alias="COHERENCE_REJECT")
    merge_threshold: float = Field(default=0.82, alias="MERGE_THRESHOLD")

    # ── CMI Relevance Filter ──
    cmi_relevance_threshold: float = Field(default=0.28, alias="CMI_RELEVANCE_THRESHOLD")
    # Hard floor: "general" articles below this CMI score are dropped (non-business content)
    cmi_hard_floor: float = Field(default=0.25, alias="CMI_HARD_FLOOR")

    # ── Article Triage Agent ──
    # Confidence floor: articles with event classifier confidence below this get LLM triage
    triage_confidence_floor: float = Field(default=0.45, alias="TRIAGE_CONFIDENCE_FLOOR")
    # Max articles to send to LLM triage per run.
    # 50 articles × batch_size=15 = ~4 LLM calls.
    triage_max_articles: int = Field(default=50, alias="TRIAGE_MAX_ARTICLES")
    # Batch size for LLM triage calls (15 optimal — fewer calls, same quality)
    triage_batch_size: int = Field(default=15, alias="TRIAGE_BATCH_SIZE")

    # ── Subclustering ──
    max_children_per_parent: int = Field(default=6, alias="MAX_CHILDREN_PER_PARENT")
    min_articles_for_subclustering: int = Field(default=6, alias="MIN_ARTICLES_FOR_SUBCLUSTERING")
    min_subcluster_coherence: float = Field(default=0.35, alias="MIN_SUBCLUSTER_COHERENCE")

    # ── Scraping ──
    semantic_dedup_max_articles: int = Field(default=2000, alias="SEMANTIC_DEDUP_MAX_ARTICLES")
    scrape_enabled: bool = Field(default=False, alias="SCRAPE_ENABLED")
    scrape_max_concurrent: int = Field(default=10, alias="SCRAPE_MAX_CONCURRENT")
    scrape_max_articles: int = Field(default=250, alias="SCRAPE_MAX_ARTICLES")
    summary_max_chars: int = Field(default=1500, alias="SUMMARY_MAX_CHARS")

    # ── Scoring Weights (JSON strings, same pattern as actionability_weights) ──
    trend_score_weights: str = Field(
        default='{"volume":0.30,"momentum":0.45,"diversity":0.25}',
        alias="TREND_SCORE_WEIGHTS",
    )
    cluster_quality_score_weights: str = Field(
        default='{"coherence":0.28,"source_diversity":0.25,"event_agreement":0.17,"evidence_volume":0.12,"authority":0.18}',
        alias="CLUSTER_QUALITY_SCORE_WEIGHTS",
    )
    source_credibility_weights: str = Field(
        default='{"base_authority":0.40,"cross_citation":0.25,"originality":0.20,"agreement":0.15}',
        alias="SOURCE_CREDIBILITY_WEIGHTS",
    )

    # ── Council & Agent Thresholds ──
    cmi_auto_noise_threshold: float = Field(default=0.2, alias="CMI_AUTO_NOISE_THRESHOLD")
    lead_relevance_threshold: float = Field(default=0.3, alias="LEAD_RELEVANCE_THRESHOLD")
    max_search_queries_per_impact: int = Field(default=7, alias="MAX_SEARCH_QUERIES_PER_IMPACT")
    enterprise_blocklist: str = Field(
        default="tata,reliance,infosys,wipro,hcl,hdfc,icici,bajaj,mahindra,adani,vedanta",
        alias="ENTERPRISE_BLOCKLIST",
    )
    company_min_relevance: float = Field(
        default=0.20,
        alias="COMPANY_MIN_RELEVANCE",
    )

    # ── RSS Throughput ──
    rss_max_per_source: int = Field(default=25, alias="RSS_MAX_PER_SOURCE")
    rss_hours_ago: int = Field(default=72, alias="RSS_HOURS_AGO")

    # ── Quality Gates (V9) ──
    min_synthesis_confidence: float = Field(default=0.40, alias="MIN_SYNTHESIS_CONFIDENCE")
    min_trend_confidence_for_agents: float = Field(default=0.40, alias="MIN_TREND_CONFIDENCE_FOR_AGENTS")

    # ── Cross-Validation (V10) ──
    # ValidatorAgent: scores LLM synthesis groundedness against source articles.
    # Enable/disable the validator (disable to save LLM calls during development)
    validator_enabled: bool = Field(default=True, alias="VALIDATOR_ENABLED")
    # Max back-and-forth rounds between synthesizer and validator (1 = validate only, 2+ = revise)
    validator_max_rounds: int = Field(default=3, alias="VALIDATOR_MAX_ROUNDS")
    # Overall groundedness score threshold to PASS (0.0-1.0). Below this = REVISE.
    validator_pass_threshold: float = Field(default=0.40, alias="VALIDATOR_PASS_THRESHOLD")
    # Overall groundedness score below which we REJECT outright (no revision attempt).
    validator_reject_threshold: float = Field(default=0.25, alias="VALIDATOR_REJECT_THRESHOLD")
    # Minimum entity overlap ratio (claimed entities found in sources) to pass entity check.
    # Lowered from 0.55 to 0.35: NER only sees title+summary+content[:1200] but LLM synthesis
    # correctly reads entities from full article text. 0.35 accounts for this coverage gap.
    validator_entity_overlap_min: float = Field(default=0.35, alias="VALIDATOR_ENTITY_OVERLAP_MIN")
    # Weight for NER entity overlap in overall score (0.0-1.0)
    validator_weight_entity: float = Field(default=0.35, alias="VALIDATOR_WEIGHT_ENTITY")
    # Weight for keyword overlap in overall score (0.0-1.0)
    validator_weight_keyword: float = Field(default=0.30, alias="VALIDATOR_WEIGHT_KEYWORD")
    # Weight for embedding similarity in overall score (0.0-1.0)
    validator_weight_embedding: float = Field(default=0.35, alias="VALIDATOR_WEIGHT_EMBEDDING")

    # ── Company Verification (V7) ──
    company_min_verification_confidence: float = Field(default=0.0, alias="COMPANY_MIN_VERIFICATION_CONFIDENCE")

    # Search APIs (supports multiple keys for rotation — comma-separated in .env)
    # Single key: TAVILY_API_KEYS=tvly-abc123
    # Multiple:   TAVILY_API_KEYS=tvly-abc123,tvly-def456
    # Set TAVILY_ENABLED=false to skip Tavily entirely and use DDG+ScrapeGraphAI
    tavily_enabled: bool = Field(default=False, alias="TAVILY_ENABLED")
    tavily_api_keys: str = Field(default="", alias="TAVILY_API_KEYS")

    # SearXNG (self-hosted meta-search — free, aggregates Google+Bing+DDG)
    # Deploy: docker run -d -p 8888:8080 searxng/searxng:latest
    searxng_url: str = Field(default="http://localhost:8888", alias="SEARXNG_URL")
    searxng_enabled: bool = Field(default=False, alias="SEARXNG_ENABLED")

    # Ollama dual-model: llama3.2:3b = tool calling, phi3.5-custom = generation
    # llama3.2:3b is the ONLY local model with confirmed tool calling (MX550 GPU)
    # phi3.5-custom is faster for pure generation but has NO tool calling support
    ollama_gen_model: str = Field(default="phi3.5-custom:latest", alias="OLLAMA_GEN_MODEL")
    ollama_tool_model: str = Field(default="llama3.2:3b", alias="OLLAMA_TOOL_MODEL")

    # News & Trend Detection APIs
    newsapi_org_key: str = Field(default="", alias="NEWSAPI_ORG_KEY")
    rapidapi_key: str = Field(default="", alias="RAPIDAPI_KEY")
    gnews_api_key: str = Field(default="", alias="GNEWS_API_KEY")
    mediastack_api_key: str = Field(default="", alias="MEDIASTACK_API_KEY")
    thenewsapi_key: str = Field(default="", alias="THENEWSAPI_KEY")
    
    # Email Finder APIs
    apollo_api_key: str = Field(default="", alias="APOLLO_API_KEY")
    hunter_api_key: str = Field(default="", alias="HUNTER_API_KEY")
    
    # Application Settings
    country: str = Field(default="India", alias="COUNTRY")
    country_code: str = Field(default="IN", alias="COUNTRY_CODE")
    max_trends: int = Field(default=12, alias="MAX_TRENDS")
    max_companies_per_trend: int = Field(default=15, alias="MAX_COMPANIES_PER_TREND")
    max_contacts_per_company: int = Field(default=6, alias="MAX_CONTACTS_PER_COMPANY")  # 3 DMs + 3 influencers
    email_confidence_threshold: int = Field(default=70, alias="EMAIL_CONFIDENCE_THRESHOLD")
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")
    show_tooltips: bool = Field(default=True, alias="SHOW_TOOLTIPS")
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./leads.db",
        alias="DATABASE_URL"
    )

    # ── API Configuration ──
    api_key: str = Field(default="", alias="API_KEY")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        alias="CORS_ORIGINS",
    )
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ── Engine Phase Timeouts (seconds) ──
    # High timeouts: prefer patience over skipping. GCP rate limits = just wait.
    engine_event_class_timeout: float = Field(default=600.0, alias="ENGINE_EVENT_CLASS_TIMEOUT")
    engine_clustering_timeout: float = Field(default=120.0, alias="ENGINE_CLUSTERING_TIMEOUT")
    engine_validation_timeout: float = Field(default=600.0, alias="ENGINE_VALIDATION_TIMEOUT")
    engine_synthesis_timeout: float = Field(default=600.0, alias="ENGINE_SYNTHESIS_TIMEOUT")
    engine_causal_timeout: float = Field(default=300.0, alias="ENGINE_CAUSAL_TIMEOUT")

    # ── Provider Resilience ──
    # Rate limit cooldown: base seconds per 429 failure (doubles each time, capped)
    provider_ratelimit_base_seconds: float = Field(default=15.0, alias="PROVIDER_RATELIMIT_BASE_SECONDS")
    provider_ratelimit_max_seconds: float = Field(default=120.0, alias="PROVIDER_RATELIMIT_MAX_SECONDS")
    # Generic error cooldown: seconds per non-429 failure
    provider_error_base_seconds: float = Field(default=30.0, alias="PROVIDER_ERROR_BASE_SECONDS")
    # Failures before marking provider "broken" (still usable after backoff expires)
    provider_broken_threshold: int = Field(default=8, alias="PROVIDER_BROKEN_THRESHOLD")

    # ── Cross-Trend Causal Council (Layer 6 in engine) ──
    cross_trend_max_candidates: int = Field(default=5, alias="CROSS_TREND_MAX_CANDIDATES")
    cross_trend_max_cascades: int = Field(default=2, alias="CROSS_TREND_MAX_CASCADES")
    cross_trend_inner_timeout: float = Field(default=120.0, alias="CROSS_TREND_INNER_TIMEOUT")

    # ── Per-Trend Causal Council (Step 3.7 in orchestrator) ──
    per_trend_max_impacts: int = Field(default=8, alias="PER_TREND_MAX_IMPACTS")

    # ── Lead Gen Agent ──
    # Max seconds for the entire lead_gen step (company + contact + outreach)
    # High timeout: let it try all companies patiently with rate-limited providers
    lead_gen_timeout: float = Field(default=900.0, alias="LEAD_GEN_TIMEOUT")

    # ── LLM Service ──
    # Max seconds to wait when all providers are in cooldown
    # 300s = patient wait for GCP rate limit recovery (was 30s = too aggressive)
    llm_max_provider_wait: float = Field(default=300.0, alias="LLM_MAX_PROVIDER_WAIT")

    # ── Coherence Grade Boundaries ──
    coherence_grade_a: float = Field(default=0.70, alias="COHERENCE_GRADE_A")
    coherence_grade_b: float = Field(default=0.55, alias="COHERENCE_GRADE_B")
    coherence_grade_c: float = Field(default=0.40, alias="COHERENCE_GRADE_C")
    coherence_grade_d: float = Field(default=0.25, alias="COHERENCE_GRADE_D")

    @property
    def enterprise_blocklist_set(self) -> frozenset:
        """Parse enterprise_blocklist CSV into a frozenset for O(1) lookup."""
        return frozenset(t.strip().lower() for t in self.enterprise_blocklist.split(",") if t.strip())

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

# Coherent Market Insights Service Catalog (Full)
CMI_SERVICES = {
    "procurement_intelligence": {
        "name": "Procurement Intelligence",
        "offerings": [
            "Supplier identification and profiling",
            "Cost structure and should-cost analysis",
            "Commodity and category market analysis",
            "Supply base risk assessment and mitigation",
            "Benchmarking of procurement practices",
            "Supplier performance evaluation",
            "Procurement process optimization",
            "Contract and negotiation support",
            "Spend analysis and savings opportunity identification",
        ],
        "keywords": ["supply chain", "procurement", "supplier", "sourcing", "cost", "vendor", "raw material", "spend", "contract", "negotiation", "commodity"]
    },
    "market_intelligence": {
        "name": "Market Intelligence",
        "offerings": [
            "Market sizing and segmentation",
            "Market trends and growth forecasts",
            "Customer needs and behavior analysis",
            "Regulatory and policy landscape assessment",
            "Channel and distribution analysis",
            "Opportunity and threat identification",
            "Product and service landscape mapping",
            "Market entry and expansion feasibility",
            "Trade Analysis (Export-import analysis)",
            "Pricing Analysis",
        ],
        "keywords": ["market", "growth", "expansion", "entry", "fta", "trade", "export", "import", "demand", "pricing", "channel", "distribution", "segmentation"]
    },
    "competitive_intelligence": {
        "name": "Competitive Intelligence",
        "offerings": [
            "Competitor profiling and benchmarking",
            "Analysis of competitor strategies, strengths, and weaknesses",
            "Product and service comparisons",
            "Pricing and go-to-market analysis",
            "Tracking competitor marketing and sales activities",
            "Monitoring of new product launches and innovations",
            "Mergers, acquisitions, and partnership tracking",
            "Sector and industry trend analysis",
        ],
        "keywords": ["competitor", "competition", "merger", "acquisition", "market share", "benchmark", "pricing", "product launch", "innovation"]
    },
    "market_monitoring": {
        "name": "Market Monitoring",
        "offerings": [
            "Ongoing tracking of market trends and developments",
            "Real-time updates on regulatory and economic changes",
            "Monitoring competitor and supplier activities",
            "Periodic market and industry reports",
            "Alerts on key market events and disruptions",
            "Tracking customer sentiment and feedback",
            "Early warning systems for emerging risks",
        ],
        "keywords": ["regulation", "policy", "compliance", "disruption", "risk", "change", "update", "monitoring", "alert", "sentiment", "tracking"]
    },
    "industry_analysis": {
        "name": "Industry Analysis",
        "offerings": [
            "Industry structure and value chain mapping",
            "Key industry drivers and challenges",
            "Regulatory and compliance environment review",
            "Analysis of technological advancements and disruptions",
            "Industry benchmarking and best practices",
            "Demand and supply dynamics assessment",
            "Identification of key players and market shares",
        ],
        "keywords": ["industry", "sector", "manufacturing", "pharma", "automotive", "electronics", "chemical", "value chain", "benchmarking", "best practices"]
    },
    "technology_research": {
        "name": "Technology Research",
        "offerings": [
            "Technology landscape and trends analysis",
            "Assessment of emerging and disruptive technologies",
            "Technology adoption and impact studies",
            "Patent and intellectual property analysis",
            "Vendor and solution evaluation",
            "R&D pipeline and innovation tracking",
            "Technology feasibility and ROI assessment",
        ],
        "keywords": ["technology", "AI", "automation", "digital", "innovation", "R&D", "tech", "software", "patent", "IP", "feasibility", "ROI", "vendor"]
    },
    "cross_border_expansion": {
        "name": "Cross Border Expansion",
        "offerings": [
            "Market entry strategy and feasibility studies",
            "Regulatory and compliance advisory for new markets",
            "Local partner and supplier identification",
            "Cultural and consumer behavior analysis",
            "Go-to-market planning and localization",
            "Competitive landscape in target geographies",
            "Risk assessment and mitigation for international operations",
        ],
        "keywords": ["expansion", "international", "export", "import", "FTA", "global", "cross-border", "foreign", "localization", "geography", "entry"]
    },
    "consumer_insights": {
        "name": "Consumer Insights",
        "offerings": [
            "Consumer behavior and attitude analysis",
            "Segmentation and persona development",
            "Customer journey mapping",
            "Brand perception and loyalty studies",
            "Product and service usage analysis",
            "Voice of customer (VoC) research",
            "Socio-demographic and psychographic profiling",
            "Customer satisfaction and NPS tracking",
        ],
        "keywords": ["consumer", "customer", "brand", "retail", "FMCG", "D2C", "e-commerce", "journey", "persona", "NPS", "loyalty", "VoC"]
    },
    "consulting_advisory": {
        "name": "Consulting and Advisory Services",
        "offerings": [
            "Strategic planning and business transformation",
            "Financial advisory and performance improvement",
            "Operational efficiency and process optimization",
            "Technology and digital transformation advisory",
            "Change management and organizational development",
            "Market entry and growth strategy",
        ],
        "keywords": ["strategy", "transformation", "growth", "efficiency", "optimization", "advisory", "financial", "change management", "digital transformation", "performance"]
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
    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1-2: Additional Business Publications (RSS - Added 2026-02-19)
    # ─────────────────────────────────────────────────────────────────────────
    "cnbctv18": {
        "id": "cnbctv18",
        "name": "CNBC TV18",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.cnbctv18.com",
        "rss_url": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/most-recent.xml",
        "categories": ["business", "markets", "economy"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "cnbctv18_market": {
        "id": "cnbctv18_market",
        "name": "CNBC TV18 Market",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.cnbctv18.com/market",
        "rss_url": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
        "categories": ["markets", "stocks", "finance"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "zeebiz": {
        "id": "zeebiz",
        "name": "Zee Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.zeebiz.com",
        "rss_url": "https://www.zeebiz.com/india-economy.xml",
        "categories": ["economy", "business", "industry"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "indiatoday_business": {
        "id": "indiatoday_business",
        "name": "India Today Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://www.indiatoday.in/business",
        "rss_url": "https://www.indiatoday.in/rss/1206574",
        "categories": ["business", "economy", "companies"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "mint_markets": {
        "id": "mint_markets",
        "name": "Mint Markets",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://www.livemint.com/market",
        "rss_url": "https://www.livemint.com/rss/markets",
        "categories": ["markets", "stocks", "IPO"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "mint_economy": {
        "id": "mint_economy",
        "name": "Mint Economy",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://www.livemint.com/economy",
        "rss_url": "https://www.livemint.com/rss/economy",
        "categories": ["economy", "policy", "macro"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "mint_industry": {
        "id": "mint_industry",
        "name": "Mint Industry",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://www.livemint.com/industry",
        "rss_url": "https://www.livemint.com/rss/industry",
        "categories": ["industry", "manufacturing", "sectors"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "mc_topnews": {
        "id": "mc_topnews",
        "name": "Moneycontrol Top News",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.92,
        "url": "https://www.moneycontrol.com",
        "rss_url": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "categories": ["business", "markets", "top_stories"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "theprint": {
        "id": "theprint",
        "name": "ThePrint",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://theprint.in",
        "rss_url": "https://theprint.in/feed/",
        "categories": ["policy", "economy", "analysis"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "scrollin": {
        "id": "scrollin",
        "name": "Scroll.in",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.86,
        "url": "https://scroll.in",
        "rss_url": "http://feeds.feedburner.com/ScrollinArticles.rss",
        "categories": ["business", "policy", "analysis"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bbc_india": {
        "id": "bbc_india",
        "name": "BBC News India",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://www.bbc.com/news/world/asia/india",
        "rss_url": "http://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
        "categories": ["business", "economy", "geopolitical"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Bing News RSS (Free, query-based - added 2026-02-19)
    # Format: https://www.bing.com/news/search?format=RSS&q=<query>
    # ─────────────────────────────────────────────────────────────────────────
    "bing_india_business": {
        "id": "bing_india_business",
        "name": "Bing News India Business",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://www.bing.com/news",
        "rss_url": "https://www.bing.com/news/search?format=RSS&q=India+business+news",
        "categories": ["business", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bing_india_economy": {
        "id": "bing_india_economy",
        "name": "Bing News India Economy",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://www.bing.com/news",
        "rss_url": "https://www.bing.com/news/search?format=RSS&q=India+economy+policy",
        "categories": ["economy", "policy", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bing_india_startup": {
        "id": "bing_india_startup",
        "name": "Bing News India Startup",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://www.bing.com/news",
        "rss_url": "https://www.bing.com/news/search?format=RSS&q=India+startup+funding",
        "categories": ["startups", "funding", "aggregator"],
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
    "livemint", "mint_companies", "mint_markets", "mint_economy", "mint_industry",
    "moneycontrol", "mc_topnews",
    "cnbctv18", "cnbctv18_market",
    # bbc_india: 403 Forbidden
    # business_standard, bs_companies — 403 Forbidden again as of 2026-02-18
    # Tier 2: Startup & Tech (RSS - Unlimited)
    "yourstory", "inc42",
    # vccircle removed: returns 200 but malformed XML, feedparser fails
    # Tier 2: Additional Publications (RSS - Unlimited)
    # zeebiz: 403 Forbidden as of 2026-02-21
    "indiatoday_business", "theprint", "scrollin",
    # Government (RSS - often blocked by anti-bot)
    # pib: empty response as of 2026-02-21
    # Other Publications (RSS - Unlimited)
    "hindu_business", "ndtv_profit", "techcrunch_india",
    # Google News (RSS - Unofficial but works)
    "google_news_india_business", "google_news_india_tech",
    # Bing News (RSS - Free, query-based aggregator)
    "bing_india_business", "bing_india_economy", "bing_india_startup",
    # APIs (set env vars to activate — gracefully skipped if key missing)
    "newsapi_org",           # NEWSAPI_ORG_KEY - 100 calls/day (BEST)
    "rapidapi_realtime_news",         # RAPIDAPI_KEY - 500/day
    "rapidapi_google_news",           # RAPIDAPI_KEY - Google News
    "rapidapi_google_trends_news",    # RAPIDAPI_KEY - trending news
    "gnews",                          # GNEWS_API_KEY - 100/day
    "mediastack",                     # MEDIASTACK_API_KEY - 500/month
    "newsdata",                       # NEWSDATA_API_KEY - 500/month
    "thenewsapi",                     # THENEWSAPI_KEY - 100/day
    "webz_news",                      # WEBZ_API_KEY - 1000/month
    # gdelt_india, gdelt_india_business: connection failures removed
]
