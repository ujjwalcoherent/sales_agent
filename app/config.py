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
    spacy_model: str = Field(default="en_core_web_lg", alias="SPACY_MODEL")

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
    # source_credibility_weights — REMOVED (0 callers, March 2026 audit)

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
    rss_hours_ago: int = Field(default=120, alias="RSS_HOURS_AGO")
    rss_fetch_concurrency: int = Field(default=6, alias="RSS_FETCH_CONCURRENCY")
    rss_per_source_timeout: float = Field(default=15.0, alias="RSS_PER_SOURCE_TIMEOUT")
    rss_httpx_timeout: float = Field(default=12.0, alias="RSS_HTTPX_TIMEOUT")

    # ── Dedup Entity Thresholds ──
    # Entity fingerprint dedup: min shared entities for same-topic match
    dedup_entity_same_topic_min: int = Field(default=2, alias="DEDUP_ENTITY_SAME_TOPIC_MIN")
    # Entity fingerprint dedup: min shared entities for cross-topic match
    dedup_entity_cross_topic_min: int = Field(default=4, alias="DEDUP_ENTITY_CROSS_TOPIC_MIN")

    # hybrid_w_semantic/lexical/event/temporal — REMOVED (0 callers, March 2026 audit)
    # hybrid_same_source_penalty — REMOVED (0 callers)
    # hybrid_temporal_decay_hours — REMOVED (0 callers)
    # Similarity weights now live in intelligence/config.py (ClusteringParams)

    # ── Embedding Augmentation (engine.py) ──
    # Event-type one-hot augmentation scale
    event_augment_scale: float = Field(default=0.50, alias="EVENT_AUGMENT_SCALE")
    # Entity fingerprint augmentation scale
    entity_fp_scale: float = Field(default=0.30, alias="ENTITY_FP_SCALE")
    # Entity fingerprint hash buckets
    entity_fp_buckets: int = Field(default=32, alias="ENTITY_FP_BUCKETS")

    # ── Quality Composite Weights (coherence.py) ──
    # Weights for computing composite quality grade (must sum to 1.0)
    quality_w_coherence: float = Field(default=0.15, alias="QUALITY_W_COHERENCE")
    quality_w_entity: float = Field(default=0.10, alias="QUALITY_W_ENTITY")
    quality_w_cmi: float = Field(default=0.25, alias="QUALITY_W_CMI")
    quality_w_specificity: float = Field(default=0.25, alias="QUALITY_W_SPECIFICITY")
    quality_w_second_order: float = Field(default=0.15, alias="QUALITY_W_SECOND_ORDER")
    quality_w_source_diversity: float = Field(default=0.10, alias="QUALITY_W_SOURCE_DIVERSITY")

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
    tavily_enabled: bool = Field(default=True, alias="TAVILY_ENABLED")
    tavily_api_keys: str = Field(default="", alias="TAVILY_API_KEYS")

    # Ollama dual-model: llama3.2:3b = tool calling, phi3.5-custom = generation
    # llama3.2:3b is the ONLY local model with confirmed tool calling (MX550 GPU)
    # phi3.5-custom is faster for pure generation but has NO tool calling support
    ollama_gen_model: str = Field(default="phi3.5-custom:latest", alias="OLLAMA_GEN_MODEL")
    ollama_tool_model: str = Field(default="llama3.2:3b", alias="OLLAMA_TOOL_MODEL")

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
    
    # ── Email Sending (Brevo) ──
    # Global kill switch: set to True to enable email sending
    email_sending_enabled: bool = Field(default=False, alias="EMAIL_SENDING_ENABLED")
    # Test mode: when True, ALL emails go to email_test_recipient instead of real contacts
    email_test_mode: bool = Field(default=True, alias="EMAIL_TEST_MODE")
    email_test_recipient: str = Field(default="ujjwal@coherentmarketinsights.com", alias="EMAIL_TEST_RECIPIENT")
    # Brevo (formerly Sendinblue) transactional email API
    brevo_api_key: str = Field(default="", alias="BREVO_API_KEY")
    brevo_sender_email: str = Field(default="outreach@coherentmarketinsights.com", alias="BREVO_SENDER_EMAIL")
    brevo_sender_name: str = Field(default="Coherent Market Insights", alias="BREVO_SENDER_NAME")

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

    # coherence_grade_a/b/c/d — REMOVED (defined but never referenced, March 2026 audit)

    # ── Enrichment (ScrapeGraphAI) ──
    deep_enrichment_enabled: bool = Field(default=False, alias="DEEP_ENRICHMENT_ENABLED")
    scrapegraph_model: str = Field(default="openai/gpt-4.1-mini", alias="SCRAPEGRAPH_MODEL")
    scrapegraph_max_results: int = Field(default=3, alias="SCRAPEGRAPH_MAX_RESULTS")
    scrapegraph_timeout: int = Field(default=90, alias="SCRAPEGRAPH_TIMEOUT")
    website_scrape_enabled: bool = Field(default=True, alias="WEBSITE_SCRAPE_ENABLED")
    hiring_signals_enabled: bool = Field(default=True, alias="HIRING_SIGNALS_ENABLED")
    tech_ip_analysis_enabled: bool = Field(default=True, alias="TECH_IP_ANALYSIS_ENABLED")

    # ── Person Intelligence ──
    person_deep_intel_enabled: bool = Field(default=True, alias="PERSON_DEEP_INTEL_ENABLED")
    person_intel_sources: str = Field(
        default="medium,substack,github,company_bio,conferences",
        alias="PERSON_INTEL_SOURCES",
    )
    person_intel_staleness_days: int = Field(default=7, alias="PERSON_INTEL_STALENESS_DAYS")
    person_intel_max_urls: int = Field(default=5, alias="PERSON_INTEL_MAX_URLS")

    # ── Contact Discovery ──
    contact_role_inference: str = Field(default="llm", alias="CONTACT_ROLE_INFERENCE")  # "llm" | "default" | "manual"
    default_dm_roles: str = Field(
        default="CEO,CTO,CFO,COO,VP Operations,Founder",
        alias="DEFAULT_DM_ROLES",
    )
    default_influencer_roles: str = Field(
        default="VP Engineering,VP Product,Head of Strategy,VP Marketing,VP Sales",
        alias="DEFAULT_INFLUENCER_ROLES",
    )

    # ── News Collection ──
    news_lookback_days: int = Field(default=7, alias="NEWS_LOOKBACK_DAYS")
    news_max_articles: int = Field(default=50, alias="NEWS_MAX_ARTICLES")
    news_relevance_threshold: float = Field(default=0.5, alias="NEWS_RELEVANCE_THRESHOLD")
    historical_news_enabled: bool = Field(default=True, alias="HISTORICAL_NEWS_ENABLED")
    historical_news_months: int = Field(default=5, alias="HISTORICAL_NEWS_MONTHS")

    # ── Company Cache ──
    company_cache_days: int = Field(default=7, alias="COMPANY_CACHE_DAYS")
    company_cache_enabled: bool = Field(default=True, alias="COMPANY_CACHE_ENABLED")

    # ── Email Outreach ──
    email_personalization_depth: str = Field(default="deep", alias="EMAIL_PERSONALIZATION_DEPTH")  # "basic" | "deep"
    email_max_length: int = Field(default=300, alias="EMAIL_MAX_LENGTH")  # approximate word limit

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
    # Core trend categories
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

    # Extended categories
    "cybersecurity": ["CISO", "VP Security", "Head of Information Security", "Security Director"],
    "compliance": ["Chief Compliance Officer", "VP Legal", "Head of Risk", "General Counsel"],
    "data_privacy": ["DPO", "Chief Privacy Officer", "CISO", "VP Legal"],
    "digital_transformation": ["CTO", "CDO", "VP Engineering", "Head of Digital"],
    "ai_adoption": ["CTO", "Chief AI Officer", "VP Engineering", "Head of Data Science"],
    "cloud_migration": ["CTO", "VP Infrastructure", "Head of Cloud", "IT Director"],
    "cost_reduction": ["CFO", "COO", "VP Operations", "Head of Procurement"],
    "market_expansion": ["CEO", "CSO", "VP Business Development", "Head of Strategy"],
    "sustainability": ["Chief Sustainability Officer", "VP Sustainability", "Head of ESG", "Environmental Director"],
    "talent": ["CHRO", "VP People", "Head of Talent", "HR Director"],

    "default": ["CEO", "CTO", "CFO", "VP Operations", "Head of Strategy"],
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
    # NOTE: Topic IDs below are country-specific (India). For other countries,
    # override NEWS_SOURCES in .env or add equivalent topic IDs.
    # hl/gl/ceid params should match settings.country but topic ID is fixed.
    # ─────────────────────────────────────────────────────────────────────────
    "google_news_business": {
        "id": "google_news_business",
        "name": "Google News Business",
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
    "google_news_tech": {
        "id": "google_news_tech",
        "name": "Google News Technology",
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

    # ─────────────────────────────────────────────────────────────────────────
    # GLOBAL TIER 1: Major International Business Publications (RSS — verified 2026-03-05)
    # All tested: HTTP 200, valid RSS, returning 10-50 articles per fetch.
    # ─────────────────────────────────────────────────────────────────────────
    "cnbc_world": {
        "id": "cnbc_world",
        "name": "CNBC World News",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.94,
        "url": "https://www.cnbc.com",
        "rss_url": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "categories": ["business", "global", "markets", "economy"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "cnbc_tech": {
        "id": "cnbc_tech",
        "name": "CNBC Technology",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.94,
        "url": "https://www.cnbc.com",
        "rss_url": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "categories": ["technology", "AI", "enterprise", "global"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "yahoo_finance": {
        "id": "yahoo_finance",
        "name": "Yahoo Finance",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.90,
        "url": "https://finance.yahoo.com",
        "rss_url": "https://finance.yahoo.com/news/rssindex",
        "categories": ["markets", "finance", "earnings", "M&A"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "marketwatch": {
        "id": "marketwatch",
        "name": "MarketWatch",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.91,
        "url": "https://www.marketwatch.com",
        "rss_url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "categories": ["markets", "finance", "economy", "corporate"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # GLOBAL TIER 2: Business & Tech Publications (RSS — verified 2026-03-05)
    # ─────────────────────────────────────────────────────────────────────────
    "guardian_business": {
        "id": "guardian_business",
        "name": "The Guardian Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.92,
        "url": "https://www.theguardian.com/uk/business",
        "rss_url": "https://www.theguardian.com/uk/business/rss",
        "categories": ["business", "economy", "corporate", "global"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "forbes": {
        "id": "forbes",
        "name": "Forbes",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.89,
        "url": "https://www.forbes.com",
        "rss_url": "https://www.forbes.com/innovation/feed2/",
        "categories": ["business", "technology", "innovation", "leadership"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "fortune": {
        "id": "fortune",
        "name": "Fortune",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.91,
        "url": "https://fortune.com",
        "rss_url": "https://fortune.com/feed/",
        "categories": ["business", "leadership", "corporate", "Fortune500"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "fast_company": {
        "id": "fast_company",
        "name": "Fast Company",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.fastcompany.com",
        "rss_url": "https://www.fastcompany.com/latest/rss",
        "categories": ["business", "innovation", "technology", "leadership"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "inc_magazine": {
        "id": "inc_magazine",
        "name": "Inc. Magazine",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.86,
        "url": "https://www.inc.com",
        "rss_url": "https://www.inc.com/rss/",
        "categories": ["startups", "entrepreneurship", "business", "SMB"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "wired_business": {
        "id": "wired_business",
        "name": "Wired Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://www.wired.com/category/business/",
        "rss_url": "https://www.wired.com/feed/category/business/latest/rss",
        "categories": ["technology", "business", "enterprise", "AI"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "ars_technica": {
        "id": "ars_technica",
        "name": "Ars Technica",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.91,
        "url": "https://arstechnica.com",
        "rss_url": "https://feeds.arstechnica.com/arstechnica/index",
        "categories": ["technology", "science", "policy", "enterprise"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # GLOBAL: Sector-Specific Feeds (RSS — verified 2026-03-05)
    # High-value B2B vertical feeds: fintech, AI, enterprise tech
    # ─────────────────────────────────────────────────────────────────────────
    "finextra": {
        "id": "finextra",
        "name": "Finextra",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://www.finextra.com",
        "rss_url": "https://www.finextra.com/rss/headlines.aspx",
        "categories": ["fintech", "banking", "payments", "regulation"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "techcrunch_ai": {
        "id": "techcrunch_ai",
        "name": "TechCrunch AI",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://techcrunch.com/category/artificial-intelligence/",
        "rss_url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "categories": ["AI", "ML", "enterprise_AI", "startups"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "techcrunch_fintech": {
        "id": "techcrunch_fintech",
        "name": "TechCrunch Fintech",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://techcrunch.com/category/fintech/",
        "rss_url": "https://techcrunch.com/category/fintech/feed/",
        "categories": ["fintech", "payments", "banking", "funding"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "siliconangle": {
        "id": "siliconangle",
        "name": "SiliconAngle",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.86,
        "url": "https://siliconangle.com",
        "rss_url": "https://siliconangle.com/feed/",
        "categories": ["enterprise", "cloud", "AI", "cybersecurity", "B2B"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "zdnet": {
        "id": "zdnet",
        "name": "ZDNet",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://www.zdnet.com",
        "rss_url": "https://www.zdnet.com/news/rss.xml",
        "categories": ["enterprise", "technology", "security", "cloud"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "pymnts": {
        "id": "pymnts",
        "name": "PYMNTS",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.pymnts.com",
        "rss_url": "https://www.pymnts.com/feed/",
        "categories": ["payments", "fintech", "ecommerce", "digital"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "banking_dive": {
        "id": "banking_dive",
        "name": "Banking Dive",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.bankingdive.com",
        "rss_url": "https://www.bankingdive.com/feeds/news/",
        "categories": ["banking", "regulation", "fintech", "compliance"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "seeking_alpha": {
        "id": "seeking_alpha",
        "name": "Seeking Alpha Market Currents",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://seekingalpha.com",
        "rss_url": "https://seekingalpha.com/market_currents.xml",
        "categories": ["markets", "earnings", "analysis", "stocks"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ASIA-PACIFIC: Regional Business Coverage (RSS — verified 2026-03-05)
    # ─────────────────────────────────────────────────────────────────────────
    "al_jazeera_economy": {
        "id": "al_jazeera_economy",
        "name": "Al Jazeera Economy",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://www.aljazeera.com/economy",
        "rss_url": "https://www.aljazeera.com/xml/rss/all.xml",
        "categories": ["economy", "global", "geopolitical", "trade"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "asia_times": {
        "id": "asia_times",
        "name": "Asia Times",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://asiatimes.com",
        "rss_url": "https://asiatimes.com/feed/",
        "categories": ["business", "geopolitical", "tech", "APAC"],
        "language": "en",
        "country": "APAC",
        "rate_limit_per_day": None
    },
    "channel_news_asia": {
        "id": "channel_news_asia",
        "name": "Channel News Asia Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.89,
        "url": "https://www.channelnewsasia.com",
        "rss_url": "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511",
        "categories": ["business", "economy", "APAC", "Singapore"],
        "language": "en",
        "country": "APAC",
        "rate_limit_per_day": None
    },
    "straits_times_biz": {
        "id": "straits_times_biz",
        "name": "Straits Times Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.89,
        "url": "https://www.straitstimes.com",
        "rss_url": "https://www.straitstimes.com/news/business/rss.xml",
        "categories": ["business", "economy", "APAC", "Singapore"],
        "language": "en",
        "country": "APAC",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # WIRE SERVICES: Press Release Aggregators (RSS — verified 2026-03-05)
    # Direct corporate announcements — M&A, earnings, product launches
    # ─────────────────────────────────────────────────────────────────────────
    "globe_newswire": {
        "id": "globe_newswire",
        "name": "GlobeNewswire",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.84,
        "url": "https://www.globenewswire.com",
        "rss_url": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire",
        "categories": ["press_releases", "corporate", "M&A", "earnings"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    # NOTE: PR Newswire category-specific feeds (/rss/news-releases/*-latest-news.rss)
    # all return 404 as of 2026-03-10. Only the general feed works.
    # prnewswire_tech, prnewswire_finance, prnewswire_ma removed — use prnewswire_india
    # (which uses the working /rss/news-releases-list.rss URL) for press release coverage.

    # ─────────────────────────────────────────────────────────────────────────
    # GOVERNMENT/REGULATORY: US Federal (RSS — verified 2026-03-05)
    # ─────────────────────────────────────────────────────────────────────────
    "fed_reserve": {
        "id": "fed_reserve",
        "name": "Federal Reserve Press Releases",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://www.federalreserve.gov",
        "rss_url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "categories": ["monetary_policy", "banking", "regulation", "rates"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # COMMUNITY: High-Quality Tech Discussion (RSS — verified 2026-03-05)
    # ─────────────────────────────────────────────────────────────────────────
    "hacker_news": {
        "id": "hacker_news",
        "name": "Hacker News (50+ points)",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.82,
        "url": "https://news.ycombinator.com",
        "rss_url": "https://hnrss.org/newest?points=50",
        "categories": ["technology", "startups", "AI", "engineering"],
        "language": "en",
        "country": "GLOBAL",
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

    # ── GDELT expansion: global B2B tech + India regulatory + VC funding ──────
    # Same _fetch_gdelt handler — different query strings baked into source config.
    # Each query targets a distinct sector so overlap is minimal.
    "gdelt_global_tech": {
        "id": "gdelt_global_tech",
        "name": "GDELT Global Tech",
        "source_type": "api",
        "tier": "tier_1",
        "credibility_score": 0.87,
        "url": "https://www.gdeltproject.org",
        "api_endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "categories": ["technology", "AI", "SaaS", "cloud", "enterprise"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": 500,
    },
    "gdelt_india_funding": {
        "id": "gdelt_india_funding",
        "name": "GDELT India Funding & VC",
        "source_type": "api",
        "tier": "tier_1",
        "credibility_score": 0.88,
        "url": "https://www.gdeltproject.org",
        "api_endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "categories": ["VC", "funding", "IPO", "M&A", "startup"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 500,
    },
    "gdelt_india_regulation": {
        "id": "gdelt_india_regulation",
        "name": "GDELT India Regulation",
        "source_type": "api",
        "tier": "tier_1",
        "credibility_score": 0.88,
        "url": "https://www.gdeltproject.org",
        "api_endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "categories": ["regulation", "policy", "RBI", "SEBI", "government"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": 500,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1: Reuters via Google News (feeds.reuters.com dead since 2020)
    # Using Google News source:reuters filter as proxy — verified 2026-03-10
    # ─────────────────────────────────────────────────────────────────────────
    "reuters_business": {
        "id": "reuters_business",
        "name": "Reuters Business (via Google News)",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.97,
        "url": "https://www.reuters.com",
        "rss_url": "https://news.google.com/rss/search?q=source:reuters+business&hl=en-US&gl=US&ceid=US:en",
        "categories": ["business", "markets", "global", "corporate"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "reuters_technology": {
        "id": "reuters_technology",
        "name": "Reuters Technology (via Google News)",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.97,
        "url": "https://www.reuters.com",
        "rss_url": "https://news.google.com/rss/search?q=source:reuters+technology&hl=en-US&gl=US&ceid=US:en",
        "categories": ["technology", "AI", "enterprise", "global"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 2: Business Wire & PR Newswire (Press release aggregators)
    # Free RSS, no auth needed — direct corporate announcements
    # ─────────────────────────────────────────────────────────────────────────
    "businesswire_tech": {
        "id": "businesswire_tech",
        "name": "Business Wire Tech",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://www.businesswire.com",
        "rss_url": "https://feed.businesswire.com/rss/home/?rss=G7",
        "categories": ["press_releases", "technology", "enterprise", "funding"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "businesswire_financial": {
        "id": "businesswire_financial",
        "name": "Business Wire Financial",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.85,
        "url": "https://www.businesswire.com",
        "rss_url": "https://feed.businesswire.com/rss/home/?rss=G6",
        "categories": ["press_releases", "finance", "earnings", "M&A"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "prnewswire_india": {
        "id": "prnewswire_india",
        "name": "PR Newswire India",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.84,
        "url": "https://www.prnewswire.com",
        "rss_url": "https://www.prnewswire.com/rss/news-releases-list.rss",
        "categories": ["press_releases", "corporate", "announcements"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # US-SPECIFIC: Major US business publications (country="US")
    # ─────────────────────────────────────────────────────────────────────────
    "wsj_business": {
        "id": "wsj_business",
        "name": "Wall Street Journal Business",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.97,
        "url": "https://www.wsj.com",
        "rss_url": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
        "categories": ["business", "economy", "markets", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "wsj_tech": {
        "id": "wsj_tech",
        "name": "Wall Street Journal Technology",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.97,
        "url": "https://www.wsj.com",
        "rss_url": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
        "categories": ["technology", "AI", "cybersecurity", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "wsj_markets": {
        "id": "wsj_markets",
        "name": "Wall Street Journal Markets",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.97,
        "url": "https://www.wsj.com",
        "rss_url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "categories": ["markets", "stocks", "IPO", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "business_insider": {
        "id": "business_insider",
        "name": "Business Insider",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://www.businessinsider.com",
        "rss_url": "https://feeds2.feedburner.com/businessinsider",
        "categories": ["business", "technology", "finance", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "nyt_business": {
        "id": "nyt_business",
        "name": "New York Times Business",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.96,
        "url": "https://www.nytimes.com",
        "rss_url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "categories": ["business", "economy", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "nyt_technology": {
        "id": "nyt_technology",
        "name": "New York Times Technology",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.96,
        "url": "https://www.nytimes.com",
        "rss_url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "categories": ["technology", "AI", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "google_news_us_business": {
        "id": "google_news_us_business",
        "name": "Google News US Business",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",
        "categories": ["business", "aggregator", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },
    "google_news_us_tech": {
        "id": "google_news_us_tech",
        "name": "Google News US Technology",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",
        "categories": ["technology", "aggregator", "US"],
        "language": "en",
        "country": "US",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # EU-SPECIFIC: European business publications (country="EU")
    # ─────────────────────────────────────────────────────────────────────────
    "bbc_business": {
        "id": "bbc_business",
        "name": "BBC Business",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.96,
        "url": "https://www.bbc.co.uk/news/business",
        "rss_url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "categories": ["business", "economy", "EU"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },
    "bbc_technology": {
        "id": "bbc_technology",
        "name": "BBC Technology",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.96,
        "url": "https://www.bbc.co.uk/news/technology",
        "rss_url": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "categories": ["technology", "AI", "EU"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },
    "dw_business": {
        "id": "dw_business",
        "name": "Deutsche Welle Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://www.dw.com",
        "rss_url": "https://rss.dw.com/xml/rss-en-bus",
        "categories": ["business", "economy", "EU", "Germany"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },
    "sky_business": {
        "id": "sky_business",
        "name": "Sky News Business",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://news.sky.com",
        "rss_url": "https://news.sky.com/feeds/rss/business.xml",
        "categories": ["business", "markets", "EU", "UK"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },
    "euractiv_economy": {
        "id": "euractiv_economy",
        "name": "EurActiv Economy & Jobs",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.euractiv.com",
        "rss_url": "https://www.euractiv.com/sections/economy-jobs/feed/",
        "categories": ["economy", "policy", "EU", "regulation"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },
    "euractiv_digital": {
        "id": "euractiv_digital",
        "name": "EurActiv Digital",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://www.euractiv.com",
        "rss_url": "https://www.euractiv.com/sections/digital/feed/",
        "categories": ["technology", "digital", "EU", "regulation", "AI"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },
    "google_news_uk_business": {
        "id": "google_news_uk_business",
        "name": "Google News UK Business",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pIUWlnQVAB?hl=en-GB&gl=GB&ceid=GB:en",
        "categories": ["business", "aggregator", "UK", "EU"],
        "language": "en",
        "country": "EU",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 2: VentureBeat & TechCrunch main (Global tech B2B)
    # ─────────────────────────────────────────────────────────────────────────
    "venturebeat": {
        "id": "venturebeat",
        "name": "VentureBeat",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.88,
        "url": "https://venturebeat.com",
        "rss_url": "https://venturebeat.com/feed/",
        "categories": ["technology", "AI", "enterprise", "startups", "B2B"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
    "techcrunch_main": {
        "id": "techcrunch_main",
        "name": "TechCrunch",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.90,
        "url": "https://techcrunch.com",
        "rss_url": "https://techcrunch.com/feed/",
        "categories": ["technology", "startups", "funding", "AI", "B2B"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # DEAD: Business Standard section feeds — ALL return 403 as of 2026-03-10
    # Kept in NEWS_SOURCES for reference but removed from DEFAULT_ACTIVE_SOURCES
    # ─────────────────────────────────────────────────────────────────────────
    "bs_economy": {
        "id": "bs_economy",
        "name": "Business Standard Economy",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.business-standard.com/economy",
        "rss_url": "https://www.business-standard.com/rss/economy-policy-101.rss",
        "categories": ["economy", "policy", "budget", "macro"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bs_finance": {
        "id": "bs_finance",
        "name": "Business Standard Finance",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.business-standard.com/finance",
        "rss_url": "https://www.business-standard.com/rss/finance-102.rss",
        "categories": ["finance", "banking", "insurance", "RBI"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bs_tech": {
        "id": "bs_tech",
        "name": "Business Standard Technology",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.93,
        "url": "https://www.business-standard.com/technology",
        "rss_url": "https://www.business-standard.com/rss/technology-108.rss",
        "categories": ["technology", "IT", "startups", "AI"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 2: ET specialised feeds (B2B / CFO / BFSI / Infra)
    # ─────────────────────────────────────────────────────────────────────────
    "et_markets": {
        "id": "et_markets",
        "name": "ET Markets",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://economictimes.indiatimes.com/markets",
        "rss_url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "categories": ["markets", "stocks", "IPO", "NSE", "BSE"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "et_startup": {
        "id": "et_startup",
        "name": "ET Startup",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://economictimes.indiatimes.com/small-biz/startups",
        "rss_url": "https://economictimes.indiatimes.com/small-biz/startups/rssfeeds/15117369.cms",
        "categories": ["startups", "funding", "founders", "unicorn"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "et_bfsi": {
        "id": "et_bfsi",
        "name": "ET BFSI",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.95,
        "url": "https://bfsi.economictimes.indiatimes.com",
        "rss_url": "https://bfsi.economictimes.indiatimes.com/rss",
        "categories": ["banking", "fintech", "NBFC", "insurance", "regulation"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "et_cio": {
        "id": "et_cio",
        "name": "ET CIO",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.92,
        "url": "https://cio.economictimes.indiatimes.com",
        "rss_url": "https://cio.economictimes.indiatimes.com/rss",
        "categories": ["CIO", "IT", "enterprise", "digital_transformation"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "et_infra": {
        "id": "et_infra",
        "name": "ET Infra",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.92,
        "url": "https://infra.economictimes.indiatimes.com",
        "rss_url": "https://infra.economictimes.indiatimes.com/rss",
        "categories": ["infrastructure", "logistics", "energy", "real_estate"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 3: SEBI / RBI corrected URLs + PIB Ministry feeds
    # SEBI official RSS: /sebirss.xml (confirmed in SEBI docs)
    # RBI: /scripts/BS_PressReleaseView.aspx (XML variant)
    # PIB Ministry of Finance and Commerce specific feeds
    # ─────────────────────────────────────────────────────────────────────────
    "sebi_v2": {
        "id": "sebi_v2",
        "name": "SEBI Announcements",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://www.sebi.gov.in",
        "rss_url": "https://www.sebi.gov.in/sebirss.xml",
        "categories": ["regulation", "capital_markets", "compliance", "SEBI"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "rbi_v2": {
        "id": "rbi_v2",
        "name": "RBI Press Releases",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://www.rbi.org.in",
        "rss_url": "https://www.rbi.org.in/scripts/rss.aspx",
        "categories": ["banking", "monetary_policy", "regulation", "RBI"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "pib_finance": {
        "id": "pib_finance",
        "name": "PIB Ministry of Finance",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://pib.gov.in",
        "rss_url": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
        "categories": ["government", "policy", "budget", "taxation"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "pib_commerce": {
        "id": "pib_commerce",
        "name": "PIB Ministry of Commerce",
        "source_type": "rss",
        "tier": "tier_1",
        "credibility_score": 0.99,
        "url": "https://pib.gov.in",
        "rss_url": "https://pib.gov.in/RssMain.aspx?ModId=7&Lang=1&Regid=3",
        "categories": ["government", "trade", "export", "import", "commerce"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 2: Google News India — additional topic feeds
    # Topic IDs are stable for major categories
    # ─────────────────────────────────────────────────────────────────────────
    "google_news_economy": {
        "id": "google_news_economy",
        "name": "Google News Economy",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/headlines/section/topic/ECONOMY?hl=en-IN&gl=IN&ceid=IN:en",
        "categories": ["economy", "aggregator", "india"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "google_news_india_startup": {
        "id": "google_news_india_startup",
        "name": "Google News India Startup",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/search?q=India+startup+funding&hl=en-IN&gl=IN&ceid=IN:en",
        "categories": ["startups", "funding", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "google_news_india_fintech": {
        "id": "google_news_india_fintech",
        "name": "Google News India Fintech",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://news.google.com",
        "rss_url": "https://news.google.com/rss/search?q=India+fintech+payments&hl=en-IN&gl=IN&ceid=IN:en",
        "categories": ["fintech", "payments", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 2: The Wire (India) — investigative business/policy
    # ─────────────────────────────────────────────────────────────────────────
    "thewire_economy": {
        "id": "thewire_economy",
        "name": "The Wire Economy",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.87,
        "url": "https://thewire.in",
        "rss_url": "https://thewire.in/economy/feed",
        "categories": ["economy", "policy", "analysis", "corporate"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 2: TechInAsia (Asia-Pacific startup coverage)
    # ─────────────────────────────────────────────────────────────────────────
    "techinasia": {
        "id": "techinasia",
        "name": "Tech in Asia",
        "source_type": "rss",
        "tier": "tier_2",
        "credibility_score": 0.86,
        "url": "https://www.techinasia.com",
        "rss_url": "https://www.techinasia.com/feed",
        "categories": ["startups", "tech", "Asia", "funding", "B2B"],
        "language": "en",
        "country": "APAC",
        "rate_limit_per_day": None
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NEW TIER 3: Bing News — additional query verticals
    # ─────────────────────────────────────────────────────────────────────────
    "bing_india_vc": {
        "id": "bing_india_vc",
        "name": "Bing News India VC",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://www.bing.com/news",
        "rss_url": "https://www.bing.com/news/search?format=RSS&q=India+venture+capital+private+equity",
        "categories": ["VC", "PE", "funding", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bing_india_regulation": {
        "id": "bing_india_regulation",
        "name": "Bing News India Regulation",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://www.bing.com/news",
        "rss_url": "https://www.bing.com/news/search?format=RSS&q=India+SEBI+RBI+regulation+policy",
        "categories": ["regulation", "policy", "aggregator"],
        "language": "en",
        "country": "IN",
        "rate_limit_per_day": None
    },
    "bing_global_saas": {
        "id": "bing_global_saas",
        "name": "Bing News Global SaaS",
        "source_type": "rss",
        "tier": "tier_3",
        "credibility_score": 0.80,
        "url": "https://www.bing.com/news",
        "rss_url": "https://www.bing.com/news/search?format=RSS&q=SaaS+enterprise+software+B2B+funding",
        "categories": ["SaaS", "enterprise", "B2B", "aggregator"],
        "language": "en",
        "country": "GLOBAL",
        "rate_limit_per_day": None
    },
}

# Quick access lists
RSS_SOURCES = [src for src in NEWS_SOURCES.values() if src["source_type"] == "rss"]
API_SOURCES = [src for src in NEWS_SOURCES.values() if src["source_type"] == "api"]
TIER_1_SOURCES = [src for src in NEWS_SOURCES.values() if src["tier"] == "tier_1"]
TIER_2_SOURCES = [src for src in NEWS_SOURCES.values() if src["tier"] == "tier_2"]

# Default sources to use (can be overridden via env)
# ─────────────────────────────────────────────────────────────────────────────
# BROKEN (never re-add — permanently dead feeds as of 2026-03-05):
#   - financial_express      : 410 Gone — feed permanently removed
#   - rbi_press              : 418 — original URL anti-bot blocked (replaced by rbi_v2)
#   - business_today         : 404 Not Found — feed URL dead
#   - entrackr               : 404 Not Found — no public RSS
#   - sebi                   : 404 Not Found — old URL dead (replaced by sebi_v2)
#   - vccircle               : 200 but malformed XML — feedparser parse error
#   - bbc_india              : 403 Forbidden — BBC geo-blocks India RSS
#   - zeebiz                 : 403 Forbidden — bot-blocked
#   - theprint               : Intermittent silent failure (kept with low priority)
#   - pib (original)         : 0 articles — empty response (replaced by pib_finance/pib_commerce)
#   - business_standard      : 403 Forbidden (top-level feed) — replaced by section feeds
#   - bs_companies           : 403 Forbidden — all BS feeds bot-blocked
#   - reuters_business       : feeds.reuters.com dead since 2020 — replaced with Google News proxy (2026-03-10)
#   - reuters_technology     : feeds.reuters.com dead since 2020 — replaced with Google News proxy (2026-03-10)
#   - bs_economy             : 403 Forbidden — all business-standard.com/rss/ feeds bot-blocked (2026-03-10)
#   - bs_finance             : 403 Forbidden — all business-standard.com/rss/ feeds bot-blocked (2026-03-10)
#   - bs_tech                : 403 Forbidden — all business-standard.com/rss/ feeds bot-blocked (2026-03-10)
#   - prnewswire_tech        : 404 — category feeds removed by PR Newswire (2026-03-10)
#   - prnewswire_finance     : 404 — category feeds removed by PR Newswire (2026-03-10)
#   - prnewswire_ma          : 404 — category feeds removed by PR Newswire (2026-03-10)
#   - business_insider (old) : /sai/rss 404 — replaced with feedburner feed (2026-03-10)
#   - thewire_economy        : 0 articles — feed returns empty (removed 2026-03-05)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ACTIVE_SOURCES = [
    # ── Tier 1: Major Indian Business Publications (RSS — Working) ────────────
    "economic_times", "et_industry", "et_tech", "et_markets", "et_startup",
    "livemint", "mint_companies", "mint_markets", "mint_economy", "mint_industry",
    "moneycontrol", "mc_topnews",
    "cnbctv18", "cnbctv18_market",
    "hindu_business",

    # ── Tier 1: ET vertical portals ───────────────────────────────────────────
    "et_bfsi", "et_cio", "et_infra",

    # ── Tier 1: Global Wire Services (via Google News proxy — verified 2026-03-10) ─
    "reuters_business",       # Reuters Business (via Google News source:reuters)
    "reuters_technology",     # Reuters Technology (via Google News source:reuters)

    # ── Tier 1: Global Business Publications (RSS — verified 2026-03-05) ─────
    "cnbc_world",             # CNBC World News (30 articles/fetch)
    "cnbc_tech",              # CNBC Technology (30 articles/fetch)
    "yahoo_finance",          # Yahoo Finance (42 articles/fetch)
    "marketwatch",            # MarketWatch Top Stories

    # ── Tier 2: Global Business & Tech Publications ──────────────────────────
    "guardian_business",      # The Guardian Business (39 articles/fetch)
    "forbes",                 # Forbes Innovation
    "fortune",                # Fortune (10 articles/fetch)
    "fast_company",           # Fast Company (20 articles/fetch)
    "inc_magazine",           # Inc. Magazine (37 articles/fetch)
    "wired_business",         # Wired Business (20 articles/fetch)
    "ars_technica",           # Ars Technica (20 articles/fetch)

    # ── Tier 2: Startup & Tech India ─────────────────────────────────────────
    "yourstory", "inc42",
    "techcrunch_india",       # TechCrunch India tag (working: 20 articles)
    "techcrunch_main",        # TechCrunch global feed
    "venturebeat",            # B2B tech + AI + enterprise
    "techinasia",             # Asia-Pacific startup coverage

    # ── Tier 2: Sector-Specific (Fintech, AI, Enterprise) ───────────────────
    "finextra",               # Fintech & banking (47 articles/fetch)
    "techcrunch_ai",          # TechCrunch AI vertical (19 articles/fetch)
    "techcrunch_fintech",     # TechCrunch Fintech vertical (20 articles/fetch)
    "siliconangle",           # Enterprise tech (30 articles/fetch)
    "zdnet",                  # Enterprise IT (20 articles/fetch)
    "pymnts",                 # Payments & digital commerce (10 articles/fetch)
    "banking_dive",           # Banking industry (10 articles/fetch)
    "seeking_alpha",          # Market currents & analysis

    # ── Tier 2: Asia-Pacific ─────────────────────────────────────────────────
    "al_jazeera_economy",     # Global economy & geopolitics (25 articles/fetch)
    "asia_times",             # Asia business & tech (20 articles/fetch)
    "channel_news_asia",      # CNA Business (20 articles/fetch)
    "straits_times_biz",      # Straits Times Business (28 articles/fetch)

    # ── Tier 2: Additional Indian Publications ────────────────────────────────
    "indiatoday_business", "scrollin",
    "ndtv_profit",
    "theprint",               # Intermittent — kept, gracefully skipped on failure

    # ── Tier 2: Press Release Aggregators (direct corporate news) ────────────
    "businesswire_tech", "businesswire_financial",
    "prnewswire_india",       # General feed (category feeds 404 since 2026-03-10)
    "globe_newswire",         # GlobeNewswire (20 articles/fetch)

    # ── Tier 1: US-Specific Business Publications ─────────────────────────────
    "wsj_business",           # Wall Street Journal Business
    "wsj_tech",               # Wall Street Journal Technology
    "wsj_markets",            # Wall Street Journal Markets
    "nyt_business",           # New York Times Business
    "nyt_technology",         # New York Times Technology
    "business_insider",       # Business Insider (tech + finance)

    # ── Tier 1-2: EU-Specific Business Publications ─────────────────────────
    "bbc_business",           # BBC Business (UK/Europe)
    "bbc_technology",         # BBC Technology (UK/Europe)
    "dw_business",            # Deutsche Welle Business (Germany/Europe)
    "sky_business",           # Sky News Business (UK)
    "euractiv_economy",       # EurActiv Economy & Jobs (EU policy)
    "euractiv_digital",       # EurActiv Digital (EU tech policy)

    # ── Tier 1: Government & Regulatory ──────────────────────────────────────
    "sebi_v2",           # SEBI: /sebirss.xml (official URL from SEBI docs)
    "rbi_v2",            # RBI: /scripts/rss.aspx (alternate — test before enabling)
    "pib_finance",       # PIB Finance Ministry
    "pib_commerce",      # PIB Commerce Ministry
    "fed_reserve",       # US Federal Reserve (monetary policy, rates)

    # ── Tier 3: Google News (unofficial but reliable) ─────────────────────────
    "google_news_business", "google_news_tech",
    "google_news_economy",
    "google_news_india_startup",
    "google_news_india_fintech",
    "google_news_us_business",    # US Business Google News
    "google_news_us_tech",        # US Tech Google News
    "google_news_uk_business",    # UK Business Google News

    # ── Tier 3: Bing News (free query-based aggregator) ───────────────────────
    "bing_india_business", "bing_india_economy", "bing_india_startup",
    "bing_india_vc", "bing_india_regulation",
    "bing_global_saas",

    # ── Tier 3: Community ────────────────────────────────────────────────────
    "hacker_news",            # HN 50+ points (tech signal)

    # ── Tier 1: GDELT (free API — no key — high volume) ──────────────────────
    "gdelt_india",               # Broad India: ~236 articles/run
    "gdelt_india_business",      # India business keywords: ~179 articles/run
    "gdelt_india_funding",       # India VC/IPO/M&A events
    "gdelt_india_regulation",    # India RBI/SEBI/policy
    "gdelt_global_tech",         # Global enterprise SaaS/AI/cybersecurity

    # ── APIs (gracefully skipped if env key missing) ──────────────────────────
    "newsapi_org",                    # NEWSAPI_ORG_KEY   — 100 calls/day (BEST)
    "rapidapi_realtime_news",         # RAPIDAPI_KEY      — 500/day
    "rapidapi_google_news",           # RAPIDAPI_KEY      — Google News
    "rapidapi_google_trends_news",    # RAPIDAPI_KEY      — trending news
    "gnews",                          # GNEWS_API_KEY     — 100/day
    "mediastack",                     # MEDIASTACK_API_KEY — 500/month
    "newsdata",                       # NEWSDATA_API_KEY  — 500/month
    "thenewsapi",                     # THENEWSAPI_KEY    — 100/day
    "webz_news",                      # WEBZ_API_KEY      — 1000/month
]
