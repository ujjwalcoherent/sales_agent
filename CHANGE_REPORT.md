# Change Report — India Trend Lead Agent

**Date:** February 2026
**Base Commit:** `6b63887` — *Initial commit: India Trend Lead Agent with Streamlit UI*
**Scope:** All uncommitted changes since initial commit

---

## Executive Summary

The codebase has been significantly upgraded from a basic lead generation agent into a **full business intelligence pipeline**. The core addition is a multi-stage news processing system that ingests articles from 20+ sources, clusters them semantically, detects trends via LLM, and synthesizes structured business intelligence — all feeding into the existing company-finding and outreach pipeline.

**Impact:** 11 modified files, 4 new files, ~1,400+ new lines of code.

---

## 1. New ML/NLP Pipeline (4 New Files)

### Why — The Problem

The original system had a single `TrendAgent` that received raw article text and tried to detect trends in one shot. This had fundamental limitations:

- **No semantic understanding:** Articles about the same event (e.g., "RBI raises repo rate" from 5 different sources) were treated as 5 separate trends instead of 1 signal with high confidence. The system couldn't tell that differently-worded articles were about the same thing.
- **No signal-to-noise separation:** A single mention of a topic looked the same as 15 articles about it. Volume-as-signal was completely lost.
- **LLM context limits:** Sending 50+ raw articles to an LLM in one prompt produces shallow output. The model can't deeply analyze everything at once.
- **No structured output:** Trends came back as free text, making downstream processing (impact analysis, company matching) unreliable.

### Why — The Solution: Embeddings + Clustering + LLM Synthesis

The fix was to split "trend detection" into discrete stages, each doing one thing well:

1. **Embeddings** — Convert articles into numerical vectors so we can mathematically compare them. Two articles about the same RBI rate hike will have vectors pointing in nearly the same direction (high cosine similarity), even if they use completely different words. This is the foundation that makes everything else possible.

2. **Clustering** — Group similar articles together automatically. Instead of the LLM seeing 50 raw articles, it now sees "here are 5 clusters, each containing related articles." This is a massive noise reduction — the LLM can focus on understanding 5 distinct signals instead of parsing 50 noisy articles.

3. **Smart Trend Detection** — Use the LLM's semantic reasoning on the *clustered* output. Now the LLM isn't doing raw grouping (which it's mediocre at); it's doing synthesis and analysis (which it's excellent at). Each cluster is a pre-validated signal.

4. **Trend Synthesis** — Convert LLM analysis into structured `MajorTrend` objects with typed fields (sector, severity, impact type, confidence score). This makes downstream processing deterministic and reliable.

### What Changed

#### `app/tools/embeddings.py` (248 lines) — NEW
- **EmbeddingTool** class for generating text embeddings
- Primary backend: HuggingFace Inference API (cloud-based, `all-MiniLM-L6-v2`)
- Fallback backend: Ollama (local)
- Methods: `embed_text()`, `embed_batch()`, `compute_similarity()`, `compute_similarity_matrix()`, `find_similar()`
- Returns zero vectors on error (safe fallback — never crashes the pipeline)
- **Why embeddings specifically:** We need a way to compare articles numerically. Keyword matching fails when articles use synonyms or different phrasing. Embeddings capture *meaning*, not just words. Two articles about "startup funding winter" and "VC investment slowdown" will cluster together because their embeddings are similar, even though they share few keywords.

#### `app/tools/clustering.py` (365 lines) — NEW
- **ClusteringTool** class implementing 3-layer article clustering:
  - **Layer 1:** DBSCAN on embeddings (density-based spatial clustering)
  - **Layer 2:** Entity overlap merging (Jaccard similarity on named entities)
  - **Layer 3:** Keyword overlap merging (Jaccard similarity on keywords)
- Produces `ArticleCluster` objects with coherence scores
- Configurable via `CLUSTERING_EPS` and `CLUSTERING_MIN_SAMPLES` in config
- **Why 3 layers:** No single clustering method is perfect. DBSCAN catches semantically similar articles but may miss articles that mention the same companies/people in different contexts. Entity overlap catches those. Keyword overlap is the final safety net for domain-specific jargon that embeddings might not capture well.

#### `app/tools/smart_trend_detector.py` (369 lines) — NEW
- **SmartTrendDetector** class for LLM-native trend detection
- Strategy: deduplicate articles → send to LLM for semantic grouping and synthesis
- Supports up to 100 articles per LLM call
- Maps detected trends to sectors (via `SECTOR_KEYWORD_MAP`) and trend types (via `TREND_TYPE_MAP`)
- Generates confidence scores and severity classifications
- **Why LLM-native:** Embeddings and clustering are good at grouping, but they can't *understand* what a trend means for business. An LLM can read a cluster of articles about "India's PLI scheme for semiconductors" and reason that this is a manufacturing policy trend with positive impact on the electronics sector. Pure ML methods can't do this.

#### `app/tools/trend_synthesizer.py` (426 lines) — NEW
- **TrendSynthesizer** class that converts article clusters into structured `MajorTrend` objects
- Uses LLM to understand business impact per cluster
- Maps trends to CMI service recommendations
- Generates sector impact analysis, confidence scoring, and severity levels
- **Why separate from detection:** Detection answers "what trends exist?" Synthesis answers "what does this trend mean for business?" Separating these concerns means each prompt is focused, producing higher quality output than a single mega-prompt.

---

## 2. Multi-Source News Ingestion

### Why
The original `rss_tool.py` only fetched from Google News RSS (~1 source). This created three problems:
- **Single point of failure:** If Google News RSS went down or rate-limited us, the entire pipeline stopped.
- **Narrow coverage:** Google News aggregates general headlines. It misses niche sources like government press releases (PIB, RBI, SEBI), startup-focused outlets (YourStory, Inc42), and sector-specific feeds — all critical for India B2B intelligence.
- **No signal validation:** With 1 source, there's no way to cross-validate whether a trend is real or just one outlet's editorial angle. Multiple sources covering the same event = higher confidence signal.

### What Changed

#### `app/tools/rss_tool.py` — MODIFIED (165 → 626 lines)
- Expanded from 1 source to **20+ news sources** (RSS feeds + REST APIs)
- Sources include:
  - **Tier 1:** Economic Times, Mint, Business Standard, Moneycontrol, Financial Express
  - **Tier 2:** YourStory, Inc42, VCCircle, Entrackr (startup ecosystem)
  - **Government:** PIB, RBI, SEBI
  - **APIs:** NewsAPI.org, RapidAPI, MediaStack, GNews, NewsData.io
- Architecture changes:
  - Parallel fetching via `asyncio` (all sources fetched concurrently)
  - Article deduplication by content hash
  - Source health tracking (marks unhealthy sources, skips on repeated failures)
  - Returns `List[NewsArticle]` (rich schema) instead of `List[Dict]`
- Key new methods: `fetch_all_sources()`, `_fetch_source()`, `_fetch_rss_source()`, `_fetch_api_source()`, `_deduplicate_articles()`

---

## 3. Expanded Data Models

### Why
The original schemas were minimal — just enough for basic trend and company data. As the pipeline grew from "fetch articles → detect trends" to a multi-stage system, untyped dictionaries became a liability:
- **Bugs from typos:** A `dict` with `trend_type` in one place and `trendType` in another silently breaks. Pydantic models catch this at construction time.
- **LLM output validation:** When the LLM generates structured output (trend type, severity, sector), we need to validate it against known enums. Without this, the LLM might return "HIGH" when we expect "high", or invent a sector that doesn't exist.
- **Pipeline state tracking:** Each stage (ingestion → clustering → synthesis) needs to pass structured data to the next. State models (`NewsIngestionState`, `ClusteringState`, `TrendSynthesisState`) make the pipeline debuggable and resumable.

### What Changed

#### `app/schemas.py` — MODIFIED (~100 → 500+ lines)

**New Enums (9):**
| Enum | Purpose |
|------|---------|
| `Sector` | 12 industry sectors (IT, BFSI, Healthcare, Manufacturing, etc.) |
| `ServiceType` | 9 CMI core services (Procurement Intelligence, Market Intelligence, etc.) |
| `TrendType` | Trend classifications (regulation, policy, funding, merger, IPO, bankruptcy, etc.) |
| `ImpactType` | Direction of impact (positive, negative, mixed, neutral, disruptive) |
| `SourceType` | Data source types (RSS, API, scrape, government, social, manual) |
| `SourceTier` | Source credibility tiers (TIER_1 through TIER_4, UNKNOWN) |
| `IntentLevel` | Buying intent levels (hot, warm, cold, dormant) |
| `CompanySize` (expanded) | Added SMB, MID_MARKET, LARGE_ENTERPRISE |
| `Severity` (expanded) | Added NEGLIGIBLE level |

**New Value Objects (3):**
- `GeoLocation` — Country, state, city
- `MoneyAmount` — Amount with currency and estimated flag
- `ConfidenceScore` — Score (0–1) with factors and computed level

**New Data Models (7+):**
- `NewsSource` — Source configuration with health tracking
- `NewsArticle` — Raw article with embeddings, clustering metadata, sentiment
- `ArticleCluster` — Group of related articles with coherence score
- `MajorTrend` — Synthesized trend with impact analysis and service recommendations
- `SectorImpact` — How a trend impacts a specific industry sector
- `NewsIngestionState` — Pipeline state for ingestion phase
- `ClusteringState` / `TrendSynthesisState` — Pipeline state tracking

---

## 4. LLM Provider Refactoring

### Why
Three practical problems drove this:
- **Model access:** OpenRouter gives access to dozens of models (Claude, Llama, Mistral, etc.) through a single API key. Instead of being locked to Gemini or Groq's model catalog, we can now pick the best model for each task.
- **Async bottleneck:** The Groq client was synchronous. When the pipeline fetches 20+ sources in parallel and then needs LLM calls for clustering/synthesis, a sync client serializes everything. `AsyncGroq` lets LLM calls run concurrently with other I/O.
- **Debugging opacity:** When a trend summary was low quality, there was no way to know which LLM provider/model produced it. `last_provider` tracking makes it possible to correlate output quality with the model that generated it.

### What Changed

#### `app/tools/llm_tool.py` — MODIFIED (~150 lines changed)
- **New provider:** OpenRouter support (highest priority when configured)
- **Async Groq:** Changed from `Groq` to `AsyncGroq` client
- **Provider tracking:** New `last_provider` attribute on LLMTool — records which provider handled the last request
- **Refactored architecture:**
  - New `_configure_providers()` method (separates config from init)
  - New `_get_provider_chain()` method (builds fallback chain based on available config)
  - Provider priority: OpenRouter → Groq → Gemini → Ollama
- **Cleaner logging:** Removed emoji from log messages

---

## 5. Configuration Expansion

### Why
Supporting 20+ news sources and new ML tools requires extensive configuration — API keys, source URLs, embedding model settings, clustering parameters. Centralizing this in `config.py` keeps the rest of the codebase clean.

### What Changed

#### `app/config.py` — MODIFIED (~110 → 750+ lines)
- **OpenRouter config:** `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- **HuggingFace config:** `HF_API_TOKEN`, `HF_EMBEDDING_MODEL`
- **Clustering config:** `CLUSTERING_EPS`, `CLUSTERING_MIN_SAMPLES`
- **NEWS_SOURCES dictionary:** 20+ sources with full configuration:
  - URL, source type, tier, category tags
  - API-specific settings (headers, params, response paths)
- **DEFAULT_ACTIVE_SOURCES:** Pre-configured active sources list
- **EXTENDED_API_SOURCES:** Optional additional API sources

---

## 6. Agent Cleanup

### Why
The old `TrendAgent` was replaced by the new tool-based pipeline (`SmartTrendDetector` + `TrendSynthesizer`). Legacy helper methods in `CompanyAgent` were dead code. Constants in `ImpactAgent` were defined inline instead of at module level.

### What Changed

#### `app/agents/__init__.py` — MODIFIED
- Removed `TrendAgent` import and export

#### `app/agents/company_agent.py` — MODIFIED (~78 lines)
- Removed `_get_mock_companies()` (41 lines of dead code)
- Removed `_extract_companies_from_result()` (68 lines of dead code)
- Moved `_STOPWORDS` to class-level `frozenset` (was recreated on every call)
- Simplified `_find_company_domain()` (removed unnecessary intermediate variable)

#### `app/agents/impact_agent.py` — MODIFIED (~50 lines)
- Moved keyword-to-trend-type mapping to module-level constant `KEYWORD_TO_TREND_TYPE`
- Improved `_get_roles_from_keywords()` efficiency
- Removed stale "Legacy fields" comments

---

## 7. Streamlit UI Updates

### Why
The UI needed to support the new pipeline stages, display which LLM provider handled each request, and prevent XSS vulnerabilities from raw news content being rendered as HTML.

### What Changed

#### `streamlit_app.py` — MODIFIED (~600+ lines changed)
- **Security:** Added `escape_for_html()` function — all news content is escaped before rendering
- **Provider transparency:** Each pipeline step now shows which LLM provider was used
- **New helper functions:**
  - `_ensure_model()` — Safely convert dict to Pydantic model
  - `_rebuild_list()` — Round-trip validation through model_dump
  - `format_reasoning_text()` — Format reasoning output with bullet points
- **Pipeline state:** New session state keys for `articles`, `clusters`, `major_trends`, `active_sources`
- **Default mode change:** `mock_mode` default changed from `True` to `False` (real data by default)
- **Sidebar:** Added provider status display and news source information
- **Renderer refactor:** Replaced if/elif chain with `step_renderers` dictionary

---

## 8. Minor Changes

#### `.gitignore` — MODIFIED
- Added `.claude/` (Claude AI workspace files)
- Added `.playwright-mcp` (Playwright MCP integration files)

#### `app/tools/__init__.py` — MODIFIED
- Added exports for all 4 new tools: `EmbeddingTool`, `ClusteringTool`, `SmartTrendDetector`, `TrendSynthesizer`
- Added convenience function exports: `embed()`, `embed_batch()`, `cosine_similarity()`, `cluster_articles()`, `detect_trends_smart()`, `synthesize_trends()`

---

## New Pipeline Architecture

```
Raw News Sources (20+)
        │
        ▼
┌─────────────────┐
│  RSS Tool        │  Parallel fetch + deduplication
│  (Ingestion)     │  → List[NewsArticle]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding Tool  │  HuggingFace / Ollama
│  (Vectorization) │  → Embeddings per article
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clustering Tool │  DBSCAN + entity + keyword overlap
│  (Grouping)      │  → List[ArticleCluster]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Smart Trend     │  LLM-native trend detection
│  Detector        │  → Grouped trends
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Trend           │  LLM synthesis + impact mapping
│  Synthesizer     │  → List[MajorTrend]
└────────┬────────┘
         │
         ▼
  Existing Pipeline
  (Impact → Companies → Contacts → Emails)
```

---

## Summary Table

| Category | Files | Lines Changed | Key Reason |
|----------|-------|---------------|------------|
| New ML Pipeline | 4 new | +1,408 | Semantic article processing |
| News Ingestion | 1 modified | +461 | Multi-source coverage |
| Data Models | 1 modified | +400 | Type-safe pipeline data |
| LLM Providers | 1 modified | +150 | OpenRouter + async + tracking |
| Configuration | 1 modified | +640 | Source and ML tool settings |
| Agent Cleanup | 3 modified | -120 | Remove dead code |
| Streamlit UI | 1 modified | +600 | Security + new pipeline UI |
| Misc | 2 modified | +10 | Gitignore + tool exports |

---

## Future Iterations — Toward a Full Agentic Workflow

The current pipeline is **linear and tool-based**: each stage runs in sequence, and the orchestration logic lives in the Streamlit app. This is a solid foundation, but the architecture was designed to evolve into a proper **agentic system** where autonomous agents make decisions, retry on failure, and optimize themselves. Here's the roadmap:

### Iteration 1: LangGraph State Machine (Near-term)

**Current state:** The pipeline stages (ingest → embed → cluster → detect → synthesize) are called sequentially by `streamlit_app.py`. If clustering produces poor results, there's no way to automatically retry with different parameters.

**Target:** Wrap the pipeline in a **LangGraph StateGraph** where each stage is a node, and edges carry conditional logic:

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
Ingest → Embed → Cluster ──→ Evaluate ──→ Re-cluster
                                │          (adjust EPS)
                                │
                                ▼
                        Detect Trends → Synthesize → Output
```

- The **Evaluate** node checks cluster coherence scores. If clusters are too noisy (low coherence), it routes back to re-cluster with tighter DBSCAN `eps` — an automatic feedback loop.
- The state models (`NewsIngestionState`, `ClusteringState`, `TrendSynthesisState`) already exist in `schemas.py` — they were built for this. LangGraph's `TypedDict` state can wrap them directly.
- **Why this matters:** The system becomes self-correcting. Bad clustering doesn't silently produce bad trends; it triggers a retry.

### Iteration 2: Autonomous Agent Roles (Mid-term)

**Current state:** All "intelligence" is in the tools. The tools do computation but don't make decisions about *what* to compute next.

**Target:** Introduce specialized agents (not just tools) that can reason about their task:

- **Ingestion Agent** — Decides which sources to query based on the topic. If the user cares about fintech, it prioritizes SEBI, RBI, and Moneycontrol over general news. It can also decide to *re-fetch* from a specific source if initial results are thin.
- **Analysis Agent** — Looks at clusters and decides: "These 3 clusters are actually about the same macro-trend (India's semiconductor push) at different levels — let me merge them before synthesis." Current clustering is purely algorithmic; an agent can apply business reasoning.
- **Quality Agent** — Reviews synthesized trends and asks: "Is this actionable? Is the confidence score justified by the evidence?" Can flag low-quality trends for re-synthesis or discard them entirely.

**How embeddings enable this:** Each agent can use the embedding layer as a shared "memory" — the Analysis Agent can compare a new article's embedding against existing cluster centroids to decide if it belongs to a known trend or signals something new. This is the foundation of **continuous learning**: new articles don't restart the pipeline, they incrementally update the embedding space.

### Iteration 3: Multi-Agent Coordination with Feedback Loops (Long-term)

**Target:** A full agentic workflow where agents collaborate:

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator Agent                 │
│  (Decides pipeline strategy, allocates resources)    │
└──────┬──────────────┬───────────────┬───────────────┘
       │              │               │
       ▼              ▼               ▼
  Ingestion      Analysis        Outreach
  Agent Fleet    Agent Fleet     Agent Fleet
  (parallel      (parallel       (parallel
   source         cluster         company
   fetching)      analysis)       targeting)
       │              │               │
       └──────┬───────┘               │
              ▼                       │
        Shared Embedding              │
        Vector Store                  │
        (persistent memory)           │
              │                       │
              └───────────────────────┘
                        │
                        ▼
                  Feedback Loop:
                  - Track which trends led to successful outreach
                  - Weight future source selection based on ROI
                  - Adjust clustering parameters based on hit rate
```

Key capabilities in this iteration:

- **Persistent vector store:** Embeddings are stored in a database (e.g., ChromaDB, Pinecone) instead of computed fresh each run. The system remembers past articles and can detect *new* trends by comparing against historical embeddings. "This cluster is new — it wasn't present yesterday" becomes a first-class signal.
- **Agent-to-agent communication:** The Outreach Agent can tell the Ingestion Agent: "Trends about supply chain disruptions led to 3 meetings last week — prioritize supply chain sources." This closes the loop from intelligence to action.
- **Genetic/evolutionary optimization:** Clustering parameters (`eps`, `min_samples`), source weights, and even LLM prompt templates can be treated as a "genome." Run multiple pipeline variants in parallel, score them by downstream success (meetings booked, replies received), and evolve the best-performing configurations. The embedding layer makes this feasible because you can cheaply re-cluster the same embeddings with different parameters without re-fetching or re-embedding.
- **Human-in-the-loop checkpoints:** Agents propose actions (e.g., "I want to send outreach about this semiconductor trend to 15 companies"), and a human approves or adjusts before execution. The system learns from these adjustments.

### What's Already in Place for Agentic Evolution

| Current Component | Agentic Role It Enables |
|---|---|
| `EmbeddingTool` | Shared semantic memory across agents |
| `ClusteringTool` with configurable params | Evolutionary parameter tuning |
| `SmartTrendDetector` (LLM-native) | Analysis Agent's reasoning engine |
| `TrendSynthesizer` (structured output) | Typed contracts between agents |
| `schemas.py` state models | LangGraph state machine nodes |
| `rss_tool.py` source health tracking | Ingestion Agent's source selection |
| `llm_tool.py` provider chain | Agent-level model selection per task |
| `config.py` source tiers | Signal quality weighting for agents |

### Key Design Decisions That Enable Future Iteration

1. **Tools, not agents, for computation.** The current pipeline uses tools (stateless, composable) rather than monolithic agents. This means a future orchestrator agent can mix and match tools freely — call `embed_batch()` followed by `cluster_articles()` with custom parameters, without being locked into a rigid agent flow.

2. **Typed state at every boundary.** `NewsArticle`, `ArticleCluster`, `MajorTrend` are Pydantic models with validation. Any future agent that produces or consumes these types gets compile-time-like safety. This is critical when agents are autonomous — you need contracts, not just conventions.

3. **Embeddings as the universal interface.** Every article gets an embedding. This means any future component — a recommendation engine, a duplicate detector, a novelty scorer — can plug in by operating on the same vector space. The embedding layer is the lingua franca of the system.

4. **Source-aware architecture.** Articles carry their source tier, type, and health metadata. A future quality agent can weight a TIER_1 source (Economic Times) higher than a TIER_3 source (random blog) when scoring trend confidence. This metadata was added now precisely to enable that later.
