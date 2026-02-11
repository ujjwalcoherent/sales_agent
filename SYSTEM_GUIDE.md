# India Trend Lead Generation Agent - System Guide

**Comprehensive developer reference for the CMI Sales Agent pipeline.**

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture Overview](#architecture-overview)
3. [The 5-Step User Workflow](#the-5-step-user-workflow)
4. [Step 0: Trend Detection (12-Phase ML Pipeline)](#step-0-trend-detection-the-ml-pipeline)
5. [Step 1: Impact Analysis](#step-1-impact-analysis)
6. [Step 2: Company Discovery](#step-2-company-discovery)
7. [Step 3: Contact Finding](#step-3-contact-finding)
8. [Step 4: Email Generation + Lead Scoring](#step-4-email-generation--lead-scoring)
9. [LLM Provider Chain](#llm-provider-chain)
10. [Embedding Strategy](#embedding-strategy)
11. [Anti-Hallucination Validation Layers](#anti-hallucination-validation-layers)
12. [Configuration Reference](#configuration-reference)
13. [File Structure](#file-structure)
14. [Key Design Decisions](#key-design-decisions)
15. [UI Architecture](#ui-architecture)
16. [Testing](#testing)
17. [Typical Pipeline Metrics](#typical-pipeline-run-metrics)
18. [Troubleshooting](#troubleshooting)

---

## What This System Does

This is an AI-powered sales intelligence pipeline for **Coherent Market Insights (CMI)** that:

1. Reads Indian business news from **22+ sources** (Economic Times, Mint, RBI, SEBI, etc.)
2. Detects market trends using a **12-phase ML pipeline** (NER, embedding, UMAP, HDBSCAN clustering, LLM synthesis)
3. Analyzes business impact using **multi-provider LLM reasoning** with auto-failover
4. Finds target **mid-size companies** affected by each trend (with NER verification)
5. Finds **decision-maker contacts** (CEO, CTO, VP) at those companies
6. Generates **personalized outreach emails** with consulting pitches and lead scores (0-100)

The end result: a ranked list of leads with scores, ready for outreach. Exportable as JSON or CSV.

---

## Architecture Overview

```
Raw News Sources (22+)
        |
        v
+------------------------+
|  RecursiveTrendEngine  |   12-phase ML pipeline
|  (app/trends/engine.py)|   RSS + API -> NER -> Embed -> Cluster -> Synthesize
+----------+-------------+
           |
           v
     List[TrendNode]  (hierarchical tree with MAJOR/SUB/MICRO)
           |
           v
+-------------------------------------------+
|        LANGGRAPH PIPELINE                 |
|  (app/agents/orchestrator.py)             |
|                                           |
|  Impact Agent -> Company Agent -> Contact |
|  (LLM analysis)  (Tavily+LLM)    Agent   |
|                                  (Apollo) |
|                                     |     |
|                               Email Agent |
|                            (Apollo+Hunter) |
|                                     |     |
|                              JSON + CSV   |
+-------------------------------------------+
```

### Core Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit + React Flow | 5-step wizard with graph visualization |
| **Pipeline** | LangGraph | Agent orchestration with state management |
| **ML** | UMAP + HDBSCAN + spaCy | Clustering, NER, dimensionality reduction |
| **Embeddings** | SentenceTransformers (local) | 384-dim article vectorization |
| **Dedup** | MinHash LSH + cosine similarity | Multi-layer duplicate detection |
| **LLM** | NVIDIA/Ollama/OpenRouter/Gemini/Groq | Multi-provider with auto-failover |
| **Search** | Tavily API | Company discovery |
| **Contacts** | Apollo.io + Hunter.io | Email and contact finding |
| **Backend** | FastAPI (optional) | REST API alternative |
| **Config** | Pydantic Settings + .env | Type-safe environment configuration |

---

## The 5-Step User Workflow

The user interacts through a wizard-style Streamlit dashboard:

```
Step 0: News Detection  ->  Step 1: Impact Analysis  ->  Step 2: Company Discovery
     |                            |                              |
     v                            v                              v
  "Detect trends"           "Analyze impact"             "Find companies"
  from live news             for consulting               matching trends


Step 3: Contact Finding  ->  Step 4: Email Generation
     |                            |
     v                            v
  "Find decision-makers"    "Generate outreach"
  at target companies        with lead scores
```

Each step requires **user approval** before proceeding (human-in-the-loop). Users can select/deselect individual trends, companies, and contacts at each stage.

---

## Step 0: Trend Detection (The ML Pipeline)

This is the most complex part. When the user clicks "Detect," the `RecursiveTrendEngine` (`app/trends/engine.py`) runs a 12-phase pipeline:

### Phase 0: News Fetching (Parallel)

**What:** Fetches articles from 22+ configured sources simultaneously.

**Sources:**
- **RSS Feeds (16):**
  - *Tier 1 (highest credibility):* Economic Times, ET Industry, ET Tech, Mint, Mint Companies, Business Standard, Moneycontrol, Financial Express, PIB, RBI
  - *Tier 2:* YourStory, Inc42, NDTV Profit, Hindu Business Line
  - *Tier 3:* Google News India (Business + Tech topics)
- **APIs (6):** NewsAPI.org, MediaStack, TheNewsAPI, GNews, GDELT (India Business + India Tech)

**How:** Uses `asyncio` for parallel I/O. RSS feeds parsed with `feedparser`, APIs called with `httpx`. Each article becomes a `NewsArticle` object with title, summary, source metadata, and credibility score.

**Optional source audit:** User can enable "Audit sources first" to check health/availability before fetching. Reports healthy, broken, empty, slow, and no-key sources.

**Output:** ~100-500+ raw articles depending on `per_source` setting.

### Phase 0.5: Event Classification (Embedding-Based)

**What:** Classifies each article's business event type using embedding similarity.

**How:** Pre-defined event prototypes (15 types: regulation, policy, funding, expansion, technology, acquisition, IPO, layoffs, crisis, consumer_shift, partnership, supply_chain, leadership_change, market_entry, price_change) are embedded once. Each article's title+summary embedding is compared against these prototypes via cosine similarity. Best-matching event type is assigned.

**Why not LLM:** Runs on 200+ articles. LLM calls would be too slow/expensive. Embedding similarity is instant and accurate enough.

**Output:** Each article gets `_trigger_event` (e.g., "regulation") and `_trigger_confidence`.

### Phase 0.5 (parallel): Content Scraping

**What:** Fetches full article content from URLs (RSS only gives title + summary).

**How:** Uses `trafilatura` for clean text extraction. Runs as an async task overlapping with event classification.

### Phase 0.7: Business Relevance Filter

**What:** Removes non-business articles (sports, entertainment, celebrity).

**How:** Articles matching any business event type pass. Articles with zero event similarity get filtered. Borderline cases (confidence >= 0.15) are kept.

### Phase 1: Deduplication (MinHash LSH)

**What:** Removes near-duplicate articles from different sources covering the same story.

**How:** MinHash with LSH. Converts articles to bigram shingle sets, computes 128-function signatures. Articles with Jaccard > 0.25 are grouped; only the highest-tier source version is kept.

**Why MinHash:** Same story from ET and Mint will have different wording. MinHash catches structural similarity.

**Typical removal:** 20-40% of articles.

### Phase 2: Named Entity Recognition (spaCy NER)

**What:** Extracts people, organizations, locations, dates, monetary values.

**How:** spaCy `en_core_web_sm` with batch processing (`nlp.pipe()`). Processes title + summary.

**Entity types:** PERSON, ORG, GPE, NORP, MONEY, LAW, EVENT, LOC

Each entity gets a **salience score** (0-1) based on position (inverted pyramid principle).

**Output:** `entities`, `entity_names`, `mentioned_companies`, `mentioned_locations` per article.

### Phase 2.3: Geographic Relevance Filter

**What:** Filters articles about irrelevant foreign countries using NER entities.

**How:** Dynamic, entity-based (NOT hardcoded keywords):
1. Checks GPE/NORP/LOC entities for configured country name
2. Domestic source articles auto-pass
3. No-geo articles pass (benefit of the doubt)
4. Foreign-only articles get filtered

**Portability:** Changing `COUNTRY=Brazil` in `.env` makes filter work for Brazil without code changes.

### Phase 2.5: Entity Co-occurrence

**What:** Builds entity graph across articles. Used for bridge entity detection and cluster quality signals.

### Phase 2.7: Sentiment Analysis (VADER)

**What:** Pre-computes per-article sentiment. Uses VADER on title+summary. Populates `article.sentiment_score` for downstream signal computation.

### Phase 3: Embedding (Local SentenceTransformers)

**What:** Converts each article title into a 384-dimensional vector.

**How:** `paraphrase-multilingual-MiniLM-L12-v2` running locally (no API calls):
- 384 dimensions (compact, fast)
- Multilingual (handles Hindi-English mixed text)
- Auto-detects CUDA GPU for acceleration (~1.6x faster on NVIDIA MX550, 2GB VRAM)
- Falls back to CPU if no GPU available (~5s for 200 articles on CPU)

**Dimension Locking:** Once first embedding is produced, dimension (384) is locked. Fallback providers producing different dimensions are rejected (prevents silent mixing that crashes clustering).

### Phase 3.5: Semantic Deduplication

**What:** Catches cross-source duplicates MinHash missed (same story, very different wording).

**How:** Pairwise cosine similarity. Pairs > 0.78 are duplicates. Union-Find for transitive grouping.

**Safety gates:**
1. **Degenerate embedding check:** If random sample has avg similarity > 0.95, embeddings are broken — skip dedup entirely
2. **Removal cap:** If > 50% would be removed, threshold raised by 0.07 and re-run

### Phase 4: Dimensionality Reduction (UMAP)

**What:** Reduces 384-dim to 5-dim for clustering.

**Parameters:** `n_components=5, n_neighbors=15, min_dist=0.0, metric=cosine`

### Phase 5: Clustering (HDBSCAN)

**What:** Groups articles into clusters by topic. Each cluster = potential trend.

**How:** HDBSCAN with adaptive `min_cluster_size` based on article count (targeting 15-20 major clusters). Uses `leaf` selection method. Automatically identifies noise articles (cluster = -1).

### Phase 5.5: Coherence Validation

**What:** Verifies clusters are semantically tight in ORIGINAL 384-dim space (not UMAP 5-dim).

**Operations:**
1. **Split:** Clusters below coherence threshold (0.35) get split via agglomerative clustering
2. **Merge:** Clusters with centroids > 0.70 similarity get merged
3. **Reject:** Too-small clusters become noise

### Phase 6: Keyword Extraction (TF-IDF)

**What:** Extracts representative keywords per cluster. Words frequent in THIS cluster but rare across others.

### Phase 7: Signal Computation

**What:** Computes quantitative signals per cluster for trend strength/actionability.

**Signal modules (`app/trends/signals/`):**

| Module | Signals | What it measures |
|--------|---------|-----------------|
| `temporal` | recency, velocity | How recent and how fast articles appear |
| `content` | specificity, sentiment | Topic focus and VADER sentiment |
| `entity` | entity_focus, person | Key entities and decision-makers |
| `source` | diversity, authority | Multi-source coverage and credibility |
| `market` | regulatory, trigger, financial | Regulation, business events, monetary mentions |
| `search_interest` | google_trends | Google Trends validation |
| `composite` | overall scoring | Weighted combination -> STRONG/WEAK/NOISE |

### Phase 7.1: Entity Graph (Bridge Entities)

Entities appearing in 2+ clusters are "bridge entities" connecting different trends.

### Phase 7.5: Search Interest (Google Trends)

Validates trends against Google Trends search data for the target country.

### Phase 7.7: Trend Memory (Historical Matching)

Compares current clusters against stored centroids from past runs (`data/trend_memory.json`). Matching clusters tagged as "continuing" vs "novel." Dimension safety: auto-reset if stored dimensions mismatch.

### Phase 8: LLM Synthesis

**What:** LLM generates human-readable trend summaries from cluster data.

**Output per cluster:**
- Trend title and summary
- 5W1H analysis (Who, What, When, Where, Why, How)
- Causal chain
- Buying intent assessment
- Affected companies and regions
- Severity and trend type classification

**Concurrency:** Up to 14 parallel LLM calls.

### Phase 9: Quality Gate (V9) + Tree Assembly

**Filters out:** Empty clusters, no-title clusters, low-confidence clusters.

**Tree Assembly:** Surviving clusters become `TrendNode` objects organized into a `TrendTree`. Large clusters (>10 articles) can be recursively sub-clustered to reveal sub-trends. Results in MAJOR/SUB/MICRO depth hierarchy.

### Phase 10: Trend Linking

Creates cross-references between related trends using shared bridge entities and embedding similarity. Visualized as edges in the React Flow graph.

---

## Step 1: Impact Analysis

**Agent:** `ImpactAgent` (`app/agents/impact_agent.py`)

**Analysis structure (6 parts):**
1. **Direct Impact** — Which industries are directly affected and how
2. **Indirect Impact** — Ripple effects on adjacent industries
3. **Pain Points** — Specific pain points for mid-size companies
4. **Consulting Projects** — Actionable consulting opportunities
5. **Additional Verticals** — Non-obvious affected industries
6. **Relevant Services** — CMI services that map to the opportunity

**Compound impact synthesis:** When multiple trends are analyzed, the system identifies compound effects where trends amplify each other's impact on the same company type.

**UI:** Two-column layout with color-coded impact sections (Direct=red, Indirect=orange, Pain=pink, Consulting=cyan, Verticals=purple, Services=green). Each trend gets an expandable card with pitch angle and detailed reasoning.

---

## Step 2: Company Discovery

**Agent:** `CompanyAgent` (`app/agents/company_agent.py`)

**Search strategy (via Tavily API):**
1. **Intent-based queries:** Companies "struggling" or "expanding" in affected sectors
2. **Pain-point queries:** Based on identified pain points from Step 1
3. **Trend-specific queries:** Companies mentioned in trend context
4. **Sector queries:** Mid-size companies (50-300 employees) in affected sectors

**V7: NER-based company hallucination guard:**
Each LLM-returned company is cross-validated:
1. **NER match (confidence 0.9):** Fuzzy-matches NER entities from source articles (bidirectional substring matching with suffix normalization — removes "Pvt Ltd", "Limited", etc.)
2. **Wikipedia fallback (confidence 0.6):** If not in NER entities, checks Wikipedia search API
3. **Unverified (confidence 0.2):** Potentially hallucinated

Companies below `COMPANY_MIN_VERIFICATION_CONFIDENCE` threshold are filtered out.

**Large company filter:** Tata, Reliance, Infosys, Wipro, HCL, HDFC, ICICI, Bajaj, Mahindra, Adani, Vedanta are excluded (focus on mid-size).

**UI:** Company cards with NER verification badge (green checkmark or orange "Unverified"), industry badge, size badge, website link, and description.

---

## Step 3: Contact Finding

**Agent:** `ContactAgent` (`app/agents/contact_agent.py`)

**Strategy:**
1. **Apollo API** (primary) — Searches for contacts by company and role
2. **LLM extraction** (fallback) — Extracts contacts from web search results via Tavily
3. **Role matching** — Uses `TREND_ROLE_MAPPING` from config to target roles based on trend type (regulation trend -> CEO/CSO, technology trend -> CTO/VP Engineering)

**Output:** Person name, role, LinkedIn URL, email, email confidence score, email source.

**UI:** Contact cards with name, role, company, email with confidence color (green >= 80%, orange >= 50%, red < 50%), LinkedIn link, and source badge. HTML rendered via string concatenation (no indentation) to prevent Streamlit markdown parser from treating divs as code blocks.

---

## Step 4: Email Generation + Lead Scoring

**Agent:** `EmailAgent` (`app/agents/email_agent.py`)

**Email finding cascade:** Apollo -> Hunter -> Pattern-based generation

**Email structure:**
- Hook referencing the specific trend
- Pain point the company likely faces
- CMI service that addresses it
- Call to action

**Lead Scoring (0-100):**

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Trend severity | 25% | How urgent is the market need |
| Company fit | 25% | Mid-size + clear intent signal |
| Email confidence | 25% | Can we actually reach them |
| Impact depth | 25% | Strength of the consulting angle |

Leads sorted by score and color-coded: Green (60+), Orange (40-59), Red (<40).

**Export:** JSON and CSV download buttons on the final step.

---

## LLM Provider Chain

**Priority order:** NVIDIA -> Ollama (local) -> OpenRouter -> Gemini -> Groq

**Implementation:** `LLMTool` (`app/tools/llm_tool.py`)

| Provider | Model | Notes |
|----------|-------|-------|
| NVIDIA | `moonshotai/kimi-k2.5` | NIM API, highest priority |
| Ollama | Configurable (default: `mistral`) | Local, unlimited, no API key needed |
| OpenRouter | Configurable (default: `google/gemini-2.0-flash-001`) | Multi-model gateway |
| Gemini | Configurable (default: `gemini-2.0-flash`) | Google AI |
| Groq | Configurable (default: `openai/gpt-oss-120b`) | Fast inference |

**Failover behavior:**
- Each provider has a **5-minute cooldown** after failure
- Error messages containing "402" or "429" trigger provider cooldown
- Status codes and error text are checked explicitly (not via `raise_for_status()`)
- Groq uses `max_tokens` parameter (not `max_completion_tokens`)
- Provider status shown in sidebar with green checkmarks

**Methods:**
- `generate_text(prompt, max_tokens)` — Raw text generation
- `generate_json(prompt, max_tokens)` — JSON-parsed output
- `generate_list(prompt)` — List of dictionaries

---

## Embedding Strategy

**3-tier fallback:** Local SentenceTransformers -> Ollama -> HuggingFace API

**Implementation:** `EmbeddingTool` (`app/tools/embeddings.py`)

| Tier | Model | Dimensions | Notes |
|------|-------|------------|-------|
| 1 (Local) | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Default, no API needed, CUDA GPU auto-detected |
| 2 (Ollama) | `nomic-embed-text:latest` | 768 | Local server |
| 3 (HuggingFace) | Configurable | Varies | API fallback |

**Dimension locking:** Once the first embedding is produced, that dimension is locked for the entire pipeline run. Fallback providers producing different dimensions are rejected to prevent silent dimension mixing (which would crash clustering).

---

## Anti-Hallucination Validation Layers

| Layer | What it validates | Location |
|-------|------------------|----------|
| **V2** | LLM JSON output structure and required fields | `config.py` |
| **V3** | Synthesis quality — titles, summaries, severity | `config.py`, `trend_synthesizer.py` |
| **V6** | Event classification accuracy via embedding similarity | `event_classifier.py` |
| **V7** | Company names vs NER entities + Wikipedia | `company_agent.py` |
| **V9** | Quality gate — drops empty/no-title/low-confidence clusters | `engine.py` |
| **V10** | Cross-validation synthesis — retry with score thresholds | `config.py` |
| **Pydantic** | Field-level validators on all schema models | `schemas/*.py` |
| **URL/Domain** | URL format and domain extraction validation | `domain_utils.py` |
| **Dimension Lock** | Embedding dimension consistency enforcement | `embeddings.py` |
| **Dedup Safety** | Degenerate embedding detection + removal caps | `engine.py` |

---

## Configuration Reference

All settings loaded from `.env` via `pydantic-settings` `BaseSettings` with `Field(alias=...)`.

### LLM Providers

| Setting | Default | Purpose |
|---------|---------|---------|
| `NVIDIA_API_KEY` | - | NVIDIA NIM LLM (highest priority) |
| `NVIDIA_MODEL` | `moonshotai/kimi-k2.5` | NVIDIA model |
| `USE_OLLAMA` | `true` | Enable Ollama local LLM |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OPENROUTER_API_KEY` | - | OpenRouter multi-model API |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-001` | OpenRouter model |
| `GEMINI_API_KEY` | - | Google Gemini |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `GROQ_API_KEY` | - | Groq fast inference |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model |

### Embeddings

| Setting | Default | Purpose |
|---------|---------|---------|
| `LOCAL_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local SentenceTransformers model |
| `HF_API_KEY` | - | HuggingFace API (tier 3 fallback) |

### News Sources

| Setting | Default | Purpose |
|---------|---------|---------|
| `NEWSAPI_KEY` | - | NewsAPI.org |
| `GNEWS_API_KEY` | - | GNews API |
| `MEDIASTACK_KEY` | - | MediaStack API |
| `THENEWSAPI_KEY` | - | TheNewsAPI |

### Sales Pipeline

| Setting | Default | Purpose |
|---------|---------|---------|
| `TAVILY_API_KEY` | - | Tavily company search |
| `APOLLO_API_KEY` | - | Apollo.io contacts + emails |
| `HUNTER_API_KEY` | - | Hunter.io email (fallback) |

### Pipeline Tuning

| Setting | Default | Purpose |
|---------|---------|---------|
| `COUNTRY` | `India` | Target market (geo filter) |
| `COUNTRY_CODE` | `IN` | ISO code for source matching |
| `MAX_TRENDS` | `3` | Max trends to process |
| `MAX_COMPANIES_PER_TREND` | `3` | Companies per trend |
| `MAX_CONTACTS_PER_COMPANY` | `2` | Contacts per company |
| `EMAIL_CONFIDENCE_THRESHOLD` | `70` | Min email confidence (0-100) |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.78` | Embedding dedup sensitivity |
| `COMPANY_MIN_VERIFICATION_CONFIDENCE` | `0.5` | V7 verification threshold |
| `MOCK_MODE` | `false` | Use mock data (no API calls) |

---

## File Structure

```
sales_agent/
├── streamlit_app.py              # Main UI: 5-step wizard (800+ lines)
├── requirements.txt              # All Python dependencies
├── .env                          # API keys + configuration (gitignored)
├── .gitignore                    # Covers .env, *.png, node_modules, data/, etc.
├── SYSTEM_GUIDE.md               # This file
├── README.md                     # Quick start + overview
│
├── app/
│   ├── __init__.py
│   ├── config.py                 # Settings (pydantic-settings), sources, services, thresholds
│   ├── database.py               # SQLAlchemy models (SQLite)
│   ├── main.py                   # FastAPI entry point (alternative to Streamlit)
│   │
│   ├── agents/
│   │   ├── company_agent.py      # Tavily search + V7 NER verification
│   │   ├── contact_agent.py      # Apollo + Tavily + LLM contact discovery
│   │   ├── email_agent.py        # Apollo + Hunter email finding + LLM pitch writing
│   │   ├── impact_agent.py       # LLM impact analysis with compound synthesis
│   │   ├── orchestrator.py       # LangGraph pipeline orchestration
│   │   ├── trend_agent.py        # Trend agent wrapper
│   │   └── validator_agent.py    # Quality validation agent
│   │
│   ├── news/
│   │   ├── dedup.py              # MinHash LSH deduplication
│   │   ├── entity_extractor.py   # spaCy NER batch processing
│   │   ├── entity_cooccurrence.py # Entity co-occurrence graph
│   │   ├── event_classifier.py   # Embedding-based event classification (15 types)
│   │   └── scraper.py            # Full article extraction (trafilatura)
│   │
│   ├── schemas/
│   │   ├── __init__.py           # Re-exports all schemas
│   │   ├── base.py               # Enums: Severity, SignalStrength, CompanySize, etc.
│   │   ├── news.py               # NewsArticle, Entity, ArticleCluster
│   │   ├── trends.py             # TrendNode, TrendTree, MajorTrend
│   │   ├── sales.py              # LeadRecord, CompanyData, ContactData, OutreachEmail
│   │   ├── pipeline.py           # AgentState (LangGraph state model)
│   │   └── validation.py         # Validation result schemas
│   │
│   ├── shared/
│   │   ├── helpers.py            # _ensure_model, escape_for_html (XSS safe), truncate_text
│   │   ├── sidebar.py            # Streamlit sidebar (settings, provider status, stats)
│   │   ├── styles.py             # CSS: step bar, cards, badges, detail sidebar, theme
│   │   └── visualizations.py     # React Flow tree graph, list view
│   │
│   ├── tools/
│   │   ├── apollo_tool.py        # Apollo.io API (contacts + emails)
│   │   ├── hunter_tool.py        # Hunter.io email finder (fallback)
│   │   ├── domain_utils.py       # Domain extraction + validation
│   │   ├── embeddings.py         # 3-tier embedding: Local -> Ollama -> HuggingFace
│   │   ├── llm_tool.py           # Multi-provider LLM: NVIDIA -> Ollama -> OpenRouter -> Gemini -> Groq
│   │   ├── rss_tool.py           # 22+ source parallel news fetcher with audit
│   │   ├── tavily_tool.py        # Tavily search API wrapper
│   │   └── trend_synthesizer.py  # LLM trend summary generation
│   │
│   └── trends/
│       ├── engine.py             # RecursiveTrendEngine: 12-phase pipeline
│       ├── coherence.py          # Post-clustering coherence validation
│       ├── keywords.py           # TF-IDF keyword extraction
│       ├── reduction.py          # UMAP dimensionality reduction
│       ├── synthesis.py          # Concurrent LLM synthesis orchestration
│       ├── subclustering.py      # Recursive sub-trend detection
│       ├── tree_builder.py       # TrendTree assembly + trend linking
│       ├── trend_memory.py       # Historical trend persistence (JSON)
│       └── signals/
│           ├── __init__.py       # Signal registry
│           ├── composite.py      # Weighted combination -> STRONG/WEAK/NOISE
│           ├── content.py        # Specificity, sentiment (VADER)
│           ├── entity.py         # Entity focus, person detection, bridge entities
│           ├── market.py         # Regulatory, trigger, financial signals
│           ├── search_interest.py # Google Trends validation
│           ├── source.py         # Source diversity + authority scoring
│           └── temporal.py       # Recency + velocity computation
│
├── test_validation_layers.py     # Validation layer tests (31 tests)
├── test_pipeline.py              # Pipeline diagnostic tests (6 tests)
├── test_llm_providers.py         # LLM provider connectivity tests
├── test_stress.py                # Stress test suite
│
├── data/                         # Runtime data (trend memory, etc.)
├── screenshots/                  # Playwright MCP screenshots
└── node_modules/                 # Playwright dependencies
```

---

## Key Design Decisions

1. **Local-first embeddings:** Uses local SentenceTransformers (no API calls) for the compute-heavy embedding phase. Produces 384-dim embeddings in ~5s for 200 articles. Fallback providers rejected if they produce different dimensions.

2. **Entity-based filtering over keywords:** Geographic relevance uses spaCy NER entities, not hardcoded keyword lists. System is portable to any country by changing one `.env` value.

3. **Multiple dedup layers:** MinHash LSH (lexical) + semantic embedding (cross-lingual) + title-based exact match. Together catch ~40-50% duplicates.

4. **Coherence validation in original space:** UMAP compression can create false groupings. Coherence validator checks cluster quality in original 384-dim space and splits/merges as needed.

5. **Safety gates everywhere:** Embedding degeneration detection, dimension locking, dedup removal caps, cluster size floors, V9 quality gate. Pipeline degrades gracefully rather than silently producing bad output.

6. **NER-based company verification (V7):** Every LLM-returned company is cross-checked against NER entities from actual news articles, with Wikipedia fallback. Prevents the most common LLM hallucination: inventing company names.

7. **HTML via string concatenation:** All `st.markdown()` HTML is built with string concatenation (`'<div>' + ...`) instead of indented f-string templates. This prevents Streamlit's markdown parser from treating indented HTML as code blocks (showing raw `<div>` tags).

8. **Controls disabled during execution:** All input controls are disabled while the pipeline runs, preventing state corruption from user interaction during async operations.

9. **Modular signal computation:** Trend signals split into 7 modules for maintainability and testability.

10. **Multi-provider LLM with cooldowns:** Each provider gets a 5-minute cooldown after failure. Error strings are checked for HTTP status codes (402/429) rather than using `raise_for_status()`, ensuring the fallback logic correctly catches rate limits and quota errors.

---

## UI Architecture

### Step Indicator Bar
Top of page: numbered steps with `->` separators. Completed = green checkmark, active = cyan highlight, pending = grayed out.

### React Flow Graph (Step 0)
Interactive node graph showing trend hierarchy with MAJOR/SUB/MICRO nodes. Clicking a node opens the detail sidebar.

### Detail Sidebar
Fixed 550px overlay on right side showing:
- Trend title, summary, severity badge
- 5W1H analysis (filtered: "Not specified", "N/A" removed)
- Causal chain, buying intent
- Entities (deduplicated keywords + entities + companies)
- Signal scores grid
- Source articles (clickable, date + source name)
- Sub-topic navigation buttons

### Card Styles
- **Trend cards:** Blue-purple gradient, cyan left border
- **Company cards:** Green gradient, green left border, NER verification badge
- **Contact cards:** Purple gradient, magenta left border
- **Email cards:** Gold gradient, yellow left border, lead score

### Pipeline Log
Collapsed expander at bottom with timestamped phase-by-phase progress.

---

## Testing

### Test Suites

| Suite | File | Tests | What it validates |
|-------|------|-------|------------------|
| **Validation** | `test_validation_layers.py` | 31 | All V1-V10 validation layers, Pydantic schemas |
| **Pipeline** | `test_pipeline.py` | 6 | Config, embeddings, engine init, RSS, LLM |
| **LLM Providers** | `test_llm_providers.py` | 8 per provider | Text gen, JSON gen, list gen, error handling |
| **Stress** | `test_stress.py` | Varied | Full pipeline load testing |

### Running Tests

```bash
# All validation tests
python -m pytest test_validation_layers.py -v

# Pipeline diagnostics
python -m pytest test_pipeline.py -v

# LLM provider tests
python -m pytest test_llm_providers.py -v
```

---

## Typical Pipeline Run Metrics

| Metric | Typical Value |
|--------|--------------|
| Articles fetched | 100-500+ |
| After dedup (MinHash) | 70-85% retained |
| After semantic dedup | 95% retained |
| Clusters (HDBSCAN) | 8-20 |
| Noise articles | 10-30% |
| Major trends | 8-16 |
| Sub-trends | 2-5 per major |
| Pipeline time (Step 0) | 80-150 seconds |
| Impact analysis (Step 1) | 20-60 seconds |
| Company discovery (Step 2) | 5-20 minutes (Tavily + LLM + V7 verification) |
| Contact finding (Step 3) | 5-15 minutes (Apollo + LLM) |
| Email generation (Step 4) | 2-5 minutes |

---

## Troubleshooting

### Ollama not connecting
```bash
curl http://localhost:11434/api/tags    # Check if running
ollama serve                             # Start if needed
ollama pull mistral                      # Pull model
```

### No embeddings generated
- Local SentenceTransformers should work out of the box (no API key)
- Check `sentence-transformers` is installed
- Fallback chain: Local -> Ollama (`nomic-embed-text`) -> HuggingFace API

### Poor clustering results
- Check embedding quality: similarity distribution should show P50 ~0.4-0.5
- If all similarities are >0.95, embeddings are degenerate (model issue)
- Adjust `SEMANTIC_DEDUP_THRESHOLD` if too many/few articles removed

### Contact page showing raw HTML
- HTML in `st.markdown()` must NOT have indentation (4+ spaces = markdown code block)
- Use string concatenation: `'<div>' + content + '</div>'` instead of indented f-strings

### LLM provider errors
- Check API keys in `.env`
- NVIDIA: requires `NVIDIA_API_KEY`
- Groq: uses `max_tokens` (not `max_completion_tokens`)
- OpenRouter: check for 402 (payment required) or 429 (rate limit)
- All providers have 5-minute cooldown after failure

### V7 filtering too many companies
- Lower `COMPANY_MIN_VERIFICATION_CONFIDENCE` (default 0.5)
- Companies verified via NER get 0.9 confidence, Wikipedia gets 0.6, unverified gets 0.2

### Rate limits
| Service | Free Tier |
|---------|-----------|
| Tavily | 1,000/month |
| NewsAPI.org | 100/day |
| GNews | 100/day |
| MediaStack | 500/month |
| Apollo.io | 600/month |
| Hunter.io | 25/month |
| Gemini | 60 RPM |
| OpenRouter | Pay-per-use |
| Ollama | Unlimited (local) |
