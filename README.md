---
title: India Trend Lead Generation Agent
emoji: "\U0001F4C8"
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: "1.31.0"
app_file: streamlit_app.py
pinned: false
python_version: "3.11"
---

# India Trend Lead Generation Agent

AI-powered market trend detection and B2B lead generation for Indian mid-size companies. Ingests news from 22+ sources, clusters articles with a 12-phase ML pipeline, detects business trends via LLM, and generates targeted sales outreach with lead scoring.

## What It Does

```
153 articles from 25 sources
    -> 132 unique after dedup
    -> 16 major trends detected
    -> 14 impacts analyzed
    -> 36 target companies found
    -> Decision makers identified
    -> Personalized pitch emails generated
    -> Leads scored 0-100, exported as JSON/CSV
```

The system automatically:
1. **Fetches News** — 22+ sources (RSS feeds + APIs) in parallel
2. **Detects Trends** — 12-phase ML pipeline: NER, embeddings, UMAP, HDBSCAN, LLM synthesis
3. **Analyzes Impact** — Identifies which sectors win/lose, pain points, consulting opportunities
4. **Finds Companies** — Discovers real mid-size Indian companies via Tavily search + NER verification
5. **Locates Decision Makers** — Finds CTOs, CEOs, VPs via Apollo.io + web search
6. **Generates Outreach** — Personalized consulting pitch emails with lead scores

## Architecture

```
Raw News (22+ sources)
        |
        v
+---------------------------+
|  12-Phase Trend Engine    |   RSS/API -> Scrape -> Event Classify -> Dedup
|  (RecursiveTrendEngine)   |   -> NER -> Embed -> UMAP -> HDBSCAN
|                           |   -> Signals -> LLM Synthesis -> Tree
+----------+----------------+
           |
           v
+-------------------------------------------+
|          LANGGRAPH PIPELINE               |
|                                           |
|  Impact Agent -> Company Agent -> Contact |
|  (LLM)          (Tavily+NER)     (Apollo) |
|                                     |     |
|                               Email Agent |
|                            (Apollo+Hunter) |
+-------------------------------------------+
           |
           v
   Scored Leads (JSON/CSV)
```

### LLM Provider Chain (Auto-Failover)

```
NVIDIA (kimi-k2.5) -> Ollama (local) -> OpenRouter -> Gemini -> Groq
```

Each provider has a 5-minute cooldown on failure. First available provider is used.

### Embedding Strategy (3-Tier)

```
Local SentenceTransformers (384-dim) -> Ollama -> HuggingFace API
```

Local model auto-detects CUDA GPU for acceleration (1.6x faster on NVIDIA MX550). Falls back to CPU if no GPU. No API key needed. Dimension locking prevents silent mixing.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

Create `.env` in the project root:

```env
# LLM Providers (configure at least one)
NVIDIA_API_KEY=your_nvidia_key           # Highest priority
USE_OLLAMA=true                          # Local, unlimited
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=google/gemini-2.0-flash-001
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.0-flash
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b

# Embeddings (optional - local model works without API key)
HF_API_KEY=your_huggingface_token

# News APIs (more = better coverage)
NEWSAPI_KEY=your_newsapi_key
GNEWS_API_KEY=your_gnews_key
MEDIASTACK_KEY=your_mediastack_key
THENEWSAPI_KEY=your_thenewsapi_key

# Company Search
TAVILY_API_KEY=your_tavily_key

# Contact Finding
APOLLO_API_KEY=your_apollo_key
HUNTER_API_KEY=your_hunter_key

# Settings
COUNTRY=India
COUNTRY_CODE=IN
MAX_TRENDS=3
MAX_COMPANIES_PER_TREND=3
MAX_CONTACTS_PER_COMPANY=2
EMAIL_CONFIDENCE_THRESHOLD=70
MOCK_MODE=false
```

### 3. Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

This gives you:
- **5-step wizard** with human-in-the-loop review at each stage
- **React Flow graph** visualization of trend hierarchy
- **Provider transparency** — see which LLM/embedding provider handled each step
- **Real-time pipeline log** with timestamped phase progress
- **Manual selection** of trends, companies, and contacts
- **Export to JSON/CSV** with one click

### 4. Run via FastAPI (Optional)

```bash
# Start server
python -m app.main --server --port 8000

# Call the API
curl -X POST http://localhost:8000/run
```

## The 12-Phase Trend Detection Pipeline

| Phase | Name | What it does |
|-------|------|-------------|
| 0 | News Fetching | Parallel fetch from 22+ sources (RSS + APIs) |
| 0.5 | Event Classification | Embedding-based event type detection (15 categories) |
| 0.5 | Content Scraping | Full article extraction via trafilatura |
| 0.7 | Relevance Filter | Remove non-business content |
| 1 | Deduplication | MinHash LSH near-duplicate removal |
| 2 | NER | spaCy named entity recognition |
| 2.3 | Geo Filter | Entity-based geographic relevance |
| 2.5 | Co-occurrence | Entity relationship graph |
| 2.7 | Sentiment | VADER sentiment analysis |
| 3 | Embedding | Local SentenceTransformers (384-dim) |
| 3.5 | Semantic Dedup | Cosine similarity deduplication |
| 4 | UMAP | 384-dim -> 5-dim reduction |
| 5 | HDBSCAN | Density-based clustering |
| 5.5 | Coherence | Cluster quality validation in original space |
| 6 | Keywords | TF-IDF extraction |
| 7 | Signals | 7-module scoring (temporal, content, entity, source, market, search, composite) |
| 7.1 | Entity Graph | Bridge entity detection |
| 7.5 | Google Trends | Search interest validation |
| 7.7 | Trend Memory | Historical trend matching |
| 8 | LLM Synthesis | Human-readable summaries + 5W1H |
| 9 | Quality Gate | Drop low-quality clusters + tree assembly |
| 10 | Trend Linking | Cross-trend relationships |

## Anti-Hallucination System

| Layer | Protection |
|-------|-----------|
| V2 | JSON structure validation |
| V3 | Synthesis quality checks |
| V6 | Event classification accuracy |
| V7 | Company NER verification + Wikipedia |
| V9 | Cluster quality gate |
| V10 | Cross-validation with retry |
| Pydantic | Field-level schema validation |
| Dimension Lock | Embedding consistency |
| Dedup Safety | Degenerate embedding detection |

## News Sources

| Category | Sources |
|----------|---------|
| **Tier 1** | Economic Times, ET Industry, ET Tech, Mint, Business Standard, Moneycontrol, Financial Express, PIB, RBI |
| **Tier 2** | YourStory, Inc42, NDTV Profit, Hindu Business Line |
| **Tier 3** | Google News India (Business + Tech) |
| **APIs** | NewsAPI.org, MediaStack, TheNewsAPI, GNews, GDELT (2 feeds) |

## Project Structure

```
sales_agent/
├── streamlit_app.py              # Streamlit UI (5-step wizard)
├── requirements.txt
├── .env                          # API keys (gitignored)
├── SYSTEM_GUIDE.md               # Detailed developer reference
│
├── app/
│   ├── config.py                 # Settings + source definitions
│   ├── main.py                   # FastAPI entry point
│   │
│   ├── agents/                   # LangGraph pipeline agents
│   │   ├── impact_agent.py       # Trend impact analysis
│   │   ├── company_agent.py      # Company discovery + V7 verification
│   │   ├── contact_agent.py      # Decision maker finding
│   │   ├── email_agent.py        # Email finding + pitch generation
│   │   └── orchestrator.py       # Pipeline orchestration
│   │
│   ├── news/                     # News processing modules
│   │   ├── dedup.py              # MinHash LSH
│   │   ├── entity_extractor.py   # spaCy NER
│   │   ├── event_classifier.py   # Embedding-based classification
│   │   └── scraper.py            # Article content extraction
│   │
│   ├── schemas/                  # Pydantic data models
│   │   ├── base.py               # Enums (Severity, SignalStrength, etc.)
│   │   ├── news.py               # NewsArticle, Entity
│   │   ├── trends.py             # TrendNode, TrendTree
│   │   ├── sales.py              # CompanyData, ContactData, OutreachEmail
│   │   └── pipeline.py           # AgentState
│   │
│   ├── tools/                    # External service integrations
│   │   ├── llm_tool.py           # Multi-provider LLM (5 providers)
│   │   ├── embeddings.py         # 3-tier embeddings
│   │   ├── rss_tool.py           # 22+ source news fetcher
│   │   ├── tavily_tool.py        # Company search
│   │   ├── apollo_tool.py        # Contact + email finding
│   │   └── hunter_tool.py        # Email finding (fallback)
│   │
│   ├── trends/                   # ML pipeline modules
│   │   ├── engine.py             # RecursiveTrendEngine (12 phases)
│   │   ├── coherence.py          # Cluster validation
│   │   ├── keywords.py           # TF-IDF extraction
│   │   ├── reduction.py          # UMAP
│   │   ├── subclustering.py      # Recursive sub-trends
│   │   ├── tree_builder.py       # Hierarchy assembly
│   │   ├── trend_memory.py       # Historical persistence
│   │   └── signals/              # 7 signal modules
│   │
│   └── shared/                   # UI utilities
│       ├── styles.py             # CSS theme
│       ├── sidebar.py            # Settings panel
│       ├── helpers.py            # HTML escaping, data conversion
│       └── visualizations.py     # React Flow graph
│
└── tests/
    ├── test_validation_layers.py # 31 validation tests
    ├── test_pipeline.py          # Pipeline diagnostics
    ├── test_llm_providers.py     # Provider connectivity
    └── test_stress.py            # Load testing
```

## API Rate Limits

| Service | Free Tier | Notes |
|---------|-----------|-------|
| Tavily | 1,000/month | Company search |
| NewsAPI.org | 100/day | News ingestion |
| GNews | 100/day | News ingestion |
| MediaStack | 500/month | News ingestion |
| Apollo.io | 600/month | Primary contact finder |
| Hunter.io | 25/month | Fallback email finder |
| Gemini | 60 RPM | LLM provider |
| OpenRouter | Pay-per-use | LLM provider |
| Ollama | Unlimited | Local, recommended |

## Testing

```bash
# Validation layers (31 tests)
python -m pytest test_validation_layers.py -v

# Pipeline diagnostics
python -m pytest test_pipeline.py -v

# LLM provider connectivity
python -m pytest test_llm_providers.py -v
```

## Troubleshooting

**Ollama not connecting:**
```bash
ollama serve                    # Start server
ollama pull mistral             # Pull model
curl http://localhost:11434/api/tags  # Verify
```

**No trends detected:** Check that at least some news sources are returning articles. Enable "Audit sources" in the UI to see source health.

**LLM errors:** Ensure at least one provider has a valid API key. The system needs at least one working LLM for synthesis, impact analysis, company extraction, and email generation.

**Company search slow:** Company discovery makes multiple Tavily API calls + LLM extraction + Wikipedia verification per trend. For 14 trends, expect 5-20 minutes.