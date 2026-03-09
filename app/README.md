# app/ — Backend Package

The core Python package for the B2B Sales Intelligence Platform. FastAPI + LangGraph pipeline, 10-stage intelligence engine, 7 self-learning loops.

## Package Structure

```
app/
├── agents/          # LangGraph pipeline (orchestrator + 4 worker agents)
├── api/             # FastAPI routers (9 routers)
├── intelligence/    # 10-stage article-to-cluster pipeline
│   ├── engine/      # Math core: NLI filter, similarity, NER, classifier, validator
│   └── cluster/     # HAC + HDBSCAN + Leiden cascade
├── learning/        # 7 self-learning loops (bandit, SetFit, EWC, Q-router)
├── tools/
│   ├── crm/         # Apollo, Hunter, Brevo wrappers
│   ├── web/         # Tavily, RSS, web_intel, news_collector
│   └── llm/         # ProviderManager (7-provider fallback), embeddings
├── data/            # mock_articles.py, provider_health.json, models/
└── schemas/         # Shared Pydantic models
```

---

## Entry Points

```bash
# FastAPI server (port 8000)
uvicorn app.main:app --reload --port 8000

# CLI pipeline run
python main.py --mode industry --region IN --hours 72 --products "CRM,Analytics"

# CLI company-first
python main.py --mode company --companies "Zepto,Blinkit" --region IN

# Headless API run
python -c "import asyncio; from app.intelligence.pipeline import execute; ..."
```

---

## 10-Stage Intelligence Pipeline

Entry: `app/intelligence/pipeline.py:execute(scope)` → `IntelligenceResult`

**Principle**: deterministic backbone, scoped agency — math handles stages 1-8, LLM handles only synthesis (stage 9).

```
Stage │ Module                           │ Algorithm
──────┼──────────────────────────────────┼─────────────────────────────────────────
  1   │ intelligence/fetch.py            │ RSS + Tavily + Google News (parallel)
  2   │ intelligence/fetch.py            │ SHA-256 exact + MinHash LSH near-dedup
  3   │ intelligence/filter.py           │ NLI: DeBERTa v3 small cross-encoder
  4   │ intelligence/engine/extractor.py │ GLiNER B2B labels + SpaCy en_core_web_sm
  5   │ intelligence/engine/normalizer.py│ rapidfuzz fuzzy entity normalization
  6   │ intelligence/engine/similarity.py│ 6-signal decomposed similarity matrix
  7   │ intelligence/cluster/            │ HAC → HDBSCAN soft → Leiden cascade
  8   │ intelligence/cluster/validator.py│ 7-check math gate per cluster
  9   │ intelligence/summarizer.py       │ FIRST LLM CALL — synthesis + Reflexion
 10   │ intelligence/match.py            │ Product ↔ cluster opportunity scoring
```

---

## API Routers (FastAPI on port 8000)

All 9 routers registered in `app/main.py`:

| Mount | Router | Key endpoints |
|-------|--------|---------------|
| `/api/v1/pipeline` | `api/pipeline.py` | `POST /run`, `GET /stream/{id}`, `GET /result/{id}` |
| `/api/v1/leads` | `api/leads.py` | `GET /`, `GET /{id}`, `POST /enrich`, `POST /send-email` |
| `/api/v1/feedback` | `api/feedback.py` | `POST /`, `GET /history` |
| `/api/v1/learning` | `api/learning.py` | `GET /status`, `GET /bandit-state` |
| `/api/v1/companies` | `api/companies.py` | `POST /search`, `GET /{id}/news`, `POST /{id}/generate-leads` |
| `/api/v1/news` | `api/news.py` | `GET /` |
| `/api/v1/campaigns` | `api/campaigns.py` | `POST /`, `GET /`, `POST /{id}/run`, `DELETE /{id}` |
| `/api/v1/profiles` | `api/profiles.py` | `POST /`, `GET /`, `PUT /{id}` |
| `/health` | `api/health.py` | `GET /` — 503 if DB or provider unreachable |

SSE streaming: `GET /api/v1/pipeline/stream/{run_id}` — uses `stream_mode="updates"` (NOT `"values"` — crashes on AgentDeps serialization).

---

## Configuration (`app/config.py`)

All settings loaded from `.env` via pydantic-settings `BaseSettings`.

### LLM Provider Chain

Priority order: **OpenAI** → GeminiDirect → VertexLlama → NVIDIA → Groq → OpenRouter → Ollama

| Variable | Default | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | — | GPT-4.1-mini (500+ RPM), primary provider |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Structured output + tool calling |
| `OPENAI_LITE_MODEL` | `gpt-4.1-nano` | Classification (fast, cheap) |
| `GEMINI_API_KEY` | — | Vertex Express 300 RPM free tier |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Primary fallback |
| `GROQ_API_KEY` | — | Llama 3.3 70B — fast inference |
| `NVIDIA_API_KEY` | — | DeepSeek V3.1 |
| `USE_OLLAMA` | `false` | Local last-resort |

### Embeddings

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBEDDING_PROVIDER` | `nvidia` | Priority: nvidia → openai → api → local → ollama |
| `EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | 1024-dim |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | 1024-dim Matryoshka |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 1024-dim CPU/GPU |

### Region-Aware RSS Sources

| Region | Count | Publication categories |
|--------|-------|------------------------|
| `IN` | 50 | ET, LiveMint, Inc42, YourStory, MoneyControl, CNBCTV18, Sebi, RBI |
| `US` | 35 | WSJ, TechCrunch, Bloomberg, Crunchbase, VentureBeat |
| `EU` | 14 | BBC Business, FT, Sifted, EU-Startups |
| `SEA` | 9 | CNA, Straits Times, TechinAsia, KrASIA |
| `GLOBAL` | 107 | All DEFAULT_ACTIVE_SOURCES |

Allowlists defined in `intelligence/config.py:REGION_SOURCES`. Each region only fetches its own publications — India runs never include Forbes, Inc Magazine, or Al Jazeera.

### NLI Filter Thresholds

| Threshold | Value | Behavior |
|-----------|-------|----------|
| `nli_auto_accept` | `0.88` | Entailment ≥ 0.88 → pass immediately (no LLM) |
| `nli_auto_reject` | `0.10` | Entailment < 0.10 → drop immediately (no LLM) |
| LLM zone | 0.10–0.88 | GPT-4.1-nano batch-classifies these |

Threshold history: 0.55 → 0.75 → 0.88 (raised after specific false positives on real 120h data: Virat Kohli 0.569, PM Modi metro 0.778, ED bail hearing 0.822).

### Clustering Parameters

| Variable | Default | Notes |
|----------|---------|-------|
| `LEIDEN_K` | `20` | k-NN graph neighbors (fixed; only resolution is Optuna-tuned) |
| `LEIDEN_RESOLUTION` | Optuna-tuned | Higher = more, smaller communities |
| `LEIDEN_OPTUNA_ENABLED` | `true` | Bayesian resolution search (15 trials, 30s timeout) |
| `hac_threshold_min/max` | 0.30/0.65 | HAC dendrogram cut sweep range |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 5 | Per Campello et al. 2013 |

### Pipeline Limits

| Variable | Default | Notes |
|----------|---------|-------|
| `MAX_TRENDS` | `12` | Clusters passed to LangGraph lead gen |
| `MAX_COMPANIES_PER_TREND` | `15` | Companies researched per cluster |
| `MAX_CONTACTS_PER_COMPANY` | `6` | 3 decision-makers + 3 influencers |
| `RSS_MAX_PER_SOURCE` | `25` | Articles fetched per RSS feed |
| `RSS_HOURS_AGO` | `120` | Default look-back window |

---

## Import Paths (Critical)

```python
from app.tools.crm.apollo_tool import ApolloTool
from app.tools.web.tavily_tool import TavilyTool
from app.tools.llm.providers import ProviderManager
from app.tools.llm.llm_service import LLMService
from app.intelligence.pipeline import execute
from app.intelligence.models import DiscoveryScope, DiscoveryMode
from app.learning.company_bandit import CompanyRelevanceBandit
```

**Do NOT** import from flat `app/tools/` paths — use sub-package paths (`crm/`, `web/`, `llm/`).

---

## Adding a New LLM Provider

1. Add config fields to `app/config.py` (`Settings` class)
2. Add model builder in `app/tools/llm/providers.py` → `_build_provider_chain()`
3. Insert at correct priority position (general / structured / tool-calling / lite chain)
4. Add health check stub in `app/api/health.py`
5. Verify: `GET /health` returns provider in status dict
