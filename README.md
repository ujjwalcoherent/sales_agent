# Sales Agent -- India Trend-to-Lead Pipeline

AI pipeline that scrapes 22+ RSS sources, clusters articles into business trends via Leiden community detection, then runs a LangGraph agent pipeline to find target companies, contacts, and generate outreach emails. Self-learns through 6 feedback loops (Thompson Sampling, EMA drift, weight adaptation).

## Dev Setup

Prerequisites: Python 3.11+, Node 20+, Docker (for SearXNG only).

```bash
git clone <repo> && cd sales_agent

# 1. Install everything
npm run install:all    # pip install + spacy model + frontend npm install

# 2. Configure
cp .env.example .env   # fill in at least GEMINI_API_KEY or OPENAI_API_KEY

# 3. Start SearXNG (Docker — only external service needed)
npm run searxng         # runs searxng/searxng on port 8888

# 4. Run
npm run dev             # starts api (port 8000) + frontend (port 3000) in parallel
```

### Individual commands

```bash
npm run api             # uvicorn app.main:app --reload --port 8000
npm run frontend        # next dev on port 3000
npm run pipeline:run    # headless pipeline run (no server needed)
npm run pipeline:mock   # replay recorded data (~45s)
npm run searxng:stop    # stop SearXNG container
```

Backend and frontend run locally (no Docker). Only SearXNG needs Docker because it's a standalone search engine. The frontend connects to `http://localhost:8000` by default (configured in `frontend/lib/api.ts`).

### Full Docker deployment (optional)

For deploying the entire stack in containers:

```bash
docker compose up -d    # searxng + api + frontend
docker compose logs -f
```

Note: `NEXT_PUBLIC_API_URL` is a build-time arg in the frontend Dockerfile (Next.js inlines `NEXT_PUBLIC_*` at build time, not runtime).

## Architecture

```
22+ RSS Sources ──> source_intel ──> analysis ──> impact ──> quality
                                     (embed,      (LLM per    (confidence
                                      Leiden,      trend)       gate)
                                      LLM synth)
                                                                  |
                                                                  v
                    learning_update <── lead_gen <── lead_crystallize <── causal_council
                    (6 loops)          (SearXNG,     (causal hops       (4-agent
                                       Apollo,       to lead sheets)     reasoning)
                                       Hunter)
```

Pipeline runs ~25-40 min real, ~45s mock replay. Each step is a LangGraph node in `app/agents/orchestrator.py`.

### LLM Provider Chain (auto-failover with exponential backoff)

| Chain | Order (position 1 tried first) |
|-------|------|
| General | OpenAI (gpt-4.1-mini) -> GeminiDirect -> VertexLlama -> NVIDIA -> Groq -> OpenRouter -> Ollama |
| Structured | OpenAI -> GeminiDirect -> Groq -> Ollama |
| Tool calling | OpenAI -> GeminiDirect -> Groq -> VertexLlama -> Ollama |
| Lite | OpenAI Nano (gpt-4.1-nano) -> GeminiDirectLite -> Groq -> standard |
| Embeddings (openai mode) | OpenAI text-embedding-3-large (1536-dim, sole provider) |
| Embeddings (nvidia mode) | NVIDIA nv-embedqa-e5-v5 -> OpenAI -> HF API -> Local -> Ollama |

OpenAI is position 1 for reliability (500+ RPM, best structured output). GeminiDirect at position 2 uses $300 GCP free credits as fallback. Default embedding mode is `openai` at 1536-dim (set `EMBEDDING_PROVIDER=nvidia` for 1024-dim NVIDIA-first chain).

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Provider status, cooldown state |
| `POST` | `/api/v1/pipeline/run` | Start pipeline, returns `run_id` |
| `GET` | `/api/v1/pipeline/stream/{run_id}` | SSE real-time progress |
| `GET` | `/api/v1/pipeline/status/{run_id}` | Polling fallback |
| `GET` | `/api/v1/pipeline/result/{run_id}` | Full results (trends, leads, impacts) |
| `GET` | `/api/v1/leads` | Filtered leads (`?hop=&lead_type=&min_confidence=`) |
| `POST` | `/api/v1/feedback` | Trend/lead ratings for learning |
| `GET` | `/api/v1/feedback/summary` | Feedback stats |

## Project Structure

```
sales_agent/
├── app/                  # Python backend (see app/README.md)
│   ├── agents/           # LangGraph pipeline nodes + workers + council
│   ├── api/              # FastAPI routers
│   ├── learning/         # 6 self-learning loops
│   ├── news/             # RSS, dedup, NER, event classification
│   ├── schemas/          # Pydantic models
│   ├── search/           # SearXNG, BM25
│   ├── shared/           # Helpers (geo, stopwords)
│   ├── tools/            # LLM, embeddings, Apollo, Hunter
│   ├── trends/           # Trend engine, clustering, signals
│   ├── ui/               # Streamlit components (legacy)
│   └── main.py           # FastAPI app factory
├── frontend/             # Next.js dashboard (see frontend/README.md)
├── docker/               # Dockerfiles + SearXNG config
├── data/                 # Runtime: SQLite, JSONL logs, learned weights
├── package.json          # Monorepo scripts (npm run dev/api/frontend)
├── docker-compose.yml    # Full stack: searxng + api + frontend
└── .env.example          # Template with all vars
```

## Gotchas and Architecture Decisions

### LangGraph stream_mode
MUST use `stream_mode="updates"` not `"values"`. The `"values"` mode tries to msgpack-serialize `AgentDeps` which contains ChromaDB clients and LLM model objects -- instant crash.

### Embedding dimension locking
Embedding dimension is locked by the primary provider (1536-dim in openai mode, 1024-dim in nvidia mode). The `_dim_locked` mechanism in `embeddings.py` rejects any fallback provider that returns different dimensions. If you see "DIMENSION MISMATCH" in logs, a provider fell back to one with incompatible vector size (usually Ollama's 768-dim nomic-embed-text).

### Provider cooldown state is class-level
`ProviderManager._failed_providers` is a class-level dict shared across all instances. A 429 on GeminiDirect in the analysis step also affects the impact step. This is intentional -- prevents hammering a rate-limited provider.

### TrendData field name mismatch
Internal pipeline uses `trend_title` / `industries_affected`. The API `TrendResponse` uses `title` / `industries`. The result endpoint maps both -- check `app/api/pipeline.py` if you're adding fields.

### Provider reset between runs
Each API pipeline run MUST call `provider_health.reset_for_new_run()`, `ProviderManager.reset_cooldowns()`, and `LLMService.clear_cache()`. The CLI `run_pipeline()` does this. The API `_execute_pipeline` does this. If you add a third entry point, you must too.

### NEXT_PUBLIC_API_URL
In dev mode, the frontend defaults to `http://localhost:8000` (hardcoded fallback in `frontend/lib/api.ts`). In Docker, it's passed as a build arg because Next.js inlines `NEXT_PUBLIC_*` at build time. Setting it as a runtime env var does nothing for client-side code.

## Debugging

```bash
# Check provider health
curl http://localhost:8000/health | python -m json.tool

# Check which providers are available
python -c "from app.tools.provider_manager import ProviderManager; pm=ProviderManager(); pm.get_model(); print(pm.get_provider_names())"

# Test OpenAI specifically
python -c "from app.tools.llm_service import LLMService; import asyncio; llm=LLMService(disabled_providers=['GeminiDirect']); print(asyncio.run(llm.generate('test', system_prompt='Reply OK')))"

# Test embeddings
python -c "from app.tools.embeddings import EmbeddingTool; et=EmbeddingTool(); e=et.embed_text('test'); print(f'dim={len(e)} provider={et._active_provider}')"

# View last pipeline run
python scripts/view_run.py
```

## Package Documentation

- [app/](app/README.md) -- Backend package, config reference, adding providers
- [app/agents/](app/agents/README.md) -- LangGraph orchestrator, worker agents, council
- [app/tools/](app/tools/README.md) -- LLM service, provider manager, embeddings, API tools
- [app/trends/](app/trends/README.md) -- Trend engine, Leiden clustering, signals
- [app/learning/](app/learning/README.md) -- 6 self-learning loops, feedback
- [frontend/](frontend/README.md) -- Next.js dashboard, API layer, dev workflow
