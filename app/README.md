# app/ -- Core Python Package

The `app/` package contains all backend logic: the LangGraph pipeline, FastAPI API layer, trend engine, LLM tooling, and self-learning system.

## Package Structure

| Package | Purpose |
|---------|---------|
| `agents/` | LangGraph pipeline orchestrator, worker agents, causal council |
| `api/` | FastAPI routers: pipeline, leads, feedback, health, learning |
| `learning/` | Source bandit, weight learner, pipeline metrics, specificity scorer |
| `news/` | RSS scraping, MinHash dedup, spaCy NER, event classification |
| `schemas/` | Pydantic models (news, trends, sales, pipeline state, validation) |
| `search/` | BM25 ranking, SearXNG integration, DuckDuckGo fallback |
| `shared/` | Helpers (geo filter, stopwords, HTML escaping) |
| `tools/` | LLM service, provider manager, embeddings, Apollo, Hunter, SearXNG |
| `trends/` | Trend engine, Leiden clustering, coherence, signals, synthesis |
| `ui/` | Streamlit UI components (legacy, being replaced by Next.js) |
| `data/` | Provider health state files |

## Pipeline Flow

```
source_intel ──> analysis ──> impact ──> quality ──> causal_council
                                                          |
                                                          v
                                          lead_crystallize ──> lead_gen ──> learning_update
```

Each step is a LangGraph node in `agents/orchestrator.py`. State flows through `GraphState` (TypedDict) with typed fields for trends, impacts, companies, contacts, and emails.

## Entry Points

```bash
# FastAPI server (production)
uvicorn app.main:app --reload --port 8000

# CLI headless run
python -c "import asyncio; from app.agents.orchestrator import run_pipeline; asyncio.run(run_pipeline(mock_mode=False))"

# Streamlit (legacy)
streamlit run streamlit_app.py --server.port 8501
```

## Configuration Reference

All settings are in `.env`, loaded via `app/config.py` (pydantic-settings `BaseSettings`).

### LLM Providers

| Variable | Default | Notes |
|----------|---------|-------|
| `GEMINI_API_KEY` | -- | Vertex Express free tier (primary) |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Cheapest GA model with tool calling |
| `OPENAI_API_KEY` | -- | GPT-4.1-mini / nano |
| `GROQ_API_KEY` | -- | Fast inference (Qwen 32B) |
| `NVIDIA_API_KEY` | -- | DeepSeek V3.1 |
| `OPENROUTER_API_KEY` | -- | Multi-model proxy |
| `USE_OLLAMA` | `true` | Local fallback |
| `OLLAMA_MODEL` | `mistral` | Any Ollama model |

### Embeddings

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBEDDING_PROVIDER` | `nvidia` | `nvidia`, `api`, or `local` |
| `EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | 1024-dim |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | 1024-dim (dimensioned) |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 1024-dim local |
| `HF_API_KEY` | -- | HuggingFace Inference API |

### Trend Engine

| Variable | Default | Notes |
|----------|---------|-------|
| `LEIDEN_K` | `20` | k-NN neighbors for graph |
| `LEIDEN_RESOLUTION` | `1.0` | Higher = more, smaller clusters |
| `LEIDEN_AUTO_RESOLUTION` | `true` | Auto-tune to target range |
| `LEIDEN_OPTUNA_ENABLED` | `true` | Bayesian hyperparameter tuning |
| `LEIDEN_MIN_COMMUNITY_SIZE` | `3` | Minimum articles per cluster |
| `DEDUP_THRESHOLD` | `0.25` | MinHash LSH similarity threshold |

### External APIs

| Variable | Notes |
|----------|-------|
| `APOLLO_API_KEY` | Contact finding (600/month free) |
| `HUNTER_API_KEY` | Email verification (25/month free) |
| `SEARXNG_URL` | Default `http://localhost:8888` |
| `COUNTRY` / `COUNTRY_CODE` | Target market (default `India` / `IN`) |

### Pipeline Limits

| Variable | Default | Notes |
|----------|---------|-------|
| `MAX_TRENDS` | `3` | Trends to process in lead gen |
| `MAX_COMPANIES_PER_TREND` | `3` | Companies per trend |
| `MAX_CONTACTS_PER_COMPANY` | `2` | Contacts per company |
| `EMAIL_CONFIDENCE_THRESHOLD` | `70` | Minimum email confidence score |

## Adding a New LLM Provider

1. Add config fields to `app/config.py` (`Settings` class)
2. Add model builder in `app/tools/provider_manager.py` (`_build_provider_chain()`)
3. Insert into the appropriate chain position (general, structured, tool-calling, lite)
4. Add health check in `app/tools/provider_health.py`
5. Test: `uvicorn app.main:app --reload` then `GET /health`
