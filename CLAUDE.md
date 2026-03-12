# Sales Agent — Development Rules

## Project
B2B sales intelligence platform. Python backend (FastAPI + LangGraph + pydantic-ai), Next.js frontend (shadcn/ui).

## Run Commands
```bash
uvicorn app.main:app --reload --port 8000           # API server
cd frontend && npm run dev                            # Next.js dev
venv/Scripts/python.exe -c "import asyncio; from app.agents.orchestrator import run_pipeline; asyncio.run(run_pipeline(mock_mode=False))"  # Pipeline
```

## Critical Rules

### LangGraph
- MUST use `stream_mode="updates"` — NEVER `"values"` (crashes on AgentDeps serialization)
- TrendData field mapping: internal=`trend_title`/`industries_affected`, API=`title`/`industries`
- Provider reset between runs: `provider_health.reset_for_new_run()`, `ProviderManager.reset_cooldowns()`, `LLMService.clear_cache()`

### Search & Enrichment
- **Tavily** is the primary search provider. DDG = free fallback. SearXNG fully removed.
- 7 Tavily API keys with round-robin rotation (thread-safe via `threading.Lock`)
- Entity validation: Tavily search + LLM classify (replaced Wikidata P31)
- ScrapeGraphAI for deep company search (background only — 30-80s latency)
- Apollo for contacts + company org data (semaphore: 3). Hunter for email finding + verification (semaphore: 2).
- ChromaDB: separate collections for pipeline articles (`articles`) and news (`articles_news`) — different embedding models.

### News Collection
- Fresh news (7d): Google News RSS + Tavily news (parallel, merged, deduped)
- Historical (1-5mo): `gnews` library with monthly date-range windows
- LLM-lite relevance filter: auto-accept if company name in title/summary, LLM batch classify ambiguous articles
- `app/tools/web/news_collector.py` orchestrates both sources -> relevance filter -> content extraction -> ChromaDB storage
- gnews URLs are opaque Google News redirects — content extraction works on RSS URLs, gnews provides title/summary/date

### Code Style
- Type hints on all functions
- Pydantic models with field validators for LLM output coercion
- Lazy initialization via properties on `AgentDeps` (never import heavy deps at module level)
- `asyncio.Semaphore` for ALL external API calls (bounded concurrency)

## Folder Structure
```
app/
|-- agents/          # LangGraph pipeline nodes
|   |-- orchestrator.py, deps.py, leads.py, source_intel.py
|   +-- workers/     # Per-entity worker agents (contact, email, impact)
|-- api/             # FastAPI routers (pipeline, leads, companies, campaigns, health)
|-- intelligence/    # News clustering pipeline
|   |-- fetch.py, filter.py, match.py, summarizer.py
|   |-- cluster/     # HAC+HDBSCAN+Leiden algorithms
|   +-- engine/      # Math core (similarity, NER, clustering, validation)
|       +-- tools/   # classifier.py, extractor.py, normalizer.py, clusterer.py
|-- learning/        # Self-learning loops
|   |-- signal_bus.py, source_bandit.py, company_bandit.py, contact_bandit.py
|   |-- threshold_adapter.py, pipeline_metrics.py, experiment_tracker.py
|   +-- dataset_enhancer.py
|-- tools/           # External integrations (pure wrappers, no business logic)
|   |-- crm/         # apollo_tool.py, hunter_tool.py, brevo_tool.py
|   |-- web/         # tavily_tool.py, rss_tool.py, web_intel.py, news_collector.py
|   |-- llm/         # llm_service.py, providers.py, embeddings.py, mock_responses.py
|   +-- (root)       # domain_utils, json_repair, geo, search, company_enricher, person_intel
|-- data/            # mock_articles.py, provider_health.json, models/ (GLiNER)
+-- schemas/         # Shared Pydantic models
```

## Architecture Invariants
- `app/tools/crm/` = CRM API wrappers (Apollo, Hunter, Brevo — no business logic)
- `app/tools/web/` = web/news sources (Tavily, RSS, web_intel multi-source orchestrator)
- `app/tools/llm/` = LLM providers + embeddings (ProviderManager fallback chain)
- `app/tools/web/web_intel.py` = multi-source orchestrator (Tavily -> DDG -> Google News RSS -> gnews -> trafilatura -> ScrapeGraphAI)
- `app/tools/company_enricher.py` = CENTRAL gatekeeper (all companies pass entity validation)
- Mock data lives in `app/data/mock_articles.py` — never inline in business logic
- Learning loops (`app/learning/`) communicate via signal_bus only — no direct imports between loops
- Database: SQLite (`leads.db`) + ChromaDB (article embeddings, trend memory)
- Campaigns: orchestration layer composing existing tools (`app/api/campaigns.py`)
  - Campaign executor uses per-campaign semaphore + `asyncio.Lock` for DB updates
  - Health endpoint returns 503 when DB or critical deps are down

## Reference Context
Read these when working on specific areas:
- API/pipeline: `.claude/reference/architecture.md`
- Tavily/search: `.claude/reference/tavily-integration.md`
- Campaigns: `.claude/reference/campaign-architecture.md`
- Frontend: `.claude/reference/frontend-patterns.md`
- PRD (north star): `docs/PRD.md`

## What NOT to Do
- Do NOT add ScrapeGraphAI to interactive (user-facing) paths — 30-80s latency
- Do NOT hardcode "India" — use `settings.country`
- Do NOT create new LLM provider chains — use `ProviderManager.get_model()` from `app.tools.llm.providers`
- Do NOT import from `app.tools.wikidata` (archived in `archive/enrichment_v1/`)
- Do NOT use `from __future__ import annotations` with local classes + `get_type_hints()`
- Do NOT commit `.env`, `leads.db`, `*.log`, `data/recordings/`
- Do NOT import tools at flat paths — use sub-package paths:
  - CRM: `from app.tools.crm.apollo_tool import ApolloTool`
  - Web: `from app.tools.web.tavily_tool import TavilyTool`
  - LLM: `from app.tools.llm.providers import ProviderManager`
