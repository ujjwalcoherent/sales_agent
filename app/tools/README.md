# app/tools/ — External Integrations

Pure API wrappers and shared utilities. No business logic, no fallback chains within a single tool.

```
tools/
├── crm/                        # CRM & outreach integrations
│   ├── apollo_tool.py          # Contact search + company org data (Apollo.io)
│   ├── hunter_tool.py          # Email finding + verification (Hunter.io)
│   └── brevo_tool.py           # Email sending (Brevo/SendinBlue SMTP)
├── web/                        # Web search & news sources
│   ├── tavily_tool.py          # Primary search (rotating API keys, round-robin, thread-safe)
│   ├── rss_tool.py             # RSS feed fetcher (uses DEFAULT_ACTIVE_SOURCES from config)
│   ├── web_intel.py            # Multi-source orchestrator (Tavily→DDG→RSS→gnews→scrape)
│   ├── news_collector.py       # Company news: fresh (7d) + historical (1-5mo)
│   └── content_scraper.py      # Full-text extraction via trafilatura
├── llm/                        # LLM providers & embeddings
│   ├── providers.py            # ProviderHealthTracker + ProviderManager
│   ├── llm_service.py          # generate() / run_structured() interface
│   ├── embeddings.py           # Multi-tier embedding with dimension locking
│   └── mock_responses.py       # Deterministic mock outputs for tests
└── (root shared utilities)
    ├── company_enricher.py     # Central entity validation + enrichment gatekeeper
    ├── person_intel.py         # Person background scraping for email personalization
    ├── domain_utils.py         # Domain extraction and normalization
    ├── json_repair.py          # LLM JSON output repair (strips fences, fixes truncation)
    ├── search.py               # BM25Search (offline) + SearchManager (BM25 + DDG fallback)
    ├── article_cache.py        # ChromaDB article embedding cache
    ├── event_classifier_tool.py # Two-tier event classification (embedding + LLM validation)
    ├── feedback_store.py       # JSONL feedback collection for SetFit training
    └── run_recorder.py         # Step snapshots for replay (data/recordings/)
```

---

## Import Paths

Always use sub-package paths. Never import at module level in agents (lazy load required).

```python
from app.tools.crm.apollo_tool import ApolloTool
from app.tools.crm.hunter_tool import HunterTool
from app.tools.crm.brevo_tool import BrevoTool
from app.tools.web.tavily_tool import TavilyTool
from app.tools.web.rss_tool import RSSTool
from app.tools.web.web_intel import WebIntelTool
from app.tools.web.news_collector import NewsCollector
from app.tools.llm.providers import ProviderManager, provider_health
from app.tools.llm.llm_service import LLMService
from app.tools.llm.embeddings import EmbeddingTool
from app.tools.company_enricher import validate_entity, enrich, enrich_batch
from app.tools.search import BM25Search, SearchManager
```

---

## Concurrency Limits

All external APIs bounded by module-level `asyncio.Semaphore`:

```python
# apollo_tool.py
_APOLLO_SEM = asyncio.Semaphore(3)

# hunter_tool.py
_HUNTER_SEM = asyncio.Semaphore(2)
```

Tavily has no semaphore — it uses round-robin key rotation (thread-safe via `threading.Lock`). LLM calls are managed by pydantic-ai `FallbackModel` with CUSUM-based provider switching.

---

## CRM Tools

### `ApolloTool` (apollo_tool.py)

Semaphore: `_APOLLO_SEM = asyncio.Semaphore(3)` (module-level).

Key methods:
```python
find_email(domain, full_name, role=None) -> EmailFinderResult
search_people_at_company(domain, roles, limit) -> List[ContactResult]
get_cached_org(domain) -> dict | None    # class-level session cache, max 500 entries
```

Returns `EmailFinderResult` (from `app.schemas`). Caches Apollo org data per domain to avoid re-querying within a session.

### `HunterTool` (hunter_tool.py)

Semaphore: `_HUNTER_SEM = asyncio.Semaphore(2)` (module-level).

Key methods:
```python
find_email(domain, full_name) -> EmailFinderResult
```

Used as fallback when Apollo returns no result. Free tier: 25 searches/month.

### Hunter fallback flow in `contact_agent.py`

```
1. Apollo: search_people_at_company(domain, roles, limit)
2. For each contact without verified email:
   Hunter: find_email(domain, full_name)
3. Filter: EMAIL_CONFIDENCE_THRESHOLD = 70
```

---

## Web Tools

### `TavilyTool` (tavily_tool.py)

No semaphore — uses rotating API keys with round-robin selection:

```python
# Thread-safe key rotation
_lock = threading.Lock()   # class-level
_key_index = 0             # class-level

def _next_key() -> str:
    with self._lock:
        key = self._keys[TavilyTool._key_index % len(self._keys)]
        TavilyTool._key_index += 1
        return key
```

Keys loaded from `TAVILY_API_KEYS` env var (comma-separated). Number of keys varies by deployment.

In-memory LRU cache: `_CACHE_TTL = 300s`, `_CACHE_MAX = 100` entries, `threading.Lock`-protected.

**Methods:**
```python
search(query, search_depth="basic", max_results=5, include_answer=True, topic="general",
       time_range=None, include_domains=None, exclude_domains=None,
       include_raw_content=False, use_cache=True) -> Dict
news_search(query, max_results=5, time_range="week") -> Dict
finance_search(query, max_results=5) -> Dict
extract(url) -> Dict                          # Tavily Extract API (full page content)
deep_company_research(company_name) -> Dict   # 3 parallel calls
enrich_trend(trend_title, summary) -> Dict
```

`search_depth="advanced"` = 2 credits, 5× more content; `"basic"` = 1 credit.
`include_answer="advanced"` = multi-paragraph AI summary.

Excluded domains always: `reddit.com`, `quora.com`, `wikipedia.org`, `youtube.com`, `facebook.com`, `twitter.com`.

### `RSSTool` (rss_tool.py)

Uses `DEFAULT_ACTIVE_SOURCES` from `app.config` (103 sources as of March 2026). Language detection via `langdetect` (Google n-gram, 55+ languages) on first 500 chars of article text.

### `WebIntelTool` / `web_intel.py`

Multi-source orchestrator. Source hierarchy:
- Search: Tavily (primary, advanced depth) → DuckDuckGo (free fallback)
- Extract: trafilatura (fast local, F1=0.958) → Jina Reader (JS-heavy) → empty
- News fresh (7d): Google News RSS + Tavily news (parallel, merged, deduped)
- News historical (1-5mo): `gnews` library with monthly date-range windows
- Deep: ScrapeGraphAI SearchGraph — background only (30-80s latency, never in interactive paths)

Functions:
```python
search(query, max_results=5) -> List[Dict]
extract(url) -> str
company_news(company_name, max_articles=10) -> List[Dict]        # last 7 days
company_news_gnews(company_name, months_back=5) -> List[Dict]    # 1-5 months
filter_relevant_articles(company_name, articles) -> List[Dict]   # LLM-lite relevance filter
deep_company_search(company_name) -> Dict                        # ScrapeGraphAI background
search_industry_companies(industry) -> List[Dict]
```

---

## LLM Tools

### ProviderManager (`providers.py`)

Builds pydantic-ai `FallbackModel` instances. Class-level `_failed_providers` dict shared across all instances. Provider switching is immediate on error (`_FAILURE_COOLDOWN = 0.0`), with exponential backoff on 429s (`_RATELIMIT_COOLDOWN = 30s` base, doubles per failure, capped).

**Provider chain — `get_model()` (general text generation):**

| Priority | Provider | Model | Notes |
|----------|----------|-------|-------|
| 1 | OpenAI | gpt-4.1-mini | 500+ RPM, best structured output + tool calling |
| 2 | GeminiDirect | gemini-2.5-flash-lite (via Vertex AI) | $300 trial, rate-limited at 1.5 req/s |
| 3 | VertexLlama | llama-4-scout-17b (configurable) | Same GCP project, model garden |
| 4 | NVIDIA | deepseek-v3.1 (OpenAI-compat NIM API) | No daily token limit |
| 5 | Groq | qwen-qwen3-32b | Free, 100K tokens/day |
| 6 | OpenRouter | gemini-2.5-flash | Cloud proxy fallback |
| 7 | Ollama | mistral (configurable) | Local, last resort |

**`get_structured_output_model()` chain** (VertexLlama/NVIDIA excluded — don't support forced function calling):
1. OpenAI → 2. GeminiDirect → 3. Groq → 4. Ollama

**`get_lite_model()` chain** (cheaper models for classification):
1. OpenAI Nano (gpt-4.1-nano) → 2. GeminiDirectLite → 3. Groq → 4. standard chain fallback

**GCP rate limiter:** `_TokenBucket` at 1.5 req/s, burst=2. Prevents hitting Vertex AI DSQ ceiling.

**Provider health:** `ProviderHealthTracker` (module-level singleton `provider_health`) tracks per-provider failure counts. `provider_health.reset_for_new_run()` clears stale failures at pipeline start.

### LLMService (`llm_service.py`)

```python
llm = LLMService()
llm_lite = LLMService(lite=True)
```

**Methods:**
```python
generate(prompt, system_prompt=None, temperature=0.7, max_tokens=2000, json_mode=False) -> str
run_structured(prompt, system_prompt="", output_type=str, retries=2,
               temperature=0.3, reflect_retries=2) -> T
generate_json(prompt, system_prompt=None, schema_hint=None, pydantic_model=None,
              required_keys=None, max_retries=None) -> dict
```

`generate()` does NOT accept `model_tier` kwarg.

**ReflectAndRetry pattern in `run_structured()`:** when structured output fails validation, the error message is injected back into the prompt and retried on the same provider before triggering failover. Up to `reflect_retries=2` reflect attempts.

**Agent cache key:** `(output_type, hash(system_prompt), retries, mock_mode, lite, cooldown_state, needs_structured)` — rebuilt when provider cooldown state changes.

For text output (`output_type=str`): uses `get_model()` (VertexLlama included).
For structured output: uses `get_structured_output_model()` (VertexLlama excluded).

### EmbeddingTool (`embeddings.py`)

Dimension-locked: once first batch completes, dimension is locked. Subsequent providers returning different dimensions are silently rejected.

**Provider chain depends on `EMBEDDING_PROVIDER` env var:**

| Mode | Fallback chain |
|------|---------------|
| `nvidia` (default) | NVIDIA NIM → OpenAI → HuggingFace API → Local → Ollama |
| `openai` | OpenAI only |
| `api` | HuggingFace API → NVIDIA → OpenAI → Ollama → Local |
| `local` | Local → NVIDIA → OpenAI → HuggingFace API → Ollama |

| Provider | Model | Dimensions |
|----------|-------|-----------|
| NVIDIA NIM | nv-embedqa-e5-v5 | 1024 |
| OpenAI | text-embedding-3-large | 1024 (Matryoshka @1024) |
| HuggingFace API | BAAI/bge-large-en-v1.5 | 1024 |
| Local | BAAI/bge-large-en-v1.5 | 1024 |
| Ollama | nomic-embed-text | 768 (rejected if higher-tier ran first) |

---

## Root Shared Utilities

### `company_enricher.py`

Central entity validation gatekeeper — all companies pass through before being returned to the user.

**Validation sources:**
1. Tavily search + LLM classify (is this a business entity?)
2. Apollo org data exists for the company's domain
3. Valid corporate website domain

**Entry points:**
```python
validate_entity(company_name) -> ValidationResult
enrich(company_name) -> CompanyData
enrich_batch(company_names) -> List[CompanyData]
```

`ValidationResult` fields: `is_valid_company`, `confidence`, `validation_source` ("tavily"/"apollo"/"domain"/"llm"), `rejection_reason`.

Pre-validation heuristic rejection patterns: sentences with `?!/;`, listicle titles, questions, non-business entities (temple/church/school), CTA buttons, cookie/GDPR boilerplate. Company names > 60 chars always rejected.

### `search.py` — BM25Search + SearchManager

**`BM25Search`:** offline BM25 over already-fetched articles. Zero API calls. Build once per run:
```python
idx = BM25Search(articles)
hits = idx.search("gold jewellery suppliers Rajkot", top_k=10)
hits = idx.search_companies(segment="fintech startups", geo="Mumbai", top_k=20)
```

**`SearchManager`:** BM25 first, DDG web search fallback:
```python
sm = SearchManager()
results = await sm.search(query, top_k=10)
results = sm.search_articles(query, top_k=15)   # BM25 only, sync
results = await sm.web_search(query, max_results=10)  # DDG only
```

### `run_recorder.py`

Captures step snapshots to `data/recordings/{run_id}/` during real (non-mock) pipeline runs. Used for replay and debugging. `RunRecorder(run_id=run_id)` created in `AgentDeps.create()` when `mock_mode=False` and `run_id` is provided.
