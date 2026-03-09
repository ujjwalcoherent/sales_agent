# app/tools/ — External Integrations

Pure API wrappers and shared utilities. No business logic, no fallback chains within a single tool.

```
tools/
├── crm/                    # CRM & outreach integrations
│   ├── apollo_tool.py      # Contact search + company enrichment (Apollo.io)
│   ├── hunter_tool.py      # Email finding + verification (Hunter.io)
│   └── brevo_tool.py       # Email sending (Brevo/SendinBlue SMTP)
├── web/                    # Web search & news sources
│   ├── tavily_tool.py      # Primary search (7 rotating API keys)
│   ├── rss_tool.py         # 92 active RSS feeds, bandit-ordered
│   ├── web_intel.py        # Multi-source orchestrator (Tavily→DDG→RSS→gnews→scrape)
│   ├── news_collector.py   # Company news: fresh (7d) + historical (1-5mo)
│   └── content_scraper.py  # Full-text extraction via trafilatura
├── llm/                    # LLM providers & embeddings
│   ├── providers.py        # ProviderManager — fallback chain with cooldowns
│   ├── llm_service.py      # generate() / generate_structured() interface
│   ├── embeddings.py       # Multi-tier embedding with dimension locking
│   └── mock_responses.py   # Deterministic mock outputs for tests
└── (root shared utilities)
    ├── company_enricher.py       # Central entity validation + enrichment gatekeeper
    ├── person_intel.py           # Person background scraping for email personalization
    ├── domain_utils.py           # Domain extraction and normalization helpers
    ├── json_repair.py            # LLM JSON output repair (strips fences, fixes truncation)
    ├── search.py                 # BM25Search: keyword + DDG web search
    ├── article_cache.py          # ChromaDB article embedding cache
    ├── event_classifier_tool.py  # Two-tier event classification (embedding + LLM validation)
    ├── feedback_store.py         # JSONL feedback collection for SetFit training
    └── run_recorder.py           # Step snapshots for mock replay (data/recordings/)
```

## Provider Fallback Chain (app/tools/llm/providers.py)

`ProviderManager` builds pydantic-ai `FallbackModel` instances with cooldown-aware ordering.
When a provider fails: rate limit → 5min cooldown, timeout → 1min cooldown, then next in chain.

| Chain | Method | Order |
|-------|--------|-------|
| General | `get_model()` | OpenAI → GeminiDirect → VertexLlama → NVIDIA → Groq → OpenRouter → Ollama |
| Structured output | `get_structured_output_model()` | OpenAI → GeminiDirect → Groq → Ollama |
| Lite | `get_lite_model()` | OpenAI Nano → GeminiDirectLite → Groq → standard chain |

| Provider | Model | Notes |
|----------|-------|-------|
| OpenAI | gpt-4.1-mini / nano | Primary — structured output + tool calling |
| GeminiDirect | gemini-2.5-flash-lite | Fast, GCP free credits |
| VertexLlama | llama-4-scout-17b | GCP credits, fast tool calling |
| NVIDIA | deepseek-v3.1 | OpenAI-compatible NIM API |
| Groq | qwen-qwen3-32b | Fast inference, free tier |
| OpenRouter | gemini-2.5-flash | Multi-model proxy fallback |
| Ollama | mistral (configurable) | Local, last resort |

## LLM Service (app/tools/llm/llm_service.py)

```python
from app.tools.llm.llm_service import LLMService

llm = LLMService()

# Text generation
text: str = await llm.generate(
    prompt="Summarize this article...",
    system_prompt="You are a B2B analyst.",
    temperature=0.3,
    max_tokens=500,
)

# Structured output (returns validated Pydantic model)
result: MyModel = await llm.run_structured(
    prompt="Extract lead data from...",
    output_type=MyModel,
    temperature=0.3,
)
```

**Important:** `generate()` does NOT accept `model_tier` kwarg. Full signatures:
```python
generate(prompt, system_prompt=None, temperature=0.7, max_tokens=2000, json_mode=False)
run_structured(prompt, system_prompt="", output_type=str, retries=2, temperature=0.3, reflect_retries=2)
generate_json(prompt, system_prompt=None, schema_hint=None, pydantic_model=None, required_keys=None, max_retries=None)
```

Agent instances are cached on `(output_type, system_prompt_hash, cooldown_state)` — rebuilds when provider states change.

## Embeddings (app/tools/llm/embeddings.py)

Dimension-locked multi-tier embeddings. Once first batch completes, dimension is locked — subsequent providers returning different dimensions are silently rejected.

Provider fallback order depends on `EMBEDDING_PROVIDER` env var:

| Mode (`EMBEDDING_PROVIDER`) | Fallback chain |
|-----------------------------|----------------|
| `nvidia` (default) | NVIDIA → OpenAI → HuggingFace API → Local → Ollama |
| `openai` | OpenAI only (no fallback) |
| `api` | HuggingFace API → NVIDIA → OpenAI → Ollama → Local |
| `local` | Local → NVIDIA → OpenAI → HuggingFace API → Ollama |

| Provider | Model | Dimensions |
|----------|-------|-----------|
| NVIDIA NIM | nv-embedqa-e5-v5 | 1024 |
| OpenAI | text-embedding-3-large | 1024 (Matryoshka @1024) |
| HuggingFace API | BAAI/bge-large-en-v1.5 | 1024 |
| Local | BAAI/bge-large-en-v1.5 | 1024 |
| Ollama | nomic-embed-text | 768 (rejected if higher-tier ran first) |

## Import Paths

Always use sub-package paths. Never import at module level (lazy load required).

```python
from app.tools.crm.apollo_tool import ApolloTool
from app.tools.crm.hunter_tool import HunterTool
from app.tools.crm.brevo_tool import BrevoTool
from app.tools.web.tavily_tool import TavilyTool
from app.tools.web.rss_tool import RSSTool
from app.tools.web.web_intel import WebIntelTool
from app.tools.web.news_collector import NewsCollector
from app.tools.llm.providers import ProviderManager
from app.tools.llm.llm_service import LLMService
from app.tools.llm.embeddings import EmbeddingTool
from app.tools.company_enricher import enrich, enrich_batch
from app.tools.search import SearchManager
```

## Concurrency Limits

All external APIs bounded by `asyncio.Semaphore`:
- Apollo: 3 concurrent requests
- Hunter: 2 concurrent requests
- Tavily: 5 concurrent requests (across 7 rotating keys)
- LLM calls: 10 concurrent (managed by pydantic-ai FallbackModel)
