# app/tools/ -- LLM Providers, Embeddings, and API Tools

This package manages all external service integrations: LLM providers with auto-failover, embedding generation with dimension locking, and API tools for contacts and search.

## Provider Fallback Chain

`provider_manager.py` builds pydantic-ai `FallbackModel` instances with cooldown-aware provider ordering. When a provider fails, it enters exponential backoff cooldown:

- **Rate limit / payment error**: 5 minute cooldown
- **Timeout / transient error**: 1 minute cooldown
- **Token bucket**: Per-provider rate limiting (smooths bursts to steady RPM)

Providers are skipped during cooldown. The chain rebuilds automatically when cooldowns expire.

### Provider Chains

| Chain | Method | Use Case |
|-------|--------|----------|
| General | `get_model()` | Text generation, synthesis |
| Structured Output | `get_structured_output_model()` | Pydantic model extraction (skips providers without `tool_choice=required`) |
| Tool Calling | `get_tool_calling_model()` | Agent function calling |
| Lite | `get_lite_model()` | Classification, cheap tasks |

### Available Providers

| Provider | Model | Notes |
|----------|-------|-------|
| GeminiDirect | gemini-2.5-flash-lite | Vertex Express free tier, primary |
| OpenAI | gpt-4.1-mini / nano | Tier 1, structured output |
| VertexLlama | llama-4-scout-17b | Fast tool calling, GCP credits |
| NVIDIA | deepseek-v3.1 | OpenAI-compatible API |
| Groq | qwen-qwen3-32b | Fast inference, free tier |
| OpenRouter | gemini-2.5-flash | Multi-model proxy |
| Ollama | mistral (configurable) | Local, unlimited, last resort |

## LLM Service

`llm_service.py` provides a high-level interface with two tracks:

### Track A: Structured Output

```python
result: MyModel = await llm.generate_structured(
    output_type=MyModel,      # Pydantic model
    prompt="Extract data...",
    system_prompt="You are...",
)
```

Uses pydantic-ai `Agent` with `result_type` set to the Pydantic model. The agent cache keys on `(output_type, system_prompt_hash, cooldown_state)` so FallbackModel rebuilds when providers change state.

### Track B: Text Generation

```python
text: str = await llm.generate_text(
    prompt="Summarize...",
    system_prompt="You are...",
)
```

Returns raw string. Uses the general provider chain (includes VertexLlama).

### JSON Repair

`json_repair.py` handles malformed LLM JSON output: strips markdown fences, fixes trailing commas, repairs truncated arrays, and attempts bracket completion.

## Embedding Providers

`embeddings.py` implements a multi-tier embedding system with dimension locking:

| Tier | Provider | Model | Dimensions |
|------|----------|-------|-----------|
| 1 | NVIDIA NIM | nv-embedqa-e5-v5 | 1024 |
| 2 | OpenAI | text-embedding-3-large | 1024 (dimensioned) |
| 3 | HuggingFace API | BAAI/bge-large-en-v1.5 | 1024 |
| 4 | Local | BAAI/bge-large-en-v1.5 | 1024 |
| 5 | Ollama | nomic-embed-text | 768 (rejected if others ran first) |

### Dimension Locking

The `_locked_dim` mechanism prevents silent dimension mixing. Once the first batch of embeddings is generated, the dimension is locked. Any subsequent provider that returns a different dimension is rejected, and the next fallback is tried. This prevents downstream components (cosine similarity thresholds, ChromaDB collections) from breaking.

### Content-Aware Embedding

Articles are embedded as a composite string: `title + event_description + entities + body_excerpt`. This produces more discriminative vectors than title-only or body-only embedding.

## API Tools

| Module | Service | Purpose |
|--------|---------|---------|
| `apollo_tool.py` | Apollo.io | Contact finding, company enrichment (600/month free) |
| `hunter_tool.py` | Hunter.io | Email verification fallback (25/month free) |
| `rss_tool.py` | 22+ RSS feeds | Parallel news fetching with source bandit ordering |
| `domain_utils.py` | -- | Domain extraction and cleaning |
| `article_cache.py` | ChromaDB | Article embedding cache (avoids re-embedding) |
| `feedback.py` | JSONL | User feedback collection for learning loops |
| `run_recorder.py` | File system | Step snapshots for mock replay (`data/recordings/`) |
| `api_checker.py` | -- | API key validation on startup |
| `provider_health.py` | File system | Provider availability state (`app/data/`) |
