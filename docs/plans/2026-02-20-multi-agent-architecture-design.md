# Multi-Agent Architecture Design

**Date:** 2026-02-20
**Status:** Proposed
**Scope:** Rearchitect sales intelligence pipeline from linear processor chain into autonomous multi-agent system

---

## Problem Statement

The current pipeline (`app/agents/orchestrator.py`) is a **linear LangGraph graph** where each node is a deterministic processor. Despite being in an `agents/` directory, nothing is actually an agent:

- **ImpactAnalyzer** — deterministic processor, zero autonomy
- **CompanyDiscovery** — hardcoded search pipeline, no tool selection
- **ContactFinder** — sequential API calls, no reasoning
- **EmailGenerator** — template fill, no iteration
- **AI Council** — single LLM calls dressed as "multi-agent debate"

The pipeline needs TRUE agents that: reason about what to investigate, select their own tools, communicate findings, loop on quality checks, and improve from feedback.

---

## Approach: Hybrid pydantic-ai Agents inside LangGraph Orchestration

After researching LangGraph Swarm, pydantic-ai delegation, CrewAI, AutoGen, OpenAI Agents SDK, and Claude Agent SDK, the recommended architecture is:

| Layer | Framework | Why |
|-------|-----------|-----|
| **Orchestration** | LangGraph `StateGraph` | Conditional routing, retry loops, checkpointing, human-in-the-loop |
| **Agent Runtime** | pydantic-ai `Agent` | Typed tool calls, `RunContext` deps, `ModelRetry` self-correction, output validators |
| **Vector Memory** | ChromaDB | Article cache (1024-dim cosine), trend memory (centroid matching), semantic search |
| **Learning** | Thompson Sampling bandits | Source quality (SourceBandit), company relevance (CompanyRelevanceBandit), DPO weight learning |

**Why not pure LangGraph agents?** LangGraph excels at graph orchestration but has no native typed tool-calling runtime. You'd write raw JSON function schemas.

**Why not pure pydantic-ai?** pydantic-ai agents are autonomous but have no built-in graph orchestration, conditional routing, or checkpointing across agents.

**Why not CrewAI/AutoGen?** Heavy frameworks with opinionated abstractions that fight our existing codebase. We already have 70% of what we need.

**Hybrid = best of both:** LangGraph routes tasks between agents, handles retries, manages shared state. pydantic-ai agents reason autonomously within their assigned scope, calling tools via `@agent.tool` decorators.

---

## Agent Topology

```
                    ┌─────────────────────────┐
                    │   Research Director      │
                    │   (LangGraph Supervisor) │
                    └─────┬───────────────┬────┘
                          │               │
              ┌───────────┴──┐     ┌──────┴───────────┐
              │ Source Intel  │     │  Analysis Agent   │
              │    Agent      │     │  (Trend Pipeline) │
              └──────┬───────┘     └──────┬────────────┘
                     │                    │
              ┌──────┴───────┐     ┌──────┴────────────┐
              │ Market Impact │     │   Lead Gen Agent   │
              │    Agent      │     │ (Company+Contact)  │
              └──────┬───────┘     └──────┬────────────┘
                     │                    │
                     └────────┬───────────┘
                              │
                     ┌────────┴────────┐
                     │  Quality Agent   │
                     │ (Gate + Feedback)│
                     └─────────────────┘
```

### Agent Specifications

#### 1. Source Intel Agent
**Role:** Autonomous data acquisition from RSS feeds, web search, and cached articles.

**pydantic-ai Agent with tools:**
- `@tool fetch_rss(sources, hours_ago, max_per_source)` — wraps `RSSTool.fetch_all_sources()` with source bandit selection
- `@tool search_web(query, max_results)` — wraps `TavilyTool.search()` for targeted investigation
- `@tool check_cache(query, n_results)` — ChromaDB similarity search on ArticleCache
- `@tool scrape_article(url)` — wraps `trafilatura.extract()` for full content
- `@tool embed_articles(articles)` — wraps `EmbeddingTool.embed_batch()` for vector generation
- `@tool classify_events(articles)` — wraps `EmbeddingEventClassifier` for event tagging

**Autonomous decisions:**
- Which sources to prioritize (Thompson Sampling posterior from SourceBandit)
- Whether to do supplementary web search based on RSS coverage gaps
- How many articles to fetch per source based on source quality estimates
- Whether to use cached articles vs. fresh fetch

**Output:** `List[NewsArticle]` with embeddings, entities, event types

#### 2. Analysis Agent
**Role:** Cluster articles into trends, compute signals, build hierarchy.

**pydantic-ai Agent with tools:**
- `@tool run_clustering(articles, embeddings, params)` — Leiden community detection with Optuna optimization
- `@tool compute_coherence(cluster)` — title + entity + embedding coherence scoring
- `@tool extract_signals(cluster)` — composite signal scoring (entity, market, source, composite)
- `@tool check_trend_memory(centroid)` — ChromaDB trend memory for novelty/continuity
- `@tool build_hierarchy(clusters)` — subclustering + tree builder
- `@tool compute_correlations(clusters)` — entity bridge correlation edges

**Autonomous decisions:**
- Accept/reject clusters based on coherence thresholds (using adaptive EMA from history)
- Decide if subclustering is needed (cluster size > threshold)
- Retry clustering with different Leiden resolution if quality is poor
- Merge related sub-trends based on semantic similarity

**Output:** `List[TrendData]` with signals, hierarchy, correlations

#### 3. Market Impact Agent
**Role:** Analyze business implications of each trend with multi-perspective reasoning.

**pydantic-ai Agent with tools:**
- `@tool analyze_impact(trend, perspective)` — structured LLM analysis from specific viewpoint
- `@tool search_precedent(trend_title)` — Tavily search for historical precedents
- `@tool assess_sectors(trend)` — sector-specific impact scoring
- `@tool synthesize_perspectives(perspectives)` — moderator synthesis of 4 viewpoints
- `@tool check_causal_links(trend_a, trend_b)` — causal council edge analysis

**Autonomous decisions:**
- Which perspectives are most relevant per trend type (regulatory → prioritize Compliance viewpoint)
- Whether to search for precedents (high-confidence trends might not need it)
- How deep to go on sector analysis based on trend specificity
- Whether to trigger causal reasoning across trend pairs

**Output:** `List[ImpactAnalysis]` with confidence, sectors, causal graph

#### 4. Lead Gen Agent
**Role:** Discover companies and contacts affected by each trend.

**pydantic-ai Agent with tools:**
- `@tool find_companies(trend, industry)` — wraps CompanyDiscovery with NER-based search
- `@tool verify_company(name)` — Wikipedia/web verification for hallucination guard
- `@tool find_contacts(company, roles)` — Apollo + Hunter enrichment
- `@tool assess_relevance(company, trend)` — company relevance bandit scoring
- `@tool generate_pitch(trend, company, contact)` — personalized email generation

**Autonomous decisions:**
- Search strategy: NER-based vs. industry-based vs. hybrid per trend
- How many companies to surface per trend (quality over quantity)
- Which roles to target based on trend type (tech trends → CTO, regulatory → GC)
- Whether a company-trend fit is strong enough to pursue (relevance bandit threshold)

**Output:** `List[CompanyData]` with contacts, emails, relevance scores

#### 5. Quality Agent
**Role:** Gate keeper that validates outputs at every stage and routes feedback.

**pydantic-ai Agent with tools:**
- `@tool validate_trend(trend)` — Council Stage A: cluster quality, event type, hierarchy
- `@tool validate_impact(impact)` — confidence-based quality scoring
- `@tool validate_lead(company, trend)` — Council Stage C: company-trend fit
- `@tool record_feedback(type, item_id, rating, signals)` — JSONL feedback storage
- `@tool check_quality_bounds(metrics)` — compare against QUALITY_BOUNDS thresholds

**Autonomous decisions:**
- Accept/reject/retry at each stage based on confidence thresholds
- Route feedback to appropriate bandits (source, company, weight learner)
- Decide if the entire pipeline output meets minimum quality bar
- Flag anomalies (sudden quality drops, unusual source behavior)

**Output:** Quality gates (pass/fail), feedback records, retry signals

#### 6. Research Director (Supervisor)
**Role:** LangGraph supervisor node that routes tasks between agents.

**Implementation:** LangGraph `StateGraph` with conditional edges:
```
START → source_intel → analysis → [quality_check_1]
  → market_impact → [quality_check_2]
  → lead_gen → [quality_check_3]
  → END

quality_check_N:
  if PASS → next_agent
  if RETRY → back_to_current_agent (max 2 retries)
  if FAIL → skip_to_next (with degraded output)
```

**Shared State (GraphState):**
```python
class GraphState(TypedDict):
    # Shared deps
    deps: PipelineDeps

    # Data flowing between agents
    articles: List[NewsArticle]
    embeddings: List[List[float]]
    trends: List[TrendData]
    impacts: List[ImpactAnalysis]
    companies: List[CompanyData]
    contacts: List[ContactData]
    emails: List[OutreachEmail]

    # Quality tracking
    quality_scores: Dict[str, float]
    retry_counts: Dict[str, int]
    errors: Annotated[List[str], operator.add]

    # Causal graph (cross-trend)
    causal_edges: List[CausalEdgeResult]
    cascade_narratives: List[CascadeNarrative]
```

---

## Tool Landscape (Full Utilization)

Every existing tool gets exposed as an `@agent.tool` on the appropriate agent:

| Tool | Current State | Agent Assignment | Enhancement |
|------|--------------|-----------------|-------------|
| `RSSTool` | Called once at pipeline start | Source Intel | Bandit-weighted source selection |
| `TavilyTool` | Called in company search only | Source Intel + Market Impact + Lead Gen | Shared across 3 agents |
| `EmbeddingTool` | Called once for batch embed | Source Intel + Analysis | On-demand embedding for web search results |
| `ArticleCache` (ChromaDB) | Store/load only | Source Intel | Semantic similarity search for gap detection |
| `TrendMemory` (ChromaDB) | Novelty scoring only | Analysis | Continuity tracking + stale trend pruning |
| `LLMService` | Generic prompt → response | All agents (via pydantic-ai) | Typed structured output per agent |
| `SourceBandit` | Updated post-run | Source Intel + Quality | Real-time source prioritization |
| `CompanyRelevanceBandit` | Updated from feedback | Lead Gen + Quality | Contextual company scoring |
| `WeightLearner` | Reads JSONL on next run | Quality | Online weight updates from feedback |
| `EventClassifier` | One-shot classification | Source Intel | Event-type-aware article routing |
| `EntityExtractor` (spaCy) | NER in ingest | Source Intel + Lead Gen | Entity-based company discovery |
| `EntityNormalizer` | Fuzzy dedup | Analysis | Cross-article entity resolution |

### ChromaDB Collections (Vector Memory)
1. **`articles`** — ArticleCache: 1024-dim embeddings, cosine HNSW, dedup by article ID
2. **`trend_centroids`** — TrendMemory: cluster centroids for novelty/continuity, EMA blending
3. **Future: `company_profiles`** — Company embeddings for similarity-based lead expansion

---

## Self-Learning Loops

The pipeline already has 4 learning subsystems. The multi-agent architecture makes them more effective:

### 1. Source Bandit (Thompson Sampling)
- **Current:** Updated once after full pipeline run
- **Multi-agent:** Source Intel Agent queries `get_adaptive_credibility()` BEFORE selecting sources, uses `update_from_run()` AFTER clustering quality is known. Tighter feedback loop.

### 2. Company Relevance Bandit
- **Current:** Updated from manual feedback only
- **Multi-agent:** Lead Gen Agent queries bandit for relevance priors. Quality Agent auto-propagates validation results as implicit feedback.

### 3. DPO Weight Learner
- **Current:** Reads JSONL feedback file on next run
- **Multi-agent:** Quality Agent writes feedback in real-time. Analysis Agent queries current weights before signal computation.

### 4. Adaptive EMA Thresholds
- **Current:** Computed from pipeline_metrics history
- **Multi-agent:** Analysis Agent uses adaptive thresholds for clustering decisions. Quality Agent monitors drift and flags anomalies.

---

## What Changes vs. Current Architecture

| Component | Current | New |
|-----------|---------|-----|
| `app/agents/orchestrator.py` | Linear LangGraph pipeline | Supervisor graph with conditional edges + retry loops |
| `app/agents/deps.py` | `PipelineDeps` dataclass | `AgentToolkit` with `@agent.tool` registration |
| `app/agents/impact_agent.py` | Deterministic processor | pydantic-ai Agent with web search + multi-perspective tools |
| `app/agents/company_agent.py` | Hardcoded search pipeline | pydantic-ai Agent with autonomous search strategy |
| `app/agents/contact_agent.py` | Sequential API calls | Tool on Lead Gen Agent |
| `app/agents/email_agent.py` | Template fill | Tool on Lead Gen Agent |
| `app/agents/council/` | Single LLM calls | Tools on Quality Agent + Market Impact Agent |
| `app/trends/engine.py` | Monolithic 6-layer pipeline | Tools on Analysis Agent (layers become callable tools) |

### What Stays the Same
- All schemas (`app/schemas/`) — unchanged
- All learning subsystems (bandits, weight learner, trend memory)
- ChromaDB collections (article cache, trend memory)
- RSS/Tavily/Apollo/Hunter tool implementations
- Streamlit UI (`streamlit_app.py`)
- Config system (`app/config.py`)
- News processing (entity extraction, event classification, scraping)
- Leiden clustering, coherence scoring, signal computation
- Stress test and calibration infrastructure

**Estimated reuse: ~70% of existing codebase.**

---

## Implementation Phases

### Phase 1: Agent Toolkit + Source Intel Agent
- Create `AgentToolkit` (replaces `PipelineDeps`) with tool registration
- Build Source Intel Agent with RSS, Tavily, cache, embed, classify tools
- Wire source bandit into tool-level selection
- Test: Source Intel produces same quality articles as current `_layer_ingest()`

### Phase 2: Analysis Agent + Market Impact Agent
- Build Analysis Agent wrapping TrendPipeline layers as tools
- Build Market Impact Agent with multi-perspective analysis tools
- Wire trend memory and adaptive thresholds
- Test: Analysis + Impact produce comparable quality to current pipeline

### Phase 3: Lead Gen + Quality Agent + Full Supervisor
- Build Lead Gen Agent with company/contact/email tools
- Build Quality Agent with validation + feedback tools
- Build Research Director supervisor graph with retry loops
- Wire all learning loops (source bandit, company bandit, weight learner)
- Integration test: Full pipeline end-to-end

### Phase 4: Stress Test + Optimization
- Run stress test on 500+ articles with multi-agent pipeline
- Compare quality metrics against linear pipeline baseline
- Optimize agent system prompts based on quality analysis
- Fine-tune retry thresholds and quality gates

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Agent calls too many LLM API calls | Budget caps per agent per run (configurable in `.env`) |
| Quality regression vs. linear pipeline | Phase-by-phase comparison with stress test baseline |
| pydantic-ai tool overhead | Tools are thin wrappers — actual logic stays in existing modules |
| Agent loops forever on retry | Max retry count (2) with degraded-output fallback |
| Increased latency from autonomy | Parallel agent execution where data-independent (Source Intel ‖ initial cache check) |

---

## Success Criteria

1. **Autonomy:** Agents make observable autonomous decisions (logged tool selections differ across runs)
2. **Quality:** Composite quality score >= 0.45 on stress test (matching or exceeding linear baseline)
3. **Learning:** Source bandit posteriors update within the run (not just post-run)
4. **Resilience:** Pipeline completes even when individual agents fail (degraded output, not crash)
5. **Transparency:** Every agent decision is logged with reasoning (explainable AI)
