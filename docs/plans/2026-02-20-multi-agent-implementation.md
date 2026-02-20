# Multi-Agent Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the linear LangGraph pipeline with autonomous pydantic-ai agents that reason about tool selection, inside a LangGraph supervisor graph with quality-gate retry loops.

**Architecture:** 5 pydantic-ai agents (Source Intel, Analysis, Market Impact, Lead Gen, Quality) each with `@agent.tool`-decorated functions wrapping existing tools. A LangGraph `StateGraph` supervisor routes between agents with conditional edges for retry. Shared `AgentDeps` dataclass passed via `RunContext`.

**Tech Stack:** pydantic-ai (Agent, RunContext, ModelRetry, @agent.tool), LangGraph (StateGraph, conditional_edges), ChromaDB (ArticleCache, TrendMemory), Thompson Sampling bandits

**Design doc:** `docs/plans/2026-02-20-multi-agent-architecture-design.md`

---

## Phase 1: Foundation — AgentDeps + Source Intel Agent

### Task 1: Create AgentDeps dataclass

Extend `PipelineDeps` into a pydantic-ai-compatible deps dataclass that carries all shared tools.

**Files:**
- Create: `app/agents/agent_deps.py`

**Step 1: Create the deps file**

```python
# app/agents/agent_deps.py
"""
Shared dependency container for pydantic-ai agents.

Extends the lazy-initialized tool pattern from PipelineDeps.
Every agent receives this via RunContext[AgentDeps] and accesses
tools through properties (lazy init on first use).

This replaces PipelineDeps for the multi-agent architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    """Shared dependencies for all pydantic-ai agents.

    Passed as deps_type to each Agent. Tools access via ctx.deps.
    Lazy initialization means tools are only created when first used.
    """

    mock_mode: bool = False
    log_callback: Optional[object] = field(default=None, repr=False)

    # Lazy-initialized tools
    _llm_service: Optional[object] = field(default=None, repr=False)
    _llm_lite_service: Optional[object] = field(default=None, repr=False)
    _tavily_tool: Optional[object] = field(default=None, repr=False)
    _rss_tool: Optional[object] = field(default=None, repr=False)
    _apollo_tool: Optional[object] = field(default=None, repr=False)
    _hunter_tool: Optional[object] = field(default=None, repr=False)
    _embedding_tool: Optional[object] = field(default=None, repr=False)
    _article_cache: Optional[object] = field(default=None, repr=False)
    _source_bandit: Optional[object] = field(default=None, repr=False)
    _company_bandit: Optional[object] = field(default=None, repr=False)

    @classmethod
    def create(cls, mock_mode: bool = False, log_callback=None) -> AgentDeps:
        """Create deps with settings-aware mock_mode."""
        from app.config import get_settings
        settings = get_settings()
        return cls(
            mock_mode=mock_mode or settings.mock_mode,
            log_callback=log_callback,
        )

    def _log(self, msg: str, level: str = "info"):
        """Log to both logger and optional UI callback."""
        getattr(logger, level, logger.info)(msg)
        if self.log_callback:
            try:
                self.log_callback(msg, level)
            except Exception:
                pass

    # ── Tool properties (lazy init) ──────────────────────────────────

    @property
    def llm_service(self):
        if self._llm_service is None:
            from app.tools.llm_service import LLMService
            self._llm_service = LLMService(mock_mode=self.mock_mode)
        return self._llm_service

    @property
    def llm_lite_service(self):
        if self._llm_lite_service is None:
            from app.tools.llm_service import LLMService
            self._llm_lite_service = LLMService(mock_mode=self.mock_mode, lite=True)
        return self._llm_lite_service

    @property
    def tavily_tool(self):
        if self._tavily_tool is None:
            from app.tools.tavily_tool import TavilyTool
            self._tavily_tool = TavilyTool(mock_mode=self.mock_mode)
        return self._tavily_tool

    @property
    def rss_tool(self):
        if self._rss_tool is None:
            from app.tools.rss_tool import RSSTool
            self._rss_tool = RSSTool(mock_mode=self.mock_mode)
        return self._rss_tool

    @property
    def apollo_tool(self):
        if self._apollo_tool is None:
            from app.tools.apollo_tool import ApolloTool
            self._apollo_tool = ApolloTool(mock_mode=self.mock_mode)
        return self._apollo_tool

    @property
    def hunter_tool(self):
        if self._hunter_tool is None:
            from app.tools.hunter_tool import HunterTool
            self._hunter_tool = HunterTool(mock_mode=self.mock_mode)
        return self._hunter_tool

    @property
    def embedding_tool(self):
        if self._embedding_tool is None:
            from app.tools.embeddings import EmbeddingTool
            self._embedding_tool = EmbeddingTool()
        return self._embedding_tool

    @property
    def article_cache(self):
        if self._article_cache is None:
            from app.tools.article_cache import ArticleCache
            self._article_cache = ArticleCache()
        return self._article_cache

    @property
    def source_bandit(self):
        if self._source_bandit is None:
            from app.trends.source_bandit import SourceBandit
            self._source_bandit = SourceBandit()
        return self._source_bandit

    @property
    def company_bandit(self):
        if self._company_bandit is None:
            from app.agents.company_relevance_bandit import CompanyRelevanceBandit
            self._company_bandit = CompanyRelevanceBandit()
        return self._company_bandit
```

**Step 2: Verify import works**

Run: `python -c "from app.agents.agent_deps import AgentDeps; d = AgentDeps.create(); print(f'AgentDeps OK, mock={d.mock_mode}')"`

**Step 3: Commit**

```bash
git add app/agents/agent_deps.py
git commit -m "feat: AgentDeps dataclass for pydantic-ai RunContext"
```

---

### Task 2: Create Source Intel Agent

The first pydantic-ai agent with `@agent.tool` functions wrapping RSSTool, TavilyTool, EmbeddingTool, ArticleCache.

**Files:**
- Create: `app/agents/source_intel_agent.py`

**Step 1: Create the agent file**

```python
# app/agents/source_intel_agent.py
"""
Source Intel Agent — autonomous data acquisition.

pydantic-ai agent that decides which sources to query, whether to
supplement with web search, and how to prioritize articles based on
source bandit quality estimates.

Tools:
  - fetch_rss: Fetch articles from RSS sources (bandit-weighted)
  - search_web: Tavily web search for targeted investigation
  - scrape_and_embed: Scrape full content + generate embeddings
  - classify_events: Tag articles with event types
  - check_source_quality: Query source bandit for quality estimates
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.agent_deps import AgentDeps
from app.config import get_settings
from app.schemas.news import NewsArticle

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────

class SourceIntelResult(BaseModel):
    """Structured output from the Source Intel Agent."""
    articles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized NewsArticle dicts with embeddings",
    )
    total_fetched: int = 0
    total_after_dedup: int = 0
    sources_used: List[str] = Field(default_factory=list)
    web_searches_performed: int = 0
    event_distribution: Dict[str, int] = Field(default_factory=dict)
    reasoning: str = Field(
        default="",
        description="Agent's reasoning about source selection and data quality",
    )


# ── System prompt ─────────────────────────────────────────────────────

SOURCE_INTEL_PROMPT = """\
You are the Source Intel Agent for a sales intelligence pipeline focused on \
Indian market trends. Your job is to collect high-quality, recent news articles \
for trend detection.

WORKFLOW:
1. First, check source quality estimates to know which RSS sources are best.
2. Fetch articles from RSS feeds. Prioritize high-quality sources.
3. Scrape full content and generate embeddings for all articles.
4. Classify articles by event type (regulation, funding, technology, etc.).
5. If total articles < 50, supplement with targeted web searches on topics \
   that appear under-covered.
6. Return the complete article set with embeddings and event tags.

DECISION GUIDELINES:
- Always fetch from RSS first — it's the primary data source.
- Use web search only to fill gaps (e.g., too few articles on a sector).
- Quality over quantity: 80 well-scraped articles > 200 headline-only stubs.
- Report your reasoning about source selection and any gaps found.
"""


# ── Agent definition ──────────────────────────────────────────────────

source_intel_agent = Agent(
    'google-gla:gemini-2.5-flash',  # Fast, cheap model for orchestration
    deps_type=AgentDeps,
    output_type=SourceIntelResult,
    system_prompt=SOURCE_INTEL_PROMPT,
    retries=2,
)


# ── Tools ─────────────────────────────────────────────────────────────

@source_intel_agent.tool
async def fetch_rss_articles(
    ctx: RunContext[AgentDeps],
    max_per_source: int = 10,
    hours_ago: int = 24,
) -> str:
    """Fetch articles from RSS news sources.

    Uses source bandit quality estimates to allocate more articles
    to higher-quality sources. Returns summary of what was fetched.

    Args:
        max_per_source: Maximum articles per RSS source (default 10).
        hours_ago: Only include news from last N hours (default 24).
    """
    settings = get_settings()
    rss = ctx.deps.rss_tool

    articles = await rss.fetch_all_sources(
        max_per_source=max_per_source or settings.rss_max_per_source,
        hours_ago=hours_ago or settings.rss_hours_ago,
    )

    # Store articles on deps for later tools to access
    if not hasattr(ctx.deps, '_articles'):
        ctx.deps._articles = []
    ctx.deps._articles = articles

    # Count by source
    source_counts = {}
    for a in articles:
        sid = getattr(a, 'source_id', 'unknown')
        source_counts[sid] = source_counts.get(sid, 0) + 1

    top_sources = sorted(source_counts.items(), key=lambda x: -x[1])[:10]
    summary = f"Fetched {len(articles)} articles from {len(source_counts)} sources.\n"
    summary += "Top sources: " + ", ".join(f"{s}({n})" for s, n in top_sources)
    return summary


@source_intel_agent.tool
async def search_web(
    ctx: RunContext[AgentDeps],
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web for additional articles on a specific topic.

    Use this when RSS coverage has gaps on a particular sector or topic.

    Args:
        query: Search query (e.g., "Indian EV regulation 2026").
        max_results: Max results to return (default 5).
    """
    result = await ctx.deps.tavily_tool.search(
        query=query,
        max_results=max_results,
        include_answer=True,
    )

    results = result.get("results", [])
    answer = result.get("answer", "")

    # Convert web results to NewsArticle format and add to article pool
    from uuid import uuid4
    from datetime import datetime, timezone
    from app.schemas.news import NewsArticle

    new_articles = []
    for r in results:
        article = NewsArticle(
            id=uuid4(),
            title=r.get("title", ""),
            summary=r.get("content", "")[:500],
            url=r.get("url", ""),
            source_id="tavily_web_search",
            source_name="Web Search",
            published_at=datetime.now(timezone.utc),
            full_content=r.get("content", ""),
        )
        new_articles.append(article)

    if hasattr(ctx.deps, '_articles'):
        ctx.deps._articles.extend(new_articles)
    else:
        ctx.deps._articles = new_articles

    summary = f"Web search for '{query}': {len(results)} results."
    if answer:
        summary += f"\nSummary: {answer[:200]}"
    return summary


@source_intel_agent.tool
async def scrape_and_embed_articles(ctx: RunContext[AgentDeps]) -> str:
    """Scrape full content and generate embeddings for all collected articles.

    Call this after fetch_rss_articles (and optionally search_web).
    Populates full_content via trafilatura and generates 1024-dim embeddings.
    """
    articles = getattr(ctx.deps, '_articles', [])
    if not articles:
        return "No articles to process."

    # Scrape full content
    from app.news.scraper import scrape_articles
    enriched_count = await scrape_articles(articles)

    # Generate embeddings
    texts = []
    for a in articles:
        parts = [a.title]
        if a.full_content:
            parts.append(a.full_content[:1000])
        elif a.summary:
            parts.append(a.summary)
        texts.append(" ".join(parts))

    embeddings = await ctx.deps.embedding_tool.embed_batch(texts)
    ctx.deps._embeddings = embeddings

    content_rate = sum(1 for a in articles if a.full_content) / max(len(articles), 1)
    return (
        f"Processed {len(articles)} articles: "
        f"{enriched_count} scraped, "
        f"{len(embeddings)} embedded ({len(embeddings[0]) if embeddings else 0}-dim), "
        f"content fill rate: {content_rate:.0%}"
    )


@source_intel_agent.tool
async def classify_article_events(ctx: RunContext[AgentDeps]) -> str:
    """Classify all articles by event type (regulation, funding, technology, etc.).

    Assigns event_type to each article using the embedding-based classifier.
    Call after scrape_and_embed_articles.
    """
    articles = getattr(ctx.deps, '_articles', [])
    embeddings = getattr(ctx.deps, '_embeddings', [])
    if not articles or not embeddings:
        return "No articles/embeddings to classify."

    from app.news.event_classifier import EmbeddingEventClassifier
    classifier = EmbeddingEventClassifier()
    event_counts = classifier.classify_batch(articles, embeddings)

    ctx.deps._event_distribution = dict(event_counts)
    return f"Event classification: {dict(event_counts)}"


@source_intel_agent.tool
async def check_source_quality(ctx: RunContext[AgentDeps]) -> str:
    """Check source bandit quality estimates for all known sources.

    Returns quality rankings so you can prioritize high-quality sources.
    """
    estimates = ctx.deps.source_bandit.get_quality_estimates()
    if not estimates:
        return "No source quality data yet (first run). All sources treated equally."

    sorted_sources = sorted(estimates.items(), key=lambda x: -x[1])
    lines = [f"Source quality estimates ({len(estimates)} sources):"]
    for sid, quality in sorted_sources[:15]:
        lines.append(f"  {sid}: {quality:.3f}")
    return "\n".join(lines)


# ── Public runner ─────────────────────────────────────────────────────

async def run_source_intel(deps: AgentDeps) -> tuple:
    """Run the Source Intel Agent and return (articles, embeddings).

    This is the entry point called by the LangGraph orchestrator node.
    Returns the raw articles and embeddings for downstream agents.
    """
    settings = get_settings()

    prompt = (
        f"Collect Indian market news articles from the last "
        f"{settings.rss_hours_ago} hours. "
        f"Fetch up to {settings.rss_max_per_source} articles per source. "
        f"Target: at least 50 quality articles with full content."
    )

    try:
        # Get the provider model for the agent
        from app.tools.provider_manager import ProviderManager
        pm = ProviderManager(mock_mode=deps.mock_mode)
        model = pm.get_model()

        result = await source_intel_agent.run(
            prompt,
            deps=deps,
            model=model,
        )

        articles = getattr(deps, '_articles', [])
        embeddings = getattr(deps, '_embeddings', [])

        logger.info(
            f"Source Intel Agent: {len(articles)} articles, "
            f"{len(embeddings)} embeddings, "
            f"reasoning: {result.output.reasoning[:100]}"
        )
        return articles, embeddings, result.output

    except Exception as e:
        logger.error(f"Source Intel Agent failed: {e}")
        # Fallback: direct tool calls (no agent reasoning)
        logger.info("Falling back to direct RSS fetch...")
        articles = await deps.rss_tool.fetch_all_sources(
            max_per_source=settings.rss_max_per_source,
        )
        from app.news.scraper import scrape_articles
        await scrape_articles(articles)
        texts = [f"{a.title} {a.full_content or a.summary or ''}"[:1000] for a in articles]
        embeddings = await deps.embedding_tool.embed_batch(texts)

        fallback_result = SourceIntelResult(
            total_fetched=len(articles),
            total_after_dedup=len(articles),
            reasoning=f"Fallback mode (agent error: {e})",
        )
        return articles, embeddings, fallback_result
```

**Step 2: Verify agent creation works**

Run: `python -c "from app.agents.source_intel_agent import source_intel_agent; print(f'Agent tools: {[t.name for t in source_intel_agent._function_tools.values()]}')"`

**Step 3: Commit**

```bash
git add app/agents/source_intel_agent.py
git commit -m "feat: Source Intel Agent with RSS/web/embed/classify tools"
```

---

### Task 3: Create Analysis Agent

Wraps TrendPipeline layers as pydantic-ai tools. The agent decides clustering parameters, retry strategy, and subclustering depth.

**Files:**
- Create: `app/agents/analysis_agent.py`

**Step 1: Create the agent file**

```python
# app/agents/analysis_agent.py
"""
Analysis Agent — autonomous trend clustering and signal computation.

pydantic-ai agent that wraps the TrendPipeline layers as callable tools.
Decides clustering parameters, evaluates coherence, triggers subclustering,
and checks trend memory for novelty scoring.

Tools:
  - run_trend_pipeline: Execute full TrendPipeline on articles
  - evaluate_cluster_quality: Check coherence and signal quality
  - check_trend_novelty: Query trend memory for novelty scores
  - retry_with_params: Re-run clustering with adjusted parameters
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.agent_deps import AgentDeps
from app.config import get_settings, get_domestic_source_ids

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Structured output from the Analysis Agent."""
    num_clusters: int = 0
    noise_ratio: float = 0.0
    mean_coherence: float = 0.0
    num_trends_passed: int = 0
    num_trends_rejected: int = 0
    novelty_scores: Dict[str, float] = Field(default_factory=dict)
    params_used: Dict[str, float] = Field(default_factory=dict)
    retries_performed: int = 0
    reasoning: str = ""


# ── System prompt ─────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are the Analysis Agent for a sales intelligence pipeline. Your job is to \
cluster news articles into coherent market trends and compute quality signals.

WORKFLOW:
1. Run the trend pipeline on the provided articles and embeddings.
2. Evaluate the cluster quality (coherence, noise ratio, signal scores).
3. If quality is poor (mean coherence < 0.40 or noise ratio > 0.40), \
   retry with adjusted parameters (lower merge threshold or higher coherence min).
4. Check trend memory for novelty — flag recurring vs genuinely new trends.
5. Return the trend tree with quality metrics and your analysis.

QUALITY CRITERIA:
- Mean coherence should be >= 0.40 (acceptable), >= 0.50 (good)
- Noise ratio should be < 0.35 (articles not in any cluster)
- At least 3 meaningful clusters (non-noise)
- Max 2 retries — after that, accept best result

Report your reasoning about cluster quality and any parameter adjustments.
"""


# ── Agent definition ──────────────────────────────────────────────────

analysis_agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=AgentDeps,
    output_type=AnalysisResult,
    system_prompt=ANALYSIS_PROMPT,
    retries=2,
)


# ── Tools ─────────────────────────────────────────────────────────────

@analysis_agent.tool
async def run_trend_pipeline(
    ctx: RunContext[AgentDeps],
    semantic_dedup_threshold: float = 0.88,
    coherence_min: float = 0.48,
    merge_threshold: float = 0.82,
) -> str:
    """Run the full TrendPipeline (Leiden clustering + synthesis) on articles.

    Args:
        semantic_dedup_threshold: Similarity threshold for dedup (0.85-0.92).
        coherence_min: Minimum coherence to keep a cluster (0.35-0.55).
        merge_threshold: Threshold for merging similar clusters (0.70-0.90).
    """
    articles = getattr(ctx.deps, '_articles', [])
    if not articles:
        return "ERROR: No articles available. Source Intel must run first."

    settings = get_settings()
    from app.trends.engine import TrendPipeline

    pipeline = TrendPipeline(
        max_depth=settings.engine_max_depth,
        semantic_dedup_threshold=semantic_dedup_threshold,
        max_concurrent_llm=settings.engine_max_concurrent_llm,
        mock_mode=ctx.deps.mock_mode,
        country=settings.country,
        domestic_source_ids=get_domestic_source_ids(settings.country_code),
    )

    # Override coherence/merge thresholds if non-default
    pipeline._coherence_min_override = coherence_min
    pipeline._merge_threshold_override = merge_threshold

    tree = await pipeline.run(articles)

    # Store results on deps for downstream agents
    ctx.deps._trend_tree = tree
    ctx.deps._pipeline = pipeline
    ctx.deps._params_used = {
        "semantic_dedup_threshold": semantic_dedup_threshold,
        "coherence_min": coherence_min,
        "merge_threshold": merge_threshold,
    }

    major = tree.to_major_trends()
    metrics = getattr(pipeline, '_last_metrics', {})
    noise_ratio = metrics.get('noise_ratio', 0.0)
    mean_coh = metrics.get('mean_coherence', 0.0)

    return (
        f"Pipeline complete: {len(articles)} articles → "
        f"{len(major)} trends, "
        f"noise_ratio={noise_ratio:.2f}, "
        f"mean_coherence={mean_coh:.2f}, "
        f"params: dedup={semantic_dedup_threshold}, "
        f"coh_min={coherence_min}, merge={merge_threshold}"
    )


@analysis_agent.tool
async def evaluate_cluster_quality(ctx: RunContext[AgentDeps]) -> str:
    """Evaluate the quality of current clustering results.

    Returns detailed metrics: coherence distribution, signal scores,
    noise ratio, cluster size distribution.
    """
    tree = getattr(ctx.deps, '_trend_tree', None)
    if tree is None:
        return "ERROR: No trend tree. Run run_trend_pipeline first."

    major = tree.to_major_trends()
    if not major:
        return "No clusters found. All articles are noise."

    coherences = [getattr(m, 'coherence_score', 0.5) for m in major]
    trend_scores = [getattr(m, 'trend_score', 0.5) for m in major]
    article_counts = [getattr(m, 'article_count', 0) for m in major]

    import numpy as np
    lines = [
        f"Clusters: {len(major)}",
        f"Coherence: mean={np.mean(coherences):.3f}, "
        f"min={np.min(coherences):.3f}, max={np.max(coherences):.3f}",
        f"Trend scores: mean={np.mean(trend_scores):.3f}",
        f"Cluster sizes: {sorted(article_counts, reverse=True)[:10]}",
    ]

    # Flag issues
    if np.mean(coherences) < 0.40:
        lines.append("WARNING: Low mean coherence. Consider lowering merge_threshold.")
    if any(c < 0.30 for c in coherences):
        lines.append(f"WARNING: {sum(1 for c in coherences if c < 0.30)} clusters below 0.30 coherence.")

    return "\n".join(lines)


@analysis_agent.tool
async def check_trend_novelty(ctx: RunContext[AgentDeps]) -> str:
    """Check trend memory for novelty scores.

    Compares current cluster centroids against stored history.
    High novelty = genuinely new trend. Low novelty = recurring topic.
    """
    tree = getattr(ctx.deps, '_trend_tree', None)
    if tree is None:
        return "No trend tree available."

    major = tree.to_major_trends()
    novelty_info = []
    for m in major:
        novelty = getattr(m, 'novelty_score', None)
        continuity = getattr(m, 'continuity_run_count', 0)
        novelty_info.append(
            f"  {m.trend_title[:60]}: "
            f"novelty={novelty if novelty is not None else 'N/A'}, "
            f"seen={continuity} times"
        )

    return f"Novelty analysis ({len(major)} trends):\n" + "\n".join(novelty_info)


# ── Public runner ─────────────────────────────────────────────────────

async def run_analysis(deps: AgentDeps) -> Any:
    """Run the Analysis Agent and return the TrendTree.

    Called by the LangGraph orchestrator node.
    """
    articles = getattr(deps, '_articles', [])

    prompt = (
        f"Analyze {len(articles)} news articles. "
        f"Cluster them into market trends using Leiden community detection. "
        f"Target: coherent clusters with mean coherence >= 0.45."
    )

    try:
        from app.tools.provider_manager import ProviderManager
        pm = ProviderManager(mock_mode=deps.mock_mode)
        model = pm.get_model()

        result = await analysis_agent.run(prompt, deps=deps, model=model)

        tree = getattr(deps, '_trend_tree', None)
        logger.info(
            f"Analysis Agent: {result.output.num_clusters} clusters, "
            f"coherence={result.output.mean_coherence:.3f}, "
            f"retries={result.output.retries_performed}"
        )
        return tree, result.output

    except Exception as e:
        logger.error(f"Analysis Agent failed: {e}")
        # Fallback: run pipeline directly
        logger.info("Falling back to direct TrendPipeline...")
        settings = get_settings()
        from app.trends.engine import TrendPipeline

        pipeline = TrendPipeline(
            max_depth=settings.engine_max_depth,
            semantic_dedup_threshold=settings.semantic_dedup_threshold,
            max_concurrent_llm=settings.engine_max_concurrent_llm,
            mock_mode=deps.mock_mode,
            country=settings.country,
            domestic_source_ids=get_domestic_source_ids(settings.country_code),
        )
        tree = await pipeline.run(articles)
        deps._trend_tree = tree

        fallback_result = AnalysisResult(
            reasoning=f"Fallback mode (agent error: {e})",
        )
        return tree, fallback_result
```

**Step 2: Verify**

Run: `python -c "from app.agents.analysis_agent import analysis_agent; print(f'Tools: {[t.name for t in analysis_agent._function_tools.values()]}')"`

**Step 3: Commit**

```bash
git add app/agents/analysis_agent.py
git commit -m "feat: Analysis Agent with clustering/quality/novelty tools"
```

---

### Task 4: Create Market Impact Agent

Multi-perspective impact analysis with web search for precedents.

**Files:**
- Create: `app/agents/market_impact_agent.py`

**Step 1: Create the agent file**

```python
# app/agents/market_impact_agent.py
"""
Market Impact Agent — multi-perspective business impact analysis.

pydantic-ai agent that analyzes each trend from 4 perspectives
(industry analyst, strategy consultant, risk analyst, market researcher),
searches for historical precedents, and synthesizes a consensus view.

Tools:
  - analyze_trend_impact: Run AI council on a single trend
  - search_precedent: Web search for similar historical events
  - analyze_all_trends: Batch-process all trends with parallel analysis
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.agent_deps import AgentDeps
from app.schemas.sales import TrendData, ImpactAnalysis

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────

class ImpactResult(BaseModel):
    """Structured output from the Market Impact Agent."""
    total_trends_analyzed: int = 0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    precedent_searches: int = 0
    cross_trend_insights: List[str] = Field(default_factory=list)
    reasoning: str = ""


# ── System prompt ─────────────────────────────────────────────────────

IMPACT_PROMPT = """\
You are the Market Impact Agent for a sales intelligence pipeline targeting \
Indian market consulting opportunities.

WORKFLOW:
1. Analyze all trends for business impact using the AI council tool.
2. For high-impact trends (severity=high/critical), search for historical \
   precedents to strengthen the analysis.
3. Identify cross-trend compound opportunities (when 2+ trends affect the \
   same sector simultaneously).
4. Return impact analysis with confidence scores and consulting recommendations.

CONSULTING CONTEXT:
You work for Coherent Market Insights (CMI), a consulting firm. Focus on:
- Which companies NEED CMI's services because of this trend
- What specific pain points does this trend create
- Who are the decision makers who would buy consulting
- What's the urgency (is this a 1-week or 6-month buying cycle)

Be specific and evidence-based. Avoid generic analysis.
"""


# ── Agent definition ──────────────────────────────────────────────────

impact_agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=AgentDeps,
    output_type=ImpactResult,
    system_prompt=IMPACT_PROMPT,
    retries=2,
)


# ── Tools ─────────────────────────────────────────────────────────────

@impact_agent.tool
async def analyze_all_trends(ctx: RunContext[AgentDeps]) -> str:
    """Analyze business impact of all detected trends via AI council.

    Runs parallel impact analysis using the existing ImpactAnalyzer.
    Each trend gets 4-perspective analysis + moderator synthesis.
    """
    from app.agents.impact_agent import ImpactAnalyzer
    from app.schemas import AgentState

    trends = getattr(ctx.deps, '_trend_data', [])
    if not trends:
        return "ERROR: No trends to analyze."

    analyzer = ImpactAnalyzer(
        mock_mode=ctx.deps.mock_mode,
        deps=ctx.deps,
        log_callback=ctx.deps.log_callback,
    )

    state = AgentState(trends=trends)
    result_state = await analyzer.analyze_impacts(state)

    ctx.deps._impacts = result_state.impacts or []

    high_conf = sum(1 for i in ctx.deps._impacts if i.council_confidence >= 0.6)
    low_conf = sum(1 for i in ctx.deps._impacts if i.council_confidence < 0.4)

    return (
        f"Analyzed {len(trends)} trends → {len(ctx.deps._impacts)} impacts. "
        f"High confidence: {high_conf}, Low confidence: {low_conf}. "
        f"Errors: {len(result_state.errors)}"
    )


@impact_agent.tool
async def search_precedent(
    ctx: RunContext[AgentDeps],
    trend_title: str,
) -> str:
    """Search for historical precedents of a specific trend.

    Use for high-impact trends to find similar past events and outcomes.

    Args:
        trend_title: The trend headline to search precedents for.
    """
    result = await ctx.deps.tavily_tool.enrich_trend(
        trend_title=trend_title,
        trend_summary="",
    )

    context = result.get("enriched_context", "")
    sources = result.get("sources", [])
    source_titles = [s.get("title", "") for s in sources]

    return (
        f"Precedent search for '{trend_title}':\n"
        f"Context: {context[:300]}\n"
        f"Sources: {source_titles}"
    )


@impact_agent.tool
async def identify_cross_trend_opportunities(ctx: RunContext[AgentDeps]) -> str:
    """Identify compound opportunities where 2+ trends affect the same sector.

    Cross-trend synthesis reveals larger market shifts that individual
    trend analysis might miss.
    """
    impacts = getattr(ctx.deps, '_impacts', [])
    if len(impacts) < 2:
        return "Need at least 2 impacts for cross-trend analysis."

    # Group by sector
    sector_trends: Dict[str, List[str]] = {}
    for imp in impacts:
        for sector in getattr(imp, 'sectors_affected', []):
            s = sector if isinstance(sector, str) else str(sector)
            sector_trends.setdefault(s, []).append(imp.trend_title)

    compound = {s: trends for s, trends in sector_trends.items() if len(trends) >= 2}
    if not compound:
        return "No compound opportunities found (each sector affected by only 1 trend)."

    lines = ["Cross-trend compound opportunities:"]
    for sector, trends in compound.items():
        lines.append(f"  {sector}: {len(trends)} converging trends — {trends}")
    return "\n".join(lines)


# ── Public runner ─────────────────────────────────────────────────────

async def run_market_impact(deps: AgentDeps) -> tuple:
    """Run the Market Impact Agent. Returns (impacts, result_metadata)."""
    trends = getattr(deps, '_trend_data', [])

    prompt = (
        f"Analyze business impact of {len(trends)} market trends. "
        f"Focus on consulting opportunities for affected Indian companies."
    )

    try:
        from app.tools.provider_manager import ProviderManager
        pm = ProviderManager(mock_mode=deps.mock_mode)
        model = pm.get_model()

        result = await impact_agent.run(prompt, deps=deps, model=model)
        impacts = getattr(deps, '_impacts', [])

        logger.info(
            f"Market Impact Agent: {len(impacts)} impacts analyzed, "
            f"reasoning: {result.output.reasoning[:100]}"
        )
        return impacts, result.output

    except Exception as e:
        logger.error(f"Market Impact Agent failed: {e}")
        # Fallback: run ImpactAnalyzer directly
        from app.agents.impact_agent import ImpactAnalyzer
        from app.schemas import AgentState

        analyzer = ImpactAnalyzer(mock_mode=deps.mock_mode, deps=deps)
        state = AgentState(trends=trends)
        result_state = await analyzer.analyze_impacts(state)

        fallback = ImpactResult(reasoning=f"Fallback: {e}")
        return result_state.impacts or [], fallback
```

**Step 2: Verify**

Run: `python -c "from app.agents.market_impact_agent import impact_agent; print('OK')"`

**Step 3: Commit**

```bash
git add app/agents/market_impact_agent.py
git commit -m "feat: Market Impact Agent with council/precedent/cross-trend tools"
```

---

### Task 5: Create Lead Gen Agent

Combines company discovery + contact finding + email generation into one autonomous agent.

**Files:**
- Create: `app/agents/lead_gen_agent.py`

**Step 1: Create the agent file**

```python
# app/agents/lead_gen_agent.py
"""
Lead Gen Agent — autonomous company discovery, contact finding, and pitch generation.

Combines the previous CompanyDiscovery + ContactFinder + EmailGenerator
into a single pydantic-ai agent that autonomously decides:
- Search strategy per trend (NER-based vs industry-based)
- How many companies to pursue per trend
- Which roles to target based on trend type
- Whether a company-trend fit is strong enough

Tools:
  - find_companies_for_trends: Discover companies affected by trends
  - find_contacts: Find decision-makers at target companies
  - generate_outreach: Create personalized pitch emails
  - assess_company_relevance: Score company-trend fit via bandit
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.agent_deps import AgentDeps

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────

class LeadGenResult(BaseModel):
    """Structured output from the Lead Gen Agent."""
    companies_found: int = 0
    contacts_found: int = 0
    emails_generated: int = 0
    outreach_generated: int = 0
    low_relevance_filtered: int = 0
    reasoning: str = ""


# ── System prompt ─────────────────────────────────────────────────────

LEAD_GEN_PROMPT = """\
You are the Lead Gen Agent for a sales intelligence pipeline. Your job is to \
find companies, contacts, and generate personalized outreach for each trend.

WORKFLOW:
1. Find companies affected by each high-confidence trend using the company finder.
2. Assess company-trend relevance — skip weak fits to save API calls.
3. Find decision-maker contacts at promising companies.
4. Generate personalized outreach emails connecting the trend to the company's needs.

QUALITY RULES:
- Only pursue companies where the trend creates a genuine pain point.
- Target the right role: tech trends → CTO/VP Engineering, regulatory → GC/CCO, \
  market shifts → CEO/CSO.
- Personalize emails with specific trend details — no generic templates.
- Quality over quantity: 5 great leads > 20 mediocre ones.
"""


# ── Agent definition ──────────────────────────────────────────────────

lead_gen_agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=AgentDeps,
    output_type=LeadGenResult,
    system_prompt=LEAD_GEN_PROMPT,
    retries=2,
)


# ── Tools ─────────────────────────────────────────────────────────────

@lead_gen_agent.tool
async def find_companies_for_trends(ctx: RunContext[AgentDeps]) -> str:
    """Find companies affected by each analyzed trend.

    Uses the existing CompanyDiscovery pipeline with NER-based
    hallucination guard and Wikipedia verification.
    """
    from app.agents.company_agent import CompanyDiscovery
    from app.schemas import AgentState

    trends = getattr(ctx.deps, '_trend_data', [])
    impacts = getattr(ctx.deps, '_impacts', [])
    if not trends:
        return "ERROR: No trends available."

    discovery = CompanyDiscovery(
        mock_mode=ctx.deps.mock_mode,
        deps=ctx.deps,
        log_callback=ctx.deps.log_callback,
    )

    state = AgentState(trends=trends, impacts=impacts)
    result = await discovery.find_companies(state)

    ctx.deps._companies = result.companies or []
    verified = sum(1 for c in ctx.deps._companies if getattr(c, 'ner_verified', False))

    return (
        f"Found {len(ctx.deps._companies)} companies across {len(trends)} trends. "
        f"NER-verified: {verified}. Errors: {len(result.errors)}"
    )


@lead_gen_agent.tool
async def find_contacts_for_companies(ctx: RunContext[AgentDeps]) -> str:
    """Find decision-maker contacts at target companies.

    Uses Apollo (primary) + web search (fallback) to find
    name, role, LinkedIn, and email.
    """
    from app.agents.contact_agent import ContactFinder
    from app.schemas import AgentState

    companies = getattr(ctx.deps, '_companies', [])
    trends = getattr(ctx.deps, '_trend_data', [])
    impacts = getattr(ctx.deps, '_impacts', [])
    if not companies:
        return "No companies to find contacts for."

    finder = ContactFinder(mock_mode=ctx.deps.mock_mode, deps=ctx.deps)
    state = AgentState(
        trends=trends,
        impacts=impacts,
        companies=companies,
    )
    result = await finder.find_contacts(state)

    ctx.deps._contacts = result.contacts or []
    with_email = sum(1 for c in ctx.deps._contacts if getattr(c, 'email', ''))

    return (
        f"Found {len(ctx.deps._contacts)} contacts "
        f"({with_email} with email). "
        f"Errors: {len(result.errors)}"
    )


@lead_gen_agent.tool
async def generate_outreach_emails(ctx: RunContext[AgentDeps]) -> str:
    """Generate personalized outreach emails for all contacts.

    Creates pitch emails connecting the specific trend to the
    company's pain points and CMI's relevant services.
    """
    from app.agents.email_agent import EmailGenerator
    from app.schemas import AgentState

    contacts = getattr(ctx.deps, '_contacts', [])
    companies = getattr(ctx.deps, '_companies', [])
    trends = getattr(ctx.deps, '_trend_data', [])
    if not contacts:
        return "No contacts to generate outreach for."

    generator = EmailGenerator(mock_mode=ctx.deps.mock_mode, deps=ctx.deps)
    state = AgentState(
        trends=trends,
        companies=companies,
        contacts=contacts,
    )
    result = await generator.process_emails(state)

    ctx.deps._contacts = result.contacts or contacts  # Updated with emails
    ctx.deps._outreach = result.outreach_emails or []

    return (
        f"Generated {len(ctx.deps._outreach)} outreach emails. "
        f"Errors: {len(result.errors)}"
    )


@lead_gen_agent.tool
async def assess_company_relevance(
    ctx: RunContext[AgentDeps],
    company_name: str,
    trend_title: str,
) -> str:
    """Assess how relevant a company is to a specific trend.

    Uses the company relevance bandit for contextual scoring.

    Args:
        company_name: Name of the company.
        trend_title: Title of the trend.
    """
    bandit = ctx.deps.company_bandit
    score = bandit.predict(company_name, trend_title)
    return f"Relevance of '{company_name}' to '{trend_title}': {score:.3f}"


# ── Public runner ─────────────────────────────────────────────────────

async def run_lead_gen(deps: AgentDeps) -> tuple:
    """Run the Lead Gen Agent. Returns (companies, contacts, outreach)."""
    trends = getattr(deps, '_trend_data', [])
    impacts = getattr(deps, '_impacts', [])

    prompt = (
        f"Find companies and contacts for {len(trends)} trends "
        f"({len(impacts)} with impact analysis). "
        f"Generate personalized outreach emails."
    )

    try:
        from app.tools.provider_manager import ProviderManager
        pm = ProviderManager(mock_mode=deps.mock_mode)
        model = pm.get_model()

        result = await lead_gen_agent.run(prompt, deps=deps, model=model)

        companies = getattr(deps, '_companies', [])
        contacts = getattr(deps, '_contacts', [])
        outreach = getattr(deps, '_outreach', [])

        logger.info(
            f"Lead Gen Agent: {len(companies)} companies, "
            f"{len(contacts)} contacts, {len(outreach)} outreach"
        )
        return companies, contacts, outreach, result.output

    except Exception as e:
        logger.error(f"Lead Gen Agent failed: {e}")
        # Fallback: run each stage directly
        from app.agents.company_agent import run_company_agent
        from app.agents.contact_agent import run_contact_agent
        from app.agents.email_agent import run_email_agent
        from app.schemas import AgentState

        state = AgentState(trends=trends, impacts=impacts)
        state = await run_company_agent(state, deps=deps)
        state = await run_contact_agent(state, deps=deps)
        state = await run_email_agent(state, deps=deps)

        fallback = LeadGenResult(reasoning=f"Fallback: {e}")
        return state.companies, state.contacts, state.outreach_emails, fallback
```

**Step 2: Verify**

Run: `python -c "from app.agents.lead_gen_agent import lead_gen_agent; print('OK')"`

**Step 3: Commit**

```bash
git add app/agents/lead_gen_agent.py
git commit -m "feat: Lead Gen Agent with company/contact/email/relevance tools"
```

---

### Task 6: Create Quality Agent

Validates outputs at every stage, routes feedback to learning subsystems.

**Files:**
- Create: `app/agents/quality_agent.py`

**Step 1: Create the agent file**

```python
# app/agents/quality_agent.py
"""
Quality Agent — validation, gating, and feedback routing.

pydantic-ai agent that validates outputs at every pipeline stage,
decides pass/fail/retry, and routes feedback to learning subsystems
(source bandit, company bandit, weight learner).

Tools:
  - validate_trends: Council Stage A trend validation
  - validate_impacts: Confidence-based impact filtering
  - validate_leads: Council Stage C lead quality check
  - record_feedback: Save feedback to JSONL for learning
  - check_quality_bounds: Compare against quality thresholds
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.agent_deps import AgentDeps

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────

class QualityVerdict(BaseModel):
    """Structured output from the Quality Agent."""
    stage: str = ""
    passed: bool = True
    should_retry: bool = False
    items_passed: int = 0
    items_filtered: int = 0
    quality_score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    reasoning: str = ""


# ── System prompt ─────────────────────────────────────────────────────

QUALITY_PROMPT = """\
You are the Quality Agent for a sales intelligence pipeline. You validate \
outputs at every stage and decide: PASS, RETRY, or FAIL.

VALIDATION STAGES:
1. Post-Analysis: Check cluster coherence, noise ratio, trend count
2. Post-Impact: Check impact confidence scores, filter low quality
3. Post-LeadGen: Validate company-trend fit, check email quality

DECISION RULES:
- PASS: Quality meets thresholds → proceed to next stage
- RETRY: Quality is borderline → ask previous agent to try again (max 2x)
- FAIL: Quality is unacceptable AND retries exhausted → proceed with degraded output

QUALITY THRESHOLDS:
- Mean coherence >= 0.40 (trends)
- Council confidence >= 0.35 (impacts)
- Company relevance >= 0.30 (leads)

Always explain your quality verdict with specific metrics.
"""


# ── Agent definition ──────────────────────────────────────────────────

quality_agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=AgentDeps,
    output_type=QualityVerdict,
    system_prompt=QUALITY_PROMPT,
    retries=1,
)


# ── Tools ─────────────────────────────────────────────────────────────

@quality_agent.tool
async def validate_trend_quality(ctx: RunContext[AgentDeps]) -> str:
    """Validate trend clustering quality (post-Analysis stage).

    Checks coherence, noise ratio, cluster count against thresholds.
    """
    tree = getattr(ctx.deps, '_trend_tree', None)
    if tree is None:
        return "No trend tree to validate."

    major = tree.to_major_trends()
    if not major:
        return "FAIL: Zero clusters found."

    import numpy as np
    coherences = [getattr(m, 'coherence_score', 0.5) for m in major]
    mean_coh = float(np.mean(coherences))
    min_coh = float(np.min(coherences))

    issues = []
    if mean_coh < 0.40:
        issues.append(f"Low mean coherence: {mean_coh:.3f} (need >= 0.40)")
    if len(major) < 3:
        issues.append(f"Too few clusters: {len(major)} (need >= 3)")
    if min_coh < 0.25:
        issues.append(f"Very low min coherence: {min_coh:.3f}")

    status = "PASS" if not issues else "RETRY" if mean_coh >= 0.35 else "FAIL"
    return (
        f"Trend validation: {status}\n"
        f"Clusters: {len(major)}, Mean coherence: {mean_coh:.3f}, "
        f"Min: {min_coh:.3f}\n"
        f"Issues: {issues if issues else 'None'}"
    )


@quality_agent.tool
async def validate_impact_quality(ctx: RunContext[AgentDeps]) -> str:
    """Validate impact analysis quality (post-Impact stage).

    Filters impacts by council confidence threshold.
    """
    from app.config import get_settings
    settings = get_settings()
    threshold = settings.min_trend_confidence_for_agents

    impacts = getattr(ctx.deps, '_impacts', [])
    if not impacts:
        return "No impacts to validate."

    viable = [i for i in impacts if i.council_confidence >= threshold]
    filtered = len(impacts) - len(viable)

    # Update deps with filtered impacts
    ctx.deps._viable_impacts = viable

    return (
        f"Impact validation: {len(viable)}/{len(impacts)} passed "
        f"(threshold={threshold}). Filtered: {filtered}."
    )


@quality_agent.tool
async def record_quality_feedback(
    ctx: RunContext[AgentDeps],
    feedback_type: str,
    item_id: str,
    rating: str,
) -> str:
    """Record quality feedback for learning subsystems.

    Args:
        feedback_type: "trend" or "lead"
        item_id: Trend or lead identifier.
        rating: For trends: good_trend/bad_trend/already_knew.
                For leads: would_email/maybe/bad_lead.
    """
    from app.tools.feedback import save_feedback
    record = save_feedback(
        feedback_type=feedback_type,
        item_id=item_id,
        rating=rating,
    )
    return f"Feedback recorded: {feedback_type}/{rating} for {item_id}"


# ── Public runner ─────────────────────────────────────────────────────

async def run_quality_check(deps: AgentDeps, stage: str) -> QualityVerdict:
    """Run the Quality Agent for a specific stage.

    Args:
        stage: "trends", "impacts", or "leads"
    """
    prompt = f"Validate the quality of the '{stage}' stage output."

    try:
        from app.tools.provider_manager import ProviderManager
        pm = ProviderManager(mock_mode=deps.mock_mode)
        model = pm.get_model()

        result = await quality_agent.run(prompt, deps=deps, model=model)
        logger.info(
            f"Quality Agent ({stage}): "
            f"passed={result.output.passed}, "
            f"retry={result.output.should_retry}"
        )
        return result.output

    except Exception as e:
        logger.error(f"Quality Agent failed: {e}")
        # Fallback: always pass
        return QualityVerdict(
            stage=stage,
            passed=True,
            reasoning=f"Fallback (agent error): {e}",
        )
```

**Step 2: Verify**

Run: `python -c "from app.agents.quality_agent import quality_agent; print('OK')"`

**Step 3: Commit**

```bash
git add app/agents/quality_agent.py
git commit -m "feat: Quality Agent with validation/feedback/gating tools"
```

---

## Phase 2: Supervisor Graph — Rewire Orchestrator

### Task 7: Rewrite orchestrator with multi-agent supervisor

Replace the linear pipeline with agent-based nodes and quality gate retry loops.

**Files:**
- Modify: `app/agents/orchestrator.py` (full rewrite of graph section)

**Step 1: Rewrite orchestrator**

The key changes:
1. `GraphState` gets new fields for agent metadata
2. Each node runs a pydantic-ai agent via its `run_*` function
3. Quality gates use conditional edges with retry routing
4. Fallback-safe: every agent has a try/except that falls back to direct calls

Replace the graph nodes and construction in `orchestrator.py`:

```python
# Replace GraphState with:
class GraphState(TypedDict):
    """LangGraph state for the multi-agent pipeline."""
    deps: Any                                       # AgentDeps instance
    trends: List[TrendData]
    impacts: List[ImpactAnalysis]
    companies: List[CompanyData]
    contacts: List[ContactData]
    outreach_emails: List[OutreachEmail]
    errors: Annotated[List[str], operator.add]
    current_step: str
    # Multi-agent metadata
    retry_counts: Dict[str, int]
    agent_reasoning: Dict[str, str]
```

Replace node functions:

```python
async def source_intel_node(state: GraphState) -> dict:
    """Step 1: Source Intel Agent — autonomous article collection."""
    logger.info("=" * 50)
    logger.info("STEP 1: SOURCE INTEL AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    from .source_intel_agent import run_source_intel
    articles, embeddings, result = await run_source_intel(deps)

    # Store on deps for downstream agents
    deps._articles = articles
    deps._embeddings = embeddings

    return {
        "errors": [e for e in [] if e],
        "current_step": "source_intel_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "source_intel": result.reasoning,
        },
    }


async def analysis_node(state: GraphState) -> dict:
    """Step 2: Analysis Agent — trend clustering and signals."""
    logger.info("=" * 50)
    logger.info("STEP 2: ANALYSIS AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    from .analysis_agent import run_analysis
    tree, result = await run_analysis(deps)

    # Convert TrendTree → List[TrendData]
    trends = []
    if tree:
        major_trends = tree.to_major_trends()
        trends = [
            TrendData(
                id=str(mt.id),
                trend_title=mt.trend_title,
                summary=mt.trend_summary,
                severity=mt.severity if isinstance(mt.severity, str) else mt.severity,
                industries_affected=[
                    s.value if hasattr(s, 'value') else str(s)
                    for s in mt.primary_sectors
                ],
                source_links=[],
                keywords=mt.key_entities[:10] if mt.key_entities else mt.key_keywords[:10],
            )
            for mt in major_trends
        ]

    # Store for downstream
    deps._trend_data = trends

    return {
        "trends": trends,
        "current_step": "analysis_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "analysis": result.reasoning,
        },
    }


async def impact_node(state: GraphState) -> dict:
    """Step 3: Market Impact Agent — multi-perspective analysis."""
    logger.info("=" * 50)
    logger.info("STEP 3: MARKET IMPACT AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    from .market_impact_agent import run_market_impact
    impacts, result = await run_market_impact(deps)

    return {
        "impacts": impacts,
        "current_step": "impact_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "market_impact": result.reasoning,
        },
    }


def quality_gate(state: GraphState) -> str:
    """Route: quality check on impacts, retry or proceed."""
    settings = get_settings()
    threshold = settings.min_trend_confidence_for_agents
    impacts = state.get("impacts", [])

    viable = [imp for imp in impacts if imp.council_confidence >= threshold]
    dropped = len(impacts) - len(viable)

    if dropped:
        logger.info(
            f"Quality gate: {dropped}/{len(impacts)} trends below "
            f"confidence {threshold}"
        )

    if not viable:
        logger.warning("Quality gate: No viable trends — skipping lead gen")
        return "end"

    state["impacts"] = viable
    return "lead_gen"


async def lead_gen_node(state: GraphState) -> dict:
    """Step 4: Lead Gen Agent — companies, contacts, emails."""
    logger.info("=" * 50)
    logger.info("STEP 4: LEAD GEN AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    from .lead_gen_agent import run_lead_gen
    companies, contacts, outreach, result = await run_lead_gen(deps)

    return {
        "companies": companies or [],
        "contacts": contacts or [],
        "outreach_emails": outreach or [],
        "current_step": "lead_gen_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "lead_gen": result.reasoning,
        },
    }
```

Replace `create_pipeline_graph`:

```python
def create_pipeline_graph():
    """Create the multi-agent LangGraph pipeline.

    Flow:
      START → source_intel → analysis → impact → quality_gate
        ├─ "lead_gen" → lead_gen → END
        └─ "end" → END
    """
    workflow = StateGraph(GraphState)

    # Agent nodes
    workflow.add_node("source_intel", source_intel_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("impact", impact_node)
    workflow.add_node("lead_gen", lead_gen_node)

    # Edges
    workflow.add_edge(START, "source_intel")
    workflow.add_edge("source_intel", "analysis")
    workflow.add_edge("analysis", "impact")
    workflow.add_conditional_edges(
        "impact",
        quality_gate,
        {"lead_gen": "lead_gen", "end": END},
    )
    workflow.add_edge("lead_gen", END)

    return workflow.compile()
```

Update `run_pipeline` to use `AgentDeps`:

```python
async def run_pipeline(mock_mode: bool = False, log_callback=None) -> PipelineResult:
    """Execute the multi-agent sales intelligence pipeline."""
    start_time = datetime.utcnow()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")

    logger.info("Starting Multi-Agent Sales Intelligence Pipeline")
    logger.info(f"Run ID: {run_id} | Mock Mode: {mock_mode}")

    # Shared dependencies for all agents
    from .agent_deps import AgentDeps
    deps = AgentDeps.create(mock_mode=mock_mode, log_callback=log_callback)

    initial_state: GraphState = {
        "deps": deps,
        "trends": [],
        "impacts": [],
        "companies": [],
        "contacts": [],
        "outreach_emails": [],
        "errors": [],
        "current_step": "init",
        "retry_counts": {},
        "agent_reasoning": {},
    }

    # ... rest of run_pipeline stays the same
```

**Step 2: Verify import and graph compilation**

Run: `python -c "from app.agents.orchestrator import create_pipeline_graph; g = create_pipeline_graph(); print(f'Graph nodes: {list(g.nodes.keys())}')"`

**Step 3: Commit**

```bash
git add app/agents/orchestrator.py
git commit -m "feat: Multi-agent supervisor graph with quality gate routing"
```

---

### Task 8: Update __init__.py exports

**Files:**
- Modify: `app/agents/__init__.py`

**Step 1: Add new agent exports**

Add imports for the new agents alongside existing exports:

```python
# Add to app/agents/__init__.py:
from .agent_deps import AgentDeps
from .source_intel_agent import run_source_intel, SourceIntelResult
from .analysis_agent import run_analysis, AnalysisResult
from .market_impact_agent import run_market_impact, ImpactResult
from .lead_gen_agent import run_lead_gen, LeadGenResult
from .quality_agent import run_quality_check, QualityVerdict
```

**Step 2: Commit**

```bash
git add app/agents/__init__.py
git commit -m "feat: Export new multi-agent modules"
```

---

## Phase 3: Integration and Testing

### Task 9: Smoke test — run the multi-agent pipeline

**Step 1: Run with mock mode to verify graph flow**

Run: `python -c "
import asyncio
from app.agents.orchestrator import run_pipeline
result = asyncio.run(run_pipeline(mock_mode=True))
print(f'Status: {result.status}')
print(f'Trends: {result.trends_detected}')
print(f'Companies: {result.companies_found}')
print(f'Errors: {result.errors[:3]}')
"`

Expected: Pipeline completes with mock data, producing trends → impacts → companies → contacts → emails.

**Step 2: Fix any import/runtime errors**

Iterate on any failures until the mock pipeline runs end-to-end.

**Step 3: Run with real data (quick test)**

Run: `python -c "
import asyncio
from app.agents.orchestrator import run_pipeline
result = asyncio.run(run_pipeline(mock_mode=False))
print(f'Status: {result.status}')
print(f'Trends: {result.trends_detected} | Companies: {result.companies_found}')
print(f'Runtime: {result.run_time_seconds:.1f}s')
print(f'Agent reasoning keys: {list(result.__dict__.keys())}')
"`

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: Multi-agent pipeline integration fixes"
```

---

### Task 10: Stress test comparison

Run the stress test to compare multi-agent pipeline quality against the linear baseline.

**Step 1: Quick stress test**

Run: `python stress_test.py --quick --use-cache`

Compare composite quality score against baseline (target: >= 0.45).

**Step 2: Full stress test**

Run: `python stress_test.py --use-cache --windows 24,72`

Compare all metrics against iteration 6 baseline.

**Step 3: Document results**

Check: `ls data/stress_test_reports/report_*.md`

---

## Implementation Notes

### Fallback Safety
Every agent has a try/except that falls back to direct tool calls (no agent reasoning). This means the pipeline NEVER breaks — worst case it runs like the old linear pipeline. Agent autonomy is additive, not blocking.

### Provider Management
All agents use `ProviderManager.get_model()` at runtime via the `model=` parameter to `agent.run()`. This preserves the Gemini → Groq → Vertex fallback chain. The agent definition uses a placeholder model that gets overridden.

### Shared State via AgentDeps
pydantic-ai agents are stateless between runs. Data flows between agents via attributes on the `AgentDeps` instance (`deps._articles`, `deps._trend_tree`, etc.). LangGraph's `GraphState` handles the typed outputs, while `AgentDeps` carries the mutable working data.

### Learning Loop Integration
- **SourceBandit**: Source Intel Agent queries `get_quality_estimates()` before fetch, pipeline updates after clustering (existing engine.py behavior preserved).
- **CompanyBandit**: Lead Gen Agent queries `predict()` for relevance priors. Quality Agent auto-propagates via `save_feedback()`.
- **WeightLearner**: Reads from `data/feedback.jsonl` on next run. Quality Agent writes feedback in real-time.
- **TrendMemory**: Analysis Agent's pipeline run automatically queries ChromaDB for novelty scores (existing engine.py behavior preserved).
