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
    'test',  # Placeholder — overridden at runtime via deps.get_model()
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

    ctx.deps._articles = articles

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

    from uuid import uuid4
    from datetime import datetime, timezone

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
            content=r.get("content", ""),
        )
        new_articles.append(article)

    ctx.deps._articles.extend(new_articles)

    summary = f"Web search for '{query}': {len(results)} results."
    if answer:
        summary += f"\nSummary: {answer[:200]}"
    return summary


@source_intel_agent.tool
async def scrape_and_embed_articles(ctx: RunContext[AgentDeps]) -> str:
    """Scrape full content and generate embeddings for all collected articles.

    Call this after fetch_rss_articles (and optionally search_web).
    Populates content via trafilatura and generates 1024-dim embeddings.
    """
    articles = ctx.deps._articles
    if not articles:
        return "No articles to process."

    from app.news.scraper import scrape_articles
    enriched_count = await scrape_articles(articles)

    texts = []
    for a in articles:
        parts = [a.title]
        if a.content:
            parts.append(a.content[:1000])
        elif a.summary:
            parts.append(a.summary)
        texts.append(" ".join(parts))

    embeddings = ctx.deps.embedding_tool.embed_batch(texts)
    ctx.deps._embeddings = embeddings

    content_rate = sum(1 for a in articles if a.content) / max(len(articles), 1)
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
    articles = ctx.deps._articles
    embeddings = ctx.deps._embeddings
    if not articles or not embeddings:
        return "No articles/embeddings to classify."

    from app.news.event_classifier import EmbeddingEventClassifier
    classifier = EmbeddingEventClassifier(ctx.deps.embedding_tool)
    event_counts = classifier.classify_batch(articles)

    ctx.deps._event_distribution = dict(event_counts)
    return f"Event classification: {dict(event_counts)}"


@source_intel_agent.tool
async def check_source_quality(ctx: RunContext[AgentDeps]) -> str:
    """Check source quality via Thompson Sampling (explore + exploit).

    Returns quality rankings based on posterior SAMPLES (not just means).
    New/uncertain sources may rank higher than expected — this is
    intentional exploration that helps discover good new sources.
    """
    bandit = ctx.deps.source_bandit
    estimates = bandit.get_quality_estimates()
    if not estimates:
        return "No source quality data yet (first run). All sources treated equally."

    # Use actual Thompson Sampling — sample from posteriors
    ranked = bandit.select_sources(list(estimates.keys()))
    means = estimates

    lines = [f"Source quality (Thompson Sampling, {len(estimates)} sources):"]
    for sid in ranked[:15]:
        mean = means.get(sid, 0.5)
        p = bandit._posteriors.get(sid, {})
        total_obs = p.get("alpha", 1) + p.get("beta", 1) - 2  # subtract prior
        lines.append(f"  {sid}: mean={mean:.3f} (obs={total_obs:.0f})")
    return "\n".join(lines)


# ── Public runner ─────────────────────────────────────────────────────

async def run_source_intel(deps: AgentDeps) -> tuple:
    """Run the Source Intel Agent and return (articles, embeddings, result).

    This is the entry point called by the LangGraph orchestrator node.
    """
    settings = get_settings()

    prompt = (
        f"Collect Indian market news articles from the last "
        f"{settings.rss_hours_ago} hours. "
        f"Fetch up to {settings.rss_max_per_source} articles per source. "
        f"Target: at least 50 quality articles with full content."
    )

    try:
        model = deps.get_model()
        result = await source_intel_agent.run(prompt, deps=deps, model=model)

        articles = deps._articles
        embeddings = deps._embeddings

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
        deps._articles = articles

        from app.news.scraper import scrape_articles
        await scrape_articles(articles)
        texts = [f"{a.title} {a.content or a.summary or ''}"[:1000] for a in articles]
        embeddings = deps.embedding_tool.embed_batch(texts)
        deps._embeddings = embeddings

        fallback_result = SourceIntelResult(
            total_fetched=len(articles),
            total_after_dedup=len(articles),
            reasoning=f"Fallback mode (agent error: {e})",
        )
        return articles, embeddings, fallback_result
