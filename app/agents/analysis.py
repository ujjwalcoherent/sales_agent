"""
Analysis Agent — autonomous trend clustering and signal computation.

pydantic-ai agent powered by intelligence/pipeline.py (math-first, 22-agent system).
Math gates 1-9 run before any LLM call. First LLM call is at synthesis (step 8).
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.deps import AgentDeps
from app.config import get_settings, get_domestic_source_ids

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Structured output from the Analysis Agent."""
    num_clusters: int = 0
    noise_ratio: float = 0.0
    mean_coherence: float = 0.0
    num_trends_passed: int = 0
    num_trends_rejected: int = 0
    params_used: Dict[str, float] = Field(default_factory=dict)
    retries_performed: int = 0
    reasoning: str = ""


ANALYSIS_PROMPT = """\
You are the Analysis Agent. Cluster news articles into coherent market trends.

WORKFLOW:
1. Run the trend pipeline on articles.
2. Evaluate cluster quality (coherence, noise ratio).
3. If quality is poor (mean coherence < 0.40 or noise > 0.40), retry with \
   adjusted parameters (lower merge threshold or higher coherence min). Max 2 retries.
4. Check trend novelty via trend memory.
5. Return metrics and reasoning.

Quality targets: mean coherence >= 0.45, noise ratio < 0.35, >= 3 clusters.
"""

analysis = Agent(
    'test',  # Placeholder — overridden at runtime via deps.get_model()
    deps_type=AgentDeps,
    output_type=AnalysisResult,
    system_prompt=ANALYSIS_PROMPT,
    retries=2,
)


@analysis.tool
async def run_trend_pipeline(
    ctx: RunContext[AgentDeps],
    semantic_dedup_threshold: float = 0.88,
    coherence_min: float = 0.48,
    merge_threshold: float = 0.82,
) -> str:
    """Run intelligence/pipeline.py (math-first, 22-agent system) on current scope.

    Replaces old TrendPipeline. Runs 9 math gates before first LLM call:
    Fetch → Dedup → Salience Filter → Entity Extraction → Similarity →
    Cluster → Validate → Synthesize → Match

    Args:
        semantic_dedup_threshold: Dedup similarity threshold (0.85-0.92) — adaptive.
        coherence_min: Min coherence to keep a cluster (0.35-0.55) — adaptive.
        merge_threshold: Threshold for merging similar clusters — adaptive.
    """
    # Guard: never re-run the full pipeline in one session — it takes 10-15 minutes.
    # If already ran this run, return the cached result so the agent can evaluate and
    # produce its structured output without triggering a second 10-minute execution.
    existing = getattr(ctx.deps, "_intelligence_result", None)
    if existing is not None:
        return (
            f"Pipeline already ran this session: {existing.total_clusters} clusters, "
            f"noise={existing.noise_rate:.2f}, "
            f"coherence={existing.mean_coherence:.2f}. "
            f"Use evaluate_cluster_quality to review and proceed to output."
        )

    settings = get_settings()

    from app.intelligence.pipeline import execute as intelligence_execute
    from app.intelligence.models import DiscoveryScope, DiscoveryMode
    from app.intelligence.config import load_adaptive_params

    scope = DiscoveryScope(
        mode=DiscoveryMode.INDUSTRY_FIRST,
        industry=getattr(settings, "industry_focus", "Technology"),
        region=settings.country_code or "IN",
        hours=getattr(settings, "rss_hours_ago", 120),
    )

    params = load_adaptive_params()
    # Allow the agent's arguments to override adaptive defaults (for retry)
    params.dedup_title_threshold = max(0.80, semantic_dedup_threshold)
    params.filter_auto_accept = coherence_min * 0.9  # tighter filter = more precise

    result = await intelligence_execute(scope, params)
    ctx.deps._intelligence_result = result
    ctx.deps._trend_tree = None  # Clear old tree — intelligence result takes precedence

    return (
        f"Intelligence pipeline: {result.total_articles_fetched} articles fetched, "
        f"{result.total_articles_post_filter} after filter, "
        f"{result.total_clusters} clusters, "
        f"noise={result.noise_rate:.2f}, "
        f"coherence={result.mean_coherence:.2f}"
    )


@analysis.tool
async def evaluate_cluster_quality(ctx: RunContext[AgentDeps]) -> str:
    """Evaluate quality of current clustering results."""
    import numpy as np

    intel = getattr(ctx.deps, "_intelligence_result", None)
    if intel is not None:
        clusters = intel.clusters
        if not clusters:
            return "Zero clusters. All articles are noise."
        cohs = [c.coherence_score for c in clusters if c.coherence_score > 0]
        sizes = [len(c.article_indices) for c in clusters]
        mean_c = float(np.mean(cohs)) if cohs else 0.0
        issues = []
        if mean_c < 0.40:
            issues.append(f"Low coherence: {mean_c:.3f}")
        if len(clusters) < 3:
            issues.append(f"Few clusters: {len(clusters)}")
        status = "PASS" if not issues else "RETRY"
        return f"{status}: {len(clusters)} clusters, coherence={mean_c:.3f}, sizes={sorted(sizes, reverse=True)[:8]}, issues={issues}"

    return "No analysis result. Run pipeline first."


@analysis.tool
async def check_trend_novelty(ctx: RunContext[AgentDeps]) -> str:
    """Check trend memory for novelty scores."""
    intel = getattr(ctx.deps, "_intelligence_result", None)
    if intel is None:
        return "No intelligence result."
    clusters = intel.clusters
    lines = []
    for c in clusters[:15]:
        lines.append(f"  {(c.label or c.primary_entity or 'unknown')[:50]}: "
                     f"coherence={c.coherence_score:.3f}, articles={len(c.article_indices)}")
    return f"Clusters ({len(clusters)}):\n" + "\n".join(lines)


async def run_analysis(deps: AgentDeps) -> Any:
    """Run the Analysis Agent. Returns (intelligence_result_or_None, result_metadata).

    Powered by intelligence/pipeline.py (22-agent math-first system).
    The pydantic-ai agent calls run_trend_pipeline tool which executes the full
    intelligence pipeline. Result is stored in deps._intelligence_result.
    """
    prompt = (
        f"Run the trend pipeline to cluster news articles into market trends. "
        f"Target coherence >= 0.45. Evaluate quality after clustering."
    )

    agent_result = None
    try:
        model = deps.get_model()
        result = await analysis.run(prompt, deps=deps, model=model)
        agent_result = result.output
        intel = getattr(deps, "_intelligence_result", None)
        if intel:
            logger.info(
                f"Analysis: {intel.total_clusters} clusters, "
                f"coherence={intel.mean_coherence:.3f}, "
                f"noise={intel.noise_rate:.2f}"
            )
    except Exception as e:
        logger.error(f"Analysis Agent failed: {e}, running intelligence pipeline directly")

    # Fallback: if agent didn't run the tool, run intelligence pipeline directly
    intel = getattr(deps, "_intelligence_result", None)
    if intel is None:
        from app.intelligence.pipeline import execute as intelligence_execute
        from app.intelligence.models import DiscoveryScope, DiscoveryMode
        from app.intelligence.config import load_adaptive_params
        settings = get_settings()
        scope = DiscoveryScope(
            mode=DiscoveryMode.INDUSTRY_FIRST,
            industry=getattr(settings, "industry_focus", "Technology"),
            region=settings.country_code or "IN",
            hours=getattr(settings, "rss_hours_ago", 120),
        )
        intel = await intelligence_execute(scope, load_adaptive_params())
        deps._intelligence_result = intel

    return intel, agent_result or AnalysisResult(reasoning="Intelligence pipeline")
