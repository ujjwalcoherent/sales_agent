"""
Analysis Agent — autonomous trend clustering and signal computation.

pydantic-ai agent wrapping TrendPipeline layers as callable tools.
Decides clustering parameters, evaluates coherence, triggers retries.
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
    """Run full TrendPipeline (Leiden clustering + synthesis) on articles.

    Args:
        semantic_dedup_threshold: Dedup similarity threshold (0.85-0.92).
        coherence_min: Min coherence to keep a cluster (0.35-0.55).
        merge_threshold: Threshold for merging similar clusters (0.70-0.90).
    """
    articles = ctx.deps._articles
    if not articles:
        return "ERROR: No articles. Source Intel must run first."

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

    tree = await pipeline.run(articles)
    ctx.deps._trend_tree = tree
    ctx.deps._pipeline = pipeline
    ctx.deps._params_used = {
        "semantic_dedup_threshold": semantic_dedup_threshold,
        "coherence_min": coherence_min,
        "merge_threshold": merge_threshold,
    }

    major = tree.to_major_trends()
    metrics = getattr(pipeline, '_last_metrics', {})
    return (
        f"Pipeline: {len(articles)} articles -> {len(major)} trends, "
        f"noise={metrics.get('noise_ratio', 0):.2f}, "
        f"coherence={metrics.get('mean_coherence', 0):.2f}"
    )


@analysis.tool
async def evaluate_cluster_quality(ctx: RunContext[AgentDeps]) -> str:
    """Evaluate quality of current clustering results."""
    tree = ctx.deps._trend_tree
    if tree is None:
        return "No trend tree. Run pipeline first."

    import numpy as np
    major = tree.to_major_trends()
    if not major:
        return "Zero clusters. All articles are noise."

    cohs = [getattr(m, 'coherence_score', 0.5) for m in major]
    sizes = [getattr(m, 'article_count', 0) for m in major]
    issues = []
    mean_c = float(np.mean(cohs))
    if mean_c < 0.40:
        issues.append(f"Low coherence: {mean_c:.3f}")
    if len(major) < 3:
        issues.append(f"Few clusters: {len(major)}")

    status = "PASS" if not issues else "RETRY"
    return f"{status}: {len(major)} clusters, coherence={mean_c:.3f}, sizes={sorted(sizes, reverse=True)[:8]}, issues={issues}"


@analysis.tool
async def check_trend_novelty(ctx: RunContext[AgentDeps]) -> str:
    """Check trend memory for novelty scores."""
    tree = ctx.deps._trend_tree
    if tree is None:
        return "No trend tree."
    major = tree.to_major_trends()
    lines = []
    for m in major:
        nov = getattr(m, 'novelty_score', None)
        cont = getattr(m, 'continuity_run_count', 0)
        lines.append(f"  {m.trend_title[:50]}: novelty={nov}, seen={cont}x")
    return f"Novelty ({len(major)} trends):\n" + "\n".join(lines)


async def run_analysis(deps: AgentDeps) -> Any:
    """Run the Analysis Agent. Returns (tree, result_metadata)."""
    prompt = f"Analyze {len(deps._articles)} articles. Cluster into market trends. Target coherence >= 0.45."

    agent_result = None
    try:
        model = deps.get_model()
        result = await analysis.run(prompt, deps=deps, model=model)
        logger.info(f"Analysis: {result.output.num_clusters} clusters, coh={result.output.mean_coherence:.3f}")
        agent_result = result.output
    except Exception as e:
        logger.error(f"Analysis Agent failed: {e}, using fallback")

    # Build the tree if the agent's tool call didn't (e.g. mock mode skips tools)
    tree = deps._trend_tree
    if tree is None:
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
        tree = await pipeline.run(deps._articles)
        deps._trend_tree = tree

    return tree, agent_result or AnalysisResult(reasoning="Fallback pipeline")
