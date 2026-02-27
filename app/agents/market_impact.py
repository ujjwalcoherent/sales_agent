"""
Market Impact Agent — multi-perspective business impact analysis.

pydantic-ai agent that analyzes each trend from multiple perspectives,
searches for historical precedents, and identifies cross-trend opportunities.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.deps import AgentDeps

logger = logging.getLogger(__name__)


class ImpactResult(BaseModel):
    """Structured output from the Market Impact Agent."""
    total_trends_analyzed: int = 0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    precedent_searches: int = 0
    cross_trend_insights: List[str] = Field(default_factory=list)
    reasoning: str = ""


IMPACT_PROMPT = """\
You are the Market Impact Agent for a sales intelligence pipeline targeting \
Indian market consulting opportunities for Coherent Market Insights (CMI).

WORKFLOW:
1. Analyze all trends for business impact using the AI council tool.
2. For high-impact trends, search for historical precedents.
3. Identify cross-trend compound opportunities.
4. Return impact analysis with confidence scores.

Focus on: which companies NEED consulting, specific pain points, decision \
makers to pitch, and urgency. Be specific and evidence-based.
"""

impact_agent = Agent(
    'test',  # Placeholder — overridden at runtime via deps.get_model()
    deps_type=AgentDeps,
    output_type=ImpactResult,
    system_prompt=IMPACT_PROMPT,
    retries=2,
)


@impact_agent.tool
async def analyze_all_trends(ctx: RunContext[AgentDeps]) -> str:
    """Analyze business impact of all trends via AI council."""
    from app.agents.workers.impact_agent import ImpactAnalyzer
    from app.schemas import AgentState

    trends = ctx.deps._trend_data
    if not trends:
        return "ERROR: No trends to analyze."

    analyzer = ImpactAnalyzer(mock_mode=ctx.deps.mock_mode, deps=ctx.deps, log_callback=ctx.deps.log_callback)
    state = AgentState(trends=trends)
    result_state = await analyzer.analyze_impacts(state)
    ctx.deps._impacts = result_state.impacts or []

    high = sum(1 for i in ctx.deps._impacts if i.council_confidence >= 0.6)
    low = sum(1 for i in ctx.deps._impacts if i.council_confidence < 0.4)
    return f"Analyzed {len(trends)} trends -> {len(ctx.deps._impacts)} impacts. High confidence: {high}, Low: {low}"


@impact_agent.tool
async def search_precedent(ctx: RunContext[AgentDeps], trend_title: str) -> str:
    """Search for historical precedents of a specific trend.

    Args:
        trend_title: The trend headline to search precedents for.
    """
    result = await ctx.deps.tavily_tool.enrich_trend(trend_title, "")
    context = (result.get("enriched_context") or "")[:300]
    sources = [s.get("title", "") for s in result.get("sources", [])]
    return f"Precedent for '{trend_title[:40]}': {context} Sources: {sources}"


@impact_agent.tool
async def identify_cross_trend_opportunities(ctx: RunContext[AgentDeps]) -> str:
    """Find sectors affected by 2+ trends simultaneously."""
    impacts = ctx.deps._impacts
    if len(impacts) < 2:
        return "Need 2+ impacts for cross-trend analysis."

    sector_trends: Dict[str, List[str]] = {}
    for imp in impacts:
        for sector in getattr(imp, 'sectors_affected', []):
            s = sector if isinstance(sector, str) else str(sector)
            sector_trends.setdefault(s, []).append(imp.trend_title)

    compound = {s: t for s, t in sector_trends.items() if len(t) >= 2}
    if not compound:
        return "No compound opportunities (each sector has only 1 trend)."
    lines = [f"  {s}: {len(t)} trends - {t}" for s, t in compound.items()]
    return "Cross-trend opportunities:\n" + "\n".join(lines)


async def run_market_impact(deps: AgentDeps) -> tuple:
    """Run Market Impact Agent. Returns (impacts, result)."""
    prompt = f"Analyze business impact of {len(deps._trend_data)} trends for consulting opportunities."

    agent_result = None
    try:
        from app.tools.provider_manager import ProviderManager
        await ProviderManager.acquire_gcp_rate_limit()
        model = deps.get_model()
        result = await impact_agent.run(prompt, deps=deps, model=model)
        logger.info(f"Impact: {len(deps._impacts)} impacts analyzed")
        agent_result = result.output
    except Exception as e:
        logger.error(f"Impact Agent failed: {e}, using fallback")

    # Build impacts if agent's tool call didn't (e.g. mock mode skips tools)
    if not deps._impacts and deps._trend_data:
        from app.agents.workers.impact_agent import ImpactAnalyzer
        from app.schemas import AgentState
        analyzer = ImpactAnalyzer(mock_mode=deps.mock_mode, deps=deps)
        state = AgentState(trends=deps._trend_data)
        result_state = await analyzer.analyze_impacts(state)
        deps._impacts = result_state.impacts or []

    return deps._impacts, agent_result or ImpactResult(reasoning="Fallback analyzer")
