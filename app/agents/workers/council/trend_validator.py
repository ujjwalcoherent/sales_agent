"""
Stage A: Trend Validation Council.

Validates cluster coherence, event classification, and importance-based
hierarchy (MAJOR/SUB/MICRO) using LLM reasoning instead of volume-based rules.

Replaces the old approach where MAJOR = 8+ articles (regardless of importance).
Now: MAJOR = high business significance as judged by AI.
"""

import asyncio
import logging
from typing import Dict, List, Any

from .schemas import TrendValidation
from ....tools.llm_service import LLMService
from ....config import get_settings

logger = logging.getLogger(__name__)

VALIDATION_SYSTEM_PROMPT = """You are a senior business intelligence analyst at Coherent Market Insights (CMI),
a B2B market research and consulting firm. Your job is to evaluate news trend clusters
and determine their BUSINESS SIGNIFICANCE for CMI's consulting sales opportunities.

CMI offers these 9 services — a trend is only valuable if at least one service can be
pitched to companies affected by it:
1. Procurement Intelligence — supplier profiling, cost analysis, spend optimization
2. Market Intelligence — market sizing, growth forecasts, regulatory landscape
3. Competitive Intelligence — competitor profiling, M&A tracking, benchmarking
4. Market Monitoring — real-time regulatory/economic tracking, early warning systems
5. Industry Analysis — value chain mapping, industry drivers, technology disruptions
6. Technology Research — emerging tech assessment, patent analysis, vendor evaluation
7. Cross-Border Expansion — market entry strategy, regulatory advisory, localization
8. Consumer Insights — consumer behavior, segmentation, brand perception studies
9. Consulting & Advisory — strategic planning, digital transformation, growth strategy

You must classify each trend as:
- MAJOR: High business impact, broad sector effects, clear CMI consulting opportunity.
  At least 2+ CMI services are directly relevant. Companies will need external help.
  Examples: regulatory changes, major M&A, supply chain disruptions, policy shifts.
- SUB: Moderate impact, 1-2 CMI services applicable. Useful but not top-priority.
  Examples: company-specific expansions, product launches, regional market shifts.
- MICRO: Narrow/minor significance. Only marginally relevant to CMI services.
  Examples: routine appointments, minor product updates, local events.
- NOISE: Not a real business trend OR no CMI service can be pitched to affected companies.
  Examples: entertainment, sports, gaming, celebrity news, republished clickbait,
  or unrelated articles grouped together. MUST be discarded.

CRITICAL RULES:
- Article count does NOT determine importance. 4 articles about a critical regulatory
  change is MORE important than 20 republished articles about a product launch.
- If NO CMI service can be pitched to ANY company affected by this trend,
  classify as NOISE regardless of article count or perceived business impact.
- Gaming, entertainment, sports, celebrity, and lifestyle trends are ALWAYS NOISE.
- Focus on trends where companies will need market research, competitive analysis,
  procurement help, or strategic consulting."""


async def validate_trends(
    trends_data: List[Dict[str, Any]],
    llm_service: LLMService = None,
) -> List[TrendValidation]:
    """
    Validate a batch of trends using AI council.

    Args:
        trends_data: List of dicts with keys:
            - trend_id, trend_title, summary, article_count, source_diversity,
            - keywords, entities, coherence_score, signal_strength,
            - article_titles (list of source article titles for evidence)
        llm_service: LLM service instance (created if not provided)

    Returns:
        List of TrendValidation with AI-determined depth and reasoning
    """
    if not trends_data:
        return []

    if llm_service is None:
        llm_service = LLMService(mock_mode=get_settings().mock_mode, lite=True)

    # Process trends in parallel with semaphore
    semaphore = asyncio.Semaphore(5)  # OpenAI primary — 500+ RPM

    async def _validate_one(trend: Dict[str, Any]) -> TrendValidation:
        async with semaphore:
            return await _validate_single_trend(trend, llm_service)

    tasks = [_validate_one(t) for t in trends_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    validations = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Trend validation failed for {trends_data[i].get('trend_title', '?')}: {result}")
            # Fallback: use heuristic
            validations.append(_heuristic_validation(trends_data[i]))
        else:
            validations.append(result)

    return validations


async def _validate_single_trend(
    trend: Dict[str, Any],
    llm_service: LLMService,
) -> TrendValidation:
    """Validate a single trend with LLM reasoning."""
    article_titles = trend.get("article_titles", [])
    articles_text = "\n".join(f"  - {t}" for t in article_titles[:8])

    prompt = f"""Evaluate this news trend cluster for business consulting significance:

TREND: {trend.get('trend_title', 'Unknown')}
SUMMARY: {trend.get('summary', 'No summary')[:500]}
ARTICLES ({trend.get('article_count', 0)} total):
{articles_text}

KEYWORDS: {', '.join(trend.get('keywords', [])[:10])}
ENTITIES: {', '.join(trend.get('entities', [])[:10])}
SOURCE DIVERSITY: {trend.get('source_diversity', 0):.2f}
COHERENCE SCORE: {trend.get('coherence_score', 0):.2f}

Classify this trend and provide your analysis."""

    try:
        from app.schemas.llm_outputs import TrendValidationLLM
        result = await llm_service.run_structured(
            prompt=prompt,
            system_prompt=VALIDATION_SYSTEM_PROMPT,
            output_type=TrendValidationLLM,
        )
        # Auto-classify as NOISE if CMI relevance is very low
        from app.config import get_settings
        _noise_thresh = get_settings().cmi_auto_noise_threshold
        cmi_score = min(1.0, max(0.0, result.cmi_relevance_score))
        validated_depth = result.validated_depth
        if cmi_score < _noise_thresh and validated_depth != "NOISE":
            logger.debug(
                f"Auto-NOISE '{trend.get('trend_title', '?')}': "
                f"cmi_relevance={cmi_score:.2f} < {_noise_thresh}"
            )
            validated_depth = "NOISE"

        return TrendValidation(
            trend_id=trend.get("trend_id", ""),
            importance_score=min(1.0, max(0.0, result.importance_score)),
            validated_depth=validated_depth,
            reasoning=result.reasoning,
            cmi_relevance_score=cmi_score,
            relevant_services=result.relevant_services,
            should_subcluster=result.should_subcluster,
            subcluster_reason=result.subcluster_reason,
            validated_event_type=result.validated_event_type,
            event_type_reasoning=result.event_type_reasoning,
        )
    except Exception as e:
        logger.warning(f"LLM validation failed, using heuristic: {e}")
        return _heuristic_validation(trend)


def _heuristic_validation(trend: Dict[str, Any]) -> TrendValidation:
    """Fallback heuristic when LLM is unavailable."""
    article_count = trend.get("article_count", 0)
    source_diversity = trend.get("source_diversity", 0)
    coherence = trend.get("coherence_score", 0)

    # Simple heuristic combining signals
    score = (
        min(1.0, article_count / 10) * 0.3
        + source_diversity * 0.3
        + coherence * 0.2
        + (0.2 if trend.get("signal_strength") == "strong" else 0.1)
    )

    if score >= 0.6:
        depth = "MAJOR"
    elif score >= 0.35:
        depth = "SUB"
    elif score >= 0.15:
        depth = "MICRO"
    else:
        depth = "NOISE"

    return TrendValidation(
        trend_id=trend.get("trend_id", ""),
        importance_score=round(score, 2),
        validated_depth=depth,
        reasoning=f"Heuristic: {article_count} articles, {source_diversity:.2f} diversity, {coherence:.2f} coherence",
        should_subcluster=article_count >= 8 and score >= 0.5,
        subcluster_reason="Volume-based fallback",
    )
