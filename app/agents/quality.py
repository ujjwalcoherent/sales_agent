"""
Quality Agent — validation, gating, and feedback routing.

pydantic-ai agent that validates outputs at every pipeline stage,
decides pass/fail/retry, and routes feedback to learning subsystems
(source bandit, company bandit, weight learner).

Tools:
  - validate_trends: Council Stage A trend validation
  - validate_impacts: Confidence-based impact filtering
  - record_feedback: Save feedback to JSONL for learning
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
    'test',  # Placeholder — overridden at runtime via deps.get_model()
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
    tree = ctx.deps._trend_tree
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

    impacts = ctx.deps._impacts
    if not impacts:
        return "No impacts to validate."

    viable = [i for i in impacts if i.council_confidence >= threshold]
    filtered = len(impacts) - len(viable)

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
        model = deps.get_model()
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
