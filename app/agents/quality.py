"""
Quality Agent — deterministic validation, gating, and retry logic.

Replaced LLM-based quality agent (March 2026 audit):
  - LLM agent always returned quality_score=0.0 (wasted 1-2 LLM calls + 5-10s)
  - Deterministic formula: quality = 0.40 * mean_coherence + 0.30 * (1 - noise_rate) + 0.30 * mean_oss
  - Same retry/pass/fail logic, no LLM call

The quality_score flows into:
  - learning_update_node → signal_bus → threshold_adapter (next run's thresholds)
  - run recordings (data/recordings/*/quality_complete.json)
"""

import logging
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


async def run_quality_check(deps, stage: str) -> QualityVerdict:
    """Deterministic quality gate — no LLM needed.

    Trends: quality = 0.40 * mean_coherence + 0.30 * (1 - noise_rate) + 0.30 * mean_oss
    Impacts: filter by council_confidence >= threshold
    """
    if stage == "trends":
        return _check_trends(deps)
    elif stage == "impacts":
        return _check_impacts(deps)
    else:
        return QualityVerdict(stage=stage, passed=True)


def _check_trends(deps) -> QualityVerdict:
    """Validate trend clustering quality using deterministic formula."""
    tree = getattr(deps, "_trend_tree", None)
    if tree is None:
        return QualityVerdict(stage="trends", passed=True, reasoning="No trend tree")

    major = tree.to_major_trends()
    if not major:
        return QualityVerdict(
            stage="trends", passed=False, quality_score=0.0,
            issues=["Zero clusters found"], reasoning="FAIL: no clusters",
        )

    # Compute signals from existing cluster data
    coherences = [getattr(m, 'coherence_score', 0.5) for m in major]
    mean_coh = sum(coherences) / max(len(coherences), 1)
    min_coh = min(coherences) if coherences else 0.0

    pipeline = getattr(deps, "_pipeline", None)
    noise_rate = getattr(pipeline, "_noise_rate", 0.0) if pipeline else 0.0
    mean_oss = sum(getattr(m, 'oss_score', 0) for m in major) / max(len(major), 1)

    # Deterministic quality formula (weights sum to 1.0)
    quality = round(
        0.40 * mean_coh + 0.30 * (1 - noise_rate) + 0.30 * mean_oss,
        3,
    )

    issues = []
    if mean_coh < 0.40:
        issues.append(f"Low mean coherence: {mean_coh:.3f} (need >= 0.40)")
    if len(major) < 3:
        issues.append(f"Too few clusters: {len(major)} (need >= 3)")
    if min_coh < 0.25:
        issues.append(f"Very low min coherence: {min_coh:.3f}")

    # Decision logic (same thresholds as old LLM agent)
    if not issues:
        passed, should_retry = True, False
    elif mean_coh >= 0.35:
        passed, should_retry = True, True  # borderline → retry if allowed
    else:
        passed, should_retry = False, False

    reasoning = (
        f"Deterministic quality: {quality:.3f} "
        f"(coherence={mean_coh:.3f}, noise={noise_rate:.1%}, oss={mean_oss:.3f}). "
        f"Clusters: {len(major)}, min_coh: {min_coh:.3f}"
    )

    logger.info(f"Quality gate (trends): score={quality:.3f}, passed={passed}, retry={should_retry}")

    return QualityVerdict(
        stage="trends",
        passed=passed,
        should_retry=should_retry,
        items_passed=len(major),
        items_filtered=0,
        quality_score=quality,
        issues=issues,
        reasoning=reasoning,
    )


def _check_impacts(deps) -> QualityVerdict:
    """Validate impact analysis — filter by council confidence threshold."""
    from app.config import get_settings
    settings = get_settings()
    threshold = settings.min_trend_confidence_for_agents

    impacts = getattr(deps, "_impacts", [])
    if not impacts:
        return QualityVerdict(stage="impacts", passed=True, reasoning="No impacts to validate")

    viable = [i for i in impacts if i.council_confidence >= threshold]
    filtered = len(impacts) - len(viable)

    deps._viable_impacts = viable

    return QualityVerdict(
        stage="impacts",
        passed=len(viable) > 0,
        items_passed=len(viable),
        items_filtered=filtered,
        reasoning=f"{len(viable)}/{len(impacts)} passed (threshold={threshold}). Filtered: {filtered}.",
    )
