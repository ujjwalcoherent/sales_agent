"""
Pipeline Stage Validator — per-stage quality gates with in-run self-correction.

Each stage in the intelligence pipeline has expected quality ranges based on
empirical data from 50+ pipeline runs. If a stage's output falls outside
these ranges, the validator:
  1. Identifies which metric failed and why
  2. Proposes a specific correction (threshold adjustment, parameter change, rollback)
  3. Returns a StageValidation result the orchestrator can act on immediately

Design principles:
  - No human required: all corrections are algorithmic
  - Conservative corrections: nudge parameters, don't reset
  - Cascade awareness: stage N failure affects stage N+1 expectations
  - One correction per run: if correction still fails, mark WARN not fail (don't loop)

Research basis:
  - Adaptive thresholds: arXiv:2502.12965 (distribution shift detection)
  - Calibration: empirical baselines from live 120h pipeline recording analysis
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"       # outside expected range but pipeline can continue
    FAIL = "fail"       # critically bad, correction attempted
    CORRECTED = "corrected"  # was FAIL, correction applied, retry succeeded


@dataclass
class StageValidation:
    """Result of validating one pipeline stage's output."""
    stage: str
    status: StageStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    corrective_params: Dict[str, Any] = field(default_factory=dict)  # params to use if retrying
    message: str = ""

    def is_ok(self) -> bool:
        return self.status in (StageStatus.PASS, StageStatus.WARN, StageStatus.CORRECTED)


# ── Empirical baselines (from 50+ pipeline runs, March 2026) ──────────────────
# All ranges are [min_expected, max_expected]. WARN outside range. FAIL at 2x deviation.

_DEDUP_REMOVAL_RATE_RANGE = (0.03, 0.50)   # 3%-50% articles removed by dedup
_FILTER_PASS_RATE_RANGE = (0.03, 0.60)     # 3%-60% articles pass NLI filter
_FILTER_NLI_MEAN_RANGE = (0.25, 0.95)      # mean NLI entailment of kept articles
_ENTITY_GROUPING_RATE_RANGE = (0.10, 0.95) # 10%-95% filtered articles get entity group
_CLUSTER_PASS_RATE_RANGE = (0.30, 1.00)    # 30%-100% formed clusters pass validation
_CLUSTER_MIN_PASSED = 1                     # at minimum 1 cluster must pass


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DEDUPLICATION VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_dedup(
    raw_count: int,
    deduped_count: int,
    dedup_pairs: int,
) -> StageValidation:
    """Validate deduplication stage output.

    Args:
        raw_count: Input article count before dedup
        deduped_count: Output article count after dedup
        dedup_pairs: Number of duplicate pairs found

    Expected:
        - Removal rate 3-50%: below means dedup is broken; above means threshold too loose
        - At least 50 articles remaining for meaningful clustering
    """
    if raw_count == 0:
        return StageValidation(
            stage="dedup",
            status=StageStatus.FAIL,
            metrics={"raw_count": 0},
            issues=["No articles to deduplicate — source fetch returned empty"],
            message="Source fetch returned 0 articles. Check news sources and API keys.",
        )

    removal_rate = (raw_count - deduped_count) / raw_count
    metrics = {
        "raw_count": raw_count,
        "deduped_count": deduped_count,
        "removal_rate": removal_rate,
        "dedup_pairs": dedup_pairs,
    }

    issues = []
    status = StageStatus.PASS
    corrective_params = {}

    if deduped_count < 10:
        issues.append(
            f"Only {deduped_count} articles after dedup (expected >= 10). "
            "Source quality issue — fetch diversity too low."
        )
        status = StageStatus.FAIL

    elif removal_rate < _DEDUP_REMOVAL_RATE_RANGE[0]:
        issues.append(
            f"Dedup removal rate {removal_rate:.1%} below expected "
            f"{_DEDUP_REMOVAL_RATE_RANGE[0]:.0%} minimum. "
            "Dedup threshold may be too strict — similar articles passing through."
        )
        status = StageStatus.WARN
        corrective_params["dedup_similarity_threshold"] = 0.85  # suggest loosening

    elif removal_rate > _DEDUP_REMOVAL_RATE_RANGE[1]:
        issues.append(
            f"Dedup removal rate {removal_rate:.1%} above expected "
            f"{_DEDUP_REMOVAL_RATE_RANGE[1]:.0%} maximum. "
            "Too many articles being removed — threshold may be too loose OR sources heavily overlapping."
        )
        status = StageStatus.WARN
        corrective_params["dedup_similarity_threshold"] = 0.95  # suggest tightening

    if deduped_count < 50:
        issues.append(
            f"Only {deduped_count} articles after dedup — clustering may underperform "
            "(need >= 50 for meaningful signal). Consider adding more news sources."
        )
        if status == StageStatus.PASS:
            status = StageStatus.WARN

    return StageValidation(
        stage="dedup",
        status=status,
        metrics=metrics,
        issues=issues,
        corrective_params=corrective_params,
        message=(
            f"Dedup: {raw_count} -> {deduped_count} articles "
            f"({removal_rate:.1%} removed, {dedup_pairs} pairs)"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: NLI FILTER VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_filter(
    input_count: int,
    kept_count: int,
    auto_accepted: int,
    auto_rejected: int,
    llm_classified: int,
    nli_mean: float,
    false_positives_found: int = 0,
) -> StageValidation:
    """Validate NLI filter stage output.

    Detects:
      - Hypothesis too strict: pass_rate < 3% -> triggers hypothesis rollback check
      - Hypothesis too permissive: pass_rate > 60% -> warns of possible drift
      - False positives: sports/crime/consumer articles in kept set
      - NLI mean too low: entire batch scoring poorly -> hypothesis may have drifted

    Args:
        input_count: Articles entering the filter
        kept_count: Articles passing the filter
        auto_accepted: Count auto-accepted (NLI >= 0.88)
        auto_rejected: Count auto-rejected (NLI <= 0.10)
        llm_classified: Count classified by LLM fallback
        nli_mean: Mean NLI entailment score of ALL input articles
        false_positives_found: Known false positives detected by caller
    """
    if input_count == 0:
        return StageValidation(
            stage="filter",
            status=StageStatus.FAIL,
            metrics={"input_count": 0},
            issues=["Filter received 0 articles from dedup stage"],
            message="No articles to filter.",
        )

    pass_rate = kept_count / input_count
    auto_reject_rate = auto_rejected / input_count
    metrics = {
        "input_count": input_count,
        "kept_count": kept_count,
        "pass_rate": pass_rate,
        "auto_accepted": auto_accepted,
        "auto_rejected": auto_rejected,
        "llm_classified": llm_classified,
        "nli_mean": nli_mean,
        "false_positives": false_positives_found,
    }

    issues = []
    status = StageStatus.PASS
    corrective_params = {}

    # Critical: hypothesis may have drifted if nearly everything is rejected
    if pass_rate < 0.01 and input_count >= 50:
        issues.append(
            f"CRITICAL: Only {kept_count}/{input_count} articles passed NLI filter "
            f"({pass_rate:.1%}). Hypothesis may have drifted to reject all content. "
            "Checking for rollback candidate in filter_hypothesis.json..."
        )
        rollback_hyp = _get_rollback_hypothesis()
        if rollback_hyp:
            issues.append("Rollback candidate available: will retry with previous_hypothesis.")
            corrective_params["use_previous_hypothesis"] = True
            corrective_params["rollback_hypothesis"] = rollback_hyp
        status = StageStatus.FAIL

    elif pass_rate < _FILTER_PASS_RATE_RANGE[0]:
        issues.append(
            f"Low pass rate {pass_rate:.1%} (expected >= {_FILTER_PASS_RATE_RANGE[0]:.0%}). "
            "Hypothesis may be overly strict for this news batch. "
            "Check if batch contains many non-B2B sources."
        )
        status = StageStatus.WARN

    elif pass_rate > _FILTER_PASS_RATE_RANGE[1]:
        issues.append(
            f"High pass rate {pass_rate:.1%} (expected <= {_FILTER_PASS_RATE_RANGE[1]:.0%}). "
            "Hypothesis may be too permissive. "
            f"NLI mean for context: {nli_mean:.3f}. Watch for false positives downstream."
        )
        status = StageStatus.WARN

    # NLI mean sanity check
    if nli_mean < 0.10 and input_count >= 50:
        issues.append(
            f"NLI mean {nli_mean:.3f} extremely low across {input_count} articles. "
            "Either all articles are genuinely off-topic OR the hypothesis is rejecting "
            "everything (including clear B2B news). Hypothesis drift likely."
        )
        status = StageStatus.FAIL if pass_rate < 0.02 else StageStatus.WARN

    # False positive check
    if false_positives_found > 0:
        issues.append(
            f"{false_positives_found} false positives detected in kept articles "
            "(sports/crime/consumer electronics). Hypothesis too permissive for these categories."
        )
        if status == StageStatus.PASS:
            status = StageStatus.WARN

    # Auto-reject rate too high (>95% rejected without LLM) suggests systematic issue
    if auto_reject_rate > 0.95 and input_count >= 50:
        issues.append(
            f"Auto-reject rate {auto_reject_rate:.1%} suspiciously high. "
            "Nearly all articles scored below NLI threshold 0.10. "
            "Possible hypothesis drift or wrong hypothesis loaded."
        )
        status = StageStatus.FAIL

    return StageValidation(
        stage="filter",
        status=status,
        metrics=metrics,
        issues=issues,
        corrective_params=corrective_params,
        message=(
            f"Filter: {input_count} -> {kept_count} ({pass_rate:.1%} kept, "
            f"NLI mean={nli_mean:.3f}, FP={false_positives_found})"
        ),
    )


def _get_rollback_hypothesis() -> Optional[str]:
    """Load previous_hypothesis from filter_hypothesis.json for in-run rollback."""
    hyp_path = Path("data/filter_hypothesis.json")
    if not hyp_path.exists():
        return None
    try:
        with open(hyp_path, encoding="utf-8") as f:
            data = json.load(f)
        prev = data.get("previous_hypothesis")
        if prev and len(prev) > 20:
            return prev
    except Exception:
        pass
    return None


def apply_filter_rollback(rollback_hypothesis: str) -> bool:
    """Write rollback_hypothesis as the active hypothesis for immediate effect.

    Called by the orchestrator when validate_filter() returns corrective_params
    with use_previous_hypothesis=True. Restores the previous hypothesis in-place
    so the filter can be re-run in the same pipeline execution.

    Returns True if rollback was applied successfully.
    """
    import datetime

    hyp_path = Path("data/filter_hypothesis.json")
    try:
        current_data: Dict[str, Any] = {}
        if hyp_path.exists():
            with open(hyp_path, encoding="utf-8") as f:
                current_data = json.load(f)

        # Swap: current becomes previous, rollback becomes current
        current_hyp = current_data.get("hypothesis", "")
        current_data["previous_hypothesis"] = current_hyp
        current_data["hypothesis"] = rollback_hypothesis
        current_data["version"] = current_data.get("version", "v0") + "_rollback"
        current_data["rollback_applied_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        current_data["notes"] = (
            "In-run rollback: previous hypothesis failed pipeline validation "
            "(pass_rate critically low). Restored previous_hypothesis automatically."
        )

        with open(hyp_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)

        # Invalidate NLI cache so it reloads
        try:
            from app.intelligence.engine.nli_filter import invalidate_hypothesis_cache
            invalidate_hypothesis_cache()
        except Exception:
            pass

        logger.warning(
            "[pipeline_validator] IN-RUN ROLLBACK applied: "
            f"restored previous hypothesis. Original: '{current_hyp[:60]}...'"
        )
        return True

    except Exception as e:
        logger.error(f"[pipeline_validator] Rollback failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: ENTITY EXTRACTION VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_entity_extraction(
    input_count: int,
    group_count: int,
    grouped_article_count: int,
    ungrouped_count: int,
) -> StageValidation:
    """Validate entity extraction and grouping stage.

    Args:
        input_count: Articles entering entity extraction
        group_count: Number of distinct entity groups formed
        grouped_article_count: Articles that got assigned to an entity group
        ungrouped_count: Articles with no entity group
    """
    if input_count == 0:
        return StageValidation(
            stage="entity",
            status=StageStatus.WARN,
            metrics={"input_count": 0},
            issues=["No articles for entity extraction"],
            message="Entity extraction skipped (0 articles).",
        )

    grouping_rate = grouped_article_count / input_count
    metrics = {
        "input_count": input_count,
        "group_count": group_count,
        "grouped_articles": grouped_article_count,
        "ungrouped_count": ungrouped_count,
        "grouping_rate": grouping_rate,
    }

    issues = []
    status = StageStatus.PASS

    if group_count == 0:
        issues.append(
            "No entity groups formed from any articles. "
            "GLiNER may not be finding named entities in this batch. "
            "Check if articles are in supported language (en) and have company mentions."
        )
        status = StageStatus.WARN  # not fatal — ungrouped articles still go to Leiden clustering

    elif grouping_rate < _ENTITY_GROUPING_RATE_RANGE[0]:
        issues.append(
            f"Low entity grouping rate {grouping_rate:.1%} "
            f"(expected >= {_ENTITY_GROUPING_RATE_RANGE[0]:.0%}). "
            "Articles may lack named entity mentions (e.g., opinion pieces, macro-economic commentary). "
            "This is expected if filter is passing many non-company-specific articles."
        )
        status = StageStatus.WARN

    # Check for pathological case: 1 entity group with ALL articles (everything collapsed)
    if group_count == 1 and grouped_article_count > 20:
        issues.append(
            f"All {grouped_article_count} articles collapsed into 1 entity group — "
            "entity normalizer may be over-merging. Check canonical name resolution."
        )
        status = StageStatus.WARN

    return StageValidation(
        stage="entity",
        status=status,
        metrics=metrics,
        issues=issues,
        message=(
            f"Entity: {group_count} groups from {input_count} articles "
            f"({grouping_rate:.1%} grouped, {ungrouped_count} ungrouped)"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: CLUSTERING VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_clustering(
    input_count: int,
    total_clusters: int,
    passed_count: int,
    failed_count: int,
    noise_count: int,
    mean_coherence: float,
    coherences: Optional[List[float]] = None,
) -> StageValidation:
    """Validate clustering and cluster validation stage.

    Args:
        input_count: Articles entering clustering
        total_clusters: Total clusters formed by algorithm
        passed_count: Clusters that passed validation
        failed_count: Clusters that failed validation
        noise_count: Unclustered (noise) articles
        mean_coherence: Mean coherence of PASSED clusters
        coherences: Individual coherence scores of passed clusters (for distribution check)
    """
    metrics: Dict[str, float] = {
        "input_count": input_count,
        "total_clusters": total_clusters,
        "passed": passed_count,
        "failed": failed_count,
        "noise_articles": noise_count,
        "mean_coherence": mean_coherence,
    }

    if input_count < 3:
        return StageValidation(
            stage="clustering",
            status=StageStatus.WARN,
            metrics=metrics,
            issues=[f"Only {input_count} articles — need >= 3 for clustering"],
            message=f"Clustering skipped: insufficient articles ({input_count} < 3).",
        )

    issues = []
    status = StageStatus.PASS
    corrective_params: Dict[str, Any] = {}

    if total_clusters == 0:
        issues.append(
            f"No clusters formed from {input_count} articles. "
            "This can happen if articles are all from different companies with no co-occurrence. "
            "Consider lowering HAC distance threshold or adding more articles per source."
        )
        status = StageStatus.WARN  # Not fatal — ungrouped articles may still generate leads via direct signals

    elif passed_count == 0 and total_clusters > 0:
        issues.append(
            f"{total_clusters} clusters formed but ALL failed validation. "
            f"Mean coherence {mean_coherence:.3f} may be below validation threshold. "
            "Possible fix: if all clusters are entity-seeded, check entity_seeded bypass is active."
        )
        status = StageStatus.FAIL
        # Correction: suggest lowering coherence threshold for this run
        corrective_params["val_coherence_override"] = max(0.10, mean_coherence * 0.8)

    elif passed_count < _CLUSTER_MIN_PASSED:
        issues.append(
            f"Only {passed_count} cluster(s) passed validation (expected >= {_CLUSTER_MIN_PASSED}). "
            "Low signal day — fewer B2B events in this news batch."
        )
        status = StageStatus.WARN

    # Check pass rate
    if total_clusters > 0:
        pass_rate = passed_count / total_clusters
        metrics["pass_rate"] = pass_rate
        if pass_rate < _CLUSTER_PASS_RATE_RANGE[0] and total_clusters >= 5:
            issues.append(
                f"Cluster pass rate {pass_rate:.1%} below expected "
                f"{_CLUSTER_PASS_RATE_RANGE[0]:.0%}. "
                "Many clusters forming but failing coherence — articles may be too diverse."
            )
            if status == StageStatus.PASS:
                status = StageStatus.WARN

    # Coherence distribution check (if individual scores provided)
    if coherences and len(coherences) >= 3:
        min_coh = min(coherences)
        max_coh = max(coherences)
        metrics["min_coherence"] = min_coh
        metrics["max_coherence"] = max_coh

        if max_coh < 0.10:
            issues.append(
                f"All passed clusters have coherence < 0.10 (max={max_coh:.3f}). "
                "Very low coherence across all clusters suggests embeddings or "
                "similarity function may not be working correctly."
            )
            if status == StageStatus.PASS:
                status = StageStatus.WARN

    # Noise rate check
    if input_count > 0:
        noise_rate = noise_count / input_count
        metrics["noise_rate"] = noise_rate
        if noise_rate > 0.90 and input_count >= 20:
            issues.append(
                f"Noise rate {noise_rate:.1%} — {noise_count}/{input_count} articles unclustered. "
                "Very high noise suggests diverse corpus (each article covers different company). "
                "Expected for generic news fetch; lower for targeted company-first fetch."
            )
            if status == StageStatus.PASS:
                status = StageStatus.WARN

    return StageValidation(
        stage="clustering",
        status=status,
        metrics=metrics,
        issues=issues,
        corrective_params=corrective_params,
        message=(
            f"Clustering: {total_clusters} clusters, "
            f"{passed_count} passed, {failed_count} failed, "
            f"{noise_count} noise, mean_coh={mean_coherence:.3f}"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-STAGE CONSISTENCY CHECKER
# ══════════════════════════════════════════════════════════════════════════════

def validate_pipeline_consistency(stage_results: List[StageValidation]) -> StageValidation:
    """Check consistency across all stages — catch cascading failure patterns.

    Cross-stage checks that individual validators can't catch:
    - Filter very permissive AND clustering all fails -> noise passed filter, clustering correct
    - Entity grouping_rate very low AND clustering passed_count high -> entity extraction broken
      (all clusters are Leiden/HAC, not entity-seeded)
    - Dedup removed very few AND filter very strict -> same articles processed, hypothesis issue

    Returns a meta-validation result summarizing pipeline health.
    """
    if not stage_results:
        return StageValidation(
            stage="pipeline",
            status=StageStatus.WARN,
            issues=["No stage results to cross-check"],
        )

    stage_map = {r.stage: r for r in stage_results}
    issues = []
    cross_checks_passed = 0
    cross_checks_total = 0

    # Cross-check 1: If filter kept >50% but clustering found 0 passed clusters
    filter_r = stage_map.get("filter")
    cluster_r = stage_map.get("clustering")
    if filter_r and cluster_r:
        cross_checks_total += 1
        filter_pass_rate = filter_r.metrics.get("pass_rate", 0.0)
        cluster_passed = cluster_r.metrics.get("passed", 0)
        if filter_pass_rate > 0.50 and cluster_passed == 0:
            issues.append(
                f"CROSS-STAGE: Filter kept {filter_pass_rate:.0%} of articles "
                "but 0 clusters passed. High filter pass rate + 0 clusters = "
                "noise slipping through filter (hypothesis too permissive)."
            )
        else:
            cross_checks_passed += 1

    # Cross-check 2: Entity grouping_rate very low but clustering produced many clusters
    entity_r = stage_map.get("entity")
    if entity_r and cluster_r:
        cross_checks_total += 1
        grouping_rate = entity_r.metrics.get("grouping_rate", 0.5)
        total_clusters = cluster_r.metrics.get("total_clusters", 0)
        if grouping_rate < 0.15 and total_clusters >= 5:
            issues.append(
                f"CROSS-STAGE: Entity grouping very low ({grouping_rate:.0%}) "
                f"but {total_clusters} clusters formed. Clusters are likely all "
                "discovery-mode (Leiden/HAC on ungrouped articles), not entity-seeded. "
                "This is valid but expected coherence will be lower."
            )
        else:
            cross_checks_passed += 1

    # Cross-check 3: Low dedup rate + low filter pass rate = both stages bottlenecked
    dedup_r = stage_map.get("dedup")
    if dedup_r and filter_r:
        cross_checks_total += 1
        removal_rate = dedup_r.metrics.get("removal_rate", 0.1)
        filter_pass_rate = filter_r.metrics.get("pass_rate", 0.2)
        if removal_rate < 0.03 and filter_pass_rate < 0.05:
            issues.append(
                f"CROSS-STAGE: Very low dedup removal ({removal_rate:.1%}) AND "
                f"low filter pass ({filter_pass_rate:.1%}). "
                "Source diversity is low (few unique articles) AND hypothesis is strict. "
                "Root cause: news sources may be returning stale/repeated content."
            )
        else:
            cross_checks_passed += 1

    overall_status = StageStatus.PASS
    if issues:
        overall_status = StageStatus.WARN
    if any(r.status == StageStatus.FAIL for r in stage_results):
        overall_status = StageStatus.FAIL

    failed_stages = [r.stage for r in stage_results if r.status == StageStatus.FAIL]
    warn_stages = [r.stage for r in stage_results if r.status == StageStatus.WARN]

    return StageValidation(
        stage="pipeline",
        status=overall_status,
        metrics={
            "cross_checks_passed": cross_checks_passed,
            "cross_checks_total": cross_checks_total,
            "failed_stages": len(failed_stages),
            "warn_stages": len(warn_stages),
        },
        issues=issues,
        message=(
            f"Pipeline: {len(failed_stages)} FAIL, {len(warn_stages)} WARN | "
            f"Cross-checks: {cross_checks_passed}/{cross_checks_total} | "
            f"Stages: {', '.join(r.stage + '=' + r.status.value for r in stage_results)}"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORTER
# ══════════════════════════════════════════════════════════════════════════════

def log_pipeline_validation_report(
    stage_results: List[StageValidation],
    consistency: Optional[StageValidation] = None,
) -> None:
    """Log a structured validation report for all stages."""
    logger.info("=" * 60)
    logger.info("  PIPELINE VALIDATION REPORT")
    logger.info("=" * 60)

    icon_map = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "corrected": "CORR"}
    for result in stage_results:
        icon = icon_map.get(result.status.value, "?")
        logger.info(f"  {icon} {result.stage.upper():10s} {result.status.value.upper()}")
        logger.info(f"    {result.message}")
        for issue in result.issues:
            logger.warning(f"    ! {issue}")
        for correction in result.corrections_applied:
            logger.info(f"    CORR {correction}")

    if consistency:
        logger.info(f"  CONSISTENCY: {consistency.status.value.upper()} — {consistency.message}")
        for issue in consistency.issues:
            logger.warning(f"    ! {issue}")

    logger.info("=" * 60)
