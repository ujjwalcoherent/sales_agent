"""
Intelligence Pipeline — Entry Point for All 3 Discovery Paths.

Execute: await execute(scope) → IntelligenceResult

Pipeline steps (math gates 1-8, first LLM call at step 7):
  1. FetchAgent      — RSS + Tavily + Google News RSS
  2. DedupAgent      — cosine TF-IDF dedup (threshold=0.85)
  3. FilterAgent     — salience filter + Gap 4 rule
  4. Extract         — NERAgent + NormAgent + ClassifierAgent
  5. SimilarityAgent — 6-signal distance matrix
  6. ClusterAgent    — HAC + HDBSCAN soft + Leiden
  7. ValidationAgent — 7-check math gate (triggers retry via Signal Bus)
  8. SynthesisAgent  — FIRST LLM CALL (label + summary, Reflexion retry)
  9. MatchAgent      — product catalog ↔ cluster matching (math only)
 10. LearningUpdate  — all self-learning agents update via Signal Bus

Principle: deterministic backbone, scoped agency.
Code handles orchestration. LLM handles only synthesis reasoning.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from app.intelligence.config import ClusteringParams, load_adaptive_params
from app.intelligence.models import (
    DiscoveryScope,
    IntelligenceResult,
    PipelineState,
)

logger = logging.getLogger(__name__)


async def execute(
    scope: DiscoveryScope,
    params: Optional[ClusteringParams] = None,
) -> IntelligenceResult:
    """Run the full intelligence pipeline for the given discovery scope.

    Args:
        scope: DiscoveryScope — entry point configuration for the run.
        params: ClusteringParams — algorithm parameters (loaded from
            data/adaptive_thresholds.json if None, falling back to defaults).

    Returns:
        IntelligenceResult with all clusters, entity groups, and match results.
    """
    if params is None:
        params = load_adaptive_params()

    state = PipelineState(scope=scope, run_id=uuid4().hex[:16])
    logger.info(f"[intelligence] run_id={state.run_id} mode={scope.mode} "
                f"scope={scope.companies or scope.industry}")

    # Ensure diagnostics directory exists
    diagnostics_dir = os.path.join("data", "cluster_diagnostics", state.run_id)
    os.makedirs(diagnostics_dir, exist_ok=True)

    try:
        # ── Step 1: Fetch ────────────────────────────────────────────────────
        from app.intelligence.fetch import fetch_articles
        state.raw_articles = await fetch_articles(scope, params)
        state.add_thought("pipeline", f"Fetched {len(state.raw_articles)} raw articles",
                          action="proceeding to dedup")
        logger.info(f"[fetch] {len(state.raw_articles)} raw articles")

        # ── Step 2: Dedup ────────────────────────────────────────────────────
        from app.intelligence.fetch import dedup_articles
        dedup_result = dedup_articles(state.raw_articles, params)
        state.articles = dedup_result.articles
        state.add_thought("pipeline",
                          f"Dedup: {len(state.raw_articles)} → {len(state.articles)} "
                          f"(removed {dedup_result.removed_count})")
        logger.info(f"[dedup] {len(state.articles)} unique articles")

        # ── Step 3: Filter (NLI zero-shot + Gap 4) ───────────────────────────
        from app.intelligence.filter import filter_articles
        filter_result = await filter_articles(state.articles, scope, params)
        state.filtered_articles = filter_result.articles
        state.gap4_dropped_companies = filter_result.gap4_dropped_companies
        state.add_thought("pipeline",
                          f"Filter: {len(state.articles)} → {len(state.filtered_articles)} "
                          f"Gap4 dropped: {state.gap4_dropped_companies}, "
                          f"nli_mean={filter_result.nli_mean_entailment:.3f}")
        logger.info(f"[filter] {len(state.filtered_articles)} relevant articles, "
                    f"nli_mean={filter_result.nli_mean_entailment:.3f}, "
                    f"hypothesis={filter_result.hypothesis_version}, "
                    f"Gap4 dropped: {state.gap4_dropped_companies}")

        if not state.filtered_articles:
            logger.warning("[pipeline] No articles survived filter. Returning empty result.")
            return _build_result(state, diagnostics_dir)

        # ── Step 4: Entity Extraction ─────────────────────────────────────────
        # Uses existing entity extractor until Phase 5 migration is complete
        state.entity_groups, ungrouped = await _extract_entities(
            state.filtered_articles, scope, params
        )
        state.add_thought("pipeline",
                          f"Entity extraction: {len(state.entity_groups)} groups, "
                          f"{len(ungrouped)} ungrouped articles")
        logger.info(f"[extract] {len(state.entity_groups)} entity groups")

        # ── Steps 5-7: Similarity → Cluster → Validate ───────────────────────
        # These use the existing clustering engine until Phase 6 migration
        from app.intelligence.cluster.orchestrator import cluster_and_validate
        state.clusters, state.passed_cluster_ids, state.rejected_cluster_ids, \
            state.validation_results, state.noise_article_indices = \
            await cluster_and_validate(
                state.filtered_articles, state.entity_groups, scope, params, state
            )
        logger.info(f"[cluster] {len(state.passed_cluster_ids)} passed, "
                    f"{len(state.rejected_cluster_ids)} rejected, "
                    f"{len(state.noise_article_indices)} noise")

        # ── Step 8: Synthesis (FIRST LLM CALL) ───────────────────────────────
        from app.intelligence.summarizer import synthesize_clusters
        state.labeled_clusters = await synthesize_clusters(
            state.passed_clusters(), state.filtered_articles, params
        )
        logger.info(f"[synthesis] {len(state.labeled_clusters)} labeled clusters")

        # ── Step 8b: Critic Validation (AutoResearch quality gate) ─────────
        from app.intelligence.summarizer import critic_validate_clusters
        state.labeled_clusters = await critic_validate_clusters(
            state.labeled_clusters, state.filtered_articles, params,
            region=scope.region,
        )

        # ── Step 8c: Evidence Chains ───────────────────────────────────────
        from app.intelligence.summarizer import build_evidence_chain
        for cluster in state.labeled_clusters:
            cluster.evidence_chain = build_evidence_chain(
                cluster, state.filtered_articles
            )

        # ── Step 9: Match Engine ──────────────────────────────────────────────
        if scope.user_products:
            from app.intelligence.match import compute_match_results
            state.match_results = compute_match_results(
                state.labeled_clusters, scope.user_products, params
            )
            logger.info(f"[match] {len(state.match_results)} match results")

        # ── Step 10: Learning Update ──────────────────────────────────────────
        await _update_learning_loops(state, params, filter_result)

    except Exception as exc:
        logger.error(f"[pipeline] Fatal error in run {state.run_id}: {exc}", exc_info=True)
        raise

    return _build_result(state, diagnostics_dir)


async def _extract_entities(
    articles: list,
    scope: DiscoveryScope,
    params: ClusteringParams,
) -> tuple:
    """Bridge to existing entity extractor until Phase 5 migration."""
    try:
        from app.intelligence.engine.extractor import extract_and_group_entities
        # Synchronous call — runs NER + normalize + GLiNER validation
        groups, ungrouped = extract_and_group_entities(articles)
        return groups, ungrouped
    except Exception as exc:
        logger.warning(f"[pipeline] Entity extractor failed: {exc} — using empty groups")
        return [], list(range(len(articles)))


async def _update_learning_loops(
    state: PipelineState,
    params: ClusteringParams,
    filter_result=None,
) -> None:
    """Update all self-learning loops after pipeline completes.

    Wires NLI filter metrics → source bandit reward signal → signal bus.
    This closes the feedback loop: NLI entailment per source → Thompson Sampling
    deprioritizes noisy sources, prioritizes high-quality B2B sources.
    """
    try:
        passed = state.passed_clusters()
        coherence_scores = [c.coherence_score for c in passed if c.coherence_score > 0]
        mean_coherence = sum(coherence_scores) / max(len(coherence_scores), 1)

        nli_mean = getattr(filter_result, "nli_mean_entailment", 0.0) if filter_result else 0.0
        nli_scores_by_source = getattr(filter_result, "nli_scores_by_source", {}) if filter_result else {}
        hypothesis_ver = getattr(filter_result, "hypothesis_version", "v0") if filter_result else "v0"
        auto_accepted = getattr(filter_result, "auto_accepted_count", 0) if filter_result else 0
        auto_rejected = getattr(filter_result, "auto_rejected_count", 0) if filter_result else 0
        total_input = len(state.articles)
        nli_rejection_rate = auto_rejected / max(total_input, 1)

        logger.info(
            f"[pipeline] run_id={state.run_id} "
            f"articles_fetched={len(state.raw_articles)} "
            f"articles_filtered={len(state.filtered_articles)} "
            f"clusters_passed={len(state.passed_cluster_ids)} "
            f"noise_rate={len(state.noise_article_indices) / max(len(state.filtered_articles), 1):.2f} "
            f"mean_coherence={mean_coherence:.3f} "
            f"nli_mean={nli_mean:.3f}"
        )

        # ── Entity quality cache update (self-improving NER) ─────────────────
        # Research: NAACL 2024 naacl-short.49 — pseudo-label propagation +7-12% F1
        # Entities from validated clusters get quality score bumped.
        # Next run: high-quality entities get lower salience threshold → more groups.
        try:
            from app.intelligence.engine.extractor import update_entity_quality
            for cluster in passed:
                entity_names = getattr(cluster, "entity_names", []) or []
                primary = getattr(cluster, "primary_entity", None)
                if primary and primary not in entity_names:
                    entity_names = [primary] + list(entity_names)
                if entity_names:
                    update_entity_quality(entity_names, cluster.coherence_score)
        except Exception as eq_exc:
            logger.debug(f"[pipeline] Entity quality update failed (non-fatal): {eq_exc}")

        # ── Wire NLI scores into signal bus ───────────────────────────────────
        # Source bandit reads this via update_from_run() on next cycle.
        try:
            from app.learning.signal_bus import LearningSignalBus
            bus = LearningSignalBus()
            bus.publish_nli_filter(
                mean_entailment=nli_mean,
                rejection_rate=nli_rejection_rate,
                hypothesis_version=hypothesis_ver,
                hypothesis_updated=False,  # hypothesis_learner.py updates this separately
                scores_by_source=nli_scores_by_source,
            )
        except Exception as bus_exc:
            logger.debug(f"[pipeline] Signal bus NLI publish failed (non-fatal): {bus_exc}")

        # ── Wire NLI scores into source bandit reward ─────────────────────────
        # Research: Chapelle & Li (2011) Thompson Sampling — reward = mean NLI entailment
        # Sources with high NLI entailment scores get prioritized in future runs.
        if nli_scores_by_source:
            try:
                from app.learning.source_bandit import SourceBandit
                bandit = SourceBandit()
                # Build minimal cluster quality map from passed clusters
                cluster_quality_map = {}
                for cluster in passed:
                    for idx in getattr(cluster, "article_indices", []):
                        if idx < len(state.filtered_articles):
                            src = getattr(state.filtered_articles[idx], "source_name", "")
                            if src:
                                cluster_quality_map[src] = cluster_quality_map.get(src, [])
                                cluster_quality_map[src].append(cluster.coherence_score)

                # Aggregate per-source quality
                source_quality = {
                    src: sum(scores) / len(scores)
                    for src, scores in cluster_quality_map.items()
                }

                # Build source_articles: source_name → [article_ids from filtered set]
                source_articles_map: Dict[str, list] = {}
                for i, art in enumerate(state.filtered_articles):
                    src = getattr(art, "source_name", "") or getattr(art, "source_id", "")
                    if src:
                        source_articles_map.setdefault(src, []).append(i)

                # article_labels: article_index → cluster_id (for cluster quality)
                article_labels_map: Dict[str, int] = {}
                cluster_quality_map_final: Dict[int, float] = {}
                for ci, cluster in enumerate(passed):
                    cluster_quality_map_final[ci] = cluster.coherence_score
                    for idx in getattr(cluster, "article_indices", []):
                        article_labels_map[str(idx)] = ci

                bandit.update_from_run(
                    source_articles=source_articles_map,
                    article_labels=article_labels_map,
                    cluster_quality=cluster_quality_map_final,
                    nli_scores_by_source=nli_scores_by_source,
                )
                logger.info(f"[pipeline] Source bandit updated: {len(nli_scores_by_source)} sources")
            except Exception as bandit_exc:
                logger.debug(f"[pipeline] Source bandit update failed (non-fatal): {bandit_exc}")

        # ── Run HypothesisLearner if enough feedback ───────────────────────────
        # SetFit (arXiv:2209.11055): closes the human feedback → NLI filter loop.
        # Non-blocking: runs in background after pipeline completes.
        try:
            from app.learning.hypothesis_learner import HypothesisLearner
            learner = HypothesisLearner()
            updated = await learner.maybe_update(nli_mean_entailment=nli_mean)
            if updated:
                logger.info("[pipeline] NLI hypothesis updated by SetFit learner")
                try:
                    from app.learning.signal_bus import LearningSignalBus
                    bus = LearningSignalBus()
                    bus.publish_nli_filter(
                        mean_entailment=nli_mean,
                        rejection_rate=nli_rejection_rate,
                        hypothesis_version=hypothesis_ver,
                        hypothesis_updated=True,
                        scores_by_source=nli_scores_by_source,
                    )
                except Exception:
                    pass
        except Exception as learner_exc:
            logger.debug(f"[pipeline] HypothesisLearner failed (non-fatal): {learner_exc}")

    except Exception as exc:
        logger.warning(f"[pipeline] Learning update failed (non-fatal): {exc}")


def _build_result(state: PipelineState, diagnostics_dir: str) -> IntelligenceResult:
    """Build final IntelligenceResult from pipeline state."""
    passed = state.passed_clusters()
    labeled = state.labeled_clusters or passed

    coherence_scores = [c.coherence_score for c in labeled if c.coherence_score > 0]
    fit_scores = [m.fit_score for m in state.match_results]

    return IntelligenceResult(
        run_id=state.run_id,
        scope=state.scope,
        completed_at=datetime.now(timezone.utc),
        clusters=labeled,
        entity_groups=state.entity_groups,
        match_results=state.match_results,
        total_articles_fetched=len(state.raw_articles),
        total_articles_post_filter=len(state.filtered_articles),
        total_clusters=len(labeled),
        noise_rate=len(state.noise_article_indices) / max(len(state.filtered_articles), 1),
        mean_coherence=sum(coherence_scores) / max(len(coherence_scores), 1),
        mean_fit_score=sum(fit_scores) / max(len(fit_scores), 1),
        gap4_dropped_companies=state.gap4_dropped_companies,
        thought_log=state.thought_log,
        rounds_completed=state.round_number,
        agent_requests_processed=sum(1 for r in state.agent_requests if r.resolved),
        diagnostics_dir=diagnostics_dir,
    )
