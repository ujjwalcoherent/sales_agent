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

import copy
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from app.intelligence.config import ClusteringParams, load_adaptive_params
from app.intelligence.models import (
    DiscoveryScope,
    IntelligenceResult,
    PipelineState,
)

logger = logging.getLogger(__name__)


def _coerce_to_raw_articles(articles: List) -> List:
    """Convert NewsArticle (or any duck-typed article) to RawArticle.

    Used when mock articles from source_intel_node are passed directly to
    the intelligence pipeline, bypassing the live fetch step.
    Only converts if the objects are not already RawArticle instances.
    """
    from app.intelligence.models import RawArticle

    raw = []
    for a in articles:
        if isinstance(a, RawArticle):
            raw.append(a)
            continue
        # Duck-type conversion: extract common fields
        raw.append(RawArticle(
            url=str(getattr(a, "url", "") or ""),
            title=str(getattr(a, "title", "") or ""),
            summary=str(getattr(a, "summary", "") or getattr(a, "content", "") or ""),
            source_name=str(
                getattr(a, "source_name", "") or
                getattr(a, "source_id", "") or ""
            ),
            source_url=str(getattr(a, "url", "") or ""),
            published_at=getattr(a, "published_at", None) or datetime.now(timezone.utc),
            fetch_method="mock",
        ))
    return raw


async def execute(
    scope: DiscoveryScope,
    params: Optional[ClusteringParams] = None,
    pre_fetched_articles: Optional[List] = None,
) -> IntelligenceResult:
    """Run the full intelligence pipeline for the given discovery scope.

    Args:
        scope: DiscoveryScope — entry point configuration for the run.
        params: ClusteringParams — algorithm parameters (loaded from
            data/adaptive_thresholds.json if None, falling back to defaults).
        pre_fetched_articles: Optional list of RawArticle (or NewsArticle convertible
            to RawArticle) to use directly, skipping the fetch step. Used in mock
            mode where source_intel_node already populated deps._articles.

    Returns:
        IntelligenceResult with all clusters, entity groups, and match results.
    """
    if params is None:
        params = load_adaptive_params()

    state = PipelineState(scope=scope, run_id=uuid4().hex[:16])
    logger.info(f"[intelligence] run_id={state.run_id} mode={scope.mode} "
                f"scope={scope.companies or scope.industry}")

    try:
        # ── Step 1: Fetch (or use pre-fetched) ───────────────────────────────
        if pre_fetched_articles is not None:
            # Convert NewsArticle → RawArticle if needed (mock mode bridge).
            # Avoids a live network call when mock articles already exist.
            state.raw_articles = _coerce_to_raw_articles(pre_fetched_articles)
            logger.info(f"[fetch] Using {len(state.raw_articles)} pre-fetched articles (skipping live fetch)")
        else:
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
            return _build_result(state)

        # ── Mock fast-path: skip entity extraction + NVIDIA embeddings ─────────
        # Mock articles are pre-curated — NER produces 0 useful entity groups
        # (entities found but statistically filtered). Skip SpaCy + GLiNER (~8s
        # cold) and use TF-IDF embeddings instead of NVIDIA API (~1.3s).
        _is_mock = all(
            getattr(a, "fetch_method", "") == "mock"
            for a in state.filtered_articles
        )
        if _is_mock:
            state.entity_groups = []
            ungrouped = list(range(len(state.filtered_articles)))
            logger.info(f"[extract] Mock fast-path: skipped NER, {len(ungrouped)} articles ungrouped")

            from app.intelligence.cluster.orchestrator import _tfidf_fallback_embeddings
            texts = [
                f"{getattr(a, 'title', '')}. {(getattr(a, 'summary', '') or '')[:300]}"
                for a in state.filtered_articles
            ]
            precomputed_embs = _tfidf_fallback_embeddings(texts, dim=512)
            logger.info(f"[embed] Mock fast-path: TF-IDF {precomputed_embs.shape} (skipped NVIDIA API)")
        else:
            # ── Step 4: Entity Extraction ─────────────────────────────────────
            state.entity_groups, ungrouped = await _extract_entities(
                state.filtered_articles, scope, params
            )
            state.add_thought("pipeline",
                              f"Entity extraction: {len(state.entity_groups)} groups, "
                              f"{len(ungrouped)} ungrouped articles")
            logger.info(f"[extract] {len(state.entity_groups)} entity groups")

            # ── Entity-group cleanup: remove bad names ────────────────────────
            if state.entity_groups:
                valid_groups = []
                bad_removed = 0
                try:
                    from app.agents.leads import is_company_description
                    for g in state.entity_groups:
                        name = getattr(g, "canonical_name", "")
                        if is_company_description(name) or len(name.strip()) <= 1 or name.strip().isdigit():
                            ungrouped.extend(getattr(g, "article_indices", []))
                            logger.info(f"[entity] Removed bad entity group: '{name}'")
                            bad_removed += 1
                        else:
                            valid_groups.append(g)
                    if bad_removed:
                        state.entity_groups = valid_groups
                        logger.info(f"[entity] Removed {bad_removed} bad entity groups")
                except ImportError:
                    pass

            # ── Step 4b: Precompute embeddings ────────────────────────────────
            from app.intelligence.cluster.orchestrator import _get_embeddings
            precomputed_embs = await _get_embeddings(state.filtered_articles)
            logger.info(f"[embed] Precomputed {precomputed_embs.shape} embeddings for clustering")

        # ── Steps 5-7: Similarity → Cluster → Validate ───────────────────────
        from app.intelligence.cluster.orchestrator import cluster_and_validate
        state.clusters, state.passed_cluster_ids, state.rejected_cluster_ids, \
            state.validation_results, state.noise_article_indices = \
            await cluster_and_validate(
                state.filtered_articles, state.entity_groups, scope, params, state,
                precomputed_embeddings=precomputed_embs,
            )
        logger.info(f"[cluster] {len(state.passed_cluster_ids)} passed, "
                    f"{len(state.rejected_cluster_ids)} rejected, "
                    f"{len(state.noise_article_indices)} noise")

        # ── Zero-cluster fallback: re-validate with lowered coherence ─────────
        if len(state.passed_cluster_ids) == 0 and len(state.clusters) > 0:
            passed_clusters = state.passed_clusters()
            coherences = [c.coherence_score for c in state.clusters if c.coherence_score > 0]
            mean_coh = sum(coherences) / max(len(coherences), 1)
            override = max(0.10, mean_coh * 0.8)
            logger.info(
                f"[cluster] All {len(state.clusters)} clusters failed validation "
                f"(mean_coherence={mean_coh:.3f}). Re-validating with coherence override {override:.3f}"
            )
            from app.intelligence.cluster.orchestrator import _run_validation
            _revalidated = await _run_validation(
                state.clusters, state.filtered_articles, state.entity_groups,
                _with_coherence_override(params, override),
            )
            if _revalidated:
                new_passed, new_rejected, new_vals = _revalidated
                state.passed_cluster_ids = new_passed
                state.rejected_cluster_ids = new_rejected
                state.validation_results = new_vals
                logger.info(f"[cluster] After override: {len(new_passed)} passed")

        # ── Step 8: Synthesis (FIRST LLM CALL) ───────────────────────────────
        # Mock fast-path: skip real LLM calls for synthesis + critic.
        # Mock articles are pre-curated — LLM labeling adds ~8s with no benefit.
        _is_mock = all(
            getattr(a, "fetch_method", "") == "mock"
            for a in state.filtered_articles
        )
        if _is_mock:
            passed = state.passed_clusters()
            for cluster in passed:
                _mock_synthesize_cluster(cluster, state.filtered_articles)
            state.labeled_clusters = passed
            logger.info(f"[synthesis] Mock fast-path: {len(passed)} clusters labeled")
        else:
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

        # ── Step 8d: Populate cluster industries from article labels ─────────
        _populate_cluster_industries(state.labeled_clusters, state.filtered_articles)

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

    return _build_result(state)


def _with_coherence_override(params: ClusteringParams, override: float) -> ClusteringParams:
    """Create a copy of ClusteringParams with val_coherence_min overridden.

    Used by SPOC correction when all clusters fail validation and we need
    to re-validate with a lowered coherence threshold.
    """
    params_copy = copy.copy(params)
    params_copy.val_coherence_min = override
    return params_copy


async def _extract_entities(
    articles: list,
    scope: DiscoveryScope,
    params: ClusteringParams,
) -> tuple:
    """Extract and group named entities from filtered articles."""
    try:
        from app.intelligence.engine.extractor import extract_and_group_entities
        # Synchronous call — runs NER + normalize + GLiNER validation
        groups, ungrouped = extract_and_group_entities(articles)
        return groups, ungrouped
    except Exception as exc:
        logger.warning(f"[pipeline] Entity extractor failed: {exc} — using empty groups")
        return [], list(range(len(articles)))


def _mock_synthesize_cluster(cluster, articles: list) -> None:
    """Enrich cluster with realistic label + summary from article content (mock fast-path).

    Instead of calling LLM, extracts key facts from pre-curated mock articles to
    produce rich ClusterResult fields that downstream TrendData conversion needs.
    """
    from collections import Counter

    ca = [articles[i] for i in cluster.article_indices if i < len(articles)]
    if not ca:
        from app.intelligence.summarizer import _fallback_label
        cluster.label = _fallback_label(cluster)
        return

    # ── Label: build readable label from article titles ───────────────
    titles = [getattr(a, "title", "") for a in ca]

    # Strategy: use the most representative title (shortest that mentions
    # the key entities), trimmed to 3-8 words.
    # First find the dominant proper noun across all titles.
    _STOP = {
        "the", "a", "an", "to", "of", "in", "for", "and", "on", "at", "is",
        "are", "its", "with", "by", "as", "has", "have", "from", "new", "how",
        "why", "what", "will", "may", "can", "over", "up", "after", "amid",
        "says", "report", "reports", "according", "india", "indian",
    }
    word_freq: Counter = Counter()
    for t in titles:
        for word in t.split():
            clean = word.strip(".,;:!?\"'()[]")
            if len(clean) > 2 and clean.lower() not in _STOP:
                word_freq[clean] += 1
    top_words = [w for w, _ in word_freq.most_common(15)]

    # Pick the best title: one that contains the top entity AND is concise
    top_entity = top_words[0] if top_words else ""
    best_title = titles[0]  # fallback
    for t in titles:
        if top_entity in t and len(t.split()) <= 10:
            best_title = t
            break

    # Truncate to 8 words max
    label_words = best_title.split()[:8]
    cluster.label = " ".join(label_words)

    # ── Summary: combine key facts from first 3 articles ────────────────
    summaries = []
    for a in ca[:3]:
        content = getattr(a, "summary", "") or getattr(a, "content", "") or ""
        # Take first sentence
        first_sent = content.split(".")[0].strip()
        if first_sent and len(first_sent) > 30:
            summaries.append(first_sent + ".")
    cluster.summary = " ".join(summaries)[:500] if summaries else cluster.label

    # ── Event type: majority vote from article event_types ──────────────
    event_types = [getattr(a, "event_type", "") for a in ca if getattr(a, "event_type", "")]
    if event_types:
        cluster.event_type = Counter(event_types).most_common(1)[0][0]

    # ── Entity names: proper nouns appearing in 50%+ of titles ──────────
    entity_threshold = max(2, len(ca) * 0.5)
    entities = [
        w for w, c in word_freq.most_common(20)
        if c >= entity_threshold and w[0].isupper() and len(w) > 2
    ][:5]
    if hasattr(cluster, "entity_names"):
        cluster.entity_names = entities
    if entities and not cluster.primary_entity:
        cluster.primary_entity = entities[0]

    # ── Source names (if field exists) ───────────────────────────────────
    sources = list({getattr(a, "source_name", "") for a in ca if getattr(a, "source_name", "")})[:5]
    if hasattr(cluster, "source_names"):
        cluster.source_names = sources

    # ── Industries: derive from article content keywords ─────────────────
    # Mock articles lack industry_label (classifier skipped), so derive from content.
    # Each industry needs >=3 keyword hits to qualify (avoids false matches from
    # incidental mentions like "banking" in an IT services article).
    _INDUSTRY_KEYWORDS = {
        "IT & Enterprise Software": ["TCS", "Infosys", "Wipro", "HCL", "GenAI", "IT services", "SaaS", "enterprise software", "IT firms"],
        "Fintech & Financial Services": ["RBI", "NBFC", "fintech", "lending", "KYC", "digital lender", "BFSI", "payment gateway"],
        "Retail & E-commerce": ["Zepto", "Blinkit", "quick commerce", "dark store", "e-commerce", "retail", "FMCG", "grocery"],
        "Semiconductors & Electronics": ["semiconductor", "fab ", "chip design", "PCB", "Tata Electronics", "Vedanta", "India Semiconductor"],
        "Telecom & 5G": ["5G", "IoT", "Jio", "Airtel", "telecom", "private network", "TRAI", "spectrum"],
        "Manufacturing & Industrial": ["manufacturing", "factory floor", "industrial", "Siemens", "CNC machine", "robotic arm"],
    }
    all_text = " ".join(getattr(a, "title", "") + " " + (getattr(a, "summary", "") or "")[:200] for a in ca)
    industry_scores: dict = {}
    for ind, kws in _INDUSTRY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw.lower() in all_text.lower())
        if score >= 2:  # Need 2+ hits to qualify
            industry_scores[ind] = score
    if industry_scores:
        cluster.industries = [k for k, _ in sorted(industry_scores.items(), key=lambda x: -x[1])][:2]


def _populate_cluster_industries(
    clusters: list,
    articles: list,
) -> None:
    """Derive human-readable industry names for each cluster from member articles.

    Each filtered article may have an industry_label set by the industry classifier
    (e.g. "healthcare_pharma", "fintech_bfsi"). This function aggregates those
    labels per cluster, converts to human-readable names, and stores them on
    cluster.industries so downstream TrendData.industries_affected is populated.
    """
    # Map internal IDs to human-readable display names
    _INDUSTRY_DISPLAY: dict = {
        "healthcare_pharma": "Healthcare & Pharmaceuticals",
        "fintech_bfsi": "Fintech & Financial Services",
        "it_technology": "IT & Enterprise Software",
        "manufacturing": "Manufacturing & Industrial",
        "logistics_supply_chain": "Logistics & Supply Chain",
        "retail_fmcg": "Retail & Consumer Goods",
    }

    if not clusters or not articles:
        return

    # Build article lookup by run_index
    article_map = {}
    for art in articles:
        idx = getattr(art, "run_index", -1)
        if idx >= 0:
            article_map[idx] = art

    populated = 0
    for cluster in clusters:
        # Collect industry labels from member articles
        labels: dict = {}  # label -> count
        for idx in getattr(cluster, "article_indices", []):
            art = article_map.get(idx)
            if art is None:
                continue
            label = getattr(art, "industry_label", None)
            if label:
                labels[label] = labels.get(label, 0) + 1

        if not labels:
            continue

        # Sort by frequency (most common industry first), convert to display names
        sorted_labels = sorted(labels.items(), key=lambda x: -x[1])
        industries = []
        for label, _count in sorted_labels:
            display = _INDUSTRY_DISPLAY.get(label, label.replace("_", " ").title())
            if display not in industries:
                industries.append(display)

        cluster.industries = industries
        populated += 1

    logger.info(f"[industries] Populated industries on {populated}/{len(clusters)} clusters")


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

        # NLI scores flow via IntelligenceResult.nli_scores_by_source → orchestrator.
        # The orchestrator's learning_update_node publishes to the bus instance
        # that actually gets persisted to disk (bus.save()).
        state._nli_scores_by_source = nli_scores_by_source

    except Exception as exc:
        logger.debug(f"[pipeline] Learning update failed (non-fatal): {exc}")


def _build_result(state: PipelineState) -> IntelligenceResult:
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
        filtered_articles=state.filtered_articles,
        total_articles_fetched=len(state.raw_articles),
        total_articles_post_filter=len(state.filtered_articles),
        total_clusters=len(labeled),
        noise_rate=len(state.noise_article_indices) / max(len(state.filtered_articles), 1),
        mean_coherence=sum(coherence_scores) / max(len(coherence_scores), 1),
        mean_fit_score=sum(fit_scores) / max(len(fit_scores), 1),
        nli_scores_by_source=getattr(state, "_nli_scores_by_source", {}),
        gap4_dropped_companies=state.gap4_dropped_companies,
        thought_log=state.thought_log,
        rounds_completed=state.round_number,
        agent_requests_processed=sum(1 for r in state.agent_requests if r.resolved),
    )
