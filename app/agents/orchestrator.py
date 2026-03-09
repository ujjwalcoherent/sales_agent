"""
LangGraph Orchestrator  - multi-agent supervisor for sales intelligence pipeline.

Flow: source_intel -> analysis -> impact -> quality_validation -> lead_gen -> END
Quality gate can retry analysis (max 2x) or skip lead gen if no viable trends.
"""

import asyncio
import logging
import json
import csv
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

from ..schemas import AgentState, PipelineResult
from ..schemas.sales import (
    TrendData, ImpactAnalysis, CompanyData, ContactData, OutreachEmail,
)
from ..config import get_settings
from ..database import get_database

logger = logging.getLogger(__name__)


# -- IntelligenceResult -> TrendData bridge ------------------------------------

def _intelligence_clusters_to_trends(intel: Any) -> List["TrendData"]:
    """Convert IntelligenceResult.clusters -> List[TrendData] for downstream nodes.

    Maps math-validated cluster fields to TrendData schema.
    Missing fields (actionable_insight, buying_intent, event_5w1h, causal_chain)
    are populated with defaults  - downstream impact_node fills them via LLM.

    Severity heuristic:
      coherence >= 0.70 -> HIGH (tight, focused cluster  - likely significant event)
      coherence 0.50-0.70 -> MEDIUM
      coherence < 0.50 -> LOW
    """
    from ..schemas.sales import TrendData
    from ..schemas import Severity

    if intel is None:
        return []

    clusters = getattr(intel, "clusters", []) or []
    trends = []
    for cluster in clusters:
        label = getattr(cluster, "label", "") or ""
        summary = getattr(cluster, "summary", "") or ""
        if not label and not summary:
            continue  # Skip unlabeled clusters

        coherence = getattr(cluster, "coherence_score", 0.5) or 0.5
        if coherence >= 0.70:
            severity = "high"
        elif coherence >= 0.50:
            severity = "medium"
        else:
            severity = "low"

        industry = getattr(cluster, "industry", None)
        industries_affected = []
        if industry:
            ind_str = getattr(industry, "industry", None) or str(industry)
            industries_affected = [ind_str]

        entity = getattr(cluster, "primary_entity", None) or ""
        keywords = list(getattr(cluster, "keywords", []) or [])[:10]
        article_indices = getattr(cluster, "article_indices", []) or []
        source_names = list(getattr(cluster, "source_names", []) or [])[:5]

        try:
            trends.append(TrendData(
                id=getattr(cluster, "cluster_id", str(len(trends))),
                trend_title=label or f"Cluster {len(trends) + 1}",
                summary=summary,
                severity=severity,
                industries_affected=industries_affected,
                source_links=[],
                keywords=keywords,
                trend_type="intelligence",
                actionable_insight="",
                event_5w1h={},
                causal_chain=[],
                buying_intent={},
                affected_companies=[entity] if entity else [],
                affected_regions=[],
                trend_score=coherence,
                actionability_score=0.0,
                oss_score=0.0,
                article_count=len(article_indices),
                article_snippets=[],
                evidence_snippets=(
                    cluster.evidence_chain.key_snippets
                    if getattr(cluster, "evidence_chain", None) else []
                ),
                evidence_companies=(
                    cluster.evidence_chain.companies_cited
                    if getattr(cluster, "evidence_chain", None) else []
                ),
            ))
        except Exception as e:
            logger.debug(f"[orchestrator] Skipping cluster {getattr(cluster, 'cluster_id', '?')}: {e}")

    logger.info(f"[orchestrator] Bridged {len(trends)}/{len(clusters)} intelligence clusters -> TrendData")
    return trends


# -- Graph State --------------------------------------------------------------

class GraphState(TypedDict):
    """LangGraph state shared across all agent nodes."""
    deps: Any                                       # AgentDeps instance
    run_id: str                                     # Unique run identifier (datetime)
    trends: List[TrendData]
    impacts: List[ImpactAnalysis]
    companies: List[CompanyData]
    contacts: List[ContactData]
    outreach_emails: List[OutreachEmail]
    errors: Annotated[List[str], operator.add]
    current_step: str
    retry_counts: Dict[str, int]
    agent_reasoning: Dict[str, str]


# -- Recording helper ----------------------------------------------------------

def _record(deps, step_name: str, data: dict, t0: float):
    """Record step snapshot if recorder is active (real runs only)."""
    recorder = getattr(deps, "recorder", None)
    if recorder:
        try:
            recorder.record_step(step_name, data, _time.time() - t0)
        except Exception as e:
            logger.debug(f"Recording step {step_name} failed: {e}")


# -- Agent Nodes --------------------------------------------------------------

async def source_intel_node(state: GraphState) -> dict:
    """Source Intel Agent -- article collection with bandit-prioritized sources."""
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 1: SOURCE INTEL AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []

    try:
        from .source_intel import run_source_intel
        articles, embeddings, result = await run_source_intel(deps)

        logger.info(
            f"Source Intel: {len(articles)} articles, "
            f"{len(embeddings)} embeddings"
        )

        if len(articles) == 0:
            errors.append("0 articles collected  - cannot proceed to analysis")
            logger.warning("Source Intel: 0 articles  - pipeline will skip to completion")
        elif len(articles) < 3:
            errors.append(f"Only {len(articles)} articles  - need at least 3")

    except Exception as e:
        logger.error(f"Source Intel node failed: {e}")
        errors.append(f"Source Intel: {e}")
        result = None

    # -- MetaReasoner: reason about source quality ------------------
    try:
        reasoner = deps.meta_reasoner
        reasoner._run_id = state.get("run_id", "")
        articles_list = getattr(deps, "_articles", [])
        source_stats = {}
        for art in articles_list:
            sid = getattr(art, "source_id", "unknown")
            if sid not in source_stats:
                source_stats[sid] = {"articles": 0, "noise_rate": 0.0, "avg_quality": 0.5}
            source_stats[sid]["articles"] += 1
        bandit_top = list(deps.source_bandit.get_quality_estimates().keys())[:5]
        settings = get_settings()
        await reasoner.reason_about_sources(
            article_count=len(articles_list),
            source_stats=source_stats,
            bandit_top_sources=bandit_top,
            hours_window=settings.rss_hours_ago,
        )
    except Exception as e:
        logger.debug(f"MetaReasoner source checkpoint skipped: {e}")

    article_count = len(getattr(deps, "_articles", []))
    _record(deps, "source_intel_complete", {
        "articles": [
            {
                "title": getattr(a, "title", ""),
                "summary": getattr(a, "summary", "")[:500],
                "source_id": getattr(a, "source_id", ""),
                "url": getattr(a, "url", ""),
                "published_at": str(getattr(a, "published_at", "")),
            }
            for a in getattr(deps, "_articles", [])
        ],
        "article_count": article_count,
    }, t0)

    step = "source_intel_empty" if article_count == 0 else "source_intel_complete"

    return {
        "errors": errors,
        "current_step": step,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "source_intel": getattr(result, 'reasoning', str(errors)),
        },
    }


def source_intel_route(state: GraphState) -> str:
    """Route after source_intel: skip to learning_update if 0 articles collected."""
    if state.get("current_step") == "source_intel_empty":
        return "learning_update"
    return "analysis"


async def analysis_node(state: GraphState) -> dict:
    """Analysis Agent -- Leiden clustering, coherence validation, signal computation."""
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 2: ANALYSIS AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []

    try:
        from .analysis import run_analysis
        intel, result = await run_analysis(deps)

        # Bridge: IntelligenceResult.clusters -> List[TrendData]
        trends = _intelligence_clusters_to_trends(intel) if intel else []
        deps._trend_data = trends

        logger.info(f"Analysis: {len(trends)} trends detected")

    except Exception as e:
        logger.error(f"Analysis node failed: {e}")
        errors.append(f"Analysis: {e}")
        trends = []
        result = None

    # -- MetaReasoner: reason about trend quality -------------------
    try:
        reasoner = deps.meta_reasoner
        pipeline = getattr(deps, "_pipeline", None)
        noise_rate = 0.0
        mean_coherence = 0.0
        cluster_count = 0
        if pipeline:
            noise_rate = getattr(pipeline, "_noise_rate", 0.0)
            cluster_count = getattr(pipeline, "_n_clusters", len(trends))
            coherences = getattr(pipeline, "_coherence_scores", {})
            if coherences:
                mean_coherence = sum(coherences.values()) / max(len(coherences), 1)
        mean_oss = sum(getattr(t, "oss_score", 0) for t in trends) / max(len(trends), 1)
        await reasoner.reason_about_trends(
            trends=trends,
            cluster_count=cluster_count,
            noise_rate=noise_rate,
            mean_oss=mean_oss,
            mean_coherence=mean_coherence,
        )
    except Exception as e:
        logger.debug(f"MetaReasoner trend checkpoint skipped: {e}")

    _record(deps, "analysis_complete", {
        "trends": [t.model_dump() if hasattr(t, "model_dump") else {} for t in trends],
        "trend_count": len(trends),
    }, t0)

    return {
        "trends": trends,
        "errors": errors,
        "current_step": "analysis_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "analysis": getattr(result, 'reasoning', str(errors)),
        },
    }


async def impact_node(state: GraphState) -> dict:
    """Market Impact Agent -- AI council analysis + cross-trend synthesis."""
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 3: MARKET IMPACT AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []

    try:
        from .market_impact import run_market_impact
        impacts, result = await run_market_impact(deps)

        logger.info(f"Impact: {len(impacts)} impacts analyzed")

    except Exception as e:
        logger.error(f"Impact node failed: {e}")
        errors.append(f"Impact: {e}")
        impacts = []
        result = None

    _record(deps, "impact_complete", {
        "impacts": [i.model_dump() if hasattr(i, "model_dump") else {} for i in impacts],
        "impact_count": len(impacts),
    }, t0)

    return {
        "impacts": impacts,
        "errors": errors,
        "current_step": "impact_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "market_impact": getattr(result, 'reasoning', str(errors)),
        },
    }


async def quality_validation_node(state: GraphState) -> dict:
    """Quality Agent -- validate trends + impacts, retry or gate low quality."""
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 3.5: QUALITY VALIDATION AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []

    try:
        from .quality import run_quality_check

        trend_verdict = await run_quality_check(deps, "trends")
        logger.info(
            f"Quality (trends): passed={trend_verdict.passed}, "
            f"retry={trend_verdict.should_retry}, "
            f"score={trend_verdict.quality_score:.3f}"
        )

        retry_counts = state.get("retry_counts", {})
        if trend_verdict.should_retry and retry_counts.get("analysis", 0) < 2:
            logger.info("Quality gate: trend quality borderline, retrying analysis")
            retry_counts["analysis"] = retry_counts.get("analysis", 0) + 1
            return {
                "errors": errors,
                "current_step": "quality_retry_analysis",
                "retry_counts": retry_counts,
                "agent_reasoning": {
                    **state.get("agent_reasoning", {}),
                    "quality_trends": trend_verdict.reasoning,
                },
            }

        impact_verdict = await run_quality_check(deps, "impacts")
        logger.info(
            f"Quality (impacts): passed={impact_verdict.passed}, "
            f"items_passed={impact_verdict.items_passed}"
        )

    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        errors.append(f"Quality validation: {e}")
        trend_verdict = None
        impact_verdict = None

    viable_count = getattr(impact_verdict, 'items_passed', 0) if impact_verdict else 0
    _record(deps, "quality_complete", {
        "trend_quality": getattr(trend_verdict, 'quality_score', 0.0) if trend_verdict else 0.0,
        "impact_items_passed": viable_count,
        "viable_count": viable_count,
    }, t0)

    # Filter impacts by confidence  - do it here, not in routing function
    settings = get_settings()
    mock_mode = getattr(deps, "mock_mode", False)
    threshold = 0.0 if mock_mode else settings.min_trend_confidence_for_agents
    impacts = state.get("impacts", [])
    viable = [imp for imp in impacts if imp.council_confidence >= threshold]
    dropped = len(impacts) - len(viable)

    if dropped:
        logger.info(
            f"Quality gate: {dropped}/{len(impacts)} trends below confidence {threshold}"
        )

    if not viable:
        viable = sorted(impacts, key=lambda i: i.council_confidence, reverse=True)[:3]
        if viable:
            logger.warning(
                f"Quality gate: No impacts above {threshold}, "
                f"using top {len(viable)} as fail-open fallback "
                f"(best confidence: {viable[0].council_confidence:.2f})"
            )
        else:
            logger.warning(
                "Quality gate: 0 impacts available  - lead gen will attempt "
                "with empty impacts (source intel may have returned 0 articles)"
            )

    if deps:
        deps._viable_impacts = viable

    return {
        "errors": errors,
        "current_step": "quality_complete",
        "impacts": viable,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "quality_trends": getattr(trend_verdict, 'reasoning', ''),
            "quality_impacts": getattr(impact_verdict, 'reasoning', ''),
        },
    }


def quality_route(state: GraphState) -> str:
    """Route: "analysis" (retry) or "lead_gen" (proceed).

    Impact filtering is handled in quality_validation_node  - this function
    is kept pure (no state mutation).
    """
    if state.get("current_step") == "quality_retry_analysis":
        return "analysis"
    return "lead_gen"


async def lead_gen_node(state: GraphState) -> dict:
    """Lead Gen Agent -- company discovery, contact finding, outreach emails."""
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 4: LEAD GEN AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []
    from app.config import get_settings
    timeout = get_settings().lead_gen_timeout

    try:
        from .leads import run_lead_gen
        companies, contacts, outreach, result = await asyncio.wait_for(
            run_lead_gen(deps), timeout=timeout
        )

        logger.info(
            f"Lead Gen: {len(companies)} companies, "
            f"{len(contacts)} contacts, {len(outreach)} outreach"
        )

    except asyncio.TimeoutError:
        logger.warning(f"Lead Gen timed out after {timeout:.0f}s  - using crystallizer results only")
        errors.append(f"Lead Gen: timed out after {timeout:.0f}s")
        companies, contacts, outreach = [], [], []
        result = None

    except Exception as e:
        logger.error(f"Lead Gen node failed: {e}")
        errors.append(f"Lead Gen: {e}")
        companies, contacts, outreach = [], [], []
        result = None

    # -- MetaReasoner: reason about lead quality --------------------
    try:
        reasoner = deps.meta_reasoner
        lead_sheets = getattr(deps, "_lead_sheets", [])
        lead_summaries = []
        for ls in lead_sheets[:10]:
            lead_summaries.append({
                "company": getattr(ls, "company_name", "?"),
                "contacts": getattr(ls, "contact_count", 0),
                "confidence": getattr(ls, "confidence", 0.0),
                "event_type": getattr(ls, "event_type", "?"),
            })
        await reasoner.reason_about_leads(
            lead_count=len(lead_sheets),
            company_count=len(companies or []),
            contact_count=len(contacts or []),
            email_count=len(outreach or []),
            lead_summaries=lead_summaries,
        )
    except Exception as e:
        logger.debug(f"MetaReasoner lead checkpoint skipped: {e}")

    people = getattr(deps, "_person_profiles", [])
    _record(deps, "lead_gen_complete", {
        "companies": [c.model_dump() if hasattr(c, "model_dump") else {} for c in (companies or [])],
        "contacts": [c.model_dump() if hasattr(c, "model_dump") else {} for c in (contacts or [])],
        "outreach": [o.model_dump() if hasattr(o, "model_dump") else {} for o in (outreach or [])],
        "people": [p.model_dump() if hasattr(p, "model_dump") else {} for p in people],
        "company_count": len(companies or []),
        "contact_count": len(contacts or []),
        "people_count": len(people),
    }, t0)

    return {
        "companies": companies or [],
        "contacts": contacts or [],
        "outreach_emails": outreach or [],
        "errors": errors,
        "current_step": "lead_gen_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "lead_gen": getattr(result, 'reasoning', str(errors)),
        },
    }


# -- Causal Council + Lead Crystallizer Nodes ---------------------------------

async def causal_council_node(state: GraphState) -> dict:
    """Causal Council  - multi-hop business impact chain tracer.

    Traces: event -> directly affected companies (hop1) -> their buyers/suppliers (hop2)
            -> downstream of hop2 (hop3). Uses real pydantic-ai tool calling.
    Self-learning: stores LearningSignal for source bandit + weight learner.
    """
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 3.7: CAUSAL COUNCIL (multi-hop impact tracing)")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []
    causal_results = []

    impacts = state.get("impacts", [])
    if not impacts:
        logger.warning("Causal council: no impacts  - skipping")
        return {
            "errors": [],
            "current_step": "causal_council_complete",
            "agent_reasoning": {**state.get("agent_reasoning", {}), "causal_council": "no impacts"},
        }

    try:
        from app.agents.causal_council import run_causal_council
        from app.agents.deps import LearningSignal, HopSignal
        from app.tools.llm.providers import ProviderManager

        pm = ProviderManager()

        # Fix 3C: Clear expired rate-limit cooldowns before the most critical phase.
        # Earlier phases may have triggered 429 cooldowns that have since expired.
        # _is_cooling_down lazily clears these on next check, but we force it now
        # so the provider list is fresh for the causal council's tool-capable model.
        _now = _time.time()
        for pname in list(ProviderManager._failed_providers.keys()):
            # Calling _is_cooling_down triggers lazy cleanup of expired entries
            pm._is_cooling_down(pname, _now)

        # Build lookup: trend_id -> TrendData (for event_type, keywords, actionability_score)
        trends = state.get("trends", [])
        trend_by_id = {td.id: td for td in trends}

        # Build BM25 index from articles fetched this run
        articles = getattr(deps, "_articles", [])
        if articles and deps.search_manager:
            from app.tools.search import BM25Search
            bm25 = BM25Search(articles)
            deps.search_manager.set_bm25_index(bm25)
            logger.info(f"BM25: indexed {len(articles)} articles for causal council")

        settings = get_settings()
        _max_impacts = getattr(settings, 'per_trend_max_impacts', 5)
        geo_label = settings.country

        for impact in impacts[:_max_impacts]:
            trend_title = impact.trend_title or ""
            trend_summary = (
                impact.detailed_reasoning
                or " ".join(impact.direct_impact[:3])
                or impact.pitch_angle
                or ""
            )

            td = trend_by_id.get(impact.trend_id)
            event_type = (td.trend_type if td else "") or "general"
            keywords = (td.keywords if td else []) or list(impact.midsize_pain_points[:5])
            oss_score = (td.oss_score if td else 0.0) or 0.0
            chain = await run_causal_council(
                trend_title=trend_title,
                trend_summary=trend_summary,
                event_type=event_type,
                keywords=keywords,
                geo=geo_label,
                provider_manager=pm,
                search_manager=deps.search_manager,
            )
            causal_results.append(chain)

            # Record per-trend quality metrics for the autonomous learning loop.
            # kb_hit_rate = fraction of hops where the KB returned real company names  -
            # a direct measure of how targetable this trend segment is.
            hops_with_companies = sum(1 for h in chain.hops if h.companies_found)
            hop_sigs = [
                HopSignal(
                    hop=h.hop,
                    segment=h.segment,
                    lead_type=h.lead_type,
                    confidence=h.confidence,
                    companies_found=len(h.companies_found),
                    tool_calls=0,  # pydantic-ai doesn't expose call count; TODO: instrument
                    mechanism_specificity=min(1.0, len(h.mechanism.split()) / 15.0),
                )
                for h in chain.hops
            ]
            sig = LearningSignal(
                trend_title=trend_title,
                event_type=event_type,
                oss_score=oss_score,
                hops_generated=len(chain.hops),
                hop_signals=hop_sigs,
                kb_hit_rate=hops_with_companies / len(chain.hops) if chain.hops else 0.0,
                source_article_ids=[a.get("id", "") for a in articles[:20] if isinstance(a, dict)],
                run_id=state.get("run_id", ""),
            )
            deps._signals.append(sig)

            logger.info(
                f"  '{trend_title[:50]}' -> {len(chain.hops)} hops "
                f"(kb_hit_rate={sig.kb_hit_rate:.0%}, oss={oss_score:.2f})"
            )

        deps._causal_results = causal_results
        total_hops = sum(len(r.hops) for r in causal_results)

        # Fix 4B: If ALL causal results have 0 hops, build synthetic hops from
        # impact analysis data. This ensures lead_crystallize_node always has
        # something to work with.
        if total_hops == 0 and impacts:
            logger.warning("Causal council: 0 hops from all trends  - building synthetic hops from impact data")
            from app.agents.causal_council import CausalChainResult, CausalHop, _build_synthetic_hops

            trends = state.get("trends", [])
            trend_by_id = {td.id: td for td in trends}

            for impact in impacts[:5]:
                td = trend_by_id.get(impact.trend_id)
                keywords = (td.keywords if td else []) or []
                event_type = (td.trend_type if td else "") or "general"

                synthetic = _build_synthetic_hops(
                    trend_title=impact.trend_title or "",
                    trend_summary=impact.detailed_reasoning or " ".join(impact.direct_impact[:3]),
                    event_type=event_type,
                    keywords=keywords,
                )
                if synthetic.hops:
                    causal_results.append(synthetic)

            deps._causal_results = causal_results
            total_hops = sum(len(r.hops) for r in causal_results)
            logger.info(f"Causal council: after synthetic fallback -> {total_hops} total hops")

        logger.info(f"Causal council: {len(causal_results)} chains, {total_hops} total hops")

    except Exception as e:
        logger.error(f"Causal council node failed: {e}", exc_info=True)
        errors.append(f"CausalCouncil: {e}")

    _record(deps, "causal_council_complete", {
        "causal_results": [
            {
                "event_summary": r.event_summary,
                "event_type": r.event_type,
                "reasoning": r.reasoning[:500] if r.reasoning else "",
                "hops": [
                    {
                        "hop": h.hop,
                        "segment": h.segment,
                        "lead_type": h.lead_type,
                        "mechanism": h.mechanism,
                        "companies_found": list(h.companies_found)[:10],
                        "confidence": h.confidence,
                        "urgency_weeks": h.urgency_weeks,
                    }
                    for h in r.hops
                ],
            }
            for r in causal_results
        ],
        "total_hops": sum(len(r.hops) for r in causal_results),
    }, t0)

    return {
        "errors": errors,
        "current_step": "causal_council_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "causal_council": f"{len(causal_results)} chains traced, "
                              f"{sum(len(r.hops) for r in causal_results)} total hops",
        },
    }


async def _resolve_companies_for_hops(deps, causal_results):
    """Resolve segment descriptions to real company names via web search + LLM.

    For each causal hop that has an empty companies_found list:
    1. Search the web for real companies matching the segment
    2. Feed search results to LLM for structured extraction
    This grounds output in real web data instead of LLM hallucination.
    """
    llm = deps.llm_service
    search_mgr = deps.search_manager
    segments_to_resolve = []
    hop_refs = []  # (chain_idx, hop_idx) for each segment

    for ci, chain in enumerate(causal_results):
        for hi, hop in enumerate(chain.hops):
            if not hop.companies_found and hop.segment:
                segments_to_resolve.append(hop.segment)
                hop_refs.append((ci, hi))

    if not segments_to_resolve:
        return  # All hops already have companies

    # Step 1: Search for each segment concurrently
    settings = get_settings()
    country = settings.country
    current_year = datetime.now().year
    search_results = {}  # idx -> list of search results
    sem = asyncio.Semaphore(5)

    async def _search_segment(idx: int, segment: str):
        async with sem:
            try:
                query = f"{segment} company {country} {current_year}"
                data = await search_mgr.web_search(query, max_results=5)
                search_results[idx] = data.get("results", [])
            except Exception as e:
                logger.debug(f"Search for segment '{segment[:40]}' failed: {e}")
                search_results[idx] = []

    await asyncio.gather(*[
        _search_segment(i, seg) for i, seg in enumerate(segments_to_resolve)
    ])

    # Step 2: Build context from search results and make ONE batched LLM call
    segment_blocks = []
    for i, seg in enumerate(segments_to_resolve):
        results = search_results.get(i, [])
        if results:
            snippets = "\n".join(
                f"  - {r.get('title', '')}: {r.get('content', '')[:200]}"
                for r in results[:5]
            )
            segment_blocks.append(f"{i+1}. Segment: {seg}\n   Search results:\n{snippets}")
        else:
            segment_blocks.append(f"{i+1}. Segment: {seg}\n   Search results: (none found)")

    prompt = f"""For each business segment below, extract 2-3 REAL, specific {country} companies from the search results.
Only name companies that appear in the search results or that you are highly confident actually exist.
Include the company's city/state if mentioned in the results.

{chr(10).join(segment_blocks)}

Respond as a JSON array where each item has:
  "index": the segment number (1-based),
  "companies": [
    {{"name": "Company Name", "city": "City", "state": "State", "size_band": "sme|mid|large"}}
  ]

CRITICAL: Only include companies you can verify from the search results. If no real companies
are found in the results for a segment, return an empty companies array for that index.
Respond with the JSON array only."""

    try:
        result = await llm.generate_json(
            prompt=prompt,
            system_prompt="You are a business intelligence analyst. Extract real company names "
                          "from search results. Never invent companies not found in the data.",
        )

        # Parse result  - could be a list or {"results": [...]}
        items = result if isinstance(result, list) else result.get("results", result.get("segments", []))
        if not isinstance(items, list):
            items = []

        resolved_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index", 0) - 1  # Convert to 0-based
            if idx < 0 or idx >= len(hop_refs):
                continue
            ci, hi = hop_refs[idx]
            companies = item.get("companies", [])
            if not isinstance(companies, list):
                continue
            names = []
            for c in companies[:3]:
                if isinstance(c, dict):
                    name = c.get("name", "").strip()
                    if name and len(name) > 2:
                        names.append(name)
                        city = c.get("city", "")
                        state = c.get("state", "")
                        if city and not causal_results[ci].hops[hi].geo_hint:
                            causal_results[ci].hops[hi].geo_hint = f"{city}, {state}" if state else city
                elif isinstance(c, str) and len(c.strip()) > 2:
                    names.append(c.strip())
            if names:
                causal_results[ci].hops[hi].companies_found = names
                resolved_count += len(names)

        logger.info(
            f"Company resolution: resolved {resolved_count} companies "
            f"for {len(segments_to_resolve)} segments "
            f"({sum(len(r) for r in search_results.values())} search results used)"
        )

    except Exception as e:
        logger.warning(f"Company resolution failed (non-fatal): {e}")


async def lead_crystallize_node(state: GraphState) -> dict:
    """Lead Crystallizer  - converts causal chains into concrete call sheets.

    Each LeadSheet has: company name (real from KB), contact role, trigger event,
    pain point, service pitch, and opening line ready to use on a call.
    """
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 3.8: LEAD CRYSTALLIZER (call sheet generation)")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []
    all_leads = []

    causal_results = getattr(deps, "_causal_results", [])
    impacts = state.get("impacts", [])

    if not causal_results:
        logger.warning("Lead crystallizer: no causal results  - skipping")
        return {
            "errors": [],
            "current_step": "lead_crystallize_complete",
            "agent_reasoning": {**state.get("agent_reasoning", {}), "lead_crystallizer": "no causal results"},
        }

    # -- Resolve segment descriptions to real company names --
    try:
        await _resolve_companies_for_hops(deps, causal_results)
    except Exception as e:
        logger.warning(f"Company resolution step failed (non-fatal): {e}")

    # -- Fetch company-specific news for resolved companies --
    company_news: dict[str, list] = {}
    try:
        unique_companies: set[str] = set()
        for chain in causal_results:
            for hop in chain.hops:
                if hop.companies_found:
                    for name in hop.companies_found:
                        if not name.startswith("["):
                            unique_companies.add(name)

        if unique_companies and deps.search_manager:
            sem = asyncio.Semaphore(5)

            async def _fetch_news(name: str):
                async with sem:
                    news = await deps.search_manager.company_news_search(
                        name, months=5, max_results=3,
                    )
                    if news:
                        company_news[name] = news

            await asyncio.gather(*[_fetch_news(n) for n in unique_companies])
            logger.info(f"Company news: fetched news for {len(company_news)}/{len(unique_companies)} companies")
    except Exception as e:
        logger.debug(f"Company news fetch failed (non-fatal): {e}")

    try:
        from app.agents.leads import crystallize_leads

        # Index trends and impacts by their natural keys so we can join them
        # without relying on fragile positional alignment between two lists.
        # chain.event_summary == impact.trend_title (set by the impact agent).
        trends = state.get("trends", [])
        trend_by_id = {td.id: td for td in trends}
        impact_by_title = {imp.trend_title: imp for imp in impacts if imp.trend_title}

        for chain in causal_results:
            impact = impact_by_title.get(chain.event_summary)
            # Prefer the full council reasoning for context; fall back to the
            # causal chain's own reasoning when impact data isn't available.
            trend_summary = (
                impact.detailed_reasoning
                or " ".join(impact.direct_impact[:3])
                or ""
            ) if impact else chain.reasoning
            td = trend_by_id.get(impact.trend_id) if impact else None
            oss_score = (td.oss_score if td else 0.0) or 0.0

            leads = await crystallize_leads(
                causal_result=chain,
                trend_title=chain.event_summary,
                trend_summary=trend_summary,
                event_type=chain.event_type,
                oss_score=oss_score,
            )
            all_leads.extend(leads)

        # Attach company news to lead sheets
        if company_news:
            for lead in all_leads:
                news = company_news.get(lead.company_name, [])
                if news:
                    lead.company_news = news

        # Loop 4: Rank leads using company relevance bandit (Thompson Sampling)
        try:
            company_bandit = deps.company_bandit
            for lead in all_leads:
                arm_id = f"{lead.company_size_band}_{lead.event_type}"
                lead.confidence *= (0.7 + 0.3 * company_bandit.compute_relevance(
                    company_size=lead.company_size_band or "mid",
                    event_type=lead.event_type or "general",
                    intent_signal_strength=lead.oss_score,
                    explore=True,
                ))
                lead.confidence = min(1.0, lead.confidence)
            all_leads.sort(key=lambda l: (-l.confidence, l.urgency_weeks))
            logger.info(f"Company bandit: re-ranked {len(all_leads)} leads")
        except Exception as e:
            logger.debug(f"Company bandit ranking skipped: {e}")

        deps._lead_sheets = all_leads

        logger.info(f"Lead crystallizer: {len(all_leads)} call sheets generated")
        for sheet in all_leads[:5]:   # Preview top 5
            logger.info(
                f"  [Hop{sheet.hop}][{sheet.lead_type.upper()}] "
                f"{sheet.company_name} | {sheet.contact_role}"
            )
            logger.info(f"    -> {sheet.opening_line[:100]}")

    except Exception as e:
        logger.error(f"Lead crystallizer node failed: {e}", exc_info=True)
        errors.append(f"LeadCrystallizer: {e}")

    _record(deps, "lead_crystallize_complete", {
        "lead_sheets": [ls.model_dump() if hasattr(ls, "model_dump") else {} for ls in all_leads],
        "lead_count": len(all_leads),
    }, t0)

    return {
        "errors": errors,
        "current_step": "lead_crystallize_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "lead_crystallizer": f"{len(all_leads)} call sheets  - "
                                 f"{sum(1 for l in all_leads if l.company_cin)} with CIN from KB",
        },
    }


# -- Learning Update Node -----------------------------------------------------

async def learning_update_node(state: GraphState) -> dict:
    """Post-run autonomous learning with cross-loop signal exchange.

    Three-phase protocol prevents circular reads between loops:
      Phase 1  - Each loop updates using THIS run's data, publishes to signal bus
      Phase 2  - Bus computes cross-loop derived signals (system_confidence, exploration_budget)
      Phase 3  - Each loop applies small cross-loop adjustments

    The signal bus is the 'brain'  - it connects all 6 loops so they inform each other
    instead of operating in isolation.
    """
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 5: AUTONOMOUS LEARNING UPDATE (with cross-loop signal bus)")
    logger.info("=" * 50)

    deps = state["deps"]
    signals = getattr(deps, "_signals", [])
    articles = getattr(deps, "_articles", [])
    pipeline = getattr(deps, "_pipeline", None)

    # -- Initialize Signal Bus --------------------------------------------
    from app.learning.signal_bus import LearningSignalBus
    bus = LearningSignalBus()
    previous_bus = LearningSignalBus.load_previous()
    bus.run_id = state.get("run_id", "")
    bus.run_count = (previous_bus.run_count + 1) if previous_bus else 1

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 0: Snapshot learning state (AutoResearch pattern)       ║
    # ╚══════════════════════════════════════════════════════════════════╝
    from app.learning.experiment_tracker import (
        ExperimentTracker, ExperimentRecord, Hypothesis,
        snapshot_learning_state, restore_learning_state, cleanup_snapshot,
        pick_next_hypothesis, mark_hypothesis_tested,
    )
    tracker = ExperimentTracker()
    snapshot_learning_state()

    # Check if any learning loop should be dampened based on history
    loop_dampen = {}
    for loop_name in ["weight_learner", "threshold_adapter", "source_bandit", "company_bandit"]:
        loop_dampen[loop_name] = tracker.should_dampen(loop_name)
        if loop_dampen[loop_name]:
            logger.info(f"Loop health: {loop_name} dampened (consistently hurting quality)")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 1: Each loop updates + publishes to bus                 ║
    # ╚══════════════════════════════════════════════════════════════════╝

    learning_updates: dict = {}

    # -- 1. Source Bandit Update ------------------------------------------
    try:
        if articles and signals:
            bandit = deps.source_bandit

            # Capture pre-update posteriors (for bus degradation detection)
            pre_update_means = bandit.get_quality_estimates()

            # Group article IDs by source_id
            source_articles: dict = {}
            for art in articles:
                src_id = getattr(art, "source_id", None) or getattr(art, "source", "unknown")
                if src_id not in source_articles:
                    source_articles[src_id] = []
                art_id = str(getattr(art, "id", id(art)))
                source_articles[src_id].append(art_id)

            # Build article -> cluster label map
            article_labels: dict = {}
            cluster_quality: dict = {}
            cluster_oss: dict = {}

            if pipeline and hasattr(pipeline, "_article_cluster_map"):
                article_labels = pipeline._article_cluster_map or {}
                cluster_quality = pipeline._cluster_quality or {}

            # Build cluster_oss from LearningSignals using actual cluster IDs
            if pipeline and hasattr(pipeline, "_article_cluster_map"):
                _title_to_cluster: dict = {}
                for _art_id, _clust_id in article_labels.items():
                    if hasattr(pipeline, "_cluster_summaries"):
                        _summ = (pipeline._cluster_summaries or {}).get(_clust_id, {})
                        _t = _summ.get("trend_title", "")
                        if _t:
                            _title_to_cluster[_t] = _clust_id

                for sig in signals:
                    _cid = _title_to_cluster.get(sig.trend_title)
                    if _cid is not None:
                        cluster_oss[_cid] = sig.oss_score
                    else:
                        cluster_oss[hash(sig.trend_title) % 10000] = sig.oss_score
            else:
                for idx, sig in enumerate(signals):
                    cluster_oss[idx] = sig.oss_score

            if source_articles:
                updated = bandit.update_from_run(
                    source_articles=source_articles,
                    article_labels=article_labels,
                    cluster_quality=cluster_quality,
                    cluster_oss=cluster_oss,
                )
                top_sources = sorted(updated.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{k}={v:.3f}" for k, v in top_sources)
                logger.info(f"Source bandit updated: {len(updated)} sources | top: {top_str}")

                # PUBLISH to bus
                bus.publish_source_bandit(updated, previous_means=pre_update_means)
    except Exception as e:
        logger.warning(f"Source bandit update failed: {e}")

    # -- 1b. Record cluster signals for weight auto-learning -----------
    try:
        if cluster_quality and cluster_oss:
            from app.learning.pipeline_metrics import record_cluster_signals
            n_logged = record_cluster_signals(
                run_id=state.get("run_id", bus.run_id if hasattr(bus, 'run_id') else ""),
                cluster_signals=cluster_quality,
                cluster_oss=cluster_oss,
            )
            if n_logged:
                logger.info(f"Logged {n_logged} cluster signal records for weight learning")
    except Exception as e:
        logger.debug(f"Cluster signal logging skipped: {e}")

    # -- 2. Trend Memory: publish lifecycle distribution to bus ----------
    stale_pruned = 0
    try:
        # The TrendPipeline stores memory results in pipeline.metrics["memory"]
        # with lifecycle_counts and avg_novelty computed during the temporalize layer.
        lifecycle_counts = {"birth": 0, "growth": 0, "peak": 0, "decline": 0}
        avg_novelty = 0.5

        memory_metrics = None
        if pipeline and hasattr(pipeline, "metrics"):
            memory_metrics = pipeline.metrics.get("memory")

        used_memory_metrics = False
        if memory_metrics:
            stored_counts = memory_metrics.get("lifecycle_counts", {})
            if stored_counts and sum(stored_counts.values()) > 0:
                lifecycle_counts = {
                    "birth": stored_counts.get("birth", 0),
                    "growth": stored_counts.get("growth", 0),
                    "peak": stored_counts.get("peak", 0),
                    "decline": stored_counts.get("decline", 0),
                }
                used_memory_metrics = True
            avg_novelty = memory_metrics.get("avg_novelty", 0.5)

        if not used_memory_metrics and signals:
            # Fallback: approximate lifecycle from OSS scores
            for sig in signals:
                oss = sig.oss_score
                if oss >= 0.6:
                    lifecycle_counts["birth"] += 1
                elif oss >= 0.4:
                    lifecycle_counts["growth"] += 1
                elif oss >= 0.2:
                    lifecycle_counts["peak"] += 1
                else:
                    lifecycle_counts["decline"] += 1
            avg_novelty = sum(s.oss_score for s in signals) / max(len(signals), 1)

        if sum(lifecycle_counts.values()) > 0:
            bus.publish_trend_memory(lifecycle_counts, avg_novelty, stale_pruned)
            logger.info(
                f"Trend memory published: {lifecycle_counts}, "
                f"avg_novelty={avg_novelty:.3f}"
            )
    except Exception as e:
        logger.warning(f"Trend memory publish failed: {e}")

    # -- 3. Weight Learner: Compute + persist all 4 weight types ----------
    all_learned = {}
    try:
        if signals:
            from app.learning.weight_learner import (
                compute_learned_weights, _save_persisted_weights,
                maybe_update_stable_weights,
            )
            import json as _json

            _settings = get_settings()
            weight_types = {
                "actionability": _json.loads(_settings.actionability_weights),
                "trend_score": _json.loads(_settings.trend_score_weights),
                "cluster_quality": _json.loads(_settings.cluster_quality_score_weights),
                "confidence": {
                    "temporal_novelty": 0.30, "cluster_quality": 0.25,
                    "source_corroboration": 0.25, "evidence_specificity": 0.20,
                },
            }

            for wt_name, defaults in weight_types.items():
                all_learned[wt_name] = compute_learned_weights(wt_name, defaults)

            _save_persisted_weights(all_learned)

            # Check if stable weights need updating (every N runs)
            stable_updated = maybe_update_stable_weights(all_learned)

            # Invalidate composite.py weight cache so next scoring uses fresh weights
            # Invalidate weight cache if it exists (legacy  - noop if not present)
            try:
                from app.trends.signals.composite import invalidate_weights_cache
                invalidate_weights_cache()
            except ImportError:
                pass  # trends/ module removed  - cache invalidation handled by intelligence pipeline

            mean_oss = sum(s.oss_score for s in signals) / len(signals)
            logger.info(
                f"Weight learner: updated all 4 weight types | "
                f"{len(signals)} signals | mean OSS={mean_oss:.3f}"
                f"{' | stable weights promoted' if stable_updated else ''}"
            )

            # PUBLISH to bus  - determine learning path and data count
            learning_path = "default"
            data_count = 0
            try:
                from app.tools.feedback_store import load_feedback
                fb = load_feedback("trend") or []
                human_fb = [f for f in fb if not f.get("metadata", {}).get("auto", False)]
                if len(human_fb) >= 50:
                    learning_path = "human"
                    data_count = len(human_fb)
                else:
                    from app.learning.pipeline_metrics import load_cluster_signal_history
                    cluster_data = load_cluster_signal_history(min_runs=3)
                    if cluster_data:
                        learning_path = "outcome"
                        data_count = len(cluster_data)
            except Exception:
                pass

            bus.publish_weight_learner(all_learned, weight_types, learning_path, data_count)
    except Exception as e:
        logger.warning(f"Weight learner update failed: {e}")

    # -- 4. Persist learning signals to JSONL -----------------------------
    try:
        import json
        from pathlib import Path
        from datetime import datetime, timezone

        signals_file = Path("data/learning_signals.jsonl")
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(signals_file, "a", encoding="utf-8") as f:
            for sig in signals:
                record = sig.to_dict()
                record["run_id"] = run_id
                record["timestamp"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(record) + "\n")
        if signals:
            logger.info(f"Learning signals: {len(signals)} records appended to {signals_file}")
    except Exception as e:
        logger.warning(f"Learning signal persistence failed: {e}")

    # -- 5. Company Bandit: reward update from lead sheets (Loop 4) -------
    try:
        lead_sheets = getattr(deps, "_lead_sheets", [])
        if lead_sheets:
            company_bandit = deps.company_bandit
            company_bandit.decay()
            updated_arms = set()
            for lead in lead_sheets:
                arm_id = f"{lead.company_size_band or 'mid'}_{lead.event_type or 'general'}"
                if lead.company_cin:
                    reward = min(1.0, lead.confidence + 0.2)
                elif not lead.company_name.startswith("["):
                    reward = lead.confidence * 0.7
                else:
                    reward = 0.1
                company_bandit.update(arm_id, reward)
                updated_arms.add(arm_id)
            logger.info(f"Company bandit: updated {len(updated_arms)} arms from {len(lead_sheets)} leads")

            # PUBLISH to bus (all arms, not just updated this run)
            try:
                bus.publish_company_bandit(company_bandit.get_estimates())
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Company bandit update skipped: {e}")

    # -- 6. Quality Feedback: auto-record for weight learner (Loop 5) --
    feedback_dist = {"good_trend": 0, "already_knew": 0, "bad_trend": 0}
    quality_sum = 0.0
    try:
        if signals:
            from app.tools.feedback_store import save_feedback
            feedback_count = 0
            for sig in signals:
                quality = (
                    0.40 * sig.kb_hit_rate +
                    0.30 * min(1.0, sig.leads_with_companies / max(sig.leads_generated, 1)) +
                    0.30 * sig.oss_score
                )
                quality_sum += quality
                if quality >= 0.45:
                    rating = "good_trend"
                elif quality >= 0.25:
                    rating = "already_knew"
                else:
                    rating = "bad_trend"
                feedback_dist[rating] = feedback_dist.get(rating, 0) + 1
                save_feedback(
                    feedback_type="trend",
                    item_id=sig.trend_title[:100],
                    rating=rating,
                    metadata={
                        "auto": True,
                        "source": "auto_learning",
                        "composite_quality": round(quality, 3),
                        "oss_score": sig.oss_score,
                        "kb_hit_rate": sig.kb_hit_rate,
                        "leads_with_companies": sig.leads_with_companies,
                        "leads_generated": sig.leads_generated,
                        "hops_generated": sig.hops_generated,
                    },
                )
                feedback_count += 1
            logger.info(f"Quality feedback: auto-recorded {feedback_count} trend ratings")

            # PUBLISH to bus
            mean_q = quality_sum / max(len(signals), 1)
            bus.publish_auto_feedback(feedback_dist, mean_q)
    except Exception as e:
        logger.debug(f"Quality feedback recording skipped: {e}")

    # -- 6b. Adaptive Thresholds: publish current state to bus ---------
    try:
        from app.learning.pipeline_metrics import (
            compute_adaptive_thresholds, detect_drift, detect_drift_ewma,
        )
        thresholds = compute_adaptive_thresholds()
        drift_alerts = detect_drift({}) if not signals else []
        ewma_drift = detect_drift_ewma({}) if not signals else []
        all_drift = drift_alerts + ewma_drift
        anomaly_flags = []

        # Check for threshold anomalies
        from app.learning.pipeline_metrics import THRESHOLD_REGISTRY
        for name, at in THRESHOLD_REGISTRY.items():
            if name in thresholds and at.is_anomaly(thresholds[name]):
                anomaly_flags.append(f"{name}={thresholds[name]:.3f}")

        bus.publish_adaptive_thresholds(thresholds, anomaly_flags, all_drift)
    except Exception as e:
        logger.debug(f"Adaptive thresholds publish skipped: {e}")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 2: Compute cross-loop derived signals                   ║
    # ╚══════════════════════════════════════════════════════════════════╝

    bus.compute_derived_signals()

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 3: Cross-pollination adjustments                        ║
    # ║  Each loop reads OTHER loops' signals and makes small tweaks    ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # 3a. Source Bandit: reward novelty-producing sources more
    try:
        mod = bus.get_source_bandit_modulation()
        if mod["novelty_bonus_active"] and hasattr(deps, "source_bandit"):
            bandit = deps.source_bandit
            # When most trends are stale (peak/decline), sources producing
            # novel (birth) trends deserve a small exploration bonus
            for source_id in bus.top_sources[:3]:
                p = bandit._posteriors.get(source_id)
                if p:
                    bonus = 0.3 * mod["exploration_budget"]
                    p["alpha"] += bonus
            bandit._save()
            logger.debug(
                f"Cross-loop: source bandit novelty bonus applied "
                f"(exploration_budget={mod['exploration_budget']:.2f})"
            )
    except Exception as e:
        logger.debug(f"Cross-loop source bandit modulation skipped: {e}")

    # 3b. Company Bandit: explore more when birth trends are high
    try:
        mod = bus.get_company_bandit_modulation()
        if mod["birth_trend_ratio"] > 0.40 and hasattr(deps, "company_bandit"):
            company_bandit = deps.company_bandit
            # New trends need more company exploration  - gently pull posteriors
            # toward uniform prior to increase Thompson Sampling variance
            shrink = 0.97  # Very gentle
            adjusted = 0
            for arm_id, p in company_bandit._posteriors.items():
                total = p["alpha"] + p["beta"]
                if total > 15:
                    p["alpha"] = 1.0 + (p["alpha"] - 1.0) * shrink
                    p["beta"] = 1.0 + (p["beta"] - 1.0) * shrink
                    adjusted += 1
            if adjusted:
                company_bandit._save()
                logger.debug(
                    f"Cross-loop: company bandit exploration boost  - "
                    f"{adjusted} arms pulled toward prior (birth_ratio={mod['birth_trend_ratio']:.2f})"
                )
    except Exception as e:
        logger.debug(f"Cross-loop company bandit modulation skipped: {e}")

    # 3c. Apply previous MetaReasoner improvement hypotheses
    try:
        reasoner = deps.meta_reasoner
        prev_hypotheses = reasoner.get_active_hypotheses()
        applied_count = 0
        for hyp in prev_hypotheses:
            target = hyp.get("target", "")
            action = hyp.get("action", "").lower()
            priority = hyp.get("priority", "medium")

            # Source-related hypotheses -> adjust exploration
            if target in ("source_intel", "source_bandit", "sources"):
                if "explor" in action and hasattr(deps, "source_bandit"):
                    bandit = deps.source_bandit
                    # Increase variance on uncertain arms
                    for sid, p in bandit._posteriors.items():
                        if 0.30 <= p["alpha"] / (p["alpha"] + p["beta"]) <= 0.70:
                            nudge = 0.15 if priority == "high" else 0.08
                            p["alpha"] += nudge
                            p["beta"] += nudge
                    bandit._save()
                    applied_count += 1

            # Weight-related hypotheses -> flag for conservative learning
            elif target in ("weight_learner", "weights"):
                if "conserv" in action or "slow" in action:
                    # Already handled by signal bus lr_multiplier
                    applied_count += 1

        if applied_count:
            logger.info(
                f"Cross-loop: applied {applied_count}/{len(prev_hypotheses)} "
                f"MetaReasoner hypotheses from previous run"
            )
    except Exception as e:
        logger.debug(f"Cross-loop hypothesis application skipped: {e}")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 4: MetaReasoner retrospective + publish to bus           ║
    # ╚══════════════════════════════════════════════════════════════════╝

    avg_oss = sum(s.oss_score for s in signals) / len(signals) if signals else 0.0
    avg_kb_hit = sum(s.kb_hit_rate for s in signals) / len(signals) if signals else 0.0
    mean_quality = (quality_sum / max(len(signals), 1)) if signals else 0.0

    retro = None
    try:
        reasoner = deps.meta_reasoner
        total_duration = _time.time() - getattr(deps, "_pipeline_t0", _time.time())

        # Load previous retrospective for comparison
        prev_retro = None
        try:
            retro_path = Path("./data/reasoning_traces.jsonl")
            if retro_path.exists():
                lines = retro_path.read_text(encoding="utf-8").strip().split("\n")
                for line in reversed(lines):
                    data = json.loads(line)
                    if data.get("step") == "retrospective" and data.get("run_id") != state.get("run_id", ""):
                        prev_retro = data
                        break
        except Exception:
            pass

        retro = await reasoner.run_retrospective(
            run_id=state.get("run_id", ""),
            total_duration_s=total_duration,
            trend_count=len(state.get("trends", [])),
            lead_count=len(getattr(deps, "_lead_sheets", [])),
            mean_oss=avg_oss,
            mean_kb_hit=avg_kb_hit,
            mean_quality=mean_quality,
            signal_bus_summary={
                "system_confidence": bus.system_confidence,
                "exploration_budget": bus.exploration_budget,
                "learning_path": bus.learning_path,
                "source_degraded": bus.source_degraded,
            },
            previous_retrospective=prev_retro,
        )

        # Publish reasoning traces + retrospective to the signal bus
        run_summary = reasoner.get_run_summary()
        retro_dict = None
        if retro and retro.run_grade:
            from dataclasses import asdict as _asdict
            retro_dict = _asdict(retro)
        bus.publish_reasoning(run_summary, retro_dict)

        # Re-compute derived signals with reasoning data included
        bus.compute_derived_signals()

        logger.info(
            f"MetaReasoner retrospective complete: grade={retro.run_grade if retro else '?'}, "
            f"reasoning_quality={run_summary.get('avg_reasoning_quality', 0):.2f}"
        )
    except Exception as e:
        logger.debug(f"MetaReasoner retrospective skipped: {e}")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 4b: Dataset Enhancement  - auto-label from cluster signals ║
    # ║  Extracts positive/negative examples from cluster coherence     ║
    # ║  No human input needed  - self-improving from pipeline signals   ║
    # ╚══════════════════════════════════════════════════════════════════╝
    try:
        from app.learning.dataset_enhancer import DatasetEnhancer
        enhancer = DatasetEnhancer()

        # Cold start: bootstrap from Reuters + AG News if dataset is empty (first run)
        if enhancer.get_stats()["total"] == 0:
            logger.info("[dataset_enhancer] Cold start: bootstrapping from Reuters + AG News...")
            r_pos, r_neg = enhancer.bootstrap_from_reuters(n_per_class=30)
            ag_pos, ag_neg = enhancer.bootstrap_from_ag_news(n_per_class=50)
            logger.info(
                f"[dataset_enhancer] Bootstrap complete: "
                f"Reuters +{r_pos}/{r_neg}, AG News +{ag_pos}/{ag_neg}"
            )

        # Extract labels from this run's cluster quality signals
        clusters = getattr(deps, "_clusters", [])
        if clusters:
            pos, neg = enhancer.extract_labels_from_clusters(clusters)
            logger.info(f"[dataset_enhancer] Extracted from clusters: +{pos} pos, +{neg} neg")

        # Trigger SetFit retraining if we have enough examples
        if enhancer.should_trigger_retrain():
            try:
                from app.learning.hypothesis_learner import HypothesisLearner
                learner = HypothesisLearner()
                positives, negatives = enhancer.get_examples_for_setfit(max_per_class=64)
                # Inject dataset examples into feedback store path
                logger.info(
                    f"[dataset_enhancer] Triggering SetFit retraining: "
                    f"{len(positives)} pos / {len(negatives)} neg"
                )
                # Use HypothesisLearner's internal methods with our dataset examples
                updated = await learner._train_and_update(positives, negatives)
                if updated:
                    logger.info("[dataset_enhancer] SetFit hypothesis updated from dataset")
                else:
                    logger.warning("[dataset_enhancer] SetFit update failed  - keeping current hypothesis")
            except Exception as exc:
                logger.warning(f"[dataset_enhancer] SetFit trigger failed: {exc}")

        stats = enhancer.get_stats()
        logger.info(
            f"[dataset_enhancer] Dataset: total={stats['total']} "
            f"pos={stats['positives']} neg={stats['negatives']} "
            f"ready={stats['ready_for_retrain']}"
        )
    except Exception as exc:
        logger.warning(f"Dataset enhancement skipped: {exc}")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 5: Persist bus + finalize                               ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # Save bus state for next run
    from datetime import datetime, timezone as _tz
    bus.timestamp = datetime.now(_tz.utc).isoformat()
    bus.save()

    # -- Summary ----------------------------------------------------------
    logger.info(
        f"Learning update complete: {len(signals)} trends | "
        f"avg_oss={avg_oss:.3f} | avg_kb_hit={avg_kb_hit:.1%}"
    )
    logger.info(f"Signal bus: {bus.summary()}")

    _record(deps, "learning_update_complete", {
        "signal_count": len(signals),
        "avg_oss": round(avg_oss, 3),
        "avg_kb_hit": round(avg_kb_hit, 3),
        "signals": [s.to_dict() for s in signals] if signals else [],
        "signal_bus": {
            "system_confidence": bus.system_confidence,
            "exploration_budget": bus.exploration_budget,
            "learning_path": bus.learning_path,
            "total_drift": bus.total_drift,
            "source_degraded": bus.source_degraded,
            "feedback_distribution": bus.feedback_distribution,
            "reasoning_grade": bus.reasoning_run_grade,
            "reasoning_quality": bus.reasoning_quality,
            "strategy_adjustments": bus.reasoning_strategy_adjustments,
        },
        "retrospective": {
            "grade": retro.run_grade if retro else "",
            "summary": retro.summary if retro else "",
            "successes": retro.successes[:3] if retro else [],
            "failures": retro.failures[:3] if retro else [],
            "improvement_plan": retro.improvement_plan[:3] if retro else [],
        } if retro else None,
    }, t0)

    # Save recording manifest (finalizes the recording for replay)
    recorder = getattr(deps, "recorder", None)
    if recorder:
        total_duration = _time.time() - getattr(deps, "_pipeline_t0", _time.time())
        recorder.save_manifest(total_duration)

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PHASE 6: AutoResearch  - evaluate experiment + keep/discard    ║
    # ╚══════════════════════════════════════════════════════════════════╝
    try:
        actionable_count = sum(1 for s in signals if s.oss_score > 0.4) if signals else 0
        actionable_rate = actionable_count / max(len(signals), 1)
        noise_rate_val = getattr(pipeline, "_noise_rate", 0.0) if pipeline else 0.0

        # Check for active hypothesis
        active_hyp = getattr(deps, "_active_hypothesis", None)

        experiment = ExperimentRecord(
            run_id=state.get("run_id", bus.run_id),
            hypothesis=active_hyp.description if active_hyp else "",
            mean_oss=avg_oss,
            mean_coherence=getattr(pipeline, "_mean_coherence", 0.0) if pipeline else 0.0,
            noise_rate=noise_rate_val,
            actionable_rate=actionable_rate,
            article_count=len(articles),
            cluster_count=len(signals),
            learning_updates=learning_updates,
        )

        if tracker.is_regression(experiment):
            experiment.status = "discard"
            experiment.reason = "regression detected  - rolled back weights + thresholds"
            restore_learning_state()
        else:
            experiment.status = "keep"
            cleanup_snapshot()

        tracker.record(experiment)

        # Mark hypothesis outcome
        if active_hyp and active_hyp.param_changes:
            baseline = tracker.rolling_baseline()
            result_delta = {}
            if baseline:
                result_delta = {
                    "mean_oss": round(avg_oss - baseline["mean_oss"], 4),
                    "actionable_rate": round(actionable_rate - baseline["actionable_rate"], 4),
                }
            hyp_status = f"tested:{experiment.status}"
            mark_hypothesis_tested(
                active_hyp.id, experiment.run_id, hyp_status, result_delta,
            )

        oss_trend = tracker.trend("mean_oss")
        if oss_trend != "stable":
            logger.info(f"AutoResearch: quality trend = {oss_trend}")

    except Exception as e:
        logger.debug(f"Experiment tracking skipped: {e}")
        cleanup_snapshot()

    retro_summary = ""
    if retro and retro.run_grade:
        retro_summary = f" | grade={retro.run_grade}"
        if retro.improvement_plan:
            top_plan = retro.improvement_plan[0]
            if isinstance(top_plan, dict):
                retro_summary += f" | next: {top_plan.get('action', '')[:60]}"

    return {
        "errors": [],
        "current_step": "learning_update_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "learning_update": (
                f"{len(signals)} signals | avg_oss={avg_oss:.3f} | "
                f"avg_kb_hit={avg_kb_hit:.1%} | bus: {bus.summary()}{retro_summary}"
            ),
        },
    }


# -- Graph Construction -------------------------------------------------------

def create_pipeline_graph():
    """Build and compile the multi-agent LangGraph with quality gate + checkpointing.

    Flow:
      source_intel -> analysis -> impact -> quality_validation
        +- (retry) -> analysis
        +- (viable) -> causal_council -> lead_crystallize -> lead_gen -> END
        └- (no trends) -> END

    LangGraph best practices applied:
      - InMemorySaver checkpointer: state is snapshotted after every node
      - RetryPolicy on API-heavy nodes: retries on transient failures
    """
    _api_retry = RetryPolicy(max_attempts=2)

    workflow = StateGraph(GraphState)

    workflow.add_node("source_intel",        source_intel_node,      retry=_api_retry)
    workflow.add_node("analysis",            analysis_node)
    workflow.add_node("impact",              impact_node,            retry=_api_retry)
    workflow.add_node("quality_validation",  quality_validation_node)
    workflow.add_node("causal_council",      causal_council_node,    retry=_api_retry)
    workflow.add_node("lead_crystallize",    lead_crystallize_node)
    workflow.add_node("lead_gen",            lead_gen_node,          retry=_api_retry)
    workflow.add_node("learning_update",     learning_update_node)   # Always runs at end

    workflow.add_edge(START, "source_intel")
    workflow.add_conditional_edges(
        "source_intel",
        source_intel_route,
        {"analysis": "analysis", "learning_update": "learning_update"},
    )
    workflow.add_edge("analysis", "impact")
    workflow.add_edge("impact", "quality_validation")
    workflow.add_conditional_edges(
        "quality_validation",
        quality_route,
        # "end" now routes to learning_update (not END) so learning fires even on skip
        {"analysis": "analysis", "lead_gen": "causal_council", "end": "learning_update"},
    )
    workflow.add_edge("causal_council", "lead_crystallize")
    workflow.add_edge("lead_crystallize", "lead_gen")
    workflow.add_edge("lead_gen", "learning_update")
    workflow.add_edge("learning_update", END)

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# -- Step Progress Formatter --------------------------------------------------

_STEP_LABELS = {
    "source_intel_complete": ("FETCH+DEDUP+NLI", "Articles collected and filtered"),
    "source_intel_empty":    ("FETCH",           "No articles found - skipping to learning"),
    "analysis_complete":     ("CLUSTER+TRENDS",  "Clusters validated, trends synthesized"),
    "impact_complete":       ("IMPACT",          "Impact council analysis complete"),
    "quality_retry_analysis":("QUALITY",         "Low quality - retrying clustering"),
    "quality_complete":      ("QUALITY",         "Quality gate passed"),
    "lead_gen_complete":     ("LEAD GEN",        "Opportunity scoring complete"),
    "causal_council_complete":("CAUSAL",         "Causal chain analysis complete"),
    "lead_crystallize_complete":("LEADS",        "Call sheets crystallized"),
    "learning_update_complete": ("LEARN",        "Self-learning loops updated"),
}


def _format_step_progress(node_name: str, step: str, state: dict, deps: Any) -> str:
    """Format a one-line progress message for a pipeline step transition."""
    label, base_desc = _STEP_LABELS.get(step, (node_name.upper(), step))

    # Augment with live counts from deps/state
    articles = getattr(deps, "_articles", []) if deps else []
    trends = state.get("trends", [])
    companies = state.get("companies", [])
    contacts = state.get("contacts", [])
    lead_sheets = getattr(deps, "_lead_sheets", []) if deps else []
    impacts = state.get("impacts", [])

    detail = ""
    if step == "source_intel_complete":
        detail = f"{len(articles)} articles"
    elif step == "analysis_complete":
        detail = f"{len(trends)} trends detected"
    elif step == "impact_complete":
        detail = f"{len(impacts)} impacts analyzed"
    elif step in ("quality_complete", "quality_retry_analysis"):
        viable = len(state.get("impacts", []))
        detail = f"{viable} viable trends"
    elif step == "lead_gen_complete":
        detail = f"{len(companies)} companies scored"
    elif step in ("causal_council_complete", "lead_crystallize_complete"):
        detail = f"{len(lead_sheets)} call sheets"
    elif step == "learning_update_complete":
        detail = "source bandit + NLI hypothesis + 4 other loops"

    suffix = f"  ->  {detail}" if detail else ""
    return f"[{label}]  {base_desc}{suffix}"


# -- Public Entry Point -------------------------------------------------------

async def run_pipeline(mock_mode: bool = False, log_callback=None, scope=None) -> PipelineResult:
    """Execute the multi-agent sales intelligence pipeline.

    Uses LangGraph astream (stream_mode="updates") which yields per-node output
    deltas  - avoids serialization issues with complex AgentDeps objects.
    InMemorySaver checkpointer snapshots state after every node for resilience.

    scope: Optional DiscoveryScope from CLI  - sets region/hours/mode/companies/products.
           When None, falls back to global settings (FastAPI path).
    """
    start_time = datetime.utcnow()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")

    logger.info("Starting Multi-Agent Sales Intelligence Pipeline")
    logger.info(f"Run ID: {run_id} | Mock Mode: {mock_mode}")
    if scope:
        logger.info(
            f"Scope: mode={getattr(scope, 'mode', '?')} "
            f"region={getattr(scope, 'region', '?')} "
            f"hours={getattr(scope, 'hours', '?')}"
        )

    # Reset provider health and cooldowns so stale failures from previous runs
    # don't block this run's LLM calls (Fix 3A).
    from ..tools.llm.providers import provider_health, ProviderManager
    provider_health.reset_for_new_run()
    ProviderManager.reset_cooldowns()
    logger.info("Provider health + cooldowns reset for new run")

    from .deps import AgentDeps
    deps = AgentDeps.create(
        mock_mode=mock_mode,
        log_callback=log_callback,
        run_id=run_id,
        scope=scope,
    )

    deps._pipeline_t0 = _time.time()

    # -- AutoResearch: load & apply top hypothesis for this run ---------
    active_hypothesis = None
    try:
        from app.learning.experiment_tracker import pick_next_hypothesis
        active_hypothesis = pick_next_hypothesis()
        if active_hypothesis and active_hypothesis.param_changes:
            logger.info(
                f"AutoResearch: testing hypothesis '{active_hypothesis.description}' "
                f"(params: {active_hypothesis.param_changes})"
            )
            # Apply param changes to adaptive thresholds for this run
            from app.intelligence.config import load_adaptive_params
            params = load_adaptive_params()
            for key, value in active_hypothesis.param_changes.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            # Store modified params on deps for pipeline to use
            deps._hypothesis_params = params
            deps._active_hypothesis = active_hypothesis
    except Exception as e:
        logger.debug(f"Hypothesis loading skipped: {e}")

    initial_state: GraphState = {
        "deps": deps,
        "run_id": run_id,
        "trends": [],
        "impacts": [],
        "companies": [],
        "contacts": [],
        "outreach_emails": [],
        "errors": [],
        "current_step": "init",
        "retry_counts": {},
        "agent_reasoning": {},
    }

    # Thread ID ties checkpointer snapshots to this pipeline run
    config = {"configurable": {"thread_id": run_id}}

    final_state: dict = dict(initial_state)
    stream_errors: list = []
    node_name = "unknown"

    try:
        graph = create_pipeline_graph()

        # stream_mode="updates" emits per-node output deltas — avoids the
        # msgpack serialization of complex AgentDeps that crashes "values" mode.
        _last_step = "init"

        try:
            async for node_output in graph.astream(initial_state, config, stream_mode="updates"):
                for node_name, delta in node_output.items():
                    if isinstance(delta, dict):
                        final_state.update(delta)

                new_step = final_state.get("current_step", "")
                if new_step != _last_step:
                    msg = _format_step_progress(node_name, new_step, final_state, deps)
                    logger.info(msg)
                    if log_callback:
                        log_callback(msg, "step")
                    _last_step = new_step
        except Exception as stream_err:
            # LangGraph's InMemorySaver raises a msgpack serialization error at
            # stream cleanup when AgentDeps (with model weights etc.) is in state.
            # All pipeline nodes have already completed at this point — treat it
            # as a checkpoint warning, not a pipeline failure.
            err_str = str(stream_err)
            if "msgpack" in err_str.lower() or "serializ" in err_str.lower():
                logger.warning(f"Checkpoint serialization warning (non-fatal): {stream_err}")
            else:
                logger.error(f"Pipeline stream error: {stream_err}")
                stream_errors.append(err_str)

        trends_count = len(final_state.get("trends", []))
        companies_count = len(final_state.get("companies", []))
        contacts_count = len(final_state.get("contacts", []))
        emails_count = len([
            c for c in final_state.get("contacts", [])
            if getattr(c, "email", "")
        ])
        outreach_count = len(final_state.get("outreach_emails", []))

        output_file = await save_outputs(final_state, run_id)

        runtime = (datetime.utcnow() - start_time).total_seconds()

        reasoning = final_state.get("agent_reasoning", {})
        for agent_name, reason in reasoning.items():
            logger.info(f"Agent [{agent_name}] reasoning: {str(reason)[:200]}")

        logger.info("=" * 50)
        logger.info("MULTI-AGENT PIPELINE COMPLETED")
        logger.info(f"Trends: {trends_count} | Companies: {companies_count} | "
                     f"Contacts: {contacts_count} | Emails: {emails_count} | "
                     f"Outreach: {outreach_count} | Runtime: {runtime:.1f}s")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 50)

        all_errors = final_state.get("errors", []) + stream_errors
        return PipelineResult(
            status="success",
            leads_generated=outreach_count,
            trends_detected=trends_count,
            companies_found=companies_count,
            emails_found=emails_count,
            output_file=output_file,
            errors=all_errors,
            run_time_seconds=runtime,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return PipelineResult(
            status="error",
            errors=[str(e)],
            run_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
        )


# -- Output Persistence -------------------------------------------------------

async def save_outputs(state: GraphState, run_id: str) -> str:
    """Save pipeline outputs to JSON and CSV files."""
    outputs_dir = Path("app/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    trends = state.get("trends", [])
    impacts = state.get("impacts", [])
    companies = state.get("companies", [])
    contacts = state.get("contacts", [])
    outreach_list = state.get("outreach_emails", [])

    trend_map = {t.id: t for t in trends}
    impact_map = {i.trend_id: i for i in impacts}
    company_map = {c.id: c for c in companies}
    contact_map = {c.id: c for c in contacts}

    leads = []
    for outreach in outreach_list:
        contact = contact_map.get(outreach.contact_id)
        company = company_map.get(contact.company_id) if contact else None
        trend_id = company.trend_id if company else ""
        trend = trend_map.get(trend_id)
        impact = impact_map.get(trend_id)

        imp = impact_map.get(trend_id) if trend_id else None
        lead = {
            "id": outreach.id,
            "trend": {
                "id": trend.id if trend else "",
                "title": trend.trend_title if trend else outreach.trend_title,
                "summary": trend.summary if trend else "",
                "severity": trend.severity if trend else "medium",
                "industries": trend.industries_affected if trend else [],
                "keywords": trend.keywords if trend else [],
                "trend_score": trend.trend_score if trend else 0.0,
                "actionability_score": trend.actionability_score if trend else 0.0,
                "actionable_insight": trend.actionable_insight if trend else "",
                "event_5w1h": trend.event_5w1h if trend else {},
                "causal_chain": trend.causal_chain if trend else [],
                "buying_intent": trend.buying_intent if trend else {},
                "affected_companies": trend.affected_companies if trend else [],
                "affected_regions": trend.affected_regions if trend else [],
                "article_count": trend.article_count if trend else 0,
            },
            "impact": {
                "first_order": imp.direct_impact if imp else [],
                "second_order": imp.indirect_impact if imp else [],
                "pain_points": imp.midsize_pain_points if imp else [],
                "pitch_angle": imp.pitch_angle if imp else "",
                "relevant_services": imp.relevant_services if imp else [],
                "confidence": imp.council_confidence if imp else 0.0,
            },
            "company": {
                "id": company.id if company else "",
                "name": company.company_name if company else outreach.company_name,
                "size": company.company_size if company else "mid",
                "industry": company.industry if company else "",
                "website": company.website if company else "",
                "domain": company.domain if company else "",
                "description": company.description if company else "",
                "reason_relevant": company.reason_relevant if company else "",
                "founded_year": company.founded_year if company else None,
                "headquarters": company.headquarters if company else "",
                "employee_count": company.employee_count if company else "",
                "stock_ticker": company.stock_ticker if company else "",
                "ceo": company.ceo if company else "",
                "funding_stage": company.funding_stage if company else "",
                "tech_stack": company.tech_stack if company else [],
                "ner_verified": company.ner_verified if company else False,
                "verification_source": company.verification_source if company else "",
                "verification_confidence": company.verification_confidence if company else 0.0,
                "target_roles": company.target_roles if company else [],
            },
            "contact": {
                "id": contact.id if contact else "",
                "name": contact.person_name if contact else outreach.person_name,
                "role": contact.role if contact else outreach.role,
                "email": contact.email if contact else outreach.email,
                "email_confidence": contact.email_confidence if contact else 0,
                "email_source": contact.email_source if contact else "",
                "email_verified": contact.verified if contact else False,
                "linkedin": contact.linkedin_url if contact else "",
            },
            "outreach": {
                "subject": outreach.subject,
                "body": outreach.body,
                "email_confidence": outreach.email_confidence,
            },
            "generated_at": outreach.generated_at.isoformat()
            if hasattr(outreach.generated_at, 'isoformat')
            else str(outreach.generated_at),
        }
        leads.append(lead)

    def _json_default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'value'):
            return obj.value
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    json_file = outputs_dir / f"leads_{run_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "total_leads": len(leads),
            "leads": leads,
        }, f, indent=2, ensure_ascii=False, default=_json_default)

    # Save structured call sheets produced by CausalCouncil + LeadCrystallizer.
    # These are the primary deliverable of the new pipeline  - one file per run,
    # separate from the old-style outreach emails so both can coexist during migration.
    deps = state.get("deps")
    lead_sheets = getattr(deps, "_lead_sheets", []) if deps else []
    if lead_sheets:
        call_sheets_file = outputs_dir / f"call_sheets_{run_id}.json"
        with open(call_sheets_file, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "generated_at": datetime.utcnow().isoformat(),
                "total_call_sheets": len(lead_sheets),
                "call_sheets": [
                    {
                        "company_name": ls.company_name,
                        "company_cin": ls.company_cin,
                        "company_state": ls.company_state,
                        "company_city": ls.company_city,
                        "company_size_band": ls.company_size_band,
                        "hop": ls.hop,
                        "lead_type": ls.lead_type,
                        "trend_title": ls.trend_title,
                        "event_type": ls.event_type,
                        "contact_role": ls.contact_role,
                        "trigger_event": ls.trigger_event,
                        "pain_point": ls.pain_point,
                        "service_pitch": ls.service_pitch,
                        "opening_line": ls.opening_line,
                        "urgency_weeks": ls.urgency_weeks,
                        "confidence": ls.confidence,
                        "reasoning": ls.reasoning,
                        "data_sources": ls.data_sources,
                        "oss_score": ls.oss_score,
                    }
                    for ls in lead_sheets
                ],
            }, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info(f"Call sheets saved: {call_sheets_file} ({len(lead_sheets)} sheets)")

    # Save learning signals for autonomous weight adaptation
    signals = getattr(deps, "_signals", []) if deps else []
    if signals:
        signals_file = outputs_dir / f"signals_{run_id}.json"
        with open(signals_file, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "signals": [s.to_dict() for s in signals],
            }, f, indent=2, ensure_ascii=False, default=_json_default)

    # -- Comprehensive run report (cluster validation + first/second-order view) --
    # Joins article titles from deps._articles into each trend card so you can
    # verify that clusters contain real, coherent news  - not garbage articles.
    articles = getattr(deps, "_articles", []) if deps else []
    article_by_id = {str(getattr(a, 'id', '')): a for a in articles}
    trend_tree = getattr(deps, "_trend_tree", None) if deps else None
    pipeline = getattr(deps, "_pipeline", None) if deps else None
    causal_results = getattr(deps, "_causal_results", []) if deps else []
    lead_sheets = getattr(deps, "_lead_sheets", []) if deps else []

    # Build article title lookup per trend (from TrendTree source_articles UUIDs)
    def _get_article_titles(trend: "TrendData") -> list:
        if trend_tree is None:
            return []
        node = trend_tree.nodes.get(trend.id)
        if node is None:
            return []
        titles = []
        for art_id in node.source_articles[:20]:  # cap at 20 for readability
            art = article_by_id.get(str(art_id))
            if art:
                titles.append({
                    "title": getattr(art, "title", ""),
                    "source": getattr(art, "source_name", getattr(art, "source_id", "")),
                    "url": getattr(art, "url", ""),
                })
        return titles

    pipeline_metrics = {}
    if pipeline is not None:
        pipeline_metrics = getattr(pipeline, "_last_metrics", {})

    run_report = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_trends": len(trends),
            "total_impacts": len(impacts),
            "total_call_sheets": len(lead_sheets),
            "total_articles_processed": len(articles),
        },
        "pipeline_metrics": pipeline_metrics,
        "trends": [
            {
                "id": t.id,
                "title": t.trend_title,
                "summary": t.summary,
                "trend_type": t.trend_type,
                "severity": t.severity if isinstance(t.severity, str) else getattr(t.severity, "value", str(t.severity)),
                "industries": t.industries_affected,
                "keywords": t.keywords,
                "trend_score": t.trend_score,
                "actionability_score": t.actionability_score,
                "actionable_insight": t.actionable_insight,
                "article_count": t.article_count,
                "causal_chain": t.causal_chain,
                "affected_companies": t.affected_companies,
                "affected_regions": t.affected_regions,
                "event_5w1h": t.event_5w1h,
                "buying_intent": t.buying_intent,
                # Article titles joined from TrendTree  - validate cluster quality here
                "source_articles": _get_article_titles(t),
                # Impact analysis (first-order direct, second-order indirect)
                "impact": (lambda imp: {
                    "first_order": imp.direct_impact if imp else [],
                    "second_order": imp.indirect_impact if imp else [],
                    "pain_points": imp.midsize_pain_points if imp else [],
                    "pitch_angle": imp.pitch_angle if imp else "",
                    "confidence": imp.council_confidence if imp else 0.0,
                    "relevant_services": imp.relevant_services if imp else [],
                })(impact_map.get(t.id)),
            }
            for t in trends
        ],
        "call_sheets": [
            {
                "trend_title": ls.trend_title,
                "company_name": ls.company_name,
                "company_cin": ls.company_cin,
                "company_city": ls.company_city,
                "company_state": ls.company_state,
                "company_size_band": ls.company_size_band,
                "hop": ls.hop,
                "lead_type": ls.lead_type,
                "contact_role": ls.contact_role,
                "trigger_event": ls.trigger_event,
                "pain_point": ls.pain_point,
                "service_pitch": ls.service_pitch,
                "opening_line": ls.opening_line,
                "urgency_weeks": ls.urgency_weeks,
                "confidence": ls.confidence,
                "oss_score": ls.oss_score,
            }
            for ls in lead_sheets
        ],
        "causal_chains": [
            {
                "event_summary": r.event_summary,
                "event_type": r.event_type,
                "hops": [
                    {
                        "hop": h.hop,
                        "segment": h.segment,
                        "lead_type": h.lead_type,
                        "mechanism": h.mechanism,
                        "companies": h.companies_found[:10],
                        "confidence": h.confidence,
                    }
                    for h in r.hops
                ],
            }
            for r in causal_results
        ],
        "errors": state.get("errors", []),
    }

    report_file = outputs_dir / f"run_report_{run_id}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info(f"Run report saved: {report_file} ({len(trends)} trends, {len(lead_sheets)} call sheets)")

    csv_file = outputs_dir / f"leads_{run_id}.csv"
    if leads:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trend", "Company", "Industry", "Website", "Domain",
                "Contact Name", "Role", "Email", "Email Confidence",
                "Subject", "Body Preview",
            ])
            for lead in leads:
                writer.writerow([
                    lead["trend"]["title"],
                    lead["company"]["name"],
                    lead["company"]["industry"],
                    lead["company"]["website"],
                    lead["company"]["domain"],
                    lead["contact"]["name"],
                    lead["contact"]["role"],
                    lead["contact"]["email"],
                    lead["contact"]["email_confidence"],
                    lead["outreach"]["subject"],
                    lead["outreach"]["body"][:100] + "...",
                ])

    # -- Persist to DB ------------------------------------------------------
    try:
        db = get_database()

        # Save pipeline run summary
        db.save_pipeline_run({
            "run_id": run_id,
            "status": "completed",
            "trends_detected": len(trends),
            "companies_found": len(companies),
            "leads_generated": len(lead_sheets),
            "contacts_found": len(contacts),
            "output_file": str(json_file),
            "errors": state.get("errors", []),
            "run_time_seconds": 0,  # Will be set by caller
        })

        # Save trends
        for trend in trends:
            db.save_trend(run_id, trend)

        # Save call sheets with enrichment data (contacts, emails from lead_gen)
        _co_by_name = {getattr(c, "company_name", ""): c for c in companies}
        _ct_by_co = {}
        for ct in contacts:
            cn = getattr(ct, "company_name", "")
            _ct_by_co.setdefault(cn, []).append(ct)
        _em_by_co = {}
        for em in getattr(deps, "_outreach", []) if deps else []:
            cn = getattr(em, "company_name", "")
            if cn not in _em_by_co:
                _em_by_co[cn] = em

        # Build person profile index  - fallback for email if _outreach is empty
        person_profiles = getattr(deps, "_person_profiles", []) if deps else []
        _pp_by_co: dict = {}
        for p in person_profiles:
            cn = getattr(p, "company_name", "")
            if cn not in _pp_by_co:
                _pp_by_co.setdefault(cn, []).append(p)
            else:
                _pp_by_co[cn].append(p)

        for sheet in lead_sheets:
            cn = getattr(sheet, "company_name", "")
            co = _co_by_name.get(cn)
            cts = _ct_by_co.get(cn, [])
            ct = next((c for c in cts if getattr(c, "email", "")), cts[0] if cts else None)
            em = _em_by_co.get(cn)

            # Primary email  - from OutreachEmail, or fallback to person profile outreach
            email_subject = getattr(em, "subject", "") if em else ""
            email_body = getattr(em, "body", "") if em else ""
            if not email_subject:
                pp_list = _pp_by_co.get(cn, [])
                pp = next((p for p in pp_list if getattr(p, "outreach_subject", "")), None)
                if pp:
                    email_subject = getattr(pp, "outreach_subject", "")
                    email_body = getattr(pp, "outreach_body", "")

            # Primary contact  - from ContactData, or fallback to person profile
            contact_name = getattr(ct, "person_name", "") if ct else ""
            contact_email = getattr(ct, "email", "") if ct else ""
            contact_role = getattr(ct, "role", "") if ct else ""
            contact_linkedin = getattr(ct, "linkedin_url", "") if ct else ""
            email_confidence = getattr(ct, "email_confidence", 0) if ct else 0
            if not contact_name:
                pp_list = _pp_by_co.get(cn, [])
                pp = next((p for p in pp_list if getattr(p, "person_name", "")), None)
                if pp:
                    contact_name = getattr(pp, "person_name", "")
                    contact_email = contact_email or getattr(pp, "email", "")
                    contact_role = contact_role or getattr(pp, "role", "")
                    contact_linkedin = contact_linkedin or getattr(pp, "linkedin_url", "")
                    email_confidence = email_confidence or getattr(pp, "email_confidence", 0)

            enrichment = {
                "company_website": getattr(co, "website", "") if co else "",
                "company_domain": getattr(co, "domain", "") if co else "",
                "reason_relevant": getattr(co, "reason_relevant", "") if co else "",
                "contact_name": contact_name,
                "contact_role": contact_role or getattr(sheet, "contact_role", ""),
                "contact_email": contact_email,
                "contact_linkedin": contact_linkedin,
                "email_confidence": email_confidence,
                "email_subject": email_subject,
                "email_body": email_body,
                "company_news": getattr(sheet, "company_news", []),
            }
            db.save_call_sheet(run_id, sheet, enrichment=enrichment)

        # Save person profiles (multiple contacts per company with reach scores)
        if person_profiles:
            saved_count = db.save_lead_contacts(run_id, person_profiles)
            logger.info(f"DB: saved {saved_count} person profiles")

        logger.info(
            f"DB: saved run + {len(trends)} trends + "
            f"{len(lead_sheets)} call sheets"
        )
    except Exception as e:
        logger.warning(f"Failed to save to database: {e}")

    return str(json_file)
