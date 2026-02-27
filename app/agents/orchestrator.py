"""
LangGraph Orchestrator — multi-agent supervisor for sales intelligence pipeline.

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


# ── Graph State ──────────────────────────────────────────────────────────────

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


# ── Recording helper ──────────────────────────────────────────────────────────

def _record(deps, step_name: str, data: dict, t0: float):
    """Record step snapshot if recorder is active (real runs only)."""
    recorder = getattr(deps, "recorder", None)
    if recorder:
        try:
            recorder.record_step(step_name, data, _time.time() - t0)
        except Exception as e:
            logger.debug(f"Recording step {step_name} failed: {e}")


# ── Agent Nodes ──────────────────────────────────────────────────────────────

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

        if len(articles) < 3:
            errors.append(f"Only {len(articles)} articles — need at least 3")

    except Exception as e:
        logger.error(f"Source Intel node failed: {e}")
        errors.append(f"Source Intel: {e}")
        result = None

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
        "article_count": len(getattr(deps, "_articles", [])),
    }, t0)

    return {
        "errors": errors,
        "current_step": "source_intel_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "source_intel": getattr(result, 'reasoning', str(errors)),
        },
    }


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
        tree, result = await run_analysis(deps)

        # Convert TrendTree -> List[TrendData] with all synthesis fields
        trends = []
        if tree:
            major_trends = tree.to_major_trends()
            trends = [
                TrendData(
                    id=str(mt.id),
                    trend_title=mt.trend_title,
                    summary=mt.trend_summary,
                    severity=mt.severity if isinstance(mt.severity, str) else mt.severity,
                    industries_affected=[
                        s.value if hasattr(s, 'value') else str(s)
                        for s in mt.primary_sectors
                    ],
                    source_links=[],
                    keywords=mt.key_entities[:10] if mt.key_entities else mt.key_keywords[:10],
                    trend_type=mt.trend_type.value if hasattr(mt.trend_type, 'value') else str(mt.trend_type),
                    actionable_insight=mt.actionable_insight or "",
                    event_5w1h=mt.event_5w1h or {},
                    causal_chain=mt.causal_chain or [],
                    buying_intent=mt.buying_intent or {},
                    affected_companies=mt.affected_companies or [],
                    affected_regions=[
                        r.value if hasattr(r, 'value') else str(r)
                        for r in (mt.affected_regions or [])
                    ],
                    trend_score=mt.trend_score,
                    actionability_score=mt.actionability_score,
                    oss_score=mt.oss_score,
                    article_count=mt.article_count,
                    article_snippets=mt.article_snippets or [],
                )
                for mt in major_trends
            ]

        deps._trend_data = trends

        logger.info(f"Analysis: {len(trends)} trends detected")

    except Exception as e:
        logger.error(f"Analysis node failed: {e}")
        errors.append(f"Analysis: {e}")
        trends = []
        result = None

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

    viable = getattr(impact_verdict, 'items_passed', 0) if impact_verdict else 0
    _record(deps, "quality_complete", {
        "trend_quality": getattr(trend_verdict, 'quality_score', 0.0) if trend_verdict else 0.0,
        "impact_items_passed": viable,
        "viable_count": viable,
    }, t0)

    return {
        "errors": errors,
        "current_step": "quality_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "quality_trends": getattr(trend_verdict, 'reasoning', ''),
            "quality_impacts": getattr(impact_verdict, 'reasoning', ''),
        },
    }


def quality_route(state: GraphState) -> str:
    """Route: "analysis" (retry), "lead_gen" (viable impacts), or "end"."""
    if state.get("current_step") == "quality_retry_analysis":
        return "analysis"

    settings = get_settings()
    deps = state.get("deps")
    mock_mode = getattr(deps, "mock_mode", False) if deps else False
    # In mock mode, mock LLM produces 0-confidence impacts — pass everything
    threshold = 0.0 if mock_mode else settings.min_trend_confidence_for_agents
    impacts = state.get("impacts", [])

    viable = [imp for imp in impacts if imp.council_confidence >= threshold]
    dropped = len(impacts) - len(viable)

    if dropped:
        logger.info(
            f"Quality route: {dropped}/{len(impacts)} trends below "
            f"confidence {threshold}"
        )

    if not viable:
        # Fail-open: always try lead gen with best available impacts.
        # The pipeline's VALUE is in lead generation — better to attempt it with
        # lower-confidence impacts than to produce 0 leads every run.
        viable = sorted(impacts, key=lambda i: i.council_confidence, reverse=True)[:3]
        if viable:
            logger.warning(
                f"Quality route: No impacts above {threshold}, "
                f"using top {len(viable)} as fail-open fallback "
                f"(best confidence: {viable[0].council_confidence:.2f})"
            )
        else:
            logger.warning(
                "Quality route: 0 impacts available — pipeline will attempt "
                "lead gen with empty impacts (source intel may have returned 0 articles)"
            )

    state["impacts"] = viable
    deps = state.get("deps")
    if deps:
        deps._viable_impacts = viable

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
        from .lead_gen import run_lead_gen
        companies, contacts, outreach, result = await asyncio.wait_for(
            run_lead_gen(deps), timeout=timeout
        )

        logger.info(
            f"Lead Gen: {len(companies)} companies, "
            f"{len(contacts)} contacts, {len(outreach)} outreach"
        )

    except asyncio.TimeoutError:
        logger.warning(f"Lead Gen timed out after {timeout:.0f}s — using crystallizer results only")
        errors.append(f"Lead Gen: timed out after {timeout:.0f}s")
        companies, contacts, outreach = [], [], []
        result = None

    except Exception as e:
        logger.error(f"Lead Gen node failed: {e}")
        errors.append(f"Lead Gen: {e}")
        companies, contacts, outreach = [], [], []
        result = None

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


# ── Causal Council + Lead Crystallizer Nodes ─────────────────────────────────

async def causal_council_node(state: GraphState) -> dict:
    """Causal Council — multi-hop business impact chain tracer.

    Traces: event → directly affected companies (hop1) → their buyers/suppliers (hop2)
            → downstream of hop2 (hop3). Uses real pydantic-ai tool calling.
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
        logger.warning("Causal council: no impacts — skipping")
        return {
            "errors": [],
            "current_step": "causal_council_complete",
            "agent_reasoning": {**state.get("agent_reasoning", {}), "causal_council": "no impacts"},
        }

    try:
        from app.agents.causal_council import run_causal_council
        from app.agents.signals import LearningSignal, HopSignal
        from app.tools.provider_manager import ProviderManager

        pm = ProviderManager()

        # Fix 3C: Clear expired rate-limit cooldowns before the most critical phase.
        # Earlier phases may have triggered 429 cooldowns that have since expired.
        # _is_cooling_down lazily clears these on next check, but we force it now
        # so the provider list is fresh for the causal council's tool-capable model.
        _now = _time.time()
        for pname in list(ProviderManager._failed_providers.keys()):
            # Calling _is_cooling_down triggers lazy cleanup of expired entries
            pm._is_cooling_down(pname, _now)

        # Build lookup: trend_id → TrendData (for event_type, keywords, actionability_score)
        trends = state.get("trends", [])
        trend_by_id = {td.id: td for td in trends}

        # Build BM25 index from articles fetched this run
        articles = getattr(deps, "_articles", [])
        if articles and deps.search_manager:
            from app.search.bm25_search import BM25Search
            bm25 = BM25Search(articles)
            deps.search_manager.set_bm25_index(bm25)
            logger.info(f"BM25: indexed {len(articles)} articles for causal council")

        settings = get_settings()
        _max_impacts = getattr(settings, 'per_trend_max_impacts', 5)
        geo_label = getattr(settings, "country", "India")

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
            # kb_hit_rate = fraction of hops where the KB returned real company names —
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
                f"  '{trend_title[:50]}' → {len(chain.hops)} hops "
                f"(kb_hit_rate={sig.kb_hit_rate:.0%}, oss={oss_score:.2f})"
            )

        deps._causal_results = causal_results
        total_hops = sum(len(r.hops) for r in causal_results)

        # Fix 4B: If ALL causal results have 0 hops, build synthetic hops from
        # impact analysis data. This ensures lead_crystallize_node always has
        # something to work with.
        if total_hops == 0 and impacts:
            logger.warning("Causal council: 0 hops from all trends — building synthetic hops from impact data")
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
            logger.info(f"Causal council: after synthetic fallback → {total_hops} total hops")

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
    """Resolve segment descriptions to real company names via SearXNG + LLM.

    For each causal hop that has an empty companies_found list:
    1. Search SearXNG for real companies matching the segment
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

    # Step 1: Search SearXNG for each segment concurrently
    search_results = {}  # idx -> list of search results
    sem = asyncio.Semaphore(5)

    async def _search_segment(idx: int, segment: str):
        async with sem:
            try:
                query = f"{segment} company India 2026"
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

    prompt = f"""For each business segment below, extract 2-3 REAL, specific Indian companies from the search results.
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

        # Parse result — could be a list or {"results": [...]}
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
    """Lead Crystallizer — converts causal chains into concrete call sheets.

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
        logger.warning("Lead crystallizer: no causal results — skipping")
        return {
            "errors": [],
            "current_step": "lead_crystallize_complete",
            "agent_reasoning": {**state.get("agent_reasoning", {}), "lead_crystallizer": "no causal results"},
        }

    # ── Resolve segment descriptions to real company names ──
    try:
        await _resolve_companies_for_hops(deps, causal_results)
    except Exception as e:
        logger.warning(f"Company resolution step failed (non-fatal): {e}")

    # ── Fetch company-specific news for resolved companies ──
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
        from app.agents.lead_crystallizer import crystallize_leads

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
            logger.info(f"    → {sheet.opening_line[:100]}")

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
            "lead_crystallizer": f"{len(all_leads)} call sheets — "
                                 f"{sum(1 for l in all_leads if l.company_cin)} with CIN from KB",
        },
    }


# ── Learning Update Node ─────────────────────────────────────────────────────

async def learning_update_node(state: GraphState) -> dict:
    """Post-run autonomous learning — runs unconditionally at graph end.

    Consumes deps._signals (LearningSignal list) to update:
    1. Source bandit: which RSS sources produced high-OSS clusters
    2. Trend memory: update OSS scores for stored centroids
    3. Weight learner: persist OSS auto-learning updates
    4. Pipeline metrics: log run summary for cross-run comparison
    5. Learning signals JSONL: append per-run record for audit trail
    """
    t0 = _time.time()
    logger.info("=" * 50)
    logger.info("STEP 5: AUTONOMOUS LEARNING UPDATE")
    logger.info("=" * 50)

    deps = state["deps"]
    signals = getattr(deps, "_signals", [])
    articles = getattr(deps, "_articles", [])
    pipeline = getattr(deps, "_pipeline", None)

    # ── 1. Source Bandit Update ──────────────────────────────────────────
    try:
        if articles and signals:
            from app.learning.source_bandit import SourceBandit

            bandit = deps.source_bandit

            # Group article IDs by source_id (not objects — they aren't hashable)
            source_articles: dict = {}
            for art in articles:
                src_id = getattr(art, "source_id", None) or getattr(art, "source", "unknown")
                if src_id not in source_articles:
                    source_articles[src_id] = []
                art_id = str(getattr(art, "id", id(art)))
                source_articles[src_id].append(art_id)

            # Build article → cluster label map (from pipeline if available)
            article_labels: dict = {}
            cluster_quality: dict = {}
            cluster_oss: dict = {}

            if pipeline and hasattr(pipeline, "_article_cluster_map"):
                article_labels = pipeline._article_cluster_map or {}
                cluster_quality = pipeline._cluster_quality or {}

            # Build cluster_oss from LearningSignals using actual cluster IDs
            # (not enumerate index — that's wrong when clusters have non-sequential IDs)
            if pipeline and hasattr(pipeline, "_article_cluster_map"):
                # Build trend_title → cluster_id reverse map
                _title_to_cluster: dict = {}
                for _art_id, _clust_id in article_labels.items():
                    # cluster_summaries keyed by cluster_id have trend_title
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
                        # Fallback: use hash of title as approximate key
                        cluster_oss[hash(sig.trend_title) % 10000] = sig.oss_score
            else:
                # No pipeline data — fall back to positional (better than nothing)
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
    except Exception as e:
        logger.warning(f"Source bandit update failed: {e}")

    # ── 2. Trend Memory OSS Update ───────────────────────────────────────
    try:
        if pipeline and hasattr(pipeline, "_trend_memory") and signals:
            tm = pipeline._trend_memory
            if tm and hasattr(pipeline, "_cluster_centroids"):
                centroids = pipeline._cluster_centroids or {}
                oss_map = {idx: sig.oss_score for idx, sig in enumerate(signals) if sig.oss_score > 0}
                if centroids and oss_map:
                    updated_count = tm.update_oss_scores(centroids, oss_map)
                    logger.info(f"Trend memory: OSS updated for {updated_count} centroids")
            # Prune stale centroids that haven't been seen in N runs
            try:
                pruned = tm.prune_stale()
                if pruned:
                    logger.info(f"Trend memory: pruned {pruned} stale centroids")
            except Exception as e:
                logger.debug(f"Trend memory prune skipped: {e}")
    except Exception as e:
        logger.warning(f"Trend memory OSS update failed: {e}")

    # ── 3. Weight Learner: Compute + persist all 4 weight types ──────────
    try:
        if signals:
            from app.learning.weight_learner import (
                compute_learned_weights, _save_persisted_weights,
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

            all_learned = {}
            for wt_name, defaults in weight_types.items():
                all_learned[wt_name] = compute_learned_weights(wt_name, defaults)

            _save_persisted_weights(all_learned)

            # Invalidate composite.py weight cache so next scoring uses fresh weights
            from app.trends.signals.composite import invalidate_weights_cache
            invalidate_weights_cache()

            mean_oss = sum(s.oss_score for s in signals) / len(signals)
            logger.info(
                f"Weight learner: updated all 4 weight types | "
                f"{len(signals)} signals | mean OSS={mean_oss:.3f}"
            )
    except Exception as e:
        logger.warning(f"Weight learner update failed: {e}")

    # ── 4. Persist learning signals to JSONL ─────────────────────────────
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

    # ── 5. Company Bandit: reward update from lead sheets (Loop 4) ───────
    try:
        lead_sheets = getattr(deps, "_lead_sheets", [])
        if lead_sheets:
            company_bandit = deps.company_bandit
            updated_arms = set()
            for lead in lead_sheets:
                arm_id = f"{lead.company_size_band or 'mid'}_{lead.event_type or 'general'}"
                # Reward: 1.0 for high-confidence KB-verified lead, 0.5 for moderate, 0.0 for placeholder
                if lead.company_cin:  # Has real CIN from KB
                    reward = min(1.0, lead.confidence + 0.2)
                elif not lead.company_name.startswith("["):
                    reward = lead.confidence * 0.7
                else:
                    reward = 0.1  # Placeholder segment — minimal reward
                company_bandit.update(arm_id, reward)
                updated_arms.add(arm_id)
            logger.info(f"Company bandit: updated {len(updated_arms)} arms from {len(lead_sheets)} leads")
    except Exception as e:
        logger.debug(f"Company bandit update skipped: {e}")

    # ── 6. Quality Feedback: auto-record for weight learner (Loop 5) ──
    try:
        if signals:
            from app.tools.feedback import save_feedback
            feedback_count = 0
            for sig in signals:
                # Composite quality: KB hits (external) + hops + OSS (text quality)
                # KB hit rate is NON-CIRCULAR — it checks against an external company database
                quality = (
                    0.40 * sig.kb_hit_rate +
                    0.30 * min(1.0, sig.leads_with_companies / max(sig.leads_generated, 1)) +
                    0.30 * sig.oss_score
                )
                if quality >= 0.45:
                    rating = "good_trend"
                elif quality >= 0.25:
                    rating = "already_knew"
                else:
                    rating = "bad_trend"
                save_feedback(
                    feedback_type="trend",
                    item_id=sig.trend_title[:100],
                    rating=rating,
                    metadata={
                        "auto": True,  # CRITICAL: weight_learner filters on this key
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
    except Exception as e:
        logger.debug(f"Quality feedback recording skipped: {e}")

    # ── 7. Summary ───────────────────────────────────────────────────────
    avg_oss = sum(s.oss_score for s in signals) / len(signals) if signals else 0.0
    avg_kb_hit = sum(s.kb_hit_rate for s in signals) / len(signals) if signals else 0.0
    logger.info(
        f"Learning update complete: {len(signals)} trends | "
        f"avg_oss={avg_oss:.3f} | avg_kb_hit={avg_kb_hit:.1%}"
    )

    _record(deps, "learning_update_complete", {
        "signal_count": len(signals),
        "avg_oss": round(avg_oss, 3),
        "avg_kb_hit": round(avg_kb_hit, 3),
        "signals": [s.to_dict() for s in signals] if signals else [],
    }, t0)

    # Save recording manifest (finalizes the recording for replay)
    recorder = getattr(deps, "recorder", None)
    if recorder:
        total_duration = _time.time() - getattr(deps, "_pipeline_t0", _time.time())
        recorder.save_manifest(total_duration)

    return {
        "errors": [],
        "current_step": "learning_update_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "learning_update": (
                f"{len(signals)} signals processed | "
                f"avg_oss={avg_oss:.3f} | avg_kb_hit={avg_kb_hit:.1%}"
            ),
        },
    }


# ── Graph Construction ───────────────────────────────────────────────────────

def create_pipeline_graph():
    """Build and compile the multi-agent LangGraph with quality gate + checkpointing.

    Flow:
      source_intel → analysis → impact → quality_validation
        ├─ (retry) → analysis
        ├─ (viable) → causal_council → lead_crystallize → lead_gen → END
        └─ (no trends) → END

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
    workflow.add_edge("source_intel", "analysis")
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


# ── Public Entry Point ───────────────────────────────────────────────────────

async def run_pipeline(mock_mode: bool = False, log_callback=None) -> PipelineResult:
    """Execute the multi-agent sales intelligence pipeline.

    Uses LangGraph astream (stream_mode="values") so each snapshot after a node
    is the complete accumulated state — Streamlit sees progress in real-time
    without needing to merge partial diffs.
    InMemorySaver checkpointer snapshots state after every node for resilience.
    """
    start_time = datetime.utcnow()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")

    logger.info("Starting Multi-Agent Sales Intelligence Pipeline")
    logger.info(f"Run ID: {run_id} | Mock Mode: {mock_mode}")

    # Reset provider health and cooldowns so stale failures from previous runs
    # don't block this run's LLM calls (Fix 3A).
    from ..tools.provider_health import provider_health
    from ..tools.provider_manager import ProviderManager
    provider_health.reset_for_new_run()
    ProviderManager.reset_cooldowns()
    logger.info("Provider health + cooldowns reset for new run")

    from .deps import AgentDeps
    deps = AgentDeps.create(
        mock_mode=mock_mode,
        log_callback=log_callback,
        run_id=run_id,
    )

    deps._pipeline_t0 = _time.time()

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

    try:
        graph = create_pipeline_graph()

        # stream_mode="values" emits the FULL accumulated state after each node,
        # giving Streamlit real-time progress without needing to merge partial diffs.
        final_state: dict = {}
        async for snapshot in graph.astream(initial_state, config, stream_mode="values"):
            final_state = snapshot
            logger.debug(f"Pipeline step: {snapshot.get('current_step', '?')}")

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

        return PipelineResult(
            status="success",
            leads_generated=outreach_count,
            trends_detected=trends_count,
            companies_found=companies_count,
            emails_found=emails_count,
            output_file=output_file,
            errors=final_state.get("errors", []),
            run_time_seconds=runtime,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return PipelineResult(
            status="error",
            errors=[str(e)],
            run_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
        )


# ── Output Persistence ───────────────────────────────────────────────────────

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

        lead = {
            "id": outreach.id,
            "trend": {
                "title": trend.trend_title if trend else outreach.trend_title,
                "summary": trend.summary if trend else "",
                "severity": trend.severity if trend else "medium",
                "industries": trend.industries_affected if trend else [],
            },
            "company": {
                "name": company.company_name if company else outreach.company_name,
                "size": company.company_size if company else "mid",
                "industry": company.industry if company else "",
                "website": company.website if company else "",
                "domain": company.domain if company else "",
            },
            "contact": {
                "name": contact.person_name if contact else outreach.person_name,
                "role": contact.role if contact else outreach.role,
                "email": contact.email if contact else outreach.email,
                "email_confidence": contact.email_confidence if contact else 0,
                "email_source": contact.email_source if contact else "",
                "linkedin": contact.linkedin_url if contact else "",
            },
            "outreach": {
                "subject": outreach.subject,
                "body": outreach.body,
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
    # These are the primary deliverable of the new pipeline — one file per run,
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

    # ── Comprehensive run report (cluster validation + first/second-order view) ──
    # Joins article titles from deps._articles into each trend card so you can
    # verify that clusters contain real, coherent news — not garbage articles.
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
                # Article titles joined from TrendTree — validate cluster quality here
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

    # ── Persist to DB ──────────────────────────────────────────────────────
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

        for sheet in lead_sheets:
            cn = getattr(sheet, "company_name", "")
            co = _co_by_name.get(cn)
            cts = _ct_by_co.get(cn, [])
            ct = next((c for c in cts if getattr(c, "email", "")), cts[0] if cts else None)
            em = _em_by_co.get(cn)
            enrichment = {
                "company_website": getattr(co, "website", "") if co else "",
                "company_domain": getattr(co, "domain", "") if co else "",
                "reason_relevant": getattr(co, "reason_relevant", "") if co else "",
                "contact_name": getattr(ct, "person_name", "") if ct else "",
                "contact_role": getattr(ct, "role", "") if ct else "",
                "contact_email": getattr(ct, "email", "") if ct else "",
                "contact_linkedin": getattr(ct, "linkedin_url", "") if ct else "",
                "email_confidence": getattr(ct, "email_confidence", 0) if ct else 0,
                "email_subject": getattr(em, "subject", "") if em else "",
                "email_body": getattr(em, "body", "") if em else "",
                "company_news": getattr(sheet, "company_news", []),
            }
            db.save_call_sheet(run_id, sheet, enrichment=enrichment)

        # Save legacy outreach leads
        for lead in leads:
            trend_id = lead.get("trend", {}).get("title", "")
            db.save_lead({
                "id": lead["id"],
                "trend": lead["trend"],
                "impact": impact_map.get(trend_id, {}),
                "company": lead["company"],
                "contact": lead["contact"],
                "outreach": lead["outreach"],
            })

        # Save person profiles (multiple contacts per company with reach scores)
        person_profiles = getattr(deps, "_person_profiles", []) if deps else []
        if person_profiles:
            saved_count = db.save_lead_contacts(run_id, person_profiles)
            logger.info(f"DB: saved {saved_count} person profiles")

        logger.info(
            f"DB: saved run + {len(trends)} trends + "
            f"{len(lead_sheets)} call sheets + {len(leads)} outreach leads"
        )
    except Exception as e:
        logger.warning(f"Failed to save to database: {e}")

    return str(json_file)
