"""
LangGraph Orchestrator — multi-agent supervisor for sales intelligence pipeline.

Flow: source_intel -> analysis -> impact -> quality_validation -> lead_gen -> END
Quality gate can retry analysis (max 2x) or skip lead gen if no viable trends.
"""

import logging
import json
import csv
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


# ── Agent Nodes ──────────────────────────────────────────────────────────────

async def source_intel_node(state: GraphState) -> dict:
    """Source Intel Agent -- article collection with bandit-prioritized sources."""
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
        logger.warning("Quality route: No viable trends — skipping lead gen")
        return "end"

    state["impacts"] = viable
    deps = state.get("deps")
    if deps:
        deps._viable_impacts = viable

    return "lead_gen"


async def lead_gen_node(state: GraphState) -> dict:
    """Lead Gen Agent -- company discovery, contact finding, outreach emails."""
    logger.info("=" * 50)
    logger.info("STEP 4: LEAD GEN AGENT")
    logger.info("=" * 50)

    deps = state["deps"]
    errors = []

    try:
        from .lead_gen import run_lead_gen
        companies, contacts, outreach, result = await run_lead_gen(deps)

        logger.info(
            f"Lead Gen: {len(companies)} companies, "
            f"{len(contacts)} contacts, {len(outreach)} outreach"
        )

    except Exception as e:
        logger.error(f"Lead Gen node failed: {e}")
        errors.append(f"Lead Gen: {e}")
        companies, contacts, outreach = [], [], []
        result = None

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

        for impact in impacts[:5]:   # Top 5 trends only (cost control)
            # ImpactAnalysis carries trend_title and detailed_reasoning.
            # We join direct_impact sentences as fallback when reasoning is empty.
            trend_title = impact.trend_title or ""
            trend_summary = (
                impact.detailed_reasoning
                or " ".join(impact.direct_impact[:3])
                or impact.pitch_angle
                or ""
            )

            # Join TrendData via trend_id to get the structured metadata
            # (event_type, keywords, OSS-based actionability) that ImpactAnalysis
            # doesn't carry on its own.
            td = trend_by_id.get(impact.trend_id)
            event_type = (td.trend_type if td else "") or "general"
            keywords = (td.keywords if td else []) or list(impact.midsize_pain_points[:5])

            # actionability_score from synthesis stage is the closest proxy for OSS —
            # it captures entity density, geo specificity, and numeric richness.
            oss_score = (td.actionability_score if td else 0.0) or 0.0

            settings = get_settings()
            geo_label = getattr(settings, "country", "India")
            chain = await run_causal_council(
                trend_title=trend_title,
                trend_summary=trend_summary,
                event_type=event_type,
                keywords=keywords,
                geo=geo_label,
                provider_manager=pm,
                company_kb=deps.company_kb,
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
        logger.info(f"Causal council: {len(causal_results)} chains, {total_hops} total hops")

    except Exception as e:
        logger.error(f"Causal council node failed: {e}", exc_info=True)
        errors.append(f"CausalCouncil: {e}")

    return {
        "errors": errors,
        "current_step": "causal_council_complete",
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "causal_council": f"{len(causal_results)} chains traced, "
                              f"{sum(len(r.hops) for r in causal_results)} total hops",
        },
    }


async def lead_crystallize_node(state: GraphState) -> dict:
    """Lead Crystallizer — converts causal chains into concrete call sheets.

    Each LeadSheet has: company name (real from KB), contact role, trigger event,
    pain point, service pitch, and opening line ready to use on a call.
    """
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
            oss_score = (td.actionability_score if td else 0.0) or 0.0

            leads = await crystallize_leads(
                causal_result=chain,
                trend_title=chain.event_summary,
                trend_summary=trend_summary,
                event_type=chain.event_type,
                company_kb=deps.company_kb,
                oss_score=oss_score,
            )
            all_leads.extend(leads)

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

            # Group articles by source_id
            source_articles: dict = {}
            for art in articles:
                src_id = getattr(art, "source_id", None) or getattr(art, "source", "unknown")
                if src_id not in source_articles:
                    source_articles[src_id] = []
                source_articles[src_id].append(art)

            # Build article → cluster label map (from pipeline if available)
            article_labels: dict = {}
            cluster_quality: dict = {}
            cluster_oss: dict = {}

            if pipeline and hasattr(pipeline, "_article_cluster_map"):
                article_labels = pipeline._article_cluster_map or {}
                cluster_quality = pipeline._cluster_quality or {}

            # Build cluster_oss from LearningSignals (trend OSS by trend_title)
            # Map: cluster_index → oss_score (approximate via article sources)
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
    except Exception as e:
        logger.warning(f"Trend memory OSS update failed: {e}")

    # ── 3. Weight Learner: Persist OSS auto-learning ─────────────────────
    try:
        if signals:
            from app.learning.weight_learner import _save_persisted_weights, _load_persisted_weights
            # Log OSS values for this run (weight learner reads from quality_report_log.jsonl)
            # The weight_learner picks up from quality_report_log on next call
            logger.info(f"Weight learner: {len(signals)} OSS signals available | "
                        f"mean OSS={sum(s.oss_score for s in signals)/len(signals):.3f}")
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

    # ── 5. Summary ───────────────────────────────────────────────────────
    avg_oss = sum(s.oss_score for s in signals) / len(signals) if signals else 0.0
    avg_kb_hit = sum(s.kb_hit_rate for s in signals) / len(signals) if signals else 0.0
    logger.info(
        f"Learning update complete: {len(signals)} trends | "
        f"avg_oss={avg_oss:.3f} | avg_kb_hit={avg_kb_hit:.1%}"
    )

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

    from .deps import AgentDeps
    deps = AgentDeps.create(mock_mode=mock_mode, log_callback=log_callback)

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

    try:
        db = get_database()
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
    except Exception as e:
        logger.warning(f"Failed to save leads to database: {e}")

    return str(json_file)
