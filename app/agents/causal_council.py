"""
Causal Council — multi-hop business impact chain tracer.

Takes a synthesized trend cluster and traces 3 hops:
  Hop 1: Companies DIRECTLY mentioned or immediately affected (first-order)
  Hop 2: Companies that BUY FROM or SELL TO hop-1 companies (second-order)
  Hop 3: Companies affected by hop-2 impacts (third-order)

This is a true pydantic-ai ReAct agent — the LLM autonomously decides which
tools to call and in what order. Primary: Groq llama-3.3-70b-versatile.
Offline fallback: Ollama llama3.2:3b (the only local model with tool calling).

Chain-of-thought: `reasoning` field captures the LLM's step-by-step analysis
BEFORE it produces the structured output — not post-hoc rationalization.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel
from pydantic_ai import RunContext  # noqa: F401 — needed for tool annotation resolution

logger = logging.getLogger(__name__)


# ── Output schema ────────────────────────────────────────────────────────────

class CausalHop(BaseModel):
    hop: int                     # 1, 2, or 3
    segment: str                 # "Tier-2 auto parts suppliers (50-200 employees)"
    lead_type: Literal["pain", "opportunity", "risk", "intelligence"]
    mechanism: str               # "Steel duty → input cost increase of ~15%"
    urgency_weeks: int           # How quickly does this materialize?
    geo_hint: str                # "Pune, Chennai, Rajkot"
    employee_band: str           # "sme" (50-500) | "mid" (500-5000)
    confidence: float            # 0.0–1.0
    companies_found: list[str] = []  # Real company names from KB + BM25


class CausalChainResult(BaseModel):
    event_summary: str
    event_type: str
    hops: list[CausalHop]
    reasoning: str   # Chain-of-thought: the LLM's step-by-step analysis


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a supply chain economist and sales intelligence analyst.

Think step by step before producing output — write your analysis in the `reasoning` field first.

Given a news trend/event, trace its business impact through 3 hops:

HOP 1 (first-order): Companies DIRECTLY mentioned or facing immediate impact.
  They are in the news itself. They need intelligence to navigate NOW.
  Lead type: opportunity if they benefit, pain if they face cost/risk.

HOP 2 (second-order): Companies that BUY FROM or SELL TO hop-1 companies.
  If steel mills face tariff (hop 1) → Tier-2 auto parts suppliers buy steel → pain.
  If gold price drops (hop 1) → gold jewellery exporters use gold → margin pain.
  Urgency: typically 2-6 weeks for cost pass-through.

HOP 3 (third-order): Companies affected by hop-2 impacts.
  If auto parts suppliers face cost squeeze (hop 2) → OEM auto companies face delays.
  Urgency: typically 4-12 weeks (delayed effects).

For each hop, you MUST:
1. Be SPECIFIC: not "technology companies" but "Tier-2 brake component suppliers, 50-200 employees, Pune"
2. Use search_articles and web_search_companies to find REAL companies in that segment
3. Estimate urgency in weeks
4. Classify as pain / opportunity / risk / intelligence

EXCLUDE:
- Fortune 500 / large enterprises (Tata, Reliance, Infosys, Wipro, HCL, HDFC, Bajaj, Mahindra)
- Hops with confidence < 0.4
- Generic "all companies" descriptions

FOCUS ON: SME and mid-size companies (50-5000 employees) in India."""


# ── Main entry point ─────────────────────────────────────────────────────────

async def run_causal_council(
    trend_title: str,
    trend_summary: str,
    event_type: str,
    keywords: list[str],
    geo: str = "India",
    provider_manager=None,
    search_manager=None,
    max_hops: int = 3,
) -> CausalChainResult:
    """
    Trace the causal chain from a news trend to affected companies.

    Returns CausalChainResult — always (never None). On failure returns empty hops
    with reasoning explaining what went wrong.
    """
    from pydantic_ai import Agent, RunContext

    @dataclass
    class _Deps:
        sm: Any

    deps = _Deps(sm=search_manager)

    agent: Agent[_Deps, CausalChainResult] = Agent(
        model=provider_manager.get_tool_capable_model() if provider_manager else "groq:llama-3.3-70b-versatile",
        output_type=CausalChainResult,
        system_prompt=_SYSTEM_PROMPT,
        deps_type=_Deps,
    )

    @agent.tool
    async def search_articles(ctx: RunContext, query: str) -> str:
        """
        Search already-fetched news articles for this run. Instant, free, offline.
        Use to find companies mentioned in context before web searching.
        Returns: Article titles mentioning relevant companies.
        """
        import json
        if ctx.deps.sm:
            hits = ctx.deps.sm.search_articles(query, top_k=10)
            if hits:
                titles = [h.get("title", "")[:80] for h in hits[:5]]
                return f"Found {len(hits)} relevant articles: {json.dumps(titles)}"
        return "No article search index available for this query"

    @agent.tool
    async def web_search_companies(ctx: RunContext, query: str) -> str:
        """
        Web search for companies in a segment — use only if KB has < 3 results.
        Returns: Top 3 search result summaries.
        """
        import json
        if ctx.deps.sm:
            result = await ctx.deps.sm.web_search(f"{query} companies India SME", max_results=5)
            if result["results"]:
                snippets = [f"{r['title']}: {r['content'][:100]}" for r in result["results"][:3]]
                return json.dumps(snippets)
        return "Web search unavailable"

    prompt = f"""Analyze this business trend and trace its causal impact chain:

TREND: {trend_title}
EVENT TYPE: {event_type}
GEOGRAPHY: {geo}
KEY ENTITIES: {", ".join(keywords[:10])}

FULL SUMMARY:
{trend_summary[:800]}

Instructions:
1. Think step by step — write your analysis in `reasoning` first.
2. Identify Hop 1 (directly affected companies — who is in the news?)
3. Trace Hop 2 (who buys from or sells to Hop 1?)
{"4. Trace Hop 3 (who is affected by Hop 2?)" if max_hops >= 3 else ""}

For each hop: use search_articles first, then web_search_companies for additional companies.
Be specific. No large enterprises. Confidence 0-1 based on directness of causal link.

Produce a complete CausalChainResult with {max_hops} hops and full reasoning."""

    # ── Attempt 1: pydantic-ai agent with tool calling ──
    chain = await _try_agent_run(agent, prompt, deps, trend_title)
    if chain and chain.hops:
        return chain

    # ── Attempt 2: Retry after 5s backoff (transient 429s often clear) ──
    logger.info(f"Causal council: retrying '{trend_title[:40]}' after 5s backoff...")
    await asyncio.sleep(5)
    chain = await _try_agent_run(agent, prompt, deps, trend_title)
    if chain and chain.hops:
        return chain

    # ── Attempt 3: Direct LLM structured output (bypasses tool calling) ──
    chain = await _try_llm_structured_fallback(
        trend_title, trend_summary, event_type, keywords, geo,
    )
    if chain and chain.hops:
        return chain

    # ── Attempt 4: Build synthetic hops from templates (no LLM needed) ──
    chain = _build_synthetic_hops(
        trend_title, trend_summary, event_type, keywords,
    )
    return chain


# ── Helper functions for resilient causal council ─────────────────────────────

async def _try_agent_run(agent, prompt, deps, trend_title) -> Optional[CausalChainResult]:
    """Run pydantic-ai agent, returning None on failure."""
    try:
        from app.tools.provider_manager import ProviderManager
        await ProviderManager.acquire_gcp_rate_limit()
        result = await agent.run(prompt, deps=deps)
        chain = result.output
        logger.info(f"Causal council: '{trend_title[:50]}' → {len(chain.hops)} hops")
        return chain
    except Exception as e:
        logger.warning(f"Causal council agent failed for '{trend_title[:40]}': {type(e).__name__}: {e}")
        return None


async def _try_llm_structured_fallback(
    trend_title: str,
    trend_summary: str,
    event_type: str,
    keywords: list[str],
    geo: str,
) -> Optional[CausalChainResult]:
    """Fallback: use LLMService structured output instead of pydantic-ai agent.

    Bypasses tool calling entirely — just asks the LLM to produce the
    CausalChainResult JSON directly. Works when providers support structured
    output but not function calling (e.g., NVIDIA DeepSeek).
    """
    try:
        from app.tools.llm_service import LLMService
        llm = LLMService()

        prompt = f"""Analyze this business trend and trace its causal impact chain.

TREND: {trend_title}
EVENT TYPE: {event_type}
GEOGRAPHY: {geo}
KEY ENTITIES: {", ".join(keywords[:10])}
SUMMARY: {trend_summary[:600]}

For each hop, identify specific company SEGMENTS (not individual companies).
Focus on mid-size Indian companies (50-5000 employees).

Return JSON with:
- event_summary: the trend title
- event_type: "{event_type}"
- reasoning: your step-by-step causal chain analysis
- hops: array of 2-3 hops, each with:
  - hop: 1/2/3
  - segment: specific company segment (e.g. "Tier-2 auto parts suppliers, 50-200 employees, Pune")
  - lead_type: "pain" or "opportunity" or "risk" or "intelligence"
  - mechanism: how this trend affects them (e.g. "Steel duty → input cost increase of ~15%")
  - urgency_weeks: estimated weeks until impact materializes
  - geo_hint: specific Indian cities/states
  - employee_band: "sme" (50-500) or "mid" (500-5000)
  - confidence: 0.0-1.0 based on directness of causal link
  - companies_found: [] (leave empty)"""

        raw = await llm.generate_json(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
        )
        if not isinstance(raw, dict) or "error" in raw:
            logger.warning(f"Structured fallback returned error: {raw}")
            return None

        chain = CausalChainResult(**{
            "event_summary": raw.get("event_summary", trend_title),
            "event_type": raw.get("event_type", event_type),
            "reasoning": raw.get("reasoning", "LLM structured fallback"),
            "hops": [CausalHop(**h) for h in raw.get("hops", []) if isinstance(h, dict)],
        })

        logger.info(f"Causal council (structured fallback): '{trend_title[:50]}' → {len(chain.hops)} hops")
        return chain

    except Exception as e:
        logger.warning(f"Causal council structured fallback failed for '{trend_title[:40]}': {type(e).__name__}: {e}")
        return None


def _build_synthetic_hops(
    trend_title: str,
    trend_summary: str,
    event_type: str,
    keywords: list[str],
) -> CausalChainResult:
    """Last-resort: build template hops without any LLM calls.

    Constructs hop-1 and hop-2 entries from event type and keywords.
    Lower quality but guarantees non-zero hops when all LLM providers
    are exhausted. Companies will be found later by the company agent
    via web search.
    """
    # Event type → default mechanism templates
    mechanism_templates = {
        "funding": "New funding round creates demand for market intelligence and growth strategy",
        "acquisition": "M&A activity creates demand for due diligence and market sizing",
        "regulation": "New regulation creates compliance burden and need for advisory",
        "crisis": "Crisis event creates urgent need for risk assessment and crisis response",
        "technology": "Technology shift creates need for competitive analysis and strategy pivot",
        "market": "Market shift creates need for repositioning and competitive intelligence",
        "policy": "Policy change creates need for impact assessment and compliance planning",
        "ipo": "IPO activity creates demand for market positioning and valuation analysis",
    }
    base_mechanism = mechanism_templates.get(event_type, "Market event creates demand for consulting intelligence")
    sector_label = ", ".join(keywords[:3]) if keywords else "general"

    hops = [
        CausalHop(
            hop=1,
            segment=f"Companies in {sector_label} sector",
            lead_type="pain" if event_type in ("crisis", "regulation") else "opportunity",
            mechanism=base_mechanism,
            urgency_weeks=2 if event_type in ("crisis", "regulation") else 4,
            geo_hint="India",
            employee_band="sme",
            confidence=0.45,
        ),
        CausalHop(
            hop=2,
            segment=f"Suppliers and service providers to {sector_label} companies",
            lead_type="intelligence",
            mechanism=f"Downstream effect: {base_mechanism.lower()}",
            urgency_weeks=6,
            geo_hint="India",
            employee_band="sme",
            confidence=0.40,
        ),
    ]

    logger.info(
        f"Causal council (synthetic fallback): '{trend_title[:50]}' → {len(hops)} template hops"
    )

    return CausalChainResult(
        event_summary=trend_title,
        event_type=event_type,
        hops=hops,
        reasoning=f"Synthetic template hops (all LLM providers exhausted). Keywords: {', '.join(keywords[:5])}",
    )
