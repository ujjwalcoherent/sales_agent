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

import logging
from dataclasses import dataclass
from typing import Any, Literal

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
2. Call search_company_kb to find REAL companies in that segment
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
    company_kb=None,
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
        kb: Any
        sm: Any

    deps = _Deps(kb=company_kb, sm=search_manager)

    agent: Agent[_Deps, CausalChainResult] = Agent(
        model=provider_manager.get_tool_capable_model() if provider_manager else "groq:llama-3.3-70b-versatile",
        output_type=CausalChainResult,
        system_prompt=_SYSTEM_PROMPT,
        deps_type=_Deps,
    )

    @agent.tool
    async def search_company_kb(ctx: RunContext, segment: str, state: str = "") -> str:
        """
        Search the local India company database (1.8M+ companies, SQLite FTS5).
        Call this for EVERY hop to find real company names. Always try before web search.
        Returns: JSON list of up to 5 matching companies with name and state.
        """
        import json
        if ctx.deps.kb and ctx.deps.kb.is_loaded:
            results = ctx.deps.kb.search(segment, state=state or None, limit=10)
            if results:
                names = [f"{r['name']} ({r['state']})" for r in results[:5]]
                return f"Found {len(results)} companies: {json.dumps(names)}"
        return "Company KB empty or not loaded — try web_search_companies instead"

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

For each hop: use search_company_kb, then search_articles, then web_search_companies as fallback.
Be specific. No large enterprises. Confidence 0-1 based on directness of causal link.

Produce a complete CausalChainResult with {max_hops} hops and full reasoning."""

    try:
        result = await agent.run(prompt, deps=deps)
        chain = result.output
        # Post-process: fill companies_found from KB if agent missed it
        for hop in chain.hops:
            if not hop.companies_found and company_kb and company_kb.is_loaded:
                kb_results = company_kb.search(hop.segment, limit=5)
                hop.companies_found = [r["name"] for r in kb_results]
        logger.info(f"Causal council: '{trend_title[:50]}' → {len(chain.hops)} hops")
        return chain
    except Exception as e:
        logger.error(f"Causal council failed for '{trend_title[:40]}': {e}")
        return CausalChainResult(
            event_summary=trend_title,
            event_type=event_type,
            hops=[],
            reasoning=f"Agent failed: {e}",
        )
