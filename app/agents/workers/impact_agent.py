"""
Impact Mapping Agent - Consultant Edition.
Thinks like a consultant to identify which companies need Coherent Market Insights services.

V7: Parallel impact analysis with asyncio.gather + Semaphore.
I1: Cross-trend impact synthesis — compound opportunity detection.
"""

import asyncio
import logging
from typing import List, Optional

from ...schemas import TrendData, ImpactAnalysis, AgentState
from ...tools.llm_service import LLMService
from ...config import get_settings, TREND_ROLE_MAPPING, CMI_SERVICES
from .council.impact_council import run_impact_council

logger = logging.getLogger(__name__)


class ImpactAnalyzer:
    """
    Analyzes each trend's business impact for CMI consulting.

    Per trend, determines:
    - Which industries are impacted by THIS specific trend and HOW
    - Which of CMI's 9 services apply to THIS trend's affected companies
    - What specific consulting projects companies would pay for
    - Who are the decision makers to pitch

    NOTE: This is a deterministic pipeline stage, not an autonomous agent.
    Renamed from ImpactAgent for honest naming (see ICML 2024 research).
    """

    def __init__(self, mock_mode: bool = False, deps=None, log_callback=None):
        """Initialize impact analyzer with Groq-preferred provider chain."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self._log_callback = log_callback
        if deps:
            self.llm_service = deps.llm_service
        else:
            self.llm_service = LLMService(mock_mode=self.mock_mode, force_groq=True)
        provider = self.llm_service.settings.get_llm_config().get("provider", "unknown")
        logger.info(f"Impact Agent initialized (provider={provider})")

    def _log(self, msg: str, level: str = "info"):
        """Log to both Python logger and optional UI callback."""
        logger.info(msg) if level == "info" else logger.warning(msg)
        if self._log_callback:
            try:
                self._log_callback(msg, level)
            except Exception:
                pass
    
    async def analyze_impacts(self, state: AgentState) -> AgentState:
        """
        Analyze impact of all detected trends.

        V7: Uses asyncio.gather + Semaphore for parallel analysis.
        I1: Runs cross-trend synthesis after individual analyses.

        Args:
            state: Current agent state with trends

        Returns:
            Updated state with impact analyses
        """
        self._log("Starting impact analysis...")

        if not state.trends:
            self._log("No trends to analyze", "warning")
            return state

        self._log(f"Analyzing {len(state.trends)} trends through AI council...")

        # V7: Parallel impact analysis with concurrency limit
        semaphore = asyncio.Semaphore(4)
        completed = 0

        async def _analyze_one(trend: TrendData) -> ImpactAnalysis:
            nonlocal completed
            self._log(f"Council analyzing: {trend.trend_title[:60]}...")
            async with semaphore:
                try:
                    impact = await self._analyze_trend_impact(trend)
                    completed += 1
                    confidence = f"{impact.council_confidence:.0%}" if impact.council_confidence else "N/A"
                    self._log(f"[{completed}/{len(state.trends)}] Done: {trend.trend_title[:40]}... (confidence={confidence})")
                    return impact
                except Exception as e:
                    completed += 1
                    self._log(f"[{completed}/{len(state.trends)}] Fallback for '{trend.trend_title[:30]}': {e}", "warning")
                    return self._create_basic_impact(trend)

        tasks = [_analyze_one(t) for t in state.trends]
        impacts = await asyncio.gather(*tasks)
        impacts = list(impacts)  # Convert from tuple

        # I1: Cross-trend synthesis if multiple impacts
        if len(impacts) >= 2:
            try:
                self._log(f"Cross-trend synthesis: Looking for compound opportunities across {len(impacts)} trends...")
                cross_insight = await self._synthesize_cross_trend_impacts(impacts)
                if cross_insight:
                    state.cross_trend_insight = cross_insight
                    compound = cross_insight.get('compound_impacts', [])
                    self._log(f"Cross-trend synthesis: {len(compound)} compound impacts found")
                    if cross_insight.get('mega_opportunity'):
                        self._log(f"  Mega opportunity: {cross_insight['mega_opportunity'][:100]}...")
                else:
                    self._log("Cross-trend synthesis: No compound impacts detected")
            except Exception as e:
                self._log(f"Cross-trend synthesis failed: {e}", "warning")

        state.impacts = impacts
        state.current_step = "impacts_analyzed"
        self._log(f"Impact analysis complete: {len(impacts)} trends analyzed", "success")

        return state
    
    async def _analyze_trend_impact(self, trend: TrendData) -> ImpactAnalysis:
        """
        Multi-perspective impact analysis via unified council call.

        Uses a single structured LLM call with all 4 analytical perspectives
        (industry, strategy, risk, fact-check) embedded in the system prompt.
        Falls back to direct single-call analysis if council fails.
        """
        try:
            return await self._council_analyze(trend)
        except Exception as e:
            self._log(f"  Council failed for '{trend.trend_title[:30]}', using fallback analysis: {e}", "warning")
            return await self._single_call_analyze(trend)

    async def _council_analyze(self, trend: TrendData) -> ImpactAnalysis:
        """Run unified multi-perspective council and convert to ImpactAnalysis."""
        self._log(f"  Preparing council input for: {trend.trend_title[:50]}...")
        article_excerpts = [trend.summary]

        # Real article evidence — lets council identify FIRST-ORDER (directly named) companies
        if trend.article_snippets:
            for i, snippet in enumerate(trend.article_snippets[:5], 1):
                article_excerpts.append(f"RAW ARTICLE EVIDENCE [{i}]: {snippet}")
            self._log(f"  Including {len(trend.article_snippets[:5])} article snippets as evidence")

        # V4: Pass richer context — causal chain, buying intent, actionable insight
        # These are synthesis-derived fields that give the council concrete facts
        if trend.causal_chain:
            for chain_step in trend.causal_chain[:4]:
                article_excerpts.append(f"CAUSAL: {chain_step}")

        if trend.actionable_insight:
            article_excerpts.append(f"INSIGHT: {trend.actionable_insight}")

        if trend.buying_intent:
            intent = trend.buying_intent
            if isinstance(intent, dict):
                who = intent.get("who_needs_help", "")
                what = intent.get("what_they_need", "")
                hook = intent.get("pitch_hook", "")
                if who:
                    article_excerpts.append(f"WHO NEEDS HELP: {who}")
                if what:
                    article_excerpts.append(f"WHAT THEY NEED: {what}")
                if hook:
                    article_excerpts.append(f"PITCH HOOK: {hook}")

        if trend.affected_companies:
            article_excerpts.append(f"AFFECTED COMPANIES: {', '.join(trend.affected_companies[:10])}")

        if trend.affected_regions:
            article_excerpts.append(f"REGIONS: {', '.join(trend.affected_regions[:5])}")

        if trend.source_links:
            article_excerpts.extend(
                f"Source: {link}" for link in trend.source_links[:5]
            )
            self._log(f"  Including {len(trend.source_links[:5])} source links as evidence")
        if trend.keywords:
            article_excerpts.append(f"Key topics: {', '.join(trend.keywords[:15])}")
            self._log(f"  Topics: {', '.join(trend.keywords[:8])}...")

        self._log(f"  Sending to AI council (4-lens analysis)...")
        council_result = await run_impact_council(
            trend_title=trend.trend_title,
            trend_summary=trend.summary,
            article_excerpts=article_excerpts,
            keywords=trend.keywords,
            entities=trend.industries_affected,
            signals={},
            llm_service=self.llm_service,
            log_callback=self._log_callback,
        )

        target_roles = council_result.target_roles or self._get_roles_from_keywords(trend.keywords)
        who_needs_help = ""
        if trend.buying_intent and isinstance(trend.buying_intent, dict):
            who_needs_help = trend.buying_intent.get("who_needs_help", "")
        self._log(f"  Council complete — building impact report (confidence={council_result.overall_confidence:.0%})")

        # ── Parse first/second order from debate_summary ──────────────────
        # _build_council_result encodes them as:
        # "First-order (N segments): X; Y | Second-order (M segments): A; B"
        first_order_list: list = []
        second_order_list: list = []
        debate = council_result.debate_summary or ""
        if "First-order" in debate and "Second-order" in debate:
            try:
                fo_part, so_part = debate.split(" | Second-order", 1)
                first_order_list = [s.strip() for s in fo_part.split(": ", 1)[-1].split(";") if s.strip()]
                second_order_list = [s.strip() for s in so_part.split(": ", 1)[-1].split(";") if s.strip()]
            except Exception:
                pass
        # Fallback: use first half / second half of affected_company_types
        if not first_order_list:
            all_types = council_result.affected_company_types or []
            half = max(1, len(all_types) // 2)
            first_order_list = all_types[:half]
            second_order_list = all_types[half:]

        return ImpactAnalysis(
            trend_id=trend.id,
            trend_title=trend.trend_title,
            who_needs_help=who_needs_help,
            direct_impact=first_order_list[:4],
            direct_impact_reasoning=council_result.detailed_reasoning,
            indirect_impact=second_order_list[:4],
            indirect_impact_reasoning=council_result.consensus_reasoning,
            additional_verticals=council_result.affected_sectors,
            additional_verticals_reasoning=council_result.debate_summary,
            midsize_pain_points=council_result.pain_points,
            consulting_projects=[
                f"{r.service_name}: {r.offering}" for r in council_result.service_recommendations
            ],
            positive_sectors=council_result.affected_sectors,
            negative_sectors=[],
            business_opportunities=council_result.business_opportunities,
            relevant_services=[r.service_name for r in council_result.service_recommendations],
            target_roles=target_roles,
            pitch_angle=council_result.pitch_angle,
            detailed_reasoning=council_result.detailed_reasoning,
            council_perspectives=[
                {
                    "role": p.agent_role,
                    "analysis": p.analysis,
                    "key_findings": p.key_findings,
                    "confidence": p.confidence,
                }
                for p in council_result.perspectives
            ],
            debate_summary=council_result.debate_summary,
            evidence_citations=council_result.evidence_citations,
            service_recommendations=[
                {
                    "service": r.service_name,
                    "offering": r.offering,
                    "justification": r.justification,
                    "urgency": r.urgency,
                }
                for r in council_result.service_recommendations
            ],
            council_confidence=council_result.overall_confidence,
        )

    async def _single_call_analyze(self, trend: TrendData) -> ImpactAnalysis:
        """Fallback: single LLM call analysis (original approach)."""
        # Build services context — NO truncation, show ALL offerings
        services_context = "\n".join([
            f"- {svc['name']}: {', '.join(svc['offerings'])}"
            for svc in CMI_SERVICES.values()
        ])
        
        prompt = f"""You are a business development consultant at Coherent Market Insights targeting MID-SIZE INDIAN COMPANIES (50-300 employees).

===== NEWS TO ANALYZE =====
HEADLINE: {trend.trend_title}
DETAILS: {trend.summary}

===== YOUR MISSION =====
Identify where MID-SIZE COMPANIES (not large enterprises, not tiny startups) will STRUGGLE due to this news.
Mid-size companies typically:
- Have limited internal strategy/research teams
- Cannot afford Big 4 consulting (McKinsey, BCG, Deloitte)
- Need actionable market intelligence to compete
- Face resource constraints but ambitious growth goals
- Lack bandwidth to track market changes themselves

Think about SPECIFIC BUSINESS CHALLENGES that create consulting opportunities.

===== EXAMPLE: How to Think Deep =====
NEWS: "India-EU FTA signed - Auto tariffs reduced 30%"

WRONG (too superficial): "Auto industry affected"

RIGHT (mid-size company focused):
- Tier-2 Auto Component Suppliers (50-200 employees): Will face pressure from OEMs to reduce prices OR risk losing contracts to EU suppliers. They need: (1) Cost benchmarking vs EU competitors, (2) Should-cost analysis to negotiate with OEMs, (3) Supplier diversification strategy.
- Mid-size Auto Ancillary Exporters: New opportunity to export to EU, but lack market intelligence on EU regulations, certification requirements, potential buyers. They need: Market entry feasibility study, regulatory compliance mapping.
- Regional Logistics Companies: EU cars need different spare parts distribution. They need: Supply chain reconfiguration study, warehouse location optimization.

===== NOW ANALYZE: {trend.trend_title} =====

Think: What specific mid-size company TYPES will face what SPECIFIC CHALLENGES?

Return this EXACT JSON:

{{
  "direct_impact": [
    "Specific mid-size company type 1 (e.g., 'Tier-2 Oil Field Equipment Suppliers')",
    "Specific mid-size company type 2 (e.g., 'Regional Fuel Distributors')",
    "Specific mid-size company type 3",
    "Specific mid-size company type 4"
  ],
  "direct_impact_reasoning": "For EACH company type above, explain: (1) What specific challenge they face, (2) What decision they need to make, (3) What information/analysis they lack. Be very specific about the business problem, not generic sector impact.",
  
  "indirect_impact": [
    "Mid-size company type affected indirectly 1",
    "Mid-size company type affected indirectly 2", 
    "Mid-size company type affected indirectly 3",
    "Mid-size company type affected indirectly 4"
  ],
  "indirect_impact_reasoning": "Explain the CHAIN: [News] causes [Direct Effect] which creates [Challenge for this mid-size company type]. What specific business decision do they now face? What analysis would help them?",
  
  "additional_verticals": [
    "Non-obvious mid-size company type 1",
    "Non-obvious mid-size company type 2",
    "Non-obvious mid-size company type 3",
    "Non-obvious mid-size company type 4",
    "Non-obvious mid-size company type 5"
  ],
  "additional_verticals_reasoning": "These are mid-size companies most people would NOT think of. Explain the non-obvious connection and what business challenge they face.",
  
  "midsize_pain_points": [
    "Specific pain point 1: e.g., 'Need to renegotiate supplier contracts but lack cost benchmarking data'",
    "Specific pain point 2: e.g., 'Facing margin pressure but dont know competitors pricing strategy'",
    "Specific pain point 3: e.g., 'Want to enter new market segment but lack feasibility analysis'",
    "Specific pain point 4: e.g., 'Board asking for impact assessment but no internal research team'",
    "Specific pain point 5"
  ],
  
  "consulting_projects": [
    "Specific deliverable 1: e.g., 'Cost structure benchmarking for oil field equipment manufacturers'",
    "Specific deliverable 2: e.g., 'Supplier risk assessment and alternative sourcing strategy'",
    "Specific deliverable 3: e.g., 'Market entry feasibility for [specific opportunity]'",
    "Specific deliverable 4: e.g., 'Competitive intelligence on how peers are responding to [this trend]'",
    "Specific deliverable 5"
  ],
  
  "positive_sectors": ["Sectors where mid-size companies will need help navigating change"],
  "negative_sectors": ["Sectors where mid-size companies might cut consulting budgets"],
  "relevant_services": ["Most relevant CMI service 1", "Service 2", "Service 3"],
  "target_roles": ["CEO", "CFO", "VP Strategy", "Director Business Development"],
  "pitch_angle": "One line showing you understand their specific challenge (max 100 chars)",
  "reasoning": "Why would a mid-size company CEO pay for consulting RIGHT NOW based on this news?"
}}

CMI SERVICES WE CAN OFFER:
{services_context}

REMEMBER:
- Focus on MID-SIZE companies (50-300 employees), not Tata/Reliance/Infosys
- Be SPECIFIC about business challenges, not generic sector impacts
- Think about what DECISIONS these companies need to make
- Identify where they LACK INFORMATION that CMI can provide
- Write in PLAIN ENGLISH — avoid consulting jargon and buzzwords
  Say "find cheaper suppliers" not "procurement cost optimization"
  Say "figure out what competitors charge" not "competitive pricing intelligence"
  The reader is a business owner, not a strategy consultant"""

        system_prompt = """You are a business development expert who understands mid-size Indian companies deeply.
You know that mid-size companies (50-300 employees) have unique challenges:
- They're too big to ignore market changes, too small to have in-house research teams
- They compete against both large players AND hungry startups
- They need actionable intelligence, not 200-page reports
- They make decisions fast but need data to back them up
- Their C-suite is accessible and makes buying decisions quickly

Your job is to identify SPECIFIC business challenges where Coherent Market Insights can help.
Always respond with valid JSON only."""

        # Try Track A: structured output (works with Gemini, may fail on NVIDIA)
        try:
            from app.schemas.llm_outputs import ImpactAnalysisLLM
            result = await self.llm_service.run_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                output_type=ImpactAnalysisLLM,
            )
            return self._build_impact_from_llm(result, trend)

        except Exception as e:
            logger.warning(f"Structured impact analysis failed: {e}, trying generate_json...")

        # Try Track B: generate_json() — works with NVIDIA DeepSeek
        try:
            raw = await self.llm_service.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if isinstance(raw, dict) and "error" not in raw:
                raw = self._validate_impact_response(raw)
                from app.schemas.llm_outputs import ImpactAnalysisLLM
                result = ImpactAnalysisLLM(**{
                    k: v for k, v in raw.items()
                    if k in ImpactAnalysisLLM.model_fields
                })
                logger.info(f"generate_json fallback succeeded with {len(result.direct_impact)} direct impacts")
                return self._build_impact_from_llm(result, trend)
            else:
                logger.warning(f"generate_json returned error: {raw.get('error', 'unknown') if isinstance(raw, dict) else raw}")
        except Exception as e2:
            logger.warning(f"generate_json fallback also failed: {e2}")

        return self._create_basic_impact(trend)
    
    async def _synthesize_cross_trend_impacts(
        self, impacts: List[ImpactAnalysis]
    ) -> Optional[dict]:
        """
        I1: Cross-trend impact synthesis — our differentiation.

        After individual impact analyses, run ONE additional LLM call to find
        companies hit by MULTIPLE trends simultaneously. This is what Meltwater
        and Feedly DON'T do — they show trends independently.

        Returns:
            Dict with cross_trend_insight, compound_impacts, mega_opportunity.
            None if synthesis fails or is not valuable.
        """
        # Build summary of all impacts for the LLM
        impact_summaries = []
        for i, impact in enumerate(impacts[:6], 1):  # Cap at 6 to fit context
            direct = ', '.join(impact.direct_impact[:3]) if impact.direct_impact else 'N/A'
            pain = ', '.join(impact.midsize_pain_points[:2]) if impact.midsize_pain_points else 'N/A'
            impact_summaries.append(
                f"Trend {i}: {impact.trend_title}\n"
                f"  Affected: {direct}\n"
                f"  Pain points: {pain}"
            )

        prompt = f"""These {len(impact_summaries)} trends are happening SIMULTANEOUSLY in India right now.

{chr(10).join(impact_summaries)}

Identify COMPOUND OPPORTUNITIES — company types hit by MULTIPLE trends at once.

Return JSON:
{{
    "cross_trend_insight": "1-2 paragraph narrative about the compound impact landscape",
    "compound_impacts": [
        {{
            "company_type": "Specific mid-size company type (e.g., 'Regional pharma distributors')",
            "affected_by_trends": [1, 3],
            "compound_challenge": "Why being hit by BOTH trends simultaneously is worse than either alone",
            "consulting_opportunity": "What specific analysis/service they need urgently"
        }}
    ],
    "mega_opportunity": "The single best pitch combining multiple trends (1 sentence)"
}}

Focus on mid-size Indian companies (50-300 employees).
Find 2-4 company types affected by 2+ trends simultaneously."""

        try:
            result = await self.llm_service.generate_json(
                prompt=prompt,
                system_prompt="You are a senior strategy consultant. Identify compound market impacts. Respond with JSON only."
            )
            if isinstance(result, dict) and "error" not in result:
                # V5: Validate cross-trend synthesis structure
                if not isinstance(result.get("compound_impacts"), list):
                    result["compound_impacts"] = []
                    logger.debug("Coerced compound_impacts to empty list")
                if not isinstance(result.get("mega_opportunity"), str):
                    result["mega_opportunity"] = str(result.get("mega_opportunity", ""))
                    logger.debug("Coerced mega_opportunity to string")
                if not isinstance(result.get("cross_trend_insight"), str):
                    result["cross_trend_insight"] = str(result.get("cross_trend_insight", ""))
                return result
            if isinstance(result, dict):
                logger.debug(f"Cross-trend synthesis returned error: {result.get('error', 'unknown')}")
            else:
                logger.debug(f"Cross-trend synthesis returned {type(result).__name__} instead of dict")
            return None
        except Exception as e:
            logger.warning(f"Cross-trend synthesis LLM call failed: {e}")
            return None

    @staticmethod
    def _validate_impact_response(result: dict) -> dict:
        """
        V5: Validate and coerce all LLM impact response fields.

        Ensures every field has the correct type before ImpactAnalysis
        creation. Logs every coercion for observability.

        Coercions:
        - List fields: None→[], str→[str], non-list→[str(val)]
        - String fields: None→"", non-str→str(val)
        - pitch_angle: truncate to 150 chars
        """
        coercions = 0

        # List fields that must be List[str]
        list_fields = [
            "direct_impact", "indirect_impact", "additional_verticals",
            "midsize_pain_points", "consulting_projects", "positive_sectors",
            "negative_sectors", "business_opportunities", "relevant_services",
            "target_roles",
        ]
        for field in list_fields:
            val = result.get(field)
            if val is None:
                result[field] = []
                coercions += 1
            elif isinstance(val, str):
                result[field] = [val.strip()] if val.strip() else []
                coercions += 1
            elif not isinstance(val, list):
                result[field] = [str(val)] if val else []
                coercions += 1
            else:
                # Filter None/empty items from existing lists
                result[field] = [str(item).strip() for item in val if item and str(item).strip()]

        # String fields that must be str
        str_fields = [
            "direct_impact_reasoning", "indirect_impact_reasoning",
            "additional_verticals_reasoning", "pitch_angle",
        ]
        for field in str_fields:
            val = result.get(field)
            if val is None:
                result[field] = ""
                coercions += 1
            elif not isinstance(val, str):
                result[field] = str(val)
                coercions += 1

        # Truncate pitch_angle
        pitch = result.get("pitch_angle", "")
        if len(pitch) > 150:
            result["pitch_angle"] = pitch[:147] + "..."
            coercions += 1

        if coercions > 0:
            logger.info(f"V5: Coerced {coercions} impact response field(s)")

        return result

    def _build_impact_from_llm(self, result, trend: TrendData) -> ImpactAnalysis:
        """Build ImpactAnalysis from ImpactAnalysisLLM result (shared by structured + json paths)."""
        target_roles = result.target_roles
        if not target_roles:
            target_roles = self._get_roles_from_keywords(trend.keywords)

        impact = ImpactAnalysis(
            trend_id=trend.id,
            trend_title=trend.trend_title,
            direct_impact=result.direct_impact,
            direct_impact_reasoning=result.direct_impact_reasoning,
            indirect_impact=result.indirect_impact,
            indirect_impact_reasoning=result.indirect_impact_reasoning,
            additional_verticals=result.additional_verticals,
            additional_verticals_reasoning=result.additional_verticals_reasoning,
            midsize_pain_points=result.midsize_pain_points,
            consulting_projects=result.consulting_projects,
            positive_sectors=result.positive_sectors or trend.industries_affected,
            negative_sectors=result.negative_sectors,
            business_opportunities=result.consulting_projects,
            target_roles=target_roles,
            relevant_services=result.relevant_services,
            pitch_angle=result.pitch_angle,
        )
        logger.info(f"Created ImpactAnalysis with {len(impact.direct_impact)} direct impacts")
        return impact

    def _create_basic_impact(self, trend: TrendData) -> ImpactAnalysis:
        """Create a basic impact analysis when LLM fails."""
        return ImpactAnalysis(
            trend_id=trend.id,
            trend_title=trend.trend_title,
            # Deep analysis fields with defaults
            direct_impact=trend.industries_affected[:4] if trend.industries_affected else [],
            direct_impact_reasoning=f"These industries are directly mentioned or affected by: {trend.trend_title}. Further analysis with LLM recommended.",
            indirect_impact=[],
            indirect_impact_reasoning="Indirect impact analysis requires LLM model. Please check API configuration.",
            additional_verticals=[],
            additional_verticals_reasoning="Additional verticals analysis requires LLM model. Please check API configuration.",
            positive_sectors=trend.industries_affected,
            negative_sectors=[],
            business_opportunities=[f"Capitalize on {trend.trend_title}"],
            target_roles=self._get_roles_from_keywords(trend.keywords),
            relevant_services=["Market Intelligence", "Industry Analysis"],
            pitch_angle=f"Expert insights on {trend.trend_title[:50]}...",
        )
    
    def _get_roles_from_keywords(self, keywords: List[str]) -> List[str]:
        """Determine target roles using TREND_ROLE_MAPPING keys directly.

        The LLM already classifies event types (regulation, funding, etc.)
        which map directly to TREND_ROLE_MAPPING keys. No keyword matching
        needed — just check if any keyword IS a known trend type.
        """
        roles = set()

        known_types = set(TREND_ROLE_MAPPING.keys())
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in known_types:
                roles.update(TREND_ROLE_MAPPING[keyword_lower])

        if not roles:
            roles.update(TREND_ROLE_MAPPING.get("default", []))

        return list(roles)[:5]


# Backward compatibility alias
ImpactAgent = ImpactAnalyzer


async def run_impact_agent(state: AgentState, deps=None) -> AgentState:
    """Wrapper function for LangGraph."""
    analyzer = ImpactAnalyzer(deps=deps) if deps else ImpactAnalyzer()
    return await analyzer.analyze_impacts(state)
