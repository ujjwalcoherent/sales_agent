"""
Impact Mapping Agent - Consultant Edition.
Thinks like a consultant to identify which companies need Coherent Market Insights services.

V7: Parallel impact analysis with asyncio.gather + Semaphore.
I1: Cross-trend impact synthesis — compound opportunity detection.
"""

import asyncio
import logging
from typing import List, Optional

from ..schemas import TrendData, ImpactAnalysis, AgentState
from ..tools.llm_tool import LLMTool
from ..config import get_settings, TREND_ROLE_MAPPING, CMI_SERVICES

logger = logging.getLogger(__name__)


class ImpactAgent:
    """
    Agent that thinks like a CONSULTANT to analyze market trends.
    
    For each trend, determines:
    - Which industries are impacted and HOW
    - Which CMI services are most relevant
    - What specific consulting opportunities exist
    - Who are the decision makers who would buy these services
    """
    
    def __init__(self, mock_mode: bool = False):
        """Initialize impact agent with standard provider fallback chain."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        # Use normal provider chain (NVIDIA→Ollama→OpenRouter→Gemini→Groq)
        # Previously force_groq=True bypassed the entire fallback chain
        self.llm_tool = LLMTool(mock_mode=self.mock_mode)
        provider = self.llm_tool.settings.get_llm_config().get("provider", "unknown")
        logger.info(f"Impact Agent initialized (provider={provider})")
    
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
        logger.info("Starting impact analysis...")

        if not state.trends:
            logger.warning("No trends to analyze")
            return state

        # V7: Parallel impact analysis with concurrency limit
        semaphore = asyncio.Semaphore(4)

        async def _analyze_one(trend: TrendData) -> ImpactAnalysis:
            async with semaphore:
                try:
                    impact = await self._analyze_trend_impact(trend)
                    logger.info(f"Analyzed impact for: {trend.trend_title[:40]}...")
                    return impact
                except Exception as e:
                    logger.warning(f"Failed to analyze impact for '{trend.trend_title[:30]}': {e}")
                    return self._create_basic_impact(trend)

        tasks = [_analyze_one(t) for t in state.trends]
        impacts = await asyncio.gather(*tasks)
        impacts = list(impacts)  # Convert from tuple

        # I1: Cross-trend synthesis if multiple impacts
        if len(impacts) >= 2:
            try:
                cross_insight = await self._synthesize_cross_trend_impacts(impacts)
                if cross_insight:
                    state.cross_trend_insight = cross_insight
                    logger.info(f"Cross-trend synthesis complete: {len(cross_insight.get('compound_impacts', []))} compound impacts")
            except Exception as e:
                logger.warning(f"Cross-trend synthesis failed: {e}")

        state.impacts = impacts
        state.current_step = "impacts_analyzed"
        logger.info(f"Completed {len(impacts)} impact analyses")

        return state
    
    async def _analyze_trend_impact(self, trend: TrendData) -> ImpactAnalysis:
        """
        Deep 3-part consultant analysis of a trend using LLM (Groq 120B primary).
        
        Args:
            trend: Trend data to analyze
            
        Returns:
            ImpactAnalysis with direct, indirect, and additional vertical impacts
        """
        # Build services context
        services_context = "\n".join([
            f"- {svc['name']}: {', '.join(svc['offerings'][:3])}"
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
- Identify where they LACK INFORMATION that CMI can provide"""

        system_prompt = """You are a business development expert who understands mid-size Indian companies deeply.
You know that mid-size companies (50-300 employees) have unique challenges:
- They're too big to ignore market changes, too small to have in-house research teams
- They compete against both large players AND hungry startups
- They need actionable intelligence, not 200-page reports
- They make decisions fast but need data to back them up
- Their C-suite is accessible and makes buying decisions quickly

Your job is to identify SPECIFIC business challenges where Coherent Market Insights can help.
Always respond with valid JSON only."""

        try:
            result = await self.llm_tool.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )

            # H2: Check for error dict from failed JSON parsing
            if isinstance(result, dict) and "error" in result:
                logger.warning(f"LLM returned error dict: {result['error']}. Falling back to basic impact.")
                return self._create_basic_impact(trend)

            logger.info(f"LLM returned impact analysis with keys: {result.keys()}")

            # V5: Validate and coerce all fields before creating ImpactAnalysis
            result = self._validate_impact_response(result)

            # Enhance target roles based on trend keywords
            target_roles = result.get("target_roles", [])
            if not target_roles:
                target_roles = self._get_roles_from_keywords(trend.keywords)

            # Extract the deep mid-size company focused fields
            impact = ImpactAnalysis(
                trend_id=trend.id,
                trend_title=trend.trend_title,
                # Part 1: Direct Impact on Mid-Size Companies
                direct_impact=result.get("direct_impact", []),
                direct_impact_reasoning=result.get("direct_impact_reasoning", ""),
                # Part 2: Indirect Impact
                indirect_impact=result.get("indirect_impact", []),
                indirect_impact_reasoning=result.get("indirect_impact_reasoning", ""),
                # Part 3: Additional Verticals
                additional_verticals=result.get("additional_verticals", []),
                additional_verticals_reasoning=result.get("additional_verticals_reasoning", ""),
                # NEW: Mid-size company pain points
                midsize_pain_points=result.get("midsize_pain_points", []),
                # NEW: Specific consulting projects
                consulting_projects=result.get("consulting_projects", []),
                # Consulting opportunities
                positive_sectors=result.get("positive_sectors", trend.industries_affected),
                negative_sectors=result.get("negative_sectors", []),
                business_opportunities=result.get("consulting_projects", result.get("business_opportunities", [])),
                target_roles=target_roles,
                relevant_services=result.get("relevant_services", []),
                pitch_angle=result.get("pitch_angle", ""),
                reasoning=result.get("reasoning", "")
            )

            logger.info(f"Created ImpactAnalysis with direct_impact: {impact.direct_impact}")
            logger.info(f"Direct impact reasoning: {impact.direct_impact_reasoning[:100]}..." if impact.direct_impact_reasoning else "No direct reasoning")

            return impact

        except Exception as e:
            logger.warning(f"LLM impact analysis failed: {e}")
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
            result = await self.llm_tool.generate_json(
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
            logger.debug(f"Cross-trend synthesis returned error: {result.get('error', 'unknown')}")
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
            "additional_verticals_reasoning", "pitch_angle", "reasoning",
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
            reasoning="Market trend with potential impact on Indian industries. LLM-based deep analysis recommended."
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


async def run_impact_agent(state: AgentState) -> AgentState:
    """Wrapper function for LangGraph."""
    agent = ImpactAgent()
    return await agent.analyze_impacts(state)
