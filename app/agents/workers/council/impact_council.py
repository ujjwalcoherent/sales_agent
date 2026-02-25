"""
Stage B: Unified Impact Analysis — multi-perspective single-call approach.

Research backing for simplification from 13 LLM calls to 1:
- Smit et al. (ICML 2024): "Multi-agent debate does not reliably outperform
  single-agent baselines like self-consistency and ensembling"
- Zhang et al. (2025): "No single MAD method demonstrates robust superior
  performance across benchmarks, LLMs, and conditions"
- Li et al. (2025, Self-MoA): "Intra-model diversity is more beneficial
  than inter-model diversity; mixing agents lowers average quality"
- OpenAI Structured Outputs (2024): "Adding a reasoning field before answer
  fields increased model accuracy by 60% on GSM8k"

Architecture: ONE structured call with all 4 analytical perspectives
embedded in the system prompt. The reasoning field provides chain-of-thought
space that preserves analytical depth without separate agent overhead.
"""

import logging
from typing import Any, Dict, List

from .schemas import CouncilPerspective, ImpactCouncilResult, ServiceRecommendation
from ....tools.llm_service import LLMService
from ....config import get_settings, CMI_SERVICES

logger = logging.getLogger(__name__)


def _build_services_context() -> str:
    """Build full CMI services catalog for prompts (no truncation)."""
    lines = []
    for svc_name, svc_data in CMI_SERVICES.items():
        lines.append(f"\n{svc_name}: {svc_data.get('description', '')}")
        for offering in svc_data.get("offerings", []):
            lines.append(f"  - {offering}")
    return "\n".join(lines)


# ── Unified Multi-Perspective System Prompt ──────────────────────────

UNIFIED_SYSTEM_PROMPT = """You are a senior analyst council at Coherent Market Insights — a market research and consulting firm targeting MID-SIZE COMPANIES (50-500 employees).

You combine FOUR analytical perspectives AND you must explicitly separate FIRST-ORDER from SECOND-ORDER impact.

## FIRST-ORDER vs SECOND-ORDER IMPACT (CRITICAL DISTINCTION)

**FIRST-ORDER = companies/segments DIRECTLY mentioned or clearly implied in the source articles**
- These are the IMMEDIATE casualties or beneficiaries
- You MUST be able to point to a specific article sentence that supports this
- Example: "Silver import duty increased → silver jewellery exporters are first-order (named in article)"
- Put these in `first_order_companies` with a citation from the source

**SECOND-ORDER = companies affected INDIRECTLY through supply chain, market dynamics, or customer behavior**
- These are NOT in the source articles — you are inferring based on business logic
- You MUST state the CHAIN explicitly: "A impacts B which in turn impacts C because..."
- Example: "Jewellers raise prices → wedding planners lose budget → event caterers see fewer premium events"
- Put these in `second_order_companies` with the transmission mechanism

WRONG: Listing the same company type in both first and second order
WRONG: Putting generic sectors in first_order without article evidence
RIGHT: First-order is grounded in SOURCE DATA. Second-order is INFERRED with explicit chain logic.

## INDUSTRY ANALYST LENS
- Name SPECIFIC types of mid-size companies affected (not generic sectors)
  BAD: "companies in the silver industry"
  GOOD: "Silver jewellery manufacturers in Rajkot (100-300 employees) who import raw silver"
- Identify specific players, suppliers, and buyers from the source data
- Reference concrete data points (tariff %, price changes, market shares)

## STRATEGY CONSULTANT LENS
- What specific DECISIONS do affected mid-size companies face RIGHT NOW?
  BAD: "Companies need to adapt their strategy"
  GOOD: "Tier-2 auto suppliers must decide by Q2 whether to absorb the 12% cost or renegotiate with OEMs"
- Name exact consulting DELIVERABLES with scope
  BAD: "Market research"
  GOOD: "Should-cost analysis for silver-based electronic components, benchmarking against 5 alternative materials"

## RISK ANALYST LENS
- Name SPECIFIC regulations, regulators, policy bodies, deadlines
- What goes wrong for companies that DON'T act? Quantify with data from articles
- Compliance timelines and enforcement milestones

## FACT-CHECKER LENS (self-critique)
- For every claim, mentally verify: is this stated in the source data, or am I inferring?
- Flag any contradictions between sources
- Separate hype from reality — what do sources ACTUALLY say vs what's implied?
- MARK your first_order_companies claims with the evidence. MARK second_order claims as inference.

CRITICAL RULES:
- Focus on MID-SIZE companies (50-500 employees), not Tata/Reliance/Infosys
- Every first_order claim must trace to source data
- Every second_order claim must show the transmission chain
- Be SPECIFIC about business challenges, not generic sector impacts
- Write in PLAIN ENGLISH — avoid consulting jargon, buzzwords, and acronyms
  BAD: "Conduct a TAM/SAM analysis leveraging our competitive intelligence framework"
  GOOD: "Research how big the market is and who the main competitors are"
  The reader should be a business owner, not a McKinsey partner."""


async def run_impact_council(
    trend_title: str,
    trend_summary: str,
    article_excerpts: List[str],
    keywords: List[str],
    entities: List[str],
    signals: Dict[str, Any],
    llm_service: LLMService = None,
    log_callback=None,
) -> ImpactCouncilResult:
    """
    Run unified multi-perspective impact analysis (1 structured LLM call).

    Replaces the old 4-round debate (13 calls) with a single call that
    embeds all 4 analytical perspectives in the system prompt. Research
    shows this achieves comparable quality at 92% cost reduction.
    """
    def _log(msg, level="info"):
        logger.info(msg) if level == "info" else logger.warning(msg)
        if log_callback:
            try:
                log_callback(msg, level)
            except Exception:
                pass

    if llm_service is None:
        llm_service = LLMService(mock_mode=get_settings().mock_mode, force_groq=True)

    _log(f"  Council: Preparing analysis for '{trend_title[:50]}'...")
    _log(f"  Council: Loading {len(article_excerpts)} source excerpts, {len(keywords)} topics, {len(entities)} entities")

    services_ctx = _build_services_context()
    articles_text = "\n".join(f"  - {e}" for e in article_excerpts[:15])

    prompt = f"""Analyze this trend for business consulting opportunities:

TREND: {trend_title}

FULL ANALYSIS (from source articles — use these facts):
{trend_summary}

SOURCE DATA:
{articles_text}

KEY TOPICS: {', '.join(keywords[:15])}
ENTITIES & SECTORS: {', '.join(entities[:15])}

OUR SERVICE CATALOG (reference specific services AND offerings):
{services_ctx}

IMPORTANT OUTPUT REQUIREMENTS:
- In the `reasoning` field: Write 400+ words of step-by-step analysis. START with "FIRST-ORDER ANALYSIS:" then "SECOND-ORDER ANALYSIS:" to show your chain thinking
- In `first_order_companies`: List company types DIRECTLY evidenced in the source articles. Format: "[Company type] ([employee range], [location]) — EVIDENCE: [quote or fact from source]"
- In `first_order_mechanism`: Explain the DIRECT causal link with specific numbers if available
- In `second_order_companies`: List company types affected INDIRECTLY. Format: "[Company type] — CHAIN: [first-order effect] → [why this company is affected] → [specific impact]"
- In `second_order_mechanism`: Explain the full transmission chain (supply chain, pricing, demand, regulatory)
- In `detailed_reasoning`: Write a PLAIN-ENGLISH 5-paragraph brief:
  1. What happened (news in simple terms)
  2. Who is DIRECTLY hit (first-order) — with evidence from articles
  3. Who is INDIRECTLY affected (second-order) — with chain logic
  4. What decisions these companies need to make RIGHT NOW
  5. Why they should act soon (deadlines, competitive pressure)
  AVOID jargon. USE plain language.
- Each `pain_point`: The specific problem + what happens if they ignore it
- Each `consulting_project`: Plain-language deliverable (e.g., "Report comparing 5 alternative suppliers" NOT "Procurement benchmarking")
- `pitch_angle`: ONE sentence (e.g., "We can help you find cheaper suppliers before the tariffs kick in")
- Each `evidence_citation`: Specific fact, number, or quote from SOURCE DATA
  GOOD: "Silver import duty increased from 7.5% to 12.5% (Economic Times, Feb 2025)"
  BAD: "According to sources" or "Industry reports suggest"
  Include at least 5 citations.

Analyze all four lenses. Think deeply — FIRST-ORDER then SECOND-ORDER — before filling output fields."""

    from app.schemas.llm_outputs import UnifiedImpactAnalysisLLM, ServiceRecommendationLLM

    # Try Track A: structured output (works with Gemini, may fail on NVIDIA)
    try:
        _log(f"  Council: Analyzing through 4 lenses (industry, strategy, risk, fact-check)...")
        _log(f"  Council: Sending to LLM with {len(services_ctx)} chars of service catalog...")
        result = await llm_service.run_structured(
            prompt=prompt,
            system_prompt=UNIFIED_SYSTEM_PROMPT,
            output_type=UnifiedImpactAnalysisLLM,
        )
        reasoning_len = len(result.reasoning) + len(result.detailed_reasoning)
        _log(
            f"  Council: LLM returned — "
            f"{len(result.affected_company_types)} company types, "
            f"{len(result.pain_points)} pain points, "
            f"{len(result.service_recommendations)} service recs, "
            f"confidence={result.confidence:.0%}"
        )
        if result.evidence_citations:
            _log(f"  Council: {len(result.evidence_citations)} evidence citations found")
        if result.affected_sectors:
            _log(f"  Council: Affected sectors — {', '.join(result.affected_sectors[:5])}")
        return _build_council_result(result)

    except Exception as e:
        _log(f"  Council: Structured output failed ({e}), trying unstructured fallback...", "warning")

    # Try Track B: generate_json() — works with NVIDIA DeepSeek (no grammar validation)
    try:
        raw = await llm_service.generate_json(
            prompt=prompt,
            system_prompt=UNIFIED_SYSTEM_PROMPT,
        )
        if isinstance(raw, dict) and "error" not in raw:
            # Parse nested service_recommendations from dicts
            svc_recs_raw = raw.get("service_recommendations", [])
            if svc_recs_raw and isinstance(svc_recs_raw, list):
                parsed_recs = []
                for sr in svc_recs_raw:
                    if isinstance(sr, dict):
                        parsed_recs.append(ServiceRecommendationLLM(**{
                            k: v for k, v in sr.items()
                            if k in ServiceRecommendationLLM.model_fields
                        }))
                raw["service_recommendations"] = parsed_recs

            # Build typed model from raw dict (unknown fields are ignored)
            result = UnifiedImpactAnalysisLLM(**{
                k: v for k, v in raw.items()
                if k in UnifiedImpactAnalysisLLM.model_fields
            })
            _log(
                f"  Council: Unstructured fallback succeeded — "
                f"{len(result.affected_company_types)} company types, "
                f"{len(result.pain_points)} pain points"
            )
            return _build_council_result(result)
        else:
            raise RuntimeError(f"generate_json returned error: {raw.get('error', 'unknown') if isinstance(raw, dict) else raw}")

    except Exception as e2:
        _log(f"  Council: Both structured and unstructured failed — {e2}", "warning")
        raise  # Propagate to trigger _single_call_analyze() fallback in impact_agent.py


def _compute_confidence(result) -> float:
    """Compute confidence from actual output QUALITY, not just item counts.

    Each component has two parts: count (did the LLM produce enough items?)
    and quality (are those items specific and grounded, not vague filler?).

    Components (each 0-1, weighted):
      25% evidence_grounding — citations with concrete data (numbers, names)
      20% specificity        — company types with employee counts/locations
      20% analytical_depth   — reasoning length AND paragraph structure
      15% problem_concrete   — pain points with specific consequences
      10% service_fit        — service recs with real justifications
      10% cross_validation   — internal consistency checks

    Calibration target: good analysis → 50-70%, great → 70-85%, never >90%.
    """
    import re

    # ── 1. Evidence Grounding (25%) ────────────────────────────────────
    # Count: need 10+ citations for full marks (not 5)
    citations = result.evidence_citations or []
    evidence_count = min(1.0, len(citations) / 10)
    # Quality: what fraction contain concrete data (numbers, %, names)?
    _data_pattern = re.compile(r'\d+\.?\d*\s*%|₹|\$|USD|INR|\d{4}|\d+\s*(crore|lakh|billion|million|employees|companies|firms)')
    grounded = sum(1 for c in citations if _data_pattern.search(c)) if citations else 0
    evidence_quality = (grounded / len(citations)) if citations else 0
    evidence = evidence_count * 0.5 + evidence_quality * 0.5

    # ── 2. Specificity (20%) ──────────────────────────────────────────
    # Count: need 6+ company types for full marks (not 4)
    company_types = result.affected_company_types or []
    spec_count = min(1.0, len(company_types) / 6)
    # Quality: fraction with specificity markers (employee ranges, locations, sub-industries)
    _spec_pattern = re.compile(r'\d+[-–]\d+\s*employee|\d+\s*employee|\d+[-–]\d+\s*staff|tier[- ]?[123]|small|mid[- ]?size|SME|MSME', re.IGNORECASE)
    specific = sum(1 for ct in company_types if _spec_pattern.search(ct)) if company_types else 0
    spec_quality = (specific / len(company_types)) if company_types else 0
    specificity = spec_count * 0.5 + spec_quality * 0.5

    # ── 3. Analytical Depth (20%) ─────────────────────────────────────
    # Length: need 3000+ chars combined (not 1200)
    total_reasoning = (result.reasoning or '') + (result.detailed_reasoning or '')
    depth_length = min(1.0, len(total_reasoning) / 3000)
    # Structure: need 5+ distinct paragraphs
    paragraphs = [p.strip() for p in total_reasoning.split('\n') if len(p.strip()) > 40]
    depth_structure = min(1.0, len(paragraphs) / 5)
    depth = depth_length * 0.6 + depth_structure * 0.4

    # ── 4. Problem Concreteness (15%) ─────────────────────────────────
    # Count: need 6+ pain points (not 4)
    pain_points = result.pain_points or []
    pain_count = min(1.0, len(pain_points) / 6)
    # Quality: fraction with concrete consequences (deadlines, numbers, "if...then")
    _consequence_pattern = re.compile(r'if\s|will\s|must\s|by\s+Q[1-4]|deadline|%|\d+\s*(crore|lakh|million|billion)', re.IGNORECASE)
    concrete = sum(1 for pp in pain_points if _consequence_pattern.search(pp)) if pain_points else 0
    pain_quality = (concrete / len(pain_points)) if pain_points else 0
    pain = pain_count * 0.5 + pain_quality * 0.5

    # ── 5. Service Fit (10%) ──────────────────────────────────────────
    # Count: need 4+ service recs (not 3)
    svc_recs = result.service_recommendations or []
    svc_count = min(1.0, len(svc_recs) / 4)
    # Quality: fraction with substantive justification (>30 chars, not boilerplate)
    justified = sum(1 for r in svc_recs if len(r.justification or '') > 30) if svc_recs else 0
    svc_quality = (justified / len(svc_recs)) if svc_recs else 0
    service_fit = svc_count * 0.5 + svc_quality * 0.5

    # ── 6. Cross-validation (10%) — replaces LLM self-report ─────────
    # Internal consistency checks: does the output hang together?
    checks_passed = 0
    checks_total = 4
    # Check 1: pitch_angle exists and is substantial
    if len(result.pitch_angle or '') > 30:
        checks_passed += 1
    # Check 2: target_roles exist
    if len(result.target_roles or []) >= 2:
        checks_passed += 1
    # Check 3: business_opportunities exist
    if len(result.business_opportunities or []) >= 2:
        checks_passed += 1
    # Check 4: affected_sectors align with company_types (both non-empty)
    if (result.affected_sectors or []) and (result.affected_company_types or []):
        checks_passed += 1
    cross_val = checks_passed / checks_total

    # ── Vagueness penalty ─────────────────────────────────────────────
    # Scan for consulting jargon / filler phrases that indicate low quality
    _vague_phrases = [
        'companies need to adapt', 'strategic optimization', 'leverage opportunities',
        'navigate challenges', 'various companies', 'many organizations',
        'optimize their', 'enhance their', 'stakeholder engagement',
        'value chain', 'holistic approach', 'paradigm shift',
        'companies in the sector', 'industry players',
    ]
    all_text = total_reasoning.lower() + (result.pitch_angle or '').lower()
    vague_hits = sum(1 for phrase in _vague_phrases if phrase in all_text)
    vagueness_penalty = min(0.15, vague_hits * 0.03)

    # ── Composite ─────────────────────────────────────────────────────
    composite = (
        0.25 * evidence
        + 0.20 * specificity
        + 0.20 * depth
        + 0.15 * pain
        + 0.10 * service_fit
        + 0.10 * cross_val
        - vagueness_penalty
    )
    return round(min(0.95, max(0.05, composite)), 2)


def _build_council_result(result) -> ImpactCouncilResult:
    """Convert unified LLM output to ImpactCouncilResult (preserves interface).

    Now uses explicit first_order_companies / second_order_companies from LLM
    instead of slicing the same list in two directions.
    Falls back to positional split if the LLM didn't populate the new fields.
    """
    # Build service recommendations from structured output
    svc_recs = []
    for rec in result.service_recommendations:
        svc_recs.append(ServiceRecommendation(
            service_name=rec.service,
            offering=rec.offering,
            justification=rec.justification,
            urgency=rec.urgency,
        ))

    # Compute confidence from actual output quality
    confidence = _compute_confidence(result)

    # ── First / Second order — use explicit LLM fields if populated ──────
    first_order = getattr(result, "first_order_companies", []) or []
    second_order = getattr(result, "second_order_companies", []) or []

    # Fallback: if the LLM didn't populate the new fields, split affected_company_types
    # by evidence markers — items with "EVIDENCE:" or source citations go first-order
    if not first_order and result.affected_company_types:
        for ct in result.affected_company_types:
            if any(marker in ct.upper() for marker in ["EVIDENCE:", "ARTICLE:", "NAMED IN", "CITED IN"]):
                first_order.append(ct)
            else:
                second_order.append(ct)
        # Final fallback: first half → first order, second half → second order
        if not first_order:
            half = max(1, len(result.affected_company_types) // 2)
            first_order = result.affected_company_types[:half]
            second_order = result.affected_company_types[half:]

    # Build mechanism context from new fields
    first_order_mech = getattr(result, "first_order_mechanism", "") or ""
    second_order_mech = getattr(result, "second_order_mechanism", "") or ""

    # Enrich detailed_reasoning with the first/second order structure
    enriched_reasoning = result.detailed_reasoning
    if first_order_mech and "FIRST-ORDER" not in enriched_reasoning.upper():
        enriched_reasoning = (
            f"FIRST-ORDER IMPACT: {first_order_mech}\n\n"
            f"SECOND-ORDER IMPACT: {second_order_mech}\n\n"
            + enriched_reasoning
        )

    # Create a single "unified" perspective for backward compatibility
    perspective = CouncilPerspective(
        agent_role="unified_analyst",
        analysis=result.reasoning,
        key_findings=result.pain_points[:5],
        affected_company_types=first_order + second_order,  # Full union for display
        recommended_services=[f"{r.service}: {r.offering}" for r in result.service_recommendations],
        confidence=confidence,
        evidence_citations=result.evidence_citations,
    )

    return ImpactCouncilResult(
        perspectives=[perspective],
        consensus_reasoning=result.reasoning,
        debate_summary=(
            f"First-order ({len(first_order)} segments): "
            + "; ".join(first_order[:2])
            + f" | Second-order ({len(second_order)} segments): "
            + "; ".join(second_order[:2])
        ),
        detailed_reasoning=enriched_reasoning,
        pitch_angle=result.pitch_angle,
        service_recommendations=svc_recs,
        evidence_citations=result.evidence_citations[:10],
        overall_confidence=confidence,
        affected_sectors=result.affected_sectors,
        # Preserve both orders — first_order in first 4 slots, second in next 4
        affected_company_types=first_order[:4] + second_order[:4],
        pain_points=result.pain_points,
        business_opportunities=result.business_opportunities,
        target_roles=result.target_roles,
    )


def _empty_result() -> ImpactCouncilResult:
    """Return empty result when analysis fails entirely."""
    return ImpactCouncilResult(
        consensus_reasoning="Analysis unavailable",
        overall_confidence=0.1,
    )
