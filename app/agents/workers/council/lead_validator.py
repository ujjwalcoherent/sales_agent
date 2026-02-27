"""
Stage C: Lead Quality Validation Council.

Validates that company-trend pairings are genuinely relevant and that
pitch angles are defensible. Filters out weak leads before they enter
the outreach pipeline.
"""

import logging
from typing import Dict, Any, Optional

from .schemas import LeadValidation
from ....tools.llm_service import LLMService
from ....config import get_settings, CMI_SERVICES

logger = logging.getLogger(__name__)

LEAD_VALIDATION_PROMPT = """You are a senior sales intelligence analyst.
Evaluate whether this company is GENUINELY affected by this trend and whether
the proposed pitch is defensible.

Reject leads that are:
- Generic (any company could be "affected")
- Based on speculation rather than evidence
- Pitching services the company doesn't need

Accept leads that:
- Have a clear causal chain from trend to company impact
- Reference specific pain points this company would have
- Recommend services that directly address identified needs"""


async def validate_lead(
    company_name: str,
    company_description: str,
    company_industry: str,
    trend_title: str,
    trend_summary: str,
    proposed_pitch: str,
    proposed_service: str,
    llm_service: LLMService = None,
) -> LeadValidation:
    """
    Validate a single company-trend pairing.

    Returns validation with relevance score, improved pitch, and reasoning.
    """
    if llm_service is None:
        llm_service = LLMService(mock_mode=get_settings().mock_mode, lite=True)

    # Build services context
    services_text = ""
    for svc_name, svc_data in CMI_SERVICES.items():
        services_text += f"\n{svc_name}: {', '.join(svc_data.get('offerings', [])[:5])}"

    prompt = f"""Evaluate this lead:

COMPANY: {company_name}
INDUSTRY: {company_industry}
DESCRIPTION: {company_description[:300]}

TREND: {trend_title}
TREND SUMMARY: {trend_summary[:400]}

PROPOSED PITCH: {proposed_pitch}
PROPOSED SERVICE: {proposed_service}

OUR SERVICES: {services_text}

Evaluate this lead and respond as JSON with these exact keys:
- relevance_score: float 0.0 to 1.0 (how relevant is this lead?)
- is_relevant: boolean (should we pursue this lead?)
- reasoning: string (2-3 sentence explanation of your assessment)
- improved_pitch: string (a better pitch angle if you have one, empty string if current pitch is good)
- recommended_service: string (best CMI service key for this lead)
- recommended_offering: string (specific offering within that service)"""

    from ...schemas.llm_outputs import LeadValidationLLM

    def _build_validation(result: LeadValidationLLM) -> LeadValidation:
        return LeadValidation(
            company_name=company_name,
            trend_title=trend_title,
            relevance_score=min(1.0, max(0.0, result.relevance_score)),
            is_relevant=result.is_relevant,
            reasoning=result.reasoning,
            improved_pitch=result.improved_pitch,
            recommended_service=result.recommended_service,
            recommended_offering=result.recommended_offering,
        )

    # Track A: structured output (works with Gemini, may fail on NVIDIA)
    try:
        result = await llm_service.run_structured(
            prompt=prompt,
            system_prompt=LEAD_VALIDATION_PROMPT,
            output_type=LeadValidationLLM,
        )
        return _build_validation(result)
    except Exception as e:
        logger.warning(f"Structured lead validation failed for {company_name}: {e}")

    # Track B: generate_json() — works with NVIDIA DeepSeek (no grammar validation)
    try:
        raw = await llm_service.generate_json(
            prompt=prompt,
            system_prompt=LEAD_VALIDATION_PROMPT,
        )
        if isinstance(raw, dict) and "error" not in raw:
            result = LeadValidationLLM(**{
                k: v for k, v in raw.items()
                if k in LeadValidationLLM.model_fields
            })
            logger.info(f"Lead validation via generate_json for {company_name}: relevant={result.is_relevant}, score={result.relevance_score:.2f}")
            return _build_validation(result)
        else:
            logger.warning(f"generate_json returned error for {company_name}: {raw}")
    except Exception as e2:
        logger.warning(f"generate_json lead validation also failed for {company_name}: {e2}")

    # Track C: default — keep lead with low confidence
    return LeadValidation(
        company_name=company_name,
        trend_title=trend_title,
        relevance_score=0.5,
        is_relevant=True,
        reasoning="Validation unavailable — both structured and unstructured LLM calls failed",
    )
