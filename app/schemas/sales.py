"""
Sales pipeline data models.

These models represent the downstream sales outreach pipeline:
impact analysis, company intelligence, contact information, and emails.

Hierarchy: MajorTrend → SectorImpact → CompanyData → ContactData → OutreachEmail

V1 VALIDATION: All models now include field validators that catch/coerce
bad LLM output. Lists coerce None/strings, URLs are validated, enums
are checked. Every coercion is logged for observability.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator
from uuid import UUID, uuid4

from .base import (
    Severity, CompanySize, Sector, ServiceType, ImpactType,
)

logger = logging.getLogger(__name__)

# Generic names that LLMs hallucinate as company names
_GENERIC_COMPANY_NAMES = frozenset({
    "company", "unknown", "n/a", "na", "none", "tbd", "various",
    "multiple", "several", "others", "etc", "general", "industry",
})

_EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def _coerce_to_str_list(v: Any, field_name: str) -> List[str]:
    """Coerce value to List[str]: None→[], str→[str], filter empties."""
    if v is None:
        return []
    if isinstance(v, str):
        v = v.strip()
        if v:
            logger.debug(f"Coerced {field_name} from string to [string]")
            return [v]
        return []
    if isinstance(v, list):
        result = []
        for item in v:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                result.append(s)
        return result
    # Iterable fallback
    if hasattr(v, '__iter__'):
        logger.debug(f"Coerced {field_name} from {type(v).__name__} to list")
        return [str(item).strip() for item in v if item is not None and str(item).strip()]
    logger.warning(f"Cannot coerce {field_name} ({type(v).__name__}) to list, wrapping")
    return [str(v)]


class SectorImpact(BaseModel):
    """How a trend impacts a specific sector."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Parent reference
    trend_id: UUID
    trend_title: str

    # Sector identity
    sector: Sector

    # Impact classification
    impact_type: ImpactType = ImpactType.NEUTRAL
    impact_severity: Severity = Severity.MEDIUM
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)

    # Impact analysis
    impact_summary: str = ""
    direct_effects: List[str] = Field(default_factory=list)
    indirect_effects: List[str] = Field(default_factory=list)

    # Challenges & opportunities
    challenges: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)

    # Consulting angle
    pain_points: List[str] = Field(default_factory=list)
    urgent_needs: List[str] = Field(default_factory=list)

    # Target personas
    target_roles: List[str] = Field(default_factory=list)

    # Recommended services
    recommended_services: List[ServiceType] = Field(default_factory=list)

    # Pitch elements
    pitch_angle: str = ""
    pitch_hooks: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class TrendData(BaseModel):
    """Market trend detected from RSS/Tavily.

    V2: Extended with synthesis-derived fields so LLM output flows to
    downstream agents (Impact, Lead Gen) instead of being discarded.
    """
    id: str = Field(default="")
    trend_title: str
    summary: str
    severity: Severity = Severity.MEDIUM
    industries_affected: List[str] = Field(default_factory=list)
    source_links: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Synthesis-derived fields (V2 — flow-through from TrendNode) ──
    trend_type: str = "general"
    actionable_insight: str = ""
    event_5w1h: Dict[str, str] = Field(default_factory=dict)
    causal_chain: List[str] = Field(default_factory=list)
    buying_intent: Dict[str, str] = Field(default_factory=dict)
    affected_companies: List[str] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    trend_score: float = 0.0
    actionability_score: float = 0.0
    oss_score: float = 0.0
    article_count: int = 0
    article_snippets: List[str] = Field(default_factory=list)  # Top-5 raw excerpts for impact council

    @field_validator('industries_affected', 'source_links', 'keywords',
                     'causal_chain', 'affected_companies', 'affected_regions',
                     mode='before')
    @classmethod
    def coerce_str_lists(cls, v, info):
        return _coerce_to_str_list(v, info.field_name)

    class Config:
        use_enum_values = True


class ImpactAnalysis(BaseModel):
    """Deep mid-size company focused impact analysis.

    V2: Now powered by AI Council (4 specialist agents + moderator).
    All list fields validate/coerce LLM output. Model validator warns
    when ALL impact lists are empty (pipeline will produce thin results).
    """
    trend_id: str
    trend_title: str

    # Part 1: Direct Impact on Mid-Size Companies
    direct_impact: List[str] = Field(default_factory=list)
    direct_impact_reasoning: str = ""

    # Part 2: Indirect Impact
    indirect_impact: List[str] = Field(default_factory=list)
    indirect_impact_reasoning: str = ""

    # Part 3: Additional Industry Verticals
    additional_verticals: List[str] = Field(default_factory=list)
    additional_verticals_reasoning: str = ""

    # Mid-size company specific
    midsize_pain_points: List[str] = Field(default_factory=list)
    consulting_projects: List[str] = Field(default_factory=list)

    # Consulting opportunities
    positive_sectors: List[str] = Field(default_factory=list)
    negative_sectors: List[str] = Field(default_factory=list)
    business_opportunities: List[str] = Field(default_factory=list)
    relevant_services: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    pitch_angle: str = ""

    # ── V2: Council-powered fields ─────────────────────────────────
    # Multi-paragraph detailed reasoning (replaces single-line reasoning)
    detailed_reasoning: str = ""
    # Each specialist agent's perspective
    council_perspectives: List[Dict[str, Any]] = Field(default_factory=list)
    # Key disagreements between agents and resolution
    debate_summary: str = ""
    # Source article references backing claims
    evidence_citations: List[str] = Field(default_factory=list)
    # Specific CMI service + offering recommendations with justification
    service_recommendations: List[Dict[str, str]] = Field(default_factory=list)
    # Council's confidence in this analysis
    council_confidence: float = 0.0
    # Phase 3B: who_needs_help from synthesis buying_intent — used for targeted company search
    who_needs_help: str = ""

    @field_validator(
        'direct_impact', 'indirect_impact', 'additional_verticals',
        'midsize_pain_points', 'consulting_projects', 'positive_sectors',
        'negative_sectors', 'business_opportunities', 'relevant_services',
        'target_roles', 'evidence_citations',
        mode='before',
    )
    @classmethod
    def coerce_list_fields(cls, v, info):
        return _coerce_to_str_list(v, info.field_name)

    @field_validator(
        'direct_impact_reasoning', 'indirect_impact_reasoning',
        'additional_verticals_reasoning', 'pitch_angle',
        'detailed_reasoning', 'debate_summary',
        mode='before',
    )
    @classmethod
    def coerce_str_fields(cls, v, info):
        if v is None:
            return ""
        return str(v)

    @field_validator('pitch_angle', mode='after')
    @classmethod
    def allow_longer_pitch(cls, v):
        # V2: Allow longer pitch angles (up to 500 chars) for service-specific detail
        if len(v) > 500:
            logger.debug(f"Truncated pitch_angle from {len(v)} to 500 chars")
            return v[:497] + "..."
        return v

    @model_validator(mode='after')
    def warn_empty_impacts(self):
        if (not self.direct_impact
                and not self.indirect_impact
                and not self.additional_verticals):
            logger.warning(
                f"ImpactAnalysis for '{self.trend_title[:40]}' has no impact "
                f"data (all impact lists empty)"
            )
        return self


class CompanyData(BaseModel):
    """Company information found via search.

    V1: Validates company_name (rejects generic names), website (URL format),
    domain (tldextract validation). NER verification fields for anti-hallucination.
    """
    id: str = Field(default="")
    company_name: str
    company_size: CompanySize = CompanySize.MID
    industry: str
    website: str = ""
    domain: str = ""
    description: str = ""
    reason_relevant: str = ""
    trend_id: str = ""
    # V7: NER verification fields
    ner_verified: bool = False
    verification_source: str = ""  # "ner_match", "wikipedia", "web_search"
    verification_confidence: float = 0.0

    @field_validator('company_name', mode='before')
    @classmethod
    def validate_company_name(cls, v):
        if v is None:
            raise ValueError("company_name cannot be None")
        v = str(v).strip()
        if len(v) < 2:
            raise ValueError(f"company_name too short: '{v}'")
        if v.lower() in _GENERIC_COMPANY_NAMES:
            raise ValueError(f"company_name is generic/placeholder: '{v}'")
        return v

    @field_validator('website', mode='before')
    @classmethod
    def validate_website(cls, v):
        if not v:
            return ""
        v = str(v).strip()
        if not v:
            return ""
        # Prepend https:// if missing scheme
        if not v.startswith(('http://', 'https://')):
            v = f"https://{v}"
        try:
            parsed = urlparse(v)
            if not parsed.netloc or '.' not in parsed.netloc:
                logger.debug(f"Invalid website URL '{v}', clearing")
                return ""
            return v
        except Exception:
            logger.debug(f"Failed to parse website URL '{v}', clearing")
            return ""

    @field_validator('domain', mode='before')
    @classmethod
    def validate_domain(cls, v):
        if not v:
            return ""
        v = str(v).strip()
        if not v or '.' not in v:
            return ""
        return v

    class Config:
        use_enum_values = True


class ContactData(BaseModel):
    """Decision-maker contact information."""
    id: str = Field(default="")
    company_id: str = ""
    company_name: str
    person_name: str
    role: str
    linkedin_url: str = ""
    email: str = ""
    email_confidence: int = 0
    email_source: str = ""  # "apollo", "hunter", "pattern"
    verified: bool = False

    @field_validator('email', mode='before')
    @classmethod
    def validate_email_format(cls, v):
        if not v:
            return ""
        v = str(v).strip()
        if not v:
            return ""
        if not _EMAIL_RE.match(v):
            logger.debug(f"Invalid email format '{v}', clearing")
            return ""
        return v


class OutreachEmail(BaseModel):
    """Generated personalized email.

    V1: Subject max 80 chars (truncated), body 30-300 words (truncated),
    email format validated.
    """
    id: str = Field(default="")
    contact_id: str = ""
    trend_title: str
    company_name: str
    person_name: str
    role: str
    email: str
    subject: str
    body: str
    email_confidence: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('subject', mode='before')
    @classmethod
    def validate_subject(cls, v):
        if v is None:
            return ""
        v = str(v).strip()
        if len(v) > 80:
            logger.debug(f"Truncated email subject from {len(v)} to 80 chars")
            v = v[:77] + "..."
        return v

    @field_validator('body', mode='before')
    @classmethod
    def validate_body(cls, v):
        if v is None:
            return ""
        v = str(v).strip()
        words = v.split()
        if len(words) > 300:
            logger.debug(f"Truncated email body from {len(words)} to 300 words")
            v = " ".join(words[:300])
        if len(words) < 20 and len(words) > 0:
            logger.debug(f"Email body is short ({len(words)} words)")
        return v

    @field_validator('email', mode='before')
    @classmethod
    def validate_email_format(cls, v):
        if not v:
            return ""
        v = str(v).strip()
        if not _EMAIL_RE.match(v):
            logger.debug(f"Invalid email format in OutreachEmail: '{v}'")
            return ""
        return v


class LeadRecord(BaseModel):
    """Complete lead record combining all data."""
    trend: TrendData
    impact: ImpactAnalysis
    company: CompanyData
    contact: ContactData
    outreach: OutreachEmail
    lead_score: float = Field(default=0.0, ge=0.0, le=100.0)

    def compute_score(self) -> float:
        """
        Compute composite lead score (0-100) from all signals.

        Weights:
          - Trend severity (25%): Higher severity = more urgent need
          - Company fit (25%): Mid-size + clear intent signal = best target
          - Email confidence (25%): Verified email = can actually reach them
          - Impact depth (25%): More pain points + direct impact = stronger pitch

        Returns: Score 0-100, stored in self.lead_score
        """
        score = 0.0

        # Trend severity (0-25)
        severity_scores = {"high": 25, "medium": 18, "low": 10, "negligible": 5}
        sev = self.trend.severity
        sev_str = sev.value if hasattr(sev, 'value') else str(sev)
        score += severity_scores.get(sev_str.lower(), 10)

        # Company fit (0-25)
        size = self.company.company_size
        size_str = size.value if hasattr(size, 'value') else str(size)
        size_scores = {"mid": 25, "startup": 18, "small": 15, "large_enterprise": 10}
        score += size_scores.get(size_str.lower(), 12)
        # Bonus for intent signal in reason
        if self.company.reason_relevant and len(self.company.reason_relevant) > 50:
            score += 5  # Detailed reason = clear intent
        # V7: NER verification bonus
        if self.company.ner_verified:
            score += 5

        # Email confidence (0-25)
        conf = self.contact.email_confidence
        score += min(conf / 4, 25)  # 100% confidence = 25 points

        # Impact depth (0-25)
        pain_count = len(self.impact.midsize_pain_points)
        direct_count = len(self.impact.direct_impact)
        impact_pts = min(pain_count * 4 + direct_count * 3, 25)
        score += impact_pts

        self.lead_score = min(round(score, 1), 100.0)
        return self.lead_score


class PipelineResult(BaseModel):
    """Result of running the full pipeline."""
    status: str = "success"
    leads_generated: int = 0
    trends_detected: int = 0
    companies_found: int = 0
    emails_found: int = 0
    output_file: str = ""
    errors: List[str] = Field(default_factory=list)
    run_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmailFinderResult(BaseModel):
    """Result from email finding APIs."""
    email: str = ""
    confidence: int = 0
    source: str = ""  # "apollo", "hunter", "pattern"
    verified: bool = False
    error: str = ""


class AgentState(BaseModel):
    """State passed between agents in the pipeline."""
    trends: List[TrendData] = Field(default_factory=list)
    impacts: List[ImpactAnalysis] = Field(default_factory=list)
    companies: List[CompanyData] = Field(default_factory=list)
    contacts: List[ContactData] = Field(default_factory=list)
    outreach_emails: List[OutreachEmail] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    current_step: str = "init"
    # I1: Cross-trend compound impact synthesis
    cross_trend_insight: Optional[Dict[str, Any]] = None
