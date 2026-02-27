"""API response/request schemas -- designed for Next.js frontend consumption."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -- Pipeline --

class PipelineRunRequest(BaseModel):
    mock_mode: bool = False
    max_trends: Optional[int] = None
    replay_run_id: Optional[str] = None  # Replay a specific recording (mock_mode only)
    disabled_providers: List[str] = Field(default_factory=list)  # Provider names to skip


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str  # started | running | completed | failed
    message: str


class PipelineStatusResponse(BaseModel):
    run_id: str
    status: str
    current_step: str
    progress_pct: int
    trends_detected: int = 0
    companies_found: int = 0
    leads_generated: int = 0
    errors: List[str] = Field(default_factory=list)
    started_at: str
    elapsed_seconds: float


class PipelineResultResponse(BaseModel):
    run_id: str
    status: str
    trends_detected: int
    companies_found: int
    leads_generated: int
    run_time_seconds: float
    errors: List[str] = Field(default_factory=list)
    trends: List["TrendResponse"] = Field(default_factory=list)
    leads: List["LeadResponse"] = Field(default_factory=list)


# -- Trends --

class TrendResponse(BaseModel):
    id: str = ""
    title: str
    summary: str = ""
    severity: str = ""
    trend_type: str = ""
    industries: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    trend_score: float = 0.0
    actionability_score: float = 0.0
    oss_score: float = 0.0
    article_count: int = 0
    event_5w1h: Dict[str, str] = Field(default_factory=dict)
    causal_chain: List[str] = Field(default_factory=list)
    buying_intent: Dict[str, str] = Field(default_factory=dict)
    affected_companies: List[str] = Field(default_factory=list)
    actionable_insight: str = ""
    # Source articles (top evidence snippets from clustered articles)
    article_snippets: List[str] = Field(default_factory=list)
    source_links: List[str] = Field(default_factory=list)
    # Impact analysis fields (from ImpactAnalysis, joined by trend_title)
    direct_impact: List[str] = Field(default_factory=list)
    indirect_impact: List[str] = Field(default_factory=list)
    midsize_pain_points: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    pitch_angle: str = ""
    evidence_citations: List[str] = Field(default_factory=list)
    who_needs_help: str = ""
    council_confidence: float = 0.0


# -- People (per company) --

class PersonResponse(BaseModel):
    """Person at a company with reach score and outreach strategy."""
    person_name: str = ""
    role: str = ""
    seniority_tier: str = "influencer"
    linkedin_url: str = ""
    email: str = ""
    email_confidence: int = 0
    verified: bool = False
    reach_score: int = 0
    outreach_tone: str = "consultative"
    outreach_subject: str = ""
    outreach_body: str = ""


# -- Leads (call sheets) --

class LeadResponse(BaseModel):
    id: Optional[int] = None
    company_name: str = ""
    company_cin: str = ""
    company_state: str = ""
    company_city: str = ""
    company_size_band: str = ""
    company_website: str = ""
    company_domain: str = ""
    reason_relevant: str = ""
    hop: int = 1
    lead_type: str = ""
    trend_title: str = ""
    event_type: str = ""
    # Contact enrichment (from Apollo/Hunter via lead_gen)
    contact_name: str = ""
    contact_role: str = ""
    contact_email: str = ""
    contact_linkedin: str = ""
    email_confidence: int = 0
    # Personalized outreach (from email_agent)
    email_subject: str = ""
    email_body: str = ""
    # Sales content
    trigger_event: str = ""
    pain_point: str = ""
    service_pitch: str = ""
    opening_line: str = ""
    urgency_weeks: int = 4
    confidence: float = 0.0
    oss_score: float = 0.0
    data_sources: List[str] = Field(default_factory=list)
    company_news: List[Dict[str, Any]] = Field(default_factory=list)
    # People at this company (multiple contacts with reach scores)
    people: List["PersonResponse"] = Field(default_factory=list)


class LeadListResponse(BaseModel):
    total: int
    leads: List[LeadResponse]


# -- Feedback --

class FeedbackRequest(BaseModel):
    feedback_type: str  # "trend" | "lead"
    item_id: str
    rating: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    saved: bool
    record: Dict[str, Any]


class FeedbackSummaryResponse(BaseModel):
    total: int
    trends: Dict[str, int]
    leads: Dict[str, int]


# -- Health --

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    providers: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


# Resolve forward references so FastAPI serializes all fields correctly.
# Without this, List["LeadResponse"] / List["TrendResponse"] remain unresolved
# ForwardRefs and FastAPI's schema-based serialization silently drops new fields.
LeadResponse.model_rebuild()
PipelineResultResponse.model_rebuild()
LeadListResponse.model_rebuild()
