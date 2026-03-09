"""Campaign system schemas — 3 discovery paths.

Campaign types:
  - company_first: User provides company names → enrich → contacts → outreach
  - industry_first: User picks industry → discover companies → enrich → contacts → outreach
  - report_driven: User pastes report text → LLM extracts companies → company_first flow
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CampaignType(str, Enum):
    COMPANY_FIRST = "company_first"
    INDUSTRY_FIRST = "industry_first"
    REPORT_DRIVEN = "report_driven"


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CampaignCompanyInput(BaseModel):
    """A company to include in a campaign."""
    company_name: str
    domain: str = ""
    industry: str = ""
    context: str = ""  # News trigger, report mention, why this company


class CampaignConfig(BaseModel):
    """Campaign execution configuration."""
    max_companies: int = 10
    max_contacts_per_company: int = 3
    generate_outreach: bool = True
    target_roles: list[str] = Field(default_factory=list)
    country: str = ""
    background_deep: bool = False  # Run ScrapeGraphAI deep enrichment


class CreateCampaignRequest(BaseModel):
    """Request to create a new campaign."""
    name: str = ""
    campaign_type: CampaignType = CampaignType.COMPANY_FIRST
    companies: list[CampaignCompanyInput] = Field(default_factory=list)  # For company_first
    industry: str = ""  # For industry_first
    report_text: str = ""  # For report_driven
    config: CampaignConfig = Field(default_factory=CampaignConfig)


class CampaignContact(BaseModel):
    """A contact found during campaign execution."""
    full_name: str = ""
    role: str = ""
    email: str = ""
    linkedin_url: str = ""
    seniority: str = ""
    email_confidence: float = 0.0


class CampaignEmail(BaseModel):
    """A generated outreach email."""
    recipient_name: str = ""
    recipient_role: str = ""
    subject: str = ""
    body: str = ""


class CampaignCompanyStatus(BaseModel):
    """Status of a single company within a campaign."""
    company_name: str
    status: str = "pending"  # pending | enriching | contacts | outreach | done | failed
    domain: str = ""
    industry: str = ""
    description: str = ""
    contacts_found: int = 0
    outreach_generated: int = 0
    contacts: list[CampaignContact] = Field(default_factory=list)
    emails: list[CampaignEmail] = Field(default_factory=list)
    error: str = ""


class UpdateCampaignRequest(BaseModel):
    """Request to update a draft/failed campaign before running."""
    name: Optional[str] = None
    companies: Optional[list[CampaignCompanyInput]] = None
    industry: Optional[str] = None
    report_text: Optional[str] = None
    config: Optional[CampaignConfig] = None


class CampaignResponse(BaseModel):
    """Full campaign state returned by API."""
    id: str
    name: str
    campaign_type: str
    status: str
    companies: list[CampaignCompanyStatus] = Field(default_factory=list)
    total_companies: int = 0
    completed_companies: int = 0
    total_contacts: int = 0
    total_outreach: int = 0
    created_at: str = ""
    completed_at: str = ""
    error: str = ""


class CampaignListResponse(BaseModel):
    """List of campaigns."""
    campaigns: list[CampaignResponse] = Field(default_factory=list)
    total: int = 0
