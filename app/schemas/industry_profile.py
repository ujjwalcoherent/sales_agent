"""
User Profile and Industry Profile schemas.

These drive the 3-path pipeline (Industry-First, Company-First, Report-Driven):
  - UserProfile: who the user is, what they sell, what industries they target
  - IndustryTarget: which industry + order to classify against
  - ProductEntry: the user's own products (for email personalization)
  - ContactHierarchy: role targeting by company size + event type

Used by:
  - app/api/profiles.py (CRUD endpoints)
  - app/intelligence/industry_classifier.py (builds IndustrySpec from targets)
  - app/agents/workers/email_agent.py (personalization context)
  - app/learning/threshold_adapter.py (industry-aware adaptation)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ProductEntry(BaseModel):
    """One product or service the user sells."""
    name: str
    value_prop: str                          # What pain it solves in one sentence
    case_studies: List[str] = Field(default_factory=list)   # Named company + result
    target_roles: List[str] = Field(default_factory=list)   # ["VP Operations", "CTO"]
    relevant_event_types: List[str] = Field(default_factory=list)  # ["expansion", "funding"]


class IndustryTarget(BaseModel):
    """One industry the user wants to find leads in."""
    industry_id: str                         # "healthcare_pharma", "fintech_bfsi", or custom
    display_name: str = ""                   # Human label, auto-set from industry_id if empty
    order: str = "both"                      # "1st", "2nd", or "both"
    first_order_description: str = ""        # If custom: "pharma manufacturers, hospital chains"
    second_order_description: str = ""       # If custom: "CROs, cold-chain vendors"
    use_builtin: bool = True                 # If False, use the descriptions above


class ContactHierarchyEntry(BaseModel):
    """Role priority list for a given event type + company size."""
    event_type: str                          # "funding", "expansion", "product_launch", etc.
    company_size: str                        # "smb", "mid_market", "enterprise"
    role_priority: List[str]                 # Ordered list: most contactable → final fallback


class EmailConfig(BaseModel):
    """Sender's email configuration."""
    from_name: str = ""
    from_email: str = ""
    # SMTP delivery handled by Brevo integration (brevo_tool.py) — no credentials stored here


class UserProfile(BaseModel):
    """Complete user profile driving all 3 pipeline paths.

    Stored in SQLite via /api/v1/profiles endpoints.
    Loaded at pipeline start — every pipeline decision references this.
    """
    profile_id: str
    user_name: str
    own_company: str = ""
    region: str = "global"                   # "IN", "US", "SEA", "global"

    # What the user sells
    own_products: List[ProductEntry] = Field(default_factory=list)

    # Industry-First (Path 1)
    target_industries: List[IndustryTarget] = Field(default_factory=list)

    # Company-First (Path 2) — account list
    account_list: List[str] = Field(default_factory=list)   # Company names to track

    # Report-Driven (Path 3)
    report_title: str = ""
    report_summary: str = ""                 # Paste abstract or key findings

    # Contact targeting
    contact_hierarchy: List[ContactHierarchyEntry] = Field(default_factory=list)
    min_lead_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # Email config
    email_config: EmailConfig = Field(default_factory=EmailConfig)

    # Self-learning signals (updated by learning loops — not user-editable)
    path_preference: str = "auto"            # "auto" | "industry" | "account" | "report"
    total_runs: int = 0
    total_emails_sent: int = 0
    total_replies: int = 0


# ── Default contact hierarchy (extended by user profile) ─────────────────────

DEFAULT_CONTACT_HIERARCHY: List[ContactHierarchyEntry] = [
    ContactHierarchyEntry(
        event_type="funding",
        company_size="enterprise",
        role_priority=["VP Finance", "Director of FP&A", "Head of Procurement", "CFO"],
    ),
    ContactHierarchyEntry(
        event_type="funding",
        company_size="mid_market",
        role_priority=["CFO", "VP Finance", "CEO"],
    ),
    ContactHierarchyEntry(
        event_type="funding",
        company_size="smb",
        role_priority=["CEO", "Founder", "CFO"],
    ),
    ContactHierarchyEntry(
        event_type="expansion",
        company_size="enterprise",
        role_priority=["VP Operations", "Director of Expansion", "COO", "Head of Real Estate"],
    ),
    ContactHierarchyEntry(
        event_type="expansion",
        company_size="mid_market",
        role_priority=["COO", "VP Operations", "CEO"],
    ),
    ContactHierarchyEntry(
        event_type="expansion",
        company_size="smb",
        role_priority=["CEO", "Founder", "COO"],
    ),
    ContactHierarchyEntry(
        event_type="technology_adoption",
        company_size="enterprise",
        role_priority=["VP Engineering", "Director of Digital", "Head of IT", "CTO"],
    ),
    ContactHierarchyEntry(
        event_type="technology_adoption",
        company_size="mid_market",
        role_priority=["CTO", "VP Engineering", "Head of Technology"],
    ),
    ContactHierarchyEntry(
        event_type="technology_adoption",
        company_size="smb",
        role_priority=["CEO", "CTO", "Founder"],
    ),
    ContactHierarchyEntry(
        event_type="product_launch",
        company_size="enterprise",
        role_priority=["VP Marketing", "CMO", "Director of Product Marketing"],
    ),
    ContactHierarchyEntry(
        event_type="product_launch",
        company_size="mid_market",
        role_priority=["CMO", "VP Marketing", "CEO"],
    ),
    ContactHierarchyEntry(
        event_type="product_launch",
        company_size="smb",
        role_priority=["CEO", "CMO", "Founder"],
    ),
    ContactHierarchyEntry(
        event_type="regulation_compliance",
        company_size="enterprise",
        role_priority=["Head of Compliance", "CLO", "VP Legal", "CISO"],
    ),
    ContactHierarchyEntry(
        event_type="regulation_compliance",
        company_size="mid_market",
        role_priority=["CLO", "Head of Compliance", "CEO"],
    ),
    ContactHierarchyEntry(
        event_type="regulation_compliance",
        company_size="smb",
        role_priority=["CEO", "Founder", "Legal Counsel"],
    ),
]


def get_role_priority(
    event_type: str,
    company_size: str,
    profile: Optional[UserProfile] = None,
) -> List[str]:
    """Return ordered list of roles to target for a given event + company size.

    Merges profile's custom hierarchy with defaults (profile takes priority).
    Falls back gracefully if no match found.
    """
    # Check profile-level overrides first
    if profile and profile.contact_hierarchy:
        for entry in profile.contact_hierarchy:
            if entry.event_type == event_type and entry.company_size == company_size:
                return entry.role_priority

    # Fall back to defaults
    for entry in DEFAULT_CONTACT_HIERARCHY:
        if entry.event_type == event_type and entry.company_size == company_size:
            return entry.role_priority

    # Generic fallback by size
    size_fallback = {
        "enterprise": ["VP " + event_type.replace("_", " ").title(), "Director", "C-Suite"],
        "mid_market": ["VP", "Director", "CEO"],
        "smb": ["CEO", "Founder", "Co-Founder"],
    }
    return size_fallback.get(company_size, ["CEO", "Founder"])
