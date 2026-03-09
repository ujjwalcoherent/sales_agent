"""
SQLite database — stores pipeline runs, call sheets (leads), contacts, trends, and companies.

Tables:
  - pipeline_runs: Run history with status, counts, timing
  - call_sheets: Flat lead records from LeadCrystallizer (primary deliverable)
  - lead_contacts: Multiple contacts per lead with reach scores
  - trends: Detected trends per run
  - saved_companies: Company KB with enrichment from Wikidata/Wikipedia/Apollo
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, DateTime, Boolean,
    event as sa_event,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from .config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# ── Models ───────────────────────────────────────────────────────────────────

class PipelineRunModel(Base):
    """Pipeline run history."""
    __tablename__ = "pipeline_runs"

    id = Column(String(50), primary_key=True)
    status = Column(String(20), default="running")
    mock_mode = Column(Boolean, default=False)
    trends_detected = Column(Integer, default=0)
    companies_found = Column(Integer, default=0)
    leads_generated = Column(Integer, default=0)
    contacts_found = Column(Integer, default=0)
    emails_found = Column(Integer, default=0)
    output_file = Column(String(500))
    errors = Column(Text)  # JSON array
    run_time_seconds = Column(Float, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class CallSheetModel(Base):
    """Call sheet (lead) from LeadCrystallizer — flat schema matching LeadSheet."""
    __tablename__ = "call_sheets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, index=True)

    # Company
    company_name = Column(String(300), nullable=False)
    company_cin = Column(String(50))
    company_state = Column(String(100))
    company_city = Column(String(100))
    company_size_band = Column(String(30))

    # Lead context
    hop = Column(Integer, default=1)
    lead_type = Column(String(30))  # pain | opportunity | risk | intelligence
    trend_title = Column(String(500))
    event_type = Column(String(50))

    # Company enrichment (from lead_gen)
    company_website = Column(String(500), default="")
    company_domain = Column(String(300), default="")
    reason_relevant = Column(Text, default="")

    # Contact enrichment (from Apollo/Hunter via lead_gen)
    contact_name = Column(String(300), default="")
    contact_role = Column(String(200))
    contact_email = Column(String(300), default="")
    contact_linkedin = Column(String(500), default="")
    email_confidence = Column(Integer, default=0)

    # Personalized outreach (from email_agent)
    email_subject = Column(String(500), default="")
    email_body = Column(Text, default="")

    # Sales content
    trigger_event = Column(Text)
    pain_point = Column(Text)
    service_pitch = Column(Text)
    opening_line = Column(Text)
    urgency_weeks = Column(Integer, default=4)
    confidence = Column(Float, default=0.0)

    # Metadata
    reasoning = Column(Text)
    data_sources = Column(Text)  # JSON array
    company_news = Column(Text, default="[]")  # JSON array of {title, url, date}
    oss_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class LeadContactModel(Base):
    """Multiple contacts per lead — person profiles with reach scores."""
    __tablename__ = "lead_contacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, index=True)
    company_name = Column(String(300), nullable=False, index=True)
    person_name = Column(String(300), nullable=False)
    role = Column(String(200))
    seniority_tier = Column(String(30), default="influencer")
    linkedin_url = Column(String(500), default="")
    email = Column(String(300), default="")
    email_confidence = Column(Integer, default=0)
    email_source = Column(String(30), default="")
    verified = Column(Boolean, default=False)
    reach_score = Column(Integer, default=0)
    outreach_tone = Column(String(30), default="consultative")
    outreach_subject = Column(String(500), default="")
    outreach_body = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class TrendModel(Base):
    """Detected trend per run."""
    __tablename__ = "trends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, index=True)
    trend_id = Column(String(100))

    title = Column(String(500), nullable=False)
    summary = Column(Text)
    severity = Column(String(20))
    trend_type = Column(String(50))
    industries = Column(Text)  # JSON array
    keywords = Column(Text)  # JSON array
    trend_score = Column(Float, default=0.0)
    actionability_score = Column(Float, default=0.0)
    oss_score = Column(Float, default=0.0)
    article_count = Column(Integer, default=0)
    event_5w1h = Column(Text)  # JSON object
    causal_chain = Column(Text)  # JSON array
    buying_intent = Column(Text)  # JSON object
    affected_companies = Column(Text)  # JSON array
    actionable_insight = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class SavedCompanyModel(Base):
    """Searched company — auto-saved on search, enriched on generate-leads.

    Also serves as the Company Knowledge Base (KB): cached enrichment from
    Wikidata/Wikipedia/Apollo. Fields refresh when stale (> max_age_days).
    """
    __tablename__ = "saved_companies"

    id = Column(String(50), primary_key=True)  # MD5 hash from search
    company_name = Column(String(300), nullable=False)
    domain = Column(String(300), default="")
    website = Column(String(500), default="")
    industry = Column(String(200), default="")
    # Enrichment fields (from Wikidata/Wikipedia/Apollo)
    description = Column(Text, default="")
    headquarters = Column(String(200), default="")
    employee_count = Column(String(50), default="")
    founded_year = Column(Integer)
    stock_ticker = Column(String(20), default="")
    ceo = Column(String(200), default="")
    funding_stage = Column(String(50), default="")
    wikidata_id = Column(String(20), default="")
    # Extended enrichment (from company_enricher / web_intel)
    sub_industries = Column(Text, default="[]")       # JSON array
    key_people = Column(Text, default="[]")            # JSON array of dicts
    products_services = Column(Text, default="[]")     # JSON array
    competitors = Column(Text, default="[]")           # JSON array
    revenue = Column(String(100), default="")
    total_funding = Column(String(100), default="")
    investors = Column(Text, default="[]")             # JSON array
    tech_stack = Column(Text, default="[]")            # JSON array
    validation_source = Column(String(100), default="")
    # Metadata
    reason_relevant = Column(Text, default="")
    article_count = Column(Integer, default=0)
    recent_articles = Column(Text, default="[]")   # JSON array
    live_news = Column(Text, default="[]")          # JSON array
    search_query = Column(String(500), default="")
    search_type = Column(String(30), default="company")
    contacts = Column(Text, default="[]")           # JSON array of PersonResponse dicts
    contacts_reasoning = Column(Text, default="")
    contacts_generated_at = Column(DateTime)
    last_searched_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserProfileModel(Base):
    """User profile — JSON blob storing industry targets, account list, products, etc."""
    __tablename__ = "user_profiles"

    profile_id = Column(String(100), primary_key=True)
    user_name = Column(String(200), default="")
    profile_json = Column(Text, default="{}")   # Full UserProfile JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class CampaignModel(Base):
    """Campaign — a batch of companies to process through enrichment + contacts + outreach."""
    __tablename__ = "campaigns"

    id = Column(String(50), primary_key=True)
    name = Column(String(200), default="")
    campaign_type = Column(String(30), default="company_first")
    status = Column(String(20), default="draft")
    config_json = Column(Text, default="{}")
    companies_json = Column(Text, default="[]")  # List of CampaignCompanyStatus dicts
    total_companies = Column(Integer, default=0)
    completed_companies = Column(Integer, default=0)
    total_contacts = Column(Integer, default=0)
    total_outreach = Column(Integer, default=0)
    error = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


# ── Database class ───────────────────────────────────────────────────────────

class Database:
    """Database manager — singleton, lazy-initialized."""

    def __init__(self, database_url: Optional[str] = None):
        settings = get_settings()
        url = database_url or settings.database_url
        if "aiosqlite" in url:
            url = url.replace("sqlite+aiosqlite", "sqlite")

        self.engine = create_engine(url, echo=False)

        # Enable WAL mode for concurrent reads during pipeline writes
        @sa_event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables and add any missing columns (safe to call multiple times).

        SQLAlchemy's create_all only creates NEW tables — it won't add columns
        to existing ones. For SQLite we use ALTER TABLE ADD COLUMN for each
        missing column (SQLite doesn't support DROP/ALTER COLUMN, but ADD is fine).
        """
        Base.metadata.create_all(self.engine)
        self._migrate_columns()

    def _migrate_columns(self):
        """Add any missing columns to existing tables (SQLite-compatible)."""
        from sqlalchemy import inspect as sa_inspect, text
        inspector = sa_inspect(self.engine)
        for table_name, model in [("call_sheets", CallSheetModel), ("saved_companies", SavedCompanyModel), ("campaigns", CampaignModel)]:
            if not inspector.has_table(table_name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table_name)}
            for col in model.__table__.columns:
                if col.name not in existing:
                    col_type = col.type.compile(self.engine.dialect)
                    default = "''" if "VARCHAR" in str(col_type) or "TEXT" in str(col_type) else "0"
                    sql = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type} DEFAULT {default}"
                    with self.engine.begin() as conn:
                        conn.execute(text(sql))
                    logger.info(f"Migration: added column {table_name}.{col.name} ({col_type})")

    @contextmanager
    def get_session(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Pipeline Runs ─────────────────────────────────────────────────

    def save_pipeline_run(self, run_data: Dict[str, Any]) -> str:
        """Save a pipeline run record."""
        with self.get_session() as session:
            run = PipelineRunModel(
                id=run_data["run_id"],
                status=run_data.get("status", "completed"),
                mock_mode=run_data.get("mock_mode", False),
                trends_detected=run_data.get("trends_detected", 0),
                companies_found=run_data.get("companies_found", 0),
                leads_generated=run_data.get("leads_generated", 0),
                contacts_found=run_data.get("contacts_found", 0),
                output_file=run_data.get("output_file", ""),
                errors=json.dumps(run_data.get("errors", [])),
                run_time_seconds=run_data.get("run_time_seconds", 0),
                started_at=run_data.get("started_at"),
                completed_at=run_data.get("completed_at"),
            )
            session.merge(run)  # merge = upsert
            return run.id

    def update_pipeline_run(self, run_id: str, updates: Dict[str, Any]):
        """Update specific fields of an existing pipeline run."""
        with self.get_session() as session:
            run = session.query(PipelineRunModel).filter_by(id=run_id).first()
            if run:
                for key, value in updates.items():
                    if hasattr(run, key):
                        setattr(run, key, value)

    def get_pipeline_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent pipeline runs."""
        with self.get_session() as session:
            runs = session.query(PipelineRunModel).order_by(
                PipelineRunModel.started_at.desc()
            ).limit(limit).all()
            return [
                {
                    "run_id": r.id,
                    "status": r.status,
                    "mock_mode": r.mock_mode,
                    "trends_detected": r.trends_detected,
                    "companies_found": r.companies_found,
                    "leads_generated": r.leads_generated,
                    "contacts_found": r.contacts_found,
                    "emails_found": r.emails_found,
                    "run_time_seconds": r.run_time_seconds,
                    "errors": json.loads(r.errors) if r.errors else [],
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in runs
            ]

    # ── Call Sheets (primary leads) ───────────────────────────────────

    def save_call_sheet(self, run_id: str, sheet, enrichment: dict = None) -> int:
        """Save a LeadSheet as a call_sheet row with optional enrichment. Returns the row ID.

        enrichment dict can contain: company_website, company_domain, reason_relevant,
        contact_name, contact_email, contact_linkedin, email_confidence,
        email_subject, email_body, company_news.
        """
        enr = enrichment or {}
        with self.get_session() as session:
            row = CallSheetModel(
                run_id=run_id,
                company_name=getattr(sheet, "company_name", ""),
                company_cin=getattr(sheet, "company_cin", ""),
                company_state=getattr(sheet, "company_state", ""),
                company_city=getattr(sheet, "company_city", ""),
                company_size_band=getattr(sheet, "company_size_band", ""),
                company_website=enr.get("company_website", ""),
                company_domain=enr.get("company_domain", ""),
                reason_relevant=enr.get("reason_relevant", ""),
                hop=getattr(sheet, "hop", 1),
                lead_type=getattr(sheet, "lead_type", ""),
                trend_title=getattr(sheet, "trend_title", ""),
                event_type=getattr(sheet, "event_type", ""),
                contact_name=enr.get("contact_name", ""),
                contact_role=enr.get("contact_role", "") or getattr(sheet, "contact_role", ""),
                contact_email=enr.get("contact_email", ""),
                contact_linkedin=enr.get("contact_linkedin", ""),
                email_confidence=enr.get("email_confidence", 0),
                email_subject=enr.get("email_subject", ""),
                email_body=enr.get("email_body", ""),
                trigger_event=getattr(sheet, "trigger_event", ""),
                pain_point=getattr(sheet, "pain_point", ""),
                service_pitch=getattr(sheet, "service_pitch", ""),
                opening_line=getattr(sheet, "opening_line", ""),
                urgency_weeks=getattr(sheet, "urgency_weeks", 4),
                confidence=getattr(sheet, "confidence", 0.0),
                reasoning=getattr(sheet, "reasoning", ""),
                data_sources=json.dumps(getattr(sheet, "data_sources", [])),
                company_news=json.dumps(enr.get("company_news", getattr(sheet, "company_news", []))),
                oss_score=getattr(sheet, "oss_score", 0.0),
            )
            session.add(row)
            session.flush()
            return row.id

    def save_lead_contacts(self, run_id: str, profiles: list) -> int:
        """Save person profiles for a run. Returns count saved."""
        saved = 0
        with self.get_session() as session:
            for p in profiles:
                row = LeadContactModel(
                    run_id=run_id,
                    company_name=getattr(p, "company_name", ""),
                    person_name=getattr(p, "person_name", ""),
                    role=getattr(p, "role", ""),
                    seniority_tier=getattr(p, "seniority_tier", "influencer"),
                    linkedin_url=getattr(p, "linkedin_url", ""),
                    email=getattr(p, "email", ""),
                    email_confidence=getattr(p, "email_confidence", 0),
                    email_source=getattr(p, "email_source", ""),
                    verified=getattr(p, "verified", False),
                    reach_score=getattr(p, "reach_score", 0),
                    outreach_tone=getattr(p, "outreach_tone", "consultative"),
                    outreach_subject=getattr(p, "outreach_subject", ""),
                    outreach_body=getattr(p, "outreach_body", ""),
                )
                session.add(row)
                saved += 1
        return saved

    def get_call_sheets(
        self,
        run_id: Optional[str] = None,
        limit: int = 100,
        hop: Optional[int] = None,
        lead_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Dict]:
        """Query call sheets with optional filters.

        Joins lead_contacts table to populate the people[] array for each lead.
        Also backfills primary contact/email from people[0] if call_sheet fields are empty.
        """
        with self.get_session() as session:
            q = session.query(CallSheetModel)
            if run_id:
                q = q.filter(CallSheetModel.run_id == run_id)
            if hop is not None:
                q = q.filter(CallSheetModel.hop == hop)
            if lead_type:
                q = q.filter(CallSheetModel.lead_type == lead_type)
            if min_confidence is not None:
                q = q.filter(CallSheetModel.confidence >= min_confidence)

            rows = q.order_by(CallSheetModel.confidence.desc()).limit(limit).all()
            if not rows:
                return []

            # Batch-load lead_contacts for all matching run_ids
            run_ids = list({r.run_id for r in rows})
            contacts = session.query(LeadContactModel).filter(
                LeadContactModel.run_id.in_(run_ids)
            ).all()

            # Group contacts by (run_id, company_name)
            contacts_map: Dict[tuple, list] = {}
            for c in contacts:
                key = (c.run_id, c.company_name)
                contacts_map.setdefault(key, []).append(c)

            results = []
            for r in rows:
                people_rows = contacts_map.get((r.run_id, r.company_name), [])
                people = [
                    {
                        "person_name": p.person_name or "",
                        "role": p.role or "",
                        "seniority_tier": p.seniority_tier or "influencer",
                        "linkedin_url": p.linkedin_url or "",
                        "email": p.email or "",
                        "email_confidence": p.email_confidence or 0,
                        "verified": p.verified or False,
                        "reach_score": p.reach_score or 0,
                        "outreach_tone": p.outreach_tone or "consultative",
                        "outreach_subject": p.outreach_subject or "",
                        "outreach_body": p.outreach_body or "",
                    }
                    for p in people_rows
                ]

                # Primary contact/email fields — backfill from people[0] if empty
                contact_name = getattr(r, "contact_name", "") or ""
                contact_email = getattr(r, "contact_email", "") or ""
                contact_role = r.contact_role or ""
                contact_linkedin = getattr(r, "contact_linkedin", "") or ""
                email_confidence = getattr(r, "email_confidence", 0) or 0
                email_subject = getattr(r, "email_subject", "") or ""
                email_body = getattr(r, "email_body", "") or ""

                if people:
                    p0 = people[0]
                    if not contact_name:
                        contact_name = p0["person_name"]
                    if not contact_email:
                        contact_email = p0["email"]
                    if not contact_role:
                        contact_role = p0["role"]
                    if not contact_linkedin:
                        contact_linkedin = p0["linkedin_url"]
                    if not email_confidence:
                        email_confidence = p0["email_confidence"]
                    if not email_subject:
                        email_subject = p0["outreach_subject"]
                    if not email_body:
                        email_body = p0["outreach_body"]

                results.append({
                    "id": r.id,
                    "run_id": r.run_id,
                    "company_name": r.company_name,
                    "company_cin": r.company_cin,
                    "company_state": r.company_state,
                    "company_city": r.company_city,
                    "company_size_band": r.company_size_band,
                    "company_website": getattr(r, "company_website", "") or "",
                    "company_domain": getattr(r, "company_domain", "") or "",
                    "reason_relevant": getattr(r, "reason_relevant", "") or "",
                    "hop": r.hop,
                    "lead_type": r.lead_type,
                    "trend_title": r.trend_title,
                    "event_type": r.event_type,
                    "contact_name": contact_name,
                    "contact_role": contact_role,
                    "contact_email": contact_email,
                    "contact_linkedin": contact_linkedin,
                    "email_confidence": email_confidence,
                    "email_subject": email_subject,
                    "email_body": email_body,
                    "trigger_event": r.trigger_event,
                    "pain_point": r.pain_point,
                    "service_pitch": r.service_pitch,
                    "opening_line": r.opening_line,
                    "urgency_weeks": r.urgency_weeks,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "data_sources": json.loads(r.data_sources) if r.data_sources else [],
                    "company_news": json.loads(getattr(r, "company_news", "[]") or "[]"),
                    "oss_score": r.oss_score,
                    "people": people,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                })
            return results

    # ── Trends ────────────────────────────────────────────────────────

    def save_trend(self, run_id: str, trend) -> int:
        """Save a TrendData as a trend row."""
        with self.get_session() as session:
            row = TrendModel(
                run_id=run_id,
                trend_id=getattr(trend, "id", ""),
                title=getattr(trend, "trend_title", ""),
                summary=getattr(trend, "summary", ""),
                severity=str(getattr(trend, "severity", "")),
                trend_type=getattr(trend, "trend_type", ""),
                industries=json.dumps(getattr(trend, "industries_affected", [])),
                keywords=json.dumps(getattr(trend, "keywords", [])),
                trend_score=getattr(trend, "trend_score", 0.0),
                actionability_score=getattr(trend, "actionability_score", 0.0),
                oss_score=getattr(trend, "oss_score", 0.0),
                article_count=getattr(trend, "article_count", 0),
                event_5w1h=json.dumps(getattr(trend, "event_5w1h", {})),
                causal_chain=json.dumps(getattr(trend, "causal_chain", [])),
                buying_intent=json.dumps(getattr(trend, "buying_intent", {})),
                affected_companies=json.dumps(getattr(trend, "affected_companies", [])),
                actionable_insight=getattr(trend, "actionable_insight", ""),
            )
            session.add(row)
            session.flush()
            return row.id

    # ── Saved Companies ─────────────────────────────────────────────

    @staticmethod
    def _safe_json_dumps(val) -> str:
        """Serialize to JSON string — prevents double-encoding."""
        if isinstance(val, str):
            # Already a string — check if it's valid JSON, return as-is
            try:
                json.loads(val)
                return val  # Already JSON string
            except (json.JSONDecodeError, TypeError):
                return json.dumps(val)
        return json.dumps(val if val is not None else [])

    def save_company(self, company_data: Dict[str, Any]) -> str:
        """Upsert a searched company. Returns the company ID.

        Smart merge: only updates fields that are non-empty in new data.
        Never overwrites enriched fields with empty values.
        Deduplicates by company_name (case-insensitive) — if a company with the
        same name exists under a different ID, merges into the existing record.
        """
        cid = company_data.get("id", "")
        # Validate required fields
        name = (company_data.get("company_name") or "").strip()
        if not name or len(name) < 2:
            logger.warning(f"save_company: rejected empty/short name: {name!r}")
            return cid
        with self.get_session() as session:
            # First: try exact ID match
            existing = session.query(SavedCompanyModel).filter_by(id=cid).first()
            # Second: if no ID match, check for same company name (dedup)
            if not existing:
                from sqlalchemy import func
                existing = session.query(SavedCompanyModel).filter(
                    func.lower(SavedCompanyModel.company_name) == name.lower()
                ).first()
                if existing:
                    # Found by name — use existing ID for consistency
                    cid = existing.id
            if existing:
                # Update with fresh data — never overwrite non-empty with empty
                existing.domain = company_data.get("domain", existing.domain) or existing.domain
                existing.website = company_data.get("website", existing.website) or existing.website
                existing.industry = company_data.get("industry", existing.industry) or existing.industry
                existing.description = company_data.get("description", existing.description) or existing.description
                existing.headquarters = company_data.get("headquarters", existing.headquarters) or existing.headquarters
                existing.employee_count = company_data.get("employee_count", existing.employee_count) or existing.employee_count
                existing.stock_ticker = company_data.get("stock_ticker", existing.stock_ticker) or existing.stock_ticker
                existing.ceo = company_data.get("ceo", existing.ceo) or existing.ceo
                existing.funding_stage = company_data.get("funding_stage", existing.funding_stage) or existing.funding_stage
                existing.wikidata_id = company_data.get("wikidata_id", existing.wikidata_id) or existing.wikidata_id
                if company_data.get("founded_year"):
                    existing.founded_year = company_data["founded_year"]
                _sjd = self._safe_json_dumps
                if company_data.get("sub_industries"):
                    existing.sub_industries = _sjd(company_data["sub_industries"])
                if company_data.get("key_people"):
                    existing.key_people = _sjd(company_data["key_people"])
                if company_data.get("products_services"):
                    existing.products_services = _sjd(company_data["products_services"])
                if company_data.get("competitors"):
                    existing.competitors = _sjd(company_data["competitors"])
                existing.revenue = company_data.get("revenue", existing.revenue) or existing.revenue
                existing.total_funding = company_data.get("total_funding", existing.total_funding) or existing.total_funding
                if company_data.get("investors"):
                    existing.investors = _sjd(company_data["investors"])
                if company_data.get("tech_stack"):
                    existing.tech_stack = _sjd(company_data["tech_stack"])
                existing.validation_source = company_data.get("validation_source", existing.validation_source) or existing.validation_source
                existing.reason_relevant = company_data.get("reason_relevant", existing.reason_relevant) or existing.reason_relevant
                existing.article_count = company_data.get("article_count", existing.article_count)
                if "recent_articles" in company_data:
                    existing.recent_articles = _sjd(company_data["recent_articles"])
                if "live_news" in company_data:
                    existing.live_news = _sjd(company_data["live_news"])
                existing.search_query = company_data.get("search_query", existing.search_query)
                existing.search_type = company_data.get("search_type", existing.search_type)
                existing.last_searched_at = datetime.utcnow()
            else:
                row = SavedCompanyModel(
                    id=cid,
                    company_name=company_data.get("company_name", ""),
                    domain=company_data.get("domain", ""),
                    website=company_data.get("website", ""),
                    industry=company_data.get("industry", ""),
                    description=company_data.get("description", ""),
                    headquarters=company_data.get("headquarters", ""),
                    employee_count=company_data.get("employee_count", ""),
                    founded_year=company_data.get("founded_year"),
                    stock_ticker=company_data.get("stock_ticker", ""),
                    ceo=company_data.get("ceo", ""),
                    funding_stage=company_data.get("funding_stage", ""),
                    wikidata_id=company_data.get("wikidata_id", ""),
                    sub_industries=self._safe_json_dumps(company_data.get("sub_industries", [])),
                    key_people=self._safe_json_dumps(company_data.get("key_people", [])),
                    products_services=self._safe_json_dumps(company_data.get("products_services", [])),
                    competitors=self._safe_json_dumps(company_data.get("competitors", [])),
                    revenue=company_data.get("revenue", ""),
                    total_funding=company_data.get("total_funding", ""),
                    investors=self._safe_json_dumps(company_data.get("investors", [])),
                    tech_stack=self._safe_json_dumps(company_data.get("tech_stack", [])),
                    validation_source=company_data.get("validation_source", ""),
                    reason_relevant=company_data.get("reason_relevant", ""),
                    article_count=company_data.get("article_count", 0),
                    recent_articles=self._safe_json_dumps(company_data.get("recent_articles", [])),
                    live_news=self._safe_json_dumps(company_data.get("live_news", [])),
                    search_query=company_data.get("search_query", ""),
                    search_type=company_data.get("search_type", "company"),
                )
                session.add(row)
            return cid

    def save_company_contacts(self, company_id: str, contacts: list, reasoning: str = "") -> int:
        """Save generated leads for a saved company. Returns count saved."""
        with self.get_session() as session:
            row = session.query(SavedCompanyModel).filter_by(id=company_id).first()
            if not row:
                return 0
            row.contacts = json.dumps(contacts)
            row.contacts_reasoning = reasoning
            row.contacts_generated_at = datetime.utcnow()
            return len(contacts)

    def get_saved_company(self, company_id: str) -> Optional[Dict]:
        """Get a single saved company by ID."""
        with self.get_session() as session:
            r = session.query(SavedCompanyModel).filter_by(id=company_id).first()
            if not r:
                return None
            return self._company_to_dict(r)

    def get_saved_companies(self, limit: int = 50) -> List[Dict]:
        """Get all saved companies, most recently searched first."""
        with self.get_session() as session:
            rows = session.query(SavedCompanyModel).order_by(
                SavedCompanyModel.last_searched_at.desc()
            ).limit(limit).all()
            return [self._company_to_dict(r) for r in rows]

    @staticmethod
    def _safe_json_loads(val, default=None):
        """Safe JSON parse — handles double-encoded JSON, empty strings, and already-parsed values."""
        if default is None:
            default = []
        if not val:
            return default
        # Already a list/dict — no parsing needed
        if isinstance(val, (list, dict)):
            return val
        try:
            result = json.loads(val)
            # Handle double-encoded: json.loads returns a string that's itself JSON
            if isinstance(result, str):
                if not result:
                    return default  # '""' → '' → use default
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return default  # Non-JSON string → use default
            return result
        except (json.JSONDecodeError, TypeError):
            return default

    @staticmethod
    def _company_to_dict(r) -> Dict:
        _sjl = Database._safe_json_loads
        return {
            "id": r.id,
            "company_name": r.company_name,
            "domain": r.domain or "",
            "website": r.website or "",
            "industry": r.industry or "",
            "description": getattr(r, "description", "") or "",
            "headquarters": getattr(r, "headquarters", "") or "",
            "employee_count": getattr(r, "employee_count", "") or "",
            "founded_year": getattr(r, "founded_year", None),
            "stock_ticker": getattr(r, "stock_ticker", "") or "",
            "ceo": getattr(r, "ceo", "") or "",
            "funding_stage": getattr(r, "funding_stage", "") or "",
            "wikidata_id": getattr(r, "wikidata_id", "") or "",
            "sub_industries": _sjl(getattr(r, "sub_industries", "[]")),
            "key_people": _sjl(getattr(r, "key_people", "[]")),
            "products_services": _sjl(getattr(r, "products_services", "[]")),
            "competitors": _sjl(getattr(r, "competitors", "[]")),
            "revenue": getattr(r, "revenue", "") or "",
            "total_funding": getattr(r, "total_funding", "") or "",
            "investors": _sjl(getattr(r, "investors", "[]")),
            "tech_stack": _sjl(getattr(r, "tech_stack", "[]")),
            "validation_source": getattr(r, "validation_source", "") or "",
            "reason_relevant": r.reason_relevant or "",
            "article_count": r.article_count or 0,
            "recent_articles": _sjl(r.recent_articles),
            "live_news": _sjl(r.live_news),
            "search_query": r.search_query or "",
            "search_type": r.search_type or "company",
            "contacts": _sjl(r.contacts),
            "contacts_reasoning": r.contacts_reasoning or "",
            "contacts_generated_at": r.contacts_generated_at.isoformat() if r.contacts_generated_at else None,
            "last_searched_at": r.last_searched_at.isoformat() if r.last_searched_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }

    # ── Company Knowledge Base (KB) ────────────────────────────────

    def get_or_enrich_company(self, company_name: str, max_age_days: int = 7) -> Optional[Dict]:
        """Check KB for recent company data. Returns cached profile if fresh.

        Returns None if not found or stale — caller should run enrichment.
        Prefers rows with enrichment data when multiple case variants exist.
        """
        with self.get_session() as session:
            rows = session.query(SavedCompanyModel).filter(
                SavedCompanyModel.company_name.ilike(company_name)
            ).all()
            # Prefer rows with enrichment data
            best = None
            for row in rows:
                if not row.last_searched_at:
                    continue
                age = (datetime.utcnow() - row.last_searched_at).days
                if age > max_age_days:
                    continue
                has_enrichment = bool(getattr(row, "description", "") or getattr(row, "headquarters", ""))
                if has_enrichment:
                    return self._company_to_dict(row)  # Return enriched row immediately
                if best is None:
                    best = row
            if best:
                return self._company_to_dict(best)
        return None

    def upsert_company_profile(self, company_name: str, profile: Dict) -> str:
        """Insert or update company in KB from enrichment profile.

        Smart merge: only updates fields that are non-empty in new data.
        """
        import hashlib
        cid = hashlib.md5(company_name.lower().encode()).hexdigest()[:12]
        profile["id"] = cid
        profile["company_name"] = company_name
        return self.save_company(profile)

    # ── User Profile CRUD ─────────────────────────────────────────────────

    def save_profile(self, profile_data: Dict[str, Any]) -> str:
        """Upsert a user profile. Returns profile_id."""
        pid = profile_data.get("profile_id", "")
        if not pid:
            raise ValueError("profile_data must have a profile_id")
        with self.get_session() as session:
            existing = session.query(UserProfileModel).filter_by(profile_id=pid).first()
            if existing:
                existing.user_name = profile_data.get("user_name", existing.user_name)
                existing.profile_json = json.dumps(profile_data)
                existing.updated_at = datetime.utcnow()
            else:
                session.add(UserProfileModel(
                    profile_id=pid,
                    user_name=profile_data.get("user_name", ""),
                    profile_json=json.dumps(profile_data),
                ))
        return pid

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a profile by ID. Returns None if not found."""
        with self.get_session() as session:
            row = session.query(UserProfileModel).filter_by(profile_id=profile_id).first()
            if not row:
                return None
            return json.loads(row.profile_json)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all profiles (summary fields only — no full JSON)."""
        with self.get_session() as session:
            rows = session.query(UserProfileModel).order_by(
                UserProfileModel.updated_at.desc()
            ).all()
            results = []
            for row in rows:
                try:
                    data = json.loads(row.profile_json)
                except (json.JSONDecodeError, TypeError):
                    data = {}
                results.append({
                    "profile_id": row.profile_id,
                    "user_name": row.user_name,
                    "region": data.get("region", "global"),
                    "target_industries": data.get("target_industries", []),
                    "account_list_count": len(data.get("account_list", [])),
                    "own_products_count": len(data.get("own_products", [])),
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })
            return results

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile. Returns True if deleted, False if not found."""
        with self.get_session() as session:
            row = session.query(UserProfileModel).filter_by(profile_id=profile_id).first()
            if not row:
                return False
            session.delete(row)
            return True

    # ── Campaign CRUD ──────────────────────────────────────────────────────

    def create_campaign(self, campaign_data: Dict[str, Any]) -> str:
        """Create a new campaign. Returns campaign ID."""
        with self.get_session() as session:
            campaign = CampaignModel(
                id=campaign_data["id"],
                name=campaign_data.get("name", ""),
                campaign_type=campaign_data.get("campaign_type", "company_first"),
                status=campaign_data.get("status", "draft"),
                config_json=json.dumps(campaign_data.get("config", {})),
                companies_json=json.dumps(campaign_data.get("companies", [])),
                total_companies=campaign_data.get("total_companies", 0),
            )
            session.add(campaign)
            return campaign.id

    def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> bool:
        """Update campaign fields. Returns True if found and updated."""
        with self.get_session() as session:
            campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
            if not campaign:
                return False
            for key, value in updates.items():
                if key == "companies":
                    campaign.companies_json = json.dumps(value)
                elif key == "config":
                    campaign.config_json = json.dumps(value)
                elif hasattr(campaign, key):
                    setattr(campaign, key, value)
            return True

    def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get a single campaign by ID."""
        with self.get_session() as session:
            c = session.query(CampaignModel).filter_by(id=campaign_id).first()
            if not c:
                return None
            return {
                "id": c.id,
                "name": c.name,
                "campaign_type": c.campaign_type,
                "status": c.status,
                "config": json.loads(c.config_json or "{}"),
                "companies": json.loads(c.companies_json or "[]"),
                "total_companies": c.total_companies,
                "completed_companies": c.completed_companies,
                "total_contacts": c.total_contacts,
                "total_outreach": c.total_outreach,
                "error": c.error or "",
                "created_at": c.created_at.isoformat() if c.created_at else "",
                "completed_at": c.completed_at.isoformat() if c.completed_at else "",
            }

    def list_campaigns(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all campaigns, newest first."""
        with self.get_session() as session:
            rows = session.query(CampaignModel).order_by(
                CampaignModel.created_at.desc()
            ).limit(limit).all()
            return [
                {
                    "id": c.id,
                    "name": c.name,
                    "campaign_type": c.campaign_type,
                    "status": c.status,
                    "total_companies": c.total_companies,
                    "completed_companies": c.completed_companies,
                    "total_contacts": c.total_contacts,
                    "total_outreach": c.total_outreach,
                    "error": c.error or "",
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                    "completed_at": c.completed_at.isoformat() if c.completed_at else "",
                    "companies": json.loads(c.companies_json or "[]"),
                }
                for c in rows
            ]

    def delete_campaign(self, campaign_id: str) -> bool:
        """Delete a campaign. Returns True if found and deleted."""
        with self.get_session() as session:
            campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
            if not campaign:
                return False
            session.delete(campaign)
            return True


# ── Singleton ────────────────────────────────────────────────────────────────

_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
    return _db
