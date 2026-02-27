"""
SQLite database — stores pipeline runs, call sheets (leads), and trends.

Tables:
  - pipeline_runs: Run history with status, counts, timing
  - call_sheets: Flat lead records from LeadCrystallizer (primary deliverable)
  - trends: Detected trends per run
  - leads: Legacy outreach-style leads (backward compat)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, DateTime, Boolean,
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


class LeadModel(Base):
    """Legacy outreach-style lead (backward compat with old pipeline)."""
    __tablename__ = "leads"

    id = Column(String(50), primary_key=True)
    trend_title = Column(String(500), nullable=False)
    trend_summary = Column(Text)
    trend_severity = Column(String(20))
    industries_affected = Column(Text)

    impact_positive_sectors = Column(Text)
    impact_negative_sectors = Column(Text)
    business_opportunities = Column(Text)

    company_name = Column(String(200), nullable=False)
    company_size = Column(String(20))
    company_industry = Column(String(100))
    company_website = Column(String(500))
    company_domain = Column(String(200))
    company_description = Column(Text)
    reason_relevant = Column(Text)

    contact_name = Column(String(200))
    contact_role = Column(String(200))
    contact_linkedin = Column(String(500))
    contact_email = Column(String(200))
    email_confidence = Column(Integer, default=0)
    email_source = Column(String(50))
    email_verified = Column(Boolean, default=False)

    outreach_subject = Column(String(500))
    outreach_body = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Database class ───────────────────────────────────────────────────────────

class Database:
    """Database manager — singleton, lazy-initialized."""

    def __init__(self, database_url: Optional[str] = None):
        settings = get_settings()
        url = database_url or settings.database_url
        if "aiosqlite" in url:
            url = url.replace("sqlite+aiosqlite", "sqlite")

        self.engine = create_engine(url, echo=False)
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
        for table_name, model in [("call_sheets", CallSheetModel)]:
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
                    "run_time_seconds": r.run_time_seconds,
                    "errors": json.loads(r.errors) if r.errors else [],
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in runs
            ]

    def get_latest_run(self) -> Optional[Dict]:
        runs = self.get_pipeline_runs(limit=1)
        return runs[0] if runs else None

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
        """Query call sheets with optional filters."""
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
            return [
                {
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
                    "contact_name": getattr(r, "contact_name", "") or "",
                    "contact_role": r.contact_role,
                    "contact_email": getattr(r, "contact_email", "") or "",
                    "contact_linkedin": getattr(r, "contact_linkedin", "") or "",
                    "email_confidence": getattr(r, "email_confidence", 0) or 0,
                    "email_subject": getattr(r, "email_subject", "") or "",
                    "email_body": getattr(r, "email_body", "") or "",
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
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

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

    def get_trends(
        self,
        run_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Query trends with optional run filter."""
        with self.get_session() as session:
            q = session.query(TrendModel)
            if run_id:
                q = q.filter(TrendModel.run_id == run_id)

            rows = q.order_by(TrendModel.oss_score.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "trend_id": r.trend_id,
                    "title": r.title,
                    "summary": r.summary,
                    "severity": r.severity,
                    "trend_type": r.trend_type,
                    "industries": json.loads(r.industries) if r.industries else [],
                    "keywords": json.loads(r.keywords) if r.keywords else [],
                    "trend_score": r.trend_score,
                    "actionability_score": r.actionability_score,
                    "oss_score": r.oss_score,
                    "article_count": r.article_count,
                    "event_5w1h": json.loads(r.event_5w1h) if r.event_5w1h else {},
                    "causal_chain": json.loads(r.causal_chain) if r.causal_chain else [],
                    "buying_intent": json.loads(r.buying_intent) if r.buying_intent else {},
                    "affected_companies": json.loads(r.affected_companies) if r.affected_companies else [],
                    "actionable_insight": r.actionable_insight,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    # ── Legacy leads (backward compat) ────────────────────────────────

    def save_lead(self, lead_data: dict) -> str:
        """Save a lead (old nested format)."""
        with self.get_session() as session:
            lead = LeadModel(
                id=lead_data.get("id", ""),
                trend_title=lead_data.get("trend", {}).get("trend_title", ""),
                trend_summary=lead_data.get("trend", {}).get("summary", ""),
                trend_severity=lead_data.get("trend", {}).get("severity", "medium"),
                industries_affected=json.dumps(lead_data.get("trend", {}).get("industries_affected", [])),
                impact_positive_sectors=json.dumps(lead_data.get("impact", {}).get("positive_sectors", [])),
                impact_negative_sectors=json.dumps(lead_data.get("impact", {}).get("negative_sectors", [])),
                business_opportunities=json.dumps(lead_data.get("impact", {}).get("business_opportunities", [])),
                company_name=lead_data.get("company", {}).get("company_name", ""),
                company_size=lead_data.get("company", {}).get("company_size", "mid"),
                company_industry=lead_data.get("company", {}).get("industry", ""),
                company_website=lead_data.get("company", {}).get("website", ""),
                company_domain=lead_data.get("company", {}).get("domain", ""),
                company_description=lead_data.get("company", {}).get("description", ""),
                reason_relevant=lead_data.get("company", {}).get("reason_relevant", ""),
                contact_name=lead_data.get("contact", {}).get("person_name", ""),
                contact_role=lead_data.get("contact", {}).get("role", ""),
                contact_linkedin=lead_data.get("contact", {}).get("linkedin_url", ""),
                contact_email=lead_data.get("contact", {}).get("email", ""),
                email_confidence=lead_data.get("contact", {}).get("email_confidence", 0),
                email_source=lead_data.get("contact", {}).get("email_source", ""),
                email_verified=lead_data.get("contact", {}).get("verified", False),
                outreach_subject=lead_data.get("outreach", {}).get("subject", ""),
                outreach_body=lead_data.get("outreach", {}).get("body", ""),
            )
            session.add(lead)
            return lead.id

    def get_latest_leads(self, limit: int = 50) -> List[dict]:
        """Get the most recent legacy leads."""
        with self.get_session() as session:
            leads = session.query(LeadModel).order_by(
                LeadModel.created_at.desc()
            ).limit(limit).all()
            return [
                {
                    "id": lead.id,
                    "trend": lead.trend_title,
                    "company": lead.company_name,
                    "contact": lead.contact_name,
                    "role": lead.contact_role,
                    "email": lead.contact_email,
                    "email_confidence": lead.email_confidence,
                    "subject": lead.outreach_subject,
                    "created_at": lead.created_at.isoformat() if lead.created_at else None,
                }
                for lead in leads
            ]


# ── Singleton ────────────────────────────────────────────────────────────────

_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
    return _db
