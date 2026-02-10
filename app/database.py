"""
SQLite database setup and operations for storing leads.
"""

import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from .config import get_settings

Base = declarative_base()


class LeadModel(Base):
    """SQLAlchemy model for storing leads."""
    __tablename__ = "leads"
    
    id = Column(String(50), primary_key=True)
    trend_title = Column(String(500), nullable=False)
    trend_summary = Column(Text)
    trend_severity = Column(String(20))
    industries_affected = Column(Text)  # JSON array
    
    impact_positive_sectors = Column(Text)  # JSON array
    impact_negative_sectors = Column(Text)  # JSON array
    business_opportunities = Column(Text)  # JSON array
    
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


class PipelineRunModel(Base):
    """SQLAlchemy model for tracking pipeline runs."""
    __tablename__ = "pipeline_runs"
    
    id = Column(String(50), primary_key=True)
    status = Column(String(20), default="running")
    trends_detected = Column(Integer, default=0)
    companies_found = Column(Integer, default=0)
    contacts_found = Column(Integer, default=0)
    emails_found = Column(Integer, default=0)
    leads_generated = Column(Integer, default=0)
    output_file = Column(String(500))
    errors = Column(Text)  # JSON array
    run_time_seconds = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class Database:
    """Database manager for lead storage."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        settings = get_settings()
        # Use synchronous SQLite for simplicity
        url = database_url or settings.database_url
        # Convert async URL to sync for regular SQLAlchemy
        if "aiosqlite" in url:
            url = url.replace("sqlite+aiosqlite", "sqlite")
        
        self.engine = create_engine(url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def save_lead(self, lead_data: dict) -> str:
        """Save a lead to the database."""
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
    
    def save_pipeline_run(self, run_data: dict) -> str:
        """Save pipeline run metadata."""
        with self.get_session() as session:
            run = PipelineRunModel(
                id=run_data.get("id", ""),
                status=run_data.get("status", "running"),
                trends_detected=run_data.get("trends_detected", 0),
                companies_found=run_data.get("companies_found", 0),
                contacts_found=run_data.get("contacts_found", 0),
                emails_found=run_data.get("emails_found", 0),
                leads_generated=run_data.get("leads_generated", 0),
                output_file=run_data.get("output_file", ""),
                errors=json.dumps(run_data.get("errors", [])),
                run_time_seconds=run_data.get("run_time_seconds", 0),
                started_at=run_data.get("started_at", datetime.utcnow()),
                completed_at=run_data.get("completed_at"),
            )
            session.merge(run)
            return run.id
    
    def get_latest_leads(self, limit: int = 50) -> List[dict]:
        """Get the most recent leads."""
        with self.get_session() as session:
            leads = session.query(LeadModel).order_by(
                LeadModel.created_at.desc()
            ).limit(limit).all()
            
            result = []
            for lead in leads:
                result.append({
                    "id": lead.id,
                    "trend": lead.trend_title,
                    "company": lead.company_name,
                    "contact": lead.contact_name,
                    "role": lead.contact_role,
                    "email": lead.contact_email,
                    "email_confidence": lead.email_confidence,
                    "subject": lead.outreach_subject,
                    "created_at": lead.created_at.isoformat() if lead.created_at else None
                })
            return result
    
    def get_latest_run(self) -> Optional[dict]:
        """Get the most recent pipeline run."""
        with self.get_session() as session:
            run = session.query(PipelineRunModel).order_by(
                PipelineRunModel.started_at.desc()
            ).first()
            
            if not run:
                return None
            
            return {
                "id": run.id,
                "status": run.status,
                "trends_detected": run.trends_detected,
                "companies_found": run.companies_found,
                "emails_found": run.emails_found,
                "leads_generated": run.leads_generated,
                "output_file": run.output_file,
                "errors": json.loads(run.errors) if run.errors else [],
                "run_time_seconds": run.run_time_seconds,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None
            }


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
    return _db
