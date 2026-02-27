"""Leads API router -- returns crystallized lead sheets from pipeline runs.

Data source priority:
  1. In-memory RunManager (active/recent runs with full state)
  2. Database (persisted call sheets from completed runs)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.run_manager import run_manager
from app.api.schemas import LeadListResponse, LeadResponse
from app.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter()


def _leads_from_run(run) -> list:
    """Extract LeadResponse list from a PipelineRun (in-memory).

    Supports two data paths:
      1. Real runs: deps._lead_sheets (pydantic objects) in final_state
      2. Replay runs: run.result["leads"] (dicts from JSON recordings)
    """
    if run is None:
        return []

    # Path 1: replay result (dicts from recorded JSON)
    if run.result and isinstance(run.result, dict) and run.result.get("replay"):
        sheets = run.result.get("leads", [])
        # Build enrichment indexes from recording data
        _company_by_name: dict[str, dict] = {}
        for c in run.result.get("companies", []):
            _company_by_name[c.get("company_name", "")] = c
        _contacts_by_company: dict[str, list] = {}
        for ct in run.result.get("contacts", []):
            cname = ct.get("company_name", "")
            _contacts_by_company.setdefault(cname, []).append(ct)
        _outreach_by_company: dict[str, dict] = {}
        for em in run.result.get("outreach", []):
            cname = em.get("company_name", "")
            if cname not in _outreach_by_company:
                _outreach_by_company[cname] = em

        results = []
        for i, s in enumerate(sheets):
            cname = s.get("company_name", "")
            company = _company_by_name.get(cname, {})
            contacts = _contacts_by_company.get(cname, [])
            outreach = _outreach_by_company.get(cname, {})
            contact = next((c for c in contacts if c.get("email")), contacts[0] if contacts else {})
            results.append(LeadResponse(
                id=i,
                company_name=cname,
                company_cin=s.get("company_cin", ""),
                company_state=s.get("company_state", ""),
                company_city=s.get("company_city", ""),
                company_size_band=s.get("company_size_band", ""),
                company_website=company.get("website", ""),
                company_domain=company.get("domain", ""),
                reason_relevant=company.get("reason_relevant", ""),
                hop=s.get("hop", 1),
                lead_type=s.get("lead_type", ""),
                trend_title=s.get("trend_title", ""),
                event_type=s.get("event_type", ""),
                contact_name=contact.get("person_name", ""),
                contact_role=contact.get("role", "") or s.get("contact_role", ""),
                contact_email=contact.get("email", ""),
                contact_linkedin=contact.get("linkedin_url", ""),
                email_confidence=int(contact.get("email_confidence", 0)),
                email_subject=outreach.get("subject", ""),
                email_body=outreach.get("body", ""),
                trigger_event=s.get("trigger_event", ""),
                pain_point=s.get("pain_point", ""),
                service_pitch=s.get("service_pitch", ""),
                opening_line=s.get("opening_line", ""),
                urgency_weeks=s.get("urgency_weeks", 4),
                confidence=s.get("confidence", 0.0),
                oss_score=s.get("oss_score", 0.0),
                data_sources=s.get("data_sources", []),
                company_news=s.get("company_news", []),
            ))
        return results

    # Path 2: real run with deps._lead_sheets (pydantic objects)
    if run.final_state is None:
        return []
    deps = run.final_state.get("deps")
    if deps is None:
        return []

    # Build enrichment indexes (contacts, companies, emails from lead_gen)
    _company_by_name = {}
    for c in getattr(deps, "_companies", []):
        _company_by_name[getattr(c, "company_name", "")] = c
    _contacts_by_company = {}
    for ct in getattr(deps, "_contacts", []):
        cname = getattr(ct, "company_name", "")
        _contacts_by_company.setdefault(cname, []).append(ct)
    _outreach_by_company = {}
    for em in getattr(deps, "_outreach", []):
        cname = getattr(em, "company_name", "")
        if cname not in _outreach_by_company:
            _outreach_by_company[cname] = em

    sheets = getattr(deps, "_lead_sheets", [])
    results = []
    for i, sheet in enumerate(sheets):
        cname = getattr(sheet, "company_name", "")
        company = _company_by_name.get(cname)
        contacts = _contacts_by_company.get(cname, [])
        outreach = _outreach_by_company.get(cname)
        contact = next((c for c in contacts if getattr(c, "email", "")), contacts[0] if contacts else None)
        results.append(LeadResponse(
            id=i,
            company_name=cname,
            company_cin=getattr(sheet, "company_cin", ""),
            company_state=getattr(sheet, "company_state", ""),
            company_city=getattr(sheet, "company_city", ""),
            company_size_band=getattr(sheet, "company_size_band", ""),
            company_website=getattr(company, "website", "") if company else "",
            company_domain=getattr(company, "domain", "") if company else "",
            reason_relevant=getattr(company, "reason_relevant", "") if company else "",
            hop=getattr(sheet, "hop", 1),
            lead_type=getattr(sheet, "lead_type", ""),
            trend_title=getattr(sheet, "trend_title", ""),
            event_type=getattr(sheet, "event_type", ""),
            contact_name=getattr(contact, "person_name", "") if contact else "",
            contact_role=getattr(contact, "role", "") if contact else getattr(sheet, "contact_role", ""),
            contact_email=getattr(contact, "email", "") if contact else "",
            contact_linkedin=getattr(contact, "linkedin_url", "") if contact else "",
            email_confidence=getattr(contact, "email_confidence", 0) if contact else 0,
            email_subject=getattr(outreach, "subject", "") if outreach else "",
            email_body=getattr(outreach, "body", "") if outreach else "",
            trigger_event=getattr(sheet, "trigger_event", ""),
            pain_point=getattr(sheet, "pain_point", ""),
            service_pitch=getattr(sheet, "service_pitch", ""),
            opening_line=getattr(sheet, "opening_line", ""),
            urgency_weeks=getattr(sheet, "urgency_weeks", 4),
            confidence=getattr(sheet, "confidence", 0.0),
            oss_score=getattr(sheet, "oss_score", 0.0),
            data_sources=getattr(sheet, "data_sources", []),
            company_news=getattr(sheet, "company_news", []),
        ))
    return results


def _leads_from_db(
    run_id: Optional[str] = None,
    hop: Optional[int] = None,
    lead_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 100,
) -> list:
    """Query call sheets from DB (durable across restarts)."""
    try:
        db = get_database()
        rows = db.get_call_sheets(
            run_id=run_id,
            hop=hop,
            lead_type=lead_type,
            min_confidence=min_confidence,
            limit=limit,
        )
        return [LeadResponse(**row) for row in rows]
    except Exception as e:
        logger.warning(f"DB lead query failed: {e}")
        return []


@router.get("", response_model=LeadListResponse)
async def list_leads(
    run_id: str = Query(None, description="Filter by pipeline run"),
    hop: int = Query(None, ge=1, le=3, description="Filter by hop level"),
    lead_type: str = Query(None, description="pain|opportunity|risk|intelligence"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    """List leads — tries in-memory first, falls back to recordings, then DB."""
    # Try in-memory RunManager first (faster, has active runs)
    leads = []
    if run_id:
        run = run_manager.get_run(run_id)
        if run and (run.final_state or run.result):
            leads = _leads_from_run(run)
        # Fallback: load from recordings if not in memory
        if not leads and run_id:
            from app.api.pipeline import _load_run_from_recording
            from app.tools.run_recorder import get_recording
            recording_dir = get_recording(run_id)
            if recording_dir and (recording_dir / "manifest.json").exists():
                run = _load_run_from_recording(run_id, recording_dir)
                leads = _leads_from_run(run)
    else:
        run = run_manager.get_latest_run()
        if run and run.status == "completed":
            leads = _leads_from_run(run)
        # Fallback: load latest recording
        if not leads:
            from app.api.pipeline import _load_run_from_recording
            from app.tools.run_recorder import get_latest_recording
            recording_dir = get_latest_recording()
            if recording_dir and (recording_dir / "manifest.json").exists():
                run = _load_run_from_recording(recording_dir.name, recording_dir)
                leads = _leads_from_run(run)

    # Fall back to DB if in-memory has no data
    if not leads:
        leads = _leads_from_db(
            run_id=run_id,
            hop=hop,
            lead_type=lead_type,
            min_confidence=min_confidence if min_confidence > 0 else None,
            limit=limit + offset,
        )
    else:
        # Apply in-memory filters
        if hop is not None:
            leads = [l for l in leads if l.hop == hop]
        if lead_type:
            leads = [l for l in leads if l.lead_type == lead_type]
        if min_confidence > 0:
            leads = [l for l in leads if l.confidence >= min_confidence]
        leads.sort(key=lambda l: l.confidence, reverse=True)

    total = len(leads)
    leads = leads[offset:offset + limit]

    return LeadListResponse(total=total, leads=leads)


@router.get("/latest", response_model=LeadListResponse)
async def latest_leads(limit: int = Query(50)):
    """Get leads from most recent completed run."""
    # Try in-memory first
    run = run_manager.get_latest_run()
    if run and run.status == "completed":
        leads = _leads_from_run(run)
        leads.sort(key=lambda l: l.confidence, reverse=True)
        return LeadListResponse(total=len(leads), leads=leads[:limit])

    # Fall back to DB
    leads = _leads_from_db(limit=limit)
    return LeadListResponse(total=len(leads), leads=leads)


@router.get("/{lead_id}")
async def get_lead_detail(lead_id: int):
    """Single lead by ID — tries in-memory, then DB."""
    # Try in-memory
    run = run_manager.get_latest_run()
    leads = _leads_from_run(run)
    if 0 <= lead_id < len(leads):
        return leads[lead_id]

    # Try DB
    try:
        db = get_database()
        rows = db.get_call_sheets(limit=1)
        # For DB, lead_id is the actual row ID
        all_sheets = db.get_call_sheets(limit=200)
        for sheet in all_sheets:
            if sheet["id"] == lead_id:
                return LeadResponse(**sheet)
    except Exception:
        pass

    raise HTTPException(404, "Lead not found")
