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
        return [
            LeadResponse(
                id=i,
                company_name=s.get("company_name", ""),
                company_cin=s.get("company_cin", ""),
                company_state=s.get("company_state", ""),
                company_city=s.get("company_city", ""),
                company_size_band=s.get("company_size_band", ""),
                hop=s.get("hop", 1),
                lead_type=s.get("lead_type", ""),
                trend_title=s.get("trend_title", ""),
                event_type=s.get("event_type", ""),
                contact_role=s.get("contact_role", ""),
                trigger_event=s.get("trigger_event", ""),
                pain_point=s.get("pain_point", ""),
                service_pitch=s.get("service_pitch", ""),
                opening_line=s.get("opening_line", ""),
                urgency_weeks=s.get("urgency_weeks", 4),
                confidence=s.get("confidence", 0.0),
                oss_score=s.get("oss_score", 0.0),
                data_sources=s.get("data_sources", []),
            )
            for i, s in enumerate(sheets)
        ]

    # Path 2: real run with deps._lead_sheets (pydantic objects)
    if run.final_state is None:
        return []
    deps = run.final_state.get("deps")
    if deps is None:
        return []

    sheets = getattr(deps, "_lead_sheets", [])
    return [
        LeadResponse(
            id=i,
            company_name=getattr(sheet, "company_name", ""),
            company_cin=getattr(sheet, "company_cin", ""),
            company_state=getattr(sheet, "company_state", ""),
            company_city=getattr(sheet, "company_city", ""),
            company_size_band=getattr(sheet, "company_size_band", ""),
            hop=getattr(sheet, "hop", 1),
            lead_type=getattr(sheet, "lead_type", ""),
            trend_title=getattr(sheet, "trend_title", ""),
            event_type=getattr(sheet, "event_type", ""),
            contact_role=getattr(sheet, "contact_role", ""),
            trigger_event=getattr(sheet, "trigger_event", ""),
            pain_point=getattr(sheet, "pain_point", ""),
            service_pitch=getattr(sheet, "service_pitch", ""),
            opening_line=getattr(sheet, "opening_line", ""),
            urgency_weeks=getattr(sheet, "urgency_weeks", 4),
            confidence=getattr(sheet, "confidence", 0.0),
            oss_score=getattr(sheet, "oss_score", 0.0),
            data_sources=getattr(sheet, "data_sources", []),
        )
        for i, sheet in enumerate(sheets)
    ]


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
    """List leads — tries in-memory first, falls back to DB."""
    # Try in-memory RunManager first (faster, has active runs)
    leads = []
    if run_id:
        run = run_manager.get_run(run_id)
        if run and (run.final_state or run.result):
            leads = _leads_from_run(run)
    else:
        run = run_manager.get_latest_run()
        if run and run.status == "completed":
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
