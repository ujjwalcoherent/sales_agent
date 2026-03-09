"""Campaign API router — create, list, run, stream, delete campaigns.

Campaigns orchestrate company enrichment, contact finding, and outreach
generation for lists of companies.  Three discovery paths: company_first,
industry_first, report_driven.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.database import get_database
from app.schemas.campaign import (
    CampaignType,
    CampaignCompanyInput,
    CampaignCompanyStatus,
    CampaignConfig,
    CampaignListResponse,
    CampaignResponse,
    CreateCampaignRequest,
    UpdateCampaignRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# SSE queues keyed by campaign_id — populated on /run, consumed on /stream
_active_queues: dict[str, asyncio.Queue] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _db_to_response(row: dict) -> CampaignResponse:
    companies_raw = row.get("companies") or []
    companies = [
        CampaignCompanyStatus(**c) if isinstance(c, dict) else c
        for c in companies_raw
    ]
    return CampaignResponse(
        id=row["id"],
        name=row.get("name", ""),
        campaign_type=row.get("campaign_type", "company_first"),
        status=row.get("status", "draft"),
        companies=companies,
        total_companies=row.get("total_companies", 0),
        completed_companies=row.get("completed_companies", 0),
        total_contacts=row.get("total_contacts", 0),
        total_outreach=row.get("total_outreach", 0),
        created_at=row.get("created_at", ""),
        completed_at=row.get("completed_at", ""),
        error=row.get("error", ""),
    )


def _auto_name(req: CreateCampaignRequest) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if req.campaign_type == CampaignType.COMPANY_FIRST and req.companies:
        names = [c.company_name for c in req.companies[:3]]
        label = ", ".join(names)
        if len(req.companies) > 3:
            label += f" +{len(req.companies) - 3}"
        return f"{label} ({date_str})"
    if req.campaign_type == CampaignType.INDUSTRY_FIRST and req.industry:
        return f"{req.industry} ({date_str})"
    return f"Campaign {date_str}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=CampaignResponse)
async def create_campaign(req: CreateCampaignRequest):
    """Create a new campaign."""
    # Validate inputs per campaign type
    if req.campaign_type == CampaignType.COMPANY_FIRST and not req.companies:
        raise HTTPException(400, "company_first campaigns require a non-empty companies list")
    if req.campaign_type == CampaignType.INDUSTRY_FIRST and not req.industry.strip():
        raise HTTPException(400, "industry_first campaigns require an industry")
    if req.campaign_type == CampaignType.REPORT_DRIVEN and not req.report_text.strip():
        raise HTTPException(400, "report_driven campaigns require report_text")

    cid = f"camp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    name = req.name.strip() or _auto_name(req)

    companies_list = [
        {
            "company_name": c.company_name,
            "status": "pending",
            "domain": c.domain,
            "industry": c.industry,
            "context": c.context,
        }
        for c in req.companies
    ]

    # Merge top-level fields into config for the executor to find
    config_dict = req.config.model_dump()
    if req.industry:
        config_dict["industry"] = req.industry
    if req.report_text:
        config_dict["report_text"] = req.report_text

    db = get_database()
    db.create_campaign({
        "id": cid,
        "name": name,
        "campaign_type": req.campaign_type.value,
        "config": config_dict,
        "companies": companies_list,
        "total_companies": len(companies_list),
    })

    row = db.get_campaign(cid)
    if not row:
        raise HTTPException(500, "Failed to persist campaign")

    logger.info("Created campaign %s (%s) with %d companies", cid, name, len(companies_list))
    return _db_to_response(row)


@router.get("/", response_model=CampaignListResponse)
async def list_campaigns(limit: int = Query(50, ge=1, le=200)):
    """List all campaigns, newest first."""
    db = get_database()
    rows = db.list_campaigns(limit=limit)
    return CampaignListResponse(
        campaigns=[_db_to_response(r) for r in rows],
        total=len(rows),
    )


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: str):
    """Get full campaign details."""
    db = get_database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(404, f"Campaign {campaign_id} not found")
    return _db_to_response(row)


@router.post("/{campaign_id}/run")
async def run_campaign(campaign_id: str, background_tasks: BackgroundTasks):
    """Start campaign execution as a background task with SSE streaming."""
    db = get_database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(404, f"Campaign {campaign_id} not found")
    if row["status"] == "running":
        raise HTTPException(409, f"Campaign {campaign_id} is already running")

    event_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    _active_queues[campaign_id] = event_queue

    db.update_campaign(campaign_id, {"status": "running"})

    from app.api.campaign_executor import execute_campaign

    background_tasks.add_task(execute_campaign, campaign_id, event_queue)

    logger.info("Started campaign %s", campaign_id)
    return {"status": "started", "campaign_id": campaign_id}


@router.get("/{campaign_id}/stream")
async def stream_campaign(campaign_id: str):
    """SSE stream for campaign progress events."""
    if campaign_id not in _active_queues:
        raise HTTPException(404, f"No active stream for campaign {campaign_id}")

    event_queue = _active_queues[campaign_id]

    async def event_generator():
        heartbeat_interval = 15
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(), timeout=heartbeat_interval,
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("event") in ("campaign_done", "campaign_error"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
                except Exception:
                    break
        finally:
            _active_queues.pop(campaign_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{campaign_id}/export/csv")
async def export_campaign_csv(campaign_id: str):
    """Export campaign contacts + emails as CSV download."""
    from fastapi.responses import Response
    import csv
    import io

    db = get_database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(404, f"Campaign {campaign_id} not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Company", "Domain", "Industry", "Contact Name", "Role",
        "Email", "LinkedIn", "Seniority", "Confidence",
        "Email Subject", "Email Body",
    ])

    for company in row.get("companies", []):
        contacts = company.get("contacts", [])
        emails = company.get("emails", [])
        # Map emails by recipient name for pairing
        email_map: dict[str, dict] = {}
        for e in emails:
            key = (e.get("recipient_name", "") or "").lower().strip()
            if key:
                email_map[key] = e

        if contacts:
            for c in contacts:
                name = c.get("full_name", "")
                matched_email = email_map.get(name.lower().strip(), {})
                writer.writerow([
                    company["company_name"],
                    company.get("domain", ""),
                    company.get("industry", ""),
                    name,
                    c.get("role", ""),
                    c.get("email", ""),
                    c.get("linkedin_url", ""),
                    c.get("seniority", ""),
                    c.get("email_confidence", ""),
                    matched_email.get("subject", ""),
                    matched_email.get("body", ""),
                ])
        elif company.get("status") == "done":
            # Company processed but no contacts found
            writer.writerow([
                company["company_name"], company.get("domain", ""),
                company.get("industry", ""),
                "", "", "", "", "", "", "", "",
            ])

    campaign_name = row.get("name", campaign_id).replace(" ", "_")[:40]
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{campaign_name}_contacts.csv"'},
    )


@router.patch("/{campaign_id}", response_model=CampaignResponse)
async def update_campaign_endpoint(campaign_id: str, req: UpdateCampaignRequest):
    """Update a draft or failed campaign. Running/completed campaigns are locked."""
    db = get_database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(404, f"Campaign {campaign_id} not found")
    if row["status"] in ("running", "completed"):
        raise HTTPException(409, f"Cannot edit a {row['status']} campaign")

    updates: dict = {}
    if req.name is not None:
        updates["name"] = req.name.strip()
    if req.companies is not None:
        companies_list = [
            {"company_name": c.company_name, "status": "pending",
             "domain": c.domain, "industry": c.industry, "context": c.context}
            for c in req.companies
        ]
        updates["companies"] = companies_list
        updates["total_companies"] = len(companies_list)
    if req.config is not None:
        config_dict = req.config.model_dump()
        # Preserve industry/report_text if already set
        existing_config = row.get("config", {})
        if req.industry is not None:
            config_dict["industry"] = req.industry
        elif "industry" in existing_config:
            config_dict["industry"] = existing_config["industry"]
        if req.report_text is not None:
            config_dict["report_text"] = req.report_text
        elif "report_text" in existing_config:
            config_dict["report_text"] = existing_config["report_text"]
        updates["config"] = config_dict
    elif req.industry is not None or req.report_text is not None:
        config_dict = row.get("config", {})
        if req.industry is not None:
            config_dict["industry"] = req.industry
        if req.report_text is not None:
            config_dict["report_text"] = req.report_text
        updates["config"] = config_dict

    # Reset status to draft if was failed
    if row["status"] == "failed":
        updates["status"] = "draft"
        updates["error"] = ""

    if updates:
        db.update_campaign(campaign_id, updates)

    return _db_to_response(db.get_campaign(campaign_id))


@router.post("/{campaign_id}/send-email")
async def send_campaign_email(campaign_id: str, req: dict):
    """Send an outreach email from campaign data via Brevo.

    Body: { company_name, recipient_name, recipient_email, subject, body }
    Reuses the same Brevo tool + branded template as the leads email sender.
    """
    db = get_database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(404, f"Campaign {campaign_id} not found")

    to_email = req.get("recipient_email", "")
    to_name = req.get("recipient_name", "")
    subject = req.get("subject", "")
    body = req.get("body", "")
    company = req.get("company_name", "")

    # Fallback to test recipient when no email available (same as leads page)
    if not to_email:
        from app.config import get_settings
        _settings = get_settings()
        if _settings.email_test_mode and _settings.email_test_recipient:
            to_email = _settings.email_test_recipient
            logger.info(f"Campaign send: no contact email — using test recipient: {to_email}")
        else:
            raise HTTPException(400, "No recipient email provided")
    if not subject or not body:
        raise HTTPException(400, "Subject and body are required")

    from app.tools.crm.brevo_tool import BrevoTool
    brevo = BrevoTool()
    result = brevo.send_email(
        to_email=to_email,
        to_name=to_name,
        subject=subject,
        body=body,
        dry_run=False,
        tone="consultative",
        trend_title="",
        company_name=company,
        lead_type="opportunity",
    )
    return result.to_dict()


@router.delete("/{campaign_id}")
async def delete_campaign(campaign_id: str):
    """Delete a campaign."""
    db = get_database()
    if not db.delete_campaign(campaign_id):
        raise HTTPException(404, f"Campaign {campaign_id} not found")
    _active_queues.pop(campaign_id, None)
    logger.info("Deleted campaign %s", campaign_id)
    return {"deleted": True}
