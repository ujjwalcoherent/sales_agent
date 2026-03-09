"""Feedback router -- wraps existing feedback collection for API access.

Rating translation: frontend sends simple labels (good/bad/known/ok), this
endpoint maps them to the domain-specific ratings that the internal learning
system expects (good_trend/bad_trend/already_knew for trends, would_email/
bad_lead/maybe for leads).
"""

from typing import Optional

from fastapi import APIRouter, Query

from app.api.schemas import (
    FeedbackRequest, FeedbackResponse, FeedbackSummaryResponse,
    FeedbackRecord, FeedbackHistoryResponse,
)
from app.tools.feedback_store import save_feedback, get_feedback_summary, load_feedback

router = APIRouter()

# Map frontend labels → internal domain-specific ratings
_RATING_MAP = {
    "trend": {"good": "good_trend", "bad": "bad_trend", "ok": "already_knew", "known": "already_knew"},
    "lead":  {"good": "would_email", "bad": "bad_lead", "ok": "maybe", "known": "maybe"},
}


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    """Submit trend or lead feedback for the learning system."""
    mapping = _RATING_MAP.get(body.feedback_type, {})
    internal_rating = mapping.get(body.rating, body.rating)

    record = save_feedback(
        feedback_type=body.feedback_type,
        item_id=body.item_id,
        rating=internal_rating,
        metadata=body.metadata,
    )
    return FeedbackResponse(saved=True, record=record)


@router.get("/summary", response_model=FeedbackSummaryResponse)
async def feedback_summary():
    """Get aggregated feedback counts."""
    summary = get_feedback_summary()
    return FeedbackSummaryResponse(**summary)


@router.get("/history", response_model=FeedbackHistoryResponse)
async def feedback_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    feedback_type: Optional[str] = Query(None, description="Filter: trend | lead"),
):
    """Get paginated feedback history — newest first."""
    records = load_feedback(feedback_type=feedback_type)
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    total = len(records)
    start = (page - 1) * per_page
    page_records = records[start : start + per_page]

    items = [
        FeedbackRecord(
            timestamp=r.get("timestamp", ""),
            feedback_type=r.get("type", ""),
            item_id=r.get("item_id", ""),
            rating=r.get("rating", ""),
            metadata=r.get("metadata", {}),
        )
        for r in page_records
    ]

    return FeedbackHistoryResponse(items=items, total=total, page=page, per_page=per_page)
