"""Feedback router -- wraps existing feedback collection for API access."""

from fastapi import APIRouter

from app.api.schemas import FeedbackRequest, FeedbackResponse, FeedbackSummaryResponse
from app.tools.feedback import save_feedback, get_feedback_summary

router = APIRouter()


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    """Submit trend or lead feedback for the learning system."""
    record = save_feedback(
        feedback_type=body.feedback_type,
        item_id=body.item_id,
        rating=body.rating,
        metadata=body.metadata,
    )
    return FeedbackResponse(saved=True, record=record)


@router.get("/summary", response_model=FeedbackSummaryResponse)
async def feedback_summary():
    """Get aggregated feedback counts."""
    summary = get_feedback_summary()
    return FeedbackSummaryResponse(**summary)
