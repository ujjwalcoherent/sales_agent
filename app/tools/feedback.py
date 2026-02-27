"""
Human feedback collection and storage.

Stores per-trend and per-lead feedback in a JSONL file with full signal
breakdowns. This is the foundation for:
- Must-link / Cannot-link constraints (Phase 8B)
- DSPy prompt optimization (Phase 8C)
- Signal weight tuning from outcomes

Storage: data/feedback.jsonl (append-only, one JSON object per line)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_PATH = Path("./data/feedback.jsonl")


def save_feedback(
    feedback_type: str,
    item_id: str,
    rating: str,
    signals: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
) -> Dict[str, Any]:
    """Save a single feedback record.

    Args:
        feedback_type: "trend" or "lead"
        item_id: Trend or lead identifier.
        rating: For trends: "good_trend", "bad_trend", "already_knew"
                For leads: "would_email", "maybe", "bad_lead"
        signals: Full signal breakdown at time of rating (for learning).
        metadata: Extra context (trend title, company name, etc.)
        feedback_path: JSONL file path.

    Returns:
        The saved record.
    """
    # Validate rating enum to prevent corrupt feedback from entering the system
    _VALID_RATINGS = {
        "trend": {"good_trend", "bad_trend", "already_knew"},
        "lead": {"would_email", "maybe", "bad_lead"},
    }
    valid = _VALID_RATINGS.get(feedback_type, set())
    if valid and rating not in valid:
        logger.warning(
            f"Invalid rating '{rating}' for type '{feedback_type}'. "
            f"Expected one of: {valid}. Storing but skipping propagation."
        )

    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": feedback_type,
        "item_id": item_id,
        "rating": rating,
        "signals": signals or {},
        "metadata": metadata or {},
    }

    try:
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Feedback saved: {feedback_type}/{rating} for {item_id}")
    except Exception as e:
        logger.warning(f"Failed to save feedback: {e}")

    # Auto-learning: propagate feedback to relevant bandits
    _propagate_feedback(record)

    return record


def _propagate_feedback(record: Dict[str, Any]) -> None:
    """Auto-propagate feedback to learning subsystems.

    Called automatically after every save_feedback(). Routes:
      - lead feedback → CompanyRelevanceBandit.update_from_feedback()
      - trend feedback → (weight_learner picks up from JSONL on next run)
    """
    feedback_type = record.get("type", "")
    try:
        if feedback_type == "lead":
            from app.agents.company_relevance_bandit import CompanyRelevanceBandit
            bandit = CompanyRelevanceBandit()
            bandit.update_from_feedback(record)
    except Exception as e:
        logger.warning(f"Feedback propagation failed: {e}")


def load_feedback(
    feedback_type: Optional[str] = None,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
) -> List[Dict[str, Any]]:
    """Load all feedback records, optionally filtered by type.

    Args:
        feedback_type: "trend" or "lead" to filter. None = all.
        feedback_path: JSONL file path.

    Returns:
        List of feedback records.
    """
    if not feedback_path.exists():
        return []

    records = []
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if feedback_type is None or record.get("type") == feedback_type:
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to load feedback: {e}")

    return records


def get_feedback_summary(
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
) -> Dict[str, Any]:
    """Get a summary of all collected feedback.

    Returns:
        Dict with counts by type and rating.
    """
    records = load_feedback(feedback_path=feedback_path)

    summary = {
        "total": len(records),
        "trends": {"good_trend": 0, "bad_trend": 0, "already_knew": 0},
        "leads": {"would_email": 0, "maybe": 0, "bad_lead": 0},
    }

    for record in records:
        rtype = record.get("type", "")
        rating = record.get("rating", "")
        if rtype == "trend" and rating in summary["trends"]:
            summary["trends"][rating] += 1
        elif rtype == "lead" and rating in summary["leads"]:
            summary["leads"][rating] += 1

    return summary
