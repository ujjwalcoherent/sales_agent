"""Learning status API — exposes all 6 self-learning loop states."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = Path("data")


class LearningStatusResponse(BaseModel):
    """Full learning system state — all 6 loops."""
    source_bandit: Dict[str, Any] = Field(default_factory=dict)
    weight_learner: Dict[str, Any] = Field(default_factory=dict)
    company_bandit: Dict[str, Any] = Field(default_factory=dict)
    adaptive_thresholds: Dict[str, Any] = Field(default_factory=dict)
    trend_memory: Dict[str, Any] = Field(default_factory=dict)
    feedback: Dict[str, Any] = Field(default_factory=dict)


def _load_json(path: Path) -> Dict:
    """Safely load a JSON file, returning {} on any error."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"Failed to load {path}: {e}")
    return {}


def _count_jsonl(path: Path) -> int:
    """Count lines in a JSONL file."""
    try:
        if path.exists():
            return sum(1 for line in path.open(encoding="utf-8") if line.strip())
    except Exception:
        pass
    return 0


@router.get("/status", response_model=LearningStatusResponse)
async def learning_status():
    """Get the current state of all 6 self-learning loops."""

    # 1. Source Bandit — Thompson Sampling posteriors per RSS source
    # File stores arms directly at top level (source_id: {alpha, beta}) or
    # nested under "arms" key — handle both formats
    source_data = _load_json(DATA_DIR / "source_bandit.json")
    source_bandit = {}
    if source_data:
        arms = source_data.get("arms", None)
        if arms is None:
            # Top-level format: each key is a source_id with {alpha, beta}
            arms = {k: v for k, v in source_data.items()
                    if isinstance(v, dict) and "alpha" in v}
        # Top 10 sources by posterior mean (alpha / (alpha + beta))
        ranked = sorted(
            arms.items(),
            key=lambda kv: kv[1].get("alpha", 1) / (kv[1].get("alpha", 1) + kv[1].get("beta", 1)),
            reverse=True,
        )[:10]
        source_bandit = {
            "total_arms": len(arms),
            "top_sources": [
                {
                    "source": k,
                    "alpha": v.get("alpha", 1),
                    "beta": v.get("beta", 1),
                    "mean": round(v.get("alpha", 1) / (v.get("alpha", 1) + v.get("beta", 1)), 3),
                    "pulls": v.get("pulls", 0),
                }
                for k, v in ranked
            ],
        }

    # 2. Weight Learner — current learned weights
    # File stores weights directly at top level (actionability, trend_score, ...)
    # or nested under "weights" key — handle both formats
    weight_data = _load_json(DATA_DIR / "learned_weights.json")
    weight_learner = {}
    if weight_data:
        weights = weight_data.get("weights", None)
        if weights is None:
            # Top-level format: {actionability: {...}, trend_score: {...}, ...}
            weights = {k: v for k, v in weight_data.items()
                       if k not in ("data_count", "last_updated")}
        weight_learner = {
            "weights": weights,
            "data_count": weight_data.get("data_count", 0),
            "last_updated": weight_data.get("last_updated", ""),
        }

    # 3. Company Bandit — (size, event_type) arms
    # File stores arms directly at top level or nested under "arms"
    company_data = _load_json(DATA_DIR / "company_bandit.json")
    company_bandit = {}
    if company_data:
        arms = company_data.get("arms", None)
        if arms is None:
            arms = {k: v for k, v in company_data.items()
                    if isinstance(v, dict) and "alpha" in v}
        ranked = sorted(
            arms.items(),
            key=lambda kv: kv[1].get("alpha", 1) / (kv[1].get("alpha", 1) + kv[1].get("beta", 1)),
            reverse=True,
        )[:10]
        company_bandit = {
            "total_arms": len(arms),
            "top_arms": [
                {
                    "arm": k,
                    "alpha": v.get("alpha", 1),
                    "beta": v.get("beta", 1),
                    "mean": round(v.get("alpha", 1) / (v.get("alpha", 1) + v.get("beta", 1)), 3),
                }
                for k, v in ranked
            ],
        }

    # 4. Adaptive Thresholds — EMA-based threshold adaptation
    threshold_data = _load_json(DATA_DIR / "adaptive_thresholds.json")
    adaptive_thresholds = {}
    if threshold_data:
        adaptive_thresholds = {
            "thresholds": threshold_data.get("thresholds", {}),
            "update_count": threshold_data.get("update_count", 0),
        }

    # 5. Trend Memory — ChromaDB novelty scoring
    trend_memory = {}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
        collections = client.list_collections()
        trend_cols = [c for c in collections if "trend" in c.name.lower()]
        if trend_cols:
            col = trend_cols[0]
            count = col.count()
            trend_memory = {
                "collection": col.name,
                "trend_count": count,
            }
    except Exception as e:
        logger.debug(f"Trend memory check failed: {e}")
        trend_memory = {"status": "unavailable"}

    # 6. Feedback — human + auto ratings
    feedback_count = _count_jsonl(DATA_DIR / "feedback.jsonl")
    feedback = {"total_records": feedback_count}
    if feedback_count > 0:
        # Count by type from last 100 records
        try:
            lines = []
            with open(DATA_DIR / "feedback.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lines.append(json.loads(line))
            recent = lines[-100:]
            auto = sum(1 for r in recent if r.get("metadata", {}).get("auto"))
            human = len(recent) - auto
            by_rating = {}
            for r in recent:
                rating = r.get("rating", "unknown")
                by_rating[rating] = by_rating.get(rating, 0) + 1
            feedback["recent_100"] = {
                "auto": auto,
                "human": human,
                "by_rating": by_rating,
            }
        except Exception:
            pass

    return LearningStatusResponse(
        source_bandit=source_bandit,
        weight_learner=weight_learner,
        company_bandit=company_bandit,
        adaptive_thresholds=adaptive_thresholds,
        trend_memory=trend_memory,
        feedback=feedback,
    )
