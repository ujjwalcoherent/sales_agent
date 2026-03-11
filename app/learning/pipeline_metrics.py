"""
Pipeline metric logging — appends a JSON record per run to a JSONL file.

`record_pipeline_run()` writes to pipeline_run_log.jsonl at the end of each run.
`record_cluster_signals()` logs per-cluster signal breakdowns for weight auto-learning.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("./data/pipeline_run_log.jsonl")
CLUSTER_SIGNAL_LOG_PATH = Path("./data/cluster_signal_log.jsonl")


def record_pipeline_run(
    run_id: str,
    coherence_scores: List[float],
    trend_scores: List[float],
    confidence_scores: List[float],
    article_count: int = 0,
    cluster_count: int = 0,
    lead_count: int = 0,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one run record to pipeline_run_log.jsonl.

    Computes percentile distributions from raw scores.
    Called from orchestrator learning_update_node at end of each run.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _percentiles(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        arr = np.array(vals)
        return {
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
        }

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "article_count": article_count,
        "cluster_count": cluster_count,
        "lead_count": lead_count,
        "coherence_distribution": _percentiles(coherence_scores),
        "trend_score_distribution": _percentiles(trend_scores),
        "confidence_distribution": _percentiles(confidence_scores),
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(
            f"Pipeline run logged: {cluster_count} clusters, "
            f"coherence_p50={record['coherence_distribution']['p50']:.3f}"
        )
    except Exception as e:
        logger.warning(f"Failed to log pipeline run: {e}")


def load_history(
    log_path: Path = DEFAULT_LOG_PATH,
    last_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load run history from JSONL log, oldest first. Optionally last N only."""
    if not log_path.exists():
        return []

    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to load pipeline history: {e}")
        return []

    if last_n is not None:
        records = records[-last_n:]

    return records


def record_cluster_signals(
    run_id: str,
    cluster_signals: Dict[int, Dict[str, Any]],
    cluster_oss: Dict[int, float],
    cluster_outcomes: Optional[Dict[int, Dict[str, float]]] = None,
    log_path: Path = CLUSTER_SIGNAL_LOG_PATH,
) -> int:
    """Log per-cluster signal breakdowns + outcome scores for weight auto-learning.

    The outcome_score is a NON-CIRCULAR composite:
      40% kb_hit_rate + 30% lead_quality + 30% oss
    This breaks the circular dependency where OSS alone drove weight updates.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    records_written = 0
    cluster_outcomes = cluster_outcomes or {}

    breakdown_keys = [
        "actionability_breakdown",
        "trend_score_breakdown",
        "cluster_quality_breakdown",
        "confidence_breakdown",
    ]

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for cid, signals in cluster_signals.items():
                oss = cluster_oss.get(cid, 0.0)
                if oss is None:
                    oss = 0.0

                breakdowns = {}
                for key in breakdown_keys:
                    bd = signals.get(key)
                    if bd and isinstance(bd, dict):
                        breakdowns[key] = bd

                if not breakdowns:
                    continue

                # Compute composite outcome score from external signals
                outcomes = cluster_outcomes.get(cid, {})
                kb_hit = outcomes.get("kb_hit_rate", 0.0)
                lead_quality = outcomes.get("lead_quality", 0.0)
                outcome_score = 0.40 * kb_hit + 0.30 * lead_quality + 0.30 * float(oss)

                record = {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "cluster_id": cid,
                    "oss": round(float(oss), 4),
                    "outcome_score": round(outcome_score, 4),
                    "kb_hit_rate": round(kb_hit, 4),
                    "lead_quality": round(lead_quality, 4),
                    "breakdowns": breakdowns,
                }
                f.write(json.dumps(record, default=str) + "\n")
                records_written += 1

        if records_written > 0:
            logger.info(
                f"Logged {records_written} cluster signal records "
                f"for weight auto-learning (run={run_id})"
            )
    except Exception as e:
        logger.warning(f"Failed to log cluster signals: {e}")

    return records_written


