"""
Run Recorder — captures per-step pipeline snapshots for mock replay.

During a real pipeline run, each LangGraph node calls `record_step()` after
completing. This produces a set of JSON files in `data/recordings/{run_id}/`
plus a manifest that maps step names to durations.

Mock replay (see `app/api/pipeline.py`) reads these recordings and streams
them through the same SSE channel with compressed timing (~45s demo).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("data/recordings")


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not handled by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):  # enums
        return obj.value
    if hasattr(obj, "model_dump"):  # pydantic models
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class RunRecorder:
    """Records pipeline step outputs for mock replay."""

    def __init__(self, run_id: str, recording_dir: Path = RECORDINGS_DIR):
        self.run_id = run_id
        self.dir = recording_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps: List[Dict[str, Any]] = []
        self._log_buffer: List[str] = []

    def buffer_log(self, message: str):
        """Buffer a log message for the current step."""
        self._log_buffer.append(message)

    def record_step(
        self,
        step_name: str,
        data: Dict[str, Any],
        duration_seconds: float,
    ):
        """Save a step snapshot as JSON.

        Args:
            step_name: LangGraph step name (e.g. "source_intel_complete").
            data: Serializable dict of step outputs.
            duration_seconds: How long this step took in the real run.
        """
        order = len(self.steps)
        entry = {
            "step": step_name,
            "duration_s": round(duration_seconds, 2),
            "order": order,
        }
        self.steps.append(entry)

        # Attach buffered log messages
        data["log_messages"] = self._log_buffer.copy()
        self._log_buffer.clear()

        step_file = self.dir / f"{order:02d}_{step_name}.json"
        try:
            with open(step_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
            logger.debug(f"Recorded step {order}: {step_name} ({duration_seconds:.1f}s)")
        except Exception as e:
            logger.warning(f"Failed to record step {step_name}: {e}")

    def save_manifest(self, total_duration: float):
        """Save manifest with step ordering and timing."""
        manifest = {
            "run_id": self.run_id,
            "recorded_at": datetime.utcnow().isoformat(),
            "total_duration_s": round(total_duration, 2),
            "step_count": len(self.steps),
            "steps": self.steps,
        }
        manifest_file = self.dir / "manifest.json"
        try:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(
                f"Recording manifest saved: {manifest_file} "
                f"({len(self.steps)} steps, {total_duration:.0f}s total)"
            )
        except Exception as e:
            logger.warning(f"Failed to save recording manifest: {e}")


def get_latest_recording() -> Optional[Path]:
    """Find the most recent recording directory (by manifest timestamp)."""
    if not RECORDINGS_DIR.exists():
        return None

    manifests = []
    for d in RECORDINGS_DIR.iterdir():
        if d.is_dir():
            m = d / "manifest.json"
            if m.exists():
                manifests.append(m)

    if not manifests:
        return None

    # Sort by file modification time (most recent first)
    manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0].parent


def get_recording(run_id: str) -> Optional[Path]:
    """Find a specific recording by run_id."""
    path = RECORDINGS_DIR / run_id
    if path.is_dir() and (path / "manifest.json").exists():
        return path
    return None
