"""Global application settings API — mock mode toggle, recording selection."""

import logging
import os
from pathlib import Path
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory global mock mode state ──────────────────────────────────────────
# Persists for the life of the server process. Toggled via API.
_GLOBAL_MOCK_MODE: bool = False
_SELECTED_RECORDING: str = ""   # empty = use best available


def get_global_mock_mode() -> bool:
    return _GLOBAL_MOCK_MODE


def _list_recordings() -> list[dict]:
    """Return available recordings sorted newest-first with metadata."""
    recordings_dir = Path("data/recordings")
    if not recordings_dir.exists():
        return []

    results = []
    for d in sorted(recordings_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        # Count checkpoint files to estimate richness
        files = list(d.glob("*.json"))
        # Prefer runs with impact + leads checkpoints
        has_leads = any("leads" in f.name or "lead" in f.name for f in files)
        has_impact = any("impact" in f.name for f in files)
        results.append({
            "run_id": d.name,
            "files": len(files),
            "has_leads": has_leads,
            "has_impact": has_impact,
            # Mark the richest known run (120h March 10 data)
            "recommended": d.name == "20260310_115444",
        })
    return results


def _pick_best_recording() -> str:
    """Auto-select the best recording: prefer recommended, else newest with most files."""
    recordings = _list_recordings()
    if not recordings:
        return ""
    # Prefer the known-best recording
    for r in recordings:
        if r["recommended"]:
            return r["run_id"]
    # Fallback: most files (richest data)
    return max(recordings, key=lambda r: r["files"])["run_id"]


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.get("/mock-mode")
async def get_mock_mode():
    """Get current global mock mode state."""
    recording = _SELECTED_RECORDING or _pick_best_recording()
    recordings = _list_recordings()
    return {
        "enabled": _GLOBAL_MOCK_MODE,
        "selected_recording": recording,
        "available_recordings": recordings[:20],  # cap for UI
        "recommendation": (
            "Enable mock mode to test all 3 campaign types using real data from "
            f"recording {recording!r} without API costs."
            if not _GLOBAL_MOCK_MODE else
            f"Mock mode ON — using recording {recording!r}. "
            "All pipeline runs and campaigns will use cached data."
        ),
    }


@router.post("/mock-mode")
async def set_mock_mode(body: dict):
    """Toggle global mock mode on/off. Optionally select a recording."""
    global _GLOBAL_MOCK_MODE, _SELECTED_RECORDING

    _GLOBAL_MOCK_MODE = bool(body.get("enabled", False))

    if "recording" in body:
        _SELECTED_RECORDING = str(body["recording"])
    elif _GLOBAL_MOCK_MODE and not _SELECTED_RECORDING:
        # Auto-select best recording when enabling
        _SELECTED_RECORDING = _pick_best_recording()

    recording = _SELECTED_RECORDING or _pick_best_recording()
    logger.info(
        "Global mock mode %s (recording: %s)",
        "ENABLED" if _GLOBAL_MOCK_MODE else "DISABLED",
        recording,
    )

    # Also set the env-based MOCK_MODE so pipeline runs pick it up
    os.environ["MOCK_MODE"] = "true" if _GLOBAL_MOCK_MODE else "false"
    if _GLOBAL_MOCK_MODE and recording:
        os.environ["REPLAY_RUN_ID"] = recording

    return {
        "enabled": _GLOBAL_MOCK_MODE,
        "selected_recording": recording,
        "message": (
            f"Mock mode enabled — using recording {recording!r}"
            if _GLOBAL_MOCK_MODE else
            "Mock mode disabled — live API calls active"
        ),
    }
