"""
User Profile API — CRUD for UserProfile (Path router configuration).

Endpoints:
  GET    /api/v1/profiles              — list all profiles
  POST   /api/v1/profiles              — create profile
  GET    /api/v1/profiles/{profile_id} — get profile
  PUT    /api/v1/profiles/{profile_id} — full update
  PATCH  /api/v1/profiles/{profile_id} — partial update
  DELETE /api/v1/profiles/{profile_id} — delete profile

Profiles are stored as JSON blobs in SQLite — schema-flexible without migrations.
The active profile_id is used at pipeline start to load user context (industries,
account list, products, contact hierarchy, etc.).
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from app.schemas.industry_profile import UserProfile

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_db(request: Request):
    return request.app.state.db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    return profile.model_dump()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[Dict])
async def list_profiles(request: Request):
    """List all user profiles."""
    db = _get_db(request)
    profiles = db.list_profiles()
    return profiles


@router.post("", response_model=Dict, status_code=201)
async def create_profile(profile: UserProfile, request: Request):
    """Create a new user profile."""
    db = _get_db(request)
    existing = db.get_profile(profile.profile_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Profile '{profile.profile_id}' already exists. Use PUT to update.")

    db.save_profile(_profile_to_dict(profile))
    logger.info(f"[profiles] Created profile: {profile.profile_id}")
    return _profile_to_dict(profile)


@router.get("/{profile_id}", response_model=Dict)
async def get_profile(profile_id: str, request: Request):
    """Get a profile by ID."""
    db = _get_db(request)
    data = db.get_profile(profile_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return data


@router.put("/{profile_id}", response_model=Dict)
async def update_profile(profile_id: str, profile: UserProfile, request: Request):
    """Full update — replaces the entire profile."""
    if profile.profile_id != profile_id:
        raise HTTPException(status_code=422, detail="profile_id in body must match URL")

    db = _get_db(request)
    existing = db.get_profile(profile_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    db.save_profile(_profile_to_dict(profile))
    logger.info(f"[profiles] Updated profile: {profile_id}")
    return _profile_to_dict(profile)


@router.patch("/{profile_id}", response_model=Dict)
async def patch_profile(profile_id: str, updates: Dict[str, Any], request: Request):
    """Partial update — merges provided fields into existing profile."""
    db = _get_db(request)
    existing = db.get_profile(profile_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    # Merge updates into existing data
    merged = {**existing, **updates}
    merged["profile_id"] = profile_id  # Prevent ID change via patch

    try:
        profile = UserProfile(**merged)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid profile data: {exc}")

    db.save_profile(profile.model_dump())
    logger.info(f"[profiles] Patched profile: {profile_id} (fields: {list(updates.keys())})")
    return profile.model_dump()


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(profile_id: str, request: Request):
    """Delete a profile."""
    db = _get_db(request)
    deleted = db.delete_profile(profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    logger.info(f"[profiles] Deleted profile: {profile_id}")


@router.get("/{profile_id}/industries", response_model=List[Dict])
async def get_profile_industries(profile_id: str, request: Request):
    """Return the list of industry targets for a profile."""
    db = _get_db(request)
    data = db.get_profile(profile_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return data.get("target_industries", [])


@router.get("/{profile_id}/account-list", response_model=List[str])
async def get_account_list(profile_id: str, request: Request):
    """Return the account list (Path 2 companies) for a profile."""
    db = _get_db(request)
    data = db.get_profile(profile_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return data.get("account_list", [])
