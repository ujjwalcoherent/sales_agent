"""FastAPI dependency injection -- Depends() patterns using app.state from lifespan."""

from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request

from app.config import Settings, get_settings
from app.database import Database


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
):
    """API key gate. Empty API_KEY env var = dev mode (all requests pass)."""
    settings = get_settings()
    required_key = settings.api_key
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Type aliases for cleaner route signatures
DB = Annotated[Database, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]
