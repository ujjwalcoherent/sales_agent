"""FastAPI application factory with lifespan hooks.

Run: uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import get_database

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: cleanup."""
    settings = get_settings()

    # Database
    db = get_database()
    db.create_tables()
    app.state.db = db
    app.state.settings = settings
    logger.info("Database initialized")

    yield  # App is running

    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Sales Intelligence API",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS for Next.js frontend
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    from app.api.health import router as health_router
    from app.api.pipeline import router as pipeline_router
    from app.api.leads import router as leads_router
    from app.api.feedback import router as feedback_router
    from app.api.learning import router as learning_router

    app.include_router(health_router, tags=["health"])
    app.include_router(pipeline_router, prefix="/api/v1/pipeline", tags=["pipeline"])
    app.include_router(leads_router, prefix="/api/v1/leads", tags=["leads"])
    app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["feedback"])
    app.include_router(learning_router, prefix="/api/v1/learning", tags=["learning"])

    return app


app = create_app()
