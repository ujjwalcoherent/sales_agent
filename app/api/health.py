"""Health check router -- provider status, DB status, config summary."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()


@router.get("/")
async def root():
    return {"service": "Sales Intelligence API", "version": "2.0.0"}


@router.get("/health")
async def health():
    settings = get_settings()

    # Check provider health (non-blocking, catch errors)
    providers = {}
    try:
        from app.tools.provider_health import provider_health
        providers = provider_health.status_report()
    except Exception:
        providers = {"error": "provider_health unavailable"}

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
        "config": {
            # Target market
            "country": settings.country,
            "max_trends": settings.max_trends,
            "mock_mode": settings.mock_mode,
            # Collection
            "rss_hours_ago": settings.rss_hours_ago,
            "rss_max_per_source": settings.rss_max_per_source,
            # Quality gates
            "coherence_min": settings.coherence_min,
            "merge_threshold": settings.merge_threshold,
            "min_synthesis_confidence": settings.min_synthesis_confidence,
            "company_min_relevance": settings.company_min_relevance,
            # Timeouts
            "engine_synthesis_timeout": settings.engine_synthesis_timeout,
            "engine_causal_timeout": settings.engine_causal_timeout,
            # Search services
            "searxng_enabled": settings.searxng_enabled,
            "searxng_url": settings.searxng_url if settings.searxng_enabled else None,
            "use_ddg_fallback": settings.use_ddg_fallback,
            "tavily_enabled": settings.tavily_enabled,
        },
    }
