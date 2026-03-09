"""Health check router -- provider status, DB status, config summary, runtime settings."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings

router = APIRouter()


class SettingsUpdateRequest(BaseModel):
    # Pipeline
    country: Optional[str] = None
    max_trends: Optional[int] = Field(None, ge=3, le=30)
    # Quality gates
    coherence_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    merge_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_synthesis_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    company_min_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Timeouts
    engine_synthesis_timeout: Optional[float] = Field(None, ge=30, le=1800)
    engine_causal_timeout: Optional[float] = Field(None, ge=30, le=1800)
    lead_gen_timeout: Optional[float] = Field(None, ge=60, le=3600)
    # Enrichment
    deep_enrichment_enabled: Optional[bool] = None
    scrapegraph_model: Optional[str] = None
    scrapegraph_max_results: Optional[int] = Field(None, ge=1, le=10)
    scrapegraph_timeout: Optional[int] = Field(None, ge=30, le=300)
    website_scrape_enabled: Optional[bool] = None
    hiring_signals_enabled: Optional[bool] = None
    tech_ip_analysis_enabled: Optional[bool] = None
    # Person intel
    person_deep_intel_enabled: Optional[bool] = None
    person_intel_sources: Optional[str] = None
    person_intel_staleness_days: Optional[int] = Field(None, ge=1, le=30)
    person_intel_max_urls: Optional[int] = Field(None, ge=1, le=20)
    # Contacts
    max_contacts_per_company: Optional[int] = Field(None, ge=1, le=20)
    contact_role_inference: Optional[str] = None
    default_dm_roles: Optional[str] = None
    default_influencer_roles: Optional[str] = None
    # News
    news_lookback_days: Optional[int] = Field(None, ge=1, le=30)
    news_max_articles: Optional[int] = Field(None, ge=5, le=200)
    news_relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    historical_news_enabled: Optional[bool] = None
    historical_news_months: Optional[int] = Field(None, ge=1, le=12)
    # Cache
    company_cache_days: Optional[int] = Field(None, ge=1, le=30)
    company_cache_enabled: Optional[bool] = None
    # Email
    email_personalization_depth: Optional[str] = None
    email_max_length: Optional[int] = Field(None, ge=50, le=1000)
    email_sending_enabled: Optional[bool] = None
    email_test_mode: Optional[bool] = None
    email_test_recipient: Optional[str] = None


class SettingsUpdateResponse(BaseModel):
    updated: dict
    current: dict


@router.get("/")
async def root():
    return {"service": "Sales Intelligence API", "version": "2.0.0"}


@router.get("/health")
async def health():
    settings = get_settings()
    checks = {"db": "ok", "providers": "ok"}

    # Check DB connection
    try:
        from app.database import get_database
        db = get_database()
        db.get_pipeline_runs(limit=1)
    except Exception as e:
        checks["db"] = f"error: {e}"

    # Check provider health
    providers = {}
    try:
        from app.tools.llm.providers import provider_health
        providers = provider_health.status_report()
    except Exception:
        providers = {"error": "provider_health unavailable"}
        checks["providers"] = "unavailable"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    body = {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
        "config": {
            # Pipeline
            "country": settings.country,
            "max_trends": settings.max_trends,
            "mock_mode": settings.mock_mode,
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
            "lead_gen_timeout": settings.lead_gen_timeout,
            # Search
            "search_providers": ["tavily", "ddg_fallback"],
            "tavily_enabled": settings.tavily_enabled,
            "use_ddg_fallback": settings.use_ddg_fallback,
            # Enrichment
            "deep_enrichment_enabled": settings.deep_enrichment_enabled,
            "scrapegraph_model": settings.scrapegraph_model,
            "scrapegraph_max_results": settings.scrapegraph_max_results,
            "scrapegraph_timeout": settings.scrapegraph_timeout,
            "website_scrape_enabled": settings.website_scrape_enabled,
            "hiring_signals_enabled": settings.hiring_signals_enabled,
            "tech_ip_analysis_enabled": settings.tech_ip_analysis_enabled,
            # Person intel
            "person_deep_intel_enabled": settings.person_deep_intel_enabled,
            "person_intel_sources": settings.person_intel_sources,
            "person_intel_staleness_days": settings.person_intel_staleness_days,
            "person_intel_max_urls": settings.person_intel_max_urls,
            # Contacts
            "max_contacts_per_company": settings.max_contacts_per_company,
            "contact_role_inference": settings.contact_role_inference,
            "default_dm_roles": settings.default_dm_roles,
            "default_influencer_roles": settings.default_influencer_roles,
            # News
            "news_lookback_days": settings.news_lookback_days,
            "news_max_articles": settings.news_max_articles,
            "news_relevance_threshold": settings.news_relevance_threshold,
            "historical_news_enabled": settings.historical_news_enabled,
            "historical_news_months": settings.historical_news_months,
            # Cache
            "company_cache_days": settings.company_cache_days,
            "company_cache_enabled": settings.company_cache_enabled,
            # Email
            "email_personalization_depth": settings.email_personalization_depth,
            "email_max_length": settings.email_max_length,
            "email_sending_enabled": settings.email_sending_enabled,
            "email_test_mode": settings.email_test_mode,
            "email_test_recipient": settings.email_test_recipient,
        },
    }

    return JSONResponse(content=body, status_code=status_code)


@router.post("/settings", response_model=SettingsUpdateResponse)
async def update_settings(body: SettingsUpdateRequest):
    """Update runtime settings. Persists in-memory until restart.

    Accepts any field from SettingsUpdateRequest. Only non-None values are applied.
    """
    settings = get_settings()
    updated = {}

    # Country code auto-derivation when country changes
    _COUNTRY_CODES = {
        "india": "IN", "united states": "US", "united kingdom": "GB",
        "germany": "DE", "france": "FR", "japan": "JP", "china": "CN",
        "canada": "CA", "australia": "AU", "brazil": "BR", "singapore": "SG",
        "israel": "IL", "south korea": "KR", "netherlands": "NL", "sweden": "SE",
        "switzerland": "CH", "spain": "ES", "italy": "IT", "mexico": "MX",
        "indonesia": "ID", "thailand": "TH", "vietnam": "VN", "philippines": "PH",
        "malaysia": "MY", "uae": "AE", "united arab emirates": "AE", "saudi arabia": "SA",
        "south africa": "ZA", "nigeria": "NG", "kenya": "KE", "egypt": "EG",
        "turkey": "TR", "poland": "PL", "ireland": "IE", "new zealand": "NZ",
    }

    for field_name, value in body.model_dump(exclude_none=True).items():
        if hasattr(settings, field_name):
            setattr(settings, field_name, value)
            updated[field_name] = value

            # Auto-derive country_code when country changes
            if field_name == "country":
                code = _COUNTRY_CODES.get(value.lower(), "")
                if code:
                    settings.country_code = code
                    updated["country_code"] = code

    return SettingsUpdateResponse(
        updated=updated,
        current={
            "country": settings.country,
            "max_trends": settings.max_trends,
            "deep_enrichment_enabled": settings.deep_enrichment_enabled,
            "person_deep_intel_enabled": settings.person_deep_intel_enabled,
            "max_contacts_per_company": settings.max_contacts_per_company,
            "contact_role_inference": settings.contact_role_inference,
            "company_cache_days": settings.company_cache_days,
            "email_personalization_depth": settings.email_personalization_depth,
        },
    )
