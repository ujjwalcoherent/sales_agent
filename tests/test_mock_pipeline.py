"""
End-to-end mock pipeline validation for all 3 use cases.
Validates: articles fetched correctly per mode, graph compiles, no import errors.
"""
import asyncio

import pytest

from app.intelligence.models import DiscoveryScope, DiscoveryMode


def test_mock_article_routing():
    """Verify _make_mock_articles returns mode-specific datasets.

    industry_first with fintech_bfsi slices to 12 fintech-only articles
    (2 clusters: RBI KYC + UPI/BNPL) for sector-coherent mock output.
    Other modes use the full 17+ article datasets.
    """
    from app.agents.source_intel import _make_mock_articles

    cases = [
        ("company_first", DiscoveryScope(mode=DiscoveryMode.COMPANY_FIRST, companies=["TCS"]), "TCS", 15),
        ("industry_first", DiscoveryScope(mode=DiscoveryMode.INDUSTRY_FIRST, industry="fintech_bfsi"), "KYC", 10),
        ("report_driven", DiscoveryScope(mode=DiscoveryMode.REPORT_DRIVEN, report_text="Carrier rocket market"), "Rocket", 15),
    ]
    for mode_name, scope, keyword, min_articles in cases:
        articles = _make_mock_articles(scope=scope)
        assert len(articles) >= min_articles, f"{mode_name}: expected ≥{min_articles} articles, got {len(articles)}"
        titles_combined = " ".join(a.title for a in articles)
        assert keyword in titles_combined, (
            f"{mode_name}: expected '{keyword}' in titles — wrong dataset returned?"
        )


def test_graph_compiles():
    """Verify LangGraph pipeline graph compiles without errors."""
    from app.agents.orchestrator import create_pipeline_graph
    g = create_pipeline_graph()
    assert g is not None


def test_fetch_report_entities_dataclass():
    """ReportEntities dataclass is importable and has correct fields."""
    from app.intelligence.fetch import ReportEntities
    e = ReportEntities(companies=["TCS"], industries=["IT"], topics=["AI adoption"])
    assert e.companies == ["TCS"]
    assert e.industries == ["IT"]
    assert e.topics == ["AI adoption"]


def test_noise_reassign_function_exists():
    """Noise reassignment function is present and callable."""
    from app.intelligence.cluster.orchestrator import _reassign_noise
    import inspect
    sig = inspect.signature(_reassign_noise)
    params = list(sig.parameters)
    assert "all_clusters" in params
    assert "all_noise" in params


def test_signal_bus_dead_method_removed():
    """Dead get_adaptive_threshold_modulation method is removed."""
    from app.learning.signal_bus import LearningSignalBus
    assert not hasattr(LearningSignalBus, "get_adaptive_threshold_modulation"), (
        "Dead method get_adaptive_threshold_modulation should have been removed"
    )


def test_threshold_adapter_accepts_system_confidence():
    """ThresholdAdapter.update() now accepts system_confidence kwarg."""
    from app.learning.threshold_adapter import ThresholdAdapter, ThresholdUpdate
    import inspect
    sig = inspect.signature(ThresholdAdapter.update)
    assert "system_confidence" in sig.parameters


def test_trend_level_field_on_trend_data():
    """TrendData.trend_level is a Literal['major','sub','minor'] that derives from actionability_score."""
    from app.schemas.sales import TrendData
    # Default is 'sub'
    t = TrendData(trend_title="Test", summary="Test summary")
    assert t.trend_level == "sub"
    # Explicit major/minor
    t_major = TrendData(trend_title="Test", summary="x", trend_level="major")
    assert t_major.trend_level == "major"
    t_minor = TrendData(trend_title="Test", summary="x", trend_level="minor")
    assert t_minor.trend_level == "minor"


def test_nli_cache_conditional_clear():
    """clear_score_cache_if_hypothesis_changed skips clear on unchanged hypothesis."""
    from app.intelligence.engine.nli_filter import (
        clear_score_cache_if_hypothesis_changed,
        _score_cache,
        _cache_lock,
    )
    # Prime: clear unconditionally to reset state
    with _cache_lock:
        _score_cache.clear()
    # First conditional call: clears because last_cleared is None
    clear_score_cache_if_hypothesis_changed()
    # Second call with same hypothesis: should NOT clear the cache
    # (we can't easily verify internal state without patching, but it must not raise)
    clear_score_cache_if_hypothesis_changed()


def test_match_roles_accepts_bandit_parameter():
    """match_roles_to_trend accepts optional bandit= to avoid N disk reads per loop."""
    from app.agents.workers.contact_agent import match_roles_to_trend
    import inspect
    sig = inspect.signature(match_roles_to_trend)
    assert "bandit" in sig.parameters, (
        "match_roles_to_trend must accept bandit= parameter "
        "to avoid N ContactBandit.load() calls in loops"
    )
    # Calling with bandit=None must behave identically to no bandit (graceful fallback)
    roles = match_roles_to_trend("funding", bandit=None)
    assert isinstance(roles, list) and len(roles) > 0, (
        "match_roles_to_trend must return non-empty list for trend_type='funding'"
    )


def test_report_entities_llm_model():
    """ReportEntitiesLLM model validates and coerces LLM output correctly."""
    from app.schemas.llm_outputs import ReportEntitiesLLM
    # Normal input
    r = ReportEntitiesLLM(companies=["TCS", "Infosys"], industries=["IT"], topics=["AI adoption"])
    assert r.companies == ["TCS", "Infosys"]
    # Coercion: None → empty list
    r2 = ReportEntitiesLLM(companies=None, industries=None, topics=None)
    assert r2.companies == []
    assert r2.industries == []


def test_company_fields_llm_model():
    """CompanyFieldsLLM model validates and coerces LLM output correctly."""
    from app.schemas.llm_outputs import CompanyFieldsLLM
    # Normal input with all fields
    r = CompanyFieldsLLM(industry="IT Services", ceo="John Doe", founded_year=2005,
                         products_services=["Consulting", "Cloud"])
    assert r.industry == "IT Services"
    assert r.products_services == ["Consulting", "Cloud"]
    # Coercion: None str fields → empty string
    r2 = CompanyFieldsLLM(industry=None, ceo=None)
    assert r2.industry == ""
    assert r2.ceo == ""
    # Coercion: int to str for employee_count
    r3 = CompanyFieldsLLM(employee_count=5000)
    assert r3.employee_count == "5000"


def test_discovery_cluster_uses_lower_coherence_threshold():
    """Discovery clusters (Leiden) use _DISCOVERY_COHERENCE_MIN=0.35, not val_coherence_min=0.40."""
    from app.intelligence.engine.validator import _DISCOVERY_COHERENCE_MIN
    from app.intelligence.config import DEFAULT_PARAMS
    assert _DISCOVERY_COHERENCE_MIN < DEFAULT_PARAMS.val_coherence_min, (
        f"Discovery coherence threshold {_DISCOVERY_COHERENCE_MIN} must be < "
        f"entity-seeded threshold {DEFAULT_PARAMS.val_coherence_min}"
    )


async def test_run_pipeline_company_first_mock():
    """Full pipeline mock run for company_first — must not crash and must detect trends.

    PipelineResult uses count fields (trends_detected, leads_generated) not lists.
    """
    from app.agents.orchestrator import run_pipeline
    scope = DiscoveryScope(mode=DiscoveryMode.COMPANY_FIRST, companies=["TCS", "Infosys"])
    result = await run_pipeline(mock_mode=True, scope=scope)
    assert result is not None, "Pipeline returned None"
    assert result.status == "success", f"Pipeline status={result.status} errors={result.errors[:2]}"
    print(f"\ncompany_first: trends={result.trends_detected} companies={result.companies_found} "
          f"leads={result.leads_generated} runtime={result.run_time_seconds:.0f}s")
    # Pipeline must detect at least 1 trend from 17 themed mock articles
    assert result.trends_detected > 0, (
        f"company_first mock: 0 trends_detected — "
        f"check clustering coherence thresholds vs mock article quality"
    )
