"""
Comprehensive tests for the self-learning inter-agent communication system.

Tests cover:
- StageAdvisory protocol
- SPOC verify-then-correct gate
- All 8 verify_*() functions
- Strategic signal score
- ThresholdAdapter wiring
- ContactBandit wiring
- ExperienceLibrary (EvolveR pattern)
- Integration checks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from uuid import uuid4


# ── Test 1: StageAdvisory creation ────────────────────────────────
def test_stage_advisory():
    from app.learning.pipeline_validator import StageAdvisory
    adv = StageAdvisory(
        from_stage="filter",
        quality_level="nominal",
        metrics={"nli_mean": 0.65, "pass_rate": 0.35},
        warnings=[],
        suggested_adjustments={},
    )
    assert adv.from_stage == "filter"
    assert adv.quality_level == "nominal"
    d = adv.to_dict()
    assert isinstance(d, dict)
    assert d["from_stage"] == "filter"
    print("  [PASS] test_stage_advisory")


# ── Test 2: should_correct gate ───────────────────────────────────
def test_should_correct():
    from app.learning.pipeline_validator import (
        StageValidation, StageStatus, should_correct,
    )
    # PASS status → no correction
    val_ok = StageValidation(
        stage="test", status=StageStatus.PASS,
        metrics={}, issues=[], corrective_params={},
    )
    assert not should_correct(val_ok)

    # FAIL but no corrective_params → no correction
    val_no_params = StageValidation(
        stage="test", status=StageStatus.FAIL,
        metrics={}, issues=["bad"], corrective_params={},
    )
    assert not should_correct(val_no_params)

    # FAIL with corrective_params and high confidence → CORRECT
    val_high = StageValidation(
        stage="test", status=StageStatus.FAIL,
        metrics={"confidence_in_error": 0.85},
        issues=["issue"],
        corrective_params={"fix": True},
    )
    assert should_correct(val_high)

    # FAIL with corrective_params but LOW confidence → NO correction
    val_low = StageValidation(
        stage="test", status=StageStatus.FAIL,
        metrics={"confidence_in_error": 0.50},
        issues=["issue"],
        corrective_params={"fix": True},
    )
    assert not should_correct(val_low)
    print("  [PASS] test_should_correct")


# ── Test 3: verify_dedup ──────────────────────────────────────────
def test_verify_dedup():
    from app.learning.pipeline_validator import verify_dedup

    val, adv = verify_dedup(
        raw_count=100,
        deduped_count=85,
        dedup_pairs=8,
    )
    assert adv.from_stage == "dedup"
    assert "removal_rate" in adv.metrics or "deduped_count" in adv.metrics
    print("  [PASS] test_verify_dedup")


# ── Test 4: verify_filter ─────────────────────────────────────────
def test_verify_filter():
    from app.learning.pipeline_validator import verify_filter

    val, adv = verify_filter(
        input_count=100,
        kept_count=35,
        auto_accepted=10,
        auto_rejected=40,
        llm_classified=50,
        nli_mean=0.55,
        false_positives_found=2,
    )
    assert adv.from_stage == "filter"
    assert "pass_rate" in adv.metrics
    assert "nli_mean" in adv.metrics
    print("  [PASS] test_verify_filter")


# ── Test 5: verify_entity_extraction ──────────────────────────────
def test_verify_entity_extraction():
    from app.learning.pipeline_validator import verify_entity_extraction

    val, adv = verify_entity_extraction(
        input_count=80,
        group_count=15,
        grouped_article_count=60,
        ungrouped_count=20,
    )
    assert adv.from_stage == "entity"
    assert "grouping_rate" in adv.metrics
    print("  [PASS] test_verify_entity_extraction")


# ── Test 6: verify_clustering ─────────────────────────────────────
def test_verify_clustering():
    from app.learning.pipeline_validator import verify_clustering

    val, adv = verify_clustering(
        input_count=60,
        total_clusters=12,
        passed_count=8,
        failed_count=4,
        noise_count=5,
        mean_coherence=0.45,
    )
    assert adv.from_stage == "clustering"
    assert "mean_coherence" in adv.metrics
    print("  [PASS] test_verify_clustering")


# ── Test 7: verify_lead_crystallization ───────────────────────────
def test_verify_lead_crystallization():
    from app.learning.pipeline_validator import verify_lead_crystallization

    class MockLead:
        def __init__(self, name, conf, etype):
            self.company_name = name
            self.confidence = conf
            self.event_type = etype

    leads = [
        MockLead("Apollo Hospitals", 0.85, "m_and_a"),
        MockLead("Tata Motors", 0.70, "expansion"),
        MockLead("A leading provider of cloud solutions that helps enterprise customers", 0.15, "general"),
    ]

    val, adv = verify_lead_crystallization(leads, trend_count=3)
    assert adv.from_stage in ("leads", "lead_crystallization")
    assert "total_leads" in val.metrics or "lead_count" in val.metrics
    print("  [PASS] test_verify_lead_crystallization")


# ── Test 8: verify_synthesis ──────────────────────────────────────
def test_verify_synthesis():
    from app.learning.pipeline_validator import verify_synthesis

    class MockTrend:
        def __init__(self, label, etype):
            self.trend_title = label
            self.label = label
            self.event_type = etype

    trends = [
        MockTrend("Apollo Hospitals Acquires Karnataka Facilities", "m_and_a"),
        MockTrend("Technology", "general"),
    ]

    val, adv = verify_synthesis(trends)
    assert adv.from_stage == "synthesis"
    print("  [PASS] test_verify_synthesis")


# ── Test 9: verify_company_enrichment ─────────────────────────────
def test_verify_company_enrichment():
    from app.learning.pipeline_validator import verify_company_enrichment

    class MockCompany:
        def __init__(self, name, desc, domain, ner):
            self.name = name
            self.company_name = name
            self.description = desc
            self.domain = domain
            self.ner_verified = ner

    companies = [
        MockCompany("Apollo Hospitals", "Healthcare chain", "apollo.com", True),
        MockCompany("BadCorp", "", "", False),  # triple failure
    ]
    leads = []  # Not actually used by verify_company_enrichment logic beyond count

    val, adv = verify_company_enrichment(companies, leads)
    assert adv.from_stage in ("enrichment", "company_enrichment")
    assert "enrichment_rate" in val.metrics
    assert val.metrics.get("triple_failures", 0) >= 1
    print("  [PASS] test_verify_company_enrichment")


# ── Test 10: verify_contacts ──────────────────────────────────────
def test_verify_contacts():
    from app.learning.pipeline_validator import verify_contacts

    class MockContact:
        def __init__(self, name, email, company):
            self.name = name
            self.email = email
            self.company_name = company

    contacts = [
        MockContact("John Doe", "john@apollo.com", "Apollo Hospitals"),
        MockContact("Jane Smith", "", "Apollo Hospitals"),
        MockContact("Bob Lee", "bob@tata.com", "Tata Motors"),
    ]
    leads = []

    val, adv = verify_contacts(contacts, leads)
    assert adv.from_stage == "contacts"
    assert "email_rate" in val.metrics or "email_rate" in adv.metrics
    print("  [PASS] test_verify_contacts")


# ── Test 11: Strategic Signal Score ───────────────────────────────
def test_strategic_score():
    from app.learning.pipeline_validator import _compute_strategic_score

    # Create mock cluster objects with the attributes _compute_strategic_score reads
    class MockCluster:
        def __init__(self, label, summary, primary_entity, industries=None, article_indices=None):
            self.label = label
            self.summary = summary
            self.primary_entity = primary_entity
            self.industries = industries or []
            self.article_indices = article_indices or []
            self.cluster_id = "test"

    high_cluster = MockCluster(
        label="Apollo Hospitals Acquires Karnataka Facilities",
        summary="Apollo Hospitals Enterprise Ltd has acquired three hospital facilities in Karnataka",
        primary_entity="Apollo Hospitals",
        industries=["Healthcare"],
    )

    low_cluster = MockCluster(
        label="Jim Cramer Stock Commentary",
        summary="Jim Cramer discusses market trends on CNBC",
        primary_entity="Jim Cramer",
    )

    high = _compute_strategic_score(high_cluster)
    low = _compute_strategic_score(low_cluster)

    assert high > low, f"High ({high:.3f}) should beat Low ({low:.3f})"
    assert high >= 0.50, f"M&A cluster should be HIGH priority, got {high:.3f}"
    print(f"  [PASS] test_strategic_score (M&A={high:.3f}, Commentary={low:.3f})")


# ── Test 12: ExperienceLibrary ────────────────────────────────────
def test_experience_library():
    from app.learning.experience_library import ExperienceLibrary, ExperienceEntry
    from pathlib import Path
    import tempfile

    # Use temp file
    tmp = Path(tempfile.mktemp(suffix=".json"))
    lib = ExperienceLibrary(path=tmp)

    # Record a successful correction
    lib.record(ExperienceEntry(
        situation_stage="clustering",
        situation_metric="coherence",
        situation_value=0.28,
        action_type="tighten_coherence",
        action_params={"delta": 0.05},
        outcome_improved=True,
        outcome_delta=0.12,
    ))

    # Record a failed correction (should NOT be stored)
    lib.record(ExperienceEntry(
        situation_stage="clustering",
        situation_metric="coherence",
        situation_value=0.30,
        action_type="loosen_coherence",
        action_params={"delta": -0.05},
        outcome_improved=False,
        outcome_delta=-0.05,
    ))

    # Find proven fix
    fix = lib.find_proven_fix("clustering", "coherence", 0.30)
    assert fix is not None, "Should find a proven fix"
    assert fix.action_type == "tighten_coherence"
    assert fix.outcome_delta == 0.12

    # Stats
    stats = lib.get_stats()
    assert stats["total_entries"] == 1  # Only successful one stored

    # Cleanup
    tmp.unlink(missing_ok=True)
    print("  [PASS] test_experience_library")


# ── Test 13: ThresholdAdapter exists and has correct API ──────────
def test_threshold_adapter_api():
    from app.learning.threshold_adapter import get_threshold_adapter, ThresholdUpdate

    adapter = get_threshold_adapter()
    assert hasattr(adapter, "update")

    # ThresholdUpdate should accept these fields
    tu = ThresholdUpdate(
        observed_filter_accept_rate=0.35,
        observed_coherence=0.45,
        observed_pass_rate=0.60,
        observed_noise_rate=0.10,
        run_id="test-run",
    )
    assert tu.observed_filter_accept_rate == 0.35
    assert tu.observed_noise_rate == 0.10
    assert tu.run_id == "test-run"
    print("  [PASS] test_threshold_adapter_api")


# ── Test 14: ContactBandit exists and has correct API ─────────────
def test_contact_bandit_api():
    from app.learning.contact_bandit import ContactBandit

    cb = ContactBandit.load()
    assert hasattr(cb, "rank_roles")
    assert hasattr(cb, "update")
    assert hasattr(cb, "save")

    # rank_roles returns list of (role, score) tuples
    ranked = cb.rank_roles(
        roles=["CTO", "CFO", "VP Engineering"],
        event_type="technology",
        company_size="large",
    )
    assert isinstance(ranked, list)
    assert len(ranked) == 3
    assert isinstance(ranked[0], tuple)
    assert len(ranked[0]) == 2
    print("  [PASS] test_contact_bandit_api")


# ── Test 15: GraphState has stage_advisories ──────────────────────
def test_graphstate_advisories():
    from app.agents.orchestrator import GraphState
    assert "stage_advisories" in GraphState.__annotations__, (
        "GraphState must have stage_advisories field"
    )
    print("  [PASS] test_graphstate_advisories")


# ── Test 16: PipelineState has stage_advisories ───────────────────
def test_pipeline_state_advisories():
    from app.intelligence.models import PipelineState
    fields = PipelineState.model_fields if hasattr(PipelineState, 'model_fields') else PipelineState.__fields__
    assert "stage_advisories" in fields, (
        "PipelineState must have stage_advisories field"
    )
    print("  [PASS] test_pipeline_state_advisories")


# ── Test 17: ClusterResult has strategic_score ────────────────────
def test_cluster_result_strategic_score():
    from app.intelligence.models import ClusterResult
    fields = ClusterResult.model_fields if hasattr(ClusterResult, 'model_fields') else ClusterResult.__fields__
    assert "strategic_score" in fields, (
        "ClusterResult must have strategic_score field"
    )
    print("  [PASS] test_cluster_result_strategic_score")


# ── Test 18: NewsArticle has embedding field ──────────────────────
def test_news_article_embedding():
    from app.schemas.news import NewsArticle
    fields = NewsArticle.model_fields if hasattr(NewsArticle, 'model_fields') else NewsArticle.__fields__
    assert "embedding" in fields, "NewsArticle must have embedding field"
    print("  [PASS] test_news_article_embedding")


# ── Test 19: MetaReasoner has run_retrospective method ────────────
def test_meta_reasoner_structured():
    from app.learning.meta_reasoner import MetaReasoner
    mr = MetaReasoner()
    assert hasattr(mr, "run_retrospective"), "MetaReasoner must have run_retrospective()"
    print("  [PASS] test_meta_reasoner_structured")


# ── Test 20: Integration — advisory flows through pipeline state ──
def test_advisory_flow_integration():
    from app.learning.pipeline_validator import StageAdvisory
    from app.intelligence.models import PipelineState

    adv = StageAdvisory(
        from_stage="filter",
        quality_level="degraded",
        metrics={"pass_rate": 0.02},
        warnings=["Very low pass rate"],
        suggested_adjustments={"rollback_hypothesis": True},
    )

    # PipelineState should accept advisory dicts
    state = PipelineState(stage_advisories=[adv.to_dict()])
    assert len(state.stage_advisories) == 1
    assert state.stage_advisories[0]["from_stage"] == "filter"
    print("  [PASS] test_advisory_flow_integration")


# ── Test 21: SPOC correction boundary check ───────────────────────
def test_spoc_correction_threshold():
    from app.learning.pipeline_validator import (
        StageValidation, StageStatus, should_correct,
    )

    # At boundary: 0.75 exactly — FAIL with confidence_in_error = 0.75
    val_boundary = StageValidation(
        stage="test", status=StageStatus.FAIL,
        metrics={"confidence_in_error": 0.75},
        issues=["borderline"],
        corrective_params={"fix": True},
    )
    assert should_correct(val_boundary), "0.75 should trigger correction"

    # Just below: 0.74
    val_below = StageValidation(
        stage="test", status=StageStatus.FAIL,
        metrics={"confidence_in_error": 0.74},
        issues=["borderline"],
        corrective_params={"fix": True},
    )
    assert not should_correct(val_below), "0.74 should NOT trigger correction"
    print("  [PASS] test_spoc_correction_threshold")


# ── Test 22: Experience library FIFO eviction ─────────────────────
def test_experience_fifo():
    from app.learning.experience_library import ExperienceLibrary, ExperienceEntry
    from pathlib import Path
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".json"))
    lib = ExperienceLibrary(path=tmp)

    # Record 205 entries (max is 200)
    for i in range(205):
        lib.record(ExperienceEntry(
            situation_stage="test",
            situation_metric="metric",
            situation_value=float(i),
            action_type=f"action_{i}",
            action_params={},
            outcome_improved=True,
            outcome_delta=0.01,
        ))

    stats = lib.get_stats()
    assert stats["total_entries"] == 200, f"Expected 200, got {stats['total_entries']}"

    tmp.unlink(missing_ok=True)
    print("  [PASS] test_experience_fifo")


# ── Test 23: Verify functions return correct tuple types ──────────
def test_verify_return_types():
    from app.learning.pipeline_validator import (
        verify_dedup, verify_filter, verify_entity_extraction,
        verify_clustering, verify_lead_crystallization,
        verify_company_enrichment, verify_contacts,
        StageValidation, StageAdvisory,
    )

    # Test each returns (StageValidation, StageAdvisory)
    funcs_and_args = [
        (verify_dedup, (100, 85, 5)),
        (verify_filter, (100, 35, 10, 40, 50, 0.55, 2)),
        (verify_entity_extraction, (80, 15, 60, 20)),
        (verify_clustering, (60, 12, 8, 4, 5, 0.45)),
    ]

    for func, args in funcs_and_args:
        result = func(*args)
        assert isinstance(result, tuple), f"{func.__name__} should return tuple"
        assert len(result) == 2, f"{func.__name__} should return 2-tuple"
        val, adv = result
        assert isinstance(val, StageValidation), f"{func.__name__}[0] should be StageValidation"
        assert isinstance(adv, StageAdvisory), f"{func.__name__}[1] should be StageAdvisory"

    print("  [PASS] test_verify_return_types")


# ── Test 24: Strategic score edge cases ───────────────────────────
def test_strategic_score_edge_cases():
    from app.learning.pipeline_validator import _compute_strategic_score

    class MockCluster:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Empty inputs
    empty_cluster = MockCluster(
        label="", summary="", primary_entity="",
        industries=[], article_indices=[], cluster_id="test",
    )
    score = _compute_strategic_score(empty_cluster)
    assert 0.0 <= score <= 1.0, f"Score should be in [0,1], got {score}"

    # Perfect M&A inputs
    perfect_cluster = MockCluster(
        label="Google Acquires DeepMind AI Research Lab",
        summary="Google has completed its acquisition of DeepMind for $500M",
        primary_entity="Google",
        industries=["Technology"],
        article_indices=[], cluster_id="test",
    )
    score_perfect = _compute_strategic_score(perfect_cluster)
    assert score_perfect >= 0.70, f"Perfect M&A should score high, got {score_perfect:.3f}"

    print(f"  [PASS] test_strategic_score_edge_cases (empty={score:.3f}, perfect={score_perfect:.3f})")


# ── Run all tests ─────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_stage_advisory,
        test_should_correct,
        test_verify_dedup,
        test_verify_filter,
        test_verify_entity_extraction,
        test_verify_clustering,
        test_verify_lead_crystallization,
        test_verify_synthesis,
        test_verify_company_enrichment,
        test_verify_contacts,
        test_strategic_score,
        # test_experience_library — removed (experience_library.py deleted, March 2026 audit)
        test_threshold_adapter_api,
        test_contact_bandit_api,
        test_graphstate_advisories,
        test_pipeline_state_advisories,
        test_cluster_result_strategic_score,
        test_news_article_embedding,
        test_meta_reasoner_structured,
        test_advisory_flow_integration,
        test_spoc_correction_threshold,
        # test_experience_fifo — removed (experience_library.py deleted, March 2026 audit)
        test_verify_return_types,
        test_strategic_score_edge_cases,
    ]

    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*60}")
    print(f"  Self-Learning System Test Suite - {len(tests)} tests")
    print(f"{'='*60}\n")

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    sys.exit(0 if failed == 0 else 1)
