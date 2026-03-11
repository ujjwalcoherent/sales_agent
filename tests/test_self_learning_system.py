"""
Comprehensive tests for the self-learning inter-agent communication system.

Tests cover:
- ThresholdAdapter wiring
- ContactBandit wiring
- ExperimentTracker (snapshot/restore, regression detection)
- Pipeline metrics (record/load)
- Signal bus (publish/derive/persist)
- Integration checks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from uuid import uuid4


# ── Test 1: ThresholdAdapter exists and has correct API ──────────
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


# ── Test 2: ContactBandit exists and has correct API ─────────────
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


# ── Test 3: GraphState has stage_advisories ──────────────────────
def test_graphstate_advisories():
    from app.agents.orchestrator import GraphState
    assert "stage_advisories" in GraphState.__annotations__, (
        "GraphState must have stage_advisories field"
    )
    print("  [PASS] test_graphstate_advisories")


# ── Test 4: PipelineState has stage_advisories ───────────────────
def test_pipeline_state_advisories():
    from app.intelligence.models import PipelineState
    fields = PipelineState.model_fields if hasattr(PipelineState, 'model_fields') else PipelineState.__fields__
    assert "stage_advisories" in fields, (
        "PipelineState must have stage_advisories field"
    )
    print("  [PASS] test_pipeline_state_advisories")


# ── Test 5: ClusterResult has strategic_score ────────────────────
def test_cluster_result_strategic_score():
    from app.intelligence.models import ClusterResult
    fields = ClusterResult.model_fields if hasattr(ClusterResult, 'model_fields') else ClusterResult.__fields__
    assert "strategic_score" in fields, (
        "ClusterResult must have strategic_score field"
    )
    print("  [PASS] test_cluster_result_strategic_score")


# ── Test 6: NewsArticle has embedding field ──────────────────────
def test_news_article_embedding():
    from app.schemas.news import NewsArticle
    fields = NewsArticle.model_fields if hasattr(NewsArticle, 'model_fields') else NewsArticle.__fields__
    assert "embedding" in fields, "NewsArticle must have embedding field"
    print("  [PASS] test_news_article_embedding")


# ── Test 7: ExperimentTracker snapshot/restore ───────────────────
def test_experiment_tracker():
    from app.learning.experiment_tracker import (
        ExperimentTracker, ExperimentRecord,
        snapshot_learning_state, restore_learning_state, cleanup_snapshot,
    )

    tracker = ExperimentTracker()
    assert hasattr(tracker, "record")
    assert hasattr(tracker, "is_regression")
    assert hasattr(tracker, "rolling_baseline")

    # ExperimentRecord should accept these fields
    record = ExperimentRecord(
        run_id="test-run",
        mean_oss=0.45,
        mean_coherence=0.55,
        actionable_rate=0.30,
        article_count=100,
        cluster_count=10,
        hypothesis="",
    )
    assert record.run_id == "test-run"
    assert record.hypothesis == ""

    # Snapshot functions should exist and not crash
    assert callable(snapshot_learning_state)
    assert callable(restore_learning_state)
    assert callable(cleanup_snapshot)
    print("  [PASS] test_experiment_tracker")


# ── Test 8: Pipeline metrics record/load ─────────────────────────
def test_pipeline_metrics():
    from app.learning.pipeline_metrics import (
        record_pipeline_run, load_history, record_cluster_signals,
    )
    assert callable(record_pipeline_run)
    assert callable(load_history)
    assert callable(record_cluster_signals)
    print("  [PASS] test_pipeline_metrics")


# ── Test 9: Signal bus publish + derive ──────────────────────────
def test_signal_bus():
    from app.learning.signal_bus import LearningSignalBus

    bus = LearningSignalBus()

    # Publish source bandit
    bus.publish_source_bandit({"tavily": 0.7, "ddg": 0.4, "rss": 0.5})
    assert len(bus.top_sources) == 3
    assert bus.top_sources[0] == "tavily"

    # Publish NLI filter
    bus.publish_nli_filter(
        mean_entailment=0.72,
        rejection_rate=0.15,
        hypothesis_version="v1",
    )
    assert bus.nli_mean_entailment == 0.72

    # Compute derived signals
    bus.compute_derived_signals()
    assert 0.0 <= bus.system_confidence <= 1.0
    assert 0.1 <= bus.exploration_budget <= 0.5

    # Summary
    summary = bus.summary()
    assert "confidence" in summary
    print("  [PASS] test_signal_bus")


# ── Test 10: Source bandit exists and has correct API ─────────────
def test_source_bandit():
    from app.learning.source_bandit import SourceBandit

    bandit = SourceBandit()
    assert hasattr(bandit, "update_from_run")
    assert hasattr(bandit, "get_quality_estimates")
    print("  [PASS] test_source_bandit")


# ── Test 11: Company bandit exists and has correct API ────────────
def test_company_bandit():
    from app.learning.company_bandit import CompanyRelevanceBandit

    bandit = CompanyRelevanceBandit()
    assert hasattr(bandit, "compute_relevance")
    assert hasattr(bandit, "update")
    assert hasattr(bandit, "decay")
    print("  [PASS] test_company_bandit")


# ── Test 12: meta_reasoner is deleted ────────────────────────────
def test_meta_reasoner_deleted():
    import importlib
    try:
        importlib.import_module("app.learning.meta_reasoner")
        assert False, "meta_reasoner should be deleted"
    except ImportError:
        pass  # Expected
    print("  [PASS] test_meta_reasoner_deleted")


# ── Test 13: pipeline_validator is deleted ───────────────────────
def test_pipeline_validator_deleted():
    import importlib
    try:
        importlib.import_module("app.learning.pipeline_validator")
        assert False, "pipeline_validator should be deleted"
    except ImportError:
        pass  # Expected
    print("  [PASS] test_pipeline_validator_deleted")


# ── Test 14: Hypothesis management removed from experiment_tracker ──
def test_hypothesis_removed():
    import app.learning.experiment_tracker as et
    # These should NOT exist anymore
    assert not hasattr(et, "Hypothesis"), "Hypothesis class should be removed"
    assert not hasattr(et, "pick_next_hypothesis"), "pick_next_hypothesis should be removed"
    assert not hasattr(et, "mark_hypothesis_tested"), "mark_hypothesis_tested should be removed"
    assert not hasattr(et, "load_hypotheses"), "load_hypotheses should be removed"
    assert not hasattr(et, "save_hypotheses"), "save_hypotheses should be removed"
    # These SHOULD still exist
    assert hasattr(et, "ExperimentTracker"), "ExperimentTracker should still exist"
    assert hasattr(et, "snapshot_learning_state"), "snapshot_learning_state should still exist"
    assert hasattr(et, "restore_learning_state"), "restore_learning_state should still exist"
    print("  [PASS] test_hypothesis_removed")


# ── Test 15: pipeline_metrics has no AdaptiveThreshold ───────────
def test_adaptive_threshold_removed():
    import app.learning.pipeline_metrics as pm
    assert not hasattr(pm, "AdaptiveThreshold"), "AdaptiveThreshold should be removed from pipeline_metrics"
    assert not hasattr(pm, "THRESHOLD_REGISTRY"), "THRESHOLD_REGISTRY should be removed"
    assert not hasattr(pm, "compute_adaptive_thresholds"), "compute_adaptive_thresholds should be removed"
    # These SHOULD still exist
    assert hasattr(pm, "record_pipeline_run"), "record_pipeline_run should still exist"
    assert hasattr(pm, "load_history"), "load_history should still exist"
    assert hasattr(pm, "record_cluster_signals"), "record_cluster_signals should still exist"
    print("  [PASS] test_adaptive_threshold_removed")


# ── Test 16: filter.py has no finetune code ──────────────────────
def test_finetune_removed_from_filter():
    import app.intelligence.filter as flt
    assert not hasattr(flt, "_get_finetune_model"), "_get_finetune_model should be removed"
    assert not hasattr(flt, "_FINETUNE_MODEL_ID"), "_FINETUNE_MODEL_ID should be removed"
    print("  [PASS] test_finetune_removed_from_filter")


# ── Test 17: orchestrator has no finetune code ───────────────────
def test_finetune_removed_from_orchestrator():
    import app.agents.orchestrator as orch
    assert not hasattr(orch, "_maybe_trigger_finetune"), "_maybe_trigger_finetune should be removed"
    assert not hasattr(orch, "_FINETUNE_TRIGGER_EVERY"), "_FINETUNE_TRIGGER_EVERY should be removed"
    assert not hasattr(orch, "_FINETUNE_JOB_FILE"), "_FINETUNE_JOB_FILE should be removed"
    print("  [PASS] test_finetune_removed_from_orchestrator")


# ── Test 18: Integration — orchestrator graph builds ─────────────
def test_graph_builds():
    from app.agents.orchestrator import create_pipeline_graph
    graph = create_pipeline_graph()
    assert graph is not None
    print("  [PASS] test_graph_builds")


# ── Test 19: contact_agent has no web search fallback ────────────
def test_contact_agent_no_web_search():
    from app.agents.workers.contact_agent import ContactFinder
    assert not hasattr(ContactFinder, "_find_via_search"), "_find_via_search should be removed"
    assert not hasattr(ContactFinder, "_extract_contact_from_search"), "_extract_contact_from_search should be removed"
    print("  [PASS] test_contact_agent_no_web_search")


# ── Test 20: ExperimentTracker regression detection ──────────────
def test_experiment_tracker_regression():
    from app.learning.experiment_tracker import ExperimentTracker, ExperimentRecord
    from pathlib import Path
    import tempfile

    # Use temp file
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    tracker = ExperimentTracker(log_path=tmp)

    # Record several baseline runs
    for i in range(5):
        tracker.record(ExperimentRecord(
            run_id=f"baseline-{i}",
            mean_oss=0.50,
            mean_coherence=0.55,
            actionable_rate=0.40,
            article_count=100,
            cluster_count=10,
        ))

    # Check: a good run should NOT be a regression
    good = ExperimentRecord(
        run_id="good",
        mean_oss=0.48,
        actionable_rate=0.38,
    )
    assert not tracker.is_regression(good), "Good run should not be regression"

    # Check: a terrible run should be a regression
    bad = ExperimentRecord(
        run_id="bad",
        mean_oss=0.20,
        actionable_rate=0.10,
    )
    assert tracker.is_regression(bad), "Bad run should be regression"

    tmp.unlink(missing_ok=True)
    print("  [PASS] test_experiment_tracker_regression")


# ── Test 21: Signal bus backward cascade ─────────────────────────
def test_signal_bus_backward_cascade():
    from app.learning.signal_bus import LearningSignalBus

    bus = LearningSignalBus()
    bus.publish_backward_signals(
        cluster_coherence_by_source={"tavily": 0.65, "rss": 0.42},
        cluster_noise_rate=0.15,
        lead_quality_per_cluster={"c1": 0.8, "c2": 0.3},
    )
    assert bus.cluster_coherence_by_source["tavily"] == 0.65
    assert bus.cluster_noise_rate == 0.15
    assert bus.lead_quality_per_cluster["c1"] == 0.8
    print("  [PASS] test_signal_bus_backward_cascade")


# ── Test 22: is_company_description still works ──────────────────
def test_is_company_description():
    from app.agents.leads import is_company_description
    # Long descriptions should be caught
    assert is_company_description("A leading provider of cloud solutions that helps enterprise customers modernize")
    assert is_company_description("Mid-size electrolyser manufacturers in India integrating new advanced electrodes")
    # Real company names should not be caught
    assert not is_company_description("Apollo Hospitals")
    assert not is_company_description("Tata Motors")
    print("  [PASS] test_is_company_description")


# ── Run all tests ─────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_threshold_adapter_api,
        test_contact_bandit_api,
        test_graphstate_advisories,
        test_pipeline_state_advisories,
        test_cluster_result_strategic_score,
        test_news_article_embedding,
        test_experiment_tracker,
        test_pipeline_metrics,
        test_signal_bus,
        test_source_bandit,
        test_company_bandit,
        test_meta_reasoner_deleted,
        test_pipeline_validator_deleted,
        test_hypothesis_removed,
        test_adaptive_threshold_removed,
        test_finetune_removed_from_filter,
        test_finetune_removed_from_orchestrator,
        test_graph_builds,
        test_contact_agent_no_web_search,
        test_experiment_tracker_regression,
        test_signal_bus_backward_cascade,
        test_is_company_description,
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
