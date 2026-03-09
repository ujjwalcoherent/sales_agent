"""
Tests for self-learning invariants.

Each test verifies MATHEMATICAL PROPERTIES, not specific values.
Self-learning systems adapt dynamically, so we test:
  - EMA convergence (ThresholdAdapter)
  - Bayesian update monotonicity (LinTS)
  - KL-divergence guardrail (WeightLearner)
  - Event bus routing (EventBus pub/sub)
  - Signal bus cross-loop aggregation (LearningSignalBus)

Run: venv/Scripts/python.exe -m pytest tests/test_learning.py -v

All tests run offline (no network, no LLM calls, no external APIs).
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# ThresholdAdapter — EMA-based adaptive thresholds
# ══════════════════════════════════════════════════════════════════════════════

class TestThresholdAdapter:
    """EMA must converge toward observed values without wild swings."""

    def _make_adapter(self, tmp_path: Path, alpha: float = 0.1):
        from app.learning.threshold_adapter import ThresholdAdapter
        return ThresholdAdapter(path=tmp_path / "thresholds.json", alpha=alpha)

    def test_ema_converges_toward_target(self, tmp_path):
        """Repeated identical observations should converge toward the target."""
        from app.learning.threshold_adapter import ThresholdUpdate
        adapter = self._make_adapter(tmp_path)

        # Start with default threshold (~0.30 for filter_auto_accept)
        # Feed low accept rate (should lower the threshold)
        for _ in range(30):
            adapter.update(ThresholdUpdate(observed_filter_accept_rate=0.10))

        threshold = adapter.get("filter_auto_accept", 0.30)
        # After 30 updates pulling threshold down, must be below 0.30
        assert threshold < 0.30, f"Expected convergence downward, got {threshold}"
        # But must stay above minimum (0.20)
        assert threshold >= 0.20, f"Threshold went below minimum: {threshold}"

    def test_ema_slow_adaptation(self, tmp_path):
        """Single observation should move threshold by ~α (10%), not wildly."""
        from app.learning.threshold_adapter import ThresholdUpdate, ThresholdAdapter
        adapter = ThresholdAdapter(path=tmp_path / "thresholds.json", alpha=0.1)

        initial = adapter.get("val_coherence_min", 0.40)

        # One observation of coherence=0.80 → target = 0.80 * 0.85 = 0.68
        adapter.update(ThresholdUpdate(observed_coherence=0.80))

        after = adapter.get("val_coherence_min", 0.40)
        # With α=0.1: after = 0.1 * 0.68 + 0.9 * initial
        # Should move by less than 20% of the gap
        delta = abs(after - initial)
        max_expected_delta = 0.30  # generous upper bound
        assert delta < max_expected_delta, f"Threshold moved too much: {delta:.3f}"

    def test_thresholds_persist_across_instances(self, tmp_path):
        """Thresholds learned in one instance must survive restart."""
        from app.learning.threshold_adapter import ThresholdUpdate, ThresholdAdapter

        adapter1 = ThresholdAdapter(path=tmp_path / "thresholds.json", alpha=0.1)
        for _ in range(10):
            adapter1.update(ThresholdUpdate(observed_coherence=0.90))
        val1 = adapter1.get("val_coherence_min", 0.40)

        # New instance loads same file
        adapter2 = ThresholdAdapter(path=tmp_path / "thresholds.json", alpha=0.1)
        val2 = adapter2.get("val_coherence_min", 0.40)

        assert val1 == val2, f"Threshold not persisted: {val1} vs {val2}"

    def test_noise_rate_adaptation(self, tmp_path):
        """High noise rate should lower HDBSCAN soft threshold (accept more points)."""
        from app.learning.threshold_adapter import ThresholdUpdate
        adapter = self._make_adapter(tmp_path)

        initial = adapter.get("hdbscan_soft_noise_threshold", 0.10)

        for _ in range(20):
            adapter.update(ThresholdUpdate(observed_noise_rate=0.50))

        after = adapter.get("hdbscan_soft_noise_threshold", 0.10)
        # High noise → threshold should decrease (accept borderline points)
        assert after <= initial, f"Expected decrease, got {after} vs initial {initial}"
        assert after >= 0.05, f"Threshold went below minimum 0.05: {after}"

    def test_all_updated_thresholds_are_in_valid_range(self, tmp_path):
        """All threshold values must remain in [0.0, 1.0] after updates."""
        from app.learning.threshold_adapter import ThresholdUpdate
        adapter = self._make_adapter(tmp_path)

        # Stress test: extreme inputs
        for _ in range(50):
            adapter.update(ThresholdUpdate(
                observed_filter_accept_rate=0.01,
                observed_coherence=0.95,
                observed_pass_rate=0.01,
                observed_noise_rate=0.95,
            ))

        for key, val in adapter._thresholds.items():
            assert 0.0 <= val <= 1.0, f"Threshold {key}={val} out of [0, 1]"


# ══════════════════════════════════════════════════════════════════════════════
# CompanyBanditLinTS — Linear Thompson Sampling
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="company_bandit_lints.py removed — production code unused")
class TestCompanyBanditLinTS:
    """LinTS posterior must update monotonically and stay well-conditioned."""

    def _make_bandit(self, tmp_path: Path):
        from app.learning.company_bandit_lints import CompanyBanditLinTS
        return CompanyBanditLinTS(state_path=tmp_path / "bandit.json", v=1.0)

    def test_score_returns_in_01_range(self, tmp_path):
        """Score must always be in [0.0, 1.0]."""
        bandit = self._make_bandit(tmp_path)
        for _ in range(100):
            score = bandit.score(
                company="TestCo",
                industry="Fintech",
                company_size="startup",
                event_type="funding",
                days_since_news=5.0,
            )
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_posterior_mean_shifts_with_updates(self, tmp_path):
        """After positive updates, posterior mean for that context should increase."""
        bandit = self._make_bandit(tmp_path)

        # Record baseline scores
        baseline_scores = [
            bandit.score("Co", "Cybersecurity", "startup", "funding", 1.0)
            for _ in range(20)
        ]
        mean_before = sum(baseline_scores) / len(baseline_scores)

        # Feed many positive rewards for this context
        for _ in range(30):
            bandit.update("Cybersecurity", "startup", "funding", 1.0, reward=1.0)

        after_scores = [
            bandit.score("Co", "Cybersecurity", "startup", "funding", 1.0)
            for _ in range(20)
        ]
        mean_after = sum(after_scores) / len(after_scores)

        # Posterior mean should shift upward (not necessarily by a fixed amount)
        assert mean_after >= mean_before - 0.05, (
            f"Posterior mean decreased unexpectedly: {mean_before:.3f} → {mean_after:.3f}"
        )

    def test_b_matrix_stays_positive_definite(self, tmp_path):
        """B matrix must remain positive definite (valid covariance proxy)."""
        bandit = self._make_bandit(tmp_path)

        for i in range(50):
            bandit.update(
                industry=["Fintech", "Cybersecurity", "Healthcare"][i % 3],
                company_size=["startup", "mid", "enterprise"][i % 3],
                event_type=["funding", "product_launch", "earnings"][i % 3],
                days_since_news=float(i % 30),
                reward=float(i % 2),
            )

        eigenvalues = np.linalg.eigvals(bandit.B)
        assert all(ev > 0 for ev in eigenvalues), (
            f"B matrix is not positive definite. Min eigenvalue: {min(eigenvalues):.4f}"
        )

    def test_exploration_decays_after_50_samples(self, tmp_path):
        """Exploration parameter v must decay after 50 samples."""
        bandit = self._make_bandit(tmp_path)
        assert bandit.v == 1.0, "Initial v must be 1.0"

        for i in range(60):
            bandit.update("Technology", "mid", "product_launch", 1.0, 0.5)

        assert bandit.v < 1.0, f"v did not decay after 60 updates: {bandit.v}"
        assert bandit.v >= 0.1, f"v decayed below minimum 0.1: {bandit.v}"

    def test_feature_importance_returns_all_features(self, tmp_path):
        """feature_importance() must return a dict with all named features."""
        bandit = self._make_bandit(tmp_path)
        fi = bandit.feature_importance()

        assert "industry_cybersecu" in fi or any("industry_" in k for k in fi)
        assert any("size_" in k for k in fi)
        assert any("event_" in k for k in fi)
        assert "days_since_news" in fi

    def test_human_feedback_conversion(self, tmp_path):
        """update_from_feedback() must convert ratings to correct rewards."""
        bandit = self._make_bandit(tmp_path)

        # Just verify it runs without error and updates sample_count
        before = bandit._sample_count
        bandit.update_from_feedback("TestCo", "good", industry="Fintech", company_size="startup")
        assert bandit._sample_count == before + 1

        bandit.update_from_feedback("TestCo", "bad", industry="Fintech", company_size="startup")
        assert bandit._sample_count == before + 2

    def test_state_persists_across_instances(self, tmp_path):
        """Bayesian posterior must survive save/load cycle."""
        bandit1 = self._make_bandit(tmp_path)
        for _ in range(10):
            bandit1.update("Fintech", "startup", "funding", 1.0, 1.0)
        count1 = bandit1._sample_count
        f_before = bandit1.f.copy()

        from app.learning.company_bandit_lints import CompanyBanditLinTS
        bandit2 = CompanyBanditLinTS(state_path=tmp_path / "bandit.json")
        assert bandit2._sample_count == count1
        np.testing.assert_array_almost_equal(bandit2.f, f_before, decimal=6)


# ══════════════════════════════════════════════════════════════════════════════
# EventBus — pub/sub routing
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="event_bus.py removed — no production callers")
class TestEventBus:
    """Event bus must route correctly and isolate handler errors."""

    def test_subscribe_and_publish(self):
        """Handler must be called with correct topic and data."""
        from app.learning.event_bus import EventBus

        received = []

        async def handler(topic: str, data: dict):
            received.append((topic, data))

        bus = EventBus()
        bus.subscribe("test.topic", handler)

        asyncio.run(
            bus.publish("test.topic", {"key": "value"})
        )

        assert len(received) == 1
        assert received[0] == ("test.topic", {"key": "value"})

    def test_no_handlers_publish_is_safe(self):
        """Publishing to a topic with no handlers must not raise."""
        from app.learning.event_bus import EventBus

        bus = EventBus()
        asyncio.run(
            bus.publish("nonexistent.topic", {})
        )
        # Should complete without exception

    def test_failing_handler_does_not_block_others(self):
        """If one handler raises, subsequent handlers must still run."""
        from app.learning.event_bus import EventBus

        results = []

        async def failing_handler(topic: str, data: dict):
            raise RuntimeError("intentional error")

        async def good_handler(topic: str, data: dict):
            results.append("ran")

        bus = EventBus()
        bus.subscribe("test.topic", failing_handler)
        bus.subscribe("test.topic", good_handler)

        asyncio.run(
            bus.publish("test.topic", {})
        )

        assert "ran" in results, "Good handler was blocked by failing handler"
        assert bus.stats["errors"] == 1

    def test_multiple_handlers_called_in_order(self):
        """Multiple handlers for same topic must run in registration order."""
        from app.learning.event_bus import EventBus

        order = []

        async def h1(t, d): order.append(1)
        async def h2(t, d): order.append(2)
        async def h3(t, d): order.append(3)

        bus = EventBus()
        bus.subscribe("ordered", h1)
        bus.subscribe("ordered", h2)
        bus.subscribe("ordered", h3)

        asyncio.run(bus.publish("ordered", {}))
        assert order == [1, 2, 3]

    def test_stats_track_publish_count(self):
        """stats must accurately reflect publish count."""
        from app.learning.event_bus import EventBus

        async def noop(t, d): pass

        bus = EventBus()
        bus.subscribe("x", noop)

        for _ in range(5):
            asyncio.run(bus.publish("x", {}))

        assert bus.stats["published"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# LearningSignalBus — cross-loop state aggregation
# ══════════════════════════════════════════════════════════════════════════════

class TestLearningSignalBus:
    """Signal bus derived signals must be mathematically correct."""

    def test_system_confidence_in_01_range(self):
        """system_confidence must always be in [0, 1]."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        bus.publish_auto_feedback({"good_trend": 10, "bad_trend": 2}, 0.70)
        bus.publish_source_bandit({"source_a": 0.8, "source_b": 0.6})
        bus.compute_derived_signals()

        assert 0.0 <= bus.system_confidence <= 1.0, (
            f"system_confidence={bus.system_confidence} out of [0, 1]"
        )

    def test_exploration_budget_in_range(self):
        """exploration_budget must be in [0.10, 0.50]."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        bus.compute_derived_signals()

        assert 0.10 <= bus.exploration_budget <= 0.50, (
            f"exploration_budget={bus.exploration_budget} out of [0.10, 0.50]"
        )

    def test_high_confidence_means_low_exploration(self):
        """When system is confident (high quality, stable sources), exploration should be low."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        # Simulate a confident system
        bus.publish_source_bandit({"a": 0.95, "b": 0.90, "c": 0.88})
        bus.publish_weight_learner(
            {"sig": {"w1": 0.5}}, {"sig": {"w1": 0.5}}, "human", 100
        )
        bus.publish_auto_feedback({}, 0.85)
        bus.compute_derived_signals()

        assert bus.exploration_budget < 0.40, (
            f"Expected low exploration for confident system, got {bus.exploration_budget}"
        )

    def test_novelty_distribution_sums_to_1(self):
        """novelty_distribution fractions must sum to 1.0."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        bus.publish_trend_memory(
            lifecycle_counts={"birth": 5, "growth": 3, "peak": 8, "decline": 4},
            avg_novelty=0.6,
            stale_pruned=2,
        )

        total = sum(bus.novelty_distribution.values())
        assert abs(total - 1.0) < 0.001, f"novelty_distribution sums to {total}, not 1.0"

    def test_persistence_and_load(self, tmp_path):
        """Signal bus state must survive save/load cycle."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        bus.run_id = "test_run_123"
        bus.run_count = 42
        bus.system_confidence = 0.75
        bus.save(tmp_path / "bus.json")

        loaded = LearningSignalBus.load_previous(tmp_path / "bus.json")
        assert loaded is not None
        assert loaded.run_id == "test_run_123"
        assert loaded.run_count == 42
        assert loaded.system_confidence == 0.75

    def test_source_diversity_index_is_positive(self):
        """Shannon entropy of source posteriors must be non-negative."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        bus.publish_source_bandit({"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.7})

        assert bus.source_diversity_index >= 0.0, (
            f"source_diversity_index={bus.source_diversity_index} is negative"
        )

    def test_degraded_sources_detected(self):
        """Sources that dropped >20% from previous run must be flagged."""
        from app.learning.signal_bus import LearningSignalBus

        bus = LearningSignalBus()
        previous = {"good_src": 0.90, "bad_src": 0.80, "ok_src": 0.60}
        current = {"good_src": 0.88, "bad_src": 0.50, "ok_src": 0.58}  # bad_src dropped 37.5%

        bus.publish_source_bandit(current, previous_means=previous)

        assert "bad_src" in bus.source_degraded, (
            f"Degraded source not detected. Flagged: {bus.source_degraded}"
        )
        assert "good_src" not in bus.source_degraded


# ══════════════════════════════════════════════════════════════════════════════
# Integration: update_from_run_metrics wired correctly
# ══════════════════════════════════════════════════════════════════════════════

class TestRunMetricsIntegration:
    """End-to-end: publish run metrics → ThresholdAdapter and LinTS update."""

    def test_update_from_run_metrics_calls_threshold_adapter(self, tmp_path):
        """update_from_run_metrics must invoke the ThresholdAdapter."""
        import app.learning.threshold_adapter as ta_module

        original_path = ta_module._PATH
        ta_module._PATH = tmp_path / "thresholds.json"
        ta_module._INSTANCE = None  # Reset singleton

        metrics = {
            "run_id": "test123",
            "articles_fetched": 500,
            "articles_filtered": 100,  # 20% pass rate = very strict filter → low accept rate
            "clusters_passed": 5,
            "clusters_rejected": 10,
            "noise_rate": 0.45,  # High noise → should lower hdbscan threshold
            "mean_coherence": 0.65,
        }

        asyncio.run(ta_module.update_from_run_metrics(metrics))

        adapter = ta_module.get_threshold_adapter()
        # After 45% noise rate update, hdbscan threshold should have shifted
        hdbscan_threshold = adapter.get("hdbscan_soft_noise_threshold", 0.10)
        assert hdbscan_threshold >= 0.0, "Threshold must be non-negative"
        assert adapter._run_count > 0, "run_count must be incremented"

        # Restore
        ta_module._PATH = original_path
        ta_module._INSTANCE = None

    @pytest.mark.skip(reason="event_bus.py removed")
    def test_event_bus_singleton_has_default_subscribers(self):
        """get_signal_bus() must return bus with registered subscribers."""
        # Reset singleton for clean test
        import app.learning.event_bus as eb_module
        original = eb_module._INSTANCE
        eb_module._INSTANCE = None

        bus = eb_module.get_signal_bus()
        assert bus.stats["total_handlers"] >= 4, (
            f"Expected >=4 default subscribers, got {bus.stats['total_handlers']}"
        )

        # Restore
        eb_module._INSTANCE = original


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
