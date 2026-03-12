"""
Cross-Loop Learning Signal Bus — shared data structure connecting 4 learning loops.

Each loop PUBLISHES summary statistics after updating, READS other loops' summaries
for cross-pollination. The bus computes DERIVED signals (system_confidence,
exploration_budget) that no single loop produces alone.

Three-phase protocol prevents circular reads:
  Phase 1 — Each loop publishes using THIS run's data
  Phase 2 — Bus computes cross-loop derived signals
  Phase 3 — Each loop applies small cross-loop adjustments

Active loops: Source Bandit, Company Bandit, Threshold Adapter, NLI Filter.

Persisted to data/signal_bus.json between runs so next run starts warm.

REF: Collaborative Multi-Armed Bandits (Landgren et al., 2016)
     HEBO shared observation table pattern (Cowen-Rivers et al., NeurIPS 2020)
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BUS_PATH = Path("./data/signal_bus.json")


@dataclass
class LearningSignalBus:
    """Cross-loop signal exchange medium.

    One instance per pipeline run. Each loop writes its section,
    then reads others' sections for cross-pollination adjustments.
    """

    # ── Source Bandit publishes ──────────────────────────────────
    source_posterior_means: Dict[str, float] = field(default_factory=dict)
    source_diversity_index: float = 0.0       # Shannon entropy of posteriors
    top_sources: List[str] = field(default_factory=list)  # Top 5 by quality
    source_exploration_rate: float = 0.0      # % of arms with high variance
    source_degraded: List[str] = field(default_factory=list)  # Sources that dropped >20%

    # ── Trend Memory publishes ──────────────────────────────────
    novelty_distribution: Dict[str, float] = field(default_factory=dict)
    # {"birth": 0.3, "growth": 0.2, "peak": 0.4, "decline": 0.1}
    avg_novelty: float = 0.5
    lifecycle_counts: Dict[str, int] = field(default_factory=dict)

    # ── Company Bandit publishes ────────────────────────────────
    winning_company_types: List[str] = field(default_factory=list)
    company_exploration_rate: float = 0.0

    # ── Adaptive Thresholds publishes ───────────────────────────
    anomaly_flags: List[str] = field(default_factory=list)
    drift_alerts: List[str] = field(default_factory=list)

    # ── Auto-Feedback publishes ─────────────────────────────────
    feedback_distribution: Dict[str, int] = field(default_factory=dict)
    # {"good_trend": 15, "already_knew": 5, "bad_trend": 3}
    composite_quality_mean: float = 0.0

    # ── NLI Filter publishes ─────────────────────────────────────
    # NLI zero-shot classifier (arXiv:1909.00161) replaces keyword salience.
    nli_mean_entailment: float = 0.0       # Mean entailment of kept articles [0,1]
    nli_rejection_rate: float = 0.0        # Fraction of articles NLI auto-rejected
    nli_hypothesis_version: str = "v0"     # Hypothesis version used this run
    nli_scores_by_source: Dict[str, float] = field(default_factory=dict)
    # {source_name: mean_entailment} — used by source bandit reward calculation

    # ── Backward cascade signals (Phase D) ──────────────────────
    # Downstream quality feeds back to upstream loops for cascade learning.
    # REF: Cascade learning (Rendle 2010 BPR), backward reward (Kulkarni et al. 2016 NIPS).
    cluster_coherence_by_source: Dict[str, float] = field(default_factory=dict)
    # {source_name: mean_coherence_of_clusters_using_articles_from_source}
    # Used by source bandit: secondary reward signal (NLI is primary).
    lead_quality_per_cluster: Dict[str, float] = field(default_factory=dict)
    # {cluster_id: mean_lead_score} → company bandit backward reward.

    # ── Cross-loop derived signals (Phase 2) ────────────────────
    system_confidence: float = 0.5   # Overall system health [0,1]
    exploration_budget: float = 0.3  # How aggressively to explore [0.1, 0.5]

    # ── Metadata ────────────────────────────────────────────────
    run_id: str = ""
    timestamp: str = ""
    run_count: int = 0  # Total runs seen (for knowing when stable weights can update)

    # ──────────────────────────────────────────────────────────────
    # Phase 1: Publish methods (called by each loop after its update)
    # ──────────────────────────────────────────────────────────────

    def publish_source_bandit(
        self,
        posterior_means: Dict[str, float],
        previous_means: Optional[Dict[str, float]] = None,
    ) -> None:
        """Source Bandit publishes its state after update."""
        self.source_posterior_means = posterior_means
        self.top_sources = sorted(posterior_means, key=posterior_means.get, reverse=True)[:5]

        # Shannon entropy of posteriors (higher = more diverse, lower = converged)
        values = list(posterior_means.values())
        if values:
            total = sum(values) or 1.0
            probs = [v / total for v in values if v > 0]
            self.source_diversity_index = -sum(p * math.log(p + 1e-10) for p in probs)
        else:
            self.source_diversity_index = 0.0

        # Exploration rate: fraction of sources with posterior mean in [0.35, 0.65]
        # (uncertain = still exploring)
        uncertain = sum(1 for v in values if 0.35 <= v <= 0.65)
        self.source_exploration_rate = uncertain / max(len(values), 1)

        # Detect degraded sources (dropped >20% from previous run)
        self.source_degraded = []
        if previous_means:
            for sid, current in posterior_means.items():
                prev = previous_means.get(sid, current)
                if prev > 0 and (prev - current) / prev > 0.20:
                    self.source_degraded.append(sid)

    def publish_trend_memory(
        self,
        lifecycle_counts: Dict[str, int],
        avg_novelty: float,
    ) -> None:
        """Trend Memory publishes lifecycle distribution after update."""
        self.lifecycle_counts = lifecycle_counts
        total = sum(lifecycle_counts.values()) or 1
        self.novelty_distribution = {k: round(v / total, 3) for k, v in lifecycle_counts.items()}
        self.avg_novelty = avg_novelty

    def publish_company_bandit(
        self,
        arm_means: Dict[str, float],
    ) -> None:
        """Company Bandit publishes arm posterior means."""
        self.winning_company_types = sorted(arm_means, key=arm_means.get, reverse=True)[:3]

        uncertain = sum(1 for v in arm_means.values() if 0.35 <= v <= 0.65)
        self.company_exploration_rate = uncertain / max(len(arm_means), 1)

    def publish_adaptive_thresholds(
        self,
        anomalies: List[str],
        drift: List[str],
    ) -> None:
        """Adaptive Thresholds publishes anomalies and drift alerts."""
        self.anomaly_flags = anomalies
        self.drift_alerts = drift

    def publish_auto_feedback(
        self,
        distribution: Dict[str, int],
        mean_quality: float,
    ) -> None:
        """Auto-Feedback publishes quality distribution."""
        self.feedback_distribution = distribution
        self.composite_quality_mean = round(mean_quality, 4)

    def publish_nli_filter(
        self,
        mean_entailment: float,
        rejection_rate: float,
        hypothesis_version: str,
        scores_by_source: Optional[Dict[str, float]] = None,
    ) -> None:
        """NLI Filter publishes quality diagnostics after filtering.

        Called after filter_articles() completes. Provides signals for:
        - Source Bandit: which sources produce high-entailment articles
        - Adaptive Thresholds: should nli_auto_accept be tightened/loosened?
        - Distribution shift detection: if mean drops >10% → log warning.
          (arXiv:2502.12965 — distribution shift survey recommends monitoring input dist.)
        """
        self.nli_mean_entailment = round(mean_entailment, 4)
        self.nli_rejection_rate = round(rejection_rate, 4)
        self.nli_hypothesis_version = hypothesis_version
        self.nli_scores_by_source = scores_by_source or {}

    def publish_backward_signals(
        self,
        cluster_coherence_by_source: Optional[Dict[str, float]] = None,
        lead_quality_per_cluster: Optional[Dict[str, float]] = None,
    ) -> None:
        """Publish cascade backward signals (Phase D).

        Called after clustering + lead scoring complete. Propagates downstream
        quality back to upstream loops:
        - cluster_coherence_by_source → source bandit secondary reward
        - lead_quality_per_cluster → company bandit backward reward
        """
        if cluster_coherence_by_source:
            self.cluster_coherence_by_source = cluster_coherence_by_source
        if lead_quality_per_cluster:
            self.lead_quality_per_cluster = lead_quality_per_cluster

    # ──────────────────────────────────────────────────────────────
    # Phase 2: Compute derived cross-loop signals
    # ──────────────────────────────────────────────────────────────

    def compute_derived_signals(self) -> None:
        """Compute aggregate signals that no single loop can produce alone.

        Must be called AFTER all loops have published (Phase 1 complete).
        system_confidence: blend of source stability, quality, anomaly flags.
        exploration_budget: inversely proportional to confidence.
        """
        signals = []

        # Source stability: low exploration rate = high confidence in sources
        signals.append(1.0 - self.source_exploration_rate)

        # Company bandit stability: low exploration = converged company preferences
        signals.append(1.0 - self.company_exploration_rate)

        # No anomalies or drift = stable system
        anomaly_count = len(self.anomaly_flags) + len(self.drift_alerts)
        signals.append(1.0 if anomaly_count == 0 else max(0.3, 1.0 - anomaly_count * 0.2))

        # Source diversity: Shannon entropy collapse → over-concentration risk
        # Normalise to [0,1]: log(n) is max entropy for n sources.
        if self.source_diversity_index > 0 and self.source_posterior_means:
            max_entropy = math.log(max(len(self.source_posterior_means), 2))
            signals.append(min(1.0, self.source_diversity_index / max_entropy))

        # Quality above baseline (0.35 is "okay", 0.5+ is "good")
        if self.composite_quality_mean > 0:
            signals.append(min(1.0, self.composite_quality_mean / 0.5))
        else:
            signals.append(0.5)  # No data yet

        # No degraded sources
        degradation_penalty = min(0.5, len(self.source_degraded) * 0.15)
        signals.append(1.0 - degradation_penalty)

        # NLI filter quality — high entailment = filter is working
        if self.nli_mean_entailment > 0:
            signals.append(min(1.0, self.nli_mean_entailment / 0.7))

        self.system_confidence = round(
            sum(signals) / max(len(signals), 1), 4
        )

        # Exploration budget: inversely proportional to confidence
        self.exploration_budget = round(
            max(0.10, min(0.50, 1.0 - self.system_confidence)), 4
        )

        logger.info(
            f"Signal bus derived: confidence={self.system_confidence:.3f}, "
            f"exploration_budget={self.exploration_budget:.3f}, "
            f"degraded_sources={len(self.source_degraded)}, "
            f"anomalies={len(self.anomaly_flags)}"
        )

    # ──────────────────────────────────────────────────────────────
    # Phase 3: Cross-pollination helpers (consumed by loops)
    # ──────────────────────────────────────────────────────────────

    def get_source_bandit_modulation(self) -> Dict[str, Any]:
        """Signals for Source Bandit cross-pollination.

        Source Bandit uses these to:
        - Reward sources producing novel (birth) trends more
        - Adjust exploration based on system exploration budget
        - Factor in which company types are winning
        """
        birth_ratio = self.novelty_distribution.get("birth", 0.0)
        return {
            "novelty_bonus_active": birth_ratio < 0.30,
            # If most trends are stale, sources producing novel ones get a bonus
            "exploration_budget": self.exploration_budget,
        }

    def get_company_bandit_modulation(self) -> Dict[str, Any]:
        """Signals for Company Bandit cross-pollination.

        Company Bandit uses these to:
        - Explore more for 'birth' lifecycle trends
        - Use source quality as a prior for company quality
        """
        return {
            "exploration_budget": self.exploration_budget,
            "birth_trend_ratio": self.novelty_distribution.get("birth", 0.0),
        }

    # ──────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────

    def save(self, path: Path = _BUS_PATH) -> None:
        """Persist bus state so next run can read previous cross-loop signals."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Signal bus persisted to {path}")
        except Exception as e:
            logger.warning(f"Signal bus save failed: {e}")

    @classmethod
    def load_previous(cls, path: Path = _BUS_PATH) -> Optional["LearningSignalBus"]:
        """Load previous run's bus state for cross-run continuity."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bus = cls()
            for key, value in data.items():
                if hasattr(bus, key):
                    setattr(bus, key, value)
            logger.debug(f"Signal bus loaded from previous run (run_count={bus.run_count})")
            return bus
        except Exception as e:
            logger.warning(f"Signal bus load failed: {e}")
            return None

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = [
            f"confidence={self.system_confidence:.2f}",
            f"explore={self.exploration_budget:.2f}",
            f"quality={self.composite_quality_mean:.3f}",
            f"nli={self.nli_mean_entailment:.2f}",
        ]
        if self.source_degraded:
            parts.append(f"degraded={self.source_degraded}")
        if self.anomaly_flags:
            parts.append(f"anomalies={len(self.anomaly_flags)}")
        return " | ".join(parts)
