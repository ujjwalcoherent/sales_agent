"""
Cross-Loop Learning Signal Bus — the neural pathways connecting all 6 learning loops.

Instead of swarm/agent routing (adds LLM cost + latency for routing decisions
that are fully determined at design time), this uses a passive shared data structure.

Each loop PUBLISHES summary statistics after updating.
Each loop READS other loops' summaries before/during its computation.
The bus computes DERIVED signals that no single loop can compute alone.

Three-phase protocol prevents circular reads:
  Phase 1 — Each loop updates using THIS run's data, publishes to bus
  Phase 2 — Bus computes cross-loop derived signals
  Phase 3 — Each loop applies small cross-loop adjustments

Persisted to data/signal_bus.json between runs so next run starts warm.

REF: Collaborative Multi-Armed Bandits (Landgren et al., 2016)
     HEBO shared observation table pattern (Cowen-Rivers et al., NeurIPS 2020)
     Multi-task representation sharing (Du et al., NeurIPS 2021)
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
    stale_pruned_count: int = 0

    # ── Weight Learner publishes ────────────────────────────────
    weight_drift: Dict[str, float] = field(default_factory=dict)
    # e.g. {"actionability": 0.03, "trend_score": -0.01, ...}
    learning_path: str = "default"   # "human", "outcome", or "default"
    weight_confidence: float = 0.0   # [0,1] how much data backs current weights
    total_drift: float = 0.0         # sum of abs(drift) across all weight types

    # ── Company Bandit publishes ────────────────────────────────
    company_arm_means: Dict[str, float] = field(default_factory=dict)
    winning_company_types: List[str] = field(default_factory=list)
    company_exploration_rate: float = 0.0

    # ── Adaptive Thresholds publishes ───────────────────────────
    threshold_values: Dict[str, float] = field(default_factory=dict)
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
    nli_hypothesis_updated: bool = False   # Whether hypothesis was updated by SetFit
    nli_scores_by_source: Dict[str, float] = field(default_factory=dict)
    # {source_name: mean_entailment} — used by source bandit reward calculation

    # ── MetaReasoner publishes ───────────────────────────────────
    reasoning_quality: float = 0.0         # Avg quality score across reasoning traces
    reasoning_concerns: List[str] = field(default_factory=list)   # Top concerns
    reasoning_hypotheses: List[str] = field(default_factory=list) # Top improvement ideas
    reasoning_run_grade: str = ""          # Retrospective grade (A-F)
    reasoning_improvement_plan: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_strategy_adjustments: Dict[str, Any] = field(default_factory=dict)

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
        stale_pruned: int = 0,
    ) -> None:
        """Trend Memory publishes lifecycle distribution after update."""
        self.lifecycle_counts = lifecycle_counts
        total = sum(lifecycle_counts.values()) or 1
        self.novelty_distribution = {k: round(v / total, 3) for k, v in lifecycle_counts.items()}
        self.avg_novelty = avg_novelty
        self.stale_pruned_count = stale_pruned

    def publish_weight_learner(
        self,
        current_weights: Dict[str, Dict[str, float]],
        default_weights: Dict[str, Dict[str, float]],
        learning_path: str,
        data_count: int,
    ) -> None:
        """Weight Learner publishes drift and confidence after update."""
        self.learning_path = learning_path

        # Compute per-type drift (total change from defaults)
        total_drift = 0.0
        per_type_drift: Dict[str, float] = {}
        for wt_name, current in current_weights.items():
            defaults = default_weights.get(wt_name, current)
            drift = sum(abs(current.get(k, 0) - defaults.get(k, 0)) for k in defaults)
            per_type_drift[wt_name] = round(drift, 4)
            total_drift += drift

        self.weight_drift = per_type_drift
        self.total_drift = round(total_drift, 4)

        # Confidence: how much data backs the weights (caps at 1.0 after 50 records)
        self.weight_confidence = min(1.0, data_count / 50.0)

    def publish_company_bandit(
        self,
        arm_means: Dict[str, float],
    ) -> None:
        """Company Bandit publishes arm posterior means."""
        self.company_arm_means = arm_means
        self.winning_company_types = sorted(arm_means, key=arm_means.get, reverse=True)[:3]

        uncertain = sum(1 for v in arm_means.values() if 0.35 <= v <= 0.65)
        self.company_exploration_rate = uncertain / max(len(arm_means), 1)

    def publish_adaptive_thresholds(
        self,
        thresholds: Dict[str, float],
        anomalies: List[str],
        drift: List[str],
    ) -> None:
        """Adaptive Thresholds publishes current values and anomalies."""
        self.threshold_values = thresholds
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
        hypothesis_updated: bool = False,
        scores_by_source: Optional[Dict[str, float]] = None,
    ) -> None:
        """NLI Filter publishes quality diagnostics after filtering.

        Called after filter_articles() completes. Provides signals for:
        - Source Bandit: which sources produce high-entailment articles
        - Adaptive Thresholds: should nli_auto_accept be tightened/loosened?
        - MetaReasoner: is filter quality above baseline (>0.60 mean entailment)?
        - Distribution shift detection: if mean drops >10% → trigger SetFit retraining
          (arXiv:2502.12965 — distribution shift survey recommends monitoring input dist.)
        """
        self.nli_mean_entailment = round(mean_entailment, 4)
        self.nli_rejection_rate = round(rejection_rate, 4)
        self.nli_hypothesis_version = hypothesis_version
        self.nli_hypothesis_updated = hypothesis_updated
        self.nli_scores_by_source = scores_by_source or {}

    def publish_reasoning(
        self,
        run_summary: Dict[str, Any],
        retrospective: Optional[Dict[str, Any]] = None,
    ) -> None:
        """MetaReasoner publishes reasoning insights for cross-loop learning.

        Called after the retrospective. Provides:
        - Quality assessments from each step to modulate system confidence
        - Improvement hypotheses that other loops can act on
        - Run grade as an overall signal
        """
        self.reasoning_quality = run_summary.get("avg_reasoning_quality", 0.0)
        self.reasoning_concerns = run_summary.get("top_concerns", [])[:5]
        self.reasoning_hypotheses = run_summary.get("top_hypotheses", [])[:5]
        self.reasoning_strategy_adjustments = run_summary.get("strategy_adjustments", {})

        if retrospective:
            self.reasoning_run_grade = retrospective.get("run_grade", "")
            self.reasoning_improvement_plan = retrospective.get("improvement_plan", [])[:5]

    # ──────────────────────────────────────────────────────────────
    # Phase 2: Compute derived cross-loop signals
    # ──────────────────────────────────────────────────────────────

    def compute_derived_signals(self) -> None:
        """Compute aggregate signals that no single loop can produce alone.

        Must be called AFTER all loops have published (Phase 1 complete).
        """
        # System confidence: weighted blend of loop-level confidence metrics
        signals = []

        # Source stability: low exploration rate = high confidence in sources
        signals.append(1.0 - self.source_exploration_rate)

        # Weight data backing: higher = more data behind learned weights
        signals.append(self.weight_confidence)

        # No anomalies = stable system
        signals.append(1.0 if not self.anomaly_flags else 0.5)

        # Quality above baseline (0.35 is "okay", 0.5+ is "good")
        if self.composite_quality_mean > 0:
            signals.append(min(1.0, self.composite_quality_mean / 0.5))
        else:
            signals.append(0.5)  # No data yet

        # No degraded sources
        degradation_penalty = min(0.5, len(self.source_degraded) * 0.15)
        signals.append(1.0 - degradation_penalty)

        # MetaReasoner quality assessment (if available)
        if self.reasoning_quality > 0:
            signals.append(self.reasoning_quality)

        self.system_confidence = round(
            sum(signals) / max(len(signals), 1), 4
        )

        # Exploration budget: inversely proportional to confidence
        # High confidence → exploit what works. Low confidence → explore more.
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
            "winning_company_types": self.winning_company_types,
            "system_confidence": self.system_confidence,
        }

    def get_weight_learner_modulation(self) -> Dict[str, Any]:
        """Signals for Weight Learner cross-pollination.

        Weight Learner uses these to:
        - Scale learning rate by system confidence (learn slower when uncertain)
        - Factor in source diversity (biased data = conservative learning)
        - Use feedback quality as additional signal
        """
        return {
            "lr_multiplier": max(0.3, self.system_confidence),
            # Low confidence → slower learning (0.3x). High → full speed (1.0x).
            "source_diversity": self.source_diversity_index,
            "company_exploration_high": self.company_exploration_rate > 0.40,
            # High company exploration → lead quality signals are noisy
            "composite_quality_mean": self.composite_quality_mean,
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
            "top_sources": self.top_sources,
        }

    def get_adaptive_threshold_modulation(self) -> Dict[str, Any]:
        """Signals for Adaptive Thresholds cross-pollination.

        Adaptive Thresholds uses these to:
        - Shift with weight drift (weight changes imply score distribution changes)
        - Tighten when novelty is low (recycled trends need higher bar)
        - Account for source diversity
        """
        return {
            "weight_drift": self.total_drift,
            "avg_novelty": self.avg_novelty,
            "source_diversity": self.source_diversity_index,
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
            f"path={self.learning_path}",
            f"drift={self.total_drift:.3f}",
            f"quality={self.composite_quality_mean:.3f}",
        ]
        if self.reasoning_run_grade:
            parts.append(f"grade={self.reasoning_run_grade}")
        if self.source_degraded:
            parts.append(f"degraded={self.source_degraded}")
        if self.anomaly_flags:
            parts.append(f"anomalies={len(self.anomaly_flags)}")
        if self.reasoning_strategy_adjustments:
            parts.append(f"strategy_adjustments={len(self.reasoning_strategy_adjustments)} steps")
        return " | ".join(parts)
