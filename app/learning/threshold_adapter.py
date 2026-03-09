"""
ThresholdAdapterAgent — Adaptive threshold learning via EMA (α=0.1).

Updates all ClusteringParams thresholds after each run based on observed
optimal values. Uses Exponential Moving Average for stability.

Formula: threshold_t = α * observed_optimal_t + (1 - α) * threshold_{t-1}
  α=0.1: 10% weight to current run, 90% to history.
  This prevents wild swings from a single bad run.

Persisted to: data/adaptive_thresholds.json
Loaded by: intelligence/config.py::load_adaptive_params()

Thresholds adapted:
  - filter_auto_accept: salience threshold for auto-accept
  - filter_auto_reject: salience threshold for auto-reject
  - val_coherence_min: cluster coherence threshold
  - val_entity_consistency_min: entity coverage threshold
  - dedup_title_threshold: title dedup cosine threshold
  - hdbscan_soft_noise_threshold: soft membership cutoff

REF: Exponential Moving Average — standard practice for online learning.
     α=0.1 selected per: Sutton & Barto "Reinforcement Learning" Ch. 2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PATH = Path("./data/adaptive_thresholds.json")
_EMA_ALPHA = 0.1    # Slow adaptation — stability over responsiveness


@dataclass
class ThresholdUpdate:
    """Observed optimal thresholds from one pipeline run."""
    # Filter
    observed_filter_accept_rate: Optional[float] = None    # fraction auto-accepted
    observed_filter_reject_rate: Optional[float] = None    # fraction auto-rejected

    # Clustering quality
    observed_coherence: Optional[float] = None             # mean cluster coherence
    observed_pass_rate: Optional[float] = None             # fraction clusters passing validation

    # Noise
    observed_noise_rate: Optional[float] = None            # fraction articles in noise

    # Meta
    run_id: str = ""
    source: str = "auto"   # "auto" or "human_feedback"


class ThresholdAdapter:
    """EMA-based threshold adaptation for all pipeline thresholds.

    Called after each pipeline run with observed quality metrics.
    Gradually shifts thresholds toward values that produce better outputs.
    """

    def __init__(self, path: Path = _PATH, alpha: float = _EMA_ALPHA):
        self.path = path
        self.alpha = alpha
        self._thresholds: Dict[str, float] = {}
        self._run_count: int = 0
        self._load()

    def get(self, key: str, default: float) -> float:
        """Get current adapted threshold (or default if not yet learned)."""
        return self._thresholds.get(key, default)

    def update(self, update: ThresholdUpdate) -> None:
        """Apply EMA update from one pipeline run.

        Only updates thresholds where we have a meaningful observed signal.
        """
        self._run_count += 1
        changed = False

        # ── Filter thresholds ─────────────────────────────────────────────────
        # If accept rate is very low (< 20%), auto-accept threshold is too strict → lower it
        if update.observed_filter_accept_rate is not None:
            if update.observed_filter_accept_rate < 0.2:
                target = max(0.20, self.get("filter_auto_accept", 0.30) - 0.02)
                self._ema_update("filter_auto_accept", target)
                changed = True
            elif update.observed_filter_accept_rate > 0.8:
                # Too many articles accepted — threshold too loose
                target = min(0.50, self.get("filter_auto_accept", 0.30) + 0.02)
                self._ema_update("filter_auto_accept", target)
                changed = True

        # ── Coherence threshold ───────────────────────────────────────────────
        # Adjust coherence min based on observed mean coherence
        if update.observed_coherence is not None and update.observed_coherence > 0.0:
            # Target: threshold = 0.85 * observed_mean (allows some spread)
            target_coherence = update.observed_coherence * 0.85
            target_coherence = max(0.30, min(0.65, target_coherence))
            self._ema_update("val_coherence_min", target_coherence)
            changed = True

        # ── Pass rate threshold ───────────────────────────────────────────────
        # If pass rate is too low (< 30%), validation is too strict
        if update.observed_pass_rate is not None:
            if update.observed_pass_rate < 0.30:
                # Loosen composite reject threshold
                target = max(0.30, self.get("val_composite_reject", 0.50) - 0.03)
                self._ema_update("val_composite_reject", target)
                changed = True
            elif update.observed_pass_rate > 0.80:
                # Too many passing — tighten
                target = min(0.70, self.get("val_composite_reject", 0.50) + 0.03)
                self._ema_update("val_composite_reject", target)
                changed = True

        # ── Noise rate threshold ──────────────────────────────────────────────
        # If noise rate is high (> 40%), HDBSCAN soft threshold may be too strict
        if update.observed_noise_rate is not None:
            if update.observed_noise_rate > 0.40:
                # Lower soft noise threshold to accept more borderline points
                target = max(0.05, self.get("hdbscan_soft_noise_threshold", 0.10) - 0.01)
                self._ema_update("hdbscan_soft_noise_threshold", target)
                changed = True
            elif update.observed_noise_rate < 0.10:
                # Noise rate very low — maybe too permissive
                target = min(0.20, self.get("hdbscan_soft_noise_threshold", 0.10) + 0.01)
                self._ema_update("hdbscan_soft_noise_threshold", target)
                changed = True

        if changed:
            self._save()
            coh = self._thresholds.get("val_coherence_min")
            fac = self._thresholds.get("filter_auto_accept")
            coh_str = f"{coh:.3f}" if coh is not None else "N/A"
            fac_str = f"{fac:.3f}" if fac is not None else "N/A"
            logger.info(
                f"[threshold_adapter] Run #{self._run_count}: thresholds updated "
                f"(coherence={coh_str}, filter_accept={fac_str})"
            )

    def _ema_update(self, key: str, observed: float) -> None:
        """Apply EMA: new = α * observed + (1-α) * current."""
        current = self._thresholds.get(key, observed)  # use observed as prior if no history
        updated = self.alpha * observed + (1 - self.alpha) * current
        self._thresholds[key] = round(updated, 4)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._thresholds = data.get("thresholds", {})
            self._run_count = data.get("run_count", 0)
            logger.debug(f"[threshold_adapter] Loaded {len(self._thresholds)} thresholds (runs={self._run_count})")
        except Exception as exc:
            logger.warning(f"[threshold_adapter] Load failed: {exc}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({
                    "thresholds": self._thresholds,
                    "run_count": self._run_count,
                    "alpha": self.alpha,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as exc:
            logger.warning(f"[threshold_adapter] Save failed: {exc}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_INSTANCE: Optional[ThresholdAdapter] = None


def get_threshold_adapter() -> ThresholdAdapter:
    """Get or create the ThresholdAdapter singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ThresholdAdapter()
    return _INSTANCE


async def update_from_run_metrics(metrics: Dict[str, Any]) -> None:
    """Update thresholds from Signal Bus intelligence.run.complete event."""
    adapter = get_threshold_adapter()
    update = ThresholdUpdate(
        observed_filter_accept_rate=metrics.get("articles_filtered", 0) / max(metrics.get("articles_fetched", 1), 1),
        observed_coherence=metrics.get("mean_coherence"),
        observed_pass_rate=(
            metrics.get("clusters_passed", 0) /
            max(metrics.get("clusters_passed", 0) + metrics.get("clusters_rejected", 0), 1)
        ),
        observed_noise_rate=metrics.get("noise_rate"),
        run_id=metrics.get("run_id", ""),
    )
    adapter.update(update)
