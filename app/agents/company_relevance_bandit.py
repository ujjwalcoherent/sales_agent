"""
Company Relevance Bandit — Thompson Sampling for company targeting.

Tracks which companies have historically led to successful consulting
engagement signals. Uses the same Beta(alpha, beta) posterior as
the source bandit, persisted across runs in data/company_bandit.json.

Reward sources:
  - 1.0: company appeared in a confirmed causal chain hop (lead sheet generated)
  - 0.5: company appeared in impact analysis with high confidence
  - 0.0: company discarded as too generic / enterprise / not actionable

Not circular: we measure post-pipeline outcomes, not the algorithm itself.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("./data/company_bandit.json")


class CompanyRelevanceBandit:
    """Multi-armed bandit for company targeting quality.

    Posterior mean gives the expected probability this company type
    will produce an actionable lead. New companies start at 0.5 (uniform prior).
    """

    def __init__(self, bandit_path: Path = _DEFAULT_PATH):
        self._path = bandit_path
        self._posteriors: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._posteriors = json.load(f)
                logger.debug(f"Company bandit loaded: {len(self._posteriors)} companies")
            except Exception as e:
                logger.warning(f"Failed to load company bandit: {e}")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._posteriors, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save company bandit: {e}")

    def _ensure(self, company_id: str) -> None:
        if company_id not in self._posteriors:
            self._posteriors[company_id] = {"alpha": 1.0, "beta": 1.0}

    def score(self, company_id: str) -> float:
        """Thompson sample from posterior — used for ranking candidates."""
        if company_id not in self._posteriors:
            return float(np.random.beta(1.0, 1.0))
        p = self._posteriors[company_id]
        return float(np.random.beta(p["alpha"], p["beta"]))

    def posterior_mean(self, company_id: str) -> float:
        """Deterministic expected value (for logging/display)."""
        if company_id not in self._posteriors:
            return 0.5
        p = self._posteriors[company_id]
        return p["alpha"] / (p["alpha"] + p["beta"])

    def update(self, company_id: str, reward: float) -> None:
        """Update posterior with observed reward (0.0–1.0)."""
        self._ensure(company_id)
        p = self._posteriors[company_id]
        p["alpha"] += reward
        p["beta"] += 1.0 - reward
        self._save()

    def rank(self, company_ids: List[str]) -> List[str]:
        """Return company_ids sorted by Thompson sample score (best first)."""
        scored = [(cid, self.score(cid)) for cid in company_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored]

    def compute_relevance(
        self,
        company_size: str,
        event_type: str,
        industry_match: float = 0.5,
        trend_severity: str = "medium",
        intent_signal_strength: float = 0.5,
        explore: bool = True,
    ) -> float:
        """Compute contextual relevance score for a company-trend pairing.

        Combines Thompson Sampling posterior for the (size, event_type) arm
        with contextual features. Returns 0.0–1.0; higher = better fit.

        Args:
            company_size: "startup", "mid", "large", or "enterprise".
            event_type: "regulation", "funding", "technology", etc.
            industry_match: Sector overlap between company and trend (0.0–1.0).
            trend_severity: "high", "medium", or "low".
            intent_signal_strength: Buying intent proxy from article signals (0.0–1.0).
            explore: If True, Thompson-sample (explore); if False, use posterior mean.
        """
        arm_id = f"{company_size}_{event_type}"
        bandit_score = self.score(arm_id) if explore else self.posterior_mean(arm_id)

        severity_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
        severity_mult = severity_map.get(str(trend_severity).lower(), 0.7)

        score = (
            0.35 * bandit_score
            + 0.30 * max(0.0, min(1.0, industry_match))
            + 0.20 * max(0.0, min(1.0, intent_signal_strength))
            + 0.15 * severity_mult
        )
        return min(1.0, max(0.0, score))

    def get_estimates(self) -> Dict[str, float]:
        """Return {company_id: posterior_mean} for all tracked companies."""
        return {
            cid: p["alpha"] / (p["alpha"] + p["beta"])
            for cid, p in self._posteriors.items()
        }
