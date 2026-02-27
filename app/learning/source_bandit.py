"""
Source Quality Bandit — Thompson Sampling for RSS source selection.

Each source maintains a Beta(alpha, beta) posterior, persisted in
`data/source_bandit.json`. Articles are scored by clustering contribution;
high-quality sources get higher posterior estimates across runs.

REF: Chapelle & Li 2011, "An Empirical Evaluation of Thompson Sampling"
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BANDIT_PATH = Path("./data/source_bandit.json")


class SourceBandit:
    """Multi-armed bandit for source quality via Thompson Sampling.

    Each source has a Beta(alpha, beta) posterior where
    alpha ~ good outcomes, beta ~ bad outcomes.
    Posterior mean = alpha / (alpha + beta).
    """

    def __init__(self, bandit_path: Path = DEFAULT_BANDIT_PATH):
        self._path = bandit_path
        self._posteriors: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._posteriors = json.load(f)
                logger.debug(
                    f"Source bandit loaded: {len(self._posteriors)} sources"
                )
            except Exception as e:
                logger.warning(f"Failed to load source bandit: {e}")
                self._posteriors = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._posteriors, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save source bandit: {e}")

    def _ensure_source(self, source_id: str) -> None:
        if source_id not in self._posteriors:
            self._posteriors[source_id] = {"alpha": 1.0, "beta": 1.0}

    def update_from_run(
        self,
        source_articles: Dict[str, List[Any]],
        article_labels: Dict[str, int],
        cluster_quality: Dict[int, float],
        dedup_rates: Optional[Dict[str, float]] = None,
        entity_richness: Optional[Dict[str, float]] = None,
        content_quality: Optional[Dict[str, float]] = None,
        cluster_oss: Optional[Dict[int, float]] = None,
        decay_gamma: float = 0.97,
    ) -> Dict[str, float]:
        """Update posteriors from a pipeline run.

        Reward blends pre-clustering signals (uniqueness, entity richness,
        content quality), post-clustering signals (inclusion rate, cluster
        quality), and cross-level OSS (synthesis specificity).

        Returns {source_id: posterior_mean} after update.
        """
        dedup_rates = dedup_rates or {}
        entity_richness = entity_richness or {}
        content_quality = content_quality or {}
        cluster_oss = cluster_oss or {}

        # Decay posteriors toward prior (recency bias, floor 1.5 prevents collapse)
        for source_id in self._posteriors:
            p = self._posteriors[source_id]
            p["alpha"] = max(1.5, 1.0 + (p["alpha"] - 1.0) * decay_gamma)
            p["beta"] = max(1.5, 1.0 + (p["beta"] - 1.0) * decay_gamma)

        for source_id, article_ids in source_articles.items():
            self._ensure_source(source_id)
            if not article_ids:
                continue

            # Pre-clustering signals
            uniqueness = 1.0 - dedup_rates.get(source_id, 0.0)
            raw_richness = entity_richness.get(source_id, 2.5)
            ent_score = min(1.0, raw_richness / 5.0)
            content_score = content_quality.get(source_id, 0.5)

            # Post-clustering signals
            labels = [article_labels.get(aid, -1) for aid in article_ids]
            clustered = sum(1 for l in labels if l >= 0)
            inclusion_rate = clustered / len(labels)
            cluster_ids = [l for l in labels if l >= 0]
            if cluster_ids:
                avg_quality = sum(
                    cluster_quality.get(cid, 0.5) for cid in cluster_ids
                ) / len(cluster_ids)
            else:
                avg_quality = 0.0

            # Cross-level OSS signal (synthesis specificity)
            if cluster_oss and cluster_ids:
                source_oss_scores = [
                    cluster_oss.get(cid, 0.0) for cid in cluster_ids
                    if cid in cluster_oss
                ]
                avg_oss = (
                    sum(source_oss_scores) / len(source_oss_scores)
                    if source_oss_scores else 0.0
                )
            else:
                avg_oss = 0.0

            # Composite reward: 45% pre-clustering + 35% post-clustering + 20% OSS
            reward = (
                0.20 * uniqueness
                + 0.15 * ent_score
                + 0.10 * content_score
                + 0.20 * inclusion_rate
                + 0.15 * avg_quality
                + 0.20 * avg_oss
            )

            # Update Beta posterior (capped at 5 pseudo-observations per run)
            n_obs = min(len(article_ids), 5)
            self._posteriors[source_id]["alpha"] += reward * n_obs
            self._posteriors[source_id]["beta"] += (1.0 - reward) * n_obs

            # Cap total to prevent numerical instability in np.random.beta
            p = self._posteriors[source_id]
            total = p["alpha"] + p["beta"]
            max_total = 200.0
            if total > max_total:
                scale = max_total / total
                p["alpha"] *= scale
                p["beta"] *= scale

        self._save()

        return self.get_quality_estimates()

    def get_quality_estimates(self) -> Dict[str, float]:
        """Return {source_id: posterior_mean} for all sources."""
        return {
            sid: round(p["alpha"] / (p["alpha"] + p["beta"]), 4)
            for sid, p in self._posteriors.items()
        }

    def sample_quality(self, source_id: str) -> float:
        """Sample from Beta(alpha, beta) posterior (Thompson Sampling).

        High-variance samples for uncertain sources drive exploration.
        """
        self._ensure_source(source_id)
        p = self._posteriors[source_id]
        return float(np.random.beta(p["alpha"], p["beta"]))

    def select_sources(
        self,
        candidate_sources: List[str],
        n_select: Optional[int] = None,
    ) -> List[str]:
        """Rank sources by Thompson Sample, return top n_select (or all)."""
        if not candidate_sources:
            return []

        samples = {src: self.sample_quality(src) for src in candidate_sources}
        ranked = sorted(samples.items(), key=lambda x: x[1], reverse=True)

        selected = [src for src, _ in ranked]
        if n_select is not None:
            selected = selected[:n_select]

        logger.debug(
            f"Thompson Sampling selected {len(selected)}/{len(candidate_sources)} sources. "
            f"Top 5: {[(s, f'{v:.3f}') for s, v in ranked[:5]]}"
        )
        return selected

    def get_adaptive_credibility(self, source_id: str) -> float:
        """Return 50/50 blend of static credibility and bandit posterior mean."""
        try:
            from app.config import NEWS_SOURCES
            static = NEWS_SOURCES.get(source_id, {}).get(
                "credibility_score", 0.5
            )
        except Exception:
            static = 0.5

        if source_id not in self._posteriors:
            return static

        p = self._posteriors[source_id]
        posterior_mean = p["alpha"] / (p["alpha"] + p["beta"])
        return round(0.50 * static + 0.50 * posterior_mean, 4)

    @property
    def source_count(self) -> int:
        return len(self._posteriors)
