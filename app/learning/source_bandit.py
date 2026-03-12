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

# Category tags that indicate B2B-signal-rich or noise-rich sources
_B2B_SIGNAL_TAGS = frozenset({
    "funding", "VC", "PE", "M&A", "startup", "fintech", "payments",
    "press_releases", "B2B", "SaaS", "enterprise", "IPO", "unicorn",
    "NBFC", "banking", "insurance",
})
_NOISE_TAGS = frozenset({
    "geopolitical", "macro", "trade", "aggregator",
})


def _informed_prior(source_id: str) -> tuple[float, float]:
    """Return Beta(alpha, beta) informed prior for a source.

    Uses NEWS_SOURCES category metadata to set an informative prior:
    - B2B signal sources (funding/VC/startup): Beta(3,1) = prior mean 0.75
    - Noise-prone sources (macro/geo/policy):  Beta(1,3) = prior mean 0.25
    - Unknown / general sources:               Beta(1,1) = prior mean 0.50

    Research: Russo et al. 2018 "A Tutorial on Thompson Sampling" (arXiv:1707.02038)
    — informed priors accelerate Thompson Sampling convergence significantly.
    """
    try:
        from app.config import NEWS_SOURCES
        cats = set(NEWS_SOURCES.get(source_id, {}).get("categories", []))
    except Exception:
        return 1.0, 1.0

    if cats & _B2B_SIGNAL_TAGS:
        return 3.0, 1.0   # prior mean = 0.75
    if cats & _NOISE_TAGS:
        return 1.0, 3.0   # prior mean = 0.25
    return 1.0, 1.0        # uninformed prior mean = 0.50


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
            # Informed prior from source category metadata (not blind 0.5)
            # B2B-signal sources start at Beta(3,1)=0.75; noise sources at Beta(1,3)=0.25
            # Research: informed priors in Thompson Sampling → faster convergence
            # (Russo et al. 2018, "A Tutorial on Thompson Sampling", arXiv:1707.02038)
            alpha, beta = _informed_prior(source_id)
            self._posteriors[source_id] = {"alpha": alpha, "beta": beta}

    def update_from_run(
        self,
        source_articles: Dict[str, List[Any]],
        article_labels: Dict[str, int],
        cluster_quality: Dict[int, float],
        dedup_rates: Optional[Dict[str, float]] = None,
        entity_richness: Optional[Dict[str, float]] = None,
        content_quality: Optional[Dict[str, float]] = None,
        cluster_oss: Optional[Dict[int, float]] = None,
        nli_scores_by_source: Optional[Dict[str, float]] = None,
        cluster_coherence_by_source: Optional[Dict[str, float]] = None,
        decay_gamma: float = 0.97,
    ) -> Dict[str, float]:
        """Update posteriors from a pipeline run.

        Reward = blend of forward signals (NLI, uniqueness, entity richness)
        and backward cascade signal (cluster_coherence_by_source).

        Forward signals (computed during this run):
          - NLI entailment (30%): does this source produce B2B articles?
          - Cluster quality (20%): do articles form meaningful clusters?
          - Uniqueness (15%): low duplicate rate
          - Entity richness (10%): named entities per article
          - Content quality (10%): article text quality
          - OSS (5%): synthesis specificity

        Backward cascade signal (from downstream):
          - cluster_coherence_by_source (10%): mean coherence of clusters
            containing this source's articles. Low coherence → articles scatter
            into noise → penalize source. This completes the feedback loop:
            source → filter → cluster → coherence → source reward.
            REF: Cascade learning (Rendle 2010 BPR).

        Returns {source_id: posterior_mean} after update.
        """
        dedup_rates = dedup_rates or {}
        entity_richness = entity_richness or {}
        content_quality = content_quality or {}
        cluster_oss = cluster_oss or {}
        nli_scores_by_source = nli_scores_by_source or {}
        cluster_coherence_by_source = cluster_coherence_by_source or {}

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

            # NLI relevance: mean entailment for this source's articles.
            # PRIMARY signal — sources with high B2B entailment get rewarded.
            nli_score = nli_scores_by_source.get(source_id, 0.40)

            # Post-clustering signals
            labels = [article_labels.get(aid, -1) for aid in article_ids]
            cluster_ids = [l for l in labels if l >= 0]
            if cluster_ids:
                avg_quality = sum(
                    cluster_quality.get(cid, 0.5) for cid in cluster_ids
                ) / len(cluster_ids)
            else:
                avg_quality = 0.0

            # OSS (synthesis specificity)
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

            # Backward cascade: cluster coherence for this source's articles.
            # If articles scatter into incoherent clusters → low reward.
            backward_coh = cluster_coherence_by_source.get(source_id, 0.50)

            # Composite reward (weights sum to 1.0)
            reward = (
                0.30 * nli_score          # PRIMARY: B2B relevance
                + 0.20 * avg_quality      # Cluster quality (forward)
                + 0.15 * uniqueness       # Deduplication quality
                + 0.10 * ent_score        # Entity richness
                + 0.10 * content_score    # Content quality
                + 0.10 * backward_coh    # BACKWARD CASCADE: cluster coherence
                + 0.05 * avg_oss          # Synthesis specificity
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

