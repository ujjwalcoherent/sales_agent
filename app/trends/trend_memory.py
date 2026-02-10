"""
Historical trend memory — persistent store for past trends.

THE PROBLEM THIS SOLVES:
  Every pipeline run starts from scratch. If "RBI repo rate hike" was a trend
  yesterday, and similar articles appear today, the system treats it as if it's
  seeing this topic for the first time. No continuity, no evolution tracking.

WHAT THIS ENABLES:
  1. CONTINUITY: "This trend has been developing for 5 days" → higher confidence
  2. EVOLUTION: "New development in [past trend]" → links related trends
  3. NOVELTY: Truly new trends get boosted (no historical match = novel)
  4. DECAY: Old trends fade over time (configurable half-life)

HOW IT WORKS:
  - Store: JSON file with trend centroids (mean embedding), titles, dates, metadata
  - Match: New cluster centroid → cosine similarity against stored centroids
  - Score: trend_continuity_score (0-1) based on match strength + age
  - Prune: Auto-remove trends older than max_age_days

DATA STORED PER TREND:
  - centroid: Mean normalized embedding of the cluster (dimension varies by model)
  - title: LLM-generated trend title (for human reference)
  - keywords: Top c-TF-IDF keywords
  - first_seen: ISO datetime (when this trend first appeared)
  - last_seen: ISO datetime (most recent match)
  - seen_count: How many pipeline runs detected this trend
  - peak_score: Highest trend_score ever recorded
  - article_count_total: Cumulative articles across all appearances

PERFORMANCE: <100ms for 1000 stored trends (vectorized cosine similarity).

STORAGE: ~200 bytes per trend → 1000 trends ≈ 200KB. Negligible.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default storage path (relative to project root)
_DEFAULT_MEMORY_PATH = Path("data/trend_memory.json")
_MEMORY_VERSION = 2  # Bumped: now stores embedding_dim for compatibility checks


class TrendMemory:
    """
    Persistent trend memory. Stores centroids + metadata, matches new trends
    against history, and provides continuity/novelty signals.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        similarity_threshold: float = 0.60,
        max_age_days: int = 30,
        decay_half_life_days: float = 7.0,
    ):
        """
        Args:
            storage_path: Path to JSON store file. Created if doesn't exist.
            similarity_threshold: Min cosine similarity to consider a match with past trend.
            max_age_days: Trends older than this are pruned on load.
            decay_half_life_days: Half-life for recency decay of stored trends.
        """
        self.storage_path = storage_path or _DEFAULT_MEMORY_PATH
        self.similarity_threshold = similarity_threshold
        self.max_age_days = max_age_days
        self.decay_half_life_days = decay_half_life_days

        self._trends: List[Dict[str, Any]] = []
        self._centroids: Optional[np.ndarray] = None  # Cached (N, dim) matrix
        self._stored_embedding_dim: Optional[int] = None  # Dim from file
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load from disk on first access."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self):
        """Load trend memory from JSON file. Prunes old trends."""
        if not self.storage_path.exists():
            self._trends = []
            self._centroids = None
            logger.debug(f"Trend memory: no file at {self.storage_path}, starting fresh")
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stored_version = data.get("version", 1)
            if stored_version < _MEMORY_VERSION:
                logger.warning(
                    f"Trend memory version {stored_version} < {_MEMORY_VERSION}, "
                    f"resetting (old data incompatible)"
                )
                self._trends = []
                self._centroids = None
                return

            self._stored_embedding_dim = data.get("embedding_dim")

            raw_trends = data.get("trends", [])

            # Prune old trends
            cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)
            cutoff_iso = cutoff.isoformat()
            self._trends = [
                t for t in raw_trends
                if t.get("last_seen", "") >= cutoff_iso
            ]

            pruned = len(raw_trends) - len(self._trends)
            if pruned > 0:
                logger.info(f"Trend memory: pruned {pruned} expired trends (>{self.max_age_days} days)")

            # Build centroid matrix for fast similarity
            self._rebuild_centroid_matrix()

            logger.info(
                f"Trend memory: loaded {len(self._trends)} trends "
                f"(dim={self._stored_embedding_dim}) from {self.storage_path}"
            )

        except Exception as e:
            logger.warning(f"Failed to load trend memory: {e}")
            self._trends = []
            self._centroids = None

    def _rebuild_centroid_matrix(self):
        """Build/rebuild the normalized centroid matrix for vectorized similarity."""
        if not self._trends:
            self._centroids = None
            return

        # Detect dimension from actual centroid data (not hardcoded)
        detected_dim = None
        for t in self._trends:
            c = t.get("centroid")
            if c and isinstance(c, list) and len(c) > 0:
                detected_dim = len(c)
                break

        if detected_dim is None:
            self._centroids = None
            return

        self._stored_embedding_dim = detected_dim

        centroids = []
        for t in self._trends:
            c = t.get("centroid")
            if c and isinstance(c, list) and len(c) == detected_dim:
                centroids.append(c)
            else:
                centroids.append([0.0] * detected_dim)  # Zero vector for missing/mismatched

        self._centroids = np.array(centroids)
        norms = np.linalg.norm(self._centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._centroids = self._centroids / norms

    def save(self):
        """Persist current state to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Don't persist centroid vectors in the JSON — they're large and rebuilt on load
            # Actually, we DO need them for matching. But strip to 4 decimal places.
            serializable_trends = []
            for t in self._trends:
                st = dict(t)
                if "centroid" in st and isinstance(st["centroid"], list):
                    st["centroid"] = [round(v, 4) for v in st["centroid"]]
                serializable_trends.append(st)

            # Detect embedding dim from centroids
            embedding_dim = None
            for t in serializable_trends:
                c = t.get("centroid")
                if c and isinstance(c, list):
                    embedding_dim = len(c)
                    break

            data = {
                "version": _MEMORY_VERSION,
                "updated_at": datetime.utcnow().isoformat(),
                "embedding_dim": embedding_dim,
                "trend_count": len(self._trends),
                "trends": serializable_trends,
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1, ensure_ascii=False)

            logger.debug(f"Trend memory: saved {len(self._trends)} trends to {self.storage_path}")

        except Exception as e:
            logger.warning(f"Failed to save trend memory: {e}")

    def match_cluster(
        self,
        cluster_centroid: np.ndarray,
        cluster_title: str = "",
    ) -> Dict[str, Any]:
        """
        Match a new cluster against stored trends.

        Args:
            cluster_centroid: Normalized mean embedding of the new cluster.
            cluster_title: Title for logging/debugging.

        Returns:
            Dict with:
              - is_continuation: bool — matches a known past trend
              - continuity_score: float (0-1) — strength of match × recency
              - novelty_score: float (0-1) — inverse of continuity (1=completely new)
              - matched_trend_title: str — title of the matched past trend (if any)
              - matched_trend_age_days: float — how old the matched trend is
              - matched_trend_seen_count: int — how many times we've seen it
        """
        self._ensure_loaded()

        if self._centroids is None or len(self._centroids) == 0:
            return {
                "is_continuation": False,
                "continuity_score": 0.0,
                "novelty_score": 1.0,
                "matched_trend_title": "",
                "matched_trend_age_days": 0.0,
                "matched_trend_seen_count": 0,
            }

        # Ensure centroid is normalized
        centroid = np.array(cluster_centroid).flatten()
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Vectorized cosine similarity against all stored trends
        sims = np.dot(self._centroids, centroid)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.similarity_threshold:
            return {
                "is_continuation": False,
                "continuity_score": 0.0,
                "novelty_score": 1.0,
                "matched_trend_title": "",
                "matched_trend_age_days": 0.0,
                "matched_trend_seen_count": 0,
                "matched_trend_idx": -1,
            }

        matched = self._trends[best_idx]

        # Compute recency decay: how old is the matched trend?
        try:
            last_seen = datetime.fromisoformat(matched["last_seen"])
            age_days = (datetime.utcnow() - last_seen).total_seconds() / 86400
        except Exception:
            age_days = self.max_age_days

        # Exponential decay: recent matches get higher continuity score
        decay = 2.0 ** (-age_days / self.decay_half_life_days)
        continuity_score = best_sim * decay

        logger.debug(
            f"  Trend memory match: '{cluster_title[:40]}' ↔ '{matched.get('title', '')[:40]}' "
            f"sim={best_sim:.3f}, age={age_days:.1f}d, continuity={continuity_score:.3f}"
        )

        return {
            "is_continuation": True,
            "continuity_score": round(continuity_score, 4),
            "novelty_score": round(1.0 - min(1.0, continuity_score), 4),
            "matched_trend_title": matched.get("title", ""),
            "matched_trend_age_days": round(age_days, 1),
            "matched_trend_seen_count": matched.get("seen_count", 1),
            "matched_trend_idx": best_idx,
        }

    def match_clusters_batch(
        self,
        cluster_centroids: Dict[int, np.ndarray],
        cluster_titles: Optional[Dict[int, str]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Batch match all clusters at once (vectorized, fast).

        Args:
            cluster_centroids: {cluster_id: centroid_embedding}
            cluster_titles: {cluster_id: title} for logging

        Returns:
            {cluster_id: match_result_dict} (same format as match_cluster)
        """
        self._ensure_loaded()
        titles = cluster_titles or {}
        results = {}

        if self._centroids is None or len(self._centroids) == 0:
            for cid in cluster_centroids:
                results[cid] = {
                    "is_continuation": False,
                    "continuity_score": 0.0,
                    "novelty_score": 1.0,
                    "matched_trend_title": "",
                    "matched_trend_age_days": 0.0,
                    "matched_trend_seen_count": 0,
                }
            return results

        # Build matrix of new centroids
        cids = list(cluster_centroids.keys())
        if not cids:
            return results

        new_centroids = np.array([cluster_centroids[cid] for cid in cids])
        # Ensure 2D even for single cluster (prevents IndexError on all_sims[i, best_idx])
        if new_centroids.ndim == 1:
            new_centroids = new_centroids.reshape(1, -1)

        # Dimension compatibility check — if stored centroids have different dim
        # than new centroids, matching is impossible. Reset and treat all as novel.
        new_dim = new_centroids.shape[1]
        stored_dim = self._centroids.shape[1]
        if new_dim != stored_dim:
            logger.warning(
                f"Trend memory dimension mismatch: new={new_dim}, stored={stored_dim}. "
                f"Resetting memory (embedding model changed)."
            )
            self._trends = []
            self._centroids = None
            for cid in cluster_centroids:
                results[cid] = {
                    "is_continuation": False,
                    "continuity_score": 0.0,
                    "novelty_score": 1.0,
                    "matched_trend_title": "",
                    "matched_trend_age_days": 0.0,
                    "matched_trend_seen_count": 0,
                }
            return results

        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        new_centroids_norm = new_centroids / norms

        # All-vs-all similarity: (n_new, n_stored)
        all_sims = np.dot(new_centroids_norm, self._centroids.T)

        for i, cid in enumerate(cids):
            best_idx = int(np.argmax(all_sims[i]))
            best_sim = float(all_sims[i, best_idx])

            if best_sim < self.similarity_threshold:
                results[cid] = {
                    "is_continuation": False,
                    "continuity_score": 0.0,
                    "novelty_score": 1.0,
                    "matched_trend_title": "",
                    "matched_trend_age_days": 0.0,
                    "matched_trend_seen_count": 0,
                }
                continue

            matched = self._trends[best_idx]
            try:
                last_seen = datetime.fromisoformat(matched["last_seen"])
                age_days = (datetime.utcnow() - last_seen).total_seconds() / 86400
            except Exception:
                age_days = self.max_age_days

            decay = 2.0 ** (-age_days / self.decay_half_life_days)
            continuity_score = best_sim * decay

            results[cid] = {
                "is_continuation": True,
                "continuity_score": round(continuity_score, 4),
                "novelty_score": round(1.0 - min(1.0, continuity_score), 4),
                "matched_trend_title": matched.get("title", ""),
                "matched_trend_age_days": round(age_days, 1),
                "matched_trend_seen_count": matched.get("seen_count", 1),
            }

        continuations = sum(1 for r in results.values() if r["is_continuation"])
        logger.info(
            f"Trend memory: {continuations}/{len(results)} clusters match past trends, "
            f"{len(results) - continuations} novel"
        )

        return results

    def store_trends(
        self,
        cluster_centroids: Dict[int, np.ndarray],
        cluster_titles: Dict[int, str],
        cluster_keywords: Dict[int, List[str]],
        cluster_scores: Optional[Dict[int, float]] = None,
        cluster_article_counts: Optional[Dict[int, int]] = None,
    ):
        """
        Store current run's trends into memory for future matching.

        Merges with existing trends: if a new trend matches an existing one,
        UPDATE it (bump seen_count, update last_seen, merge keywords).
        If no match, INSERT as new.

        Args:
            cluster_centroids: {cluster_id: mean_embedding}
            cluster_titles: {cluster_id: LLM-generated title}
            cluster_keywords: {cluster_id: [top keywords]}
            cluster_scores: {cluster_id: trend_score} (optional)
            cluster_article_counts: {cluster_id: article_count} (optional)
        """
        self._ensure_loaded()
        now = datetime.utcnow().isoformat()
        scores = cluster_scores or {}
        counts = cluster_article_counts or {}
        stored = 0
        updated = 0

        for cid in cluster_centroids:
            centroid = cluster_centroids[cid]
            title = cluster_titles.get(cid, f"Cluster {cid}")
            keywords = cluster_keywords.get(cid, [])
            score = scores.get(cid, 0.0)
            count = counts.get(cid, 0)

            # Check if this matches an existing trend
            match = self.match_cluster(centroid, title)

            if match["is_continuation"] and match["continuity_score"] > 0.3:
                # UPDATE existing trend — use matched index (not title search)
                matched_idx = match.get("matched_trend_idx", -1)
                if 0 <= matched_idx < len(self._trends):
                    t = self._trends[matched_idx]
                    t["last_seen"] = now
                    t["seen_count"] = t.get("seen_count", 1) + 1
                    t["peak_score"] = max(t.get("peak_score", 0), score)
                    t["article_count_total"] = t.get("article_count_total", 0) + count
                    # Merge keywords (keep unique, limit to 15)
                    existing_kw = set(t.get("keywords", []))
                    existing_kw.update(keywords[:5])
                    t["keywords"] = list(existing_kw)[:15]
                    # Update centroid with exponential moving average
                    old_c = np.array(t.get("centroid", centroid))
                    new_c = np.array(centroid)
                    blended = 0.7 * old_c + 0.3 * new_c
                    norm = np.linalg.norm(blended)
                    if norm > 0:
                        blended = blended / norm
                    t["centroid"] = blended.tolist()
                    updated += 1
                    # Rebuild matrix after update so next match uses fresh data
                    self._rebuild_centroid_matrix()
            else:
                # INSERT new trend
                self._trends.append({
                    "title": title,
                    "keywords": keywords[:10],
                    "centroid": np.array(centroid).flatten().tolist(),
                    "first_seen": now,
                    "last_seen": now,
                    "seen_count": 1,
                    "peak_score": score,
                    "article_count_total": count,
                })
                stored += 1
                # Rebuild matrix after insert so next match sees new entry
                self._rebuild_centroid_matrix()

        logger.info(f"Trend memory: stored {stored} new, updated {updated} existing, total={len(self._trends)}")

        # Auto-save
        self.save()

    @property
    def trend_count(self) -> int:
        """Number of stored trends."""
        self._ensure_loaded()
        return len(self._trends)
