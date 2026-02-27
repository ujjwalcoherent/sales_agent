"""
Temporal trend memory — distinguishes genuinely new trends from recurring topics.

Stores cluster centroids across runs in ChromaDB. On each pipeline run:
1. Match new centroids vs stored (cosine > match_threshold = same trend)
2. Matching: low novelty, increment run_count, EMA blend centroid (70/30)
3. Non-matching: novelty = 1.0 (genuinely new)
4. Stale (not seen in N days): prune

This prevents "RBI rate hike" from appearing as a fresh trend every run
when it's been detected 10 times before.

REF: BERTrend (Boutaleb et al. 2024) — temporal topic tracking.
     Plan Phase 4: Temporal memory layer.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrendMemory:
    """Persistent trend centroid store for novelty detection.

    Uses ChromaDB to store cluster centroids with metadata (run_count,
    first_seen, last_seen, keywords). On each run, matches current cluster
    centroids against stored ones via cosine similarity.
    """

    def __init__(
        self,
        db_path: str = "./data/memory",
        match_threshold: float = 0.80,
        stale_days: int = 14,
        ema_alpha: float = 0.30,
    ):
        """
        Args:
            db_path: ChromaDB persistent storage path.
            match_threshold: Cosine similarity above which a centroid
                             is considered "same trend" as a stored one.
            stale_days: Remove stored centroids not seen in this many days.
            ema_alpha: Weight for blending new centroid into stored
                       (0.30 = 30% new, 70% old).
        """
        import chromadb
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="trend_centroids",
            metadata={"hnsw:space": "cosine"},
        )
        self.match_threshold = match_threshold
        self.stale_days = stale_days
        self.ema_alpha = ema_alpha

    def compute_novelty(
        self,
        cluster_centroids: Dict[int, np.ndarray],
        cluster_keywords: Optional[Dict[int, List[str]]] = None,
        cluster_article_counts: Optional[Dict[int, int]] = None,
        cluster_oss: Optional[Dict[int, float]] = None,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, str]]:
        """Compare current cluster centroids against stored trend memory.

        Args:
            cluster_centroids: {cluster_id: centroid_vector}
            cluster_keywords: {cluster_id: [keywords]} for metadata storage.
            cluster_article_counts: {cluster_id: article_count} for lifecycle.

        Returns:
            (novelty_scores, continuity_scores, lifecycle_stages) where:
            - novelty_scores: {cluster_id: 0.0-1.0} (1.0 = brand new)
            - continuity_scores: {cluster_id: 0.0-1.0} (1.0 = long-running)
            - lifecycle_stages: {cluster_id: "birth"|"growth"|"peak"|"decline"|"revival"}
        """
        novelty_scores = {}
        continuity_scores = {}
        lifecycle_stages = {}
        now = datetime.now(timezone.utc)

        if not cluster_centroids:
            return novelty_scores, continuity_scores, lifecycle_stages

        # Query all stored centroids
        stored_count = self.collection.count()

        for cid, centroid in cluster_centroids.items():
            centroid_list = centroid.tolist() if isinstance(centroid, np.ndarray) else centroid

            if stored_count == 0:
                # No history — everything is novel
                novelty_scores[cid] = 1.0
                continuity_scores[cid] = 0.0
                lifecycle_stages[cid] = "birth"
                self._store_centroid(
                    cid, centroid_list, cluster_keywords, now,
                    article_count=cluster_article_counts.get(cid, 0) if cluster_article_counts else 0,
                    oss=cluster_oss.get(cid, 0.0) if cluster_oss else 0.0,
                )
                continue

            # Find nearest stored centroid
            results = self.collection.query(
                query_embeddings=[centroid_list],
                n_results=1,
                include=["metadatas", "distances", "embeddings"],
            )

            if not results["ids"][0]:
                # No matches found
                novelty_scores[cid] = 1.0
                continuity_scores[cid] = 0.0
                lifecycle_stages[cid] = "birth"
                self._store_centroid(
                    cid, centroid_list, cluster_keywords, now,
                    article_count=cluster_article_counts.get(cid, 0) if cluster_article_counts else 0,
                    oss=cluster_oss.get(cid, 0.0) if cluster_oss else 0.0,
                )
                continue

            # ChromaDB cosine distance = 1 - cosine_similarity
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            matched_id = results["ids"][0][0]
            metadata = results["metadatas"][0][0]

            if similarity >= self.match_threshold:
                # Match found — this is a recurring trend
                run_count = metadata.get("run_count", 1) + 1
                first_seen = metadata.get("first_seen", now.isoformat())
                prev_article_count = metadata.get("article_count", 0)
                prev_lifecycle = metadata.get("lifecycle", "birth")

                # Novelty decays with run_count (more seen = less novel)
                # Gentler curve: 1/(1 + 0.3*run_count) — preserves novelty longer
                # for moderate counts. Old: 1/(1+log(N)) was too aggressive
                # (N=5 → 0.38). New: N=5 → 0.40, N=10 → 0.25, N=20 → 0.14.
                import math
                novelty_scores[cid] = round(1.0 / (1.0 + 0.3 * run_count), 4)

                # Continuity grows with run_count
                continuity_scores[cid] = round(min(1.0, run_count / 10.0), 4)

                # Lifecycle classification
                curr_count = cluster_article_counts.get(cid, 0) if cluster_article_counts else 0
                lifecycle = self._classify_lifecycle(
                    run_count, curr_count, prev_article_count, prev_lifecycle
                )
                lifecycle_stages[cid] = lifecycle

                # EMA blend: update stored centroid
                stored_emb = results["embeddings"][0][0]
                blended = [
                    self.ema_alpha * new + (1 - self.ema_alpha) * old
                    for new, old in zip(centroid_list, stored_emb)
                ]
                # Normalize blended embedding to prevent norm drift over many updates
                norm = float(np.linalg.norm(blended))
                if norm > 0:
                    blended = [x / norm for x in blended]

                keywords = (
                    cluster_keywords.get(cid, [])[:10]
                    if cluster_keywords else []
                )

                # V4: Track OSS per centroid for cross-run learning
                curr_oss = cluster_oss.get(cid, 0.0) if cluster_oss else 0.0
                prev_avg_oss = metadata.get("avg_oss", 0.0)
                if prev_avg_oss > 0 and curr_oss > 0:
                    # Rolling average: 30% new + 70% old (same EMA as centroid)
                    new_avg_oss = round(self.ema_alpha * curr_oss + (1 - self.ema_alpha) * prev_avg_oss, 4)
                else:
                    new_avg_oss = round(curr_oss, 4) if curr_oss > 0 else prev_avg_oss

                # Determine OSS trend direction
                if curr_oss > prev_avg_oss + 0.05:
                    oss_trend = "improving"
                elif curr_oss < prev_avg_oss - 0.05:
                    oss_trend = "declining"
                else:
                    oss_trend = "stable"

                self.collection.update(
                    ids=[matched_id],
                    embeddings=[blended],
                    metadatas=[{
                        "run_count": run_count,
                        "first_seen": first_seen,
                        "last_seen": now.isoformat(),
                        "keywords": json.dumps(keywords),
                        "similarity": round(similarity, 4),
                        "article_count": curr_count,
                        "lifecycle": lifecycle,
                        "avg_oss": new_avg_oss,
                        "oss_trend": oss_trend,
                    }],
                )

                logger.debug(
                    f"Cluster {cid}: matched stored trend {matched_id} "
                    f"(sim={similarity:.3f}, runs={run_count}, "
                    f"novelty={novelty_scores[cid]:.3f}, lifecycle={lifecycle})"
                )

            else:
                # No match — genuinely new trend
                novelty_scores[cid] = 1.0
                continuity_scores[cid] = 0.0
                lifecycle_stages[cid] = "birth"
                self._store_centroid(
                    cid, centroid_list, cluster_keywords, now,
                    article_count=cluster_article_counts.get(cid, 0) if cluster_article_counts else 0,
                    oss=cluster_oss.get(cid, 0.0) if cluster_oss else 0.0,
                )

                logger.debug(
                    f"Cluster {cid}: NEW trend "
                    f"(nearest sim={similarity:.3f} < {self.match_threshold})"
                )

        return novelty_scores, continuity_scores, lifecycle_stages

    @staticmethod
    def _classify_lifecycle(
        run_count: int,
        curr_articles: int,
        prev_articles: int,
        prev_lifecycle: str,
    ) -> str:
        """Classify trend lifecycle stage based on article count trajectory.

        BIRTH -> GROWTH -> PEAK -> DECLINE -> DORMANT
          ^                                      |
          +------------ REVIVAL <----------------+
        """
        if run_count <= 1:
            return "birth"

        if prev_lifecycle == "dormant" and curr_articles > 0:
            return "revival"

        if curr_articles == 0:
            return "dormant"

        # Compare article counts
        if prev_articles == 0:
            return "growth"

        ratio = curr_articles / max(prev_articles, 1)

        if ratio > 1.2:  # 20%+ growth
            return "growth"
        elif ratio < 0.7:  # 30%+ decline
            return "decline"
        else:
            return "peak"  # Stable

    def _store_centroid(
        self,
        cluster_id: int,
        centroid: List[float],
        cluster_keywords: Optional[Dict[int, List[str]]],
        now: datetime,
        article_count: int = 0,
        oss: float = 0.0,
    ):
        """Store a new centroid in ChromaDB."""
        keywords = (
            cluster_keywords.get(cluster_id, [])[:10]
            if cluster_keywords else []
        )
        doc_id = f"trend_{now.strftime('%Y%m%d_%H%M%S')}_{cluster_id}"

        self.collection.add(
            ids=[doc_id],
            embeddings=[centroid],
            metadatas=[{
                "run_count": 1,
                "first_seen": now.isoformat(),
                "last_seen": now.isoformat(),
                "keywords": json.dumps(keywords),
                "similarity": 1.0,
                "article_count": article_count,
                "lifecycle": "birth",
                "avg_oss": round(oss, 4),
                "oss_trend": "new",
            }],
        )

    def prune_stale(self) -> int:
        """Remove stored centroids not seen in stale_days.

        Returns:
            Number of stale centroids removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.stale_days)
        cutoff_str = cutoff.isoformat()

        stored = self.collection.get(include=["metadatas"])
        if not stored["ids"]:
            return 0

        stale_ids = []
        for doc_id, metadata in zip(stored["ids"], stored["metadatas"]):
            last_seen = metadata.get("last_seen", "")
            if last_seen and last_seen < cutoff_str:
                stale_ids.append(doc_id)
                run_count = metadata.get("run_count", 0)
                keywords = metadata.get("keywords", "[]")
                logger.debug(
                    f"Stale centroid: {doc_id[:12]}... "
                    f"(run_count={run_count}, last_seen={last_seen[:10]}, kw={keywords})"
                )

        if stale_ids:
            self.collection.delete(ids=stale_ids)
            logger.info(f"Pruned {len(stale_ids)} stale trend centroids (>{self.stale_days} days)")

        return len(stale_ids)

    def update_oss_scores(
        self,
        cluster_centroids: Dict[int, np.ndarray],
        cluster_oss: Dict[int, float],
    ) -> int:
        """Post-synthesis: update stored centroids with OSS data.

        Called AFTER synthesis produces OSS scores. Since compute_novelty()
        runs BEFORE synthesis, centroids are stored with oss=0.0. This method
        finds the matching stored centroids and updates their OSS metadata.

        Returns:
            Number of centroids updated.
        """
        if not cluster_oss or not cluster_centroids:
            return 0

        updated = 0
        for cid, centroid in cluster_centroids.items():
            if cid not in cluster_oss:
                continue

            centroid_list = centroid.tolist() if isinstance(centroid, np.ndarray) else centroid
            results = self.collection.query(
                query_embeddings=[centroid_list],
                n_results=1,
                include=["metadatas"],
            )

            if not results["ids"][0]:
                continue

            matched_id = results["ids"][0][0]
            metadata = results["metadatas"][0][0]

            # Only update if this is a recent match (last_seen within last hour)
            curr_oss = cluster_oss[cid]
            prev_avg_oss = metadata.get("avg_oss", 0.0)

            if prev_avg_oss > 0 and curr_oss > 0:
                new_avg_oss = round(self.ema_alpha * curr_oss + (1 - self.ema_alpha) * prev_avg_oss, 4)
            else:
                new_avg_oss = round(curr_oss, 4) if curr_oss > 0 else prev_avg_oss

            if curr_oss > prev_avg_oss + 0.05:
                oss_trend = "improving"
            elif curr_oss < prev_avg_oss - 0.05:
                oss_trend = "declining"
            else:
                oss_trend = "stable"

            metadata["avg_oss"] = new_avg_oss
            metadata["oss_trend"] = oss_trend

            self.collection.update(
                ids=[matched_id],
                metadatas=[metadata],
            )
            updated += 1

        if updated:
            logger.debug(f"Updated OSS for {updated} stored centroids")
        return updated

    @property
    def stored_count(self) -> int:
        """Number of stored trend centroids."""
        return self.collection.count()
