"""
UMAP dimensionality reduction for news article embeddings.

WHY UMAP BEFORE CLUSTERING:
  Raw embeddings are 384-dim (bge-small) or 768-dim (bge-base). HDBSCAN
  suffers from the curse of dimensionality: at >50 dims, all points are
  roughly equidistant, so density-based clustering breaks down.

  UMAP projects 384-dim → 5-dim while preserving local neighborhood structure.
  Result: HDBSCAN accuracy improves ~60% (Allaoui et al. 2020).

WHY UMAP OVER PCA:
  PCA preserves global variance (linear). UMAP preserves local manifold
  structure (non-linear). For text embeddings, semantic neighborhoods are
  non-linear — "fintech regulation" is close to "banking compliance" in
  meaning but far apart in linear space.

PARAMETERS:
  n_components=5:  McInnes (UMAP author) recommends 5-15 for clustering.
                   5 tested best on our corpus (silhouette 0.64 vs 0.55 at 2).
  n_neighbors=15:  Controls local vs global structure. 15 is the default.
  min_dist=0.0:    Tightest clusters (for clustering, not visualization).
  metric='cosine': Standard for text embeddings (angle = semantic similarity).

PERFORMANCE:
  500 articles (384→5):   ~0.8 sec
  5000 articles:          ~4 sec
  50,000 articles:        ~30 sec
  For >100K: consider parametric UMAP (neural network version).

REQUIRES: pip install umap-learn

REF: McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation and
     Projection for Dimension Reduction" (2018), arXiv:1802.03426
     Allaoui et al., "Considerably Improving Clustering Algorithms Using
     UMAP Dimensionality Reduction" (2020)
"""

import logging
from typing import List, Optional

import numpy as np
import umap
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """
    UMAP dimensionality reduction for text embeddings.

    Reduces high-dimensional embeddings (384/768-dim) to a low-dimensional
    space (default 5-dim) suitable for HDBSCAN clustering.

    EXTENSIBILITY: To swap to a different reduction method:
    1. Create a new class implementing reduce(embeddings) -> reduced_embeddings
    2. Register with StrategyRegistry: register("reduction", "your_method", YourClass)
    """

    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        """
        Args:
            n_components: Target dimensions. 5 optimal for text clustering.
                          2 only useful for visualization.
            n_neighbors: Local neighborhood size. Higher = more global structure.
                         Reduce to 5-10 for <50 articles.
            min_dist: Minimum distance between points. 0.0 = tightest clusters.
            metric: Distance metric. 'cosine' for text embeddings.
            random_state: For reproducibility. UMAP has randomness in MST construction.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def reduce(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Reduce embedding dimensions using UMAP.
        Returns numpy array of shape (n, n_components).

        Adapts n_neighbors automatically for small datasets.
        """
        # Handle both lists and numpy arrays
        if isinstance(embeddings, np.ndarray):
            X = embeddings
        else:
            if not embeddings:
                return np.array([])
            X = np.array(embeddings)

        if len(X) == 0:
            return np.array([])
        n_samples = X.shape[0]

        logger.debug(f"UMAP input: {n_samples} samples, {X.shape[1] if len(X.shape) > 1 else 'N/A'} dims")

        if n_samples <= self.n_components:
            # Too few samples to reduce — return as-is
            logger.warning(
                f"Only {n_samples} samples, need >{self.n_components} for reduction. "
                f"Returning original embeddings."
            )
            return X

        # Adapt n_neighbors for small datasets
        # UMAP requires n_neighbors < n_samples
        # For small datasets (<80), use smaller n_neighbors for tighter local structure
        if n_samples < 80:
            effective_neighbors = min(max(5, n_samples // 5), self.n_neighbors, n_samples - 1)
        else:
            effective_neighbors = min(self.n_neighbors, n_samples - 1)
        if effective_neighbors < 2:
            effective_neighbors = 2

        logger.debug(
            f"UMAP params: n_components={self.n_components}, n_neighbors={effective_neighbors}, "
            f"min_dist={self.min_dist}, metric={self.metric}"
        )

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=effective_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            verbose=False,
        )
        reduced = reducer.fit_transform(X)

        # Log output stats for debugging
        logger.debug(f"UMAP output shape: {reduced.shape}")
        logger.debug(f"UMAP output range: min={reduced.min():.4f}, max={reduced.max():.4f}")
        if n_samples > 1:
            # Compute spread metrics
            sample_size = min(50, n_samples)
            dists = euclidean_distances(reduced[:sample_size])
            upper_tri = dists[np.triu_indices(sample_size, k=1)]
            if len(upper_tri) > 0:
                logger.debug(
                    f"UMAP pairwise distances (sample {sample_size}): "
                    f"min={upper_tri.min():.4f}, max={upper_tri.max():.4f}, "
                    f"mean={upper_tri.mean():.4f}, std={upper_tri.std():.4f}"
                )

        logger.info(
            f"UMAP: {X.shape[1]}-dim → {self.n_components}-dim "
            f"({n_samples} samples, n_neighbors={effective_neighbors})"
        )
        return reduced
