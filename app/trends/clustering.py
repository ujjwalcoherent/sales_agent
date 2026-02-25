"""
Leiden-based clustering for news article embeddings.

Pipeline: embeddings -> k-NN graph -> Leiden community detection -> clusters

Operates directly on cosine similarity k-NN graph (no dimensionality reduction).
Deterministic with fixed seed. Supports hybrid similarity matrices.

Optimization via Optuna TPE with meta-feature warm-starting from historical runs.
"""

import logging
import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_deps():
    """Import leidenalg + igraph, raising clear error if missing."""
    try:
        import igraph as ig
        import leidenalg
        return ig, leidenalg
    except ImportError as e:
        raise ImportError(
            "Leiden clustering requires: pip install leidenalg python-igraph\n"
            f"Original error: {e}"
        )


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 20,
    mutual: bool = True,
):
    """Build a k-NN graph from embeddings with cosine similarity edge weights.

    When mutual=True, only keeps edges where both nodes are in each other's k-NN.
    """
    ig, _ = _ensure_deps()
    from sklearn.neighbors import NearestNeighbors

    n_samples = embeddings.shape[0]
    effective_k = min(k, n_samples - 1)

    nn = NearestNeighbors(
        n_neighbors=effective_k + 1,  # +1 because self is included
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    edges = []
    weights = []
    seen = set()

    for i in range(n_samples):
        for j_idx in range(1, effective_k + 1):  # skip self (index 0)
            j = int(indices[i, j_idx])
            cos_dist = float(distances[i, j_idx])
            cos_sim = 1.0 - cos_dist

            if cos_sim <= 0:
                continue

            edge_key = (min(i, j), max(i, j))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            if mutual and i not in indices[j, 1:]:
                continue

            edges.append(edge_key)
            weights.append(cos_sim)

    g = ig.Graph(n=n_samples, edges=edges, directed=False)
    g.es["weight"] = weights

    if weights:
        logger.debug(
            f"k-NN graph: {n_samples} nodes, {len(edges)} edges, "
            f"k={effective_k}, mutual={mutual}, "
            f"avg_weight={np.mean(weights):.3f}"
        )
    else:
        logger.debug(f"k-NN graph: {n_samples} nodes, 0 edges (all isolated)")
    return g


def build_knn_graph_from_similarity(
    sim_matrix: np.ndarray,
    k: int = 20,
    mutual: bool = True,
):
    """Build a k-NN graph from a precomputed similarity matrix (e.g., hybrid blend)."""
    ig, _ = _ensure_deps()
    n = sim_matrix.shape[0]
    effective_k = min(k, n - 1)

    edges = []
    weights = []
    seen = set()

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # exclude self
        top_k = np.argsort(sims)[-effective_k:][::-1]

        for j in top_k:
            j = int(j)
            sim = float(sim_matrix[i, j])
            if sim <= 0:
                continue
            edge_key = (min(i, j), max(i, j))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            if mutual:
                j_sims = sim_matrix[j].copy()
                j_sims[j] = -1.0
                j_top_k = set(np.argsort(j_sims)[-effective_k:])
                if i not in j_top_k:
                    continue

            edges.append(edge_key)
            weights.append(sim)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    if weights:
        logger.debug(
            f"Hybrid k-NN graph: {n} nodes, {len(edges)} edges, "
            f"k={effective_k}, mutual={mutual}, "
            f"avg_weight={np.mean(weights):.3f}"
        )
    return g


def cluster_leiden(
    embeddings: np.ndarray,
    k: int = 20,
    resolution: float = 1.0,
    seed: int = 42,
    mutual_knn: bool = True,
    min_community_size: int = 4,
    quality_function: str = "RBConfiguration",
    precomputed_sim: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Cluster embeddings using k-NN graph + Leiden community detection.

    Returns (labels, noise_count, metrics) where labels has -1 for noise.
    """
    _, leidenalg = _ensure_deps()

    n_samples = embeddings.shape[0]
    t_start = time.time()

    if n_samples < 3:
        return np.full(n_samples, -1, dtype=int), n_samples, {
            "n_clusters": 0, "noise_count": n_samples,
        }

    t_graph = time.time()
    if precomputed_sim is not None:
        graph = build_knn_graph_from_similarity(precomputed_sim, k=k, mutual=mutual_knn)
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normed = embeddings / norms
        graph = build_knn_graph(emb_normed, k=k, mutual=mutual_knn)
    graph_time = time.time() - t_graph

    t_leiden = time.time()

    partition_types = {
        "RBConfiguration": leidenalg.RBConfigurationVertexPartition,
        "Modularity": leidenalg.ModularityVertexPartition,
        "CPM": leidenalg.CPMVertexPartition,
        "Significance": leidenalg.SignificanceVertexPartition,
    }
    partition_type = partition_types.get(
        quality_function, leidenalg.RBConfigurationVertexPartition
    )

    leiden_kwargs = {"weights": "weight", "seed": seed, "n_iterations": -1}
    if quality_function in ("RBConfiguration", "CPM"):
        leiden_kwargs["resolution_parameter"] = resolution

    partition = leidenalg.find_partition(graph, partition_type, **leiden_kwargs)
    leiden_time = time.time() - t_leiden

    raw_labels = np.array(partition.membership)
    labels = raw_labels.copy()

    community_sizes = Counter(raw_labels)
    noise_communities = {
        c for c, size in community_sizes.items() if size < min_community_size
    }
    for c in noise_communities:
        labels[raw_labels == c] = -1

    unique_labels = sorted(set(labels) - {-1})
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = -1
    labels = np.array([label_map[l] for l in labels])

    noise_count = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    total_time = time.time() - t_start

    cluster_sizes = sorted(
        [s for c, s in community_sizes.items() if c not in noise_communities],
        reverse=True,
    )
    metrics = {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_pct": noise_count / max(n_samples, 1),
        "modularity": float(partition.modularity),
        "quality": float(partition.quality()),
        "graph_edges": graph.ecount(),
        "graph_time_s": round(graph_time, 3),
        "leiden_time_s": round(leiden_time, 3),
        "total_time_s": round(total_time, 3),
        "resolution": resolution,
        "k": k,
        "cluster_sizes": cluster_sizes,
    }

    logger.info(
        f"Leiden: {n_samples} articles → {n_clusters} clusters, "
        f"{noise_count} noise ({noise_count * 100 // max(n_samples, 1)}%), "
        f"modularity={partition.modularity:.3f}, "
        f"sizes={cluster_sizes[:5]}{'...' if len(cluster_sizes) > 5 else ''}, "
        f"graph={graph_time:.2f}s, leiden={leiden_time:.3f}s"
    )

    return labels, noise_count, metrics


def optimize_leiden(
    embeddings: np.ndarray,
    n_trials: int = 15,
    seed: int = 42,
    warm_start_params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    precomputed_sim: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Bayesian optimization of Leiden parameters using Optuna TPE.

    Returns dict with {k, resolution, min_community_size, score}.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(embeddings)
    target_clusters = max(5, int(math.sqrt(n)))

    # Adaptive min_community_size range based on dataset size
    if n >= 500:
        min_comm_lo, min_comm_hi = 3, 6
    elif n >= 200:
        min_comm_lo, min_comm_hi = 2, 5
    else:
        min_comm_lo, min_comm_hi = 2, 4

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = embeddings / norms

    def objective(trial):
        k = trial.suggest_int("k", 8, min(30, n - 1))
        resolution = trial.suggest_float("resolution", 0.3, 3.0, log=True)
        min_community = trial.suggest_int("min_community_size", min_comm_lo, min_comm_hi)
        use_mutual = trial.suggest_categorical("mutual_knn", [True, False])

        labels, _, metrics = cluster_leiden(
            emb_normed, k=k, resolution=resolution, seed=seed,
            min_community_size=min_community,
            mutual_knn=use_mutual,
            precomputed_sim=precomputed_sim,
        )

        n_clusters = metrics["n_clusters"]
        noise_pct = metrics["noise_pct"]
        modularity = metrics.get("modularity", 0)

        quality = compute_leiden_quality(emb_normed, labels)
        coherence = quality.get("avg_coherence", 0)
        min_coherence = quality.get("min_coherence", 0)

        if n_clusters == 0:
            return 0.0

        cluster_diff = n_clusters - target_clusters
        if cluster_diff < 0:
            cluster_penalty = abs(cluster_diff) / max(target_clusters, 1)
        else:
            cluster_penalty = 0.7 * cluster_diff / max(target_clusters, 1)
        cluster_penalty = min(cluster_penalty, 1.0)

        # Adaptive noise penalty
        if n >= 500:
            noise_floor, noise_ceiling = 0.05, 0.45   # 5-50% range
        elif n >= 200:
            noise_floor, noise_ceiling = 0.05, 0.45   # same
        else:
            noise_floor, noise_ceiling = 0.10, 0.55   # more lenient for small datasets
        noise_penalty = max(0, (noise_pct - noise_floor) / max(noise_ceiling - noise_floor, 0.01))
        noise_penalty = min(noise_penalty, 1.0)

        score = (
            0.25 * coherence                              # cluster tightness
            + 0.15 * min(1.0, min_coherence / 0.35)      # no garbage clusters (raised)
            + 0.10 * max(0, modularity)                   # graph structure quality
            + 0.15 * (1 - cluster_penalty)                # right number of clusters
            + 0.20 * (1 - noise_penalty)                  # noise control
            + 0.10 * quality.get("silhouette_cosine", 0)  # cluster separation
            + 0.05 * min(1.0, n_clusters / max(target_clusters, 1))  # reward having enough clusters
        )

        return score

    logger.debug(
        f"Adaptive clustering: n={n}, min_community_range=[{min_comm_lo}, {min_comm_hi}], "
        f"target_clusters={target_clusters}"
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if warm_start_params:
        study.enqueue_trial({
            "k": warm_start_params.get("k", 20),
            "resolution": warm_start_params.get("resolution", 1.0),
            "min_community_size": warm_start_params.get("min_community_size", 4),
            "mutual_knn": warm_start_params.get("mutual_knn", True),
        })

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params
    best["score"] = round(study.best_value, 4)
    best["n_trials_completed"] = len(study.trials)

    logger.info(
        f"Optuna Leiden optimization: k={best['k']}, "
        f"resolution={best['resolution']:.3f}, "
        f"min_community={best['min_community_size']}, "
        f"score={best['score']:.4f} "
        f"({best['n_trials_completed']} trials)"
    )

    return best


def compute_meta_features(embeddings: np.ndarray) -> Dict[str, float]:
    """Extract dataset meta-features for similarity-based warm-starting.

    Returns dict characterizing the embedding space shape for matching
    against historical runs.
    """
    n = len(embeddings)
    features: Dict[str, float] = {"n_articles": float(n)}

    if n < 3:
        return features

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = embeddings / norms

    sample_size = min(200, n)
    if n > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, sample_size, replace=False)
        sample = emb_norm[idx]
    else:
        sample = emb_norm

    sims = np.dot(sample, sample.T)
    upper_tri = sims[np.triu_indices(len(sample), k=1)]

    if len(upper_tri) > 0:
        features["sim_mean"] = round(float(np.mean(upper_tri)), 4)
        features["sim_std"] = round(float(np.std(upper_tri)), 4)
        features["sim_p10"] = round(float(np.percentile(upper_tri, 10)), 4)
        features["sim_p90"] = round(float(np.percentile(upper_tri, 90)), 4)

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, n - 1, emb_norm.shape[1]))
        pca.fit(sample)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.searchsorted(cumvar, 0.95) + 1)
        features["intrinsic_dim"] = float(intrinsic_dim)
    except Exception:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors
        k_nn = min(10, len(sample) - 1)
        nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine", algorithm="brute")
        nn.fit(sample)
        dists, _ = nn.kneighbors(sample)
        avg_knn_dist = float(dists[:, 1:].mean())
        median_dist = float(np.median(upper_tri)) if len(upper_tri) > 0 else 1.0
        features["density_ratio"] = round(
            avg_knn_dist / max(median_dist, 0.001), 4
        )
    except Exception:
        pass

    return features


def load_best_params_for_data(
    meta_features: Dict[str, float],
    log_path: Optional[str] = None,
    max_history: int = 20,
    min_history: int = 5,
) -> Optional[Dict[str, Any]]:
    """Load best Optuna params from the most similar historical run.

    Returns {k, resolution, min_community_size} or None.
    """
    try:
        from app.learning.pipeline_metrics import load_history
        history = load_history(last_n=max_history)
    except Exception:
        return None

    valid_runs = [
        r for r in history
        if r.get("optuna_best_params") and r.get("meta_features")
    ]

    if len(valid_runs) < min_history:
        return load_last_best_params(log_path)

    feature_keys = ["n_articles", "sim_mean", "sim_std", "sim_p10", "sim_p90",
                    "intrinsic_dim", "density_ratio"]

    current_vec = np.array([meta_features.get(k, 0.0) for k in feature_keys])

    all_vecs = []
    for r in valid_runs:
        mf = r["meta_features"]
        all_vecs.append([mf.get(k, 0.0) for k in feature_keys])
    all_vecs = np.array(all_vecs)

    ranges = all_vecs.max(axis=0) - all_vecs.min(axis=0)
    ranges[ranges < 0.001] = 1.0

    current_norm = (current_vec - all_vecs.min(axis=0)) / ranges
    history_norm = (all_vecs - all_vecs.min(axis=0)) / ranges

    distances = np.linalg.norm(history_norm - current_norm, axis=1)
    top_indices = np.argsort(distances)[:3]

    top_dists = distances[top_indices]
    weights = 1.0 / (top_dists + 0.01)
    weights /= weights.sum()

    param_keys = ["k", "resolution", "min_community_size"]
    blended = {}
    for pk in param_keys:
        vals = []
        for idx in top_indices:
            params = valid_runs[idx]["optuna_best_params"]
            vals.append(params.get(pk, 0))
        blended[pk] = sum(v * w for v, w in zip(vals, weights))

    blended["k"] = int(round(blended["k"]))
    blended["min_community_size"] = int(round(blended["min_community_size"]))
    blended["resolution"] = round(blended["resolution"], 4)

    best_match = valid_runs[top_indices[0]]
    logger.info(
        f"Meta-feature warm-start: matched run {best_match.get('run_id', '?')} "
        f"(distance={top_dists[0]:.3f}), blended params: {blended}"
    )

    return blended


def load_last_best_params(
    log_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load the last Optuna best params from pipeline run log for warm-starting.

    Returns:
        Dict with {k, resolution, min_community_size} or None if no history.
    """
    try:
        from app.learning.pipeline_metrics import load_history
        history = load_history(last_n=5)
        for record in reversed(history):
            optuna_params = record.get("optuna_best_params")
            if optuna_params and isinstance(optuna_params, dict):
                return optuna_params
    except Exception:
        pass
    return None


def auto_resolve_resolution(
    embeddings: np.ndarray,
    k: int = 20,
    target_clusters: Optional[Tuple[int, int]] = None,
    seed: int = 42,
    precomputed_sim: Optional[np.ndarray] = None,
) -> float:
    """Binary search fallback for resolution (lightweight, no Optuna)."""
    n = len(embeddings)
    if target_clusters is None:
        target_min = max(3, int(math.sqrt(n) / 3))
        target_max = max(target_min + 2, int(math.sqrt(n)))
        target_clusters = (target_min, target_max)

    lo, hi = 0.1, 5.0
    best_res = 1.0
    best_diff = float("inf")

    for _ in range(10):
        mid = (lo + hi) / 2.0
        labels, _, metrics = cluster_leiden(
            embeddings, k=k, resolution=mid, seed=seed,
            precomputed_sim=precomputed_sim,
        )
        n_clusters = metrics["n_clusters"]

        if target_clusters[0] <= n_clusters <= target_clusters[1]:
            return mid

        diff = abs(n_clusters - (target_clusters[0] + target_clusters[1]) / 2)
        if diff < best_diff:
            best_diff = diff
            best_res = mid

        if n_clusters < target_clusters[0]:
            lo = mid
        else:
            hi = mid

    logger.info(
        f"Auto-resolution: best={best_res:.2f} "
        f"(target={target_clusters}, achieved ~{n_clusters} clusters)"
    )
    return best_res


def compute_leiden_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute quality metrics: silhouette, per-cluster coherence, and aggregates."""
    from sklearn.metrics import silhouette_score
    from app.tools.embeddings import mean_pairwise_cosine

    mask = labels >= 0
    valid_embeddings = embeddings[mask]
    valid_labels = labels[mask]

    metrics: Dict[str, Any] = {}

    if len(set(valid_labels)) > 1 and len(valid_labels) > 2:
        try:
            metrics["silhouette_cosine"] = float(
                silhouette_score(valid_embeddings, valid_labels, metric="cosine")
            )
        except Exception:
            pass

    cluster_coherences = {}
    for cid in sorted(set(valid_labels)):
        cluster_embs = valid_embeddings[valid_labels == cid]
        cluster_coherences[int(cid)] = mean_pairwise_cosine(cluster_embs)

    metrics["cluster_coherences"] = cluster_coherences
    if cluster_coherences:
        coherence_vals = list(cluster_coherences.values())
        metrics["avg_coherence"] = float(np.mean(coherence_vals))
        metrics["min_coherence"] = float(np.min(coherence_vals))
        metrics["coherence_p25"] = float(np.percentile(coherence_vals, 25))

    return metrics
