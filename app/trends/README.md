# app/trends/ -- Trend Engine

The trend engine transforms raw RSS articles into structured, scored, and validated business trends. It runs as part of the `analysis` step in the LangGraph pipeline.

## Full Pipeline

```
RSS articles (300-500)
    |
    v
[scrape] -> full text extraction (trafilatura)
    |
[event_classify] -> 16 event categories via embedding similarity
    |
[dedup] -> MinHash LSH near-duplicate removal (~30% removed)
    |
[NER] -> spaCy named entity recognition + normalization
    |
[geo_filter] -> entity-based geographic relevance (India focus)
    |
[embed] -> 1024-dim content-aware embeddings (NVIDIA/OpenAI/HF/local)
    |
[semantic_dedup] -> cosine similarity deduplication
    |
[Leiden clustering] -> k-NN graph + community detection (Optuna-tuned)
    |
[coherence] -> validate clusters, split incoherent, merge redundant
    |
[keywords] -> TF-IDF extraction per cluster
    |
[signals] -> 6-module scoring (temporal, content, entity, source, market, composite)
    |
[trend_memory] -> ChromaDB novelty scoring (new vs recurring)
    |
[synthesis] -> LLM generates human-readable trend summaries
    |
[tree] -> TrendTree assembly with quality gate
```

## Modules

| Module | Purpose |
|--------|---------|
| `engine.py` | `TrendPipeline` -- orchestrates all layers above |
| `coherence.py` | Post-clustering validation: split, merge, reject, IQR outlier detection |
| `subclustering.py` | Agglomerative sub-trend detection within large clusters |
| `synthesis.py` | Concurrent LLM synthesis (5W1H summaries, causal chains) |
| `enrichment.py` | Entity consolidation, company activity scoring, cluster validation |
| `tree.py` | `TrendTree` assembly + AI council trend classification |
| `keywords.py` | TF-IDF keyword extraction per cluster |
| `memory.py` | ChromaDB centroid storage for novelty detection across runs |
| `correlation.py` | Cross-trend entity bridges, sector chains, temporal lag |

## Signal Types (`signals/`)

Each signal module computes per-cluster scores normalized to [0, 1]:

| Module | Signals | Description |
|--------|---------|-------------|
| `temporal.py` | recency, velocity | Article freshness + publication acceleration |
| `content.py` | specificity, sentiment | OSS-derived specificity + VADER sentiment |
| `entity.py` | entity_focus, person_count | Entity concentration + decision-maker mentions |
| `source.py` | diversity, authority | Source variety + bandit-adaptive quality weighting |
| `market.py` | regulatory, trigger, financial | Regulatory keywords, 6sense event triggers, financial terms |
| `composite.py` | actionability, trend, quality, confidence | Final weighted combinations with factor breakdowns |

### Composite Scoring

`composite.py` produces four final scores per trend:

- **actionability_score**: How valuable for sales outreach (uses learned weights from feedback)
- **trend_score**: Raw importance (BERTrend + Reddit Hot algorithm inspired)
- **cluster_quality_score**: Cluster formation quality (coherence, diversity, authority)
- **confidence_score**: Probability this is a real, new trend (quality + novelty + evidence)

Weights are loaded from `data/learned_weights.json` when available (output of the weight learner).

## Clustering Approach

The engine uses Leiden community detection on a k-NN graph built from 1024-dimensional embeddings (no dimensionality reduction):

1. **k-NN graph**: Connect each article to its k nearest neighbors by cosine similarity
2. **Event-type augmentation**: Append scaled one-hot event-type dimensions to break the "India business news gravity well" where all articles cluster together
3. **Leiden algorithm**: Find communities optimizing modularity (Traag et al. 2019)
4. **Optuna optimization**: Bayesian TPE tunes `(k, resolution, min_community_size)` over ~15 trials to maximize cluster quality (coherence * diversity * size balance)

### Post-Clustering Refinement

`coherence.py` runs 5 operations on Leiden output:
1. Validate coherence (mean pairwise cosine in original embedding space)
2. Split incoherent clusters (agglomerative)
3. Merge redundant clusters (centroid similarity)
4. Reject very low coherence (demote to noise)
5. Multi-signal outlier detection + vocabulary-based refinement

## Key Tuning Parameters

| Variable | Default | Effect |
|----------|---------|--------|
| `LEIDEN_K` | 20 | More neighbors = larger, fewer clusters |
| `LEIDEN_RESOLUTION` | 1.0 | Higher = more, smaller clusters |
| `LEIDEN_OPTUNA_ENABLED` | true | Bayesian auto-tuning (recommended) |
| `LEIDEN_OPTUNA_TRIALS` | 15 | More trials = better params, slower |
| `LEIDEN_MIN_COMMUNITY_SIZE` | 3 | Below this = noise |
| `DEDUP_THRESHOLD` | 0.25 | Lower = more aggressive dedup |
| `COHERENCE_THRESHOLD` | 0.35 | EMA-adaptive, minimum cluster coherence |
| `MERGE_THRESHOLD` | 0.80 | EMA-adaptive, centroid similarity for merge |
