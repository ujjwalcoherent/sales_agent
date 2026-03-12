# app/intelligence/ — News Clustering Pipeline

Multi-stage deterministic pipeline: raw RSS articles → validated, synthesized trend clusters.
Entry: `await execute(scope: DiscoveryScope) → IntelligenceResult`

**Principle:** math handles stages 1–7, LLM handles only synthesis (stage 8). First LLM call happens at stage 8.

```
intelligence/
├── pipeline.py             # Orchestrates all stages, publishes learning signals
├── fetch.py                # Stage 1+2: RSS fetch + TF-IDF title/body dedup
├── filter.py               # Stage 3: NLI filter + LLM fallback for ambiguous zone
├── match.py                # Stage 9: product-to-cluster opportunity scoring
├── summarizer.py           # Stage 8: LLM synthesis with Reflexion retry
├── models.py               # Domain models: Article, Cluster, IntelligenceResult
├── config.py               # ClusteringParams defaults + REGION_SOURCES allowlists + INDUSTRY_TAXONOMY
├── engine/                 # Math core (local, no LLM)
│   ├── nli_filter.py       # cross-encoder/nli-deberta-v3-small inference
│   ├── extractor.py        # GLiNER (B2B labels) + spaCy NER
│   ├── similarity.py       # 6-signal decomposed N×N similarity matrix
│   ├── classifier.py       # Article event type classification
│   ├── clusterer.py        # HAC + HDBSCAN + Leiden adaptive routing
│   ├── normalizer.py       # Entity name normalization (rapidfuzz WRatio ≥ 85)
│   └── validator.py        # 7-check cluster validation gate
└── cluster/                # Cluster algorithm implementations
    ├── algorithms.py       # HAC (scipy average linkage), HDBSCAN soft-membership, Leiden (igraph)
    ├── orchestrator.py     # Routes by article count and group type → algorithm selection
    └── validator.py        # Post-clustering coherence + signal checks
```

---

## Stage 1: Fetch

Sources: region-filtered RSS feeds + Tavily news API, fetched in parallel.

**Region allowlists** (`config.py:REGION_SOURCES`):

| Region | Key publications |
|--------|-----------------|
| `IN` | ET/LiveMint/MoneyControl/Inc42/YourStory/CNBCTV18/SEBI/RBI + ~50 sources |
| `US` | WSJ/TechCrunch/Bloomberg/Crunchbase/VentureBeat + ~30 sources |
| `EU` | BBC Business/FT/Guardian/Euractiv + ~15 sources |
| `SEA` | CNA/Straits Times/TechinAsia/KrASIA + ~8 sources |
| `GLOBAL` | `None` → uses `DEFAULT_ACTIVE_SOURCES` (103 active sources) |

Explicit allowlists — fetching for a region only pulls sources in that region's list. India runs never fetch Forbes, Inc Magazine, or Al Jazeera.

Language filter: `langdetect` library detects language on first 500 chars; articles in non-target language dropped before RSS parse completes.

Source fetch ordering: source bandit posterior sample → higher-quality sources (higher Beta mean) fetched first.

---

## Stage 2: Dedup

Three-pass duplicate removal. All passes keep the **earliest-published** copy.

**Pass 0 — URL normalization** (O(n)):
```python
norm_url = article.url.split("?")[0].rstrip("/")
```
Catches identical articles published by two RSS sub-feeds.

**Pass 1 — Title dedup** (TF-IDF cosine, threshold = `dedup_title_threshold` default 0.85):
```
TF-IDF(title, ngram_range=(1,2), max_features=10000)
cosine_similarity(matrix) >= 0.85 → keep earliest, drop rest
```
Catches syndicated reposts.

**Pass 2 — Body dedup** (TF-IDF cosine, threshold = `dedup_body_threshold` default 0.70):
```
text = full_text or summary or title
TF-IDF(text, ngram_range=(1,2), max_features=10000)
cosine_similarity(matrix) >= 0.70 → keep earliest, drop rest
```
Lower threshold (0.70) catches rewritten versions that share key phrases.

Real data: ~10% articles removed per run.

---

## Stage 3: NLI Filter

**Model:** `cross-encoder/nli-deberta-v3-small` (~60MB, CPU-only, ~50ms/article)

**Architecture:** DeBERTa cross-encoder — joint encoding of `(article_text, hypothesis)`, output via `[CLS]` → 3 logits → softmax.

**Label order** (verified from model `config.json`): `{0: contradiction, 1: entailment, 2: neutral}`

**Entailment score:**
```python
scores = softmax(model_logits, dim=1)   # shape (batch, 3)
entailment = scores[:, 1]               # _ENTAILMENT_IDX = 1
```

**Decision thresholds** (from `ClusteringParams`):
```
entailment >= nli_auto_accept (default 0.88)  →  AUTO-ACCEPT (no LLM)
entailment <= nli_auto_reject (default 0.10)  →  AUTO-REJECT (no LLM)
0.10 < score < 0.88                           →  LLM batch classify (GPT-4.1-nano, batch=10)
```

~80% of decisions are pure math. LLM only handles the ambiguous zone.

**Score cache:** LRU cache (capacity 2048) keyed on `hash(text_prefix + hypothesis)`. Thread-safe via `_cache_lock`.

**Hypothesis** loaded from `data/filter_hypothesis.json` at first inference call. Updated by `HypothesisLearner` when SetFit retraining runs.

**Current hypothesis:**
```
"This article reports on a specific company named in the text that is growing,
raising capital, launching a product, releasing a model or technology, making
an acquisition, facing a regulatory action, issuing a product recall, signing
a major contract, filing for an IPO, entering a partnership, or making a
strategic business move."
```

**Hypothesis grammar rules** — violation causes near-zero entailment on ALL inputs:
1. Prefix `"This article reports on a specific company named in the text that is..."` — NEVER change
2. NEVER add negation — NLI models score near-zero entailment for negated hypotheses
3. NEVER use meta-descriptions ("discusses", "business news report") — NLI trained on factual assertions
4. Only extend the action verb list at the end

**Benchmark results** (real India 120h data, 2026-03-08):
- B2B articles: mean entailment 0.859 (Zepto=0.989, Infosys Azure=0.988, TCS Q3=0.981)
- Noise articles: mean entailment 0.333 (PM-US deal=0.116, India-US oil=0.008, ship insurance=0.001)

**Filter pass rate in practice:** 13-35% of articles pass (correct behavior — India RSS includes SEBI filings, political news, economic commentary; NLI only passes articles with a company explicitly doing a business action).

**Confusion matrix logging:** TP/FP/TN/FN counts are logged per-run in `filter.py` console output, not stored on the signal bus.

**Gap 4 rule:** if a target company (in COMPANY_FIRST mode) has 0 articles in the last N days → company dropped from scope.

**Research:** Yin et al. (2019) arXiv:1909.00161; arXiv:2312.17543 (+9.4% F1 over keyword classifiers).

---

## Stage 4: Entity NER (Two-Pass)

**Pass 1 — GLiNER** (`engine/extractor.py`):
- Model: `urchade/gliner_small-v2.1` (local, `data/models/gliner_small/`)
- B2B labels: `company`, `startup`, `financial_institution`, `venture_capital_firm`, `enterprise_software`, `executive`
- Span detection on article title + first 500 chars of summary
- Research: Zaratiana et al. (2024) NAACL 2024 naacl-long.300 (+8.2% F1 vs ChatGPT zero-shot NER)

**Pass 2 — spaCy** (`en_core_web_sm`):
- `ORG` / `PERSON` / `GPE` entity types
- Merged with GLiNER, conflicts resolved by confidence score

**Entity normalization** (`engine/normalizer.py`): rapidfuzz `WRatio >= 85` → merge to canonical name (highest-frequency mention).

**Entity quality cache** (`data/entity_quality.json`): per-entity score = EMA of cluster coherence when that entity appeared as primary. After score >= 3.0 → `MIN_SEED_SALIENCE` lowered from 0.30 to 0.22.

---

## Stage 5: 6-Signal Similarity Matrix

`engine/similarity.py:compute_decomposed_similarity()` → `Dict[str, np.ndarray]` (N×N each)

### Signal 1: Semantic (weight=0.35)
Cosine similarity on 1024-dim neural embeddings:
```
S_semantic[i,j] = cosine(e_i, e_j)    (L2-normalized, clamped [0,1])
```

### Signal 2: Entity Overlap (weight=0.25)
Jaccard similarity on B2B entity name sets:
```
S_entity[i,j] = |E_i ∩ E_j| / |E_i ∪ E_j|
```
Vectorized via scipy sparse matrix.

### Signal 3: Lexical (weight=0.05)
Title word overlap (BM25 proxy):
```
S_lexical[i,j] = |W_i ∩ W_j| / |W_i ∪ W_j|
W_i = word set from title (doubled) + first 100 chars of summary
```
Titles double-counted — title words are higher-precision signal than summary.

### Signal 4: Event Match (weight=0.10)
5W alignment score:
```
S_event[i,j] = 0.40 × I(entity_i == entity_j)        # who match
             + 0.30 × I(event_type_i == event_type_j)  # what match
             + 0.30 × word_overlap(what_i, what_j)     # what content
```

### Signal 5: Temporal Decay (weight=0.10)
Dual-sigma Gaussian:
```
Δt[i,j] = |published_i − published_j|  (seconds)
S_short[i,j] = exp(−Δt² / 2σ_short²)   σ_short = 8h
S_long[i,j]  = exp(−Δt² / 2σ_long²)    σ_long  = 72h
S_temporal[i,j] = max(S_short, S_long)
```
Dual-sigma: breaking news clusters via σ=8h; week-long stories cluster via σ=72h.

### Signal 6: Source Penalty (weight=0.15)
```
S_source[i,j] = 0.3  if source_i == source_j   (same_source_penalty)
              = 1.0  otherwise
```
Same-source pairs have 70% lower effective similarity to enforce source diversity in clusters.

### Blending
```
S_blended = Σ_k (w_k / Σ w_k) × S_k
Default: semantic=0.35, entity=0.25, source=0.15, event=0.10, temporal=0.10, lexical=0.05
```

---

## Stage 6: Clustering Cascade

`cluster/orchestrator.py` routes by group type and article count:

```
Entity groups (articles about the same named entity):
  n < 5        →  single cluster (too small to subdivide)
  5 <= n <= 50 →  HAC (scipy average linkage, silhouette sweep)
  n > 50       →  HDBSCAN (soft-membership)

Discovery mode (ungrouped articles after entity extraction):
  Any n        →  Leiden community detection on k-NN similarity graph
```

Leiden handles all articles that did NOT join an entity group. HAC/HDBSCAN handle entity-grouped articles only.

### HAC (Hierarchical Agglomerative Clustering)

`cluster/algorithms.py:cluster_hac()`

Linkage method: **average** (cosine-safe; Ward requires Euclidean space).

**Dendrogram silhouette sweep:**
```
for t in arange(t_min, t_max, step=0.05):
    labels = fcluster(Z, t, criterion="distance")
    sil = silhouette_score(dist_matrix, labels, metric="precomputed")
    adjusted_sil = sil − (n_singletons / n) × singleton_penalty  (singleton_penalty = 0.5)
    → pick t* = argmax(adjusted_sil)
```

Adaptive threshold range by group size:
- n < 15: `t_min = max(0.15, hac_threshold_min - 0.05)`, `t_max = 0.80`
- n >= 30: `t_min = max(0.25, hac_threshold_min + 0.05)`, `t_max = 0.60`
- default: `t_min = 0.30`, `t_max = 0.65`

Outlier removal: per-sample silhouette `< hac_outlier_silhouette (-0.1)` → moved to noise.

Cophenetic correlation `ρ = cophenet(Z, condensed_dist)` logged as cluster quality signal.

**Research:** FANATIC (EMNLP 2021) singleton penalty for unbalanced clusters.

### HDBSCAN Soft-Membership

`cluster/algorithms.py:cluster_hdbscan_soft()` uses `hdbscan.HDBSCAN(prediction_data=True)`.

**Soft-membership assignment** (Campello et al. 2013):
```
For each noise point p:
    soft_scores = approximate_predict(p)      # shape (n_clusters,)
    if max(soft_scores) > SOFT_NOISE_THRESHOLD (0.10):
        assign p to argmax(soft_scores)
    else:
        keep as noise
```
Recovers ~15-25% of articles HDBSCAN would discard as noise.

Parameters (adaptive):
- n < 30: `min_cluster_size=3, min_samples=2`
- n >= 30: `min_cluster_size=5, min_samples=3`

**Research:** Campello et al. (2013) JMLR soft-membership HDBSCAN.

### Leiden Community Detection

`cluster/algorithms.py:cluster_leiden()` via `leidenalg` + `python-igraph`.

1. Build k-NN similarity graph: k=20 nearest neighbors per node
2. Edge weight = blended similarity score
3. Run Leiden with `resolution` parameter
4. Communities with `size < leiden_min_community_size` → noise

**Optuna tuning** (15 trials, 30s timeout per run):
- Search space: `k ∈ [8, 30]`, `resolution ∈ [0.5, 3.0]`
- Objective: maximize `n_communities × mean_coherence`

Real result: k=8, resolution=1.757 → 6 communities, modularity=0.614.

**Research:** Traag et al. (2019) Sci Reports 9:5233 — better modularity than Louvain.

---

## Stage 7: Cluster Validation (7 Checks)

`engine/validator.py` — cluster PASSES only if ALL checks pass.

| Check | Metric | Threshold | Notes |
|-------|--------|-----------|-------|
| 1. Size | article count | >= 2 | Singletons always fail |
| 2. Coherence | mean pairwise cosine | >= adaptive (from threshold_adapter) | Entity-seeded clusters bypass this check |
| 3. Entity density | articles with >=1 B2B entity / total | >= 0.40 | Filters abstract clusters |
| 4. Source diversity | unique sources / cluster size | >= 0.30 | Prevents single-source echo clusters |
| 5. Temporal span | max date - min date | <= 30 days | Prevents stale cross-time clusters |
| 6. Signal strength | composite weighted NLI score | >= 0.25 | Based on NLI scores of member articles |
| 7. Duplicate gate | near-identical pairs / total pairs | <= 0.50 | Catches dedup false negatives |

**Entity-seeded clusters** bypass the coherence hard-veto — entity co-occurrence is the relatedness signal.

**Adaptive coherence threshold:** `threshold_adapter.py` adjusts `val_coherence_min` via EMA after each run.

---

## Stage 8: LLM Synthesis

`summarizer.py` — first LLM call in the entire pipeline.

Input per cluster: up to 8 representative articles (highest similarity to cluster centroid), entity list, event type distribution.

Output fields in `TrendData`:
- `trend_title` (internal) / `title` (API)
- `summary`: 2-3 sentence synthesis
- `evidence_chain`: key quotes + citations
- `industries_affected` (internal) / `industries` (API)
- `synthesis_confidence`: self-assessed 0.0-1.0

**Reflexion retry** (Shinn et al. 2023, arXiv:2303.11366):
```
if synthesis_confidence < 0.60:
    critique = LLM("What's weak about this synthesis?")
    retry_synthesis = LLM(original_prompt + critique)   # max 2 retries
```

LLM model tier: `get_lite_model()` — OpenAI GPT-4.1-nano → GeminiDirectLite → Groq → standard chain.

---

## Stage 9: Product Matching

`match.py` scores each cluster against `scope.user_products`:

```
fit_score = 0.60 × cosine(cluster_centroid_embedding, product_embedding)
          + 0.40 × keyword_overlap(cluster_entities, product_keywords)
```

Output: `List[MatchResult]` sorted by `fit_score` descending.

---

## Stage 10: Learning Update

After run completes, `pipeline.py` publishes to Signal Bus:
```python
bus.publish_nli_filter(mean_nli, rejection_rate, hypothesis_version, scores_by_source)
bus.publish_backward_signals(cluster_coherence_by_source, noise_rate, lead_quality_by_cluster)
source_bandit.update_from_run(source_nli_means, cluster_contributions, ...)
entity_quality_cache.update(entity_name → cluster_coherence)
threshold_adapter.update(ThresholdUpdate(...))
dataset_enhancer.extract_labels_from_clusters(clusters, nli_scores)
```

---

## IntelligenceResult Schema

```python
@dataclass
class IntelligenceResult:
    run_id: str
    completed_at: datetime
    clusters: List[TrendData]           # validated + synthesized clusters
    total_articles_fetched: int
    total_articles_post_filter: int
    total_clusters: int
    noise_rate: float                   # fraction of articles assigned to no cluster
    mean_coherence: float               # mean pairwise cosine across all clusters
    mean_fit_score: float               # mean product match score (0.0 if no products)
    match_results: List[MatchResult]    # product ↔ cluster top matches
```

**TrendData field names:** internal pipeline uses `trend_title` / `industries_affected`; API serialization uses `title` / `industries`.

---

## DiscoveryScope — Pipeline Input

```python
@dataclass
class DiscoveryScope:
    mode: DiscoveryMode          # INDUSTRY_FIRST | COMPANY_FIRST | REPORT_DRIVEN
    companies: List[str]         # for COMPANY_FIRST
    industry: Optional[str]      # e.g. "Healthcare > Pharma"
    report_text: Optional[str]   # for REPORT_DRIVEN
    region: str                  # "IN" | "US" | "EU" | "SEA" | "GLOBAL"
    hours: int                   # look-back window (default: 120)
    user_products: List[str]     # for Stage 9 product matching
```

---

## Real Performance (from `data/recordings/`)

| Metric | 120h India run (March 10, 2026) |
|--------|--------------------------------|
| Articles fetched | 367 |
| After NLI filter | 130 (35%) |
| Clusters | 9 |
| Duration | 37 min |
| Companies found | 55 |
| Leads generated | 57 |

---

## Algorithm References

| Algorithm | Paper | Key result |
|-----------|-------|-----------|
| NLI zero-shot | Yin et al. (2019) arXiv:1909.00161 | +9.4% F1 over keyword classifiers |
| GLiNER NER | Zaratiana et al. (2024) NAACL naacl-long.300 | +8.2% F1 vs ChatGPT zero-shot |
| HDBSCAN soft | Campello et al. (2013) JMLR | Noise recovery via soft membership |
| Leiden community | Traag et al. (2019) Sci Reports 9:5233 | Better modularity than Louvain |
| HAC singleton penalty | FANATIC (EMNLP 2021) | Adjusted silhouette for unbalanced clusters |
| Reflexion retry | Shinn et al. (2023) arXiv:2303.11366 | LLM self-critique improves synthesis |
| SetFit hypothesis | Tunstall et al. (2022) arXiv:2209.11055 | 8 examples ≈ 3000-example fine-tune |
