# app/intelligence/ — News Clustering Pipeline

10-stage deterministic pipeline: raw RSS articles → validated, synthesized trend clusters.
Entry: `await execute(scope: DiscoveryScope) → IntelligenceResult`

**Principle**: math handles stages 1-8, LLM handles only synthesis (stage 9). First LLM call happens at stage 9.

```
intelligence/
├── pipeline.py            # Orchestrates all 10 stages, publishes learning signals
├── fetch.py               # Stage 1+2: RSS fetch + SHA-256/MinHash dedup
├── filter.py              # Stage 3: NLI filter + LLM fallback for 0.10–0.88 zone
├── match.py               # Stage 9: Product-to-cluster opportunity scoring
├── summarizer.py          # Stage 8: LLM synthesis with Reflexion retry
├── industry_classifier.py # 2-pass NLI: 1st-order + 2nd-order industry classification
├── models.py              # Domain models: Article, Cluster, IntelligenceResult
├── config.py              # ClusteringParams + REGION_SOURCES allowlists
├── engine/                # Math core (local, no LLM)
│   ├── nli_filter.py      # cross-encoder/nli-deberta-v3-small inference
│   ├── extractor.py       # GLiNER (B2B labels) + spaCy NER
│   ├── similarity.py      # 6-signal decomposed N×N similarity matrix
│   ├── classifier.py      # Article event type classification (5W extraction)
│   ├── clusterer.py       # HAC + HDBSCAN + Leiden adaptive routing
│   ├── normalizer.py      # Entity name normalization + deduplication
│   └── validator.py       # 7-check cluster validation gate
└── cluster/               # Cluster algorithm implementations
    ├── algorithms.py      # HAC (scipy), HDBSCAN soft-membership, Leiden (igraph)
    ├── orchestrator.py    # Routes by article count → algorithm selection
    └── validator.py       # Post-clustering coherence + signal checks
```

---

## Stage 1: Fetch

```python
articles = await fetch_articles(scope: DiscoveryScope)
```

Sources: region-filtered RSS feeds + Tavily news API, fetched in parallel. Merged, sorted by publication time.

**Region allowlists** (`config.py:REGION_SOURCES`):

| Region | Count | Key publications |
|--------|-------|-----------------|
| `IN` | 49 | ET/LiveMint/MoneyControl/Inc42/YourStory/CNBCTV18/SEBI/RBI |
| `US` | 35 | WSJ/TechCrunch/Bloomberg/Crunchbase/VentureBeat |
| `EU` | 14 | BBC Business/FT/Sifted/EU-Startups/Reuters |
| `SEA` | 9 | CNA/Straits Times/TechinAsia/KrASIA |
| `GLOBAL` | 107 | All DEFAULT_ACTIVE_SOURCES |

Explicit per-region allowlists — India runs never fetch Forbes, Inc Magazine, or Al Jazeera (these sources polluted earlier runs with geopolitical noise).

Language filter: `langdetect` drops articles in non-target language before RSS parse is complete.

Source fetch ordering: source bandit posterior sample → higher-quality sources (higher Beta mean) fetched first, up to `RSS_MAX_PER_SOURCE` (default 25).

---

## Stage 2: Dedup

Three-pass duplicate removal. All passes keep the **earliest-published** copy.

**Pass 0 — URL normalization** (O(n)):
```python
norm_url = article.url.split("?")[0].rstrip("/")  # strip query params + trailing slash
```
Catches identical articles published by two RSS sub-feeds (e.g. "Economic Times" + "ET Industry" publish same URL — TF-IDF title cosine ≈ 0.34, below threshold, would both survive without this pass).

**Pass 1 — Title dedup** (TF-IDF cosine, threshold=0.85):
```
TF-IDF(title, ngram_range=(1,2), max_features=10000)
cosine_similarity(matrix) ≥ 0.85 → keep earliest, drop rest
```
Catches syndicated reposts. Title-only for speed — full TF-IDF matrix is O(n²) but vectorized via BLAS (single `cosine_similarity()` call on sparse matrix).

**Pass 2 — Body dedup** (TF-IDF cosine, threshold=0.70):
```
text = article.full_text or article.summary or article.title
TF-IDF(text, ngram_range=(1,2), max_features=10000)
cosine_similarity(matrix) ≥ 0.70 → keep earliest, drop rest
```
Uses `full_text` (scraped content) if available, falls back to `summary` then `title`. Lower threshold (0.70) catches rewritten versions that share key phrases.

Why body threshold is lower than title (0.70 < 0.85): different articles can share common opening phrases ("According to sources...", "The company announced...") but completely different content — 0.70 is aggressive enough to catch rewrites without over-deduplicating different stories that mention the same company.

Real data: ~10% articles removed per run.

---

## Stage 3: NLI Filter

**Model**: `cross-encoder/nli-deberta-v3-small` (~60MB, CPU-only, ~50ms/article)

**Architecture**: DeBERTa cross-encoder — joint encoding of `(article_text, hypothesis)` as token pair, output via `[CLS]` projection → 3 logits → softmax.

**Entailment score** (label index 1, verified from model's `config.json`):
```python
scores = softmax(model_logits, dim=1)  # shape (batch, 3)
entailment = scores[:, 1]              # index 1 = "entailment"
```

Label order: `{0: contradiction, 1: entailment, 2: neutral}` — verified from HuggingFace model card.

**Decision logic**:
```
entailment ≥ 0.88  →  AUTO-ACCEPT  (no LLM call)
entailment ≤ 0.10  →  AUTO-REJECT  (no LLM call)
0.10 < score < 0.88 →  LLM batch classify (GPT-4.1-nano)
```

~80% of decisions are pure math. LLM only handles the 20% ambiguous zone.

**Current hypothesis** (`data/filter_hypothesis.json`):
```
"This article reports on a specific company named in the text that is growing,
raising capital, launching a product, releasing a model or technology, making
an acquisition, facing a regulatory action, issuing a product recall, signing
a major contract, filing for an IPO, entering a partnership, or making a
strategic business move."
```

**Grammar rules** (critical — violation causes model-wide rejection, see arXiv:1909.00161 §3):
1. Structure: `"This article reports on a specific company named in the text that is [ACTION]..."` — NEVER change the prefix
2. NEVER add negation (NOT/except/unless) — NLI models score near-zero entailment for negated hypotheses on ALL inputs
3. NEVER use meta-descriptions ("business news report", "discusses") — NLI trained on factual assertions, scores near-zero for meta
4. Only extend action verb list at the end

**Benchmark results** (2026-03-08, real India 120h data):
- B2B articles: mean entailment 0.859 (Zepto=0.989, Infosys Azure=0.988, TCS Q3=0.981)
- Noise articles: mean entailment 0.333 (PM-US deal=0.116, India-US oil=0.008, ship insurance=0.001)

**NLI threshold history** (each raised from real 120h data audits):
- 0.55 → 0.75: Virat Kohli (0.569), Mukesh Ambani (0.650) were auto-accepting
- 0.75 → 0.88: PM Modi metro (0.778), ED bail hearing (0.822) were auto-accepting
- reject 0.15 → 0.10: Sarvam AI open-source (0.092), Euler Motors PLI (0.076) were falsely rejected

**Research**: Yin et al. (2019) arXiv:1909.00161; arXiv:2312.17543 (+9.4% F1 over keyword classifiers on zero-shot tasks).

---

## Stage 4: Entity NER (Two-Pass)

**Pass 1 — GLiNER** (`engine/extractor.py`):
- Model: `urchade/gliner_small-v2.1` (local, `data/models/gliner_small/`)
- B2B-specific labels: `company`, `startup`, `financial_institution`, `venture_capital_firm`, `enterprise_software`, `executive`
- Span detection on article title + first 500 chars of summary
- Research: Zaratiana et al. (2024) NAACL 2024 naacl-long.300 (+8.2% F1 vs ChatGPT zero-shot NER)

**Pass 2 — spaCy** (`en_core_web_sm`):
- ORG / PERSON / GPE entity types
- Catches standard company names not recognized by GLiNER (especially Indian company names not in training data)
- Merged with GLiNER output, conflicts resolved by confidence score

**Entity quality cache** (`data/entity_quality.json`):
- Per-entity score = EMA of cluster coherence when that entity appeared as primary
- After score ≥ 3.0 → entity promoted: `MIN_SEED_SALIENCE` threshold lowered from 0.30 to 0.22
- Self-improving: entities that consistently produce good clusters get easier seeding over time
- Research: NAACL 2024 naacl-short.49 (+7-12% F1 on entity-centric clustering over time)

**Normalization** (`engine/normalizer.py`):
- rapidfuzz `WRatio ≥ 85` → merge to canonical name
- "Tata Consultancy Services" ↔ "TCS" → unified to highest-frequency mention
- Prevents same company appearing as multiple separate entities

---

## Stage 5: 6-Signal Similarity Matrix

`engine/similarity.py:compute_decomposed_similarity()` → `Dict[str, np.ndarray]` (N×N each)

### Signal 1: Semantic (weight=0.35)

Cosine similarity on 1024-dim neural embeddings:

```
S_semantic[i,j] = (e_i · e_j) / (‖e_i‖ ‖e_j‖)

where e_i = L2-normalized embedding of article i
```

Implementation: `norms = ‖E‖₂`, then `sim = (E/norms) @ (E/norms).T`. Clamped to [0, 1].

### Signal 2: Entity Overlap (weight=0.25)

Jaccard similarity on B2B entity name sets:

```
S_entity[i,j] = |E_i ∩ E_j| / |E_i ∪ E_j|

where E_i = set of B2B entity names in article i (lowercased)
```

Vectorized via scipy sparse: `mat` (N×V binary), `intersection = mat @ mat.T`, `union = rowsums_i + rowsums_j - intersection`.

### Signal 3: Lexical (weight=0.05)

Title word overlap (BM25 proxy):

```
S_lexical[i,j] = |W_i ∩ W_j| / |W_i ∪ W_j|

where W_i = word set from title (doubled) + first 100 chars of summary
```

Titles double-counted intentionally — title words are high-precision signal vs. summary.

### Signal 4: Event Match (weight=0.10)

5W alignment score:

```
S_event[i,j] = 0.40 × I(entity_i == entity_j)          # who match
             + 0.30 × I(event_type_i == event_type_j)    # what match
             + 0.30 × (word_overlap(what_i, what_j))      # what content match
```

where `entity` = primary named entity, `event_type` ∈ {funding, expansion, product_launch, regulatory, m&a, ...}.

### Signal 5: Temporal Decay (weight=0.10)

Dual-sigma Gaussian — avoids pre-knowing event duration:

```
Δt[i,j] = |published_i − published_j|  (in seconds)

S_short[i,j] = exp(−Δt²  / 2σ_short²)    σ_short = 8h
S_long[i,j]  = exp(−Δt²  / 2σ_long²)     σ_long  = 72h

S_temporal[i,j] = max(S_short, S_long)
```

**Why dual-sigma**: breaking news (same-day) clusters via σ=8h; week-long stories still cluster via σ=72h. Single sigma forces a tradeoff. max() picks whichever fits the event type naturally.

### Signal 6: Source Penalty (weight=0.15)

Same-source articles penalized to enforce source diversity in clusters:

```
S_source[i,j] = 0.3   if source_i == source_j   (same_source_penalty)
              = 1.0   otherwise
```

`same_source_penalty = 0.3` → same-source pairs have 70% lower effective similarity.

### Blending

```
S_blended[i,j] = Σ_k  (w_k / Σ_k w_k)  ×  S_k[i,j]

Default weights: semantic=0.35, entity=0.25, source=0.15,
                 event=0.10, temporal=0.10, lexical=0.05
```

All weights sum to 1.0. Weights are learned/updated by `learning/weight_learner.py` (EWC dual-weight system). Signals stored separately in `Dict["semantic"|"entity"|"lexical"|"event"|"temporal"|"source"|"blended"]` for inspection.

---

## Stage 6: Clustering Cascade

`cluster/orchestrator.py` routes by group type and article count:

```
Entity groups (articles about same named entity):
  n < 5        →  single cluster (too small to subdivide)
  5 ≤ n ≤ 50   →  HAC (scipy average linkage, silhouette sweep)
  n > 50       →  HDBSCAN (soft-membership)

Discovery mode (ungrouped articles after entity extraction):
  Any n        →  Leiden community detection on k-NN similarity graph
```

Leiden is NOT triggered by count — it handles all articles that did NOT join
an entity group. HAC/HDBSCAN handle entity-grouped articles only.

### HAC (Hierarchical Agglomerative Clustering)

`cluster/algorithms.py:cluster_hac()`

Linkage method: **average** (cosine-safe; Ward requires Euclidean space).

**Dendrogram silhouette sweep**:
```
for t in arange(t_min, t_max, step=0.05):         # e.g. 0.30 → 0.65
    labels = fcluster(Z, t, criterion="distance")
    sil = silhouette_score(dist_matrix, labels, metric="precomputed")
    adjusted_sil = sil − (n_singletons / n) × singleton_penalty
    → pick t* = argmax(adjusted_sil)
```

`singleton_penalty = 0.5` (FANATIC/EMNLP 2021) — penalizes cuts that create many isolated articles.

**Cophenetic correlation** `ρ = cophenet(Z, condensed_dist)` logged as cluster quality signal.

Adaptive threshold range by group size:
- n < 15: `t_min = max(0.15, hac_threshold_min - 0.05)`, `t_max = 0.80`
- n ≥ 30: `t_min = max(0.25, hac_threshold_min + 0.05)`, `t_max = 0.60`
- default: `t_min = 0.30`, `t_max = 0.65`

Outlier removal: per-sample silhouette `< hac_outlier_silhouette (-0.1)` → moved to noise.
Only articles with actively negative silhouette (worse fit than random) are removed.

### HDBSCAN Soft-Membership

`cluster/algorithms.py:cluster_hdbscan_soft()`

Uses `hdbscan.HDBSCAN(prediction_data=True)` + `hdbscan.approximate_predict()`.

**Soft-membership assignment** (Campello et al. 2013, JMLR):
```
For each noise point p:
    soft_scores = approximate_predict(p)      # shape (n_clusters,)
    if max(soft_scores) > SOFT_NOISE_THRESHOLD (0.10):
        assign p to argmax(soft_scores)
    else:
        keep as noise
```

This recovers ~15-25% of "noise" articles that HDBSCAN typically discards.

Parameters (adaptive by group size):
- n < 30: `min_cluster_size=3, min_samples=2`
- n ≥ 30: `min_cluster_size=5, min_samples=3`

### Leiden (Community Detection)

`cluster/algorithms.py:cluster_leiden()` via `leidenalg` + `python-igraph`.

1. Build k-NN similarity graph: k=20 nearest neighbors per node (from blended similarity matrix)
2. Edge weight = blended similarity score
3. Run Leiden algorithm with `resolution` parameter (Optuna-tuned, default 1.0)
4. Evaluate modularity `Q = (edges_in - expected_edges) / total_edges`
5. Communities with `size < leiden_min_community_size` → noise

**Optuna tuning** (15 trials, 30s timeout per run):
- Search space: `k ∈ [8, 30]`, `resolution ∈ [0.5, 3.0]`
- Objective: maximize `n_communities * mean_coherence`

Real result: k=8, resolution=1.757 → 6 communities, modularity=0.614.

---

## Stage 7: Cluster Validation (7 Checks)

`engine/validator.py` applies 7 math checks per cluster. Cluster PASSES only if ALL checks pass.

| Check | Metric | Threshold | Notes |
|-------|--------|-----------|-------|
| 1. Size | article count | ≥ 2 | Singletons always fail |
| 2. Coherence | mean pairwise cosine similarity | ≥ adaptive (from `threshold_adapter`) | Entity-seeded bypass this check |
| 3. Entity density | articles with ≥1 B2B entity / total | ≥ 0.40 | Filters abstract/meta clusters |
| 4. Source diversity | unique sources / cluster size | ≥ 0.30 | Prevents single-source echo clusters |
| 5. Temporal span | max date - min date | ≤ 30 days | Prevents stale cross-time clusters |
| 6. Signal strength | composite weighted score | ≥ 0.25 | Based on NLI scores of member articles |
| 7. Duplicate gate | near-identical pairs / total pairs | ≤ 0.50 | Catches MinHash false negatives |

**Entity-seeded clusters** (built around a specific named entity by Stage 4): bypass the coherence hard-veto.
Entity co-occurrence IS the relatedness signal; requiring high cosine coherence would eliminate legitimate multi-faceted company coverage.

**Adaptive coherence threshold**: `threshold_adapter.py` adjusts after each run via EMA:
```
EMA_t = α × observed_coherence_t + (1-α) × EMA_{t-1}    α = 0.3
threshold_t = EMA_t × 0.80                               (80% of recent mean)
```

Real data: 6 Leiden clusters → 5/6 passed validation (1 failed source diversity check).

---

## Stage 8: LLM Synthesis

`summarizer.py` — **FIRST LLM CALL** in the entire pipeline.

Input per cluster:
- Up to 8 representative articles (highest similarity to cluster centroid)
- Entity list + event type distribution
- Prior cluster attempts for Reflexion (if retry)

Output (`TrendData`):
- `trend_title`: concise cluster label
- `summary`: 2-3 sentence synthesis
- `evidence_chain`: key quotes + citations
- `industries_affected`: 1st-order and 2nd-order industries
- `synthesis_confidence`: self-assessed 0.0-1.0

**Reflexion retry** (Shinn et al. 2023, arXiv:2303.11366):
```
if synthesis_confidence < 0.60:
    critique = LLM("What's weak about this synthesis? Be specific.")
    retry_synthesis = LLM(original_prompt + critique)     # max 2 retries
```

LLM model tier: `get_lite_model()` — OpenAI GPT-4.1-nano → GeminiDirectLite → Groq Llama 70B.

---

## Stage 9: Product Matching

`match.py` scores each cluster against user's product catalog (`scope.user_products`):

```
fit_score = 0.60 × cosine(cluster_centroid_embedding, product_embedding)
          + 0.40 × keyword_overlap(cluster_entities, product_keywords)
```

Output: `List[MatchResult]` sorted by `fit_score` descending, returned in `IntelligenceResult.match_results`.

---

## Stage 10: Learning Update

After run completes, `pipeline.py` publishes to Signal Bus:

```python
signal_bus.publish_nli_filter(mean_nli, rejection_rate, hypothesis_version, scores_by_source)
signal_bus.publish_clustering(n_clusters, mean_coherence, entity_coverage, noise_rate)
source_bandit.update_from_run(source_nli_means, cluster_contributions)
entity_quality_cache.update(entity_name → cluster coherence)
threshold_adapter.update(coherence_observed, merge_observed)
dataset_enhancer.maybe_update(articles, clusters)    # triggers SetFit if ≥50 examples
```

All 7 loops update in parallel via Signal Bus. Loops read each others' summaries only — no direct imports.

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

| Metric | 48h India run | 120h India run (best) |
|--------|--------------|----------------------|
| Articles fetched | 433 | ~1200 |
| After dedup | 386 (10.9% removed) | ~1080 |
| After NLI filter | 53 (13.7% pass rate) | ~200 (18%) |
| NLI mean (kept) | 0.788 | 0.721 |
| Entity groups | 8 | ~20 |
| Leiden clusters | 6 | ~14 |
| Clusters passed | 5/6 (83%) | ~12/14 (86%) |
| Leiden modularity | 0.614 | 0.55–0.65 |
| Pipeline time | ~14 min | 14–30 min |
| LLM calls (synthesis) | ~30 | ~80 |

**Why NLI pass rate is low (13-18%)**: correct behavior. India RSS includes SEBI filings (company disclosures, not B2B signals), economic commentary, political news. NLI is strict — only articles that explicitly mention a company doing a business action pass. Goal is precision, not recall.

---

## Algorithm References

| Algorithm | Paper | Key result |
|-----------|-------|-----------|
| NLI zero-shot | Yin et al. (2019) arXiv:1909.00161 | +9.4% F1 over keyword classifiers |
| GLiNER NER | Zaratiana et al. (2024) NAACL naacl-long.300 | +8.2% F1 vs ChatGPT zero-shot |
| HDBSCAN soft | Campello et al. (2013) JMLR | Noise recovery via soft membership |
| Leiden community | Traag et al. (2019) Sci Reports 9:5233 | Better modularity than Louvain |
| HAC singleton penalty | FANATIC (EMNLP 2021) | Adjusted silhouette for unbalanced clusters |
| Reflexion retry | Shinn et al. (2023) arXiv:2303.11366 | LLM self-critique improves synthesis quality |
| SetFit hypothesis | Tunstall et al. (2022) arXiv:2209.11055 | 8 examples ≈ 3000-example fine-tune |
| Entity quality cache | NAACL 2024 naacl-short.49 | +7-12% F1 on entity-centric clustering |
