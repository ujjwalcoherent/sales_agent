# app/learning/ — Self-Learning System

5 active learning loops that improve the pipeline between runs. All loops communicate via `signal_bus.py` exclusively — no direct imports between loops.

```
learning/
├── signal_bus.py          # Cross-loop pub/sub backbone (LearningSignalBus dataclass)
├── source_bandit.py       # Loop 1: Thompson Sampling over RSS source arms
├── company_bandit.py      # Loop 2: Thompson Sampling over company-trend (size × event_type) arms
├── contact_bandit.py      # Loop 3: Thompson Sampling over (role × event_type × company_size) arms
├── threshold_adapter.py   # Loop 4: EMA-based threshold adaptation with CUSUM guard
├── dataset_enhancer.py    # Auto-labels articles → SetFit bootstrap dataset (training data only)
├── experiment_tracker.py  # Logs experiment results; snapshot/restore for rollback
└── pipeline_metrics.py    # Metrics collector → data/pipeline_run_log.jsonl
```

**Removed (as of March 2026 pipeline surgery):**
- `weight_learner.py` — DELETED (EWC weight learning; weights never loaded back, 0 effect after 36 runs)
- `path_router.py` — DELETED (unused routing layer)
- `experience_library.py` — DELETED (write-only; find_proven_fix never called)
- `hypothesis_learner.py` — DELETED (SetFit training loop removed; dataset_enhancer still auto-labels)
- `pipeline_validator.py` — DELETED (self-correcting gates; corrections gated behind 0.75 confidence that never fired)
- `meta_reasoner.py` — DELETED (all methods returned empty stubs; `_enabled = False` always)

---

## Signal Bus — `signal_bus.py`

`LearningSignalBus` is a plain dataclass. One instance per pipeline run. Loops write their section after updating, then read other loops' sections for cross-pollination.

**Three-phase protocol** prevents circular reads:
1. Phase 1 — each loop publishes from THIS run's raw data
2. Phase 2 — bus computes derived signals (`system_confidence`, `exploration_budget`)
3. Phase 3 — each loop reads others' summaries, applies small adjustments

**Publish methods:**
```python
bus.publish_source_bandit(posterior_means, previous_means)
bus.publish_trend_memory(lifecycle_counts, avg_novelty)
bus.publish_company_bandit(arm_means)
bus.publish_adaptive_thresholds(anomalies, drift)
bus.publish_auto_feedback(distribution, mean_quality)
bus.publish_nli_filter(mean_entailment, rejection_rate, hypothesis_version, scores_by_source)
bus.publish_backward_signals(cluster_coherence_by_source, lead_quality_per_cluster)
```

**Published fields (key):**
- Source Bandit section: `source_posterior_means`, `top_sources`, `source_diversity_index`, `source_exploration_rate`, `source_degraded`
- NLI Filter section: `nli_mean_entailment`, `nli_rejection_rate`, `nli_hypothesis_version`, `nli_scores_by_source`
- Clustering section (backward cascade): `cluster_coherence_by_source`, `lead_quality_per_cluster`
- Derived: `system_confidence` = weighted mean of 5 stability signals; `exploration_budget = max(0.10, min(0.50, 1.0 - system_confidence))`

**Persisted:** `data/signal_bus.json` (warm start across runs)

**Research:** Landgren et al. (2016) Collaborative Multi-Armed Bandits; Cowen-Rivers et al. (NeurIPS 2020) HEBO.

---

## Loop 1 — Source Bandit (`source_bandit.py`)

**What it learns:** Which RSS sources produce B2B-signal articles worth clustering.

**Algorithm:** Thompson Sampling over Beta(α, β) posteriors. One arm per source.

**Reward formula** (weights sum to 1.0):
```
reward = 0.30 × nli_score              # PRIMARY: mean NLI entailment for this source
       + 0.20 × avg_cluster_quality    # cluster coherence of clusters source contributed to
       + 0.15 × uniqueness             # 1 - dedup_rate (low dedup = high uniqueness)
       + 0.10 × entity_richness        # min(1.0, raw_entity_richness / 5.0)
       + 0.10 × content_score          # full-text availability rate
       + 0.10 × backward_coherence     # cluster_coherence_by_source (cascade signal)
       + 0.05 × avg_oss               # synthesis specificity score

n_obs = min(len(article_ids), 5)       # capped at 5 pseudo-observations per run
α_s += reward × n_obs
β_s += (1 - reward) × n_obs
```

**Posterior mean** = α / (α + β). Sample `θ_s ~ Beta(α_s, β_s)` at fetch time; fetch sources in descending `θ_s` order.

**Decay** (recency bias): `α = max(1.5, 1.0 + (α - 1.0) × 0.97)` per run before update.

**Cap**: total α + β capped at 200 to prevent numerical instability.

**Informed priors** (Russo et al. 2018):
- B2B-tagged sources (funding/VC/startup/fintech tags): Beta(3, 1) → prior mean 0.75
- Noise-tagged sources (geopolitical/macro/trade/aggregator tags): Beta(1, 3) → prior mean 0.25
- Unknown sources: Beta(1, 1) → flat prior mean 0.50

`get_adaptive_credibility(source_id)` returns 50/50 blend of static credibility + posterior mean.

**Persisted:** `data/source_bandit.json`

**Research:** Chapelle & Li (2011) "An Empirical Evaluation of Thompson Sampling"; Russo et al. (2018) arXiv:1707.02038.

---

## Loop 2 — Company Bandit (`company_bandit.py`)

**What it learns:** Which company-trend pairings convert to actionable leads.

**Algorithm:** Thompson Sampling on `company_id` arms. Arms are strings — either `"{company_size}_{event_type}"` for contextual scoring, or raw company IDs for per-company tracking.

**Reward sources:**
- `1.0` — company appeared in confirmed causal chain hop (lead sheet generated)
- `0.5` — company appeared in impact analysis with high confidence
- `0.0` — company discarded as too generic or not actionable

**Update rule:** `α += reward; β += (1 - reward)`

**Decay:** `α = max(1.0, 1.0 + (α - 1.0) × 0.97)` prevents Thompson variance collapse.

**`compute_relevance()` formula:**
```
score = 0.35 × bandit_score(company_size_event_type arm)
      + 0.30 × industry_match        [0.0-1.0]
      + 0.20 × intent_signal_strength [0.0-1.0]
      + 0.15 × severity_mult          {high→1.0, medium→0.7, low→0.4}
```

**Persisted:** `data/company_bandit.json`

---

## Loop 3 — Contact Bandit (`contact_bandit.py`)

**What it learns:** Which contact role to approach per (event_type × company_size) combination.

**Algorithm:** Thompson Sampling on `(role × event_type × company_size)` arms. `ContactArm` dataclass with Beta(α, β) posterior.

**Reward constants:**
```python
REWARD_EMAIL_OPEN      = 0.3    # weak positive
REWARD_EMAIL_REPLY     = 1.0    # strong positive
REWARD_EMAIL_BOUNCE    = 0.0    # neutral
REWARD_LEAD_CONVERTED  = 1.5    # strongest signal
REWARD_EMAIL_SKIPPED   = -0.1   # implicit negative
```

**Update rule:** rewards > 1.0 are scaled to [0, 1] for α; β gets 0 penalty for full-success rewards.

**Informed priors by role category:**
| Role category | α | β | Prior mean |
|--------------|---|---|-----------|
| VP, Director | 3.0 | 1.0 | 0.75 |
| Head of | 2.5 | 1.0 | 0.71 |
| CISO, Founder | 2.5 | 1.5 | 0.63 |
| CTO, CMO | 2.0 | 2.0 | 0.50 |
| CEO (enterprise) | 1.5 | 4.5 | 0.25 |

`rank_roles()` draws Thompson samples; `get_top_roles()` uses posterior mean (no sampling — for reporting only).

**Persisted:** `data/contact_bandit.json`

**Research:** Chapelle & Li (2011) arXiv:1111.1797; Li et al. (2010) LinUCB WWW 2010.

---

## Loop 4 — Threshold Adapter (`threshold_adapter.py`)

**What it learns:** Optimal thresholds for NLI filter, coherence validation, HDBSCAN noise, and pass rate gating.

**Algorithm:** Exponential Moving Average (EMA) with α = 0.10 (code constant `_EMA_ALPHA`).

**EMA formula:**
```
new_threshold = 0.10 × observed + 0.90 × current
```

**Thresholds adapted:**
- `filter_auto_accept`: if accept_rate < 20% → lower by 0.02; if > 80% → raise by 0.02 (clamped [0.20, 0.50])
- `val_coherence_min`: target = `0.85 × observed_coherence` (clamped [0.30, 0.65])
- `val_composite_reject`: if pass_rate < 30% → lower by 0.03; if > 80% → raise by 0.03 (clamped [0.30, 0.70])
- `hdbscan_soft_noise_threshold`: if noise_rate > 40% → lower by 0.01; if < 10% → raise by 0.01 (clamped [0.05, 0.20])

**CUSUM guard** (Page 1954) — freezes adaptation when quality degrades:
```
deviations = [0.4 - q for q in quality_history[-5:]]
if cusum_detect(deviations, threshold=0.3, drift=0.05):
    _frozen = True   # adaptation suspended until quality_score >= 0.45
```

`cusum_detect()`: Page's cumulative sum change detection. Triggers if `s_pos > 2.0` or `s_neg > 2.0` (default threshold).

**Persisted:** `data/adaptive_thresholds.json`

**Research:** EMA standard practice per Sutton & Barto "Reinforcement Learning" Ch. 2; Page (1954) "Continuous Inspection Schemes" Biometrika.

---

## Dataset Enhancer (`dataset_enhancer.py`)

Auto-builds a labeled training dataset from run results — no human annotation required. Produces `data/dynamic_dataset.jsonl` for downstream fine-tuning or analysis.

**Thresholds:**
```python
POSITIVE_COHERENCE_THRESHOLD = 0.70   # cluster coherence
POSITIVE_NLI_THRESHOLD       = 0.85   # NLI entailment (dual-gate)
NEGATIVE_COHERENCE_THRESHOLD = 0.25   # cluster coherence
NEGATIVE_NLI_THRESHOLD       = 0.10   # NLI entailment
N_RETRAIN_THRESHOLD          = 50     # total examples before triggering retrain
MAX_CLASS_RATIO              = 2.0    # max positive:negative imbalance
MAX_DATASET_SIZE             = 5000   # cap to prevent training blowup
```

**Labeling rules:**

| Source | Label | Condition |
|--------|-------|-----------|
| Cluster articles | POSITIVE | coherence > 0.70 AND NLI > 0.85 (dual gate) |
| NLI-rejected articles | NEGATIVE | NLI score < 0.10 |
| Cluster articles | NEGATIVE | coherence < 0.25 |
| Reuters-21578 earn/acq/corp | POSITIVE | gold-standard |
| Reuters-21578 grain/macro/dlr | NEGATIVE | gold-standard |
| AG News Business class | POSITIVE | ground truth |
| AG News Sports/World classes | NEGATIVE | ground truth |

**Why dual gate for positive cluster examples:** coherence-only would make topically similar non-B2B clusters (e.g. cricket commentary) positive training data. NLI-only would include articles in low-quality clusters. Both gates together confirm genuine B2B signal.

**Deduplication:** MD5 hash of text content. Stored in `data/dynamic_dataset.jsonl`.

---

## Experiment Tracker (`experiment_tracker.py`)

Each pipeline run = one `ExperimentRecord` appended to `data/experiments.jsonl`.

**Snapshot files** (for regression rollback):
```python
_SNAPSHOT_FILES = [
    "data/adaptive_thresholds.json",
    "data/filter_hypothesis.json",   # hypothesis must roll back with other learning state
    "data/nli_baseline.json",        # stale baseline triggers false retraining after rollback
]
```

**Snapshot directory:** `data/_learning_snapshot/`

**ExperimentRecord fields:** `run_id`, `mean_oss`, `mean_coherence`, `noise_rate`, `actionable_rate`, `article_count`, `cluster_count`, `status` ("keep"/"discard"/"crash"), `learning_updates`.

---

## Key Invariants

- Loops communicate via `signal_bus.py` ONLY — no direct cross-loop imports
- All persistent state in `data/*.json` or `data/*.jsonl` — restart-safe
- Every loop loads state on `__init__`, saves immediately after each update
- Learning updates run in `learning_update_node` (final pipeline stage) — always executes even when no viable trends found
- `WeightLearner`, `ExperienceLibrary`, `PathRouter`, `HypothesisLearner`, `PipelineValidator`, `MetaReasoner` are all DELETED — do not reference them
