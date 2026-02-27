# app/learning/ -- Self-Learning System

The pipeline improves autonomously through 6 feedback loops that update between runs. Each loop measures a different aspect of pipeline quality and adjusts parameters accordingly.

## 6 Learning Loops

### 1. Source Bandit (`source_bandit.py`)

**Mechanism**: Thompson Sampling with Beta(alpha, beta) posteriors per RSS source.

Each source maintains a Beta distribution. After each run, sources whose articles ended up in high-quality clusters get alpha incremented (reward); sources whose articles were filtered or fell into noise get beta incremented (penalty). On the next run, sources are sampled from their posteriors -- high-quality sources get fetched first.

**Persistence**: `data/source_bandit.json`

### 2. Trend Memory (`memory.py` in `app/trends/`)

**Mechanism**: ChromaDB centroid storage with cosine matching.

Stores cluster centroids after each run. On the next run, new centroids are compared against stored ones. Matches (cosine > 0.80) get low novelty scores and blended centroids (EMA 70/30). Non-matches are flagged as genuinely new trends. Stale centroids (not seen in 14 days) are pruned.

**Persistence**: `data/memory/` (ChromaDB)

### 3. Weight Learner (`weight_learner.py`)

**Mechanism**: Dual-path weight adaptation.

Adjusts the scoring weights used by `composite.py` to rank trends:

| Path | Trigger | Learning Rate |
|------|---------|--------------|
| Human feedback | 50+ feedback records | 3x base rate |
| Auto-learning | 5+ pipeline runs | Base rate, decaying |
| Default | Cold start | Static weights |

**Reward signal**: Composite quality = 40% KB hit rate + 30% lead quality + 30% OSS. Specifically not 100% OSS (which would be circular -- OSS measures synthesis text while weights influence trend filtering).

**Safety**: Weights clamped to [0.02, 0.40], normalized to sum 1.0, variance check prevents learning from undiscriminating scores.

**Persistence**: `data/learned_weights.json`

### 4. Company Bandit (`company_relevance_bandit.py` in `app/agents/`)

**Mechanism**: Thompson Sampling over (company_size, event_type) arms.

Tracks which company types produce actionable leads. Rewards:
- 1.0: Company appeared in a confirmed causal chain hop (lead sheet generated)
- 0.5: Company in impact analysis with high confidence
- 0.0: Company discarded as too generic / not actionable

**Persistence**: `data/company_bandit.json`

### 5. Adaptive Thresholds (`pipeline_metrics.py`)

**Mechanism**: EMA (Exponential Moving Average) on per-run metrics.

Tracks coherence thresholds, merge thresholds, and signal cutoffs across runs. When cluster quality drifts, thresholds auto-adjust. The JSONL log also supports z-score drift detection for alerting.

**Persistence**: `data/pipeline_run_log.jsonl`

### 6. Auto-Feedback (`app/tools/feedback.py`)

**Mechanism**: Automated trend rating for learning loop seeding.

After each run, trends are auto-rated based on objective metrics (cluster size, source diversity, entity count, OSS). Auto-generated feedback is tagged with `metadata.auto=True` to separate it from human feedback. This provides warm-start data for the weight learner before any human feedback exists.

**Persistence**: `data/feedback.jsonl`

## Data Flow

```
Pipeline Run
    |
    +---> pipeline_metrics.py  ---> data/pipeline_run_log.jsonl
    |         |
    |         +---> EMA thresholds (loop 5)
    |
    +---> source_bandit.py     ---> data/source_bandit.json (loop 1)
    |
    +---> trend memory         ---> data/memory/ ChromaDB (loop 2)
    |
    +---> auto-feedback        ---> data/feedback.jsonl (loop 6)
    |
    +---> company_bandit       ---> data/company_bandit.json (loop 4)
    |
    v
weight_learner.py  ---> data/learned_weights.json (loop 3)
    |
    v
composite.py scoring weights (next run)
```

## Objective Specificity Score (OSS)

Defined in `specificity.py`. A pure text-analysis metric (no LLM self-rating) that measures how actionable a trend synthesis is:

| Component | Weight | Measures |
|-----------|--------|----------|
| entity_density | 0.25 | Named entities per total words |
| numeric_density | 0.20 | Numbers, percentages, dates, currency |
| geo_specificity | 0.20 | Specific locations (not just "India") |
| size_mention | 0.15 | Employee count ranges present |
| industry_depth | 0.20 | Specific sub-segment vs generic "sector" |

OSS is computed after synthesis, before grading. It serves as the primary autonomous quality signal for the weight learner. The score is not circular: OSS measures text properties of synthesis output while scoring weights influence which trends survive filtering -- different systems measuring different things.

## Feedback API Integration

The FastAPI endpoint `POST /api/v1/feedback` accepts human ratings:

```json
{
  "item_type": "trend",
  "item_id": "trend-abc123",
  "rating": 4,
  "comment": "Highly relevant to my portfolio"
}
```

Feedback is stored in `data/feedback.jsonl`. When 50+ human records accumulate, the weight learner switches from auto-learning to human-feedback-driven adaptation with a 3x learning rate boost.

Summary endpoint: `GET /api/v1/feedback/summary` returns counts and average ratings by item type.
