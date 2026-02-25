# Pipeline Status Report — 2026-02-25

## Vector DB — FRESH START ✅
- Old data archived to: `data/_archive_vectordb_20260225_170812/`
- `data/memory/` (trend centroids): CLEARED → 0 entries
- `data/article_cache/` (1470 articles): CLEARED → 0 entries
- Next run will start building fresh learning history

## Bugs Fixed This Session

### 1. Learning Loop Never Fired ✅ FIXED
**File:** `app/agents/orchestrator.py`
- Added `learning_update_node` that runs at END of every pipeline run
- Graph was: `lead_gen → END` (learning never happened)
- Graph now: `lead_gen → learning_update → END`
- Also: quality "skip" path now goes to `learning_update` not `END`
- Source bandit `update_from_run()` now gets called after every run
- Learning signals from `deps._signals` are now consumed + persisted to `data/learning_signals.jsonl`

### 2. Hardcoded `geo="India"` ✅ FIXED
**File:** `app/agents/orchestrator.py` line ~379
- Was: `geo="India"` (hardcoded string)
- Now: `geo=get_settings().country` (reads from config/env)
- Set `COUNTRY=India` in `.env` for current runs (default preserved)

### 3. First/Second Order NOT Real ✅ FIXED
**Files:** `app/schemas/llm_outputs.py`, `app/agents/workers/council/impact_council.py`, `app/agents/workers/impact_agent.py`
- Was: `direct_impact = affected_company_types[:4]` and `indirect_impact = business_opportunities[:4]` (SAME call, sliced differently)
- Now:
  - Added `first_order_companies` + `first_order_mechanism` to LLM schema
  - Added `second_order_companies` + `second_order_mechanism` to LLM schema
  - Impact council system prompt now EXPLICITLY instructs:
    - First-order = must be evidenced in source articles (cite the article)
    - Second-order = must show transmission chain (A→B→C with mechanism)
  - `impact_agent.py` now reads from explicit first/second order fields

## Remaining Bugs (Next Session)

### CRITICAL
1. **specificity.py** has 40+ hardcoded India city names in regex → needs `geonamescache` dynamic loader
2. **source_bandit.json was empty** — learning node now wired, but need a test run to confirm it populates
3. **Article cluster map not stored in deps** — `learning_update_node` builds approximate cluster_oss but can't do full source bandit update without `_article_cluster_map` in deps

### HIGH
4. **Company KB is empty** — `data/company_kb.sqlite` exists but has no data. Need to run `scripts/load_india_mca.py` or similar
5. **Dataset search incomplete** — GLEIF, World Bank Pink Sheet, GDELT not integrated yet

## Self-Learning Architecture Status

```
Source Bandit (Thompson Sampling)  — ✅ code exists, ✅ now called at end of run
Weight Learner (OSS auto-learning) — ✅ code exists, data flows through quality_report_log.jsonl
Trend Memory (ChromaDB centroids)  — ✅ code exists, ✅ CLEARED + ready for fresh start
Article Cache (ChromaDB)           — ✅ code exists, ✅ CLEARED + ready for re-embedding
Learning Signals JSONL             — ✅ now persisted at data/learning_signals.jsonl each run
```

## Datasets To Integrate (Priority Order)

| Dataset | What It Solves | Priority |
|---------|---------------|----------|
| GLEIF LEI Golden Copy | Company KB - 2.5M global companies, replace Tavily | 🔴 CRITICAL |
| India MCA data.gov.in | 1.8M Indian companies for company KB | 🔴 CRITICAL |
| World Bank Pink Sheet | Commodity prices for synthesis hooks (gold, steel, oil) | 🔴 CRITICAL |
| GDELT 2.0 Event DB | Burst detection, ADM1 geo coding | 🟡 HIGH |
| UN Comtrade | Supply chain shocks, import/export flows | 🟡 HIGH |
| SEC EDGAR | US company triggers | 🟡 HIGH |

## How First/Second Order Works (Post-Fix)

**The causal chain architecture:**
```
Impact Council → first_order_companies (directly named in articles)
              → second_order_companies (supply chain inference with chain logic)

Causal Council → hop1 = first-order (directly affected segments)
              → hop2 = second-order (buyers/suppliers of hop1)
              → hop3 = third-order (downstream of hop2)
```

Both systems now work together: Impact Council identifies the segments, Causal Council finds actual companies within those segments using the company KB.

## How Clustering Works

1. RSS articles fetched from 40+ sources (bandit-weighted by OSS)
2. spaCy NER extraction → entity graph
3. `text-embedding-3-small` (1024-dim) embeddings via OpenAI or local
4. Leiden community detection on k-NN cosine similarity graph
5. Coherence validation (Jaccard → MAD ejection → connected components)
6. Synthesis via LLM with OSS retry if score < 0.35
7. Trend memory (ChromaDB) tracks centroids across runs for novelty scoring
8. Per-run: source bandit updated, weight learner adapts thresholds

## Next Steps For Next Session

1. Run the pipeline (`streamlit run streamlit_app.py`) and check output
2. Verify source bandit gets populated (check `data/source_bandit.json` after run)
3. Load company KB data (even a small subset for testing)
4. Download GLEIF sample + World Bank Pink Sheet
5. Fix `specificity.py` hardcoded geo patterns → use `geonamescache`
6. Test first/second order separation in actual pipeline output
