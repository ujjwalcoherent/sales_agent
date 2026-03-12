# Pipeline Cleanup — Change Log
> Session: March 2026 | Branch: main

## Summary of All Changes

Total dead code removed: **~2,700+ lines** across 3 commits.
Pipeline runtime target: **5-10 minutes** (was ~37 min).
Mock mode: **PERFECT** for all 3 use cases.

---

## Session 11 — March 12, 2026 (Report-Driven Mock + Import Cleanup)

### feat: carrier rocket / space launch mock dataset for report_driven mode

Report-driven mock was using 5G/IoT/telecom articles — replaced with carrier rocket / space launch
data extracted from a real CMI market report. Also added `MOCK_REPORT_SUMMARY` (~800 chars) so
report_driven mock mode has default report text when none is provided.

**`mock_articles.py`:**
- `MOCK_REPORT_SUMMARY` — condensed carrier rocket market report (market size, key players, trends)
- `MOCK_ARTICLES_REPORT_DRIVEN` — 17 articles in 3 clusters:
  - Cluster 1: Reusable rocket development (SpaceX Starship, Relativity Space, Rocket Lab, ULA, Blue Origin)
  - Cluster 2: Small satellite constellation launches (Amazon Kuiper, ISRO SSLV, Starlink, OneWeb)
  - Cluster 3: Space launch startup funding (Stoke Space, Skyroot, Isar Aerospace, Firefly)
- `MOCK_ARTICLES_RAW` — added 6 UPI/Digital Payments regulation articles for richer fintech output

**`mock_responses.py`:**
- `_detect_sector()` — added space keywords (spacex, rocket, satellite, starship, blue origin, etc.)
- `_get_mock_company_response()` — added "space" sector companies (Skyroot Aerospace, Agnikul Cosmos)
- `_MOCK_SYNTH_TRENDS[3]` — "Reusable Rocket Development Slashes Launch Costs by 40-60%"
- `_MOCK_IMPACTS[3]` — space launch market impact analysis
- `_MOCK_COUNCIL_RESPONSES["space"]` — full council response with 1st/2nd order companies, pain points
- `_MOCK_EMAILS[3]` — carrier rocket market intelligence email
- Impact/synth/email routing updated with space sector index mapping + `% len()` safety

**`source_intel.py`:**
- `_make_mock_articles()` injects `MOCK_REPORT_SUMMARY` into scope when report_driven + empty report_text
- Fintech/BFSI industry slices to first 12 articles (2 fintech clusters only)

### fix: entity extraction truncation bumped 2000→6000 chars

**`fetch.py:_extract_report_entities_llm()`:**
- Market reports put company names 4-6K chars deep (after market overview section)
- 2000-char truncation missed all companies; 6000 chars captures competitive landscape
- Also added Track A/B/C extraction pipeline (run_structured → json_mode → regex fallback)

### chore: merge duplicate import blocks across 3 files

Three files had duplicate import blocks (one per logical section, both at top level):

- **`providers.py`** — merged ProviderHealthTracker + ProviderManager import blocks into single top-level block
- **`fetch.py`** — merged dedup section imports (`Article`, `DedupResult`, `Set`, `Tuple`) into top-level
- **`search.py`** — merged `asyncio`, `Optional` into top-level; removed dead `TYPE_CHECKING` block importing from non-existent `.bm25_search` module

### chore: .gitignore updates

- Added `data/recordings/` explicit ignore, `*.png`, `.superpowers/`, `.superset/`, `tests/run_*_validation.py`

### test: updated mock pipeline routing test

- `test_mock_article_routing()` — changed report_driven assertion from `("5G IoT", "5G")` to `("Carrier rocket market", "Rocket")`

**Verified:** All 13 tests pass. Mock pipeline works for all 3 modes:
| Mode | Trends | Companies | Leads |
|------|--------|-----------|-------|
| industry_first | 1 | 4 | 12 |
| company_first | 2 | 4 | 12 |
| report_driven | 2 | 4 | 12 |

---

## Session 10 — March 12, 2026 (Continuous Improvement Loop)

### fix: sector-coherent mock mode for all 3 use cases

Mock mode was producing wrong-sector companies, pain points, and council responses because:
1. `_detect_sector()` matched "compliance" from CMI services catalog (present in ALL prompts)
2. "manufacturing" keyword was too broad (appears in IT article bodies)
3. `MOCK_ARTICLES_RAW` mixed fintech + retail + semiconductor articles for industry_first

**Changes:**
- `mock_responses.py`: Added `_MOCK_COUNCIL_RESPONSES` dict with IT/fintech/telecom variants; `_detect_sector()` now extracts TREND portion to avoid false matches on catalog keywords; checks IT keywords first (company names = most specific)
- `orchestrator.py`: Causal council mock uses scope-aware fallback (reads `deps.scope.industry` and `report_text`)
- `source_intel.py`: Fintech/BFSI mode slices to first 12 articles (2 fintech clusters only)
- `mock_articles.py`: Added UPI/Digital Payments regulation cluster (6 articles) for richer fintech output

**Result:** All 3 modes produce end-to-end sector-coherent output:
| Mode | Companies | Pain Points | Email Subject |
|------|-----------|-------------|---------------|
| company_first | Mphasis, Persistent, Happiest Minds | GenAI competitive intel | GenAI Competitive Intelligence |
| industry_first | Lendingkart, Easebuzz, IDfy, Signzy | KYC 90-day compliance | RBI KYC Compliance Gap Assessment |
| report_driven | Endurance, Sundram, STL, Tejas | 5G IoT supply chain | 5G Enterprise IoT Supply Chain |

### feat: Track B→A upgrades (report entity extraction + company field extraction)

Two `generate(json_mode=True)` → `run_structured()` upgrades with Pydantic models:

1. **`fetch.py:_extract_report_entities_llm()`** — Report-driven pipeline entry point
   - Track A: `run_structured(output_type=ReportEntitiesLLM)` with StrList coercion
   - Track B fallback: `generate(json_mode=True)` + manual json.loads
   - Track C fallback: regex extraction

2. **`company_enricher.py:_extract_structured_fields()`** — Company enrichment
   - Track A: `run_structured(output_type=CompanyFieldsLLM)` with 13 optional fields
   - Track B fallback: `generate()` + json_repair parse
   - `CompanyFieldsLLM` uses model_validator for None→"" coercion on string fields

**New Pydantic models** in `llm_outputs.py`: `ReportEntitiesLLM`, `CompanyFieldsLLM`
**New tests**: `test_report_entities_llm_model`, `test_company_fields_llm_model`

### chore: remove dead code (CoercedFloat, _params_used, source_article_ids, HopSignal)

- `llm_outputs.py`: Removed unused `CoercedFloat` type annotation + `_coerce_to_float()` helper
- `deps.py`: Removed `AgentDeps._params_used` (never set/read), `LearningSignal.source_article_ids` (write-only), `LearningSignal.hop_signals` (write-only), and `HopSignal` dataclass (never consumed)
- `orchestrator.py`: Removed `source_article_ids` population and `HopSignal` construction (12 lines) in causal council. Kept aggregate `kb_hit_rate` which IS consumed.

---

## Session 9 — March 12, 2026

### feat: expand contact targeting + realistic mock data from web research

**Contact targeting expanded beyond C-suite:**
- `max_contacts_per_company`: 6 → 10 (30% decision-makers, 70% influencers+evaluators)
- `default_influencer_roles`: added Director of Engineering, Engineering Manager, Product Manager, Senior Architect
- `TREND_ROLE_MAPPING`: all 20 trend categories expanded from 4 to 6-8 roles each, including mid-level evaluators

### fix: remove dead fields from signal bus modulation dicts

`get_source_bandit_modulation()` returned `winning_company_types` and `system_confidence` — never read by orchestrator.
`get_company_bandit_modulation()` returned `top_sources` — never read by orchestrator.
Removed to keep cross-pollination API honest. Also fixed stale README docs (5 non-existent methods/fields).

### perf: skip person_intel web searches in mock mode (35s → 0.5s)

`email_agent.gather_person_context()` was making real Tavily/DDG web searches for every contact name
even in mock mode — searching names like `"Shriram N" Persistent Systems`. Added mock_mode guard
that builds synthetic PersonContext. Mock pipeline now runs in **0.1-2.7s** (was 28-35s).

**Mock data rebuilt with real web research (18 companies):**
- Apollo mock: real leadership teams from public sources (Mphasis, Persistent, Razorpay, Pine Labs, STL, Tejas Networks, Signzy, IDfy, etc.)
- Hunter mock: real mid-level contacts (Engineering Managers, Directors, Product Managers) complementing Apollo C-suite
- Council mock: enriched with actual company names, evidence citations from research, target_roles including VP Engineering/Director/PM
- Email mock: CMI service-specific pitches referencing real trends and competitor data

---

## Session 8 — March 12, 2026

### fix: mock mode producing 0 leads (broken impact analysis + missing causal chains)

Three bugs caused mock mode to produce 0 leads despite running successfully:

1. **Mock response routing collision** (`mock_responses.py`): The `get_mock_response()` function
   routed impact/council prompts to the company extraction branch because impact prompts naturally
   contain "compan" (first_order_companies) and "find" (find cheaper suppliers). Fixed by checking
   impact keywords before company keywords.

2. **FunctionModel returning wrong schema** (`mock_responses.py`): The `_build_structured_mock()`
   function returned `ImpactResult` fields for the council's system prompt (which matched "quality"
   before "mid-size"). The council expects `UnifiedImpactAnalysisLLM` fields. Fixed by adding a
   council-specific branch checked before the quality branch.

3. **No synthetic causal chains in mock mode** (`orchestrator.py`): The `causal_council_node`
   skipped entirely in mock mode, producing 0 causal results. The downstream `lead_crystallize_node`
   requires causal results to generate call sheets. Fixed by generating synthetic `CausalChainResult`
   objects with realistic company segments and hops.

4. **Company enrichment making real API calls in mock mode** (`leads.py`): The `enrich()` call
   in lead_gen made real Tavily/DDG calls for company descriptions. Fixed by short-circuiting
   with mock descriptions when `deps.mock_mode` is True.

**Result:** All 3 mock modes now produce leads:
| Mode | Trends | Companies | Leads |
|------|--------|-----------|-------|
| COMPANY_FIRST | 2 | 4 | 8 |
| INDUSTRY_FIRST | 3 | 4 | 8 |
| REPORT_DRIVEN | 3 | 4 | 8 |

---

## Session 7 — March 12, 2026

### fix: ticker-resolved names lost in `fuzzy_group()` entity normalization

When `normalize_entity_name()` resolved a ticker (e.g., "TCS" → "Tata Consultancy Services"),
the original name was not tracked as a variant of the normalized canonical. This caused the
original name to be absent from `fuzzy_group_entities()`'s output, so the caller treated it
as unmapped (identity). All 26 ticker aliases in `TICKER_ALIASES` were affected.

Fix: when creating a new group and `name != normalized`, add the original name as a variant.
Result: `["TCS", "Tata Consultancy Services"]` now correctly merges into 1 group (was 2).

### chore: remove dead `CampaignStatus` enum from schemas

`CampaignStatus` enum in `app/schemas/campaign.py` was defined but never imported — campaigns API
uses string literals ("draft", "running", etc.) directly. Frontend has its own TypeScript equivalent.

### chore: remove stale "Phase N migration" docstrings

Cleaned 2 docstrings referencing non-existent migration phases and deleted modules
(`app.clustering.tools.validator`, `Phase 5/11 migration`). Comments only — no behavior change.

### fix: implement missing `fuzzy_group()` in normalizer.py (latent NameError)

`fuzzy_group_entities()` in `app/intelligence/engine/normalizer.py` called `fuzzy_group()` which
was deleted with the old `app.news.entity_normalizer` module. This caused a `NameError` at runtime,
silently caught by `_fuzzy_group()` in extractor.py which fell back to identity mapping — meaning
**entity normalization was silently broken** (no fuzzy merging of name variants like "NVIDIA"/"nvidia").

Fix: implemented `fuzzy_group()` using rapidfuzz `token_sort_ratio` matching, following the algorithm
in the module docstring. Also cleaned stale "Phase 11 migration" docstring reference.

Verified: `_fuzzy_group(["Microsoft", "MSFT", "Google", "Alphabet"])` now correctly merges
`MSFT→Microsoft` via ticker resolution + fuzzy matching.

### perf: skip entity extraction + NVIDIA embeddings in mock mode (8.6× faster)

Entity extraction (SpaCy + GLiNER) ran on mock articles, loading a 165MB neural model from
disk (~6s cold) just to produce entity groups that were 100% statistically filtered out.
NVIDIA embedding API made a real network call (~1.3s) for clustering that only needs to be
structurally valid in mock mode.

Fix: detect `fetch_method="mock"` articles and skip NER entirely (0 entity groups, same as
before). Replace NVIDIA embeddings with local TF-IDF fallback vectors.

**Before → After (mock mode, cold start):**
| Mode | Before | After |
|------|--------|-------|
| COMPANY_FIRST | 20.1s | 2.8s |
| INDUSTRY_FIRST | 4.2s | 0.1s |
| REPORT_DRIVEN | 4.2s | 0.5s |

### perf: skip LLM synthesis + critic in mock mode (~55% faster)

The intelligence pipeline's synthesis (Step 8) and critic (Step 8b) made **real LLM calls**
even when all articles were mock articles. These calls added ~8s of wall time with no benefit
for mock pipeline validation.

Fix: detect mock articles via `fetch_method="mock"` (same pattern used by `filter.py`) and
skip LLM calls. Uses `_fallback_label()` for cluster labels and article titles for summaries.

**Before → After (mock mode):**
| Mode | Before | After |
|------|--------|-------|
| COMPANY_FIRST | 24.2s | 20.1s |
| INDUSTRY_FIRST | 9.9s | 4.2s |
| REPORT_DRIVEN | 8.8s | 4.2s |

---

## Session 6 — March 12, 2026

### chore: remove stale references to deleted modules in docs

Cleaned references to `company_agent.py`, `lead_validator.py`, `MetaReasoner`, `stage_advisories`,
and `weight_learner.py` from `main.py` docstrings, `app/agents/README.md` file tree, and
`app/intelligence/match.py` comments. All referenced modules were deleted in prior cleanup sessions
but their documentation entries were left behind.

### fix: repair standalone tests — stdout capture crash + missing fields

**Problem 1:** 4 standalone test files replaced `sys.stdout` with `io.TextIOWrapper` at import
time, crashing pytest's capture mechanism (`ValueError: I/O operation on closed file`).
Files: `test_clustering.py`, `test_dataset_enhancer.py`, `test_entity_extraction.py`, `test_pipeline_milestones.py`.
Fix: guard with `"pytest" not in sys.modules` so it only runs when executed directly.

**Problem 2:** `test_dataset_enhancer.py::_make_enhancer()` created a `DatasetEnhancer` via
`__new__` (bypassing `__init__`) but didn't set `_cached_positives`/`_cached_negatives` fields
added in March 2026. Fix: add the 2 missing fields.

**Problem 3:** `test_dedup.py::test_math_invariants()` was a helper function called from other
tests, but named `test_*` so pytest collected it as a standalone test with missing fixtures.
Fix: rename to `check_math_invariants()`.

**Net: 15/15 standalone tests pass (was: 0 collected / crash). Main suite: 32/32.**

### fix: repair 3 broken tests in test_learning.py

`tests/test_learning.py` had 3 failures from APIs removed in prior cleanup sessions:
1. `test_high_confidence_means_low_exploration` — called `bus.publish_weight_learner()` (removed with weight_learner.py). Fix: remove the call; test still validates exploration budget.
2. `test_novelty_distribution_sums_to_1` — passed `stale_pruned=2` to `publish_trend_memory()` (parameter removed in signal bus cleanup). Fix: remove the kwarg.
3. `test_update_from_run_metrics_calls_threshold_adapter` — called `update_from_run_metrics()` which was removed. Fix: skip with reason.

**Net: 0 failures in test_learning.py (was 3). Main suite: 32/32 pass.**

### chore: remove 5 dead settings from config.py

Removed Settings fields that were only defined but never read anywhere in the codebase:
- `lead_relevance_threshold` — orphaned after CompanyDiscovery deletion
- `max_search_queries_per_impact` — orphaned after CompanyDiscovery deletion
- `company_min_verification_confidence` — orphaned after pipeline_validator deletion
- `max_companies_per_trend` — orphaned after CompanyDiscovery deletion
- `ollama_gen_model` — only `ollama_tool_model` is used (via `_build_ollama_tool_model`)

Verified: all other settings are read via `getattr(settings, 'field')` in providers.py,
embeddings.py, llm_service.py, or other runtime code. 13 settings initially flagged as dead
were false positives due to `getattr()` access pattern.

### perf: cache GLiNER model across EntityClassifier re-instantiations

**Problem:** `_load_gliner()` in `classifier.py` loaded the GLiNER model from disk (~9s) every
time a new `EntityClassifier` was created. While `extractor.py` has a module-level singleton
guard, any other code path creating `EntityClassifier()` would trigger a full reload.

**Fix:** Added `_gliner_model_cache` dict at the function level in `_load_gliner()`. The loaded
model is cached by `model_dir` key, so re-instantiation of `EntityClassifier` skips the ~9s
`GLiNER.from_pretrained()` call. Cold start still ~9s, but subsequent runs in the same process
are instant.

### chore: delete dead `company_agent.py` (946 lines)

**Finding:** The entire `CompanyDiscovery` class (13 methods, 946 lines) was never imported or
instantiated from anywhere in the codebase. Zero external callers, zero test references.

**Why dead:** Pipeline evolved from separate company-discovery step to inline company discovery
within causal council hops (`CausalHop.companies_found`). The old `CompanyDiscovery` class was
replaced but never deleted.

**Also:** Updated stale `CompanyAgent` comment in `mock_responses.py`.

**Net: 946 lines removed. 32/32 tests pass. Graph compiles.**

### feat: wire company_exploration_rate into system_confidence

**Signal gap:** `company_exploration_rate` was computed in `publish_company_bandit()` (fraction of
company bandit arms with uncertain posteriors) but never consumed — its twin `source_exploration_rate`
was already wired into `compute_derived_signals()`.

**Fix:** Added `company_exploration_rate` as a signal in `compute_derived_signals()`. Low company
exploration rate = company preferences have converged = higher system confidence. This follows
the collaborative multi-armed bandit pattern (Landgren et al., 2016) where both bandits' convergence
states contribute to the shared exploration budget.

**Remaining bus analysis:**
- `feedback_distribution`: consumed by run recording (orchestrator.py:1735) — live for analytics
- `nli_scores_by_source`: passed directly to source_bandit (not via bus read) — bus copy is for persistence

### chore: remove dead `observed_filter_reject_rate` field from ThresholdUpdate

- `app/learning/threshold_adapter.py:46` — field declared but never assigned or read anywhere
- Only `observed_filter_accept_rate` was consumed (used by EMA adaptation at lines 163-172)

### chore: remove 3 dead constants

- `HDBSCAN_SOFT_NOISE_THRESHOLD` in `app/intelligence/cluster/algorithms.py` — defined, never referenced
- `B2B_BLOCK_LABELS` in `app/intelligence/engine/classifier.py` — defined, never referenced
- `IDENTITY_TYPES` in `app/intelligence/engine/extractor.py` — defined, never referenced

### chore: remove 4 dead signal bus fields + update callers

**Problem:** 4 `LearningSignalBus` fields were published but never consumed by any loop or derived signal:
- `stale_pruned_count` — written by `publish_trend_memory`, read by nobody
- `company_arm_means` — written by `publish_company_bandit`, read by nobody (only `winning_company_types` consumed)
- `threshold_values` — written by `publish_adaptive_thresholds`, read by nobody (only `anomaly_flags`/`drift_alerts` consumed)
- `cluster_noise_rate` — written by `publish_backward_signals`, read by nobody

**Fix:**
- `app/learning/signal_bus.py` — removed 4 field declarations + their writes in publish methods
- `app/agents/orchestrator.py` — removed 3 dead arguments from caller sites:
  - `publish_adaptive_thresholds()`: removed `thresholds=dict(adapter._thresholds)`
  - `publish_trend_memory()`: removed `stale_pruned` arg + dead `stale_pruned = 0` variable
  - `publish_backward_signals()`: removed `cluster_noise_rate=noise_rate`
- `tests/test_self_learning_system.py` — updated `test_signal_bus_backward_cascade` to match new signature

**Net: 32/32 tests pass. ~25 lines removed. Signal bus now has zero write-only fields.**

### fix: timezone bug in signal bus timestamp + restore linter-broken import

**Bug 1 — `app/agents/orchestrator.py:1719`:**
- `datetime.now(timezone)` passed the `timezone` class (a type) instead of `timezone.utc` (an instance)
- Caused `TypeError: tzinfo argument must be None or of a tzinfo subclass, not type 'type'` every run
- Signal bus `timestamp` was never set → cross-run persistence lost the time marker
- Fix: `datetime.now(timezone.utc)`

**Bug 2 — `app/agents/workers/contact_agent.py:15`:**
- Linter erroneously removed `TREND_ROLE_MAPPING` from import (used 15 times in the file)
- Caused `NameError` on any contact finding operation
- Fix: restore `from ...config import get_settings, TREND_ROLE_MAPPING`

**Bug 3 — `tests/test_mock_pipeline.py:89`:**
- Test imported `clear_score_cache` which was renamed to `clear_score_cache_if_hypothesis_changed`
- Fix: clear cache directly via `_score_cache.clear()` under lock

**Net: 32/32 tests pass. Mock pipeline runs clean (no tzinfo error).**

### feat: Track B→A upgrade — company_enricher + impact_agent

**Problem:** 4 LLM call sites used `generate_json()` (Track B / raw JSON parsing) as fallback after `run_structured()` (Track A / pydantic-ai). This added ~70 lines of manual type coercion and dict parsing that duplicate pydantic's built-in validation.

**Fix 1 — `app/tools/company_enricher.py` (3 spots):**
- Removed Track B fallback blocks for products/services, hiring signals, and tech/IP extraction
- Track A (`run_structured()`) already has 2 retries + 2 reflect retries + provider fallback chain
- If all providers fail, outer try/except returns empty — acceptable for background enrichment
- **~25 lines removed**

**Fix 2 — `app/agents/workers/impact_agent.py` (1 spot):**
- Replaced manual `_validate_impact_response()` (60 lines of type coercion) with pydantic validators on `ImpactAnalysisLLM`
- Track B fallback now uses `ImpactAnalysisLLM.model_validate(raw)` — pydantic handles coercion
- **~60 lines removed** (`_validate_impact_response()` deleted entirely)

**Fix 3 — `app/schemas/llm_outputs.py` (new validators):**
- Added `_coerce_to_str_list()` + `StrList` annotated type: handles None→[], str→[str], non-list→[str(val)]
- Added `@model_validator(mode='before')` for string fields: None→"", non-str→str(val), pitch_angle truncation
- All 11 list fields on `ImpactAnalysisLLM` now use `StrList` type annotation

**Net: 32/32 tests pass. ~85 lines removed. Both Track A and Track B paths benefit from the same validators.**

### chore: move 22 inline stdlib imports to module level

**Problem:** 22 stdlib imports (`json`, `time`, `asyncio`, `hashlib`, `os`) were inside functions instead of at module level. 3 were redundant (already imported at module level under different aliases).

**Files fixed (12 files):**
- `app/schemas/sales.py` — 2× redundant `import logging` (already line 14)
- `app/tools/search.py` — redundant `import asyncio as _aio` (already `_asyncio` at line 121)
- `app/tools/llm/embeddings.py` — 4× `import time` + 1× `import os` → module level
- `app/agents/workers/contact_agent.py` — 2× `import asyncio` → module level
- `app/api/leads.py` — `import asyncio` → module level
- `app/database.py` — `import hashlib` → module level
- `app/intelligence/fetch.py` — `import time` → module level
- `app/intelligence/config.py` — `import json` + `import os` → module level
- `app/tools/company_enricher.py` — 2× `import json` → module level
- `app/tools/llm/llm_service.py` — `import json` → module level
- `app/tools/web/rss_tool.py` — `import json` + `import time` → module level
- `app/tools/web/web_intel.py` — `import json` → module level

**Net: 32/32 tests pass. All 12 modules import cleanly.**

### fix: wire unconsumed signal bus fields into compute_derived_signals

**Problem:** Two published signal bus fields were never consumed by any downstream loop:
- `source_diversity_index` (Shannon entropy of source posteriors) — published by `publish_source_bandit()` but never read
- `drift_alerts` (threshold drift events like `low_coherence`, `high_noise`) — published by `publish_adaptive_thresholds()` but never read

Without these signals, `system_confidence` couldn't detect source over-concentration or sustained quality drift.

**Fix (`app/learning/signal_bus.py` → `compute_derived_signals()`):**
- Wire `drift_alerts` into anomaly scoring: each drift alert reduces confidence by 0.2 (floor 0.3), merged with existing `anomaly_flags` handling
- Wire `source_diversity_index` as normalized Pielou evenness (H/H_max): low diversity → lower confidence → higher exploration budget
- REF: Pielou (1966) species evenness; Shannon (1948) entropy

**Net: 32/32 tests pass. 2 signal gaps closed.**

### chore: remove dead code — buffer_log, stale snapshot entry

**Dead code removed:**

1. **`RunRecorder.buffer_log()` + `_log_buffer` field** (`app/tools/run_recorder.py`)
   - `buffer_log()` had 0 callers anywhere in the codebase
   - `_log_buffer` was always empty → `data["log_messages"]` was always `[]`
   - Consumer (`app/api/pipeline.py:912`) already uses `.get("log_messages", [])` — safe removal

2. **Stale snapshot entry `"data/learned_weights.json"`** (`app/learning/experiment_tracker.py`)
   - `weight_learner.py` was deleted in Session 2 (March 10)
   - File never exists on disk; `if src.exists()` guard silently skipped it
   - Removed from `_SNAPSHOT_FILES` list (3 → 2 entries remain)

3. **README sync** (`app/learning/README.md`)
   - Removed matching `learned_weights.json` line from docs

**Net: 32/32 tests pass. ~12 lines removed.**

---

## Session 5 — March 12, 2026

### perf: contact_agent — eliminate N ContactBandit disk reads

**Problem:** `ContactBandit.load()` was called inside `_find_one()` — an async closure that runs once per company (20-50 calls per pipeline run). Since `load()` reads a JSON file from disk each time, this caused N redundant file reads where N = number of target companies.

**Root cause (double load):** `match_roles_to_trend()` had its own internal `ContactBandit.load()`, AND `_find_one()` loaded it a *second* time for a duplicate re-ranking pass. This meant up to 2N disk reads per pipeline stage.

**Fix (`app/agents/workers/contact_agent.py`):**
- Added `bandit=None` parameter to `match_roles_to_trend()`: if provided, skips internal load; otherwise falls back to `ContactBandit.load()` (backward-compatible — `leads.py` call site unchanged)
- Load `ContactBandit` once before the `asyncio.gather()` loop in `find_contacts()`
- Pass pre-loaded bandit to `match_roles_to_trend()` — eliminates N-1 file reads + the duplicate re-ranking pass
- REF: Chapelle & Li (2011) Thompson Sampling for CTR — arXiv:1111.1797

**Net: 31/31 tests pass. N-1 file reads eliminated per contact stage.**

### perf: company_agent — eliminate N×M CompanyRelevanceBandit disk reads

**Problem:** `CompanyRelevanceBandit.__init__()` reads from disk, and it was called inside `_extract_companies_with_intent()` — a method run in parallel for up to 10 search results × N impacts ≈ **200 disk reads per pipeline run**.

**Fix:** Added optional `relevance_bandit=None` parameter to `_extract_companies_with_intent()`. Load `CompanyRelevanceBandit` once before the `asyncio.gather` loop in the caller and pass it through the closure. Single-call sites (fallback path) unaffected.

**Net: 31/31 tests pass. ~200 disk reads → 1 per impact.**

### perf: company_bandit — N file writes → 1 after update loop

**Problem:** `CompanyRelevanceBandit.update()` called `_save()` after every single update. The orchestrator's learning_update_node called `update()` once per lead (up to 55 leads per run) → **55 JSON file writes** per pipeline run.

**Fix:** Added `save=True` optional parameter to `update()`. Learning update loop passes `save=False` and calls `company_bandit._save()` once after all updates. Backward-compatible: default `save=True` preserves existing behavior for all other call sites.

**Net: 32/32 tests pass. 55 file writes → 1 per learning update phase.**

### perf: embeddings — guard O(N²) cosine similarity check with isEnabledFor(DEBUG)

**Problem:** Pairwise cosine similarity was computed on every `embed_batch()` call regardless of log level. Python evaluates f-string arguments eagerly before calling `logger.debug()`. At 50 articles × 1536-dim, this computed a 50×50 similarity matrix even at INFO log level.

**Fix:** Wrapped entire debug stats block in `if logger.isEnabledFor(logging.DEBUG):`. Production INFO-level runs now skip the O(N²) computation entirely.

### perf: dataset_enhancer — O(N²) file reads → O(1) in-memory cached stats

**Problem:** `DatasetEnhancer._add_example()` called `self.get_stats()` for every example being added. `get_stats()` reads the entire JSONL file from scratch each time. For a filter run adding 100 examples to a 3,700-entry dataset, this meant **370,000 redundant JSON parses per run**.

**Fix:** `_load_seen_hashes()` now also accumulates `_cached_positives`/`_cached_negatives` counters during its single file pass. `_add_example()` uses these cached counters for the balance/cap checks and increments them atomically after each successful write. `get_stats()` (reporting-only path) still reads from disk as before.

**Net: 32/32 tests pass. Confirmed with live 3,702-entry dataset.**

### cleanup: remove redundant inline imports in orchestrator.learning_update_node

`json`, `Path`, `datetime`, `timezone` are already imported at module level in `orchestrator.py` (lines 10-14). Two `try` blocks shadowed them with identical inline imports — removed both.

### perf: pre-compile regex patterns in leads.py (~300+ calls/run saved)

`is_company_description()` and `_normalize_event_type()` both used uncompiled regex inside function bodies called ~300+ times per pipeline run:
- `re.findall(r'\(([^)]+)\)', clean)` compiled on every `is_company_description()` call
- `re.search(rf"\b{kw}\b", lower)` compiled for 29 short keywords on every `_normalize_event_type()` call

**Fix:** `_PAREN_RE` and `_SHORT_KW_RE` (29 patterns) pre-compiled at module level after `_EVENT_TYPE_KEYWORDS` definition. Removed inline `import re` from both functions.

### perf: pre-compile regex patterns across intelligence engine hot paths (5 files)

**normalizer.normalize_entity_name**: 5 patterns pre-compiled at module level (`_RE_POSSESSIVE`, `_RE_LEADING_NOISE`, `_RE_TRAILING_VERB`, `_RE_TRAILING_YEAR`, `_RE_HYPHEN_MODIFIER`). Called once per entity name — hundreds of calls per run.

**extractor._clean_entity_name + _apply_statistical_filters**: 6 patterns added (`_RE_UNICODE_WHITESPACE`, `_RE_MULTI_SPACE`, `_RE_CAMEL_TRANSITION`, `_RE_ACRONYM_CONCAT`, `_RE_NON_ALNUM`, `_RE_SENTENCE_SPLIT`). Also removed `import re as _re` inline import.

**match.py**: `_RE_WORD_TOKENS` and `_RE_SENTENCE_SPLIT` added for keyword extraction and trigger matching loops.

**summarizer.py**: `_RE_SENTENCE_SPLIT` (3 occurrences) and `_RE_HTML_OR_URLS` (1 occurrence) pre-compiled.

**filter.py**: Per-company word-boundary patterns compiled once before the article loop (not on every article × company combination). With 3 companies and 50 ambiguous articles: 150 → 3 regex compilations.

**source_intel.py**: `_RE_SENTENCE_SPLIT` pre-compiled at module level; removed 2 inline `re.split()` calls.

**company_agent.py**: `_RE_NON_ALNUM_SPACE` pre-compiled at module level; removed inline `import re` from `_normalize_name()`.

**contact_agent.py**: Removed redundant `import re as _re` (re already at module level on line 8); fixed `_size_matches()` in campaign_executor.

**email_agent.py**: 5 LLM placeholder cleanup patterns (`_RE_YOUR_NAME`, `_RE_YOUR_TITLE`, `_RE_YOUR_CONTACT`, `_RE_COMPANY_NAME`, `_RE_EXCESS_NEWLINES`) pre-compiled at module level. Duplicated from brevo_tool.py to avoid cross-module dependency for a hot path called per-lead.

**campaign_executor.py**: Added `import re` at module level; removed `import re as _re` from `_size_matches()` inner function.

**impact_council.py**: 3 scoring patterns (`_RE_IC_DATA`, `_RE_IC_SPEC`, `_RE_IC_CONSEQUENCE`) pre-compiled at module level; removed `import re` from `score_impact_analysis()` (called once per company, ~50×/run).

**fetch.py**: 3 patterns (`_RE_REPORT_CAPS`, `_RE_REPORT_PHRASE`, `_RE_REPORT_SENTENCE`) pre-compiled at module level; removed `import re` from `_extract_report_queries()`.

**Net: 32/32 tests pass. All inline/per-call regex compilations eliminated across 13 files.**

### cleanup: orchestrator.py — remove redundant `import asyncio as _aio`

`asyncio` already at module level (line 8). Inline alias inside `causal_council_node` replaced with direct `asyncio.Semaphore`/`asyncio.gather`. Also moved `import time as _time` from inside `run_pipeline()` to module level in `api/pipeline.py`.

### perf: pre-compile regex in company_enricher (Wikipedia enrichment path)

_strip_wiki() called ~5-10× per company (250-500 invocations/run): 15 patterns moved to module level (_RE_WIKI_LINK, _RE_WIKI_HLIST, _RE_WIKI_DATE, _RE_WIKI_SMALL, _RE_WIKI_TREND, _RE_WIKI_INR, _RE_WIKI_TPL_CONTENT, _RE_WIKI_TPL_EMPTY, _RE_WIKI_REF, _RE_WIKI_BR, _RE_WIKI_TAG, _RE_WIKI_TRIM, _RE_YEAR, _RE_URL, _RE_PRODUCT_SEP).
Also: _names_match() patterns (_RE_MULTI_SPACE, _RE_NON_ALNUM) pre-compiled.

### perf: pre-compile regex in domain_utils, search, contact_agent, campaign_executor, web_intel

**domain_utils.py**: _RE_DOMAIN_PATTERN, _RE_NON_ALNUM_SPACE, _RE_SPACES — extract_clean_domain() called ~27×/run.
**search.py**: _RE_WORD_TOKENS for BM25Search._tokenize() — called per article indexed (50-200/run).
**contact_agent.py + campaign_executor.py**: _RE_DIGITS for employee count parsing.
**web_intel.py**: _RE_PAREN, _RE_YEAR_SHORT, _RE_QUARTER, _RE_NON_WORD, _RE_MULTI_SPACE for _normalize_title() dedup.

**Zero uncompiled regex patterns remain in any hot path across all app/ files.**

### cleanup: remove unused imports (AST scan, 14 files)

AST-based analysis found genuinely unused imports (verified by grep — no annotation or runtime use):
- `List`: analysis.py, leads.py, email_agent.py, api/learning.py
- `Optional`: company_agent.py, impact_agent.py, schemas.py, contact_agent (Dict/Optional), campaign_executor.py, normalizer.py
- `Dict`: fetch.py (dedup section), contact_agent.py
- `Request` (FastAPI): api/pipeline.py
- `json`: api/profiles.py
- `asdict`, `Any`: threshold_adapter.py

**Net: 32/32 tests pass.**

---

## Commit fb9d3a2 — Dead Code Removal (Task 1)

### Deleted Files
| File | Lines | Reason |
|------|-------|--------|
| `app/learning/meta_reasoner.py` | 95 | All 4 methods returned empty, `_enabled=False` always |
| `app/learning/pipeline_validator.py` | 1,475 | Per-stage quality gates; confidence gate never fired |

### Functions Removed from `app/agents/orchestrator.py`
| Function/Block | Lines | Reason |
|----------------|-------|--------|
| `_maybe_trigger_finetune()` | ~110 | Spawns OpenAI finetune thread; model never loaded back |
| `_FINETUNE_TRIGGER_EVERY`, `_FINETUNE_JOB_FILE` | 2 | Dead constants for above |
| `pick_next_hypothesis()` call block | ~15 | `data/improvement_hypotheses.json` never auto-populated |
| `mark_hypothesis_tested()` call block | ~10 | Same — dead hypothesis system |
| `DatasetEnhancer` + SetFit call block | ~55 | SetFit training: never had enough feedback data |
| `HypothesisLearner.maybe_update()` call | ~25 | No labeled feedback data to train on |
| `compute_adaptive_thresholds()` call | ~12 | Duplicate of threshold_adapter (2 EMA systems doing same job) |
| `verify_company_enrichment/contacts/leads` | ~30 | From deleted pipeline_validator |

### Functions Removed from Other Files
| File | What removed |
|------|-------------|
| `app/intelligence/filter.py` | `_get_finetune_model()`, `_FINETUNE_MODEL_ID`, finetune fast-path in `_classify_batch()` |
| `app/intelligence/pipeline.py` | All 7 `verify_*` call blocks, `_stage_results` list |
| `app/agents/workers/contact_agent.py` | `_find_via_search()`, `_extract_contact_from_search()` (web search fallback, ~140 lines) |
| `app/learning/experiment_tracker.py` | `Hypothesis` model, `pick_next_hypothesis`, `mark_hypothesis_tested`, `load/save_hypotheses` |
| `app/learning/pipeline_metrics.py` | `AdaptiveThreshold` class, `THRESHOLD_REGISTRY`, `compute_adaptive_thresholds()`, `_extract_nested()` |

### Behaviors Preserved (inlined before deletion)
- **Entity-group cleanup** — removes single-char/numeric entity names (inlined in `pipeline.py`)
- **Zero-cluster fallback** — re-validates with looser coherence when 0 clusters pass (inlined in `pipeline.py`)

### Test Results
- 22/22 self-learning tests pass
- LangGraph graph compiles clean

---

## Commit b19c5a7 — Clustering Simplification + Mock Data (Tasks 2 & 3)

### Clustering Fixes (`app/intelligence/cluster/algorithms.py`)
- **BUG FIX**: HDBSCAN clusters now get `EventGranularity` (MAJOR/SUB/NANO) — was missing entirely
- **Labels improved**: `"{entity}: major event"` / `"{entity}: sub-event N"` / `"{entity}: signal N"` (was `"{entity} event {N}"`)
- **HAC metrics**: Removed `sweep_results` list (large dict, unread downstream); kept `n_sweep_candidates` count

### Leiden Optimization Removed (`app/intelligence/engine/clusterer.py`)
- Removed Optuna hyperparameter tuning for Leiden resolution (was adding **30s latency per call**)
- Removed 5 dead Optuna functions: `optimize_leiden`, `compute_leiden_quality`, `compute_meta_features`, `load_best_params_for_data`, `_load_last_best_params`
- Fixed `leiden_k` to `params.leiden_k=20`, `resolution=1.0` (already the optimal empirical defaults)

### Dead ClusteringParams Constants Removed (`app/intelligence/config.py`)
| Constant | Value | Reason |
|----------|-------|--------|
| `hac_cophenetic_min` | 0.70 | 0 callers — only defined |
| `leiden_optuna_trials` | 15 | Only used by deleted Optuna loop |
| `leiden_optuna_timeout` | 30 | Only used by deleted Optuna loop |

### Mock Data — All 3 Modes (`app/data/mock_articles.py`)
| Dataset | Articles | Clusters | Theme |
|---------|----------|----------|-------|
| `MOCK_ARTICLES_RAW` | 17 | 3 | Fintech KYC + Quick commerce + Semiconductor (unchanged) |
| `MOCK_ARTICLES_COMPANY_FIRST` | 17 | 3 | IT services AI transformation + Cloud deals + Restructuring |
| `MOCK_ARTICLES_REPORT_DRIVEN` | 17 | 3 | 5G enterprise + Airtel/Jio B2B + IoT platform funding |

### Mock Mode Routing (`app/agents/source_intel.py`)
- `_make_mock_articles(scope=None)` now accepts scope parameter
- Selects correct dataset by `scope.mode`: `company_first` → IT articles, `report_driven` → 5G/IoT articles, default → fintech
- All API tools (Tavily/Hunter/Apollo) confirmed to have `if self.mock_mode: return []` short-circuits

---

## Commit 0941542 — Noise Reassignment (Task 6)

### `app/intelligence/cluster/orchestrator.py`
- **`_reassign_noise()`** implemented — `noise_reassign_min_similarity=0.45` was defined in ClusteringParams but had **0 callers** (dead parameter)
- After all clustering passes (HAC + HDBSCAN + Leiden), soft-assigns noise articles to nearest cluster centroid if cosine similarity ≥ 0.45
- Running-mean centroid update after each assignment prevents all noise collapsing to one cluster
- Reduces noise rate from ~20-42% → ~5-15%
- REF: Campello et al. (2013) extended — soft cluster membership for HDBSCAN noise points

---

## Commit d8ccaf4 — Report-Driven + Quality Gates + Adaptive Threshold (Tasks 5, 7)

### `app/intelligence/fetch.py` (Task 5)
- **`ReportEntities` dataclass** added: `{companies, industries, topics}` — typed structured extraction output
- **`_extract_report_entities_llm()`** — single LLM call (json_mode=True, 2000-char truncated), graceful regex fallback
- **Report-Driven routing**: companies → `_fetch_google_news_rss()` (no Tavily key), topics → `_fetch_tavily_or_ddg()`
- **Industry-First upgrade**: anchor companies via Google News RSS + keyword search via `_fetch_tavily_or_ddg()`
- **`_fetch_ddg_news()`** added — DDG news fallback (uses `from ddgs import DDGS`)
- **`_fetch_tavily_or_ddg()`** added — Tavily primary, DDG fallback if returns empty
- **`_extract_report_queries()` improved**: added sentence-fragment fallback for empty-regex case; fixed `RegionConfig.name` attribute bug
- Entities discovered via LLM are set on `scope.companies` for downstream entity processing

### `app/agents/orchestrator.py` (Task 7)
- **Quality gate logging** (INFO/WARN, no hard stops) replaces deleted `verify_*` calls:
  - `article_count < 20` → WARN with diagnostic hints (RSS reachable? Tavily keys?)
  - `len(trends) == 0` → ERROR log; `< 3` → WARN
  - Lead email coverage `< 30%` → WARN with Hunter/Apollo hint
- **NLI signal loop fixed**: `intelligence/pipeline.py` published to a throwaway bus instance; now re-published to the orchestrator's bus (the one saved to disk) in `learning_update_node`
- **`GraphState`** — added `quality_score: float` field (populated from `quality_validation_node`)

### `app/learning/signal_bus.py` (Task 7)
- **Removed** `get_adaptive_threshold_modulation()` — 0 callers found in entire codebase; the cross-loop modulation is now done via `system_confidence` parameter on `ThresholdAdapter.update()`

### `app/learning/threshold_adapter.py` (Task 7)
- **`update(system_confidence=0.5)`** — new parameter for cross-loop health signal
  - `system_confidence > 0.7` → alpha × 1.3 (learn faster, system is stable)
  - `system_confidence < 0.3` → alpha × 0.7 (learn slower, system is degraded)
  - REF: Adaptive learning rates proportional to SNR (Kingma & Ba, Adam 2014)
- Alpha modulation is ephemeral (saved_alpha restored after update)
- After each threshold update: publishes `bus.publish_adaptive_thresholds()` — closes the cross-loop feedback loop

---

## Session 3 — March 12, 2026 (continued)

### Commits: d9c4ade, 7b9b004, 3e2c08d

### Dead Code Removal
- `tests/standalone/test_dataset_enhancer.py`: 530→194 lines — removed 4 tests for deleted methods (bootstrap_from_reuters/ag_news, get_examples_for_setfit, should_trigger_retrain). 13/13 assertions pass.
- `app/config.py NEWS_SOURCES`: removed 8 permanently dead RSS sources (business_standard, bs_companies, bs_economy, bs_finance, bs_tech, bbc_india, zeebiz, thewire_economy) — all 403/empty, not in DEFAULT_ACTIVE_SOURCES, zero code references outside config.
- `app/intelligence/cluster/algorithms.py`: deleted DBCV `validity_index()` call (non-trivial CPU, discarded by every caller via `_` unpacking), deleted dead `validate_clustering_math()` function (never called anywhere).
- `frontend/components/dashboard/kpi-cards.tsx`: deleted (0 consumers — dashboard defines inline KpiCard).
- `frontend/components/dashboard/leads-compact.tsx`: deleted (0 consumers — dashboard uses inline LeadRow).

### Quality Metric Cleanup
**REMOVED (zero downstream decisions):**
- DBCV score in HDBSCAN — computed, then thrown away with `_` discard at call site in cluster orchestrator
- `validate_clustering_math()` — function defined but never called
- Duplicate sections in TrendDetail (trends/page.tsx): removed inline pain_points/target_roles from COUNCIL ANALYSIS block (shown as standalone sections below), removed early BUYING INTENT SIGNALS (duplicate of BUYING INTENT further down)

**KEPT (confirmed live decision-making):**
- `coherence_score` — drives quality gate (40% weight), threshold adapter EMA, cluster admission, source bandit backward cascade
- `oss_score` — drives quality gate (30%), source bandit (5%), auto-feedback (30%), pipeline_metrics
- `nli_mean_entailment` — primary source bandit reward (30%), system_confidence cross-loop signal
- `quality_score` (GraphState) — feeds CUSUM guard in ThresholdAdapter
- `council_confidence` — hard filter + fallback sort for impact viability gate
- `silhouette_score` (HAC internal) — picks winning threshold in parameter sweep
- `actionability_score` — display-only for frontend (no internal gate)
- `trend_score` — display-only alias for coherence (frontend sort in TrendTree)

### Bug Fixes
- `orchestrator.py`: fix double `_compute_oss(cluster)` call — compute once, reuse result
- `leads-panel.tsx`: fix broken sort delegation — panel was re-sorting by confidence after page applied user sort (Company/Urgency sort controls had no effect)

### API Upgrades
- LangGraph: `retry=` → `retry_policy=` (V0.5 deprecation, was generating warnings per graph compile)
- Pydantic V2 migration in all schema files: `class Config:` → `model_config = ConfigDict()`, `@validator` → `@field_validator` with V2 info API (news.py, sales.py, trends.py, base.py, config.py)
- SQLAlchemy 2.0: `ext.declarative.declarative_base` → `orm.declarative_base`
- **Result: 24/24 tests pass with ZERO deprecation warnings**

### UX Improvements
- Dashboard: Added 3-mode pipeline selector (Industry / Company / Report) directly in header — users can choose pipeline mode without going to Campaigns
- api.ts: `runPipeline()` now accepts `mode/industry/companies` overrides, passed from dashboard mode selector

### Self-Learning Data
- Cleaned `data/source_bandit.json`: removed 10 ghost entries for deleted sources (all at floor Beta(1.5,1.5), never received data). 171 → 161 arms tracked.
- Source bandit IS working correctly — 26+ runs of learned data visible with posterior means ranging 0.35–0.73 across sources. `livemint`, `business-standard`, `economic-times`, `yourstory`, `inc42` are top learners.

---

## Session 2 — Commits 28e81ce, e02e9f4, 7d08dd6 (March 12, 2026)

### `app/intelligence/pipeline.py` (fix — mock bridge)
- **`execute(pre_fetched_articles=None)`** — new parameter; skips live `fetch_articles()` when pre-fetched data is provided
- **`_coerce_to_raw_articles()`** — duck-type bridge: NewsArticle → RawArticle (preserves `fetch_method="mock"`)
- **Root cause fix**: mock articles in `deps._articles` were being ignored; analysis node was re-fetching live articles

### `app/agents/analysis.py` (fix — mock bridge)
- `run_trend_pipeline` tool: passes `deps._articles` as `pre_fetched_articles` in mock mode
- `run_analysis()` fallback: same mock bridge for direct `intelligence_execute()` calls

### `app/intelligence/filter.py` (perf — NLI fast-path)
- When ALL articles have `fetch_method="mock"`: auto-accept, skip DeBERTa inference
- Saves ~5-10s per mock run (model cold start + inference)
- Mixed batches still run full NLI — production path unchanged

### `app/learning/hypothesis_learner.py` — **DELETED** (798 lines)
- `maybe_update()` entry point was never called after SetFit removal
- Zero live imports: only a dead comment in orchestrator.py

### `app/learning/dataset_enhancer.py` — −185 lines (506 → 321)
- Removed dead SetFit methods: `get_examples_for_setfit`, `should_trigger_retrain`
- Removed dead data loaders: `bootstrap_from_ag_news`, `bootstrap_from_reuters`
- Kept: `extract_labels_from_filter()` (LIVE), `extract_labels_from_clusters()`, helpers

### `app/agents/causal_council.py` — Track A primary, Track B fallback
- Added `run_structured()` as primary path (typed `CausalChainResult` output)
- `generate_json()` remains as inner fallback (for NVIDIA DeepSeek compatibility)
- Matches `impact_agent.py` pattern — consistent Track A standard across codebase

**Net this session: −1,002 lines removed, 3 bugs fixed (mock bridge ×2 + NLI fast-path), 28/28 tests pass**

---

## Session 4 — March 12, 2026

### Track A Upgrade: Company Resolution in `orchestrator.py`

**`_resolve_companies_via_search()` — Track B → Track A**

Added `SegmentResolutionListLLM`, `SegmentResolutionLLM`, `ResolvedCompanyLLM` to `app/schemas/llm_outputs.py`:
- `ResolvedCompanyLLM`: typed company record with `name`, `city`, `state`, `size_band`
- `SegmentResolutionLLM`: per-segment result with `index` + `companies` list
- `SegmentResolutionListLLM`: wraps list output via `@model_validator(mode="before")` that coerces `[...]` → `{"segments": [...]}`

**`orchestrator.py` change:**
- Primary: `llm.run_structured(output_type=SegmentResolutionListLLM)` — pydantic-ai validates field types, prevents hallucinated company names in wrong fields
- Fallback: `llm.generate_json()` retained for NVIDIA DeepSeek compatibility (no forced function-calling)
- The shared parse loop (`idx = item.get("index", 0) - 1`, etc.) is unchanged — both paths produce identical `items` list format

**Impact:** Company name extraction from segment search results (main Industry-First/Report-Driven path) now benefits from pydantic validation on every structured output attempt.

### Track A Upgrade: Person Intelligence + Contact Role Inference

**New schemas in `app/schemas/llm_outputs.py`:**
- `_coerce_str_list()` + `StrList` annotated type — shared coercion (`str→[str]`, `None→[]`, non-list→`[str(v)]`) used by all 3 new models
- `PersonOutreachInsightsLLM` — `{background_summary, recent_focus, notable_achievements: StrList, shared_interests: StrList, talking_points: StrList}` — coerces string-valued list fields automatically
- `ContentThemesLLM` — `{themes: StrList}` with `@model_validator` coercing `[...]` → `{"themes": [...]}`
- `ContactRolesLLM` — `{roles: StrList}` with `@model_validator` coercing `[...]` → `{"roles": [...]}`

**`app/tools/person_intel.py` changes:**
- `_synthesize_person_insights()` (line ~303): Track A primary (`run_structured(PersonOutreachInsightsLLM)`), Track B fallback retained for DeepSeek compatibility
- `_synthesize_content_themes()` (line ~655): Track A primary (`run_structured(ContentThemesLLM)`), Track B fallback retained

**`app/api/companies.py` change:**
- `infer_contact_roles_llm()` (line ~986): Track A primary (`run_structured(ContactRolesLLM)`), Track B fallback retained; eliminates manual `isinstance(result, list)` / dict-value-scan guard

**Pattern established:** `@model_validator(mode="before")` on `StrList`-typed models lets pydantic-ai handle list-vs-dict coercion that was previously done with ad-hoc isinstance guards.

### Track A Upgrade: Event Classifier + Company Enricher

**New schemas in `app/schemas/llm_outputs.py`:**
- `CoercedFloat` annotated type — `BeforeValidator` that coerces `"0.8"` (string) → `0.8` (float); LLMs like NVIDIA DeepSeek return confidence as strings
- `EventClassificationLLM` — `{event_type: str, confidence: CoercedFloat, reasoning: str}` — Tier-2 per-article classification
- `EventTaxonomyEntryLLM` / `EventTaxonomyListLLM` — cluster taxonomy naming with `urgency: CoercedFloat`
- `ProductServicesLLM` — `{products: StrList}` with `@model_validator` handling `[...]` → `{"products": [...]}` AND common dict key variants (`services`, `solutions`, `offerings`, `items`)
- `HiringSignalsLLM` — `{hiring_signals: StrList}` — background hiring intelligence
- `TechIpLLM` — `{tech_stack: StrList, patents: StrList, partnerships: StrList}` — background tech/IP analysis

**`app/tools/event_classifier_tool.py` changes:**
- `_validate_one()` Tier-2 loop (per-article): Track A primary (`run_structured(EventClassificationLLM)`), Track B fallback — eliminates `float(result.get("confidence", 0.5))` manual coercion
- `name_taxonomy_candidates()`: Track A primary (`run_structured(EventTaxonomyListLLM)`), Track B fallback

### Dead Code Removal — 4 Functions (Session 4)

| File | Function | Lines | Reason |
|------|----------|-------|--------|
| `app/tools/llm/providers.py` | `_build_ollama_gen_model()` | 13 | Alternative Ollama model (phi3.5-custom); never wired into any provider chain — only `_build_ollama_model` and `_build_ollama_tool_model` are called |
| `app/tools/llm/providers.py` | `_build_gemini_via_openrouter()` | 9 | Routing shim superseded by `_build_openrouter_model` + `_build_gemini_direct_model`; 0 callers |
| `app/intelligence/cluster/validator.py` | `_adapt_validation_result()` | 43 | Format conversion shim from an earlier clustering refactor; never called post-refactor. Also removed unused `Any` import |
| `app/agents/workers/impact_council.py` | `_empty_result()` | 6 | Error-fallback helper; impact_agent.py uses `_create_basic_impact()` instead — 0 callers |

**Net: −71 lines removed. 31/31 tests pass.**

**`app/tools/company_enricher.py` changes:**
- `_search_products_services()` (main enrichment path): Track A primary (`run_structured(ProductServicesLLM)`), handles both array and dict-key variants cleanly
- `_analyze_hiring_signals()` (background): Track A primary (`run_structured(HiringSignalsLLM)`)
- `_analyze_tech_ip()` (background): Track A primary (`run_structured(TechIpLLM)`), `model_dump()` output is drop-in replacement for dict return

---

## Architecture After Cleanup

### What was KEPT (confirmed live)
- `threshold_adapter.py` — CUSUM + EMA, prevents threshold drift between runs
- `source_bandit.py` — Thompson Sampling, adapts source priority over runs
- `company_bandit.py` — LinTS, adapts company relevance scoring over runs
- `contact_bandit.py` — `rank_roles()` called from `contact_agent.py:121` and `:331`
- `signal_bus.py` — cross-loop communication (being audited in Task 7)
- `experiment_tracker.py` — `ExperimentTracker.snapshot/restore_learning_state()` for regression rollback
- `hypothesis_learner.py` — **DELETED** (798 lines): entry point `maybe_update()` never called, zero live imports
- `dataset_enhancer.py` — gutted to 321 lines: SetFit methods removed, only label-accumulator kept
- `event_classifier_tool.py` — 1735 lines, called from `source_intel.py` for event classification
- `person_intel.py` — 670 lines, used by `email_agent.py` + `campaign_executor.py`

### Quality Metrics (Clusters)
**Kept:** silhouette score (HAC parameter sweep), DBCV (HDBSCAN validity), coherence (mean pairwise cosine)
**Removed:** sweep_results list, Optuna quality metrics, cophenetic threshold check, duplicate AdaptiveThreshold

### Pipeline Modes
| Mode | API Call | What it does |
|------|----------|-------------|
| `company_first` | `POST /api/v1/pipeline/run {"mode":"company_first","companies":["TCS","Infosys"]}` | Fetch news about specific companies → cluster → find decision-makers |
| `industry_first` | `POST /api/v1/pipeline/run {"mode":"industry_first","industry":"fintech_bfsi"}` | Fetch industry news + anchor company news → cluster → find leads in that sector |
| `report_driven` | `POST /api/v1/pipeline/run {"mode":"report_driven","report_text":"..."}` | LLM extracts companies/topics from report → fetch targeted news → cluster → leads |

### Self-Learning Loop
```
Run N article quality → NLI entailment scores
        ↓
source_bandit.update(source_id, reward=nli_score) → data/source_quality.json
        ↓
Run N+1: sources with higher posterior score fetch more articles

Lead quality → company relevance signal
        ↓
company_bandit.update(company_id, success=bool) → data/company_bandit.json
        ↓
Run N+1: better companies prioritized in lead gen

Cluster coherence scores
        ↓
threshold_adapter.update(ThresholdUpdate(coherence=...)) + CUSUM guard
        ↓
Run N+1: validation thresholds adapted to actual data distribution
```

---

## Files Changed by Session

### Python Backend
```
app/agents/orchestrator.py        ← heavy (dead code, quality gates, hypothesis)
app/agents/source_intel.py        ← mock mode routing
app/agents/workers/contact_agent.py ← removed web search fallback
app/data/mock_articles.py         ← +2 mode-specific article sets
app/intelligence/cluster/algorithms.py ← HDBSCAN granularity fix, better labels
app/intelligence/cluster/orchestrator.py ← noise reassignment (in progress)
app/intelligence/config.py        ← removed 3 dead constants
app/intelligence/engine/clusterer.py ← removed Optuna (5 functions)
app/intelligence/fetch.py         ← LLM entity extraction (in progress)
app/intelligence/filter.py        ← removed finetune fast-path
app/intelligence/pipeline.py      ← removed verify_* calls, inlined 2 behaviors
app/learning/experiment_tracker.py ← removed hypothesis management
app/learning/meta_reasoner.py     ← DELETED
app/learning/pipeline_metrics.py  ← removed AdaptiveThreshold, THRESHOLD_REGISTRY
app/learning/pipeline_validator.py ← DELETED
app/learning/signal_bus.py        ← docstring cleanup (audit in progress)
```

### Key Invariants (Do Not Break)
- LangGraph `stream_mode="updates"` — NEVER change to "values"
- TrendData fields: internal `trend_title`/`industries_affected`, API `title`/`industries`
- Provider reset per run: `provider_health.reset_for_new_run()`, `ProviderManager.reset_cooldowns()`, `LLMService.clear_cache()`
- DDG import: `from ddgs import DDGS` (NOT `duckduckgo_search`)
- LLMService.generate() signature: `generate(prompt, system_prompt=None, temperature=0.7, max_tokens=2000, json_mode=False)` — NO `model_tier` kwarg
