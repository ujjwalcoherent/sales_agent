"""
Live 120h pipeline component validation.

Runs Dedup -> NLI Filter -> Entity Extraction -> Clustering on real recording data.
Reports quality metrics for each stage. No mocks.

Usage:
    PYTHONPATH=. venv/Scripts/python.exe tests/standalone/test_120h_live.py
"""
import asyncio
import json
import logging
import pathlib
import sys
import time
import warnings
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

# ── Load latest complete recording ────────────────────────────────────────────
RECORDINGS = sorted(pathlib.Path("data/recordings").glob("*/00_source_intel_complete.json"))
if not RECORDINGS:
    print("ERROR: No recordings found. Run the pipeline once first.")
    sys.exit(1)

rec_file = RECORDINGS[-1]
print(f"\n{'='*60}")
print(f"  120h Pipeline Live Component Validation")
print(f"  Recording: {rec_file.parent.name}")
print(f"{'='*60}\n")

with open(rec_file, encoding="utf-8") as f:
    src = json.load(f)

raw = src.get("articles", [])
print(f"Raw articles loaded: {len(raw)}")

from app.intelligence.models import Article, DiscoveryScope, PipelineState
from app.intelligence.config import DEFAULT_PARAMS
from app.intelligence.fetch import dedup_articles
from app.intelligence.filter import filter_articles
from app.intelligence.engine.extractor import extract_and_group_entities

articles: list[Article] = []
for i, a in enumerate(raw):
    pub = a.get("published_at")
    try:
        pub = datetime.fromisoformat(str(pub).replace("Z", "+00:00")) if pub else datetime.now(timezone.utc)
    except Exception:
        pub = datetime.now(timezone.utc)
    articles.append(Article(
        id=a.get("id", str(i)),
        url=a.get("url", f"https://test.com/{i}"),
        title=a.get("title", ""),
        summary=a.get("summary", ""),
        source_name=a.get("source_id", "unknown"),
        source_url="https://test.com",
        published_at=pub,
    ))

print()

# ═══════════════════════════════════════════════════════════════
# STAGE 1: DEDUPLICATION
# ═══════════════════════════════════════════════════════════════
print("STAGE 1: DEDUPLICATION")
print("-" * 45)
t0 = time.time()
dedup_result = dedup_articles(articles, DEFAULT_PARAMS)
deduped = dedup_result.articles
elapsed = time.time() - t0

removed = dedup_result.removed_count
pct = removed / max(len(articles), 1) * 100
print(f"  Input:        {len(articles)} articles")
print(f"  Output:       {len(deduped)} articles")
print(f"  Removed:      {removed} ({pct:.1f}%)")
print(f"  Dedup pairs:  {len(dedup_result.dedup_pairs)}")
print(f"  Time:         {elapsed:.1f}s")

stage1_ok = len(deduped) >= 50
print(f"  Result:       {'PASS' if stage1_ok else 'WARN'} (output >= 50 articles)")
print()

# ═══════════════════════════════════════════════════════════════
# STAGE 2: NLI FILTER (full corpus for realistic pass rate)
# ═══════════════════════════════════════════════════════════════
print(f"STAGE 2: NLI FILTER  (all {len(deduped)} articles)")
print("-" * 45)

t0 = time.time()
fr = asyncio.run(filter_articles(
    deduped,
    scope=DiscoveryScope(),  # generic scope — no company/industry constraint
    params=DEFAULT_PARAMS,
))
elapsed = time.time() - t0

kept = fr.articles
pass_rate = len(kept) / max(len(deduped), 1) * 100
print(f"  Input:        {len(deduped)} articles")
print(f"  Kept:         {len(kept)} ({pass_rate:.1f}%)")
print(f"  Auto-accept:  {fr.auto_accepted_count}")
print(f"  Auto-reject:  {fr.auto_rejected_count}")
print(f"  LLM zone:     {fr.llm_classified_count}")
print(f"  NLI mean:     {fr.nli_mean_entailment:.3f}")
print(f"  Time:         {elapsed:.1f}s")

bad_keywords = ["cricket", "ipl", "murder", "bail", "iphone", "galaxy s", "oneplus", "nothing phone"]
false_pos = [a for a in kept if any(kw in a.title.lower() for kw in bad_keywords)]

stage2_ok = pass_rate >= 2 and len(false_pos) == 0
print(f"  False positives (sports/crime/consumer): {len(false_pos)}")
print(f"  Result:       {'PASS' if stage2_ok else 'WARN'} (pass rate >= 2%, 0 false positives)")
print()
print("  Sample B2B kept:")
for a in kept[:10]:
    print(f"    [{a.source_name[:18]:18s}] {a.title[:60]}")
print()

# ═══════════════════════════════════════════════════════════════
# STAGE 3: ENTITY EXTRACTION
# ═══════════════════════════════════════════════════════════════
print("STAGE 3: ENTITY EXTRACTION")
print("-" * 45)
t0 = time.time()
groups, ungrouped = extract_and_group_entities(kept)
elapsed = time.time() - t0

n_articles = len(kept)
n_grouped = n_articles - len(ungrouped)
grouped_rate = n_grouped / max(n_articles, 1) * 100

print(f"  Input articles:  {n_articles}")
print(f"  Entity groups:   {len(groups)}")
print(f"  Grouped:         {n_grouped}/{n_articles} ({grouped_rate:.1f}%)")
print(f"  Ungrouped:       {len(ungrouped)}")
print(f"  Time:            {elapsed:.1f}s")

if groups:
    print()
    print("  Top entity groups:")
    for g in sorted(groups, key=lambda x: -x.mention_count)[:10]:
        print(f"    {g.canonical_name:30s}  type={g.entity_type:6s}  articles={g.mention_count}  sal={g.avg_salience:.2f}")

stage3_ok = len(groups) >= 1
print(f"\n  Result:       {'PASS' if stage3_ok else 'WARN'} (>= 1 entity group)")
print()

# ═══════════════════════════════════════════════════════════════
# STAGE 4: CLUSTERING
# ═══════════════════════════════════════════════════════════════
print("STAGE 4: CLUSTERING")
print("-" * 45)
if len(kept) >= 3:
    from app.intelligence.cluster.orchestrator import cluster_and_validate

    t0 = time.time()
    scope = DiscoveryScope()
    state = PipelineState()
    clusters, passed_ids, rejected_ids, val_results, noise_idx = asyncio.run(
        cluster_and_validate(
            articles=kept,
            entity_groups=groups,
            scope=scope,
            params=DEFAULT_PARAMS,
            state=state,
        )
    )
    elapsed = time.time() - t0

    passed_set = set(passed_ids)
    passed = [c for c in clusters if c.cluster_id in passed_set]
    failed = [c for c in clusters if c.cluster_id not in passed_set]

    coherences = [c.coherence_score for c in passed if c.coherence_score]
    mean_coh = sum(coherences) / len(coherences) if coherences else 0.0

    print(f"  Total clusters:  {len(clusters)}")
    print(f"  Passed:          {len(passed)}")
    print(f"  Failed:          {len(failed)}")
    print(f"  Noise articles:  {len(noise_idx)}")
    print(f"  Mean coherence:  {mean_coh:.3f}")
    print(f"  Time:            {elapsed:.1f}s")

    if passed:
        print()
        print("  Validated clusters:")
        for c in sorted(passed, key=lambda x: -x.coherence_score)[:6]:
            # Resolve first article from article_indices
            art_sample = kept[c.article_indices[0]].title[:50] if c.article_indices else "?"
            print(f"    coh={c.coherence_score:.3f}  n={c.article_count}  {art_sample}")

    stage4_ok = len(passed) >= 1
    print(f"\n  Result:       {'PASS' if stage4_ok else 'WARN'} (>= 1 validated cluster)")
else:
    print(f"  SKIP: only {len(kept)} articles kept, need >= 3 for clustering")
    stage4_ok = False
print()

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
stages = [
    ("STAGE 1  DEDUP",   stage1_ok),
    ("STAGE 2  FILTER",  stage2_ok),
    ("STAGE 3  ENTITY",  stage3_ok),
    ("STAGE 4  CLUSTER", stage4_ok),
]
all_pass = all(ok for _, ok in stages)
for name, ok in stages:
    print(f"  {name:20s}  {'PASS' if ok else 'WARN'}")
print()
print(f"  Overall: {'ALL PASS' if all_pass else 'SOME WARNINGS'}")
print("=" * 60)
