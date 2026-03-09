"""
Pipeline Milestone Quality Scorer -- CUDA-Agent pattern.

Implements milestone-based discrete rewards at each pipeline stage,
mirroring BytedTsinghua-SIA CUDA-Agent's staged quality scoring.

Key concepts applied:
  - Milestone reward at each discrete stage (fetch → filter → entity → cluster)
  - Pipeline score = product of stage scores (penalises ANY weak stage)
  - Protected verification: assertions checked before accepting stage output
  - RFT signal: only high-scoring pipeline runs should contribute training data

Run:
    venv/Scripts/python.exe tests/standalone/test_pipeline_milestones.py
    venv/Scripts/python.exe tests/standalone/test_pipeline_milestones.py --all-runs
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Thresholds (CUDA-Agent "verification scripts") ────────────────────────────
# These are protected constants — changing them changes what counts as GOOD/WARN/FAIL.
# In CUDA-Agent terms, these are the "milestone reward thresholds".

FETCH_DEDUP_GOOD = 0.70   # Dedup keeps >= 70% → sources not over-duplicated
FETCH_DEDUP_WARN = 0.40   # Below 40% → source overlap problem

FILTER_PRECISION_GOOD = 0.40  # >= 40% auto-accepted (NLI confidence) → GOOD
FILTER_PRECISION_WARN = 0.15  # < 15% auto-accepted → mostly ambiguous, LLM-heavy

ENTITY_SALIENCE_GOOD = 0.35   # Mean coherence/salience proxy >= 0.35 → GOOD
ENTITY_SALIENCE_WARN = 0.20   # < 0.20 → weak entity signal

CLUSTER_VALIDATION_GOOD = 0.60  # >= 60% clusters pass validation → GOOD
CLUSTER_VALIDATION_WARN = 0.30  # < 30% → pipeline producing mostly noise clusters

OVERALL_GOOD = 0.20   # Product of all 4 stages (multiplicative, so naturally low)
OVERALL_WARN = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Stage scorers
# ══════════════════════════════════════════════════════════════════════════════

def score_fetch_stage(run: dict) -> Tuple[float, dict]:
    """Stage 1 — Fetch quality: deduplication efficiency.

    Metric: deduped / raw
    Signal: how much raw input was redundant (lower = more duplicate sources)
    CUDA-Agent milestone: reward high dedup efficiency (clean input = better downstream)
    """
    raw = run.get("article_counts.input", 0)
    deduped = run.get("article_counts.after_dedup", 0)

    if raw == 0:
        return 0.0, {"raw": 0, "deduped": 0, "quality": 0.0, "status": "FAIL"}

    quality = deduped / raw

    # Secondary signals
    noise_filtered = run.get("article_counts.noise_filtered", 0)
    language_filtered = run.get("article_counts.language_filtered", 0)

    status = (
        "GOOD" if quality >= FETCH_DEDUP_GOOD
        else "WARN" if quality >= FETCH_DEDUP_WARN
        else "FAIL"
    )

    return quality, {
        "raw": raw,
        "deduped": deduped,
        "noise_filtered": noise_filtered,
        "language_filtered": language_filtered,
        "quality": round(quality, 3),
        "status": status,
    }


def score_filter_stage(run: dict) -> Tuple[float, dict]:
    """Stage 2 — Filter quality: NLI auto-accept rate.

    Metric: auto_accepted / total_processed
    Signal: fraction of articles NLI classified confidently (no LLM needed)
    Higher auto-accept = NLI model is well-calibrated for this domain.

    NOTE: The old pipeline used CMI scorer (article_counts.after_cmi_scorer).
    We reconstruct filter stats from CMI: kept = after_cmi_scorer,
    total input to filter = after_dedup.

    CUDA-Agent milestone: reward high NLI precision (less LLM dependency)
    """
    deduped = run.get("article_counts.after_dedup", 0)
    after_cmi = run.get("article_counts.after_cmi_scorer", 0)
    cmi_hard_dropped = run.get("article_counts.cmi_hard_floor_dropped", 0)
    cmi_low_relevance = run.get("cmi_low_relevance", 0)

    if deduped == 0:
        return 0.0, {"kept": 0, "total": 0, "quality": 0.0, "status": "FAIL"}

    # Precision: how many passed the filter vs total processed
    # auto-accepted = those that cleared the hard NLI floor without LLM
    auto_accepted = after_cmi  # CMI score = NLI-based auto-accept proxy
    total = deduped
    quality = auto_accepted / total if total > 0 else 0.0

    status = (
        "GOOD" if quality >= FILTER_PRECISION_GOOD
        else "WARN" if quality >= FILTER_PRECISION_WARN
        else "FAIL"
    )

    return quality, {
        "kept": after_cmi,
        "total_input": deduped,
        "cmi_hard_dropped": cmi_hard_dropped,
        "cmi_low_relevance": cmi_low_relevance,
        "quality": round(quality, 3),
        "status": status,
    }


def score_entity_stage(run: dict) -> Tuple[float, dict]:
    """Stage 3 — Entity quality: mean cluster coherence as entity signal proxy.

    Metric: mean coherence of all clusters (coherence encodes entity consistency)
    Signal: avg_intra_cluster_cosine or cluster_quality.avg_coherence

    CUDA-Agent milestone: reward coherent entity grouping (tight clusters = confirmed entities)
    """
    # Primary: avg_intra_cluster_cosine (direct cosine similarity between articles in cluster)
    avg_cosine = run.get("avg_intra_cluster_cosine", None)
    min_cosine = run.get("min_intra_cluster_cosine", None)

    # Secondary: cluster_quality avg_coherence
    cq = run.get("cluster_quality", {})
    if isinstance(cq, str):
        try:
            cq = json.loads(cq)
        except Exception:
            cq = {}

    avg_coherence = cq.get("avg_coherence", None)
    coherences = list(cq.get("cluster_coherences", {}).values())
    n_entity_groups = run.get("enrichment", {})
    if isinstance(n_entity_groups, str):
        try:
            n_entity_groups = json.loads(n_entity_groups)
        except Exception:
            n_entity_groups = {}

    total_entities = n_entity_groups.get("total_entities", 0)
    total_companies = n_entity_groups.get("total_companies", 0)

    # Quality signal: prefer avg_coherence, fall back to avg_cosine
    if avg_coherence is not None:
        quality = avg_coherence
    elif avg_cosine is not None:
        quality = avg_cosine
    elif coherences:
        quality = sum(coherences) / len(coherences)
    else:
        quality = 0.0

    status = (
        "GOOD" if quality >= ENTITY_SALIENCE_GOOD
        else "WARN" if quality >= ENTITY_SALIENCE_WARN
        else "FAIL"
    )

    return quality, {
        "total_entities": total_entities,
        "total_companies": total_companies,
        "avg_coherence": round(avg_coherence, 3) if avg_coherence is not None else None,
        "avg_intra_cluster_cosine": round(avg_cosine, 3) if avg_cosine is not None else None,
        "n_coherences_computed": len(coherences),
        "quality": round(quality, 3),
        "status": status,
    }


def score_cluster_stage(run: dict) -> Tuple[float, dict]:
    """Stage 4 — Cluster quality: validated cluster rate.

    Metric: validated_clusters / total_clusters
    Signal: fraction of clusters that passed quality gate (LLM validation)

    CUDA-Agent milestone: reward high cluster validation rate (not just quantity)
    """
    total_clusters = run.get("n_clusters", 0)
    leiden = run.get("leiden", {})
    if isinstance(leiden, str):
        try:
            leiden = json.loads(leiden)
        except Exception:
            leiden = {}

    noise_count = leiden.get("noise_count", run.get("noise_count", 0))
    modularity = leiden.get("modularity", 0.0)

    llm_val = run.get("llm_validation", {})
    if isinstance(llm_val, str):
        try:
            llm_val = json.loads(llm_val)
        except Exception:
            llm_val = {}

    clusters_validated = llm_val.get("clusters_validated", 0)
    clusters_rejected = llm_val.get("clusters_rejected", 0)
    avg_coherence_score = llm_val.get("avg_coherence_score", 0.0)

    enrichment = run.get("enrichment", {})
    if isinstance(enrichment, str):
        try:
            enrichment = json.loads(enrichment)
        except Exception:
            enrichment = {}

    valid_clusters = enrichment.get("valid_clusters", clusters_validated)
    total_in_enrichment = enrichment.get("total_clusters", total_clusters)

    if total_clusters == 0:
        return 0.0, {"total": 0, "validated": 0, "quality": 0.0, "status": "FAIL"}

    # Quality = validated / total (excludes noise)
    effective_total = total_clusters - noise_count if total_clusters > noise_count else total_clusters
    if effective_total <= 0:
        effective_total = total_clusters

    validated = valid_clusters if valid_clusters > 0 else clusters_validated
    quality = validated / effective_total if effective_total > 0 else 0.0
    quality = min(quality, 1.0)  # cap at 1.0 in case of data inconsistency

    status = (
        "GOOD" if quality >= CLUSTER_VALIDATION_GOOD
        else "WARN" if quality >= CLUSTER_VALIDATION_WARN
        else "FAIL"
    )

    return quality, {
        "total_clusters": total_clusters,
        "noise_count": noise_count,
        "effective_total": effective_total,
        "validated": validated,
        "llm_validated": clusters_validated,
        "llm_rejected": clusters_rejected,
        "outlier_ejections": llm_val.get("outlier_ejections", 0),
        "avg_coherence_score": avg_coherence_score,
        "modularity": round(modularity, 4) if modularity else 0.0,
        "quality": round(quality, 3),
        "status": status,
    }


def compute_pipeline_score(
    fetch_q: float,
    filter_q: float,
    entity_q: float,
    cluster_q: float,
) -> Tuple[float, str]:
    """Overall pipeline score — CUDA-Agent milestone product.

    Product (not average) means a single failing stage tanks the whole score.
    This is the key RFT signal: only pipeline runs above threshold contribute
    high-quality training examples to the dataset enhancer.
    """
    score = fetch_q * filter_q * entity_q * cluster_q
    status = (
        "GOOD" if score >= OVERALL_GOOD
        else "WARN" if score >= OVERALL_WARN
        else "FAIL"
    )
    return round(score, 4), status


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard renderer
# ══════════════════════════════════════════════════════════════════════════════

def _bar(quality: float, width: int = 20) -> str:
    filled = int(quality * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render_dashboard(run: dict, run_id: str = "") -> dict:
    """Compute all stage scores and print the CUDA-Agent style dashboard."""

    fetch_q, fetch_info = score_fetch_stage(run)
    filter_q, filter_info = score_filter_stage(run)
    entity_q, entity_info = score_entity_stage(run)
    cluster_q, cluster_info = score_cluster_stage(run)
    overall, overall_status = compute_pipeline_score(fetch_q, filter_q, entity_q, cluster_q)

    ts = run.get("timestamp", "?")[:16]

    print(f"\n{'=' * 65}")
    print(f"  PIPELINE HEALTH DASHBOARD  |  run={run_id or run.get('run_id','?')}  |  {ts}")
    print(f"{'=' * 65}")

    print(f"\nStage 1  FETCH   "
          f"raw={fetch_info['raw']:<5}  deduped={fetch_info['deduped']:<5}  "
          f"noise_rm={fetch_info.get('noise_filtered',0):<3}  "
          f"quality={fetch_info['quality']:.3f}  {_bar(fetch_q)}  [{fetch_info['status']}]")

    print(f"Stage 2  FILTER  "
          f"kept={filter_info['kept']:<5}  input={filter_info['total_input']:<5}  "
          f"hard_drop={filter_info.get('cmi_hard_dropped',0):<4}  "
          f"quality={filter_info['quality']:.3f}  {_bar(filter_q)}  [{filter_info['status']}]")

    entity_q_display = entity_info.get('avg_coherence') or entity_info.get('avg_intra_cluster_cosine') or 0.0
    print(f"Stage 3  ENTITY  "
          f"companies={entity_info.get('total_companies',0):<4}  "
          f"entities={entity_info.get('total_entities',0):<5}  "
          f"mean_coh={entity_q_display:.3f}        "
          f"quality={entity_info['quality']:.3f}  {_bar(entity_q)}  [{entity_info['status']}]")

    print(f"Stage 4  CLUSTER "
          f"total={cluster_info['total_clusters']:<4}  "
          f"validated={cluster_info['validated']:<3}  "
          f"noise={cluster_info['noise_count']:<3}  "
          f"mod={cluster_info.get('modularity',0):.3f}    "
          f"quality={cluster_info['quality']:.3f}  {_bar(cluster_q)}  [{cluster_info['status']}]")

    print(f"\n{'-' * 65}")
    print(f"OVERALL SCORE:  {overall:.4f}  {_bar(overall, 30)}  [{overall_status}]")
    print(f"  = fetch({fetch_q:.3f}) x filter({filter_q:.3f}) x entity({entity_q:.3f}) x cluster({cluster_q:.3f})")
    print(f"{'-' * 65}")

    # RFT signal: would this run's examples be used for training?
    rft_eligible = overall >= OVERALL_WARN
    rft_high_quality = overall >= OVERALL_GOOD
    print(f"\nCUDA-Agent RFT signal:")
    print(f"  Examples eligible for dataset (score >= {OVERALL_WARN}): {'YES' if rft_eligible else 'NO'}")
    print(f"  High-quality training examples (score >= {OVERALL_GOOD}):  {'YES ✓' if rft_high_quality else 'NO  ✗'}")

    return {
        "run_id": run_id or run.get("run_id", "?"),
        "timestamp": ts,
        "fetch": {"quality": fetch_q, **fetch_info},
        "filter": {"quality": filter_q, **filter_info},
        "entity": {"quality": entity_q, **entity_info},
        "cluster": {"quality": cluster_q, **cluster_info},
        "overall": overall,
        "overall_status": overall_status,
        "rft_eligible": rft_eligible,
        "rft_high_quality": rft_high_quality,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Assertions (CUDA-Agent "protected verification scripts")
# ══════════════════════════════════════════════════════════════════════════════

_results = []


def assert_milestone(name: str, condition: bool, detail: str = "") -> None:
    status = "[PASS]" if condition else "[FAIL]"
    _results.append((name, condition, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


def verify_dashboard_output(result: dict) -> None:
    """Protected verification: check each stage produces valid output."""
    print("\n--- Protected Verification Scripts ---")

    # Stage outputs must be in [0, 1]
    for stage in ("fetch", "filter", "entity", "cluster"):
        q = result[stage]["quality"]
        assert_milestone(
            f"{stage}: quality in [0, 1]",
            0.0 <= q <= 1.0,
            f"quality={q}",
        )

    # Overall score must be <= min of stage scores (product property)
    stage_scores = [result[s]["quality"] for s in ("fetch", "filter", "entity", "cluster")]
    overall = result["overall"]
    assert_milestone(
        "overall <= min(stage scores)",
        overall <= min(stage_scores) + 1e-6,
        f"overall={overall} min_stage={min(stage_scores):.3f}",
    )

    # Fetch: deduped cannot exceed raw
    assert_milestone(
        "fetch: deduped <= raw articles",
        result["fetch"]["deduped"] <= result["fetch"]["raw"],
        f"deduped={result['fetch']['deduped']} raw={result['fetch']['raw']}",
    )

    # Filter: kept <= deduped input
    assert_milestone(
        "filter: kept <= deduped input",
        result["filter"]["kept"] <= result["filter"]["total_input"],
        f"kept={result['filter']['kept']} input={result['filter']['total_input']}",
    )

    # Cluster: validated <= total
    assert_milestone(
        "cluster: validated <= total_clusters",
        result["cluster"]["validated"] <= result["cluster"]["total_clusters"] + 1,
        f"validated={result['cluster']['validated']} total={result['cluster']['total_clusters']}",
    )

    # Overall must be non-negative
    assert_milestone(
        "overall: pipeline score >= 0",
        overall >= 0.0,
        f"overall={overall}",
    )

    # RFT eligibility is bool
    assert_milestone(
        "rft_eligible is boolean",
        isinstance(result["rft_eligible"], bool),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def load_runs(log_path: Path) -> List[dict]:
    runs = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
    return runs


def main() -> None:
    all_runs_flag = "--all-runs" in sys.argv

    log_path = ROOT / "data" / "pipeline_run_log.jsonl"
    if not log_path.exists():
        print(f"[ERROR] pipeline_run_log.jsonl not found at {log_path}")
        sys.exit(1)

    runs = load_runs(log_path)
    if not runs:
        print("[ERROR] No runs found in pipeline_run_log.jsonl")
        sys.exit(1)

    print(f"\nLoaded {len(runs)} pipeline runs from {log_path.name}")

    if all_runs_flag:
        # Score every run
        target_runs = runs
    else:
        # Default: use the richest run (highest article count with full cluster data)
        best_run = max(
            runs,
            key=lambda r: (
                r.get("article_counts.input", 0),
                r.get("n_clusters", 0),
            ),
        )
        target_runs = [best_run]
        print(f"Using richest run: {best_run.get('run_id','?')} "
              f"({best_run.get('article_counts.input',0)} articles, "
              f"{best_run.get('n_clusters',0)} clusters)")

    dashboard_results = []
    for run in target_runs:
        result = render_dashboard(run)
        dashboard_results.append(result)
        verify_dashboard_output(result)

    # Summary across all scored runs
    if len(dashboard_results) > 1:
        print(f"\n{'=' * 65}")
        print(f"  MULTI-RUN SUMMARY ({len(dashboard_results)} runs)")
        print(f"{'=' * 65}")
        print(f"{'Timestamp':<18} {'Fetch':>7} {'Filter':>7} {'Entity':>7} {'Cluster':>8} {'Overall':>8}  RFT")
        print(f"{'-' * 65}")
        for r in sorted(dashboard_results, key=lambda x: x["timestamp"], reverse=True):
            rft_mark = "HQ" if r["rft_high_quality"] else ("ok" if r["rft_eligible"] else "--")
            print(
                f"{r['timestamp']:<18} "
                f"{r['fetch']['quality']:>7.3f} "
                f"{r['filter']['quality']:>7.3f} "
                f"{r['entity']['quality']:>7.3f} "
                f"{r['cluster']['quality']:>8.3f} "
                f"{r['overall']:>8.4f}  {rft_mark}"
            )
        overalls = [r["overall"] for r in dashboard_results]
        print(f"\n  Mean overall: {sum(overalls)/len(overalls):.4f}")
        print(f"  Best run:     {max(overalls):.4f}")
        print(f"  Worst run:    {min(overalls):.4f}")
        hq_count = sum(1 for r in dashboard_results if r["rft_high_quality"])
        print(f"  RFT high-quality runs: {hq_count}/{len(dashboard_results)}")

    # Final assertion summary
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    print(f"\n{'=' * 65}")
    print(f"  VERIFICATION: {passed} passed, {failed} failed out of {len(_results)} checks")
    print(f"{'=' * 65}")

    if failed:
        print("\nFailed checks:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)
    else:
        print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
