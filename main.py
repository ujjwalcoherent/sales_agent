"""
Sales Intelligence Platform — CLI Entry Point

Usage:
  python main.py                                      # India industry-first (default)
  python main.py --region US --industry "Fintech"     # US fintech
  python main.py --mode company --companies "NVIDIA,Apple" --region US
  python main.py --hours 72 --region IN               # 72h window, India
  python main.py --list-sources --region IN           # show which RSS sources will run
  python main.py --remove-sources sebi_v2             # drop a source for this run
  python main.py --add-sources forbes,techcrunch_main # add sources for this run
  python main.py --server                             # start FastAPI server instead
  python main.py --test                               # run all standalone tests

Region codes: IN (India), US (United States), EU (Europe), SEA (Southeast Asia), GLOBAL (all)
Each region fetches only its curated source list — no cross-region noise.

Pipeline stages (full multi-agent LangGraph):
   1. source_intel  — fetch RSS + web search (region-aware, bandit-ordered)
   2. dedup         — URL exact-match + TF-IDF cosine (title=0.85, body=0.70)
   3. NLI filter    — cross-encoder/nli-deberta-v3-small (auto_accept=0.88, auto_reject=0.10)
   4. entity NER    — GLiNER + SpaCy extraction
   5. clustering    — HAC / HDBSCAN / Leiden cascade, 7-check validation
   6. analysis      — trend synthesis + impact council (LLM with Reflexion retry)
   7. lead_gen      — opportunity scoring + lead crystallization
   8. company_agent — entity enrichment via Tavily + Apollo
   9. contact_agent — Hunter.io email find + role matching
  10. email_agent   — hyper-personalized email generation (GPT-4.1-mini structured output)
  11. learning      — 6 self-learning loops update (source bandit, NLI hypothesis, etc.)
  12. DB save       — leads, trends, contacts saved to leads.db

Results: view via frontend (http://localhost:3000) or API (GET /api/v1/leads)
Real run times: 14-30 min for India runs. First run +2 min for NLI model download.
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTHONPATH", ".")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sales Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["company", "industry", "report"],
                   default="industry", help="Discovery mode (default: industry)")
    p.add_argument("--companies", default="",
                   help="Comma-separated company names for company-first mode")
    p.add_argument("--industry", default="Technology > B2B SaaS",
                   help="Industry path (e.g. 'Healthcare > Pharma', 'Fintech')")
    p.add_argument("--hours", type=int, default=120,
                   help="Look-back window in hours (default: 120)")
    p.add_argument("--region", default="IN",
                   help="Region: IN, US, EU, SEA, GLOBAL (default: IN). Controls which RSS sources are fetched.")
    p.add_argument("--products", default="",
                   help="Comma-separated products you sell (for match scoring)")
    p.add_argument("--report", default="",
                   help="Report text for report-driven mode")

    # ── Source tuning ──────────────────────────────────────────────────────────
    p.add_argument("--add-sources", default="",
                   help="Extra source IDs to add beyond region defaults (comma-separated). "
                        "Run --list-sources to see all available IDs.")
    p.add_argument("--remove-sources", default="",
                   help="Source IDs to remove from region defaults (comma-separated).")
    p.add_argument("--list-sources", action="store_true",
                   help="Print all sources for the given --region and exit.")

    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show INFO-level logs")
    p.add_argument("--server", action="store_true",
                   help="Start FastAPI server (port 8000) instead of CLI run")
    p.add_argument("--test", action="store_true",
                   help="Run all standalone validation tests")
    return p.parse_args()


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests() -> int:
    """Run all standalone tests and return exit code."""
    import subprocess
    tests = [
        "tests/standalone/test_nli_filter.py",
        "tests/standalone/test_entity_extraction.py",
        "tests/standalone/test_clustering.py",
        "tests/standalone/test_dedup.py",
        "tests/standalone/test_pipeline_milestones.py",
    ]
    print("\n" + "=" * 70)
    print("  RUNNING STANDALONE TESTS")
    print("=" * 70)
    all_pass = True
    for t in tests:
        if not os.path.exists(t):
            print(f"  SKIP  {t} (not found)")
            continue
        result = subprocess.run(
            [sys.executable, t],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        status = "PASS" if result.returncode == 0 else "FAIL"
        if result.returncode != 0:
            all_pass = False
        last = [l for l in result.stdout.splitlines() if l.strip()][-3:] if result.stdout else []
        print(f"  {status}  {t}")
        for line in last:
            print(f"       {line}")
    print("=" * 70)
    print(f"  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70 + "\n")
    return 0 if all_pass else 1


# ── Quality report printer ────────────────────────────────────────────────────

def _bar(val: float, width: int = 20) -> str:
    filled = int(round(val * width))
    return "#" * filled + "-" * (width - filled)


def print_pipeline_report(result) -> None:
    """Print a structured quality report from PipelineResult (full multi-agent run)."""
    print()
    print("=" * 70)
    print("  FULL PIPELINE RESULTS  (LangGraph multi-agent)")
    print("=" * 70)
    print(f"  Status       : {result.status}")
    print(f"  Runtime      : {result.run_time_seconds:.0f}s  ({result.run_time_seconds/60:.1f} min)")
    ts = getattr(result, "timestamp", None)
    if ts:
        print(f"  Completed    : {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Outcome funnel
    print("  OUTCOME FUNNEL")
    print("  " + "-" * 40)
    print(f"  Trends detected   : {result.trends_detected:4d}")
    print(f"  Companies found   : {result.companies_found:4d}")
    print(f"  Leads generated   : {result.leads_generated:4d}")
    print(f"  Emails found      : {result.emails_found:4d}")
    print()

    # Output files — derive all paths from the leads json path
    if result.output_file:
        from pathlib import Path
        leads_path = Path(result.output_file)
        outputs_dir = leads_path.parent
        run_id = leads_path.stem.replace("leads_", "")  # e.g. "20260309_143022"
        print("  OUTPUT FILES")
        print("  " + "-" * 40)
        files = [
            ("run_report", outputs_dir / f"run_report_{run_id}.json",
             "everything: trends + impacts + articles + causal chains + call sheets"),
            ("leads",      leads_path,
             "leads with full email body per contact"),
            ("call_sheets", outputs_dir / f"call_sheets_{run_id}.json",
             "structured pitch sheets (trigger, pain point, opening line)"),
            ("csv",        outputs_dir / f"leads_{run_id}.csv",
             "spreadsheet — import to CRM"),
        ]
        for label, path, desc in files:
            exists = "✓" if path.exists() else "?"
            print(f"  {exists} {str(path)}")
            print(f"      {desc}")
        print()

    # Errors
    if result.errors:
        print(f"  ERRORS ({len(result.errors)})")
        print("  " + "-" * 40)
        for e in result.errors[:5]:
            print(f"  - {str(e)[:80]}")
        print()

    # Learning loop status
    print("  SELF-LEARNING STATUS")
    print("  " + "-" * 40)
    _print_learning_status()
    print()
    print("  DB: results saved to leads.db (view via frontend or GET /api/v1/leads)")
    print("=" * 70)
    print()


def _print_learning_status() -> None:
    """Print current state of all self-learning loops."""
    import json
    from pathlib import Path

    checks = {
        "Source Bandit"     : ("data/source_bandit_state.json",   "posteriors",   None),
        "NLI Hypothesis"    : ("data/filter_hypothesis.json",      "version",      None),
        "Entity Quality"    : ("data/entity_quality.json",         None,           None),
        "Adaptive Thresholds": ("data/adaptive_thresholds.json",   None,           None),
        "Signal Bus"        : ("data/signal_bus.json",             "timestamp",    None),
        "Experiment Log"    : ("data/experiments.jsonl",           None,           None),
    }
    for name, (path, key, _) in checks.items():
        p = Path(path)
        if not p.exists():
            print(f"  {name:22s} : not initialized")
            continue
        try:
            if path.endswith(".jsonl"):
                lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                print(f"  {name:22s} : {len(lines)} records")
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                if key and key in data:
                    print(f"  {name:22s} : {data[key]}")
                else:
                    size = len(data) if isinstance(data, (dict, list)) else "ok"
                    print(f"  {name:22s} : {size} entries")
        except Exception:
            print(f"  {name:22s} : (error reading)")


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _list_sources(region: str) -> None:
    """Print all source IDs active for a given region and exit."""
    from app.intelligence.config import get_region
    region_cfg = get_region(region)
    source_ids = region_cfg.source_ids
    print(f"\nActive sources for region={region} ({len(source_ids)} total):")
    print("-" * 50)
    try:
        from app.config import NEWS_SOURCES
        for sid in sorted(source_ids):
            cfg = NEWS_SOURCES.get(sid, {})
            name = cfg.get("name", sid)
            tier = cfg.get("tier", "?")
            print(f"  {sid:<35}  {tier}  {name}")
    except Exception:
        for sid in sorted(source_ids):
            print(f"  {sid}")
    print()


async def run_pipeline(args: argparse.Namespace) -> int:
    """Build DiscoveryScope from CLI args and run the FULL multi-agent pipeline.

    Calls app.agents.orchestrator.run_pipeline() — the complete LangGraph pipeline:
      source_intel → analysis → impact → quality → lead_gen → company_agent →
      contact_agent → email_agent → learning updates → DB save
    All results (leads, trends, emails) are saved to leads.db.
    """
    from app.intelligence.models import DiscoveryScope, DiscoveryMode
    from app.agents.orchestrator import run_pipeline as agents_run_pipeline

    # ── Source overrides (--add-sources / --remove-sources) ──────────────────
    # Apply before pipeline starts so overrides take effect for this run only.
    if args.add_sources or args.remove_sources:
        from app.intelligence.config import REGION_SOURCES, get_region
        region_code = args.region.upper()
        base = list(get_region(region_code).source_ids)  # current list for region
        if args.add_sources:
            extras = [s.strip() for s in args.add_sources.split(",") if s.strip()]
            for s in extras:
                if s not in base:
                    base.append(s)
            print(f"  + Added sources: {extras}")
        if args.remove_sources:
            removes = {s.strip() for s in args.remove_sources.split(",") if s.strip()}
            before = len(base)
            base = [s for s in base if s not in removes]
            print(f"  - Removed {before - len(base)} sources: {removes}")
        # Temporarily override REGION_SOURCES for this run
        REGION_SOURCES[region_code] = base

    # Build scope
    mode_map = {
        "company": DiscoveryMode.COMPANY_FIRST,
        "industry": DiscoveryMode.INDUSTRY_FIRST,
        "report": DiscoveryMode.REPORT_DRIVEN,
    }
    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    products = [p.strip() for p in args.products.split(",") if p.strip()]

    scope = DiscoveryScope(
        mode=mode_map[args.mode],
        companies=companies,
        industry=args.industry if args.mode == "industry" else None,
        report_text=args.report if args.mode == "report" else None,
        region=args.region,
        hours=args.hours,
        user_products=products,
    )

    # Show active source count for transparency
    from app.intelligence.config import get_region as _get_region
    n_sources = len(_get_region(scope.region).source_ids)

    print()
    print("=" * 70)
    print("  SALES INTELLIGENCE — FULL MULTI-AGENT PIPELINE")
    print("=" * 70)
    print(f"  Mode     : {scope.mode.value}")
    if scope.companies:
        print(f"  Companies: {', '.join(scope.companies)}")
    if scope.industry:
        print(f"  Industry : {scope.industry}")
    if scope.user_products:
        print(f"  Products : {', '.join(scope.user_products)}")
    print(f"  Region   : {scope.region} ({n_sources} sources)  |  Window: {scope.hours}h")
    print(f"  Started  : {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    print("  Stages: fetch → dedup → NLI filter → cluster → trends →")
    print("          impact analysis → lead gen → company enrich →")
    print("          contact find → email gen → learning update → DB save")
    print("  (NLI model loads on first run: ~2 min extra)")
    print()

    def log_cb(msg: str, level: str = "info") -> None:
        """Stream agent progress to stdout — sanitizes non-ASCII for Windows terminal."""
        # Terminal uses cp1252 on Windows; sanitize only for display here.
        # Actual data (JSON files, DB, frontend) keeps original Unicode untouched.
        safe = msg.encode("cp1252", errors="replace").decode("cp1252")
        if level == "step":
            print(f"\n  -- {safe}")
        elif level == "warning":
            print(f"  [WARN] {safe}")
        elif level == "error":
            print(f"  [ERR ] {safe}")
        elif level == "info":
            if len(safe) <= 120:
                print(f"         {safe}")

    try:
        result = await agents_run_pipeline(scope=scope, log_callback=log_cb)
        print_pipeline_report(result)
        return 0 if result.status == "success" else 1
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        print(f"\n  ERROR: {e}")
        print("  Run with --verbose for full traceback.")
        return 1


# ── Server ────────────────────────────────────────────────────────────────────

def run_server() -> None:
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000")
    print("Frontend: http://localhost:3000  (run: cd frontend && npm run dev)")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("app").setLevel(logging.INFO)

    if args.server:
        run_server()
        return 0

    if args.test:
        return run_tests()

    if args.list_sources:
        _list_sources(args.region)
        return 0

    return asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    sys.exit(main())
