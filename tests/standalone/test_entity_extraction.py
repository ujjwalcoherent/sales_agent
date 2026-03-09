"""
Standalone entity extraction validation test.

Validates that GLiNER + spaCy correctly extract B2B entities from B2B news articles.

Run:
    venv/Scripts/python.exe tests/standalone/test_entity_extraction.py

What this tests:
  1. Known-entity articles: 5 synthetic articles with ground-truth entities.
     Reports PASS/FAIL per entity, type accuracy, salience ordering.
  2. Real-data sweep: 20 articles from latest recording (last 72h).
     Reports top-10 entities by salience, type distribution, zero-entity failures.

CUDA-agent "Milestone reward" quality scoring:
  - Synthetic: score = entities_found / expected_entities per article
  - Real: score = 1.0 if top entity salience > 0.30, else top_salience / 0.30
"""

import io
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── project root on path ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress noisy library logs during test
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.WARNING)
logging.getLogger("gliner").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── imports ────────────────────────────────────────────────────────────────
from app.intelligence.engine.extractor import extract_and_group_entities
from app.schemas.news import NewsArticle, Entity


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_spacy_ner_on_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    """Pre-populate article.entities using spaCy so the extractor skips its NER step.

    The extractor's _ensure_ner() checks for articles missing both `entities`
    and `entities_raw`. NewsArticle has `entities: List[Entity]` (not entities_raw),
    so we pre-run spaCy here and populate the typed Entity list directly.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return articles  # Graceful degradation — let extractor handle it

    for art in articles:
        if art.entities:
            continue  # Already has entities
        text = (art.title or "") + " " + (art.summary or "")
        doc = nlp(text[:600])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "PRODUCT", "GPE"):
                art.entities.append(Entity(
                    text=ent.text,
                    type=ent.label_,
                    salience=0.5,
                ))
                art.entity_names.append(ent.text)
    return articles


def _make_article(
    title: str,
    summary: str,
    source_id: str = "test",
    source_name: str = "TestSource",
    url: Optional[str] = None,
) -> NewsArticle:
    """Build a minimal NewsArticle for testing."""
    return NewsArticle(
        title=title,
        summary=summary,
        url=url or f"https://test.example.com/{hash(title) % 99999}",
        source_id=source_id,
        source_name=source_name,
        published_at=datetime.now(timezone.utc),
    )


def _articles_from_recording(path: str, max_articles: int = 20) -> List[NewsArticle]:
    """Load articles from a run-recorder JSON file.

    Loads up to *max_articles* from the 72h window nearest to the recording's
    latest timestamp (newest articles first).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("articles", [])
    if not raw:
        return []

    # Filter to 72h window from latest article date
    dates = [r["published_at"] for r in raw if r.get("published_at")]
    if dates:
        latest = max(datetime.fromisoformat(d) for d in dates)
        cutoff = latest - timedelta(hours=72)
        raw = [
            r for r in raw
            if r.get("published_at") and datetime.fromisoformat(r["published_at"]) >= cutoff
        ]

    # Sort newest-first, cap at max_articles
    raw.sort(key=lambda r: r.get("published_at", ""), reverse=True)
    raw = raw[:max_articles]

    articles = []
    for r in raw:
        try:
            articles.append(NewsArticle(
                title=r.get("title", ""),
                summary=r.get("summary", ""),
                url=r.get("url", "https://test.example.com/unknown"),
                source_id=r.get("source_id", "recording"),
                source_name=r.get("source_name") or r.get("source_id", "recording"),
                published_at=datetime.fromisoformat(r["published_at"]),
            ))
        except Exception:
            pass  # Skip malformed records
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpectedEntity:
    """One expected entity in a test article."""
    name: str                          # Canonical name (case-insensitive match)
    entity_type: str                   # Expected GLiNER-corrected type: ORG / PERSON / PRODUCT
    # Aliases the extractor might produce after normalization/fuzzy-grouping
    aliases: List[str] = field(default_factory=list)


@dataclass
class TestCase:
    """A test article with known ground-truth entities."""
    article_id: str
    title: str
    summary: str
    expected: List[ExpectedEntity]
    description: str = ""


TEST_CASES: List[TestCase] = [
    TestCase(
        article_id="TC01",
        title="Zepto raises $200M Series G led by General Catalyst",
        summary=(
            "Quick-commerce startup Zepto has closed a $200 million Series G funding round "
            "led by General Catalyst, with participation from Nexus Venture Partners. "
            "Zepto CEO Aadit Palicha said the funds will be used to expand dark-store coverage "
            "across India's Tier 2 cities."
        ),
        expected=[
            ExpectedEntity("Zepto", "ORG", aliases=["zepto"]),
            ExpectedEntity("General Catalyst", "ORG", aliases=["general catalyst"]),
        ],
        description="Startup funding — Indian startup + US VC firm",
    ),
    TestCase(
        article_id="TC02",
        title="Infosys signs cloud migration deal with HDFC Bank worth Rs 2000 crore",
        summary=(
            "Infosys announced a multi-year cloud migration engagement with HDFC Bank, "
            "one of India's largest private sector banks. The deal is valued at approximately "
            "Rs 2000 crore and covers migration of core banking workloads to AWS infrastructure. "
            "Infosys will deploy its Cobalt platform for the transformation."
        ),
        expected=[
            ExpectedEntity("Infosys", "ORG", aliases=["infosys"]),
            ExpectedEntity("HDFC Bank", "ORG", aliases=["hdfc bank", "hdfc"]),
        ],
        description="IT services deal — Indian IT company + Indian financial institution",
    ),
    TestCase(
        article_id="TC03",
        title="Razorpay CEO Harshil Mathur announces Series H fundraise",
        summary=(
            "Razorpay, India's leading payments infrastructure company, confirmed it is in "
            "advanced discussions for a Series H funding round. CEO and co-founder Harshil Mathur "
            "made the announcement at the Fintech Summit 2026, saying the company targets a "
            "$12 billion valuation. Razorpay processes over $180 billion in payment volume annually."
        ),
        expected=[
            ExpectedEntity("Razorpay", "ORG", aliases=["razorpay"]),
            ExpectedEntity("Harshil Mathur", "PERSON", aliases=["harshil mathur"]),
        ],
        description="Fintech funding — startup + executive as PERSON entity",
    ),
    TestCase(
        article_id="TC04",
        title="Freshworks acquires LevelOps for enterprise DevSecOps",
        summary=(
            "Freshworks has acquired LevelOps, a San Francisco-based DevSecOps startup, "
            "to enhance its Freshservice ITSM product with security operations capabilities. "
            "LevelOps integrates directly into CI/CD pipelines. Freshworks CEO Dennis Woodside "
            "called the acquisition a key step in the company's platform expansion strategy."
        ),
        expected=[
            ExpectedEntity("Freshworks", "ORG", aliases=["freshworks"]),
            ExpectedEntity("LevelOps", "ORG", aliases=["levelops"]),
        ],
        description="Enterprise software M&A — acquirer + acquired startup",
    ),
    TestCase(
        article_id="TC05",
        title="NVIDIA DGX H200 deployed at Tata Consultancy Services data center",
        summary=(
            "NVIDIA announced that Tata Consultancy Services has deployed the DGX H200 AI "
            "supercomputer cluster at its Hyderabad data center to accelerate enterprise AI "
            "workloads. TCS will leverage NVIDIA's full-stack AI platform including NeMo and "
            "NIM microservices. Jensen Huang, NVIDIA CEO, called TCS a strategic AI partner."
        ),
        expected=[
            ExpectedEntity("NVIDIA", "ORG", aliases=["nvidia"]),
            ExpectedEntity("Tata Consultancy Services", "ORG",
                           aliases=["tata consultancy services", "tcs"]),
        ],
        description="Enterprise AI infrastructure — NVIDIA + major IT services company",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: SYNTHETIC ARTICLE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _find_entity_in_groups(
    expected: ExpectedEntity,
    groups: List[Any],
) -> Optional[Any]:
    """Return the EntityGroup matching an expected entity (case-insensitive, alias-aware)."""
    search_names = {expected.name.lower()} | {a.lower() for a in expected.aliases}
    for g in groups:
        canonical_lower = g.canonical_name.lower()
        # Direct canonical match
        if canonical_lower in search_names:
            return g
        # Variant match
        for v in g.variant_names:
            if v.lower() in search_names:
                return g
        # Partial/substring match (e.g. "HDFC" finding "HDFC Bank")
        for sn in search_names:
            if sn in canonical_lower or canonical_lower in sn:
                return g
    return None


@dataclass
class EntityResult:
    expected_name: str
    expected_type: str
    found: bool
    found_name: str = ""
    found_type: str = ""
    found_salience: float = 0.0
    type_correct: bool = False


@dataclass
class ArticleResult:
    article_id: str
    description: str
    entity_results: List[EntityResult]
    all_extracted: List[str]          # All canonical names extracted from this article
    quality_score: float              # entities_found / expected_entities


def run_synthetic_tests(verbose: bool = True) -> List[ArticleResult]:
    """Run all synthetic test cases. Each article processed independently."""
    print("\n" + "=" * 72)
    print("PART 1 — SYNTHETIC ARTICLE TESTS (known ground-truth entities)")
    print("=" * 72)

    results: List[ArticleResult] = []

    for tc in TEST_CASES:
        # Build a single-article list — we need min_articles=1 for single-article extraction
        article = _make_article(tc.title, tc.summary)
        # Pre-run spaCy NER to populate article.entities (NewsArticle has no entities_raw field)
        articles_prepped = _run_spacy_ner_on_articles([article])

        t0 = time.perf_counter()
        groups, ungrouped = extract_and_group_entities(articles_prepped, min_articles=1)
        elapsed = time.perf_counter() - t0

        all_extracted = [g.canonical_name for g in groups]

        entity_results: List[EntityResult] = []
        found_count = 0

        for exp in tc.expected:
            matched_group = _find_entity_in_groups(exp, groups)
            if matched_group:
                found_count += 1
                type_ok = matched_group.entity_type == exp.entity_type
                entity_results.append(EntityResult(
                    expected_name=exp.name,
                    expected_type=exp.entity_type,
                    found=True,
                    found_name=matched_group.canonical_name,
                    found_type=matched_group.entity_type,
                    found_salience=matched_group.avg_salience,
                    type_correct=type_ok,
                ))
            else:
                entity_results.append(EntityResult(
                    expected_name=exp.name,
                    expected_type=exp.entity_type,
                    found=False,
                ))

        quality = found_count / len(tc.expected) if tc.expected else 0.0
        ar = ArticleResult(
            article_id=tc.article_id,
            description=tc.description,
            entity_results=entity_results,
            all_extracted=all_extracted,
            quality_score=quality,
        )
        results.append(ar)

        if verbose:
            _print_article_result(ar, elapsed)

    # Summary table
    _print_synthetic_summary(results)
    return results


def _print_article_result(ar: ArticleResult, elapsed: float) -> None:
    """Print one article's extraction result."""
    bar = "PASS" if ar.quality_score >= 1.0 else ("PARTIAL" if ar.quality_score > 0 else "FAIL")
    color = "\033[92m" if bar == "PASS" else ("\033[93m" if bar == "PARTIAL" else "\033[91m")
    reset = "\033[0m"

    print(f"\n[{ar.article_id}] {ar.description}")
    print(f"  Quality score: {ar.quality_score:.0%}  {color}{bar}{reset}  ({elapsed*1000:.0f}ms)")

    for er in ar.entity_results:
        if er.found:
            type_tag = "TYPE_OK" if er.type_correct else f"TYPE_WRONG(got {er.found_type})"
            print(f"  [PASS] '{er.expected_name}' -> found as '{er.found_name}' "
                  f"[{type_tag}]  salience={er.found_salience:.3f}")
        else:
            print(f"  [FAIL] '{er.expected_name}' (expected {er.expected_type}) — NOT FOUND")

    # Show all extracted entities (including extras / potential hallucinations)
    if ar.all_extracted:
        expected_lower = {e.expected_name.lower() for e in ar.entity_results}
        expected_aliases = set()
        for tc in TEST_CASES:
            if tc.article_id == ar.article_id:
                for exp in tc.expected:
                    expected_aliases |= {a.lower() for a in exp.aliases}
                    expected_aliases.add(exp.name.lower())

        extras = [
            name for name in ar.all_extracted
            if name.lower() not in expected_lower
            and not any(name.lower() in alias or alias in name.lower()
                        for alias in expected_aliases)
        ]
        if extras:
            print(f"  [EXTRA entities extracted — not in expected]: {extras}")
    else:
        print(f"  [WARNING] Zero entities extracted from this article")


def _print_synthetic_summary(results: List[ArticleResult]) -> None:
    """Print aggregate summary across all synthetic tests."""
    print("\n" + "-" * 72)
    print("SYNTHETIC TESTS — SUMMARY")
    print("-" * 72)

    all_entity_results = [er for ar in results for er in ar.entity_results]
    total = len(all_entity_results)
    found = sum(1 for er in all_entity_results if er.found)
    type_correct = sum(1 for er in all_entity_results if er.found and er.type_correct)
    mean_quality = sum(ar.quality_score for ar in results) / len(results) if results else 0.0

    print(f"  Entity recall:     {found}/{total}  ({found/total*100:.1f}%)")
    print(f"  Type accuracy:     {type_correct}/{found}  ({type_correct/found*100:.1f}% of found)"
          if found else "  Type accuracy:     N/A (no entities found)")
    print(f"  Mean quality score:{mean_quality:.2f}  ({mean_quality*100:.0f}%)")

    # Per-article quality table
    print()
    print(f"  {'Article':<8} {'Description':<45} {'Quality':>8}")
    print(f"  {'-'*8} {'-'*45} {'-'*8}")
    for ar in results:
        bar = "PASS" if ar.quality_score >= 1.0 else ("PARTIAL" if ar.quality_score > 0 else "FAIL")
        print(f"  {ar.article_id:<8} {ar.description[:45]:<45} {ar.quality_score:>6.0%}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: REAL DATA SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_real_data_tests(recording_path: str, n_articles: int = 20) -> None:
    """Run entity extraction on real articles and report quality metrics."""
    print("\n" + "=" * 72)
    print(f"PART 2 — REAL DATA SWEEP ({n_articles} articles from latest recording)")
    print("=" * 72)

    # Load articles
    articles = _articles_from_recording(recording_path, max_articles=n_articles)
    if not articles:
        print("  [ERROR] No articles loaded from recording — skipping real data tests.")
        return

    print(f"  Loaded {len(articles)} articles from: {recording_path}")
    print(f"  Date range: {articles[-1].published_at.strftime('%Y-%m-%d')} "
          f"to {articles[0].published_at.strftime('%Y-%m-%d')}")

    # Pre-run spaCy NER to populate article.entities
    articles = _run_spacy_ner_on_articles(articles)

    # Run extraction (need min_articles=2 to reduce noise on real data)
    t0 = time.perf_counter()
    groups, ungrouped = extract_and_group_entities(articles, min_articles=2)
    elapsed = time.perf_counter() - t0

    print(f"  Extraction time:   {elapsed:.2f}s")
    print(f"  Groups found:      {len(groups)}")
    print(f"  Ungrouped articles:{len(ungrouped)}")

    # ── Top 10 entities by avg_salience ──────────────────────────────────
    top10 = sorted(groups, key=lambda g: g.avg_salience, reverse=True)[:10]
    print()
    print("  TOP 10 ENTITIES BY SALIENCE:")
    print(f"  {'Rank':<5} {'Entity':<35} {'Type':<10} {'Salience':>8} {'Mentions':>9} {'Articles':>9}")
    print(f"  {'-'*5} {'-'*35} {'-'*10} {'-'*8} {'-'*9} {'-'*9}")
    for i, g in enumerate(top10, 1):
        print(f"  {i:<5} {g.canonical_name[:35]:<35} {g.entity_type:<10} "
              f"{g.avg_salience:>8.3f} {g.mention_count:>9} {len(g.article_indices):>9}")

    # ── Entity type distribution ──────────────────────────────────────────
    type_counts = Counter(g.entity_type for g in groups)
    total_groups = len(groups)
    print()
    print("  ENTITY TYPE DISTRIBUTION:")
    for etype, cnt in type_counts.most_common():
        pct = cnt / total_groups * 100 if total_groups else 0
        bar = "#" * int(pct / 3)
        print(f"    {etype:<12} {cnt:>5}  ({pct:>5.1f}%)  {bar}")

    # ── GLiNER label distribution (if available) ──────────────────────────
    gliner_labels = Counter()
    for g in groups:
        if g.is_validated and g.validation_evidence:
            for ev in g.validation_evidence:
                gliner_labels[ev] += 1
    if gliner_labels:
        print()
        print("  GLiNER FINE-GRAINED LABEL DISTRIBUTION (top 8):")
        for label, cnt in gliner_labels.most_common(8):
            print(f"    {label:<28} {cnt}")

    # ── Articles with zero entities extracted ─────────────────────────────
    # An article at index i has zero entities if no EntityGroup claims index i
    grouped_indices = set()
    for g in groups:
        grouped_indices.update(g.article_indices)
    zero_entity_articles = [i for i in range(len(articles)) if i not in grouped_indices]

    print()
    print(f"  ARTICLES WITH ZERO EXTRACTED ENTITIES: {len(zero_entity_articles)}/{len(articles)}")
    if zero_entity_articles:
        for idx in zero_entity_articles[:5]:
            art = articles[idx]
            print(f"    [{idx}] {art.title[:80]}")
        if len(zero_entity_articles) > 5:
            print(f"    ... and {len(zero_entity_articles) - 5} more")

    # ── Quality scoring (Milestone reward) ───────────────────────────────
    print()
    print("  QUALITY SCORING (Milestone reward):")
    print("  Score = 1.0 if top entity salience > 0.30, else top_salience / 0.30")
    if top10:
        top_salience = top10[0].avg_salience
        quality = 1.0 if top_salience >= 0.30 else top_salience / 0.30
        q_label = "GOOD" if quality >= 1.0 else ("MARGINAL" if quality >= 0.6 else "POOR")
        print(f"    Top entity: '{top10[0].canonical_name}' salience={top_salience:.3f}")
        print(f"    Quality score: {quality:.2f}  [{q_label}]")
    else:
        print("    [WARN] No entity groups found — quality score: 0.00  [POOR]")

    # Per-article quality table
    print()
    print("  PER-ARTICLE EXTRACTION QUALITY (salience_quality = top entity salience):")
    print(f"  {'#':<4} {'Entities':>8} {'Top_Salience':>13} {'Score':>7}  Title")
    print(f"  {'-'*4} {'-'*8} {'-'*13} {'-'*7}  {'-'*40}")
    for idx, art in enumerate(articles):
        # Find all groups claiming this article
        art_groups = [g for g in groups if idx in g.article_indices]
        if art_groups:
            best = max(art_groups, key=lambda g: g.avg_salience)
            top_sal = best.avg_salience
            art_score = min(1.0, top_sal / 0.30)
            n_ents = len(art_groups)
        else:
            top_sal = 0.0
            art_score = 0.0
            n_ents = 0

        score_label = "OK" if art_score >= 1.0 else ("LOW" if art_score >= 0.5 else "ZERO")
        title_short = art.title[:50]
        print(f"  {idx:<4} {n_ents:>8} {top_sal:>13.3f} {art_score:>6.2f}  "
              f"[{score_label}] {title_short}")

    # ── Deduplication check ───────────────────────────────────────────────
    canonical_names = [g.canonical_name.lower() for g in groups]
    duplicates = [n for n, c in Counter(canonical_names).items() if c > 1]
    print()
    if duplicates:
        print(f"  [WARN] Duplicate canonical names found ({len(duplicates)}): {duplicates[:5]}")
    else:
        print(f"  [PASS] Deduplication: no duplicate canonical names in {len(groups)} groups")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: SALIENCE ORDERING VALIDATION (primary vs secondary entities)
# ══════════════════════════════════════════════════════════════════════════════

def run_salience_ordering_tests() -> None:
    """Validate that primary company has higher salience than secondary mentions."""
    print("\n" + "=" * 72)
    print("PART 3 — SALIENCE ORDERING (primary entity should outrank secondary)")
    print("=" * 72)

    # Article where Zepto is the subject and General Catalyst is a secondary actor
    articles = [
        _make_article(
            title="Zepto raises $200M Series G led by General Catalyst",
            summary=(
                "Zepto closed a $200 million Series G. The round was led by General Catalyst. "
                "Zepto plans to use the capital to expand operations. Zepto's valuation now "
                "stands at $5 billion. General Catalyst joins existing investors."
            ),
        ),
    ]
    articles = _run_spacy_ner_on_articles(articles)

    groups, _ = extract_and_group_entities(articles, min_articles=1)

    if len(groups) < 2:
        print(f"  [SKIP] Only {len(groups)} entity groups found (need >=2 for ordering test)")
        for g in groups:
            print(f"    {g.canonical_name}: salience={g.avg_salience:.3f}")
        return

    zepto_group = _find_entity_in_groups(
        ExpectedEntity("Zepto", "ORG", aliases=["zepto"]), groups
    )
    gc_group = _find_entity_in_groups(
        ExpectedEntity("General Catalyst", "ORG", aliases=["general catalyst"]), groups
    )

    print(f"  {'Entity':<25} {'Salience':>10} {'Type':<10} {'Mentions':>9}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*9}")
    for g in sorted(groups, key=lambda x: x.avg_salience, reverse=True):
        flag = "<-- PRIMARY" if (zepto_group and g.canonical_name == zepto_group.canonical_name) else ""
        print(f"  {g.canonical_name[:25]:<25} {g.avg_salience:>10.3f} "
              f"{g.entity_type:<10} {g.mention_count:>9}  {flag}")

    if zepto_group and gc_group:
        primary_higher = zepto_group.avg_salience >= gc_group.avg_salience
        status = "PASS" if primary_higher else "FAIL"
        print()
        print(f"  Primary (Zepto) salience={zepto_group.avg_salience:.3f} "
              f"vs Secondary (General Catalyst) salience={gc_group.avg_salience:.3f}")
        print(f"  Salience ordering: [{status}]")
    else:
        missing = []
        if not zepto_group:
            missing.append("Zepto")
        if not gc_group:
            missing.append("General Catalyst")
        print(f"  [SKIP] Cannot test ordering — missing groups: {missing}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: DEDUPLICATION VALIDATION (same entity mentioned twice = 1 group)
# ══════════════════════════════════════════════════════════════════════════════

def run_dedup_tests() -> None:
    """Validate that the same company across 2 articles produces only 1 EntityGroup."""
    print("\n" + "=" * 72)
    print("PART 4 — DEDUPLICATION (same company in 2 articles = 1 EntityGroup)")
    print("=" * 72)

    # Infosys appears in both articles — should produce 1 group, not 2
    articles = [
        _make_article(
            title="Infosys Q4 revenue beats estimates with 8% growth",
            summary=(
                "Infosys reported Q4 revenue of $4.7 billion, beating analyst estimates. "
                "Infosys maintained its full-year guidance of 4-7% revenue growth."
            ),
        ),
        _make_article(
            title="Infosys wins 5-year deal with Deutsche Bank for cloud migration",
            summary=(
                "Infosys has secured a landmark five-year cloud transformation deal with "
                "Deutsche Bank. The deal strengthens Infosys's position in the European market."
            ),
        ),
    ]
    articles = _run_spacy_ner_on_articles(articles)

    groups, ungrouped = extract_and_group_entities(articles, min_articles=1)

    infosys_groups = [
        g for g in groups
        if "infosys" in g.canonical_name.lower()
        or any("infosys" in v.lower() for v in g.variant_names)
    ]

    print(f"  Articles: 2 (both mention Infosys)")
    print(f"  Total groups extracted: {len(groups)}")
    print(f"  Groups containing 'Infosys': {len(infosys_groups)}")

    if len(infosys_groups) == 1:
        g = infosys_groups[0]
        print(f"  [PASS] Single Infosys group: '{g.canonical_name}' "
              f"across {len(g.article_indices)} articles, "
              f"salience={g.avg_salience:.3f}")
    elif len(infosys_groups) == 0:
        print(f"  [FAIL] Infosys not extracted at all from 2 articles")
    else:
        print(f"  [FAIL] Duplicate groups for Infosys ({len(infosys_groups)} groups):")
        for g in infosys_groups:
            print(f"    '{g.canonical_name}' articles={g.article_indices}")

    # Show all groups for transparency
    if groups:
        print()
        print(f"  All extracted groups:")
        for g in groups:
            print(f"    '{g.canonical_name}' [{g.entity_type}] "
                  f"articles={g.article_indices} salience={g.avg_salience:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 72)
    print("ENTITY EXTRACTION VALIDATION — B2B Sales Intelligence Pipeline")
    print("GLiNER + spaCy two-tier extraction quality test")
    print("=" * 72)

    # Part 1: Synthetic tests with known ground-truth
    synthetic_results = run_synthetic_tests(verbose=True)

    # Part 2: Real data sweep
    recording_path = os.path.join(
        PROJECT_ROOT, "data", "recordings", "20260307_152046",
        "00_source_intel_complete.json"
    )
    if os.path.exists(recording_path):
        run_real_data_tests(recording_path, n_articles=20)
    else:
        print(f"\n[SKIP] Real data test — recording not found: {recording_path}")

    # Part 3: Salience ordering
    run_salience_ordering_tests()

    # Part 4: Deduplication
    run_dedup_tests()

    # ── Final verdict ──────────────────────────────────────────────────────
    all_entity_results = [er for ar in synthetic_results for er in ar.entity_results]
    total = len(all_entity_results)
    found = sum(1 for er in all_entity_results if er.found)
    type_correct = sum(1 for er in all_entity_results if er.found and er.type_correct)
    mean_quality = sum(ar.quality_score for ar in synthetic_results) / len(synthetic_results)

    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print(f"  Synthetic recall:   {found}/{total}  ({found/total*100:.1f}%)")
    print(f"  Type accuracy:      {type_correct}/{found}  ({type_correct/found*100:.1f}% of found)"
          if found else "  Type accuracy:      N/A")
    print(f"  Mean quality score: {mean_quality:.2f}")

    overall = "PASS" if mean_quality >= 0.8 else ("PARTIAL" if mean_quality >= 0.5 else "FAIL")
    print(f"\n  Overall: [{overall}]")
    print()


if __name__ == "__main__":
    main()
