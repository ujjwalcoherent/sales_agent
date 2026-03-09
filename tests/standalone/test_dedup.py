"""
Standalone dedup test -- validates the dedup_articles() function in isolation.

Tests:
  1. Exact duplicate removal (same URL)
  2. Near-duplicate removal (same title, different source)
  3. TF-IDF threshold behavior: reworded titles correctly NOT collapsed (0.85 is strict)
  4. Statistics: raw -> deduped in real 72h data
  5. Quality check: no legitimate distinct articles removed

Protected invariants (CUDA-Agent style):
  I1. output count <= input count
  I2. no article appears twice in output
  I3. URL uniqueness: no two articles share the same URL

Run:
  PYTHONPATH=/d/Project/sales_agent venv/Scripts/python.exe tests/standalone/test_dedup.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

# ---- Path setup -----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.intelligence.fetch import dedup_articles, fetch_articles
from app.intelligence.models import RawArticle, DedupResult, DiscoveryScope
from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS


# ---- Helpers ---------------------------------------------------------------

PASS_LABEL = "[PASS]"
FAIL_LABEL = "[FAIL]"

_results: List[Tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS_LABEL if condition else FAIL_LABEL
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {status} {label}{detail_str}")
    _results.append((label, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def make_article(
    url: str,
    title: str,
    summary: str = "",
    source_name: str = "TestSource",
    hours_ago: float = 1.0,
) -> RawArticle:
    return RawArticle(
        url=url,
        title=title,
        summary=summary or title,
        source_name=source_name,
        source_url=url,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        fetch_method="test",
    )


# ---- Test 1: Exact duplicate removal (same URL) ----------------------------

def test_exact_url_dedup() -> None:
    section("Test 1: Exact URL Deduplication")

    params = ClusteringParams(dedup_title_threshold=0.85, dedup_body_threshold=0.70)

    a1 = make_article("https://example.com/article-1", "NVIDIA announces new GPU chip for AI workloads")
    a2 = make_article("https://example.com/article-1", "NVIDIA announces new GPU chip for AI workloads",
                      source_name="AnotherSource")  # Same URL, different source
    a3 = make_article("https://example.com/article-2", "Microsoft acquires AI startup for $2 billion")

    raw = [a1, a2, a3]
    result = dedup_articles(raw, params)

    output_urls = [a.url for a in result.articles]
    unique_urls = set(output_urls)

    check("Same URL appears only once in output",
          len(unique_urls) == len(output_urls),
          f"output has {len(result.articles)} articles, {len(unique_urls)} unique URLs")
    check("Distinct article kept",
          any("article-2" in u for u in output_urls),
          "article-2 should survive")
    check("Removed count >= 1",
          result.removed_count >= 1,
          f"removed_count={result.removed_count}")
    check("Math assertion: non-increasing",
          result.assertion_count_non_increasing,
          f"output={len(result.articles)} <= input={len(raw)}")


# ---- Test 2: Near-duplicate removal (same title, different source) ----------

def test_title_near_dedup() -> None:
    section("Test 2: Near-Duplicate Removal (Same Title, Different Source)")

    params = ClusteringParams(dedup_title_threshold=0.85, dedup_body_threshold=0.70)

    # Syndicated article -- same title across sources (cosine = 1.0)
    title = "Infosys wins $500 million cloud transformation deal with European bank"
    a1 = make_article("https://economictimes.com/infosys-deal", title,
                      source_name="Economic Times", hours_ago=5)
    a2 = make_article("https://livemint.com/infosys-deal", title,
                      source_name="LiveMint", hours_ago=4)
    a3 = make_article("https://moneycontrol.com/infosys-deal", title,
                      source_name="MoneyControl", hours_ago=3)

    # Distinct article -- should NOT be removed
    a4 = make_article("https://techcrunch.com/tata-funding",
                      "Tata Digital raises $300 million in Series B funding round",
                      source_name="TechCrunch", hours_ago=2)

    raw = [a1, a2, a3, a4]
    result = dedup_articles(raw, params)

    output_titles = [a.title for a in result.articles]
    syndicated_kept = sum(1 for t in output_titles if "Infosys" in t)

    check("Syndicated copies collapsed to 1",
          syndicated_kept == 1,
          f"Found {syndicated_kept} Infosys articles in output (expected 1)")
    check("Distinct article (Tata) kept",
          any("Tata" in t for t in output_titles),
          "Tata article should not be deduped")
    check("At least 2 removed from syndicated set",
          result.removed_count >= 2,
          f"removed_count={result.removed_count}")

    print(f"\n    Dedup pairs (first 3):")
    for kept_url, removed_url in result.dedup_pairs[:3]:
        print(f"      KEPT: {kept_url}")
        print(f"      RMVD: {removed_url}")
        print()


# ---- Test 3: TF-IDF threshold behavior (0.85 is strict by design) -----------

def test_tfidf_threshold_behavior() -> None:
    """Verify that the 0.85 TF-IDF threshold is correctly calibrated.

    Key insight from Manber & Wu 1994: 0.85 is designed for near-EXACT copies
    (same words, slight reordering). Semantically similar but reworded articles
    should NOT be collapsed -- that would cause false positives.

    This test validates:
    - Identical titles (sim=1.0) ARE collapsed
    - Reworded titles (sim~0.2) are NOT collapsed (correct behavior)
    """
    section("Test 3: TF-IDF Threshold Calibration (0.85 = strict near-exact)")

    params = ClusteringParams(dedup_title_threshold=0.85, dedup_body_threshold=0.70)

    # Case A: Identical titles -- SHOULD be deduped (sim=1.0)
    identical_title = "CrowdStrike Q3 earnings revenue rises 8 percent beats expectations"
    a1 = make_article("https://src-a.com/cs", identical_title, source_name="SourceA", hours_ago=5)
    a2 = make_article("https://src-b.com/cs", identical_title, source_name="SourceB", hours_ago=4)

    # Case B: Reworded titles -- should NOT be deduped (sim~0.22, well below 0.85)
    # These are legitimately different framings of related news
    rw1 = make_article("https://src-c.com/story1",
                       "CrowdStrike reports major cybersecurity breach affecting Fortune 500 companies",
                       source_name="SourceC", hours_ago=3)
    rw2 = make_article("https://src-d.com/story2",
                       "CrowdStrike discloses major security breach impacting Fortune 500 firms",
                       source_name="SourceD", hours_ago=2)

    raw = [a1, a2, rw1, rw2]
    result = dedup_articles(raw, params)

    output_titles = [a.title for a in result.articles]
    identical_count = sum(1 for t in output_titles if "earnings" in t and "CrowdStrike" in t)
    breach_count = sum(1 for t in output_titles if "breach" in t and "CrowdStrike" in t)

    # Compute actual similarity for reporting
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(strip_accents="unicode", analyzer="word",
                              ngram_range=(1, 2), min_df=1, max_features=10000)
        mat = vec.fit_transform([rw1.title, rw2.title])
        reworded_sim = cosine_similarity(mat)[0, 1]
        identical_sim = 1.0  # same string
    except Exception:
        reworded_sim = 0.0
        identical_sim = 1.0

    print(f"\n    TF-IDF similarity scores:")
    print(f"      Identical titles: {identical_sim:.3f}  (threshold=0.85) -> should be deduped")
    print(f"      Reworded titles:  {reworded_sim:.3f}  (threshold=0.85) -> should NOT be deduped")

    check("Identical titles (sim=1.0) correctly collapsed to 1",
          identical_count == 1,
          f"found {identical_count} earnings articles (expected 1)")
    check("Reworded titles (sim~0.22) correctly kept both (no false positive)",
          breach_count == 2,
          f"found {breach_count} breach articles (expected 2 -- reworded, not duplicates)")
    check("Total output == 3 (1 deduped + 2 reworded)",
          len(result.articles) == 3,
          f"output has {len(result.articles)} articles (expected 3)")


# ---- Test 4: No false positives (distinct articles must be kept) ------------

def test_no_false_positives() -> None:
    section("Test 4: No False Positives -- Distinct Articles Kept")

    params = ClusteringParams(dedup_title_threshold=0.85, dedup_body_threshold=0.70)

    # 3 clearly distinct articles on different topics
    distinct_articles = [
        make_article(
            "https://reuters.com/tech/nvidia-blackwell",
            "NVIDIA Blackwell GPU architecture sets new AI training benchmark record",
            summary="NVIDIA's Blackwell GPU achieves 10x performance improvement for AI model training.",
            source_name="Reuters",
            hours_ago=12,
        ),
        make_article(
            "https://bloomberg.com/pharma/roche-drug",
            "Roche wins FDA approval for new Alzheimer's disease treatment drug",
            summary="The FDA granted approval to Roche's monoclonal antibody drug for Alzheimer's treatment.",
            source_name="Bloomberg",
            hours_ago=11,
        ),
        make_article(
            "https://ft.com/energy/bp-solar",
            "BP announces $10 billion solar energy investment across Southeast Asia",
            summary="BP will deploy solar farms across Vietnam, Thailand and Indonesia in major renewable push.",
            source_name="Financial Times",
            hours_ago=10,
        ),
    ]

    raw = list(distinct_articles)
    result = dedup_articles(raw, params)

    check("All 3 distinct articles kept",
          len(result.articles) == 3,
          f"output has {len(result.articles)} articles (expected 3)")
    check("Zero removed (all distinct)",
          result.removed_count == 0,
          f"removed_count={result.removed_count}")
    check("NVIDIA article present",
          any("NVIDIA" in a.title for a in result.articles))
    check("Roche article present",
          any("Roche" in a.title for a in result.articles))
    check("BP article present",
          any("BP" in a.title for a in result.articles))


# ---- Test 5a: Same URL, different title (known dedup gap) -------------------

def test_same_url_different_title() -> None:
    """BUG DETECTION: dedup_articles() relies purely on TF-IDF cosine similarity.
    It has no explicit URL equality pre-pass. When two RSS feeds publish the SAME
    URL but with slightly different title truncations, TF-IDF similarity may be
    < 0.85 and they survive dedup -- leaving duplicate URLs in the output.

    Real example found in 72h run:
      https://economictimes.indiatimes.com/industry/transportation/airlines
        - "Mid-East war: Air India adds flights on 9 routes"          (Economic Times)
        - "Air India adds 78 flights on 9 international routes..."    (ET Industry)
      TF-IDF sim ~ 0.35 -- below 0.85 threshold -- NOT deduped.

    This test confirms the gap and documents it as a known issue.
    """
    section("Test 5a: BUG -- Same URL Different Title (known dedup gap)")

    params = ClusteringParams(dedup_title_threshold=0.85, dedup_body_threshold=0.70)

    # Simulate the exact Economic Times / ET Industry duplicate pattern
    url = "https://economictimes.indiatimes.com/industry/transportation/airlines"
    a1 = make_article(url,
                      "Mid-East war: Air India adds flights on 9 routes",
                      source_name="Economic Times", hours_ago=5)
    a2 = make_article(url,
                      "Air India adds 78 flights on 9 international routes amid Middle East disruption",
                      source_name="ET Industry", hours_ago=4)
    a3 = make_article("https://reuters.com/unrelated",
                      "Unrelated article about something completely different",
                      source_name="Reuters", hours_ago=3)

    raw = [a1, a2, a3]
    result = dedup_articles(raw, params)
    output_urls = [a.url for a in result.articles]

    same_url_count = output_urls.count(url)
    url_unique = len(set(output_urls)) == len(output_urls)

    # Compute actual similarity to document the gap
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(strip_accents="unicode", analyzer="word",
                              ngram_range=(1, 2), min_df=1, max_features=10000)
        mat = vec.fit_transform([a1.title, a2.title])
        actual_sim = cosine_similarity(mat)[0, 1]
    except Exception:
        actual_sim = -1.0

    print(f"\n    TF-IDF similarity of same-URL pair: {actual_sim:.3f} (threshold=0.85)")
    print(f"    Same URL appears {same_url_count}x in output (expected: 1 if bug is fixed)")
    print(f"    BUG STATUS: {'NOT FIXED -- URL appears twice' if same_url_count > 1 else 'FIXED'}")
    print(f"    ROOT CAUSE: dedup_articles() has no URL equality pre-pass.")
    print(f"    FIX NEEDED: Add O(n) URL-hash dedup BEFORE TF-IDF cosine passes.")

    # We check what SHOULD be true (URL uniqueness), but mark it with [BUG] note
    check("[BUG] Same URL with different titles: URL uniqueness in output",
          url_unique,
          f"URL appears {same_url_count}x -- dedup missing URL equality pre-pass")


# ---- Test 5: Protected invariants (math assertions) --------------------------

def test_math_invariants(raw: List[RawArticle], result: DedupResult, label: str = "synthetic") -> None:
    section(f"Test 5: Protected Math Invariants ({label} data)")

    # I1: output count <= input count
    check("I1: output count <= input count",
          len(result.articles) <= len(raw),
          f"output={len(result.articles)}, input={len(raw)}")

    # I2: no article ID appears twice in output
    ids = [a.id for a in result.articles]
    check("I2: no duplicate IDs in output",
          len(ids) == len(set(ids)),
          f"{len(ids)} ids, {len(set(ids))} unique")

    # I3: URL uniqueness
    # NOTE: This may fail on real data due to a known bug:
    # dedup_articles() has no URL equality pre-pass. Two RSS feeds can emit the
    # same URL with slightly different title strings; TF-IDF sim < 0.85 → not deduped.
    # BUG: requires URL-hash pre-pass before TF-IDF passes. (Test 5a documents this.)
    urls = [a.url for a in result.articles]
    n_dup_urls = len(urls) - len(set(urls))
    check("I3: URL uniqueness -- no two articles share the same URL",
          len(urls) == len(set(urls)),
          f"{len(urls)} urls, {len(set(urls))} unique"
          + (f" -- {n_dup_urls} duplicates (KNOWN BUG: no URL pre-pass)" if n_dup_urls else ""))

    # I4: DedupResult assertion fields
    check("I4: assertion_count_non_increasing == True",
          result.assertion_count_non_increasing)
    check("I5: assertion_threshold_respected == True",
          result.assertion_threshold_respected)


# ---- Test 6: Real 72h data -- statistics ------------------------------------

async def test_real_data_stats() -> Tuple[List[RawArticle], DedupResult]:
    section("Test 6: Real 72h Data Statistics")

    print("  Fetching real articles (72h window)... this may take 15-30 seconds")

    scope = DiscoveryScope(
        companies=["Infosys", "TCS", "HDFC Bank", "Reliance"],
        region="IN",
        hours=72,
        mock_mode=False,
    )
    params = DEFAULT_PARAMS

    raw_articles: List[RawArticle] = []
    try:
        raw_articles = await fetch_articles(scope, params)
        print(f"  Raw articles fetched: {len(raw_articles)}")
    except Exception as e:
        print(f"  Fetch failed: {e} -- using synthetic dataset for invariant tests")

    if not raw_articles:
        print("  No articles fetched -- generating synthetic fallback dataset")
        raw_articles = _generate_synthetic_dataset(200)

    raw_count = len(raw_articles)
    result = dedup_articles(raw_articles, DEFAULT_PARAMS)
    deduped_count = len(result.articles)
    removed = result.removed_count
    removal_rate = (removed / raw_count * 100) if raw_count > 0 else 0.0

    print(f"\n  -- Statistics --")
    print(f"  Raw articles:     {raw_count}")
    print(f"  Deduped articles: {deduped_count}")
    print(f"  Removed:          {removed}")
    print(f"  Removal rate:     {removal_rate:.1f}%")
    print(f"  Dedup pairs:      {len(result.dedup_pairs)}")

    print(f"\n  -- Sample Dedup Pairs (first 5) --")
    for i, (kept_url, removed_url) in enumerate(result.dedup_pairs[:5], 1):
        print(f"  [{i}] KEPT:    {kept_url[:80]}")
        print(f"       REMOVED: {removed_url[:80]}")
        print()

    # Source breakdown of what was fetched
    from collections import Counter
    source_counts = Counter(a.source_name for a in raw_articles)
    print(f"  -- Top 5 Sources --")
    for src, cnt in source_counts.most_common(5):
        print(f"    {src[:40]:40s} {cnt:4d} articles")

    check("Removal rate is reasonable (0-80%)",
          0 <= removal_rate <= 80,
          f"{removal_rate:.1f}%")
    check("At least some articles survived dedup",
          deduped_count > 0,
          f"deduped_count={deduped_count}")

    return raw_articles, result


def _generate_synthetic_dataset(n: int) -> List[RawArticle]:
    """Generate n synthetic RawArticle records for invariant testing."""
    import random
    import string

    articles = []
    base_titles = [
        "NVIDIA announces new AI chip for enterprise data centers",
        "Microsoft expands cloud infrastructure in Asia Pacific region",
        "Tesla reports record EV deliveries in third quarter earnings",
        "Amazon Web Services launches new machine learning platform",
        "Google introduces Gemini AI model for enterprise customers",
        "Apple unveils next generation iPhone with AI capabilities",
        "Meta reports strong advertising revenue growth this quarter",
        "OpenAI secures funding for next generation AI research",
        "Salesforce acquires data analytics startup for expansion",
        "Oracle announces new database cloud service features",
    ]
    sources = ["Reuters", "Bloomberg", "TechCrunch", "Forbes", "CNBC",
               "FT", "WSJ", "The Hindu", "Economic Times", "LiveMint"]

    for i in range(n):
        # Mix: ~30% exact duplicates (same title, different URL), ~70% unique
        if i % 3 == 0 and i > 0:
            base_idx = random.randint(0, min(i - 1, len(base_titles) - 1))
            title = base_titles[base_idx % len(base_titles)]
            url = f"https://{sources[i % len(sources)].lower().replace(' ', '')}.com/article-{base_idx}"
        else:
            uid = "".join(random.choices(string.ascii_lowercase, k=6))
            title = f"{base_titles[i % len(base_titles)]} - {uid}"
            url = f"https://news-{i}.example.com/article-{i}"

        articles.append(RawArticle(
            url=url,
            title=title,
            summary=f"Summary for {title}. " * 3,
            source_name=sources[i % len(sources)],
            source_url=url,
            published_at=datetime.now(timezone.utc) - timedelta(hours=i * 0.5),
            fetch_method="synthetic",
        ))

    return articles


# ---- Test 7: Inject known duplicates + known distinct articles ---------------

def test_injection() -> None:
    section("Test 7: Injection Test -- Known Duplicates + Known Distinct")

    params = ClusteringParams(dedup_title_threshold=0.85, dedup_body_threshold=0.70)

    # 3 known exact duplicates to inject (same title, same story, different sources)
    dup_base = "Reliance Industries announces partnership with Meta for digital commerce"
    known_dups = [
        make_article("https://src-a.com/reliance-meta", dup_base,
                     source_name="SourceA", hours_ago=5),
        make_article("https://src-b.com/reliance-meta", dup_base,
                     source_name="SourceB", hours_ago=4),
        make_article("https://src-c.com/reliance-meta", dup_base,
                     source_name="SourceC", hours_ago=3),
    ]

    # 3 clearly distinct articles (must NOT be removed)
    known_distinct = [
        make_article("https://nyt.com/ai-regulation",
                     "EU passes landmark artificial intelligence regulation act",
                     summary="European Union lawmakers approved comprehensive AI rules for enterprise software.",
                     source_name="NYT", hours_ago=6),
        make_article("https://wsj.com/jpmorgan-layoffs",
                     "JPMorgan Chase announces 10000 workforce reduction amid automation push",
                     summary="JPMorgan is cutting staff as AI automates back-office banking operations.",
                     source_name="WSJ", hours_ago=7),
        make_article("https://bbc.com/climate-deal",
                     "G7 nations agree on new carbon emission reduction targets for industry",
                     summary="G7 summit concluded with binding commitments on industrial carbon reduction.",
                     source_name="BBC", hours_ago=8),
    ]

    raw = known_dups + known_distinct
    result = dedup_articles(raw, params)

    output_titles = [a.title for a in result.articles]

    # Verify all 3 known duplicates collapsed to 1
    reliance_count = sum(1 for t in output_titles if "Reliance" in t)
    check("All 3 known duplicates collapsed to 1",
          reliance_count == 1,
          f"found {reliance_count} Reliance articles (expected 1)")

    # Verify all 3 distinct articles kept
    eu_kept = any("EU" in t or "artificial intelligence regulation" in t for t in output_titles)
    jpmorgan_kept = any("JPMorgan" in t for t in output_titles)
    g7_kept = any("G7" in t or "carbon" in t for t in output_titles)

    check("Distinct: EU AI regulation kept", eu_kept, "EU article must survive dedup")
    check("Distinct: JPMorgan layoffs kept", jpmorgan_kept, "JPMorgan article must survive")
    check("Distinct: G7 climate deal kept", g7_kept, "G7 article must survive")

    check("Total output == 4 (1 deduped + 3 distinct)",
          len(result.articles) == 4,
          f"output has {len(result.articles)} articles (expected 4)")
    check("Removed == 2 (2 of the 3 dups removed)",
          result.removed_count == 2,
          f"removed_count={result.removed_count}")


# ---- Summary ----------------------------------------------------------------

def print_summary() -> None:
    section("SUMMARY")
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"  Total checks: {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {failed}")
    print()

    if failed:
        print("  Failed checks:")
        for label, ok, detail in _results:
            if not ok:
                print(f"    - {label}" + (f" ({detail})" if detail else ""))
        print()

    overall = PASS_LABEL if failed == 0 else FAIL_LABEL
    print(f"  Overall result: {overall}")
    print()


# ---- Main -------------------------------------------------------------------

async def main() -> None:
    print("\n" + "="*60)
    print("  DEDUP STANDALONE TEST SUITE")
    print("  Testing: app/intelligence/fetch.py::dedup_articles()")
    print("="*60)

    # Unit tests (pure synthetic -- fast, deterministic)
    test_exact_url_dedup()
    test_title_near_dedup()
    test_tfidf_threshold_behavior()
    test_no_false_positives()
    test_same_url_different_title()
    test_injection()

    # Synthetic invariant test (200 articles)
    synthetic_raw = _generate_synthetic_dataset(200)
    synthetic_result = dedup_articles(synthetic_raw, DEFAULT_PARAMS)
    test_math_invariants(synthetic_raw, synthetic_result, label="synthetic-200")

    # Real data test (network I/O -- fetches from RSS/Tavily)
    real_raw, real_result = await test_real_data_stats()

    # Protected invariants on real data
    section("Test 8: Protected Invariants on Real Data Output")
    test_math_invariants(real_raw, real_result, label="real-72h")

    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
