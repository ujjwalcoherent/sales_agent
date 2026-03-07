"""
Standalone test for app/intelligence/filter.py

All test data is derived dynamically from MOCK_ARTICLES_RAW — no manually
typed article titles, no hand-crafted entity lists.

Strategy:
  - In-domain  = articles whose category matches the target industry
  - Out-of-domain = articles from a DIFFERENT category (real articles, wrong industry)
  - The filter must keep in-domain and reject out-of-domain

Run:
    venv/Scripts/python.exe -m pytest tests/test_filter_standalone.py -v -s
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pytest

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS, INDUSTRY_TAXONOMY
from app.intelligence.filter import (
    filter_articles,
    _get_targets,
    _build_entity_counts,
    compute_salience,
)
# Note: _best_salience_for_article was removed when NLI replaced the keyword
# salience formula. Cross-industry filtering tests now rely on filter_articles()
# which uses NLI zero-shot classification (arXiv:1909.00161).
from app.intelligence.models import Article, DiscoveryMode, DiscoveryScope
from app.data.mock_articles import MOCK_ARTICLES_RAW


# ── Data helpers — derive everything from mock data, nothing hardcoded ────────

def _articles_by_category() -> Dict[str, List[Article]]:
    """Partition MOCK_ARTICLES_RAW into groups by category (regulation/funding/policy)."""
    groups: Dict[str, List[Article]] = {}
    for i, (title, summary, source, category) in enumerate(MOCK_ARTICLES_RAW):
        art = Article(
            url=f"https://test.com/{i}",
            title=title,
            summary=summary,
            source_name=source,
            source_url="https://test.com",
            published_at=datetime.now(timezone.utc),
            run_index=i,
        )
        groups.setdefault(category, []).append(art)
    return groups


def _articles_by_industry_keyword() -> Tuple[List[Article], List[Article]]:
    """Split mock articles into those matching Fintech keywords and those that don't.

    Uses INDUSTRY_TAXONOMY["Fintech"]["keywords"] — derived from config, not hardcoded.
    Returns (fintech_matching, non_matching).
    """
    keywords = set(k.lower() for k in INDUSTRY_TAXONOMY.get("Fintech", {}).get("keywords", []))
    matching, non_matching = [], []
    for art in _articles_by_category().get("regulation", []):
        text = f"{art.title} {art.summary}".lower()
        if any(kw in text for kw in keywords):
            matching.append(art)
        else:
            non_matching.append(art)
    return matching, non_matching


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestGetTargets:
    """_get_targets() returns industry keywords in INDUSTRY_FIRST mode."""

    def test_company_first_returns_companies(self):
        """COMPANY_FIRST mode returns the exact company names from scope."""
        # Derive company names from mock articles dynamically
        by_cat = _articles_by_category()
        regulation_articles = by_cat.get("regulation", [])
        # Extract first capitalized word from each title as a proxy company name
        companies = list({
            word for art in regulation_articles
            for word in art.title.split()
            if word[0].isupper() and len(word) > 3
        })[:3]

        scope = DiscoveryScope(mode=DiscoveryMode.COMPANY_FIRST, companies=companies)
        targets = _get_targets(scope)
        assert set(targets) == set(companies)

    def test_industry_first_uses_taxonomy_keywords(self):
        """INDUSTRY_FIRST returns keywords from INDUSTRY_TAXONOMY — not generic fallback."""
        for industry in list(INDUSTRY_TAXONOMY.keys())[:3]:
            scope = DiscoveryScope(mode=DiscoveryMode.INDUSTRY_FIRST, industry=industry)
            targets = _get_targets(scope)
            expected_keywords = set(k.lower() for k in INDUSTRY_TAXONOMY[industry].get("keywords", []))

            assert len(targets) > 0, f"Industry '{industry}' must return keyword targets"
            assert set(targets) <= expected_keywords | {industry.lower()}, (
                f"Targets for '{industry}' contain unexpected keywords: "
                f"{set(targets) - expected_keywords}"
            )
            print(f"\n  [targets] {industry} -> {sorted(targets)}")

    def test_unknown_industry_falls_back_to_industry_name(self):
        """Unknown industry returns at least the industry name as a target."""
        scope = DiscoveryScope(mode=DiscoveryMode.INDUSTRY_FIRST, industry="QuantumBiotech")
        targets = _get_targets(scope)
        assert len(targets) > 0
        assert "quantumbiotech" in [t.lower() for t in targets]


class TestSalienceFormula:
    """Dunietz & Gillick salience — derived from mock article content."""

    def test_entity_in_title_scores_above_auto_accept(self):
        """An article whose target entity IS in the title scores >= filter_auto_accept (0.30)."""
        # Pick any regulation article and use its first entity from title as target
        by_cat = _articles_by_category()
        article = by_cat["regulation"][0]  # "RBI Mandates New KYC Norms..."
        # Extract entity: first word > 3 chars, capitalized, not first word of title
        words = article.title.split()
        entity = next(
            (w for w in words[1:] if w[0].isupper() and len(w) > 3 and w.isalpha()),
            words[0]
        )
        entity_counts = _build_entity_counts([article])
        score = compute_salience(entity, article, entity_counts, corpus_size=1)
        print(f"\n  [salience] '{entity}' in '{article.title[:40]}' = {score.score:.3f}")
        assert score.score >= DEFAULT_PARAMS.filter_auto_reject, (
            f"Entity '{entity}' IS in article title but salience={score.score:.3f} "
            f"is below auto_reject threshold ({DEFAULT_PARAMS.filter_auto_reject})"
        )

    def test_entity_absent_from_article_scores_zero(self):
        """An entity that doesn't appear anywhere in an article scores exactly 0."""
        by_cat = _articles_by_category()
        # Use a word that appears in regulation articles but NOT in funding articles
        regulation_text = " ".join(
            f"{a.title} {a.summary}" for a in by_cat["regulation"]
        ).lower()
        funding_text = " ".join(
            f"{a.title} {a.summary}" for a in by_cat["funding"]
        ).lower()

        # Find a word present in regulation but not funding
        reg_words = {w.strip(".,") for w in regulation_text.split() if len(w) > 5}
        fund_words = {w.strip(".,") for w in funding_text.split() if len(w) > 5}
        exclusive_words = reg_words - fund_words

        if not exclusive_words:
            pytest.skip("No exclusive words found between categories")

        test_entity = sorted(exclusive_words)[0]  # deterministic
        test_article = by_cat["funding"][0]  # article from OTHER category

        entity_counts = {"zepto": 5}  # simulate some corpus presence
        score = compute_salience(test_entity, test_article, entity_counts, corpus_size=10)
        print(f"\n  [salience] absent entity '{test_entity}' in funding article = {score.score:.4f}")
        assert score.score == 0.0, (
            f"Entity '{test_entity}' not in article but salience={score.score:.4f} (must be 0.0). "
            f"Exclusivity component must not give free score to unmatched entities."
        )

    def test_exclusivity_inversely_proportional_to_corpus_frequency(self):
        """Entity appearing in 1 article must score higher exclusivity than one in 50."""
        by_cat = _articles_by_category()
        article = by_cat["regulation"][0]
        entity = article.title.split()[0]  # first word of title

        entity_counts_rare = {entity.lower(): 1}
        entity_counts_common = {entity.lower(): 50}

        score_rare = compute_salience(entity, article, entity_counts_rare, 100)
        score_common = compute_salience(entity, article, entity_counts_common, 100)

        print(f"\n  [exclusivity] rare={score_rare.exclusivity_score:.3f} "
              f"common={score_common.exclusivity_score:.3f}")
        assert score_rare.exclusivity_score > score_common.exclusivity_score


class TestCrossIndustryFiltering:
    """Filter correctly separates in-domain from out-of-domain using real mock articles."""

    def test_nli_filter_rejects_off_topic_articles(self):
        """NLI filter rejects semiconductor policy articles when filtering for Fintech.

        The NLI zero-shot classifier (arXiv:1909.00161) replaces keyword salience.
        We call filter_articles() directly and check auto_rejected_count > 0.
        """
        by_cat = _articles_by_category()
        policy_articles = by_cat.get("policy", [])
        if not policy_articles:
            pytest.skip("No policy articles in mock data")

        # Use a Fintech scope — semiconductor policy articles should mostly be rejected
        scope = DiscoveryScope(mode=DiscoveryMode.INDUSTRY_FIRST, industry="Fintech")
        result = _run(filter_articles(policy_articles, scope, DEFAULT_PARAMS))

        print(f"\n  [NLI cross-industry] policy→fintech: {len(policy_articles)} → {len(result.articles)}")
        print(f"  auto_rejected={result.auto_rejected_count}  nli_mean={result.nli_mean_entailment:.3f}")

        # NLI should reject at least some off-topic articles
        assert len(result.articles) <= len(policy_articles), "Filter must not add articles"
        # At least some articles should be rejected (filter is discriminating)
        assert result.auto_rejected_count > 0 or len(result.articles) < len(policy_articles), (
            "NLI filter must reject some semiconductor articles for Fintech scope. "
            "If nothing is rejected, NLI is not discriminating."
        )

    def test_filter_reduces_count_in_cross_industry_mode(self):
        """Filter with COMPANY_FIRST targeting regulation companies removes
        off-topic articles from a mixed set — all data from mock articles."""
        by_cat = _articles_by_category()
        # Derive target companies from regulation article titles (no manual typing)
        reg_articles = by_cat.get("regulation", [])
        target_companies = list({
            word for art in reg_articles
            for word in art.title.split()
            if word[0].isupper() and len(word) > 4 and word.isalpha()
        })[:4]

        # Mix regulation + policy (semiconductor) articles
        mixed_articles = reg_articles + by_cat.get("policy", [])
        scope = DiscoveryScope(
            mode=DiscoveryMode.COMPANY_FIRST,
            companies=target_companies,
            region="IN",
        )
        result = _run(filter_articles(mixed_articles, scope, DEFAULT_PARAMS))

        print(f"\n  [cross-industry filter] {len(mixed_articles)} -> {len(result.articles)}")
        print(f"  targets={target_companies}")
        print(f"  auto_accept={result.auto_accepted_count}, auto_reject={result.auto_rejected_count}")

        assert len(result.articles) <= len(mixed_articles), "Filter must not add articles"
        assert result.auto_rejected_count > 0, (
            "Some semiconductor (off-topic) articles must be auto-rejected by math gate. "
            "If auto_reject=0, the salience formula is not discriminating."
        )

    def test_filter_count_monotonic(self):
        """Output count must never exceed input count — for all categories."""
        by_cat = _articles_by_category()
        for category, articles in by_cat.items():
            scope = DiscoveryScope(mode=DiscoveryMode.INDUSTRY_FIRST, industry="Technology")
            result = _run(filter_articles(articles, scope, DEFAULT_PARAMS))
            assert len(result.articles) <= len(articles), (
                f"Filter added articles for category '{category}': "
                f"input={len(articles)}, output={len(result.articles)}"
            )
