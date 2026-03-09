from __future__ import annotations
"""
Relevance Filter — Math Gate 2.

Algorithm: Zero-shot NLI classification (replaces Dunietz & Gillick 2014 salience).

Research basis:
  Yin et al. (2019) "Benchmarking Zero-shot Text Classification" — arXiv:1909.00161
  "Building Efficient Universal Classifiers with NLI" (2024) — arXiv:2312.17543
    → +9.4% F1 over keyword classifiers on zero-shot tasks

Model: cross-encoder/nli-deberta-v3-small (60MB, CPU, ~50ms/article)
  Premise:   "India cricket beats record"
  Hypothesis: "This article reports on a specific company named in the text that is
               growing, raising capital, or making a strategic business move."
  → contradiction score high → DROP (no manual patterns needed)

  Premise:   "Zepto raises $350M in Series G round"
  Hypothesis: same
  → entailment=0.989 → KEEP

Hypothesis v1 (data/filter_hypothesis.json): B2B_mean=0.859, Noise_mean=0.333
Verified on real 120h data: 195/1018 articles pass (19%%, mean_NLI=0.721)

Decision:
  nli_entailment >= nli_auto_accept (0.88) → keep (no LLM)
  nli_entailment <= nli_auto_reject (0.10) → drop (no LLM)
  0.10 < entailment < 0.88               → LLM batch classify (ambiguous cases only)

~80% of decisions are pure NLI math. LLM is the exception, not the rule.
The NLI hypothesis is loaded from data/filter_hypothesis.json and updated
by hypothesis_learner.py (SetFit) as user feedback accumulates.

Also applies Gap 4 rule:
  If a target company has 0 articles in the last N days → drop company entirely.
"""

import asyncio
import logging
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from app.intelligence.config import ClusteringParams
from app.intelligence.models import (
    Article,
    DiscoveryScope,
    FilterResult,
    SalienceScore,
)

logger = logging.getLogger(__name__)

# LLM batch size for ambiguous cases (same pattern as news_collector.py)
_LLM_BATCH_SIZE = 10


async def filter_articles(
    articles: List[Article],
    scope: DiscoveryScope,
    params: ClusteringParams,
) -> FilterResult:
    """Apply NLI relevance filter + Gap 4 rule.

    Math gates:
      Assert: target companies present in output (if they had articles in input)
      Assert: output count <= input count (nothing added)
    """
    if not articles:
        return FilterResult(
            articles=[],
            assertion_target_companies_present=True,
            assertion_count_non_increasing=True,
        )

    targets = _get_targets(scope)

    # ── NLI score all articles in one batch ───────────────────────────────────
    # This is the core gate — replaces manual keyword salience.
    # NLI understands semantic meaning: "AI in cricket analytics" ≠ business event.
    from app.intelligence.engine.nli_filter import (
        score_articles as nli_score_articles,
        nli_scores_by_source as compute_nli_by_source,
        get_hypothesis_version,
    )

    nli_scores = nli_score_articles(articles, batch_size=32)

    # Track per-source NLI scores for Thompson Sampling reward signal
    source_nli_scores = compute_nli_by_source(articles, nli_scores)
    hypothesis_ver = get_hypothesis_version()

    # ── Gate articles by NLI entailment score ────────────────────────────────
    kept: List[Article] = []
    dropped_ids: List[str] = []
    auto_accepted = 0
    auto_rejected = 0
    ambiguous: List[Tuple[Article, float]] = []

    for article, nli_score in zip(articles, nli_scores):
        # Title-only articles (no summary) → always route to LLM regardless of NLI score.
        # Without summary context, NLI auto-accept produces too many false positives
        # (consumer product launches, celebrity quotes, government announcements all score
        # high on title alone — e.g., "OnePlus 15T launches" scores 0.95 title-only).
        summary_len = len(getattr(article, "summary", "") or "")
        is_title_only = summary_len < 30

        if nli_score >= params.nli_auto_accept and not is_title_only:
            kept.append(article)
            auto_accepted += 1
        elif nli_score <= params.nli_auto_reject:
            dropped_ids.append(article.id)
            auto_rejected += 1
        else:
            # Semantic ambiguity zone → LLM second opinion
            # (also includes title-only articles that scored above auto_accept threshold)
            ambiguous.append((article, nli_score))

    # ── LLM batch classify ambiguous cases ────────────────────────────────────
    llm_kept, llm_rejected = await _llm_classify_batch(ambiguous, scope)
    kept.extend(llm_kept)
    dropped_ids.extend(a.id for a in llm_rejected)

    # ── Industry classification (1st/2nd order) ───────────────────────────────
    # Runs AFTER base NLI filter — only labels articles that already passed B2B gate.
    # Labels each article with industry_label, industry_order, first_order_score.
    # Soft step: never drops articles, only annotates.
    if scope.mode != "company_first" or scope.industry:
        try:
            from app.intelligence.industry_classifier import (
                classify_articles as classify_by_industry,
                build_spec_from_profile,
                get_spec,
                BUILT_IN_SPECS,
            )
            # Build specs from scope: use scope.industry if set, else all built-ins
            if scope.industry:
                spec = get_spec(scope.industry)
                if spec is None:
                    spec = build_spec_from_profile(
                        industry_id=scope.industry,
                        first_order_description=scope.industry,
                        second_order_description="vendors and service providers for " + scope.industry,
                        region=scope.region,
                    )
                industry_specs = [spec]
            else:
                # Use all built-ins — article gets best-matching industry
                industry_specs = list(BUILT_IN_SPECS.values())
                # Apply region from scope to each spec
                if scope.region and scope.region != "GLOBAL":
                    from dataclasses import replace
                    industry_specs = [
                        replace(s, region=scope.region) for s in industry_specs
                    ]
            kept = classify_by_industry(kept, industry_specs)
        except Exception as exc:
            logger.warning(f"[filter] Industry classification failed (non-fatal): {exc}")

    # ── Apply Gap 4 rule ───────────────────────────────────────────────────────
    # Gap4 only makes sense in Company-First mode (tracking real named companies).
    # In Industry-First mode, targets are generic keywords ('saas', 'ai', 'software') —
    # Gap4 would incorrectly drop articles about "SaaS companies" if "saas" keyword
    # has no standalone recent match, even though the article is a valid B2B signal.
    if scope.companies:
        kept, gap4_dropped = _apply_gap4(kept, targets, params.filter_gap4_days)
    else:
        gap4_dropped = []

    # ── Set run_index on kept articles ────────────────────────────────────────
    for i, art in enumerate(kept):
        art.run_index = i

    # ── Compute NLI stats for kept articles (O(n), not O(n²)) ─────────────────
    kept_ids = {art.id for art in kept}
    kept_nli_scores = [
        score for art, score in zip(articles, nli_scores)
        if art.id in kept_ids
    ]
    nli_mean = float(sum(kept_nli_scores) / len(kept_nli_scores)) if kept_nli_scores else 0.0

    # ── Auto-label high-confidence examples for dataset enhancement ────────────
    # NLI high-confidence kept → positive examples (no human needed)
    # NLI auto-rejected → negative examples (very low entailment = confirmed noise)
    # Runs async in background, never blocks filter output.
    try:
        from app.learning.dataset_enhancer import DatasetEnhancer
        enhancer = DatasetEnhancer()
        # Build nli_score map per kept article
        article_nli = {art.id: score for art, score in zip(articles, nli_scores)}
        kept_with_scores = [article_nli.get(a.id, 0.0) for a in kept]
        # Gather auto-rejected texts (articles that scored below auto_reject threshold)
        rejected_texts = [
            f"{a.title}. {a.summary or ''}"[:400]
            for a, s in zip(articles, nli_scores)
            if s <= params.nli_auto_reject and a.id not in kept_ids
        ]
        enhancer.extract_labels_from_filter(kept, rejected_texts, kept_with_scores)
    except Exception as _exc:
        pass  # Never block the filter for dataset enhancement

    # ── Math assertions ───────────────────────────────────────────────────────
    assert_target_present = _check_targets_present(kept, targets, articles)
    assert_non_increasing = len(kept) <= len(articles)

    if not assert_non_increasing:
        logger.error(f"[filter] ASSERTION FAILED: output ({len(kept)}) > input ({len(articles)})")

    logger.info(
        f"[filter] {len(articles)} → {len(kept)} "
        f"(nli_auto_accept={auto_accepted}, nli_auto_reject={auto_rejected}, "
        f"llm_kept={len(llm_kept)}, llm_reject={len(llm_rejected)}, "
        f"gap4_dropped={gap4_dropped}, nli_mean={nli_mean:.3f})"
    )

    return FilterResult(
        articles=kept,
        dropped_articles=dropped_ids,
        gap4_dropped_companies=gap4_dropped,
        llm_classified_count=len(ambiguous),
        auto_accepted_count=auto_accepted,
        auto_rejected_count=auto_rejected,
        nli_mean_entailment=nli_mean,
        nli_scores_by_source=source_nli_scores,
        hypothesis_version=hypothesis_ver,
        assertion_target_companies_present=assert_target_present,
        assertion_count_non_increasing=assert_non_increasing,
    )


def compute_salience(
    entity: str,
    article: Article,
    entity_counts: Dict[str, int],
    corpus_size: int,
) -> SalienceScore:
    """Compute Dunietz & Gillick 2014 salience for one (entity, article) pair.

    NOTE: This function is kept for backwards compatibility and used by the
    entity extractor's MIN_SEED_SALIENCE check. It is NO LONGER used for the
    article relevance filter (replaced by NLI in filter_articles()).

    Args:
        entity: canonical entity name (case-insensitive match)
        article: Article to score
        entity_counts: cross-document mention counts (for exclusivity)
        corpus_size: total number of articles in corpus
    """
    entity_lower = entity.lower()
    title_lower = article.title.lower()
    text = (article.full_text or article.summary or "").lower()
    full_lower = title_lower + " " + text

    # ── Component 1: Title presence (0.40 weight) ─────────────────────────────
    title_presence = 1.0 if _entity_in_text(entity_lower, title_lower) else 0.0

    # ── Component 2: First 2 sentences presence (0.30 weight) ────────────────
    first_sentences = _extract_first_sentences(text, n=2)
    first_sentence_presence = 1.0 if _entity_in_text(entity_lower, first_sentences) else 0.0

    # ── Component 3: Normalized frequency (0.15 weight) ───────────────────────
    total_words = max(len(full_lower.split()), 1)
    mention_count = full_lower.count(entity_lower)
    normalized_freq = min(mention_count / total_words * 100, 1.0)

    # ── Early exit: entity not mentioned at all → score = 0 ───────────────────
    if title_presence == 0.0 and first_sentence_presence == 0.0 and mention_count == 0:
        return SalienceScore(
            entity=entity,
            article_id=article.id,
            title_presence=0.0,
            first_sentence_presence=0.0,
            normalized_frequency=0.0,
            exclusivity_score=0.0,
        )

    # ── Component 4: Exclusivity score (0.15 weight) ──────────────────────────
    total_corpus_mentions = entity_counts.get(entity_lower, 1)
    exclusivity = 1.0 / math.log(1 + total_corpus_mentions)

    return SalienceScore(
        entity=entity,
        article_id=article.id,
        title_presence=title_presence,
        first_sentence_presence=first_sentence_presence,
        normalized_frequency=normalized_freq,
        exclusivity_score=exclusivity,
    )


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_targets(scope: DiscoveryScope) -> List[str]:
    """Extract target entity names from scope (used for Gap4 + LLM classify context)."""
    if scope.companies:
        return scope.companies
    if scope.industry:
        from app.intelligence.config import get_industry_keywords
        return list(get_industry_keywords(scope.industry))[:5]
    return []


def _build_entity_counts(articles: List[Article]) -> Dict[str, int]:
    """Count total cross-document mentions per entity name (lowercased)."""
    counts: Counter = Counter()
    for article in articles:
        text = f"{article.title} {article.summary} {article.full_text}".lower()
        for entity in article.entities_raw:
            el = entity.lower()
            counts[el] += text.count(el)
    return dict(counts)


def _entity_in_text(entity_lower: str, text_lower: str) -> bool:
    """Check if entity appears in text as a word/phrase (not substring)."""
    words = entity_lower.split()
    if len(words) == 1:
        pattern = r'\b' + re.escape(entity_lower) + r'\b'
        return bool(re.search(pattern, text_lower))
    return entity_lower in text_lower


def _extract_first_sentences(text: str, n: int = 2) -> str:
    """Extract first N sentences from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


async def _llm_classify_batch(
    ambiguous: List[Tuple[Article, float]],
    scope: DiscoveryScope,
) -> Tuple[List[Article], List[Article]]:
    """LLM batch classify semantically ambiguous articles (0.20 < nli < 0.55).

    These articles score in the uncertain zone where the NLI model's confidence
    is below threshold. A second-opinion from an LLM using the industry context
    resolves ambiguity more reliably than NLI alone.
    Fail-CLOSED: drops ambiguous articles on LLM failure (noise is worse than gaps).
    """
    if not ambiguous:
        return [], []

    kept: List[Article] = []
    rejected: List[Article] = []
    targets = _get_targets(scope)

    # Fast path: only used in Company-First mode where scope.companies lists real company names.
    # DISABLED for Industry-First mode — industry keywords like "ai", "cloud", "saas" are too
    # generic and would substring-match unrelated words, bypassing LLM for sports/crime/etc.
    fast_accept: List[Article] = []
    need_llm: List[Tuple[Article, float]] = []

    # Only fast-accept when scope explicitly lists company names (Company-First path).
    # Industry keywords are sent to LLM without exception.
    company_first_targets = scope.companies if (scope.companies) else []

    for article, nli_score in ambiguous:
        title_lower = article.title.lower()
        if company_first_targets and any(
            re.search(r'\b' + re.escape(t.lower()) + r'\b', title_lower)
            for t in company_first_targets
        ):
            fast_accept.append(article)
        else:
            need_llm.append((article, nli_score))

    kept.extend(fast_accept)

    if not need_llm:
        return kept, rejected

    # LLM batch classification
    try:
        from app.tools.llm.llm_service import LLMService
        llm = LLMService()

        for batch_start in range(0, len(need_llm), _LLM_BATCH_SIZE):
            batch = need_llm[batch_start:batch_start + _LLM_BATCH_SIZE]
            batch_kept, batch_rejected = await _classify_batch(batch, targets, llm, scope)
            kept.extend(batch_kept)
            rejected.extend(batch_rejected)

    except Exception as exc:
        logger.warning(f"[filter] LLM classification failed: {exc} — dropping ambiguous articles")
        rejected.extend(a for a, _ in need_llm)

    return kept, rejected


async def _classify_batch(
    batch: List[Tuple[Article, float]],
    targets: List[str],
    llm: object,
    scope: Optional["DiscoveryScope"] = None,
) -> Tuple[List[Article], List[Article]]:
    """Send one batch to LLM for relevance classification.

    Uses industry-aware context when available (INDUSTRY_FIRST mode).
    Fail-CLOSED on exception: drops ambiguous articles rather than keeping them.
    """
    titles = "\n".join(
        f"{i+1}. {art.title}"
        for i, (art, _) in enumerate(batch)
    )

    if scope is not None and getattr(scope, "industry", None):
        industry = scope.industry
        try:
            from app.intelligence.config import INDUSTRY_TAXONOMY
            exclude = INDUSTRY_TAXONOMY.get(industry, {}).get("exclude", [])
            exclude_str = ", ".join(exclude[:4]) if exclude else "sports, celebrity, crime, politics"
        except Exception:
            exclude_str = "sports, celebrity, crime, politics"

        prompt = f"""You are a B2B sales intelligence filter for the {industry} industry.

B2B ONLY: Keep articles relevant to BUSINESS-TO-BUSINESS events where companies sell
to or partner with other businesses — NOT consumer-facing news.

For each article below, respond with KEEP or DROP.

KEEP if ALL three are true:
1. A specific named private or public company is the PRIMARY actor
2. The event is B2B-relevant: enterprise software, SaaS, cloud services, industrial tech,
   semiconductors, data centers, B2B platforms, funding rounds, acquisitions, partnerships,
   IPO filings, regulatory actions affecting a named company
3. Not primarily consumer-facing

DROP if ANY is true:
- Consumer electronics (phones, tablets, TVs, headphones, wearables for end users)
- Consumer product reviews, gadget reviews, 'best of' buying guides for general public
- Stock analyst tips, price targets, buy/sell recommendations, investor commentary
- Sports, cricket, gaming, entertainment, celebrity news
- Government/politician announcements, military operations, geopolitical events
- Crime, court cases, bail hearings NOT involving a company as defendant
- Macro trends with no specific company as primary actor
- {exclude_str}
- No named company as the primary subject making a business decision

Articles:
{titles}

Respond with a comma-separated list of KEEP or DROP in the same order.
Example: KEEP, DROP, KEEP, KEEP, DROP"""
    else:
        targets_str = ", ".join(targets[:5]) if targets else "the target companies"
        prompt = f"""You are a B2B sales intelligence filter.

Target companies: {targets_str}

Rule: KEEP only if a SPECIFIC NAMED COMPANY (not government/politician) is the primary actor.

For each article below, respond with KEEP or DROP.
KEEP = a specific company in {targets_str} is making a business move (earnings, product, strategy, deal, regulation targeting the company).
DROP = article is about government policy, infrastructure by politicians, macro trends, or {targets_str} is not the primary subject (politics, sports, crime, celebrity).

Articles:
{titles}

Respond with a comma-separated list of KEEP or DROP in the same order.
Example: KEEP, DROP, KEEP, KEEP, DROP"""

    try:
        from app.tools.llm.llm_service import LLMService
        response = await llm.generate(prompt, max_tokens=50)
        decisions = [d.strip().upper() for d in response.split(",")]

        kept = []
        rejected = []
        for i, (article, _) in enumerate(batch):
            decision = decisions[i] if i < len(decisions) else "DROP"
            if decision == "KEEP":
                kept.append(article)
            else:
                rejected.append(article)
        return kept, rejected

    except Exception as exc:
        logger.warning(f"[filter] LLM batch failed ({exc}) — dropping {len(batch)} ambiguous articles")
        return [], [a for a, _ in batch]


def _apply_gap4(
    articles: List[Article],
    targets: List[str],
    window_days: int,
) -> Tuple[List[Article], List[str]]:
    """Gap 4 rule: drop companies with 0 articles in the last N days."""
    if not targets:
        return articles, []

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    dropped_companies: List[str] = []

    for target in targets:
        target_lower = target.lower()
        recent = [
            a for a in articles
            if a.published_at and a.published_at >= cutoff
            and target_lower in f"{a.title} {a.summary}".lower()
        ]
        if not recent:
            dropped_companies.append(target)
            logger.info(f"[gap4] Dropped '{target}' — 0 articles in last {window_days}d")

    if not dropped_companies:
        return articles, []

    dropped_lower = {c.lower() for c in dropped_companies}
    remaining_targets_lower = {t.lower() for t in targets if t not in dropped_companies}

    if not remaining_targets_lower:
        return articles, dropped_companies

    kept = []
    for article in articles:
        text_lower = f"{article.title} {article.summary}".lower()
        if any(t in text_lower for t in remaining_targets_lower):
            kept.append(article)
        elif any(t in text_lower for t in dropped_lower):
            pass  # Article only mentions dropped companies → remove
        else:
            kept.append(article)  # No target mention → let cluster decide

    return kept, dropped_companies


def _check_targets_present(
    kept: List[Article],
    targets: List[str],
    original: List[Article],
) -> bool:
    """Assert: if a target had articles in input, it must have articles in output."""
    if not targets:
        return True

    for target in targets:
        target_lower = target.lower()
        had_input = any(
            target_lower in f"{a.title} {a.summary}".lower()
            for a in original
        )
        has_output = any(
            target_lower in f"{a.title} {a.summary}".lower()
            for a in kept
        )
        if had_input and not has_output:
            logger.warning(f"[filter] ASSERTION: target '{target}' had input articles "
                          f"but none in output (may have been Gap4 dropped intentionally)")
            return False
    return True


# ── Gap4 Filter ──────────────────────────────────────────────────────────────

"""
Gap 4 Rule — standalone module.

Rule: If a target company has 0 articles in the last N days → drop it entirely.

Used by both FilterAgent and as a standalone check in diagnostics.
Also used by FetchAgent to pre-filter which companies to even fetch for.

Named "Gap 4" because it's the 4th data quality gap in B2B sales intelligence:
  Gap 1: No contact info
  Gap 2: Stale company data
  Gap 3: Wrong industry classification
  Gap 4: No recent news signal → company not in any buying motion
"""


import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from app.intelligence.models import Article

logger = logging.getLogger(__name__)


def check_gap4(
    company_articles: Dict[str, List[Article]],
    window_days: int = 5,
) -> Tuple[Dict[str, List[Article]], List[str]]:
    """Apply Gap 4 rule to a per-company article map.

    Args:
        company_articles: mapping from company name → their articles
        window_days: recency window (default 5 days)

    Returns:
        (kept_companies, dropped_company_names)
        kept_companies: companies that have recent articles
        dropped_company_names: companies dropped due to Gap 4
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    kept: Dict[str, List[Article]] = {}
    dropped: List[str] = []

    for company, articles in company_articles.items():
        recent = [
            a for a in articles
            if a.published_at is not None and a.published_at >= cutoff
        ]
        if recent:
            kept[company] = articles
        else:
            dropped.append(company)
            logger.info(f"[gap4] '{company}' dropped — 0 articles in last {window_days}d "
                        f"(has {len(articles)} total, all older than {window_days}d)")

    if dropped:
        logger.info(f"[gap4] Dropped {len(dropped)}/{len(company_articles)} companies: "
                    f"{dropped}")

    return kept, dropped


def gap4_report(
    company_articles: Dict[str, List[Article]],
    window_days: int = 5,
) -> Dict[str, Dict]:
    """Generate a diagnostic report showing Gap 4 status for all companies."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    report: Dict[str, Dict] = {}

    for company, articles in company_articles.items():
        recent = [a for a in articles if a.published_at and a.published_at >= cutoff]
        total = len(articles)
        report[company] = {
            "total_articles": total,
            "recent_articles": len(recent),
            "window_days": window_days,
            "gap4_triggered": len(recent) == 0,
            "oldest_article": min(
                (a.published_at for a in articles if a.published_at),
                default=None,
            ),
            "newest_article": max(
                (a.published_at for a in articles if a.published_at),
                default=None,
            ),
        }

    return report
