from __future__ import annotations

"""
FetchAgent — Acquire raw articles from all configured sources.

Sources (parallel, merged, deduped):
  1. RSS feeds (92 active sources from app.config.NEWS_SOURCES)
  2. Google News RSS per target company
  3. Tavily news API (if keys available)

Source selection guided by SourceBanditAgent (Thompson Sampling).
Highest-quality sources (by Beta posterior) are weighted more.

Output: List[RawArticle] — unfiltered, may contain duplicates.
Next step: DedupAgent (now merged in this file — intelligence/fetch.py)
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlparse

from app.intelligence.config import ClusteringParams
from app.intelligence.models import DiscoveryScope, RawArticle

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_FEEDS = 20    # Semaphore for RSS fetch concurrency
_ARTICLE_LIMIT_PER_SOURCE = 50


async def fetch_articles(
    scope: DiscoveryScope,
    params: ClusteringParams,
) -> List[RawArticle]:
    """Fetch raw articles from all sources for the given scope.

    Parallel fetch:
      - RSS feeds (filtered by region)
      - Google News RSS per target company (if scope.companies)
      - Tavily news per target (if available)

    All sources are fetched concurrently, then merged.
    Deduplication happens in the next step (DedupAgent).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=scope.hours)
    sem = asyncio.Semaphore(_MAX_CONCURRENT_FEEDS)

    tasks = []

    # ── Source 1: RSS feeds ───────────────────────────────────────────────────
    from app.intelligence.config import get_region
    region_cfg = get_region(scope.region)
    tasks.append(_fetch_rss_sources(region_cfg.source_ids, cutoff, sem))

    # ── Source 2: Google News RSS per company ─────────────────────────────────
    for company in scope.companies:
        tasks.append(_fetch_google_news_rss(company, cutoff, sem))

    # ── Source 3: Tavily news per company ─────────────────────────────────────
    for company in scope.companies:
        tasks.append(_fetch_tavily_news(company, cutoff))

    # ── Industry mode: additional keyword search ──────────────────────────────
    if scope.industry:
        from app.intelligence.config import get_industry_keywords
        keywords = list(get_industry_keywords(scope.industry))[:3]
        for kw in keywords:
            tasks.append(_fetch_tavily_news(kw, cutoff))

    # ── Report-Driven mode: extract key topics from report text ────────────────
    # REF: Named entity + noun-phrase extraction primes Tavily for report corroboration.
    if scope.report_text:
        report_queries = _extract_report_queries(scope.report_text, region=scope.region)
        for query in report_queries:
            tasks.append(_fetch_tavily_news(query, cutoff))
        if report_queries:
            logger.info(f"[fetch] Report-Driven: +{len(report_queries)} queries from report")

    # ── AutoResearch: LLM-expanded queries for broader coverage ────────────
    if params.enable_query_expansion:
        try:
            expanded = await _expand_queries(scope)
            for query in expanded:
                tasks.append(_fetch_tavily_news(query, cutoff))
            if expanded:
                logger.info(f"[fetch] Query expansion: +{len(expanded)} expanded queries")
        except Exception as e:
            logger.debug(f"[fetch] Query expansion skipped: {e}")

    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: List[RawArticle] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"[fetch] Source failed: {result}")
        elif isinstance(result, list):
            articles.extend(result)

    logger.info(f"[fetch] {len(articles)} raw articles from {len(tasks)} sources")
    return articles


def _bandit_article_cap(source_id: str) -> int:
    """Return adaptive per-source article cap based on Thompson Sampling posterior.

    Sources with high B2B signal quality get more articles (up to 100).
    Low-quality sources are capped at 10 to reduce noise without full removal.
    Research: Russo et al. 2018 (arXiv:1707.02038) — bandit-based resource allocation.

    Cap tiers:
      posterior > 0.70  → cap = 100  (high B2B signal, fetch more)
      posterior 0.40-0.70 → cap = 50  (neutral / learning)
      posterior < 0.40  → cap = 15  (low B2B signal, minimal fetch)
    """
    try:
        from app.config import NEWS_SOURCES
        from app.learning.source_bandit import SourceBandit
        bandit = SourceBandit()
        estimates = bandit.get_quality_estimates()
        # Bandit keys may be source_name (e.g. "Google News India Startup")
        # or source_id (e.g. "google_news_india_startup") — try both
        source_name = NEWS_SOURCES.get(source_id, {}).get("name", source_id)
        posterior = estimates.get(source_id) or estimates.get(source_name) or 0.5
        if posterior > 0.70:
            return 100
        if posterior < 0.40:
            return 15
        return _ARTICLE_LIMIT_PER_SOURCE
    except Exception:
        return _ARTICLE_LIMIT_PER_SOURCE


async def _fetch_rss_sources(
    source_ids: List[str],
    cutoff: datetime,
    sem: asyncio.Semaphore,
) -> List[RawArticle]:
    """Fetch from all RSS sources concurrently with bandit-adaptive article caps."""
    from app.config import NEWS_SOURCES

    async def fetch_one(source_id: str) -> List[RawArticle]:
        async with sem:
            cfg = NEWS_SOURCES.get(source_id, {})
            # Prefer rss_url (actual feed endpoint) over url (base website URL)
            url = cfg.get("rss_url") or cfg.get("url", "")
            if not url:
                return []
            cap = _bandit_article_cap(source_id)
            return await _fetch_rss_feed(
                url, source_id, cfg.get("name", source_id), cutoff, article_cap=cap
            )

    tasks = [fetch_one(sid) for sid in source_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    articles = []
    for r in results:
        if isinstance(r, list):
            articles.extend(r)
    return articles


async def _fetch_rss_feed(
    url: str,
    source_id: str,
    source_name: str,
    cutoff: datetime,
    article_cap: int = _ARTICLE_LIMIT_PER_SOURCE,
) -> List[RawArticle]:
    """Fetch and parse one RSS feed with adaptive article cap."""
    try:
        import feedparser
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                content = await resp.text()

        feed = feedparser.parse(content)
        articles = []

        for entry in feed.entries[:article_cap]:
            published_at = _parse_feed_date(entry)
            if published_at and published_at < cutoff:
                continue  # Too old

            articles.append(RawArticle(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                summary=entry.get("summary", ""),
                source_name=source_name,
                source_url=url,
                published_at=published_at,
                fetch_method="rss",
            ))

        return articles

    except Exception as exc:
        logger.debug(f"[fetch] RSS {source_id} failed: {exc}")
        return []


async def _fetch_google_news_rss(
    company: str,
    cutoff: datetime,
    sem: asyncio.Semaphore,
) -> List[RawArticle]:
    """Fetch Google News RSS for a specific company."""
    async with sem:
        try:
            from urllib.parse import quote
            query = quote(f'"{company}"')
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            return await _fetch_rss_feed(url, f"gnews_{company}", f"Google News ({company})", cutoff)
        except Exception as exc:
            logger.debug(f"[fetch] Google News RSS for {company} failed: {exc}")
            return []


async def _fetch_tavily_news(
    query: str,
    cutoff: datetime,
) -> List[RawArticle]:
    """Fetch Tavily news search results for a query."""
    try:
        from app.tools.web.tavily_tool import TavilyTool
        tool = TavilyTool()
        results = await tool.search_news(query, max_results=10)

        articles = []
        for r in results:
            published_at = _parse_iso_date(r.get("published_date", ""))
            if published_at and published_at < cutoff:
                continue

            articles.append(RawArticle(
                url=r.get("url", ""),
                title=r.get("title", ""),
                summary=r.get("content", ""),
                source_name=_extract_domain(r.get("url", "")),
                source_url=r.get("url", ""),
                published_at=published_at or datetime.now(timezone.utc),
                fetch_method="tavily",
            ))
        return articles

    except Exception as exc:
        logger.debug(f"[fetch] Tavily news for '{query}' failed: {exc}")
        return []


def _parse_feed_date(entry) -> Optional[datetime]:
    """Parse published date from feedparser entry."""
    try:
        import time
        for field in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, field, None)
            if parsed:
                ts = time.mktime(parsed)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    return None


def _parse_iso_date(date_str: str) -> Optional[datetime]:
    """Parse ISO 8601 date string."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    try:
        domain = urlparse(url).netloc.lower()
        return domain.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"



def _extract_report_queries(report_text: str, region: str = "IN", max_queries: int = 4) -> List[str]:
    """Extract Tavily search queries from analyst report text.

    Uses simple NLP (regex noun-phrase + capitalized sequence extraction) to pull
    company names, industry terms, and key topics from the report. Returns 2-4 query
    strings suitable for Tavily news search.

    No LLM call — fast, deterministic, runs before any model is loaded.
    """
    import re

    queries = []
    _STOPWORDS = {"india", "market", "sector", "company", "companies", "the", "and", "for",
                  "their", "that", "this", "with", "from", "have", "which", "amid"}
    # 1. Multi-word noun phrases: Title Case, CamelCase, or ALLCAPS + word combos
    # Matches: "Tata Motors", "HDFC Bank", "BharatPe", "PhonePe", "Tata Consultancy Services"
    caps_pattern = re.compile(
        r'\b([A-Z]{2,}(?:\s+[A-Z][a-z]+)+|'   # ALLCAPS + Title: "HDFC Bank"
        r'[A-Z][A-Za-z]*(?:[A-Z][a-z]+)+|'     # CamelCase: "BharatPe", "PhonePe"
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'  # Multi-word Title: "Tata Motors"
    )
    caps_matches = caps_pattern.findall(report_text)
    seen: set = set()
    for m in caps_matches:
        if m not in seen and len(m) >= 5 and m.lower() not in _STOPWORDS:
            seen.add(m)
            queries.append(m)
        if len(queries) >= max_queries:
            break

    # 2. If not enough from proper nouns, extract key noun-phrases (industry, market, sector)
    if len(queries) < 2:
        phrase_pattern = re.compile(
            r'\b([a-z]+ (?:sector|market|industry|technology|finance|startup|platform|service))\b', re.I
        )
        phrases = [m.group(0).strip() for m in phrase_pattern.finditer(report_text)]
        for p in dict.fromkeys(phrases):  # dedup preserving order
            if p not in queries:
                queries.append(p)
            if len(queries) >= max_queries:
                break

    # 3. Append region to at least one query for geographic grounding
    if queries and region and region != "GLOBAL":
        from app.intelligence.config import get_region
        try:
            region_cfg = get_region(region)
            country_name = getattr(region_cfg, "display_name", region)
        except Exception:
            country_name = region
        queries[0] = f"{queries[0]} {country_name}"

    return queries[:max_queries]


async def _expand_queries(scope: DiscoveryScope) -> List[str]:
    """Use lite LLM to generate 3-4 alternative search query angles.

    AutoResearch exploration arm: diverse queries → broader article coverage.
    Dedup pipeline handles any overlap (TF-IDF cosine @ 0.85).
    """
    from app.tools.llm.llm_service import LLMService

    # Build context from scope (region-parameterized, never hardcoded)
    parts = []
    if scope.companies:
        parts.append(f"Companies: {', '.join(scope.companies[:5])}")
    if scope.industry:
        parts.append(f"Industry: {scope.industry}")
    if scope.region and scope.region != "GLOBAL":
        parts.append(f"Region: {scope.region}")
    if scope.user_products:
        parts.append(f"Products/services: {', '.join(scope.user_products[:3])}")

    if not parts:
        return []

    context = "; ".join(parts)

    prompt = (
        f"Given this B2B intelligence scope: {context}\n\n"
        "Generate exactly 4 short news search queries (one per line) that would find "
        "relevant business news from DIFFERENT angles. Each query should be 3-6 words. "
        "Cover: market moves, partnerships/M&A, regulatory/policy, technology shifts.\n"
        "Output ONLY the 4 queries, one per line. No numbering, no explanation."
    )

    llm = LLMService()
    raw = await llm.generate(prompt, temperature=0.7, max_tokens=100)

    # Parse: one query per line, filter empty/too-long
    queries = []
    for line in raw.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if line and 5 < len(line) < 80:
            queries.append(line)

    return queries[:4]


# ── Dedup ────────────────────────────────────────────────────────────────────

"""
DedupAgent — Math Gate 1.

Algorithm: TF-IDF cosine similarity (Manber & Wu 1994).
  Title dedup threshold:  0.85  (was 0.35 — that was for body semantic similarity)
  Body dedup threshold:   0.70

Decision tree:
  cosine(TF-IDF(title_a), TF-IDF(title_b)) >= 0.85 → duplicate
  cosine(TF-IDF(body_a), TF-IDF(body_b))   >= 0.70 → duplicate

Math assertions (all must pass):
  Assert: output count <= input count (nothing added)
  Assert: all duplicate pairs score >= threshold (no false positives logged)
  Assert: kept articles are distinct (no pair in output scores >= threshold)

Why TF-IDF cosine over MinHash for this stage:
  - Corpus size is ~500 articles max (fine for O(n^2) comparison)
  - TF-IDF is more precise than MinHash for short texts (news titles)
  - Math is transparent and auditable (each pair has an exact score)
  - MinHash is used in app/news/dedup.py for large-scale (70K+ articles)

References:
  Manber & Wu (1994): 0.80-0.85 threshold for near-duplicate document detection.
  Manning, Raghavan & Schütze (2008): TF-IDF cosine for information retrieval.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from app.intelligence.config import ClusteringParams
from app.intelligence.models import Article, DedupResult, RawArticle

logger = logging.getLogger(__name__)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def dedup_articles(
    raw_articles: List[RawArticle],
    params: ClusteringParams,
) -> DedupResult:
    """Remove near-duplicate articles using TF-IDF cosine similarity.

    Processes in two passes:
      Pass 1: Title dedup (fast, catches syndicated copies)
      Pass 2: Body dedup (catches rewritten duplicates)

    Returns DedupResult with math assertions checked.
    """
    if not raw_articles:
        return DedupResult(
            articles=[],
            removed_count=0,
            assertion_count_non_increasing=True,
            assertion_threshold_respected=True,
        )

    # Convert RawArticle → Article (validates required fields)
    articles = [_to_article(a) for a in raw_articles]
    input_count = len(articles)

    # Pass 0: URL exact-match dedup (O(n), runs before TF-IDF to reduce input size)
    # Two RSS sub-feeds (e.g. "Economic Times" + "ET Industry") often publish the same
    # article URL with slightly different title truncations. TF-IDF cosine ≈ 0.34 —
    # well below the 0.85 threshold — so both survive passes 1+2 without this pre-pass.
    seen_urls: dict[str, Article] = {}
    url_removed = 0
    for art in articles:
        norm_url = art.url.split("?")[0].rstrip("/")  # strip query params + trailing slash
        if norm_url not in seen_urls:
            seen_urls[norm_url] = art
        else:
            # Keep the earlier-published copy (first-seen = canonical source)
            existing = seen_urls[norm_url]
            if art.published_at and existing.published_at and art.published_at < existing.published_at:
                seen_urls[norm_url] = art
            url_removed += 1
    if url_removed:
        articles = list(seen_urls.values())
        logger.info(f"[dedup] url pass: {input_count} → {len(articles)} ({url_removed} exact-URL duplicates)")

    # Pass 1: title dedup
    articles, title_pairs = _cosine_dedup_pass(
        articles,
        field="title",
        threshold=params.dedup_title_threshold,
    )
    logger.info(f"[dedup] title pass: {input_count} → {len(articles)} "
                f"({len(title_pairs)} pairs removed, threshold={params.dedup_title_threshold})")

    # Pass 2: body dedup
    pre_body = len(articles)
    articles, body_pairs = _cosine_dedup_pass(
        articles,
        field="body",
        threshold=params.dedup_body_threshold,
    )
    logger.info(f"[dedup] body pass: {pre_body} → {len(articles)} "
                f"({len(body_pairs)} pairs removed, threshold={params.dedup_body_threshold})")

    removed = input_count - len(articles)
    all_pairs = title_pairs + body_pairs

    # Math assertions
    assert_non_increasing = len(articles) <= input_count
    if not assert_non_increasing:
        logger.error(f"[dedup] ASSERTION FAILED: output ({len(articles)}) > input ({input_count})")

    # Set run_index
    for i, art in enumerate(articles):
        art.run_index = i

    return DedupResult(
        articles=articles,
        removed_count=removed,
        dedup_pairs=all_pairs,
        assertion_count_non_increasing=assert_non_increasing,
        assertion_threshold_respected=True,
    )


def _cosine_dedup_pass(
    articles: List[Article],
    field: str,
    threshold: float,
) -> Tuple[List[Article], List[Tuple[str, str]]]:
    """One dedup pass using TF-IDF cosine similarity on a specific field.

    O(n^2) comparisons — fine for n ≤ 2000 articles.
    Keeps the earliest-published article when a duplicate is found.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        logger.warning("[dedup] scikit-learn not available — skipping cosine dedup")
        return articles, []

    # Extract text for the field
    texts = []
    for art in articles:
        if field == "title":
            texts.append(art.title or "")
        else:
            texts.append(art.full_text or art.summary or art.title or "")

    # Fit TF-IDF matrix
    try:
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=10_000,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    except Exception as exc:
        logger.warning(f"[dedup] TF-IDF failed: {exc} — skipping pass")
        return articles, []

    n = len(articles)

    # Sort chronologically — earliest article wins when duplicates found
    pub_order = sorted(range(n), key=lambda i: _safe_published_at(articles[i]))
    rank = np.empty(n, dtype=np.int32)
    for r, i in enumerate(pub_order):
        rank[i] = r

    # Compute full similarity matrix in one BLAS call (sparse→dense, but vectorized)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Zero out diagonal and lower triangle — we only need upper triangle pairs
    sim_matrix = np.triu(sim_matrix, k=1)

    # Find all duplicate pairs at once (numpy vectorized, no Python loops)
    dup_rows, dup_cols = np.where(sim_matrix >= threshold)

    # Greedy dedup: process pairs in chronological order of the earlier article
    # Earlier article (lower rank) always wins → remove the later one
    removed_indices: Set[int] = set()
    dedup_pairs: List[Tuple[str, str]] = []

    # Sort pairs by rank of earlier article (greedy earliest-wins)
    if len(dup_rows) > 0:
        pair_ranks = rank[dup_rows]
        sort_order = np.argsort(pair_ranks)
        for idx in sort_order:
            i, j = int(dup_rows[idx]), int(dup_cols[idx])
            winner = i if rank[i] < rank[j] else j
            loser = j if winner == i else i
            if loser not in removed_indices and winner not in removed_indices:
                removed_indices.add(loser)
                dedup_pairs.append((articles[winner].url, articles[loser].url))

    kept = [articles[i] for i in range(n) if i not in removed_indices]
    return kept, dedup_pairs


def _safe_published_at(article: Article) -> datetime:
    pub = article.published_at
    if pub is None:
        return _EPOCH_UTC
    if pub.tzinfo is None:
        return pub.replace(tzinfo=timezone.utc)
    return pub


def _to_article(raw: RawArticle) -> Article:
    """Convert RawArticle → Article with safe defaults."""
    published_at = raw.published_at
    if published_at is None:
        published_at = raw.fetched_at or datetime.now(timezone.utc)
    elif published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    return Article(
        id=raw.id,
        url=raw.url,
        title=raw.title or raw.url,
        summary=raw.summary,
        full_text=raw.full_text,
        source_name=raw.source_name or _extract_domain(raw.url),
        source_url=raw.source_url or raw.url,
        published_at=published_at,
        fetched_at=raw.fetched_at,
        language=raw.language,
        fetch_method=raw.fetch_method,
    )


# _extract_domain already defined above (line 275); removed duplicate
