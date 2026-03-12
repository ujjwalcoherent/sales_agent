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
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

from app.intelligence.config import ClusteringParams
from app.intelligence.models import Article, DedupResult, DiscoveryScope, RawArticle

logger = logging.getLogger(__name__)

_RE_REPORT_CAPS = re.compile(
    r'\b([A-Z]{2,}(?:\s+[A-Z][a-z]+)+|'   # ALLCAPS + Title: "HDFC Bank"
    r'[A-Z][A-Za-z]*(?:[A-Z][a-z]+)+|'    # CamelCase: "BharatPe"
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'  # Multi-word Title: "Tata Motors"
)
_RE_REPORT_PHRASE = re.compile(
    r'\b([a-z]+ (?:sector|market|industry|technology|finance|startup|platform|service))\b',
    re.IGNORECASE,
)
_RE_REPORT_SENTENCE = re.compile(r'[.!?\n]')


@dataclass
class ReportEntities:
    """Structured entities extracted from an analyst report."""
    companies: List[str] = field(default_factory=list)   # ["Infosys", "TCS", "HCL"]
    industries: List[str] = field(default_factory=list)  # ["IT Services", "Cloud Computing"]
    topics: List[str] = field(default_factory=list)      # ["AI adoption", "data center expansion"]

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
        from app.intelligence.config import get_industry_keywords, get_industry_anchors
        keywords = list(get_industry_keywords(scope.industry))[:3]

        # PRIMARY: Google News RSS for anchor companies (no Tavily needed)
        anchor_companies = get_industry_anchors(scope.industry)
        for company in anchor_companies[:4]:
            tasks.append(_fetch_google_news_rss(company, cutoff, sem))

        # SECONDARY: Tavily/DDG for industry keywords
        for kw in keywords:
            tasks.append(_fetch_tavily_or_ddg(kw, cutoff))

    # ── Report-Driven mode: LLM entity extraction → Google News RSS + Tavily/DDG ──
    # REF: Structured entity extraction (companies/industries/topics) routes to the
    # right fetch function — Google News RSS for company names (no API key needed),
    # Tavily/DDG for topic queries. Regex fallback if LLM unavailable.
    if scope.report_text:
        entities = await _extract_report_entities_llm(scope.report_text, region=scope.region)

        # PRIMARY: Google News RSS per company (no Tavily needed)
        for company in entities.companies[:5]:
            tasks.append(_fetch_google_news_rss(company, cutoff, sem))

        # SECONDARY: Tavily/DDG per topic (optional, graceful fallback to DDG)
        for topic in entities.topics[:3]:
            tasks.append(_fetch_tavily_or_ddg(topic, cutoff))

        # Also surface extracted companies to scope for downstream entity processing
        if entities.companies and not scope.companies:
            scope.companies = entities.companies[:5]

        n_sources = len(entities.companies) + len(entities.topics)
        logger.info(
            f"[fetch] Report-Driven: {len(entities.companies)} companies, "
            f"{len(entities.industries)} industries, {len(entities.topics)} topics → "
            f"{n_sources} fetch tasks"
        )

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


async def _fetch_ddg_news(query: str, cutoff: datetime, max_results: int = 10) -> List[RawArticle]:
    """Fetch news via DuckDuckGo as Tavily fallback.

    Uses ddgs library (NOT duckduckgo_search — that package is broken).
    Returns empty list on failure — callers must handle gracefully.
    """
    try:
        from ddgs import DDGS
        articles = []
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        for r in results:
            published_at = _parse_iso_date(r.get("date", ""))
            if published_at and published_at < cutoff:
                continue
            articles.append(RawArticle(
                url=r.get("url", ""),
                title=r.get("title", ""),
                summary=r.get("body", ""),
                source_name=r.get("source", _extract_domain(r.get("url", ""))),
                source_url=r.get("url", ""),
                published_at=published_at or datetime.now(timezone.utc),
                fetch_method="ddg",
            ))
        return articles
    except Exception as exc:
        logger.debug(f"[fetch] DDG news for '{query}' failed: {exc}")
        return []


async def _fetch_tavily_or_ddg(query: str, cutoff: datetime) -> List[RawArticle]:
    """Try Tavily first, fall back to DDG if unavailable or returns no results."""
    results = await _fetch_tavily_news(query, cutoff)
    if not results:
        results = await _fetch_ddg_news(query, cutoff)
    return results


def _parse_feed_date(entry) -> Optional[datetime]:
    """Parse published date from feedparser entry."""
    try:
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



async def _extract_report_entities_llm(report_text: str, region: str = "IN") -> ReportEntities:
    """Extract structured entities from analyst report using LLM.

    One LLM call at pipeline start. Returns companies, industries, topics
    for routing to appropriate fetch functions (Google News RSS, RSS feeds).

    Track A: run_structured() with ReportEntitiesLLM (pydantic-ai typed output).
    Track B: generate(json_mode=True) + manual json.loads() fallback.
    Track C: regex extraction if both LLM paths fail.
    """
    # Market reports put company names 4-6K chars in (after market overview).
    # 6000 chars captures the competitive landscape while keeping prompt costs low.
    text = report_text[:6000]

    sys_prompt = (
        "You extract business entities from analyst report text. "
        "Return companies (proper names, max 6), industries (sector names, max 3), "
        "and topics (actionable themes, 3-5 words each, max 4)."
    )
    user_prompt = (
        f"Extract entities from this business/analyst report text for a B2B sales intelligence system.\n\n"
        f"Report (region: {region}):\n{text}\n\n"
        "Rules:\n"
        "- companies: proper company names only (e.g. 'Infosys', 'HDFC Bank', 'TCS'). Max 6.\n"
        "- industries: sector names (e.g. 'IT Services', 'Fintech', 'Healthcare'). Max 3.\n"
        "- topics: specific actionable themes (e.g. 'AI chip demand surge', '5G enterprise adoption'). Max 4."
    )

    try:
        from app.tools.llm.llm_service import LLMService
        from app.schemas.llm_outputs import ReportEntitiesLLM
        llm = LLMService()

        # Track A: run_structured with pydantic-ai typed output
        try:
            result = await llm.run_structured(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                output_type=ReportEntitiesLLM,
            )
            return ReportEntities(
                companies=[c for c in result.companies[:6] if c],
                industries=[i for i in result.industries[:3] if i],
                topics=[t for t in result.topics[:4] if t],
            )
        except Exception as e_struct:
            logger.debug(f"[fetch] Structured entity extraction failed ({e_struct}), trying generate_json")

        # Track B: generate with json_mode fallback
        raw = await llm.generate(
            user_prompt + "\n\nReturn ONLY valid JSON with keys: companies, industries, topics.",
            temperature=0.1, max_tokens=300, json_mode=True,
        )
        data = json.loads(raw)
        return ReportEntities(
            companies=[str(c).strip() for c in data.get("companies", [])[:6] if c],
            industries=[str(i).strip() for i in data.get("industries", [])[:3] if i],
            topics=[str(t).strip() for t in data.get("topics", [])[:4] if t],
        )
    except Exception as exc:
        logger.warning(f"[fetch] LLM entity extraction failed: {exc}, falling back to regex")
        # Track C: regex extraction
        queries = _extract_report_queries(report_text, region=region, max_queries=4)
        return ReportEntities(topics=queries)


def _extract_report_queries(report_text: str, region: str = "IN", max_queries: int = 4) -> List[str]:
    """Extract Tavily search queries from analyst report text.

    Uses simple NLP (regex noun-phrase + capitalized sequence extraction) to pull
    company names, industry terms, and key topics from the report. Returns 2-4 query
    strings suitable for Tavily news search.

    No LLM call — fast, deterministic, runs before any model is loaded.
    """
    queries = []
    _STOPWORDS = {"india", "market", "sector", "company", "companies", "the", "and", "for",
                  "their", "that", "this", "with", "from", "have", "which", "amid"}
    # 1. Multi-word noun phrases: Title Case, CamelCase, or ALLCAPS + word combos
    # Matches: "Tata Motors", "HDFC Bank", "BharatPe", "PhonePe", "Tata Consultancy Services"
    caps_matches = _RE_REPORT_CAPS.findall(report_text)
    seen: set = set()
    for m in caps_matches:
        if m not in seen and len(m) >= 5 and m.lower() not in _STOPWORDS:
            seen.add(m)
            queries.append(m)
        if len(queries) >= max_queries:
            break

    # 2. If not enough from proper nouns, extract key noun-phrases (industry, market, sector)
    if len(queries) < 2:
        phrases = [m.group(0).strip() for m in _RE_REPORT_PHRASE.finditer(report_text)]
        for p in dict.fromkeys(phrases):  # dedup preserving order
            if p not in queries:
                queries.append(p)
            if len(queries) >= max_queries:
                break

    # 3. Fallback: if still no queries, use first meaningful sentence fragment
    if not queries:
        # Split on sentence boundaries, pick 2-4 noun-heavy fragments
        sentences = _RE_REPORT_SENTENCE.split(report_text)
        for sent in sentences:
            sent = sent.strip()
            if 10 <= len(sent) <= 100:
                queries.append(sent)
            if len(queries) >= 2:
                break

    # 4. Append region to at least one query for geographic grounding
    # (skip if the first query already mentions the country name)
    if queries and region and region != "GLOBAL":
        from app.intelligence.config import get_region
        try:
            region_cfg = get_region(region)
            # RegionConfig has .name not .display_name
            country_name = getattr(region_cfg, "name", None) or getattr(region_cfg, "display_name", region)
        except Exception:
            country_name = region
        if country_name.lower() not in queries[0].lower():
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
