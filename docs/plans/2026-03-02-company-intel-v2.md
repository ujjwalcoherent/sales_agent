# Company Intelligence v2 — Complete System Rewrite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken SearXNG-dependent company intelligence system with a validated, multi-source pipeline that extracts maximum data, finds trend-matched contacts (not just CEOs), generates hyper-personalized emails from scraped person context, and accumulates 5-7 months of company news.

**Architecture:** Three-layer design — (1) `web_intel.py` tool layer (ddgs search → trafilatura/Jina extract → ScrapeGraphAI deep enrichment), (2) `company_enricher.py` orchestration layer (Wikidata → Wikipedia → Apollo org → web scrape → LLM gap-fill, with entity validation to prevent "Temple" garbage), (3) upgraded agents (trend-aware contact matching, person-scraping for hyper-personalized outreach). All results validated before display. News accumulated in ChromaDB via background collector.

**Tech Stack:** ddgs (search), trafilatura (extraction), ScrapeGraphAI (deep enrichment), Google News RSS (news), feedparser, Wikidata/Wikipedia API, Apollo API, Hunter API, pydantic-ai, SQLite + ChromaDB.

---

## Dead Code Removal (Do First)

Before building anything new, clean house. These are confirmed dead/broken by code exploration:

| File | What to Remove | Why |
|------|---------------|-----|
| `database.py:188-223` | `LeadModel` table + `save_lead()` + `get_latest_leads()` | Never called by current pipeline |
| `domain_utils.py:186-212` | `normalize_domain()` | Never called anywhere |
| `domain_utils.py:214-245` | `extract_domains_from_text()` | Never called anywhere |
| `domain_utils.py:249-252` | Dead `patterns` list in `generate_email_pattern()` | Built but never used |
| `hunter_tool.py:249-252` | Same dead `patterns` list | Variable assigned, ignored |
| `wikidata.py:80` | `logo_url` field | Declared but never extracted (no P154 code) |
| `company_agent.py:272` | Stale "Tavily" comment | Tavily removed months ago |
| `companies.py:923-939` | `_save_mock_result()` | Writes to disk on every search — unnecessary I/O |
| `companies.py:338` | `_resolve_domain` import from `lead_gen.py` | Wrong direction — agent→router cross-import |

---

## Task 1: `app/tools/web_intel.py` — Unified Web Intelligence Tool

**Files:**
- Create: `app/tools/web_intel.py`
- Modify: `app/api/companies.py` (replace all SearXNG direct calls)
- Modify: `app/agents/workers/company_agent.py` (replace SearXNG search)

**What this replaces:** The 7 different SearXNG call sites scattered across the codebase. Every part of the system calls ONE module now.

**Design:**

```python
"""
Unified web intelligence — search, extract, news.

Replaces all direct SearXNG calls with a validated, multi-source pipeline.
Every caller in the system uses this module. No direct search engine calls elsewhere.

Architecture:
  search(query) → ddgs/SearXNG → URLs + snippets
  extract(url)  → trafilatura → Jina Reader fallback → raw content
  news(company) → Google News RSS → URLs → extract each
  deep_search(query) → ScrapeGraphAI SearchGraph → structured data (background only, 30-80s)
  deep_scrape(url, prompt) → ScrapeGraphAI SmartScraperGraph → structured extraction
"""

from pydantic import BaseModel
from typing import Optional
import asyncio, httpx, feedparser, json, logging, re, time
from ddgs import DDGS
import trafilatura

logger = logging.getLogger(__name__)

# ── Result Models ──────────────────────────────────────

class SearchResult(BaseModel):
    """Single search result with optional extracted content."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    content: str = ""          # Full extracted content (if extract=True)
    content_length: int = 0

class NewsArticle(BaseModel):
    """Single news article from RSS or search."""
    title: str = ""
    url: str = ""
    source_name: str = ""
    published_at: str = ""
    content: str = ""          # Full extracted content
    summary: str = ""          # First 2-3 sentences

class CompanyProfile(BaseModel):
    """Deep company profile from ScrapeGraphAI or structured extraction."""
    company_name: str = ""
    also_known_as: list[str] = []
    industry: str = ""
    sub_industries: list[str] = []
    description: str = ""
    founded_year: int | None = None
    headquarters: str = ""
    ceo: str = ""
    key_people: list[dict] = []    # [{"name": "...", "role": "..."}]
    employee_count: str = ""
    revenue: str = ""
    market_cap: str = ""
    stock_ticker: str = ""
    website: str = ""
    domain: str = ""
    products_services: list[str] = []
    competitors: list[str] = []
    subsidiaries: list[str] = []
    funding_stage: str = ""
    total_funding: str = ""
    investors: list[str] = []
    tech_stack: list[str] = []
    recent_events: list[str] = []
    sources: list[str] = []        # URLs where data was found


# ── Core Functions ─────────────────────────────────────

_SEARCH_SEM = asyncio.Semaphore(5)
_EXTRACT_SEM = asyncio.Semaphore(10)
_NEWS_SEM = asyncio.Semaphore(3)

async def search(
    query: str,
    max_results: int = 10,
    extract_content: bool = False,
    news_mode: bool = False,
    region: str = "wt-wt",
) -> list[SearchResult]:
    """Web search via ddgs → optional content extraction.

    Primary: ddgs (DuckDuckGo, multi-backend)
    Fallback: SearXNG (if ddgs returns 0 and SearXNG is running)

    Args:
        query: Search query
        max_results: Max results to return
        extract_content: If True, also extracts full page content via trafilatura
        news_mode: If True, uses ddgs.news() instead of ddgs.text()
        region: DuckDuckGo region code
    """
    async with _SEARCH_SEM:
        results = await asyncio.to_thread(_ddgs_search, query, max_results, news_mode, region)

    if not results:
        # Fallback to SearXNG if available
        results = await _searxng_fallback(query, max_results, news_mode)

    if extract_content and results:
        # Parallel extraction with bounded concurrency
        async def _extract_one(r):
            async with _EXTRACT_SEM:
                content = await extract(r.url)
                r.content = content
                r.content_length = len(content)
            return r
        results = await asyncio.gather(*[_extract_one(r) for r in results])

    return results


def _ddgs_search(query: str, max_results: int, news_mode: bool, region: str) -> list[SearchResult]:
    """Synchronous ddgs search (run in thread)."""
    try:
        ddgs = DDGS()
        if news_mode:
            raw = list(ddgs.news(query, max_results=max_results, region=region))
            return [SearchResult(
                title=r.get("title", ""),
                url=r.get("url", r.get("href", "")),
                snippet=r.get("body", ""),
                source=r.get("source", ""),
            ) for r in raw]
        else:
            raw = list(ddgs.text(query, max_results=max_results, region=region))
            return [SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ) for r in raw]
    except Exception as e:
        logger.warning(f"ddgs search failed: {e}")
        return []


async def _searxng_fallback(query: str, max_results: int, news_mode: bool) -> list[SearchResult]:
    """Fallback to SearXNG if ddgs fails."""
    try:
        from app.search.manager import SearchManager
        mgr = SearchManager()
        if news_mode:
            raw = await asyncio.wait_for(mgr.company_news_search(query, max_results=max_results), timeout=10)
        else:
            raw = await asyncio.wait_for(mgr.web_search(query, max_results=max_results), timeout=10)
        return [SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", r.get("snippet", "")),
        ) for r in raw.get("results", [])]
    except Exception as e:
        logger.debug(f"SearXNG fallback also failed: {e}")
        return []


async def extract(url: str, timeout: float = 15.0) -> str:
    """Extract clean content from a URL.

    Layer 1: trafilatura (fast, local, no API, F1=0.958)
    Layer 2: Jina Reader (handles JS-heavy pages)
    Layer 3: Empty string (graceful failure)
    """
    # Layer 1: trafilatura
    try:
        content = await asyncio.wait_for(
            asyncio.to_thread(_trafilatura_extract, url),
            timeout=timeout,
        )
        if content and len(content) > 100:
            return content
    except Exception:
        pass

    # Layer 2: Jina Reader
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"https://r.jina.ai/{url}", headers={"Accept": "text/plain"})
            if resp.status_code == 200 and len(resp.text) > 100:
                return resp.text[:50000]
    except Exception:
        pass

    return ""


def _trafilatura_extract(url: str) -> str:
    """Synchronous trafilatura extraction (run in thread)."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        result = trafilatura.extract(downloaded, output_format="txt", include_links=False)
        return result or ""
    return ""


async def company_news(
    company_name: str,
    max_articles: int = 20,
    time_range: str = "when:7d",
    region: str = "en-US",
    country: str = "US",
    extract_content: bool = True,
) -> list[NewsArticle]:
    """Fetch company news via Google News RSS → extract content.

    Google News RSS returns 100+ articles per query.
    Each article URL is then extracted via trafilatura for full content.
    """
    async with _NEWS_SEM:
        query = f"{company_name} {time_range}"
        feed_url = f"https://news.google.com/rss/search?q={query}&hl={region}&gl={country}&ceid={country}:{region.split('-')[0]}"

        feed = await asyncio.to_thread(feedparser.parse, feed_url)
        articles = []
        for entry in feed.entries[:max_articles]:
            source = ""
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title
            articles.append(NewsArticle(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name=source,
                published_at=entry.get("published", ""),
            ))

    # Extract content in parallel if requested
    if extract_content and articles:
        sem = asyncio.Semaphore(5)
        async def _extract_article(article):
            async with sem:
                try:
                    content = await asyncio.wait_for(extract(article.url), timeout=10)
                    article.content = content
                    # First 3 sentences as summary
                    sentences = content.split(". ")[:3]
                    article.summary = ". ".join(sentences) + "." if sentences else ""
                except Exception:
                    pass
            return article
        articles = await asyncio.gather(*[_extract_article(a) for a in articles])

    return articles


async def deep_company_search(
    query: str,
    prompt: str | None = None,
) -> CompanyProfile | None:
    """Deep company search via ScrapeGraphAI SearchGraph.

    WARNING: Takes 30-80 seconds. Use for BACKGROUND enrichment only.
    NOT for interactive/user-facing search.

    Uses OpenAI gpt-4.1-mini (cheapest viable model for ScrapeGraphAI).
    """
    try:
        import os
        from scrapegraphai.graphs import SearchGraph

        if not prompt:
            prompt = f"""Find ALL available information about "{query}". Return as JSON:
company_name, also_known_as (list), industry, sub_industries (list),
description (3-4 sentences), founded_year (int), headquarters,
ceo, key_people (list of name+role dicts), employee_count,
revenue, market_cap, stock_ticker, website, domain,
products_services (list), competitors (list), subsidiaries (list),
funding_stage, total_funding, investors (list), tech_stack (list),
recent_events (list of 3 recent headlines).
Use null for fields not found. Return ONLY business data."""

        config = {
            "llm": {
                "model": "openai/gpt-4.1-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.1,
            },
            "verbose": False,
            "headless": True,
        }

        result = await asyncio.to_thread(
            lambda: SearchGraph(prompt=prompt, config=config).run()
        )

        if isinstance(result, dict):
            data = result.get("content", result)
            return CompanyProfile(**{k: v for k, v in data.items() if k in CompanyProfile.model_fields and v is not None})

    except Exception as e:
        logger.warning(f"ScrapeGraphAI deep search failed for '{query}': {e}")

    return None


async def deep_scrape_url(
    url: str,
    prompt: str,
) -> dict | None:
    """Deep structured extraction from a URL via ScrapeGraphAI SmartScraperGraph.

    WARNING: Takes 10-60 seconds. Use for targeted extraction only.
    """
    try:
        import os
        from scrapegraphai.graphs import SmartScraperGraph

        config = {
            "llm": {
                "model": "openai/gpt-4.1-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.1,
            },
            "verbose": False,
            "headless": True,
        }

        result = await asyncio.to_thread(
            lambda: SmartScraperGraph(prompt=prompt, source=url, config=config).run()
        )

        if isinstance(result, dict):
            return result.get("content", result)

    except Exception as e:
        logger.warning(f"ScrapeGraphAI scrape failed for '{url}': {e}")

    return None


async def search_industry_companies(
    industry: str,
    max_companies: int = 10,
) -> list[CompanyProfile]:
    """Find companies in an industry. Multi-source with validation.

    Source 1: ddgs search → extract company names from snippets
    Source 2: Google News RSS for industry → extract mentioned companies
    Source 3: DB saved companies matching industry
    Source 4: LLM parametric knowledge (fallback)
    Source 5: ScrapeGraphAI SearchGraph (background enrichment for top results)

    Every company name is validated against Wikidata before return.
    This is what prevents "Temple" → Hindu temple garbage.
    """
    # This will be implemented as the orchestration of multiple sources
    # with entity validation. See Task 3 for the companies.py rewrite.
    pass
```

**Key design decisions:**
- `search()` uses ddgs primary (no Docker dependency) with SearXNG as fallback (already running)
- `extract()` uses trafilatura primary (local, fast, F1=0.958) with Jina Reader fallback (JS pages)
- `company_news()` uses Google News RSS (100+ articles free, even for tiny Indian startups like Innoviti)
- `deep_company_search()` uses ScrapeGraphAI SearchGraph (30-80s, background only, proven 10/10 in stress test)
- All functions have semaphores for bounded concurrency
- All functions have timeouts and graceful fallbacks

**Step 1:** Create `app/tools/web_intel.py` with the full implementation above.

**Step 2:** Install missing dependencies:
```bash
pip install ddgs trafilatura feedparser
# scrapegraphai already installed (v1.73.0)
```

**Step 3:** Write basic test:
```bash
python -c "
import asyncio
from app.tools.web_intel import search, extract, company_news
async def test():
    # Fast search
    r = await search('NVIDIA', max_results=3)
    print(f'search: {len(r)} results, first={r[0].title if r else None}')
    # Extract
    c = await extract('https://en.wikipedia.org/wiki/Nvidia')
    print(f'extract: {len(c)} chars')
    # News
    n = await company_news('NVIDIA', max_articles=5, extract_content=False)
    print(f'news: {len(n)} articles')
asyncio.run(test())
"
```

**Step 4:** Commit: `feat: add unified web_intel.py tool layer (ddgs + trafilatura + Jina + ScrapeGraphAI + Google News RSS)`

---

## Task 2: `app/tools/company_enricher.py` — Entity-Validated Enrichment

**Files:**
- Create: `app/tools/company_enricher.py`
- Modify: `app/tools/wikidata.py` (minor: persistent httpx client, remove dead logo_url)
- Modify: `app/tools/apollo_tool.py` (expose org data caching)

**What this solves:** The "Temple" problem. Every company name is validated against Wikidata P31 (instance-of) before it's returned to the user. No more Hindu temples, no more article titles, no more garbage.

**Design:**

```python
"""
Entity-validated company enrichment.

The CENTRAL rule: No company is returned to the user unless it passes entity validation.
Validation = confirmed as a business entity via at least one of:
  1. Wikidata P31 match (instance of: business/company/enterprise)
  2. Apollo org data exists (real company in Apollo's database)
  3. Has a valid corporate website domain

This prevents:
  - "Temple" → Hindu temple news (Wikidata P31 = "temple", not "company" → rejected)
  - "100 Top FinTech Companies" → article title (not in Wikidata → rejected)
  - "zhidao.baidu.com" → knowledge site domain (fails is_valid_company_domain → rejected)
"""

class ValidationResult(BaseModel):
    """Result of entity validation."""
    is_valid_company: bool = False
    confidence: float = 0.0  # 0-1
    validation_source: str = ""  # "wikidata" | "apollo" | "domain" | "llm"
    rejection_reason: str = ""  # If invalid, why


class EnrichedCompany(BaseModel):
    """Fully enriched company profile — the output of the enrichment pipeline."""
    # Identity
    company_name: str
    wikidata_id: str = ""
    domain: str = ""
    website: str = ""
    # Validation
    validation: ValidationResult = ValidationResult()
    # Profile (from Wikidata + Wikipedia + Apollo + ScrapeGraphAI)
    description: str = ""
    industry: str = ""
    sub_industries: list[str] = []
    founded_year: int | None = None
    headquarters: str = ""
    ceo: str = ""
    key_people: list[dict] = []
    employee_count: str = ""
    revenue: str = ""
    stock_ticker: str = ""
    stock_exchange: str = ""
    funding_stage: str = ""
    total_funding: str = ""
    investors: list[str] = []
    products_services: list[str] = []
    competitors: list[str] = []
    subsidiaries: list[str] = []
    tech_stack: list[str] = []
    # Sources
    data_sources: list[str] = []  # Which sources contributed data


async def validate_entity(name: str) -> ValidationResult:
    """Validate that a name refers to a real business entity.

    Three checks in parallel:
    1. Wikidata P31 — is it instance of business/company/enterprise?
    2. Domain check — does name.com or similar resolve to a corporate site?
    3. Quick LLM check — "Is '{name}' a company? Answer yes/no."

    Must pass AT LEAST ONE check to be considered valid.
    """


async def enrich(
    company_name: str,
    domain: str = "",
    skip_validation: bool = False,  # True for companies already in our DB
    background_deep: bool = False,  # True to also run ScrapeGraphAI (slow)
) -> EnrichedCompany | None:
    """Full enrichment pipeline with entity validation.

    Step 1: Check Company KB (SQLite) — return cached if fresh
    Step 2: Validate entity (unless skip_validation)
    Step 3: Parallel enrichment:
       - Wikidata + Wikipedia (free, 200-500ms)
       - Apollo org data (if domain known, free side-effect)
    Step 4: Domain resolution (Wikidata P856 → ddgs search → name guess)
    Step 5: LLM gap-fill (only if description still empty)
    Step 6: [Optional] ScrapeGraphAI deep scrape (background, 30-80s)
    Step 7: Save to Company KB
    """


async def enrich_batch(
    names: list[str],
    max_concurrent: int = 5,
    skip_validation: bool = False,
) -> list[EnrichedCompany]:
    """Enrich multiple companies in parallel with bounded concurrency."""
```

**The validation flow that prevents "Temple" garbage:**

```
User searches "Temple"
  → Wikidata: finds "Temple" → P31 = "religious building" → NOT a company → rejected
  → Also finds "Temple & Webster" → P31 = "public company" → VALID → returned
  → Also finds "Temple Inc" → P31 = "business" → VALID → returned

User searches "fintech"
  → classified as industry (Wikidata P31 = "concept")
  → search_industry_companies() runs
  → each extracted company name → validate_entity()
  → "100 Top FinTech Companies" → not in Wikidata, no domain → REJECTED
  → "Stripe" → Wikidata P31 = "business" → VALID
  → "Razorpay" → Wikidata P31 = "enterprise" → VALID
```

**Step 1:** Create `app/tools/company_enricher.py` with full implementation.

**Step 2:** Modify `app/tools/wikidata.py`:
- Remove `logo_url` field from `WikidataProfile` (line 80)
- Use persistent `httpx.AsyncClient` instead of `async with _get_client()` per call
- Add `is_valid_business_entity(wikidata_id)` public function that checks P31

**Step 3:** Modify `app/tools/apollo_tool.py`:
- Add `_org_cache: dict[str, dict]` class variable
- In `search_people_at_company()`, cache org data by domain
- Add `get_cached_org(domain) -> dict | None` public method

**Step 4:** Test:
```bash
python -c "
import asyncio
from app.tools.company_enricher import validate_entity, enrich
async def test():
    # Should pass
    v = await validate_entity('NVIDIA')
    print(f'NVIDIA: valid={v.is_valid_company}, source={v.validation_source}')
    # Should fail (religious building)
    v = await validate_entity('Temple')
    print(f'Temple: valid={v.is_valid_company}, reason={v.rejection_reason}')
    # Full enrichment
    c = await enrich('Infosys')
    print(f'Infosys: {c.headquarters}, {c.ceo}, {c.employee_count}')
asyncio.run(test())
"
```

**Step 5:** Commit: `feat: add entity-validated company_enricher.py (prevents Temple-type garbage)`

---

## Task 3: Rewrite `app/api/companies.py` — Clean Search

**Files:**
- Modify: `app/api/companies.py` (major rewrite of search functions)
- Modify: `app/api/schemas.py` (add new fields)

**What changes:**

### 3A. `_detect_search_type()` → Entity-first classification

Replace the current 3-layer classifier (lines 521-564) with entity validation:

```python
async def _detect_search_type(query: str) -> tuple[str, EnrichedCompany | None]:
    """Classify query using entity validation.

    If Wikidata says it's a company → "company" + enriched profile
    If Wikidata says it's a concept/industry → "industry"
    If Wikidata doesn't know → LLM classify → heuristic fallback
    """
    from app.tools.company_enricher import validate_entity

    result = await validate_entity(query)
    if result.is_valid_company:
        # It's a company — enrich it
        enriched = await enrich(query)
        return "company", enriched
    elif result.rejection_reason and "not a company" in result.rejection_reason:
        # Wikidata found it but it's not a company (e.g., "Temple" = religious building)
        return "industry", None
    else:
        # Unknown to Wikidata — use LLM/heuristic
        # ... existing Layer 2-3 fallback ...
```

### 3B. `_search_by_company_name()` → Use enricher

Replace lines 141-252 with a clean flow:
1. `company_enricher.enrich(query)` — handles Wikidata + Wikipedia + Apollo + validation
2. `web_intel.search(query, extract_content=False)` — for finding domain and articles
3. `web_intel.company_news(query, max_articles=5, extract_content=False)` — for live news
4. Merge into `CompanySearchResult`
5. Save to KB

### 3C. `_search_by_industry()` → Validated multi-source

Replace lines 258-404 with:
1. `web_intel.search(f"{industry} companies", max_results=15)` — fast discovery
2. Extract company names from snippets via LLM
3. **Validate each name** via `company_enricher.validate_entity()` — THIS IS THE KEY
4. `company_enricher.enrich_batch(valid_names)` — parallel enrichment of validated names
5. Also query DB `_db_search_companies_by_industry()` (existing, keep)
6. Also LLM knowledge fallback (existing, keep)
7. Merge, deduplicate, return top N

### 3D. Remove dead code

- Remove `_save_mock_result()` (lines 923-939)
- Remove `_resolve_domain` import from `lead_gen.py` (line 338) — use `web_intel.search` + `domain_utils` instead
- Remove duplicate `_extract_company_names_from_snippets()` (replaced by LLM call in web_intel)

### 3E. Schema additions

In `app/api/schemas.py`, add to `CompanySearchResult`:
```python
sub_industries: list[str] = []
key_people: list[dict] = []  # [{"name": "...", "role": "..."}]
products_services: list[str] = []
competitors: list[str] = []
revenue: str = ""
total_funding: str = ""
investors: list[str] = []
tech_stack: list[str] = []
validation_source: str = ""  # "wikidata" | "apollo" | "domain"
```

**Step 1:** Clean dead code from `companies.py`.
**Step 2:** Rewrite `_detect_search_type()` with entity validation.
**Step 3:** Rewrite `_search_by_company_name()` using `company_enricher.enrich()`.
**Step 4:** Rewrite `_search_by_industry()` with validation loop.
**Step 5:** Update `schemas.py` with new fields.
**Step 6:** Test: search "NVIDIA", "Temple", "fintech", "proptech", "Innoviti Technologies".
**Step 7:** Commit: `feat: rewrite companies.py with entity validation (no more Temple garbage)`

---

## Task 4: Trend-Aware Contact Matching

**Files:**
- Modify: `app/agents/workers/contact_agent.py` (major upgrade)
- Modify: `app/schemas/sales.py` (add trend_role_map)

**Problem:** Currently finds the SAME roles for every company regardless of trend. A cybersecurity trend should find CISOs, a cost-reduction trend should find CFOs, a digital transformation trend should find CTOs. Instead, the code uses a static `target_roles` list that often defaults to CEO.

**Design:**

```python
# Trend-type → target roles mapping
TREND_ROLE_MAP = {
    # Security/compliance trends
    "cybersecurity": ["CISO", "VP Security", "Head of Information Security", "Security Director"],
    "compliance": ["Chief Compliance Officer", "VP Legal", "Head of Risk", "General Counsel"],
    "data_privacy": ["DPO", "Chief Privacy Officer", "CISO", "VP Legal"],

    # Technology/digital trends
    "digital_transformation": ["CTO", "CDO", "VP Engineering", "Head of Digital"],
    "ai_adoption": ["CTO", "Chief AI Officer", "VP Engineering", "Head of Data Science"],
    "cloud_migration": ["CTO", "VP Infrastructure", "Head of Cloud", "IT Director"],

    # Business/financial trends
    "cost_reduction": ["CFO", "COO", "VP Operations", "Head of Procurement"],
    "market_expansion": ["CEO", "CSO", "VP Business Development", "Head of Strategy"],
    "supply_chain": ["COO", "VP Supply Chain", "Head of Logistics", "Procurement Director"],

    # Industry-specific
    "sustainability": ["CSO", "VP Sustainability", "Head of ESG", "Environmental Director"],
    "talent": ["CHRO", "VP People", "Head of Talent", "HR Director"],

    # Default (when trend type doesn't match)
    "default": ["CEO", "CTO", "CFO", "VP Operations", "Head of Strategy"],
}


def match_roles_to_trend(trend_type: str, pain_point: str, who_needs_help: str) -> list[str]:
    """Determine which roles to target based on the trend.

    Uses trend_type as primary signal, then scans pain_point and
    who_needs_help for role keywords. Returns ordered list of target roles.
    """
    # Check trend_type against TREND_ROLE_MAP
    roles = TREND_ROLE_MAP.get(trend_type, [])

    # Scan pain_point and who_needs_help for additional role signals
    text = f"{pain_point} {who_needs_help}".lower()
    if "security" in text or "cyber" in text:
        roles = TREND_ROLE_MAP["cybersecurity"] + roles
    if "cost" in text or "budget" in text or "expense" in text:
        roles = TREND_ROLE_MAP["cost_reduction"] + roles
    # ... more keyword scanning

    # Deduplicate while preserving order
    seen = set()
    unique_roles = []
    for r in roles:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique_roles.append(r)

    return unique_roles[:8]  # Cap at 8 roles to search
```

**Also fix:** `_extract_contact_from_search()` line 365 — change `r.get('snippet', '')` to `r.get('content', '')` to match `SearchManager` output format. This is a confirmed bug.

**Also fix:** Parallelize sequential SearXNG calls in Phase 2 (line 197 — currently iterates roles one-by-one after Apollo).

**Step 1:** Add `TREND_ROLE_MAP` and `match_roles_to_trend()` to `contact_agent.py`.
**Step 2:** Modify `_find_contacts_for_company()` to call `match_roles_to_trend()` using the company's associated trend data.
**Step 3:** Fix the `snippet` → `content` field name bug.
**Step 4:** Parallelize Phase 2 role searches.
**Step 5:** Test with a cybersecurity trend (should find CISOs, not CEOs).
**Step 6:** Commit: `feat: trend-aware contact matching (CISO for security, CFO for cost, etc.)`

---

## Task 5: Person-Context Scraping for Hyper-Personalized Outreach

**Files:**
- Create: `app/tools/person_intel.py`
- Modify: `app/agents/workers/email_agent.py` (use person context)
- Modify: `app/schemas/sales.py` (add PersonContext model)

**Problem:** Current emails are personalized only by role tier (decision_maker/influencer/gatekeeper). They don't know anything about the PERSON — their background, recent posts, company initiatives, or what they care about. Real sales tools (Apollo, ZoomInfo) scrape LinkedIn profiles, company blogs, and recent interviews to personalize.

**Design:**

```python
"""
Person intelligence — scrape context about a contact for hyper-personalized outreach.

Sources (in priority order):
1. LinkedIn profile (via ddgs search, extract public info)
2. Company website team page (via ScrapeGraphAI if URL known)
3. Recent mentions in news (via ddgs news search)
4. Conference talks/interviews (via ddgs search)

Output: PersonContext with talking points for email personalization.
"""

class PersonContext(BaseModel):
    """Scraped context about a person for email personalization."""
    person_name: str
    company_name: str
    role: str
    # Scraped intelligence
    background_summary: str = ""    # 2-3 sentences about their background
    recent_focus: str = ""          # What they've been working on recently
    notable_achievements: list[str] = []  # Awards, promotions, initiatives
    shared_interests: list[str] = []  # Topics they post about, care about
    talking_points: list[str] = []  # Specific points to reference in outreach
    # Sources
    linkedin_headline: str = ""
    recent_posts: list[str] = []    # Titles/summaries of recent content
    news_mentions: list[str] = []   # Recent news mentioning this person
    sources: list[str] = []         # URLs where context was found


async def gather_person_context(
    person_name: str,
    company_name: str,
    role: str,
    trend_context: str = "",  # The trend this outreach is about
) -> PersonContext:
    """Scrape all available context about a person.

    Step 1: ddgs search "{person_name} {company_name}" → extract snippets
    Step 2: ddgs news search for recent mentions
    Step 3: LLM synthesizes talking_points from scraped data + trend context
    """
```

**Email agent upgrade:**

```python
# In _generate_outreach(), BEFORE building the LLM prompt:

# Scrape person context (5-10s, runs in parallel with other contacts)
from app.tools.person_intel import gather_person_context
person_ctx = await gather_person_context(
    person_name=contact.person_name,
    company_name=company.company_name,
    role=contact.role,
    trend_context=impact.pitch_angle if impact else "",
)

# Build MUCH richer prompt:
prompt = f"""Write a hyper-personalized cold outreach email.

RECIPIENT:
  Name: {contact.person_name}
  Role: {contact.role} at {company.company_name}
  Background: {person_ctx.background_summary}
  Recent focus: {person_ctx.recent_focus}
  Notable: {', '.join(person_ctx.notable_achievements[:3])}

CONTEXT:
  Trend: {trend.title}
  Pain point: {impact.midsize_pain_points[0]}
  Their company's situation: {company.description}

TALKING POINTS TO REFERENCE:
{chr(10).join(f'  - {tp}' for tp in person_ctx.talking_points[:3])}

RULES:
  - Reference something SPECIFIC about them (their background, recent work, or achievement)
  - Connect the trend to THEIR specific situation
  - Tone: {tone} (based on their seniority)
  - Max {word_limit} words
  - One clear CTA
"""
```

**Step 1:** Create `app/tools/person_intel.py` with `gather_person_context()`.
**Step 2:** Add `PersonContext` to `app/schemas/sales.py`.
**Step 3:** Modify `email_agent.py` `_generate_outreach()` to scrape person context before generating.
**Step 4:** Test with a real contact (e.g., Jensen Huang at NVIDIA).
**Step 5:** Commit: `feat: person-context scraping for hyper-personalized outreach`

---

## Task 6: Background News Collector — 5-7 Months of Company News

**Files:**
- Create: `app/tools/news_collector.py`
- Modify: `app/api/companies.py` (trigger collection on search)
- Modify: `app/agents/orchestrator.py` (trigger collection after pipeline)

**Problem:** ChromaDB is ALWAYS empty in the FastAPI path because `store_articles()` is only called in the legacy Streamlit path. The pipeline processes news articles in-memory but never persists them. The company news tab shows at most 5+5 items.

**Design:**

```python
"""
Background news collector — accumulates company-specific news in ChromaDB.

Runs:
1. After pipeline completion — collects news for all discovered companies
2. On company detail page visit — collects news for that specific company
3. On-demand via API endpoint — manual trigger

Uses Google News RSS for discovery (100+ articles per query, free).
Uses trafilatura for content extraction.
Stores in ChromaDB with company_name metadata for retrieval.

Over multiple pipeline runs (5-7 months), ChromaDB accumulates a rich
archive of company-specific news — exactly what the news tab needs.
"""

async def collect_company_news(
    company_name: str,
    months_back: int = 1,
    max_articles: int = 50,
) -> int:
    """Collect news for a single company and store in ChromaDB.

    Returns: number of new articles stored (after dedup).
    """
    from app.tools.web_intel import company_news
    from app.tools.article_cache import ArticleCache

    cache = ArticleCache()
    stored = 0

    # Collect news for different time ranges
    time_ranges = ["when:7d", "when:1m"]
    if months_back >= 3:
        time_ranges.append("when:3m")
    if months_back >= 6:
        time_ranges.append("when:6m")

    for time_range in time_ranges:
        articles = await company_news(
            company_name,
            max_articles=max_articles // len(time_ranges),
            time_range=time_range,
            extract_content=True,
        )
        for article in articles:
            if not article.content:
                continue
            # Dedup by URL
            if cache.url_exists(article.url):
                continue
            # Store in ChromaDB
            cache.store_article(
                title=article.title,
                content=article.content,
                url=article.url,
                source_name=article.source_name,
                published_at=article.published_at,
                metadata={"company_name": company_name},
            )
            stored += 1

    return stored


async def collect_news_for_companies(
    company_names: list[str],
    months_back: int = 1,
    max_concurrent: int = 3,
) -> dict[str, int]:
    """Collect news for multiple companies in parallel."""
    sem = asyncio.Semaphore(max_concurrent)
    results = {}

    async def _collect_one(name):
        async with sem:
            count = await collect_company_news(name, months_back)
            results[name] = count

    await asyncio.gather(*[_collect_one(n) for n in company_names])
    return results
```

**Integration points:**

1. **After pipeline completion** (`orchestrator.py`):
```python
# In run_pipeline(), after lead_gen step:
company_names = [c.company_name for c in companies]
asyncio.create_task(collect_news_for_companies(company_names, months_back=1))
```

2. **On company search** (`companies.py`):
```python
# After returning search results, trigger background collection
asyncio.create_task(collect_company_news(company_name, months_back=3))
```

3. **On company detail page visit** (already exists: `GET /companies/{id}/news`):
```python
# Trigger background collection if articles < threshold
if article_count < 20:
    asyncio.create_task(collect_company_news(company_name, months_back=6))
```

**Step 1:** Create `app/tools/news_collector.py`.
**Step 2:** Add `url_exists()` and `store_article()` methods to `ArticleCache` if not present.
**Step 3:** Integrate into `orchestrator.py` post-pipeline.
**Step 4:** Integrate into `companies.py` search and news endpoints.
**Step 5:** Test: search "NVIDIA" → check ChromaDB article count grows.
**Step 6:** Commit: `feat: background news collector accumulates 5-7 months of company news in ChromaDB`

---

## Task 7: Upgrade Pipeline — `company_agent.py` and `lead_gen.py`

**Files:**
- Modify: `app/agents/workers/company_agent.py`
- Modify: `app/agents/lead_gen.py`

### 7A. `company_agent.py` — Use `web_intel` + `company_enricher`

Replace all direct SearXNG calls with `web_intel.search()`:

```python
# In _do_search() (line 216):
# BEFORE: results = await search_mgr.web_search(query, max_results=10)
# AFTER:
from app.tools.web_intel import search
results = await search(query, max_results=10)
```

Replace `_verify_company()` (line 793) with `company_enricher.validate_entity()` + `company_enricher.enrich()`:

```python
# BEFORE: manual Wikipedia check + manual Wikidata check
# AFTER:
from app.tools.company_enricher import validate_entity, enrich

async def _verify_company(self, company, deps):
    validation = await validate_entity(company.company_name)
    if not validation.is_valid_company:
        company.verification_confidence = 0.1
        return company

    enriched = await enrich(company.company_name, domain=company.domain, skip_validation=True)
    if enriched:
        company.description = enriched.description or company.description
        company.headquarters = enriched.headquarters or company.headquarters
        company.employee_count = enriched.employee_count or company.employee_count
        company.wikidata_id = enriched.wikidata_id
        company.verification_confidence = 0.95 if enriched.wikidata_id else 0.6
    return company
```

### 7B. `lead_gen.py` — Use `company_enricher` + remove `_resolve_domain()`

Replace `_resolve_domain()` (lines 42-57) with `web_intel.search()` + `domain_utils`:

```python
# Move domain resolution to domain_utils.py:
async def resolve_company_domain(company_name: str) -> str:
    """Find a company's domain via web search + validation."""
    from app.tools.web_intel import search
    from app.tools.domain_utils import extract_clean_domain, is_valid_company_domain

    results = await search(f"{company_name} official website", max_results=5)
    for r in results:
        d = extract_clean_domain(r.url)
        if d and is_valid_company_domain(d):
            return d
    return ""
```

Replace manual enrichment loop (lines 140-184) with `company_enricher.enrich_batch()`.

### 7C. Fix `_find_companies_for_impact()` concurrency

Add semaphore to the `asyncio.gather()` at line 229 that fires ALL search queries simultaneously:

```python
search_sem = asyncio.Semaphore(10)  # Cap concurrent searches
async def _do_search_bounded(query):
    async with search_sem:
        return await _do_search(query)
results = await asyncio.gather(*[_do_search_bounded(q) for q in queries])
```

**Step 1:** Replace SearXNG calls in `company_agent.py` with `web_intel.search()`.
**Step 2:** Replace `_verify_company()` with `company_enricher.validate_entity()` + `enrich()`.
**Step 3:** Move `_resolve_domain()` to `domain_utils.py`, add `is_valid_company_domain()` check.
**Step 4:** Replace manual enrichment in `lead_gen.py` with `company_enricher.enrich_batch()`.
**Step 5:** Add concurrency semaphore to `_find_companies_for_impact()`.
**Step 6:** Commit: `feat: upgrade pipeline to use web_intel + company_enricher`

---

## Task 8: Database + Schema Updates

**Files:**
- Modify: `app/database.py`
- Modify: `app/schemas/sales.py`
- Modify: `frontend/lib/types.ts`
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/app/(dashboard)/companies/[id]/page.tsx`

### 8A. Schema additions

`app/schemas/sales.py` — add to `CompanyData`:
```python
sub_industries: list[str] = Field(default_factory=list)
key_people: list[dict] = Field(default_factory=list)
products_services: list[str] = Field(default_factory=list)
competitors: list[str] = Field(default_factory=list)
revenue: str = ""
total_funding: str = ""
investors: list[str] = Field(default_factory=list)
tech_stack: list[str] = Field(default_factory=list)
validation_source: str = ""
```

### 8B. Database columns

`app/database.py` — add to `SavedCompanyModel`:
```python
sub_industries = Column(Text, default="")       # JSON list
key_people = Column(Text, default="")            # JSON list of dicts
products_services = Column(Text, default="")     # JSON list
competitors = Column(Text, default="")           # JSON list
revenue = Column(String(100), default="")
total_funding = Column(String(100), default="")
investors = Column(Text, default="")             # JSON list
tech_stack = Column(Text, default="")            # JSON list
validation_source = Column(String(50), default="")
```

Update `_migrate_columns()`, `save_company()`, `_company_to_dict()`.

### 8C. Remove dead LeadModel

Remove `LeadModel` (lines 188-223), `save_lead()`, `get_latest_leads()`.

### 8D. Frontend types

`frontend/lib/types.ts` — add to `CompanySearchResult` and `SavedCompany`:
```typescript
sub_industries: string[];
key_people: { name: string; role: string }[];
products_services: string[];
competitors: string[];
revenue: string;
total_funding: string;
investors: string[];
tech_stack: string[];
validation_source: string;
```

### 8E. Frontend display

In `[id]/page.tsx` OverviewTab, add new sections:
- **Products & Services** — pill list from `products_services`
- **Key People** — name + role list from `key_people`
- **Competitors** — pill list from `competitors`
- **Tech Stack** — pill list from `tech_stack`
- **Funding** — `total_funding` + `investors` list + `funding_stage`

**Step 1:** Add fields to `sales.py`, `schemas.py`, `database.py`.
**Step 2:** Update `_migrate_columns()` and serializers.
**Step 3:** Remove dead `LeadModel`.
**Step 4:** Add frontend types.
**Step 5:** Update OverviewTab with new sections.
**Step 6:** Commit: `feat: extended company data model (products, competitors, funding, tech stack)`

---

## Task 9: Fix `domain_utils.py` and `hunter_tool.py`

**Files:**
- Modify: `app/tools/domain_utils.py`
- Modify: `app/tools/hunter_tool.py`

### 9A. domain_utils.py

- Remove dead `normalize_domain()` (line 186)
- Remove dead `extract_domains_from_text()` (line 214)
- Fix `extract_domain_from_company_name()` — actually try `.in` and `.co.in` for Indian companies
- Fix `is_valid_company_domain()` — make TLD validation actually enforce (not just log)
- Add `resolve_company_domain()` function (moved from `lead_gen.py`)

### 9B. hunter_tool.py

- Remove dead `patterns` list in `generate_email_pattern()` (lines 249-252)
- Fix `verify_email()` field name: caller checks `status == "valid"` but should also check `result == "deliverable"`
- Add usage counter to prevent exceeding free tier (25 searches/month)

**Step 1:** Clean dead code from both files.
**Step 2:** Fix `is_valid_company_domain()` TLD enforcement.
**Step 3:** Add `resolve_company_domain()` to `domain_utils.py`.
**Step 4:** Add Hunter usage counter.
**Step 5:** Commit: `fix: clean domain_utils + hunter_tool dead code, fix TLD validation`

---

## Task 10: Store Contacts in DB for Reuse

**Files:**
- Modify: `app/database.py` (contacts already stored as JSON, add query methods)
- Modify: `app/agents/workers/contact_agent.py` (check DB before API calls)

**Problem:** Currently contacts are generated fresh every time. If we already found 5 contacts for NVIDIA last week, we should reuse them (and optionally refresh).

**Design:**

```python
# database.py — add:
def get_company_contacts(self, company_name: str) -> list[dict] | None:
    """Get cached contacts for a company. Returns None if not found or stale."""
    company = self.get_or_enrich_company(company_name, max_age_days=30)
    if company and company.get("contacts"):
        return json.loads(company["contacts"]) if isinstance(company["contacts"], str) else company["contacts"]
    return None

# contact_agent.py — modify find_contacts():
async def _find_contacts_for_company(self, company, deps, ...):
    # Check DB first
    cached = deps.db.get_company_contacts(company.company_name)
    if cached and len(cached) >= 3:
        logger.info(f"Using {len(cached)} cached contacts for {company.company_name}")
        return [ContactData(**c) for c in cached]

    # ... existing Apollo → Hunter → SearXNG flow ...

    # Save found contacts to DB
    deps.db.save_company_contacts(company_id, contacts_as_dicts)
```

**Step 1:** Add `get_company_contacts()` to `database.py`.
**Step 2:** Add DB check at top of `_find_contacts_for_company()`.
**Step 3:** Ensure contacts are saved after generation (already partially done via `save_company_contacts`).
**Step 4:** Test: generate contacts → check DB → generate again → should use cache.
**Step 5:** Commit: `feat: cache contacts in DB, reuse on subsequent lookups`

---

## Execution Order

```
Task 1:  web_intel.py (foundation — all other tasks depend on this)
Task 9:  domain_utils + hunter cleanup (small, unblocks Task 2)
Task 2:  company_enricher.py (depends on Task 1 + 9)
Task 3:  companies.py rewrite (depends on Task 1 + 2)
Task 4:  trend-aware contacts (independent, can parallel with 3)
Task 5:  person-context scraping (depends on Task 4)
Task 6:  news collector (depends on Task 1)
Task 7:  pipeline upgrade (depends on Task 1 + 2)
Task 8:  schema + frontend (depends on Task 2 + 3)
Task 10: contact caching (depends on Task 4)
```

```
Parallelizable:
  [Task 1] → [Task 2] → [Task 3] → [Task 8]
                       ↘ [Task 7]
  [Task 9] (parallel with Task 1)
  [Task 4] → [Task 5] → [Task 10] (parallel with Tasks 2-3)
  [Task 6] (parallel with Tasks 2-5, depends only on Task 1)
```

---

## Verification

After all tasks complete:

1. **Search "NVIDIA"** → full enriched profile (CEO, employees, revenue, products, competitors, ticker, HQ)
2. **Search "Temple"** → Temple & Webster (retail company), NOT Hindu temples
3. **Search "fintech"** → 10 validated companies (Stripe, Razorpay, etc.), NOT article titles
4. **Search "proptech"** → real proptech companies with enrichment
5. **Search "Innoviti Technologies"** → finds tiny Indian startup with funding data
6. **Company news tab** → 20+ articles, paginated, growing over time
7. **Pipeline run** → companies get full enrichment + trend-matched contacts (CISO for security, CFO for cost)
8. **Email outreach** → references person's background, recent work, specific talking points
9. **Second search for same company** → instant from KB cache (0ms enrichment)
10. **Second pipeline finding same company** → contacts from DB cache, skip API calls
