"""Company search router — entity-validated, multi-source company intelligence.

Three search paths:
  A. Company name → entity validation + enrichment + ChromaDB articles + live news
  B. Industry keyword → web search + LLM extraction + validation + batch enrichment
  C. On-demand company news from ChromaDB (5 months of pipeline-collected articles)

Entity validation: company_enricher.validate_entity() (Tavily search + LLM classify + domain check).

Data source priority:
  1. Company KB (SQLite cache) — 0ms, 7-day staleness
  2. Tavily deep research — AI answers + web results (~1s)
  3. Apollo — org data, funding, tech stack (~500ms)
  4. web_intel — live news, web search, domain resolution (~1-2s)
  5. LLM gap-fill — description only, last resort (~2s)
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    CompanySearchRequest,
    CompanySearchResponse,
    CompanySearchResult,
    GenerateLeadsRequest,
    GenerateLeadsResponse,
    PersonResponse,
    SavedCompanyResponse,
    SavedCompanyListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Module-level singletons (avoid per-request ChromaDB file lock contention) ─

_article_cache = None


def _get_article_cache():
    global _article_cache
    if _article_cache is None:
        from app.tools.article_cache import ArticleCache
        _article_cache = ArticleCache()
    return _article_cache


async def _chromadb_query(article_cache, query_texts: list, n_results: int, include: list):
    """Run ChromaDB query against the news collection (auto-embedded articles)."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                article_cache.news_collection.query,
                query_texts=query_texts,
                n_results=n_results,
                include=include,
            ),
            timeout=10.0,
        )
        return result
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"ChromaDB query failed/timed out: {e}")
        return None


# ── Helper: convert EnrichedCompany → CompanySearchResult ─────────────────


def _enriched_to_result(
    enriched,
    query: str,
    cached_articles: list | None = None,
    live_news: list | None = None,
    reason: str = "",
    industry_override: str = "",
) -> CompanySearchResult:
    """Convert an EnrichedCompany model to a CompanySearchResult for the API."""
    cid = hashlib.md5(enriched.company_name.lower().encode()).hexdigest()[:12]
    return CompanySearchResult(
        id=cid,
        company_name=enriched.company_name,
        domain=enriched.domain,
        website=enriched.website,
        industry=enriched.industry or industry_override,
        description=enriched.description,
        headquarters=enriched.headquarters,
        employee_count=enriched.employee_count,
        founded_year=enriched.founded_year,
        stock_ticker=enriched.stock_ticker,
        ceo=enriched.ceo,
        funding_stage=enriched.funding_stage,
        reason_relevant=reason or f"Direct search for '{query}'",
        article_count=len(cached_articles) if cached_articles else 0,
        recent_articles=(cached_articles or [])[:10],
        live_news=(live_news or [])[:5],
        # Extended fields
        sub_industries=enriched.sub_industries,
        key_people=[{"name": kp} if isinstance(kp, str) else kp for kp in enriched.key_people],
        products_services=enriched.products_services,
        competitors=enriched.competitors,
        revenue=enriched.revenue,
        total_funding=enriched.total_funding,
        investors=enriched.investors,
        tech_stack=enriched.tech_stack,
        validation_source=enriched.validation.validation_source if enriched.validation else "",
    )


# ── Path A: Company name search ──────────────────────────────────────


async def _search_by_company_name(
    query: str, enriched=None, max_results: int = 10,
) -> tuple[list[CompanySearchResult], int]:
    """Search for a specific company by name.

    DB-first cache strategy: check saved companies before expensive enrichment.
    Cache hit with description → build result directly (1-3s, just news).
    Cache miss → full enrichment (15-30s).
    """
    from app.database import get_database

    # Step 0: Check DB cache — instant if company was searched before
    db_cached = None
    if not enriched:
        try:
            db = get_database()
            db_cached = db.get_or_enrich_company(query, max_age_days=7)
        except Exception as e:
            logger.debug(f"DB cache lookup failed: {e}")

    if db_cached and db_cached.get("description"):
        logger.info(f"Cache HIT for '{query}' — skipping enrichment")
        # Still fetch live news + ChromaDB (fast, 1-2s each)
        news_task = asyncio.create_task(_fetch_live_news(query))
        cached_articles, cached_count = await _fetch_chromadb_articles(query, max_results)
        live_news = await _await_live_news(news_task)

        result = CompanySearchResult(
            company_name=db_cached.get("company_name", query),
            domain=db_cached.get("domain", ""),
            website=db_cached.get("website", ""),
            industry=db_cached.get("industry", ""),
            description=db_cached.get("description", ""),
            headquarters=db_cached.get("headquarters", ""),
            employee_count=db_cached.get("employee_count", ""),
            founded_year=db_cached.get("founded_year"),
            stock_ticker=db_cached.get("stock_ticker", ""),
            ceo=db_cached.get("ceo", ""),
            funding_stage=db_cached.get("funding_stage", ""),
            reason_relevant=db_cached.get("reason_relevant", f"Direct search for '{query}'"),
            article_count=len(cached_articles),
            recent_articles=cached_articles[:10],
            live_news=live_news[:5],
            sub_industries=db_cached.get("sub_industries", []),
            key_people=db_cached.get("key_people", []),
            products_services=db_cached.get("products_services", []),
            competitors=db_cached.get("competitors", []),
            revenue=db_cached.get("revenue", ""),
            total_funding=db_cached.get("total_funding", ""),
            investors=db_cached.get("investors", []),
            tech_stack=db_cached.get("tech_stack", []),
            validation_source=db_cached.get("validation_source", "cache"),
        )
        return [result], cached_count

    # Step 1: Full enrichment (cache miss or stale)
    from app.tools.company_enricher import enrich
    if not enriched:
        enriched = await enrich(query)
    if not enriched:
        return [], 0  # Validation failed — not a real company

    # Step 2: Get live news + ChromaDB in parallel
    news_task = asyncio.create_task(_fetch_live_news(query))
    cached_articles, cached_count = await _fetch_chromadb_articles(query, max_results)
    live_news = await _await_live_news(news_task)

    # Step 3: Build result
    result = _enriched_to_result(
        enriched, query,
        cached_articles=cached_articles,
        live_news=live_news,
    )

    return [result], cached_count


async def _fetch_chromadb_articles(query: str, max_results: int) -> tuple[list, int]:
    """Fetch cached articles from ChromaDB news collection."""
    article_cache = _get_article_cache()
    cached_articles = []
    if article_cache:
        # Try metadata filter first (exact company match)
        try:
            chroma_results = await asyncio.wait_for(
                asyncio.to_thread(
                    article_cache.news_collection.get,
                    where={"company_name": query.lower()},
                    limit=min(20, max_results * 3),
                    include=["metadatas"],
                ),
                timeout=8.0,
            )
            if chroma_results and chroma_results.get("metadatas"):
                for meta in chroma_results["metadatas"]:
                    cached_articles.append({
                        "title": meta.get("title", ""),
                        "summary": (meta.get("summary", "") or "")[:200],
                        "source_name": meta.get("source_name", ""),
                        "published_at": meta.get("published_at", ""),
                        "url": meta.get("url", ""),
                    })
        except Exception as e:
            logger.debug(f"ChromaDB metadata filter failed: {e}")

        # Fall back to semantic search if metadata filter returned nothing
        if not cached_articles:
            chroma_results = await _chromadb_query(
                article_cache, query_texts=[query],
                n_results=min(20, max_results * 3),
                include=["metadatas", "distances"],
            )
            if chroma_results and chroma_results.get("metadatas"):
                for meta in chroma_results["metadatas"][0]:
                    cached_articles.append({
                        "title": meta.get("title", ""),
                        "summary": (meta.get("summary", "") or "")[:200],
                        "source_name": meta.get("source_name", ""),
                        "published_at": meta.get("published_at", ""),
                        "url": meta.get("url", ""),
                    })

    return cached_articles, len(cached_articles)


async def _await_live_news(news_task: asyncio.Task) -> list:
    """Await a live news task, return formatted articles."""
    try:
        news_articles = await asyncio.wait_for(news_task, timeout=8.0)
        return [
            {
                "title": a.title,
                "url": a.url,
                "content": (a.summary or a.content or "")[:200],
            }
            for a in news_articles
        ]
    except Exception:
        return []


async def _fetch_live_news(company_name: str) -> list:
    """Fetch live news via web_intel.company_news. Returns list of NewsArticle."""
    try:
        from app.tools.web.web_intel import company_news
        return await company_news(company_name, max_articles=5, extract_content=False)
    except Exception as e:
        logger.debug(f"Live news fetch failed for '{company_name}': {e}")
        return []


# ── Industry relevance helpers ───────────────────────────────────────

# Synonym map: query term → set of words/phrases that appear in real industry names.
# Used by _industry_relevant() to match enriched company industries against search queries.
# Each entry maps a search term to related industry labels from Yahoo Finance, Wikipedia, etc.
_INDUSTRY_SYNONYMS: dict[str, set[str]] = {
    "automotive": {"automobile", "auto", "vehicle", "car", "motor", "ev", "truck", "tire"},
    "fintech": {"financial technology", "digital payment", "banking", "neobank", "insurtech", "payment", "lending"},
    "semiconductor": {"chip", "silicon", "wafer", "fab", "integrated circuit", "microelectronics", "electronic component"},
    "cybersecurity": {"security", "infosec", "cyber", "threat", "endpoint protection", "identity management"},
    "edtech": {"education technology", "e-learning", "learning", "lms", "courseware", "education"},
    "healthtech": {"health technology", "medtech", "digital health", "telehealth", "medical", "healthcare"},
    "ecommerce": {"e-commerce", "online retail", "marketplace", "online shopping"},
    "saas": {"software as a service", "cloud software", "subscription software"},
    "ai": {"artificial intelligence", "machine learning", "deep learning", "generative ai"},
    "cloud": {"cloud computing", "iaas", "paas", "cloud infrastructure"},
    "logistics": {"supply chain", "freight", "shipping", "warehouse", "transportation"},
    "proptech": {"property technology", "real estate", "realty"},
    "agritech": {"agriculture", "farming", "precision agriculture"},
    "aerospace": {"space", "satellite", "launch", "defense", "aviation", "aircraft"},
    "cleantech": {"clean energy", "renewable", "green", "solar", "wind energy"},
    "biotech": {
        "biotechnology", "biopharmaceutical", "pharmaceutical", "pharma",
        "drug discovery", "genomics", "biologics", "biosimilar", "vaccine",
        "life science", "diagnostics", "bioscience", "therapeutics",
        "clinical research", "biomanufacturing",
    },
    "pharma": {
        "pharmaceutical", "biotechnology", "biopharmaceutical", "drug",
        "vaccine", "clinical", "therapeutics", "specialty pharma", "cdmo",
    },
    "telecom": {"telecommunications", "5g", "wireless", "broadband", "mobile network"},
    "insurance": {"insurtech", "underwriting", "claims", "reinsurance"},
    "manufacturing": {"factory", "industrial", "production", "assembly"},
    "energy": {"oil", "gas", "petroleum", "power generation", "utility"},
    "mining": {"metals", "ore", "mineral", "extraction"},
    "defense": {"military", "defence", "armament", "weapon"},
    "food": {"food", "beverage", "fmcg", "consumer goods", "grocery"},
    "fashion": {"apparel", "clothing", "textile", "garment", "luxury"},
    "gaming": {"video game", "esports", "interactive entertainment"},
    "media": {"entertainment", "streaming", "content", "publishing", "broadcast"},
    "retail": {"consumer", "department store", "supermarket", "discount store"},
}


def _industry_relevant(
    industry: str,
    sub_industries: list[str],
    query: str,
    description: str = "",
    products_services: list[str] | None = None,
) -> bool:
    """Check if a company is genuinely in the searched industry.

    Multi-layer matching (fast to thorough):
      1. Direct containment in industry / sub_industries
      2. Synonym expansion against industry fields
      3. Word overlap in industry fields
      4. Synonym expansion against description + products (catches companies
         whose industry label doesn't match but whose business clearly does)

    Returns True if there's a reasonable match, False if clearly irrelevant.
    Companies with no industry data get benefit of the doubt.
    """
    if not industry:
        return True  # No data → keep

    q = query.lower().strip()
    # Split pipe-separated industries (e.g. "Biopharmaceutical|Biotechnology")
    ind_parts = [p.strip().lower() for p in industry.split("|")]
    ind = " ".join(ind_parts)  # Collapsed for substring matching

    # 1. Direct containment (either direction, against each part)
    for part in ind_parts:
        if q in part or part in q:
            return True

    # 2. Sub-industries check
    all_subs = list(sub_industries or [])
    for sub in all_subs:
        sub_l = sub.lower()
        if q in sub_l or sub_l in q:
            return True

    # 3. Synonym expansion against industry + sub_industries
    q_synonyms = _INDUSTRY_SYNONYMS.get(q, set())
    ind_text = ind + " " + " ".join(s.lower() for s in all_subs)
    for syn in q_synonyms:
        if syn in ind_text:
            return True

    # 4. Word-level overlap in industry fields
    stop = {"and", "the", "of", "in", "for", "&", "a", "an"}
    q_words = set(q.split()) - stop
    ind_words = set(ind.split()) - stop
    if q_words & ind_words:
        return True

    # ── Layer 5: Description + products scan ──────────────────────
    # A company labeled "Specialty Chemicals" might describe itself as
    # "develops biosimilars and vaccines" → clearly biotech.
    # Only check synonyms here (not the raw query, which is too broad
    # for free-text matching — "energy" would match everything).
    if q_synonyms and (description or products_services):
        desc_lower = (description or "").lower()[:500]
        prods = " ".join(p.lower() for p in (products_services or []))
        scan_text = f"{desc_lower} {prods}"
        # Require at least 2 synonym hits in description to be confident
        hits = sum(1 for syn in q_synonyms if syn in scan_text)
        if hits >= 2:
            return True

    return False


# ── A2: LLM batch industry classification ────────────────────────────


async def _llm_classify_industry_batch(
    companies: list[tuple[str, str, str]],  # [(name, industry, description), ...]
    query: str,
) -> dict[str, bool]:
    """LLM batch industry classification for borderline companies.

    Uses GPT-4.1-nano ($0.10/1M tokens). 10 companies ~ $0.001.
    Returns {company_name: is_relevant}.
    """
    from app.tools.llm.llm_service import LLMService

    llm = LLMService(lite=True)

    numbered = "\n".join(
        f"{i + 1}. {name} (labeled: {ind or 'unknown'}): {desc[:150]}"
        for i, (name, ind, desc) in enumerate(companies)
    )

    prompt = (
        f'Classify each company: does it operate in the "{query}" industry?\n'
        f"Consider products, services, R&D focus, customers — not just the label.\n\n"
        f"Companies:\n{numbered}\n\n"
        f"Reply with ONLY the number followed by Y or N (one per line):"
    )

    result = await asyncio.wait_for(
        llm.generate(prompt=prompt, temperature=0.0, max_tokens=100),
        timeout=8.0,
    )

    verdicts: dict[str, bool] = {}
    if result:
        for line in result.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0].rstrip(".")) - 1
                    if 0 <= idx < len(companies):
                        verdicts[companies[idx][0]] = parts[1].upper().startswith("Y")
                except (ValueError, IndexError):
                    continue

    logger.info(
        f"LLM industry classify '{query}': {sum(verdicts.values())}/{len(companies)} accepted"
    )
    return verdicts


# ── Path B: Industry keyword search ──────────────────────────────────


async def _search_by_industry(
    query: str, max_results: int = 10,
) -> tuple[list[CompanySearchResult], int]:
    """Search for companies in an industry.

    Multi-source company discovery with entity validation:
      1. Web search for company names in this industry
      2. LLM extraction of company names from search snippets
      3. DB saved companies matching industry
      4. LLM knowledge fallback
      5. Entity validation (the key step that prevents garbage)
      6. Batch enrichment of validated names
    """
    from app.tools.company_enricher import validate_entity, enrich_batch
    from app.tools.web.web_intel import search as web_search

    cached_count = 0

    # ChromaDB count for response metadata
    article_cache = _get_article_cache()
    if article_cache:
        chroma_results = await _chromadb_query(
            article_cache, query_texts=[query],
            n_results=50,
            include=["metadatas"],
        )
        if chroma_results and chroma_results.get("metadatas"):
            cached_count = len(chroma_results["metadatas"][0])

    from app.config import get_settings
    country = get_settings().country

    # Source 1: Multiple diverse web searches (parallel)
    async def _web_search_safe(q: str) -> list:
        try:
            return await asyncio.wait_for(web_search(q, max_results=15), timeout=10.0)
        except Exception as e:
            logger.debug(f"Web search for '{q}' failed: {e}")
            return []

    # A3: More diverse discovery queries to find companies web search might miss
    search_queries = [
        f"top {query} companies {country} {datetime.now().year}",
        f"best {query} startups {country}",
        f"{query} companies list {datetime.now().year}",
        f"leading {query} firms {country}",
        f"largest {query} companies by revenue",
    ]
    search_tasks = [_web_search_safe(q) for q in search_queries]

    # Source 3: DB saved companies matching industry (parallel with web search)
    db_task = _db_company_names_by_industry(query)

    all_web_results = await asyncio.gather(*search_tasks, db_task, return_exceptions=True)

    # Flatten web results
    web_results = []
    for i, r in enumerate(all_web_results[:-1]):  # Skip last (db_task)
        if isinstance(r, list):
            web_results.extend(r)

    db_names = all_web_results[-1] if isinstance(all_web_results[-1], list) else []

    # Source 2: Extract company names from all web snippets (LLM)
    company_names = await _extract_company_names_from_snippets(web_results, query)

    # Source 4: LLM knowledge fallback
    llm_names = []
    # A3: Lower threshold so LLM fills more gaps when web yields few names
    if len(company_names) + len(db_names) < 5:
        llm_names = await _llm_industry_knowledge(query)

    # Merge and deduplicate (preserving order)
    all_names = list(dict.fromkeys(company_names + db_names + llm_names))

    if not all_names:
        logger.info(f"Industry search '{query}': no company names found from any source")
        return [], cached_count

    # VALIDATE each name (the key step that prevents garbage)
    # DB-cached names skip validation (already validated when first saved)
    db_names_set = set(n.lower() for n in db_names)
    valid_names = [n for n in all_names if n.lower() in db_names_set]
    names_to_validate = [n for n in all_names if n.lower() not in db_names_set]

    validation_sem = asyncio.Semaphore(5)

    async def _validate_one(name: str) -> str | None:
        async with validation_sem:
            try:
                v = await asyncio.wait_for(validate_entity(name), timeout=15.0)
                if v.is_valid_company:
                    return name
            except Exception:
                pass
            return None

    if names_to_validate:
        validation_results = await asyncio.gather(
            *[_validate_one(n) for n in names_to_validate[:20]],
            return_exceptions=True,
        )
        for vr in validation_results:
            if isinstance(vr, str) and vr:
                valid_names.append(vr)

    if not valid_names:
        logger.info(f"Industry search '{query}': no names passed entity validation")
        return [], cached_count

    # Enrich validated names in parallel (bounded concurrency inside enrich_batch)
    enriched_list = await asyncio.wait_for(
        enrich_batch(valid_names[:max_results], skip_validation=True),
        timeout=100.0,
    )

    # A2: Split into pass / borderline, then LLM classify borderline companies
    results = []
    borderline = []
    filtered_out = []
    for e in enriched_list:
        if _industry_relevant(
            e.industry, e.sub_industries, query,
            description=e.description,
            products_services=e.products_services,
        ):
            # Tag the search query as sub_industry for future DB re-discovery
            q_cap = query.strip().capitalize()
            if q_cap.lower() not in [s.lower() for s in (e.sub_industries or [])]:
                e.sub_industries.append(q_cap)
            results.append(_enriched_to_result(
                e, query,
                reason=f"Company in '{query}' sector",
                industry_override=query,
            ))
        else:
            borderline.append(e)

    # LLM batch classification for borderline companies (Layer 6)
    if borderline:
        batch = [
            (e.company_name, e.industry or "", (e.description or "")[:150])
            for e in borderline
        ]
        try:
            verdicts = await _llm_classify_industry_batch(batch, query)
            for e in borderline:
                if verdicts.get(e.company_name, False):
                    q_cap = query.strip().capitalize()
                    if q_cap.lower() not in [s.lower() for s in (e.sub_industries or [])]:
                        e.sub_industries.append(q_cap)
                    results.append(_enriched_to_result(
                        e, query,
                        reason=f"LLM classified as '{query}' sector",
                        industry_override=query,
                    ))
                else:
                    filtered_out.append(f"{e.company_name} ({e.industry})")
        except Exception as e_err:
            logger.debug(f"LLM industry classification failed: {e_err}")
            # On LLM failure, borderline companies are excluded
            for e in borderline:
                filtered_out.append(f"{e.company_name} ({e.industry})")

    if filtered_out:
        logger.info(
            f"Industry search '{query}': filtered {len(filtered_out)} irrelevant: "
            f"{', '.join(filtered_out[:5])}"
        )

    return results[:max_results], cached_count


async def _db_company_names_by_industry(query: str) -> list[str]:
    """Search saved_companies DB for company names matching this industry.

    Searches across: industry, sub_industries, reason_relevant, and description fields.
    """
    names = []
    try:
        from app.database import get_database, SavedCompanyModel
        from sqlalchemy import or_

        db = get_database()
        with db.get_session() as session:
            q_lower = query.lower()
            # Primary: match on industry / sub_industries fields (high precision)
            rows = session.query(SavedCompanyModel).filter(
                or_(
                    SavedCompanyModel.industry.ilike(f"%{q_lower}%"),
                    SavedCompanyModel.sub_industries.ilike(f"%{q_lower}%"),
                    SavedCompanyModel.reason_relevant.ilike(f"%{q_lower}%"),
                )
            ).limit(30).all()

            seen = set()
            for row in rows:
                if row.company_name and row.company_name.lower() not in seen:
                    names.append(row.company_name)
                    seen.add(row.company_name.lower())
    except Exception as e:
        logger.debug(f"DB search for industry '{query}' failed: {e}")
    return names


async def _llm_industry_knowledge(industry: str) -> list[str]:
    """Fallback: Ask LLM for well-known companies in this industry.

    Used when web search, ChromaDB, and DB all return too few results.
    """
    try:
        from app.tools.llm.llm_service import LLMService
        from app.config import get_settings
        country = get_settings().country
        llm = LLMService(lite=True)
        prompt = (
            f"List 10 well-known companies that operate in the '{industry}' industry in {country}. "
            f"Include both {country} companies and major global players with {country} operations.\n\n"
            f"Return ONLY company names as a JSON list. Example: [\"Company A\", \"Company B\"]\n"
            f"Company names only (1-3 words each, no descriptions):"
        )
        response = await asyncio.wait_for(
            llm.generate(prompt=prompt, max_tokens=200), timeout=8.0,
        )
        text = response.strip()
        if "[" in text:
            text = text[text.index("["):text.rindex("]") + 1]
            names = json.loads(text)
            result = [n.strip() for n in names if isinstance(n, str) and len(n.strip()) >= 2]
            logger.debug(f"LLM knowledge for '{industry}': {result}")
            return result
    except Exception as e:
        logger.debug(f"LLM industry knowledge failed for '{industry}': {e}")
    return []


async def _extract_company_names_from_snippets(
    results: list, industry: str,
) -> list[str]:
    """Use LLM nano to extract real company names from web search snippets.

    Accepts web_intel.SearchResult objects (Pydantic models with .title, .snippet).
    """
    if not results:
        return []
    try:
        from app.tools.llm.llm_service import LLMService
        llm = LLMService(lite=True)
        snippets = "\n".join(
            f"- {r.title}: {(r.snippet or '')[:150]}"
            for r in results[:8]
        )
        prompt = (
            f"From these search results about '{industry}', extract ONLY the names of real "
            f"companies/organizations that OPERATE in this industry.\n\n"
            f"Rules:\n"
            f"- Return ONLY company names like 'Paytm', 'Razorpay', 'Zerodha' (1-4 words)\n"
            f"- Do NOT return article titles, website names, news outlets, or generic terms\n"
            f"- Do NOT return the publishing website's name (like F6S, watsoo, mymudra)\n"
            f"- Return as JSON list of strings\n\n"
            f"{snippets}\n\nCompany names (JSON list):"
        )
        response = await asyncio.wait_for(
            llm.generate(prompt=prompt, max_tokens=300),
            timeout=8.0,
        )
        text = response.strip()
        if "[" in text:
            text = text[text.index("["):text.rindex("]") + 1]
            names = json.loads(text)
            return [n.strip() for n in names if isinstance(n, str) and len(n.strip()) >= 2]
    except Exception as e:
        logger.debug(f"LLM company extraction failed: {e}")
    return []


# ── Auto-detect search type (heuristic → LLM intent → entity validation) ─

# Layer 1: Fast heuristic — common industry terms (seed set only, LLM handles rest)
_INDUSTRY_FAST_MATCH = {
    "fintech", "healthtech", "edtech", "proptech", "agritech", "insurtech",
    "regtech", "biotech", "cleantech", "martech", "legaltech", "medtech",
    "saas", "ecommerce", "cybersecurity", "blockchain", "ai", "iot",
    "pharma", "logistics", "manufacturing", "retail", "banking",
    "semiconductor", "aerospace", "energy", "healthcare", "automotive",
    "telecom", "gaming", "robotics", "nanotech", "greentech", "foodtech",
}
_INDUSTRY_SUFFIXES = {"industry", "sector", "companies", "startups", "firms", "market", "solutions"}


def _fast_industry_check(query: str) -> bool:
    """Layer 1: zero-cost heuristic for obvious industry queries."""
    q = query.strip().lower()

    # Exact match against seed set
    if q in _INDUSTRY_FAST_MATCH:
        return True

    # Single word ending in "tech"
    words = q.split()
    if len(words) == 1 and q.endswith("tech") and len(q) > 4:
        return True

    # Query ends with industry suffix ("fintech companies", "ai startups")
    if words and words[-1] in _INDUSTRY_SUFFIXES:
        return True

    return False


async def _llm_classify_intent(query: str) -> str:
    """Layer 2: LLM intent classifier — cheap and fast (~0.5-1s).

    Returns 'company', 'industry', or 'unknown'.
    """
    try:
        from app.tools.llm.llm_service import LLMService
        llm = LLMService(lite=True)
        resp = await asyncio.wait_for(llm.generate(
            f"Classify this search query: \"{query}\"\n"
            f"Is the user looking for:\n"
            f"A) A specific company by name\n"
            f"B) An industry/sector/category to find multiple companies in\n"
            f"C) A product/technology type\n"
            f"Reply with ONLY one word: company, industry, or product",
            temperature=0.0, max_tokens=10,
        ), timeout=5.0)
        if resp:
            r = resp.strip().lower().rstrip(".")
            if "company" in r:
                return "company"
            if "industry" in r or "product" in r:
                return "industry"
    except Exception as e:
        logger.debug(f"LLM intent classify failed: {e}")
    return "unknown"


async def _detect_search_type(query: str) -> tuple[str, Optional[Any]]:
    """Classify query intent, then validate if needed.

    Flow: fast heuristic → LLM intent classify → entity validation (company only).
    This ensures user INTENT is captured first, not just entity existence.
    A query like "fintech" is classified as industry BEFORE we check if fintech.com exists.

    Returns (search_type, enriched_company_or_None).
    """
    # Layer 1: Fast heuristic — known industry terms, patterns
    if _fast_industry_check(query):
        logger.info(f"Intent (heuristic): '{query}' → industry")
        return "industry", None

    # Layer 2: LLM intent classifier — handles everything the heuristic misses
    intent = await _llm_classify_intent(query)

    if intent == "industry":
        logger.info(f"Intent (LLM): '{query}' → industry")
        return "industry", None

    # Layer 3: Entity validation — confirm the company exists
    from app.tools.company_enricher import validate_entity, enrich

    async def _try_validate(label: str):
        result = await validate_entity(query)
        if result.is_valid_company:
            enriched = await enrich(query, skip_validation=True)
            logger.info(f"Intent ({label}+validate): '{query}' → company (source={result.validation_source})")
            return "company", enriched
        return None

    if intent == "company":
        try:
            validated = await _try_validate("LLM")
            if validated:
                return validated
            # LLM said company but validation failed → fall back to industry
            logger.info(f"Intent: '{query}' → LLM said company but validation failed, trying industry")
            return "industry", None
        except Exception as e:
            logger.warning(f"Entity validation failed for '{query}': {e}")
            return "company", None  # Attempt company search without enrichment

    # "unknown" intent — try entity validation
    try:
        validated = await _try_validate("fallback")
        if validated:
            return validated
    except Exception:
        pass

    # Final fallback: heuristic — short capitalized words → company
    words = query.strip().split()
    if len(words) <= 3 and all(w[0].isupper() for w in words if len(w) > 1):
        logger.info(f"Intent (capitalization heuristic): '{query}' → company")
        return "company", None

    logger.info(f"Intent (final fallback): '{query}' → industry")
    return "industry", None


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/search", response_model=CompanySearchResponse)
async def search_companies(req: CompanySearchRequest):
    """Search for companies by name or industry keyword.

    search_type="auto" uses entity validation → LLM → heuristic to auto-detect.
    """
    start = time.time()

    # Mock mode: return saved data
    if req.mock_mode:
        from app.api.mock_loader import find_mock_search
        mock = find_mock_search(req.query)
        if mock:
            return CompanySearchResponse(**mock)

    search_type = req.search_type
    enriched = None
    if search_type == "auto":
        search_type, enriched = await _detect_search_type(req.query)

    try:
        if search_type == "company":
            companies, cached_count = await asyncio.wait_for(
                _search_by_company_name(req.query, enriched=enriched, max_results=req.max_results),
                timeout=30.0,
            )
        else:
            companies, cached_count = await asyncio.wait_for(
                _search_by_industry(req.query, max_results=req.max_results),
                timeout=120.0,
            )
    except asyncio.TimeoutError:
        logger.warning(f"Company search timed out for '{req.query}'")
        companies, cached_count = [], 0
    except Exception as e:
        logger.error(f"Company search failed: {e}", exc_info=True)
        raise HTTPException(500, f"Search failed: {e}")

    elapsed_ms = int((time.time() - start) * 1000)
    response = CompanySearchResponse(
        companies=companies,
        search_type=search_type,
        query=req.query,
        cached_articles_used=cached_count,
        search_duration_ms=elapsed_ms,
    )

    # Persist searched companies to database (with enrichment data)
    try:
        from app.database import get_database
        db = get_database()
        for c in companies:
            db.save_company({
                "id": c.id,
                "company_name": c.company_name,
                "domain": c.domain,
                "website": c.website,
                "industry": c.industry,
                "description": c.description,
                "headquarters": c.headquarters,
                "employee_count": c.employee_count,
                "founded_year": c.founded_year,
                "stock_ticker": c.stock_ticker,
                "ceo": c.ceo,
                "funding_stage": c.funding_stage,
                "reason_relevant": c.reason_relevant,
                "article_count": c.article_count,
                "recent_articles": [dict(a) for a in c.recent_articles],
                "live_news": [dict(n) for n in c.live_news],
                "sub_industries": c.sub_industries,
                "key_people": c.key_people,
                "products_services": c.products_services,
                "competitors": c.competitors,
                "revenue": c.revenue,
                "total_funding": c.total_funding,
                "investors": c.investors,
                "tech_stack": c.tech_stack,
                "validation_source": c.validation_source,
                "search_query": req.query,
                "search_type": search_type,
            })
    except Exception as e:
        logger.warning(f"Failed to persist searched companies: {e}")

    # Background news collection for searched companies
    try:
        from app.tools.web.news_collector import collect_company_news
        for c in companies:
            asyncio.create_task(collect_company_news(c.company_name, months_back=5))
    except Exception:
        pass

    return response


# ── LLM Role Inference ───────────────────────────────────────────


async def _infer_target_roles(
    company_name: str,
    industry: str = "",
    description: str = "",
    trigger_event: str = "",
    news_context: str = "",
) -> list[str]:
    """Use LLM to dynamically infer the best target roles for outreach.

    Returns 6-8 job titles mixing C-suite and VP/Director, specific to the
    company's industry and current situation.  Falls back to broad defaults
    if LLM fails or times out.
    """
    from app.tools.llm.llm_service import LLMService

    settings = get_settings()
    fallback = (
        settings.default_dm_roles.split(",")[:4]
        + settings.default_influencer_roles.split(",")[:4]
    )
    fallback = [r.strip() for r in fallback if r.strip()]

    if settings.contact_role_inference != "llm":
        return fallback

    try:
        llm = LLMService(lite=True)

        context_parts = [f"Company: {company_name}"]
        if industry:
            context_parts.append(f"Industry: {industry}")
        if description:
            context_parts.append(f"About: {description[:200]}")
        if trigger_event:
            context_parts.append(f"Recent trigger: {trigger_event[:200]}")
        if news_context:
            context_parts.append(f"News: {news_context[:200]}")

        prompt = (
            f"Given this company context:\n{chr(10).join(context_parts)}\n\n"
            "List 6-8 job titles of the people most likely to be decision-makers "
            "or influencers for B2B sales outreach to this company.\n"
            "Mix C-suite and VP/Director level. Be specific to their industry.\n"
            "Return as a JSON array of role strings, e.g. "
            '[\"CEO\", \"VP Engineering\", \"Head of Data Science\"]'
        )

        from app.schemas.llm_outputs import ContactRolesLLM
        try:
            llm_result = await asyncio.wait_for(
                llm.run_structured(
                    prompt=prompt,
                    system_prompt="Return valid JSON only.",
                    output_type=ContactRolesLLM,
                ),
                timeout=8.0,
            )
            roles = llm_result.roles
        except Exception:
            raw = await asyncio.wait_for(
                llm.generate_json(prompt=prompt, system_prompt="Return valid JSON only."),
                timeout=8.0,
            )
            roles = []
            if isinstance(raw, list):
                roles = [str(r).strip() for r in raw if r and str(r).strip()]
            elif isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, list):
                        roles = [str(r).strip() for r in v if r and str(r).strip()]
                        break

        if len(roles) >= 3:
            logger.info(f"LLM inferred {len(roles)} target roles for {company_name}")
            return roles[:8]

    except Exception as e:
        logger.debug(f"LLM role inference failed for {company_name}: {e}")

    return fallback


@router.post("/{company_id}/generate-leads", response_model=GenerateLeadsResponse)
async def generate_leads_for_company(company_id: str, req: GenerateLeadsRequest):
    """On-demand lead generation for a company (ContactFinder + EmailGenerator)."""
    start = time.time()

    if not req.company_name:
        raise HTTPException(400, "company_name is required")

    # Mock mode
    if req.mock_mode:
        from app.api.mock_loader import find_mock_leads
        mock = find_mock_leads(req.company_name)
        if mock:
            return GenerateLeadsResponse(**mock)

    from app.schemas.sales import CompanyData
    from app.agents.workers.contact_agent import ContactFinder
    from app.agents.workers.email_agent import EmailGenerator
    from app.agents.leads import _build_person_profiles
    from app.schemas.sales import AgentState
    from app.agents.deps import AgentDeps

    # Quick Tavily health check — log warning if unavailable
    try:
        from app.tools.web.tavily_tool import TavilyTool
        if not TavilyTool().available:
            logger.warning("Tavily unavailable — generate-leads will rely on DDG fallback")
    except Exception:
        logger.warning("Tavily check failed — generate-leads may be limited")

    try:
        deps = AgentDeps(mock_mode=False)
    except Exception as e:
        logger.error(f"AgentDeps creation failed: {e}")
        return GenerateLeadsResponse(
            company_name=req.company_name, contacts=[],
            outreach_count=0,
            reasoning=f"Failed to initialize: {e}",
            duration_ms=int((time.time() - start) * 1000),
        )

    # Infer target roles — frontend override > LLM inference > defaults
    if req.target_roles:
        inferred_roles = req.target_roles
    else:
        inferred_roles = await _infer_target_roles(
            company_name=req.company_name,
            industry=req.industry or "",
            description=req.description or "",
        )

    company = CompanyData(
        id=company_id,
        company_name=req.company_name,
        industry=req.industry or "general",
        domain=req.domain,
        website=f"https://{req.domain}" if req.domain else "",
        reason_relevant=f"On-demand search for {req.company_name}",
        target_roles=inferred_roles,
    )

    reasoning_parts = [f"Target roles: {', '.join(inferred_roles[:4])}..."]
    contacts = []
    outreach_emails = []

    # Step 1: Find contacts (Apollo → web search → Hunter domain_search fallback)
    try:
        finder = ContactFinder(mock_mode=False, deps=deps)
        state = AgentState(companies=[company])
        result = await asyncio.wait_for(finder.find_contacts(state), timeout=120.0)
        contacts = result.contacts or []
        reasoning_parts.append(f"Found {len(contacts)} contacts")
    except asyncio.TimeoutError:
        reasoning_parts.append("Contact search timed out")
    except Exception as e:
        logger.error(f"Contact search failed for {req.company_name}: {e}")
        reasoning_parts.append(f"Contact error: {e}")

    # Step 2: Generate outreach emails (with Hunter verify_email for confidence boost)
    if contacts:
        try:
            generator = EmailGenerator(mock_mode=False, deps=deps)
            state = AgentState(companies=[company], contacts=contacts)
            result = await asyncio.wait_for(generator.process_emails(state), timeout=120.0)
            contacts = result.contacts or contacts
            outreach_emails = result.outreach_emails or []
            reasoning_parts.append(f"Generated {len(outreach_emails)} outreach emails")
        except asyncio.TimeoutError:
            reasoning_parts.append("Email generation timed out")
        except Exception as e:
            logger.error(f"Email gen failed for {req.company_name}: {e}")
            reasoning_parts.append(f"Email error: {e}")

    # Step 3: Build person profiles with reach scores
    profiles = []
    try:
        profiles = _build_person_profiles(
            contacts=contacts, outreach_emails=outreach_emails,
            impacts=[], companies=[company],
        )
        reasoning_parts.append(f"Built {len(profiles)} person profiles")
    except Exception as e:
        logger.error(f"Profile build failed: {e}")
        reasoning_parts.append(f"Profile error: {e}")

    elapsed_ms = int((time.time() - start) * 1000)
    people = [
        PersonResponse(
            person_name=p.person_name, role=p.role,
            seniority_tier=p.seniority_tier, linkedin_url=p.linkedin_url,
            email=p.email, email_confidence=p.email_confidence,
            verified=p.verified, reach_score=p.reach_score,
            outreach_tone=p.outreach_tone,
            outreach_subject=p.outreach_subject, outreach_body=p.outreach_body,
        )
        for p in profiles
    ]

    response = GenerateLeadsResponse(
        company_name=req.company_name, contacts=people,
        outreach_count=len(outreach_emails),
        reasoning=" | ".join(reasoning_parts) or "No data produced",
        duration_ms=elapsed_ms,
    )

    # Persist generated contacts to saved_companies DB
    try:
        from app.database import get_database
        db = get_database()
        db.save_company_contacts(
            company_id,
            [p.model_dump() for p in people],
            reasoning=" | ".join(reasoning_parts),
        )
    except Exception as e:
        logger.warning(f"Failed to persist contacts for {req.company_name}: {e}")

    return response


# ── Company News (on-demand from ChromaDB) ─────────────────────────


@router.get("/{company_id}/news")
async def get_company_news(
    company_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """Fetch company-specific news from ChromaDB (5 months of pipeline-collected articles).

    ChromaDB accumulates articles across pipeline runs. This endpoint queries them
    on-demand — no duplication into SQLite needed.
    """
    # Resolve company name from DB
    from app.database import get_database
    db = get_database()
    company = db.get_saved_company(company_id)
    if not company:
        raise HTTPException(404, f"Company {company_id} not found")

    company_name = company["company_name"]
    article_cache = _get_article_cache()

    all_articles = []
    seen_urls: set[str] = set()

    # Source 1: ChromaDB metadata-filtered query (articles stored with company_name tag)
    # This is the primary path — precise, no cross-company contamination.
    company_lower = company_name.lower()
    try:
        meta_results = await asyncio.wait_for(
            asyncio.to_thread(
                article_cache.news_collection.get,
                where={"company_name": company_lower},
                include=["metadatas"],
                limit=200,
            ),
            timeout=10.0,
        )
        if meta_results and meta_results.get("metadatas"):
            for meta in meta_results["metadatas"]:
                url = meta.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                all_articles.append({
                    "title": meta.get("title", ""),
                    "summary": (meta.get("summary", "") or "")[:300],
                    "source_name": meta.get("source_name", ""),
                    "published_at": meta.get("published_at", ""),
                    "url": url,
                    "sentiment_score": meta.get("sentiment_score", 0.0),
                    "source_type": "cached",
                })
            logger.info(f"News for {company_name}: {len(all_articles)} articles from metadata filter")
    except Exception as e:
        logger.warning(f"ChromaDB metadata query failed for {company_name}: {e}")

    # Source 2: ChromaDB semantic search fallback (older articles without company_name tag)
    # Only if metadata filter returned < 5 articles — fills gaps from pre-tag data.
    if len(all_articles) < 5:
        chroma_results = await _chromadb_query(
            article_cache, query_texts=[company_name],
            n_results=50,
            include=["metadatas", "distances"],
        )
        if chroma_results and chroma_results.get("metadatas"):
            distances = chroma_results.get("distances", [[]])[0]
            for idx, meta in enumerate(chroma_results["metadatas"][0]):
                url = meta.get("url", "")
                if not url or url in seen_urls:
                    continue
                # Relevance gate: company name must appear in title/summary.
                # Stricter for single-word names to avoid false positives.
                title = (meta.get("title", "") or "").lower()
                summary = (meta.get("summary", "") or "").lower()
                name_tokens = [t for t in company_lower.split() if len(t) > 2]

                if len(name_tokens) <= 1:
                    # Single-word company: require full name in title (strict)
                    if company_lower not in title:
                        continue
                else:
                    # Multi-word: at least 60% of significant words in title+summary
                    matched = sum(1 for t in name_tokens if t in title or t in summary)
                    if matched < max(2, int(len(name_tokens) * 0.6)):
                        continue
                seen_urls.add(url)
                all_articles.append({
                    "title": meta.get("title", ""),
                    "summary": (summary or "")[:300],
                    "source_name": meta.get("source_name", ""),
                    "published_at": meta.get("published_at", ""),
                    "url": url,
                    "sentiment_score": meta.get("sentiment_score", 0.0),
                    "source_type": "cached",
                })

    # Source 3: Live news via web_intel (fresh 7-day articles) — with relevance filter
    try:
        from app.tools.web.web_intel import company_news, filter_relevant_articles
        live_articles = await asyncio.wait_for(
            company_news(company_name, max_articles=10, extract_content=False),
            timeout=10.0,
        )
        # Relevance filter to cut noise — only keep articles genuinely about this company
        if live_articles:
            company_data = db.get_saved_company(company_id) or {}
            live_articles = await asyncio.wait_for(
                filter_relevant_articles(
                    company_name, live_articles,
                    domain=company_data.get("domain", ""),
                    industry=company_data.get("industry", ""),
                ),
                timeout=12.0,
            )
        for a in live_articles[:5]:
            url = a.url
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append({
                    "title": a.title,
                    "summary": (a.summary or a.content or "")[:300],
                    "source_name": a.source_name or "web",
                    "published_at": a.published_at.isoformat() if a.published_at else "",
                    "url": url,
                    "sentiment_score": 0.0,
                    "source_type": "live",
                })
    except Exception as e:
        logger.debug(f"Live news fetch for {company_name} failed: {e}")

    # Sort newest first, paginate
    all_articles.sort(key=lambda a: a.get("published_at", ""), reverse=True)
    total = len(all_articles)
    start_idx = (page - 1) * per_page
    paginated = all_articles[start_idx:start_idx + per_page]

    # Trigger background news collection if article count is sparse
    if total < 20:
        try:
            from app.tools.web.news_collector import collect_company_news
            asyncio.create_task(collect_company_news(company_name, months_back=5))
        except Exception:
            pass

    return {
        "articles": paginated,
        "total": total,
        "page": page,
        "per_page": per_page,
        "company_name": company_name,
    }


# ── Saved companies (DB-backed) ──────────────────────────────────────


@router.get("/saved", response_model=SavedCompanyListResponse)
async def list_saved_companies(limit: int = 50):
    """List all saved (previously searched) companies."""
    from app.database import get_database
    db = get_database()
    companies = db.get_saved_companies(limit=limit)
    return SavedCompanyListResponse(
        companies=[SavedCompanyResponse(**c) for c in companies],
        total=len(companies),
    )


@router.get("/saved/{company_id}", response_model=SavedCompanyResponse)
async def get_saved_company(company_id: str):
    """Get a single saved company with its contacts and news.

    Supports lookup by hash ID (primary) or by company name (fallback).
    Pipeline-discovered companies use hash IDs, but frontend may navigate by name.
    """
    from app.database import get_database
    db = get_database()

    # Try direct hash ID lookup first
    company = db.get_saved_company(company_id)

    # Fallback: try as company name (URL-decoded by FastAPI)
    if not company:
        company = db.get_or_enrich_company(company_id, max_age_days=365)

    # Fallback 2: compute hash from name and try that
    if not company:
        name_hash = hashlib.md5(company_id.lower().encode()).hexdigest()[:12]
        company = db.get_saved_company(name_hash)

    if not company:
        raise HTTPException(404, f"Company {company_id} not found")
    return SavedCompanyResponse(**company)
