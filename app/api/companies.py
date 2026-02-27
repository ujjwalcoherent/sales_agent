"""Company search router — real-time search + cached article lookup.

Two search paths:
  A. Company name → SearXNG domain + news, ChromaDB historical articles
  B. Industry keyword → ChromaDB semantic search → extract companies

On-demand lead generation per company via ContactFinder + EmailGenerator.

Reuses: SearchManager, ArticleCache, ContactFinder, EmailGenerator, _build_person_profiles
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    CompanySearchRequest,
    CompanySearchResponse,
    CompanySearchResult,
    GenerateLeadsRequest,
    GenerateLeadsResponse,
    PersonResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MOCK_DIR = Path("./data/mock")

# ── Lazy initializers (avoid import-time side effects) ───────────────


def _get_search_manager():
    from app.search.manager import SearchManager
    from app.config import get_settings
    s = get_settings()
    return SearchManager(searxng_url=s.searxng_url, searxng_enabled=s.searxng_enabled)


def _get_article_cache():
    from app.tools.article_cache import ArticleCache
    return ArticleCache()


# ── Path A: Company name search ──────────────────────────────────────


async def _search_by_company_name(
    query: str, search_mgr, article_cache, max_results: int = 10,
) -> tuple[list[CompanySearchResult], int]:
    """Search by company name — live web + cached articles in parallel."""
    cached_count = 0

    # Parallel: domain resolution + live news + cached articles
    web_task = search_mgr.web_search(f"{query} official website India", max_results=5)
    news_task = search_mgr.company_news_search(query, months=5, max_results=5)
    web_data, live_news = await asyncio.gather(web_task, news_task)

    # Extract domain
    domain = ""
    from app.tools.domain_utils import extract_clean_domain
    for r in web_data.get("results", []):
        url = r.get("url", "")
        if url:
            d = extract_clean_domain(url)
            if d:
                domain = d
                break

    # ChromaDB semantic search for historical articles
    cached_articles = []
    try:
        chroma_results = article_cache.collection.query(
            query_texts=[query],
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
            cached_count = len(cached_articles)
    except Exception as e:
        logger.warning(f"ChromaDB query failed for '{query}': {e}")

    cid = hashlib.md5(query.encode()).hexdigest()[:12]
    result = CompanySearchResult(
        id=cid,
        company_name=query,
        domain=domain,
        website=f"https://{domain}" if domain else "",
        industry="",
        reason_relevant=f"Direct search for '{query}'",
        article_count=len(cached_articles),
        recent_articles=cached_articles[:5],
        live_news=[
            {"title": n.get("title", ""), "url": n.get("url", ""), "content": (n.get("content", "") or "")[:200]}
            for n in live_news[:5]
        ],
    )

    return [result], cached_count


# ── Path B: Industry keyword search ──────────────────────────────────


async def _search_by_industry(
    query: str, search_mgr, article_cache, max_results: int = 10,
) -> tuple[list[CompanySearchResult], int]:
    """Search by industry keyword — ChromaDB semantic → extract company entities."""
    cached_count = 0
    company_articles: dict[str, list] = {}

    # ChromaDB: find articles matching the industry keyword
    try:
        chroma_results = article_cache.collection.query(
            query_texts=[query],
            n_results=min(50, max_results * 5),
            include=["metadatas", "distances"],
        )
        if chroma_results and chroma_results.get("metadatas"):
            for meta in chroma_results["metadatas"][0]:
                cached_count += 1
                entities = []
                if "entity_names_json" in meta:
                    try:
                        entities = json.loads(meta["entity_names_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                snippet = {
                    "title": meta.get("title", ""),
                    "summary": (meta.get("summary", "") or "")[:200],
                    "source_name": meta.get("source_name", ""),
                    "published_at": meta.get("published_at", ""),
                    "url": meta.get("url", ""),
                }

                # Skip short/generic entity names
                _skip = {"india", "the", "government", "rbi", "sebi", "ministry", "delhi",
                          "mumbai", "new delhi", "reuters", "bloomberg", "et", "pti"}
                for entity in entities:
                    if len(entity) < 3 or entity.lower() in _skip:
                        continue
                    company_articles.setdefault(entity, []).append(snippet)
    except Exception as e:
        logger.warning(f"ChromaDB industry query failed for '{query}': {e}")

    # Also run live web search for trending companies in this space
    try:
        web_data = await search_mgr.web_search(f"{query} companies India 2026", max_results=10)
        for r in web_data.get("results", []):
            title = r.get("title", "")
            if title and len(title) > 3 and len(title) < 60:
                company_articles.setdefault(title, [])
    except Exception as e:
        logger.debug(f"Web search for industry '{query}' failed: {e}")

    # Rank by article count, take top N
    sorted_companies = sorted(
        company_articles.items(), key=lambda x: len(x[1]), reverse=True,
    )[:max_results]

    # Resolve domains in parallel (bounded concurrency)
    sem = asyncio.Semaphore(5)

    async def _resolve_one(name: str, articles: list) -> CompanySearchResult:
        domain = ""
        try:
            async with sem:
                from app.agents.lead_gen import _resolve_domain
                domain = await _resolve_domain(search_mgr, name)
        except Exception:
            pass
        cid = hashlib.md5(name.encode()).hexdigest()[:12]
        return CompanySearchResult(
            id=cid,
            company_name=name,
            domain=domain,
            website=f"https://{domain}" if domain else "",
            industry=query,
            reason_relevant=f"Found in {len(articles)} articles about '{query}'",
            article_count=len(articles),
            recent_articles=articles[:5],
        )

    tasks = [_resolve_one(name, articles) for name, articles in sorted_companies]
    results = await asyncio.gather(*tasks) if tasks else []
    return list(results), cached_count


# ── Auto-detect search type ──────────────────────────────────────────


def _detect_search_type(query: str) -> str:
    """Heuristic: short capitalized → company, long/keyword-ish → industry."""
    words = query.strip().split()
    if not words:
        return "company"
    industry_words = {"industry", "sector", "market", "regulation", "fintech", "technology",
                      "manufacturing", "healthcare", "energy", "agriculture", "companies",
                      "startup", "startups", "digital", "automation", "ai", "pharma"}
    if any(w.lower() in industry_words for w in words):
        return "industry"
    if len(words) <= 2 and all(w[0].isupper() for w in words if w):
        return "company"
    return "industry" if len(words) > 3 else "company"


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/search", response_model=CompanySearchResponse)
async def search_companies(req: CompanySearchRequest):
    """Search for companies by name or industry keyword.

    search_type="auto" auto-detects.
    """
    start = time.time()

    # Mock mode: return saved data
    if req.mock_mode:
        from app.api.mock_loader import find_mock_search
        mock = find_mock_search(req.query)
        if mock:
            return CompanySearchResponse(**mock)

    search_type = req.search_type
    if search_type == "auto":
        search_type = _detect_search_type(req.query)

    search_mgr = _get_search_manager()
    article_cache = _get_article_cache()

    try:
        if search_type == "company":
            companies, cached_count = await asyncio.wait_for(
                _search_by_company_name(req.query, search_mgr, article_cache, req.max_results),
                timeout=30.0,
            )
        else:
            companies, cached_count = await asyncio.wait_for(
                _search_by_industry(req.query, search_mgr, article_cache, req.max_results),
                timeout=60.0,
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

    # Auto-save for mock data
    _save_mock_result("company_search", {
        "query": req.query, "search_type": search_type,
        "response": response.model_dump(),
    })

    return response


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
    from app.agents.lead_gen import _build_person_profiles
    from app.schemas.sales import AgentState
    from app.agents.deps import AgentDeps

    deps = AgentDeps(mock_mode=False)
    company = CompanyData(
        id=company_id,
        company_name=req.company_name,
        industry=req.industry or "general",
        domain=req.domain,
        website=f"https://{req.domain}" if req.domain else "",
        reason_relevant=f"On-demand search for {req.company_name}",
    )

    reasoning_parts = []
    contacts = []
    outreach_emails = []

    # Step 1: Find contacts (Apollo → SearXNG fallback)
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

    # Step 2: Generate outreach emails
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

    _save_mock_result("lead_gen", {
        "company_name": req.company_name, "company_id": company_id,
        "response": response.model_dump(),
    })
    return response


# ── Mock data persistence ─────────────────────────────────────────────


def _save_mock_result(category: str, data: dict) -> None:
    """Append result to mock data file for demo mode."""
    try:
        MOCK_DIR.mkdir(parents=True, exist_ok=True)
        mock_file = MOCK_DIR / f"{category}.json"
        existing = []
        if mock_file.exists():
            try:
                existing = json.loads(mock_file.read_text())
            except (json.JSONDecodeError, TypeError):
                existing = []
        existing.append(data)
        existing = existing[-20:]  # Keep last 20 per category
        mock_file.write_text(json.dumps(existing, indent=2, default=str))
    except Exception as e:
        logger.debug(f"Mock save failed: {e}")
