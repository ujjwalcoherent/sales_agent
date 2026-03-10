"""Entity-validated company enrichment — the CENTRAL gatekeeper.

No company is returned to the user unless it passes entity validation.
Validation = confirmed as a business entity via at least one of:
  1. Tavily search + LLM classify (is this a business?)
  2. Apollo org data exists for the company's domain
  3. Has a valid corporate website domain

This module ORCHESTRATES existing tools — it does not re-implement their logic:
  - app.tools.tavily_tool   — Tavily search, news, finance, extract
  - app.tools.apollo_tool   — Apollo people search + org data cache
  - app.tools.domain_utils  — domain extraction / validation
  - app.tools.web_intel     — web search for domain resolution
  - app.tools.llm_service   — LLM gap-fill (last resort)
  - app.database            — Company KB (SQLite cache)

Usage:
    from app.tools.company_enricher import validate_entity, enrich, enrich_batch

    result = await validate_entity("NVIDIA")           # Quick check
    company = await enrich("NVIDIA")                   # Full enrichment
    companies = await enrich_batch(["NVIDIA", "Tesla"]) # Parallel batch
"""

import asyncio
import logging
import re
import unicodedata
from typing import Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Heuristic rejection patterns ─────────────────────────────────────────
# Strings that look like sentences, lists, or non-entity text
_REJECT_PATTERNS = [
    re.compile(r"[?!;]"),                       # sentence punctuation
    re.compile(r"^\d+\s+(top|best|ways|things)", re.I),  # listicle titles
    re.compile(r"^(how|why|what|when|where)\s", re.I),   # questions
    re.compile(r"\b(temple|church|mosque|school|university)\b", re.I),  # non-business entities
    re.compile(r"^\s*$"),                        # empty / whitespace-only
    re.compile(r"(scan|download)\s+(qr|app|pdf)", re.I),  # web scraping garbage
    re.compile(r"^(click|tap|subscribe|sign up|buy now|add to cart)", re.I),  # CTA buttons
    re.compile(r"cookie|gdpr|privacy policy|terms of (service|use)", re.I),  # legal boilerplate
]

_MAX_NAME_LENGTH = 60  # Company names >60 chars are almost never real entities


# ── Models ────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Result of entity validation."""
    is_valid_company: bool = False
    confidence: float = 0.0              # 0-1
    validation_source: str = ""          # "tavily" | "apollo" | "domain" | "llm"
    rejection_reason: str = ""           # If invalid, why


class EnrichedCompany(BaseModel):
    """Fully enriched company profile."""
    # Identity
    company_name: str
    wikidata_id: str = ""  # Legacy field — kept for backward compatibility with DB/schemas
    domain: str = ""
    website: str = ""
    # Validation
    validation: ValidationResult = Field(default_factory=ValidationResult)
    # Profile
    description: str = ""
    industry: str = ""
    sub_industries: list[str] = Field(default_factory=list)
    founded_year: Optional[int] = None
    headquarters: str = ""
    ceo: str = ""
    key_people: list[dict] = Field(default_factory=list)
    employee_count: str = ""
    revenue: str = ""
    stock_ticker: str = ""
    stock_exchange: str = ""
    funding_stage: str = ""
    total_funding: str = ""
    investors: list[str] = Field(default_factory=list)
    products_services: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    subsidiaries: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    # Sources
    data_sources: list[str] = Field(default_factory=list)


# ── Heuristic pre-filter ─────────────────────────────────────────────────

def _heuristic_reject(name: str) -> str:
    """Quick heuristic check — returns rejection reason or empty string if OK."""
    if not name or not name.strip():
        return "empty name"
    stripped = name.strip()
    if len(stripped) > _MAX_NAME_LENGTH:
        return f"name too long ({len(stripped)} chars)"
    if len(stripped) < 2:
        return "name too short"
    for pattern in _REJECT_PATTERNS:
        if pattern.search(stripped):
            return f"matches rejection pattern: {pattern.pattern}"
    return ""


# ── Data Sanitization ────────────────────────────────────────────────────

# Patterns shared with _REJECT_PATTERNS are omitted (DRY).
# These are specific to enrichment text sanitization (prices, web scraping).
_GARBAGE_PATTERNS = _REJECT_PATTERNS + [
    re.compile(r"₹\s*\d+/|^\$?\d+(\.\d{2})?\s*(off|discount)", re.I),
    re.compile(r"loading\.\.\.|javascript|undefined", re.I),
]


def _sanitize_text_field(value: str, max_length: int = 500) -> str:
    """Return empty string if value is web scraping garbage."""
    if not value or len(value.strip()) < 3:
        return ""
    for pat in _GARBAGE_PATTERNS:
        if pat.search(value):
            return ""
    # Reject if less than 50% alphanumeric/space (binary, encoded, or garbled text)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in value) / max(len(value), 1)
    if alpha_ratio < 0.5 and len(value) > 10:
        return ""
    return value.strip()[:max_length]


# ── Entity Validation ─────────────────────────────────────────────────────

async def validate_entity(name: str) -> ValidationResult:
    """Validate that a name refers to a real business entity.

    Checks (in order of confidence):
      1. Heuristic pre-filter (reject garbage strings)
      2. Tavily search + LLM classify (confidence 0.9)
      3. Domain validation — has valid corporate domain (confidence 0.7)

    Must pass AT LEAST ONE real check to be valid.
    """
    # Step 0: Heuristic pre-filter
    reject_reason = _heuristic_reject(name)
    if reject_reason:
        return ValidationResult(
            is_valid_company=False,
            confidence=0.0,
            rejection_reason=reject_reason,
        )

    # Step 1: Tavily search + AI answer classification
    try:
        from .web.tavily_tool import TavilyTool

        tavily = TavilyTool()
        if tavily.available:
            result = await asyncio.wait_for(
                tavily.search(
                    query=f"{name} company",
                    max_results=3,
                    include_answer=True,
                    search_depth="basic",
                ),
                timeout=8.0,
            )

            answer = result.get("answer", "")
            results = result.get("results", [])

            # Check if the AI answer or search results describe a business
            if answer and _answer_describes_business(name, answer):
                return ValidationResult(
                    is_valid_company=True,
                    confidence=0.9,
                    validation_source="tavily",
                )

            # Check result titles/snippets for business indicators
            if results and _results_indicate_business(name, results):
                return ValidationResult(
                    is_valid_company=True,
                    confidence=0.85,
                    validation_source="tavily",
                )
    except asyncio.TimeoutError:
        logger.debug(f"Tavily validation timed out for '{name}'")
    except Exception as e:
        logger.debug(f"Tavily validation failed for '{name}': {e}")

    # Step 2: Clearbit Autocomplete — FREE, no auth, instant (confidence 0.9)
    clearbit_result = await _clearbit_validate(name)
    if clearbit_result:
        return clearbit_result

    # Step 3: DDG search + business keyword heuristic (confidence 0.8)
    ddg_result = await _ddg_validate(name)
    if ddg_result:
        return ddg_result

    # Step 4: LLM knowledge check (confidence 0.65)
    llm_result = await _llm_knowledge_validate(name)
    if llm_result:
        return llm_result

    # Step 5: Domain check (medium confidence 0.7)
    try:
        from .domain_utils import extract_domain_from_company_name, is_valid_company_domain

        guessed_domain = extract_domain_from_company_name(name)
        if guessed_domain and is_valid_company_domain(guessed_domain):
            return ValidationResult(
                is_valid_company=True,
                confidence=0.7,
                validation_source="domain",
            )
    except Exception as e:
        logger.debug(f"Domain validation failed for '{name}': {e}")

    # No check passed — invalid
    return ValidationResult(
        is_valid_company=False,
        confidence=0.0,
        rejection_reason="failed all validation checks (Tavily, Clearbit, DDG, LLM, domain all failed)",
    )


_STRIP_SUFFIXES = re.compile(
    r"\b(inc\.?|ltd\.?|llc|corp\.?|corporation|limited|pvt\.?|private|plc|gmbh|sa|ag|co\.?)\b",
    re.IGNORECASE,
)


def _names_match(a: str, b: str) -> bool:
    """Fuzzy company name match — strips suffixes, normalizes whitespace."""
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s).lower().strip()
        s = _STRIP_SUFFIXES.sub("", s).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    # Exact after normalization
    if na == nb:
        return True
    # One contains the other
    if na in nb or nb in na:
        return True
    # Token overlap: at least 80% of shorter's tokens in longer
    ta, tb = set(na.split()), set(nb.split())
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if shorter and len(shorter & longer) / len(shorter) >= 0.8:
        return True
    return False


async def _clearbit_validate(name: str) -> Optional[ValidationResult]:
    """Validate company via Clearbit Autocomplete API — FREE, no auth, instant."""
    try:
        import httpx

        url = "https://autocomplete.clearbit.com/v1/companies/suggest"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"query": name})
            if resp.status_code == 200:
                results = resp.json()
                for r in results:
                    if _names_match(name, r.get("name", "")):
                        return ValidationResult(
                            is_valid_company=True,
                            confidence=0.9,
                            validation_source="clearbit",
                        )
    except Exception as e:
        logger.debug(f"Clearbit validation failed for '{name}': {e}")
    return None


async def _ddg_validate(name: str) -> Optional[ValidationResult]:
    """Validate company via DDG search + business keyword heuristic."""
    try:
        from .web.web_intel import search as web_search

        ddg_results = await asyncio.wait_for(
            web_search(f"{name} company", max_results=5),
            timeout=8.0,
        )
        if ddg_results:
            combined = " ".join(
                f"{r.get('title', '')} {r.get('body', '') if isinstance(r, dict) else getattr(r, 'body', '')}"
                for r in ddg_results[:3]
            ).lower()
            if _answer_describes_business(name, combined):
                return ValidationResult(
                    is_valid_company=True,
                    confidence=0.8,
                    validation_source="ddg",
                )
    except asyncio.TimeoutError:
        logger.debug(f"DDG validation timed out for '{name}'")
    except Exception as e:
        logger.debug(f"DDG validation failed for '{name}': {e}")
    return None


async def _llm_knowledge_validate(name: str) -> Optional[ValidationResult]:
    """Validate company via LLM knowledge check — cheap, fast."""
    try:
        from .llm.llm_service import LLMService

        llm = LLMService(lite=True)
        resp = await asyncio.wait_for(
            llm.generate(
                f"Is '{name}' a real company or business organization? Reply YES or NO only.",
                temperature=0.0,
                max_tokens=5,
            ),
            timeout=5.0,
        )
        if resp and "yes" in resp.strip().lower():
            return ValidationResult(
                is_valid_company=True,
                confidence=0.65,
                validation_source="llm",
            )
    except Exception as e:
        logger.debug(f"LLM knowledge validation failed for '{name}': {e}")
    return None


def _answer_describes_business(name: str, answer: str) -> bool:
    """Check if a Tavily AI answer describes a business entity.

    Returns False if the answer contains negative signals (e.g. "does not exist",
    "placeholder", "fictional") even if business keywords are present.
    """
    answer_lower = answer.lower()
    name_lower = name.lower()

    # Must mention the company name (or significant parts of it)
    if name_lower not in answer_lower and not any(
        part in answer_lower for part in name_lower.split()[:2] if len(part) > 2
    ):
        return False

    # Negative signals — answer says the entity doesn't exist
    negative_signals = [
        "does not exist", "doesn't exist", "not a real", "not a company",
        "placeholder", "fictional", "no such company", "no company named",
        "could not find", "cannot find", "no information", "not found",
        "appears to be fake", "is not a known",
    ]
    # Check negatives specifically near the company name
    name_context_start = max(0, answer_lower.find(name_lower[:8].lower()) - 20)
    name_context = answer_lower[name_context_start:name_context_start + len(name_lower) + 200]
    if any(neg in name_context for neg in negative_signals):
        return False

    # Business indicator keywords — must appear near or about the company, not in unrelated text
    business_keywords = [
        "company", "corporation", "inc.", "ltd", "founded", "headquartered",
        "ceo", "revenue", "employees", "startup", "enterprise", "firm",
        "manufacturer", "provider", "platform", "services", "solutions",
        "technology", "technologies", "industries", "group", "holdings",
    ]

    # Count how many business keywords appear within proximity of the company name
    hits = sum(1 for kw in business_keywords if kw in name_context)
    return hits >= 2  # Require at least 2 business keywords (not just "company")


def _results_indicate_business(name: str, results: list[dict]) -> bool:
    """Check if search results indicate a business entity.

    Requires the company name to appear in result text alongside business
    indicators. Generic results about other companies don't count.
    """
    name_lower = name.lower()
    name_parts = [p for p in name_lower.split() if len(p) > 2]
    business_signals = 0

    for r in results[:3]:
        title = (r.get("title", "") or "").lower()
        content = (r.get("content", "") or "").lower()
        url = (r.get("url", "") or "").lower()
        text = f"{title} {content}"

        # Name must appear in at least one result
        has_name = name_lower in text or (
            name_parts and sum(1 for p in name_parts if p in text) >= max(1, len(name_parts) // 2)
        )
        if not has_name:
            continue

        # Business indicators — must be in same result that mentions the name
        biz_kws = ["company", "founded", "ceo", "revenue", "employees", "startup",
                    "headquartered", "corporation", "platform", "solutions"]
        hits = sum(1 for kw in biz_kws if kw in text)
        if hits >= 2:  # Require 2+ business keywords per result
            business_signals += 1
        # LinkedIn company page
        if "linkedin.com/company" in url:
            business_signals += 1
        # Corporate domain in results
        if any(part in url for part in name_lower.split()[:1] if len(part) > 2):
            business_signals += 1

    return business_signals >= 2


# ── Helper: merge Tavily research into EnrichedCompany ────────────────────

def _merge_tavily(company: EnrichedCompany, research: dict) -> None:
    """Merge Tavily deep_company_research() result into an EnrichedCompany."""
    if not research:
        return

    # Main answer → description
    if not company.description and research.get("answer"):
        company.description = research["answer"]

    # Extract domain from search results
    if not company.domain:
        from .domain_utils import extract_clean_domain, is_valid_company_domain
        name_lower = company.company_name.lower().replace(" ", "")
        for r in research.get("results", []):
            url = r.get("url", "")
            if url:
                domain = extract_clean_domain(url)
                if domain and is_valid_company_domain(domain):
                    domain_base = domain.split(".")[0].lower()
                    if domain_base in name_lower or name_lower in domain_base or len(name_lower) <= 4:
                        company.domain = domain
                        company.website = f"https://{domain}"
                        break

    # Finance data
    finance = research.get("finance", {})
    if finance.get("answer"):
        finance_text = finance["answer"]
        # Try to extract revenue, market cap from finance answer
        if not company.revenue:
            company.revenue = _extract_field_from_text(finance_text, ["revenue", "sales"])
        if not company.stock_ticker:
            company.stock_ticker = _extract_field_from_text(finance_text, ["ticker", "NYSE:", "NASDAQ:", "BSE:", "NSE:"])

    if "tavily" not in company.data_sources:
        company.data_sources.append("tavily")


def _extract_field_from_text(text: str, keywords: list[str]) -> str:
    """Extract a value near a keyword from text. Simple heuristic."""
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx >= 0:
            # Extract ~50 chars after the keyword
            snippet = text[idx:idx + 80]
            # Clean up to just the value portion
            parts = snippet.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip().split(".")[0].strip()[:50]
    return ""


def _merge_apollo_org(company: EnrichedCompany, org: dict) -> None:
    """Merge Apollo organization data into an EnrichedCompany (fill blanks only)."""
    if not org:
        return
    if not company.industry and org.get("industry"):
        company.industry = org["industry"]
    if not company.employee_count and org.get("employee_count"):
        company.employee_count = str(org["employee_count"])
    if company.founded_year is None and org.get("founded_year"):
        company.founded_year = org["founded_year"]
    if not company.headquarters and org.get("headquarters"):
        company.headquarters = org["headquarters"]
    if not company.description and org.get("description"):
        company.description = org["description"]
    if not company.website and org.get("website"):
        company.website = org["website"]
    if not company.funding_stage and org.get("funding_stage"):
        company.funding_stage = org["funding_stage"]
    if not company.total_funding and org.get("funding_total"):
        company.total_funding = str(org["funding_total"])
    if not company.tech_stack and org.get("tech_stack"):
        company.tech_stack = org["tech_stack"]
    if "apollo" not in company.data_sources:
        company.data_sources.append("apollo")


def _merge_kb_cache(company: EnrichedCompany, cached: dict) -> None:
    """Merge cached Company KB dict into an EnrichedCompany (fill blanks only)."""
    if not cached:
        return
    if not company.domain and cached.get("domain"):
        company.domain = cached["domain"]
    if not company.website and cached.get("website"):
        company.website = cached["website"]
    if not company.description and cached.get("description"):
        company.description = cached["description"]
    if not company.industry and cached.get("industry"):
        company.industry = cached["industry"]
    if company.founded_year is None and cached.get("founded_year"):
        company.founded_year = cached["founded_year"]
    if not company.headquarters and cached.get("headquarters"):
        company.headquarters = cached["headquarters"]
    if not company.ceo and cached.get("ceo"):
        company.ceo = cached["ceo"]
    if not company.employee_count and cached.get("employee_count"):
        company.employee_count = cached["employee_count"]
    if not company.stock_ticker and cached.get("stock_ticker"):
        company.stock_ticker = cached["stock_ticker"]
    if not company.funding_stage and cached.get("funding_stage"):
        company.funding_stage = cached["funding_stage"]
    if "kb_cache" not in company.data_sources:
        company.data_sources.append("kb_cache")


# ── Domain Resolution ─────────────────────────────────────────────────────

async def _resolve_domain(company: EnrichedCompany) -> None:
    """Resolve a company's domain from multiple sources.

    Priority: website field → web search → name guess.
    Sets both company.domain and company.website.
    """
    from .domain_utils import extract_clean_domain, is_valid_company_domain, extract_domain_from_company_name

    # If website is already set, extract domain from it
    if company.website and not company.domain:
        domain = extract_clean_domain(company.website)
        if domain and is_valid_company_domain(domain):
            company.domain = domain
            return

    # If domain is already set, done
    if company.domain and is_valid_company_domain(company.domain):
        if not company.website:
            company.website = f"https://{company.domain}"
        return

    # Try web search for domain
    try:
        from .web.web_intel import search as web_search

        results = await asyncio.wait_for(
            web_search(f"{company.company_name} official website", max_results=3),
            timeout=6.0,
        )
        for r in results:
            url = r.url
            if url:
                domain = extract_clean_domain(url)
                if domain and is_valid_company_domain(domain):
                    # Verify it's not a news/social site by checking name is in domain
                    name_lower = company.company_name.lower().replace(" ", "")
                    domain_base = domain.split(".")[0].lower()
                    if domain_base in name_lower or name_lower in domain_base or len(name_lower) <= 4:
                        company.domain = domain
                        company.website = f"https://{domain}"
                        return
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug(f"Web search for domain failed for '{company.company_name}': {e}")

    # Fallback: guess domain from company name
    guessed = extract_domain_from_company_name(company.company_name)
    if guessed and is_valid_company_domain(guessed):
        company.domain = guessed
        if not company.website:
            company.website = f"https://{guessed}"


# ── LLM Gap-Fill ──────────────────────────────────────────────────────────

async def _llm_gap_fill(company: EnrichedCompany) -> None:
    """Use LLM to fill the description if still empty. Last resort.

    Only fills description — other fields should come from structured sources.
    Uses lite mode (cheapest model) since this is a simple task.
    """
    if company.description:
        return  # Already has description, skip

    try:
        from .llm.llm_service import LLMService

        llm = LLMService(lite=True)
        prompt = (
            f"Write a concise 2-3 sentence factual description of the company '{company.company_name}'."
            f" Include what industry they operate in, what products/services they offer,"
            f" and approximately when they were founded if known."
            f" Be factual — do not speculate. If you don't know, say so."
        )
        description = await asyncio.wait_for(
            llm.generate(prompt, temperature=0.2, max_tokens=300),
            timeout=10.0,
        )
        if description and len(description.strip()) > 20:
            cleaned = _sanitize_text_field(description, 1000)
            # Verify the description actually references the company
            if cleaned and company.company_name.lower().split()[0] in cleaned.lower():
                company.description = cleaned
                if "llm" not in company.data_sources:
                    company.data_sources.append("llm")
    except Exception as e:
        logger.debug(f"LLM gap-fill failed for '{company.company_name}': {e}")


# ── Structured Field Extraction ───────────────────────────────────────────

async def _extract_structured_fields(company: EnrichedCompany, text: str) -> None:
    """Use LLM to extract ALL available structured fields from Tavily answer text.

    Extracts everything possible: industry, headquarters, CEO, employee count,
    founded year, products/services, competitors, sub-industries, tech stack,
    investors, funding stage. Only fills fields that are still empty.
    """
    if not text or len(text) < 50:
        return

    # Build dynamic field list based on what's missing
    field_defs = {
        "industry": ("string", not company.industry),
        "headquarters": ("string", not company.headquarters),
        "ceo": ("string", not company.ceo),
        "employee_count": ("string", not company.employee_count),
        "founded_year": ("integer", company.founded_year is None),
        "products_services": ("list of strings", not company.products_services),
        "competitors": ("list of strings", not company.competitors),
        "sub_industries": ("list of strings", not company.sub_industries),
        "tech_stack": ("list of strings", not company.tech_stack),
        "investors": ("list of strings", not company.investors),
        "funding_stage": ("string", not company.funding_stage),
        "revenue": ("string", not company.revenue),
        "stock_ticker": ("string", not company.stock_ticker),
    }

    missing = {k: t for k, (t, needed) in field_defs.items() if needed}
    if not missing:
        return

    try:
        from .llm.llm_service import LLMService

        llm = LLMService(lite=True)

        field_desc = ", ".join(f"{k} ({t})" for k, t in missing.items())
        prompt = (
            f"Extract all available business intelligence from this text about '{company.company_name}'.\n"
            f"Return a JSON object with any of these fields you can find: {field_desc}\n"
            f"Use null for fields not found. For list fields, return arrays.\n\n"
            f"Text: {text[:2500]}\n\nJSON:"
        )
        response = await asyncio.wait_for(
            llm.generate(prompt, temperature=0.1, max_tokens=500),
            timeout=10.0,
        )

        if not response:
            return

        from app.tools.json_repair import parse_json_response
        data = parse_json_response(response)
        if not data or data.get("error"):
            return

        def _to_list(val) -> list[str]:
            if isinstance(val, list):
                return [str(v).strip() for v in val if v and str(v).strip()]
            if isinstance(val, str) and val.strip():
                return [s.strip() for s in val.split(",") if s.strip()]
            return []

        # Fill string fields (sanitize LLM output)
        if data.get("industry"):
            llm_ind = _sanitize_text_field(str(data["industry"]), 200)
            if not company.industry:
                company.industry = llm_ind
            elif llm_ind.lower() != company.industry.lower():
                # A1: LLM may extract a different/broader label → store as sub_industry
                if llm_ind not in (company.sub_industries or []):
                    company.sub_industries.append(llm_ind)
        if not company.headquarters and data.get("headquarters"):
            company.headquarters = _sanitize_text_field(str(data["headquarters"]), 200)
        if not company.ceo and data.get("ceo"):
            company.ceo = _sanitize_text_field(str(data["ceo"]), 100)
        if not company.employee_count and data.get("employee_count"):
            company.employee_count = _sanitize_text_field(str(data["employee_count"]), 50)
        if not company.revenue and data.get("revenue"):
            company.revenue = _sanitize_text_field(str(data["revenue"]), 100)
        if not company.stock_ticker and data.get("stock_ticker"):
            company.stock_ticker = _sanitize_text_field(str(data["stock_ticker"]), 20)
        if not company.funding_stage and data.get("funding_stage"):
            company.funding_stage = _sanitize_text_field(str(data["funding_stage"]), 50)

        # Fill integer fields
        if company.founded_year is None and data.get("founded_year"):
            try:
                company.founded_year = int(data["founded_year"])
            except (ValueError, TypeError):
                pass

        # Fill list fields
        if not company.products_services and data.get("products_services"):
            company.products_services = _to_list(data["products_services"])
        if not company.competitors and data.get("competitors"):
            company.competitors = _to_list(data["competitors"])
        if data.get("sub_industries"):
            # A1: Merge LLM sub_industries with existing (don't overwrite)
            for si in _to_list(data["sub_industries"]):
                if si not in (company.sub_industries or []):
                    company.sub_industries.append(si)
        if not company.tech_stack and data.get("tech_stack"):
            company.tech_stack = _to_list(data["tech_stack"])
        if not company.investors and data.get("investors"):
            company.investors = _to_list(data["investors"])

    except Exception as e:
        logger.debug(f"Structured field extraction failed for '{company.company_name}': {e}")


# ── yfinance Enrichment (public companies) ───────────────────────────────

async def _enrich_from_yfinance(company: EnrichedCompany) -> None:
    """Enrich from Yahoo Finance — FREE, no auth. Works for public companies.

    Provides: sector, industry, employees, description, website, officers.
    Runs in thread pool since yfinance is synchronous.
    """
    try:
        import yfinance as yf

        name = company.company_name

        def _yf_lookup():
            # Try direct ticker if we have one
            if company.stock_ticker:
                t = yf.Ticker(company.stock_ticker)
                info = t.info
                if info.get("longBusinessSummary"):
                    return info

            # Try searching by company name → ticker
            try:
                search = yf.Search(name)
                quotes = search.quotes if hasattr(search, "quotes") else []
                for q in quotes[:3]:
                    symbol = q.get("symbol", "")
                    if symbol:
                        t = yf.Ticker(symbol)
                        info = t.info
                        if info.get("longBusinessSummary"):
                            return info
            except Exception:
                pass
            return {}

        info = await asyncio.wait_for(
            asyncio.to_thread(_yf_lookup), timeout=15.0,
        )

        if not info or not info.get("longBusinessSummary"):
            return

        # Fill only empty fields — structured data preferred over LLM-extracted
        if not company.industry and info.get("industry"):
            company.industry = info["industry"]
        # A1: Store sector as sub_industry if different from primary industry
        if info.get("sector") and info["sector"].lower() != (company.industry or "").lower():
            if info["sector"] not in (company.sub_industries or []):
                company.sub_industries.append(info["sector"])
        if not company.employee_count and info.get("fullTimeEmployees"):
            company.employee_count = f"{info['fullTimeEmployees']:,}"
        if not company.description and info.get("longBusinessSummary"):
            company.description = info["longBusinessSummary"]
        if not company.website and info.get("website"):
            company.website = info["website"]
            if not company.domain:
                from .domain_utils import extract_clean_domain
                company.domain = extract_clean_domain(info["website"]) or ""
        if not company.headquarters:
            city = info.get("city", "")
            country = info.get("country", "")
            if city or country:
                company.headquarters = f"{city}, {country}".strip(", ")
        if not company.stock_ticker and info.get("symbol"):
            company.stock_ticker = info["symbol"]
        if not company.stock_exchange and info.get("exchange"):
            company.stock_exchange = info["exchange"]
        # Officers from yfinance
        if not company.key_people and info.get("companyOfficers"):
            company.key_people = [
                {"name": o.get("name", ""), "title": o.get("title", "")}
                for o in info["companyOfficers"][:5]
                if o.get("name")
            ]
        if not company.ceo:
            for o in info.get("companyOfficers", []):
                if o.get("title") and "ceo" in o["title"].lower():
                    company.ceo = o.get("name", "")
                    break
        if not company.revenue and info.get("totalRevenue"):
            rev = info["totalRevenue"]
            if rev >= 1_000_000_000:
                company.revenue = f"${rev / 1_000_000_000:.1f}B"
            elif rev >= 1_000_000:
                company.revenue = f"${rev / 1_000_000:.0f}M"
        if "yfinance" not in company.data_sources:
            company.data_sources.append("yfinance")
        logger.debug(f"yfinance enrichment succeeded for '{company.company_name}'")

    except asyncio.TimeoutError:
        logger.debug(f"yfinance timed out for '{company.company_name}'")
    except Exception as e:
        logger.debug(f"yfinance enrichment failed for '{company.company_name}': {e}")


# ── Wikipedia Enrichment (notable companies) ─────────────────────────────

async def _enrich_from_wikipedia(company: EnrichedCompany) -> None:
    """Enrich from Wikipedia infobox — FREE. Structured data for notable companies.

    Provides: industry, HQ, founded, employees, website, products, key people.
    Uses wptools for direct infobox parsing. Runs in thread pool (synchronous lib).
    """
    try:
        def _wp_lookup():
            import wptools
            page = wptools.page(company.company_name, silent=True)
            page.get_parse()
            return page.data.get("infobox", {})

        infobox = await asyncio.wait_for(
            asyncio.to_thread(_wp_lookup), timeout=12.0,
        )

        if not infobox:
            return

        # Field mapping: Wikipedia key variants → our field
        def _strip_wiki(val: str) -> str:
            """Strip wiki markup while preserving meaningful content."""
            # Links: [[Target|Display]] → Display, [[Target]] → Target
            val = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", val)
            # Templates with content: {{hlist|A|B}} → A, B
            # {{Unbulleted list|A|B}} → A, B
            val = re.sub(r"\{\{(?:hlist|Unbulleted list|ubl|plainlist)\|([^}]*)\}\}", r"\1", val, flags=re.I)
            # Date templates: {{Start date and age|2 July 1981}} → 2 July 1981
            val = re.sub(r"\{\{Start date(?: and age)?\|([^}]*)\}\}", r"\1", val, flags=re.I)
            # Small template: {{small|(text)}} → text
            val = re.sub(r"\{\{small\|([^}]*)\}\}", r"\1", val, flags=re.I)
            # Increase/Decrease markers
            val = re.sub(r"\{\{(?:Increase|Decrease|Steady)\}\}", "", val, flags=re.I)
            # Currency conversion: {{INRConvert|...}} → remove
            val = re.sub(r"\{\{INRConvert\|[^}]*\}\}", "", val, flags=re.I)
            # Other templates: strip remaining {{...}} but try to keep first arg
            val = re.sub(r"\{\{[^|}]*\|([^|}]*?)(?:\|[^}]*)?\}\}", r"\1", val)
            val = re.sub(r"\{\{[^}]*\}\}", "", val)
            # HTML tags and refs
            val = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^/]*/?>", "", val, flags=re.DOTALL)
            val = re.sub(r"<br\s*/?>", ", ", val, flags=re.I)
            val = re.sub(r"<[^>]+>", "", val)
            # Clean up
            val = re.sub(r"\s+", " ", val)
            val = re.sub(r"^[,\s|]+|[,\s|]+$", "", val)
            return val.strip()

        def _get(keys: list[str]) -> str:
            for k in keys:
                val = infobox.get(k)
                if val and isinstance(val, str):
                    val = _strip_wiki(val)
                    if val:
                        return val
            return ""

        if not company.industry:
            ind = _get(["industry", "Industry"])
            if ind:
                company.industry = ind
        else:
            # A1: Wikipedia may have broader/different industry label → store as sub_industry
            wiki_ind = _get(["industry", "Industry"])
            if wiki_ind and wiki_ind.lower() != (company.industry or "").lower():
                if wiki_ind not in (company.sub_industries or []):
                    company.sub_industries.append(wiki_ind)
        if not company.headquarters:
            city = _get(["hq_location_city", "hq_location", "headquarters", "location", "location_city"])
            country = _get(["hq_location_country"])
            hq = f"{city}, {country}".strip(", ") if city or country else ""
            if hq:
                company.headquarters = hq
        if not company.employee_count:
            emp = _get(["num_employees", "employees", "number_of_employees"])
            if emp:
                company.employee_count = emp
        if company.founded_year is None:
            founded = _get(["founded", "foundation", "formation_date"])
            if founded:
                year_match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", founded)
                if year_match:
                    company.founded_year = int(year_match.group(1))
        if not company.website:
            # Raw value before wiki markup stripping — extract URL from {{URL|...}}
            raw_site = ""
            for key in ["homepage", "website", "official_website"]:
                raw_val = infobox.get(key, "")
                if raw_val:
                    url_match = re.search(r"https?://[^\s|}\]]+", str(raw_val))
                    if url_match:
                        raw_site = url_match.group(0)
                        break
            if not raw_site:
                raw_site = _get(["homepage", "website", "official_website"])
            if raw_site and ("http" in raw_site or "." in raw_site):
                if not raw_site.startswith("http"):
                    raw_site = f"https://{raw_site}"
                company.website = raw_site
                if not company.domain:
                    from .domain_utils import extract_clean_domain
                    company.domain = extract_clean_domain(raw_site) or ""
        if not company.products_services:
            products = _get(["products", "Products", "services"])
            if products:
                # Split on common separators (including pipe | and bullet *)
                items = re.split(r"[,\n•·|*]", products)
                company.products_services = [
                    p.strip() for p in items
                    if p.strip() and len(p.strip()) > 2 and not p.strip().startswith("(")
                ][:10]
        if not company.ceo:
            kp = _get(["key_people", "founder", "CEO"])
            if kp and len(kp) < 100:
                company.ceo = kp

        if "wikipedia" not in company.data_sources:
            company.data_sources.append("wikipedia")
        logger.debug(f"Wikipedia enrichment succeeded for '{company.company_name}'")

    except asyncio.TimeoutError:
        logger.debug(f"Wikipedia timed out for '{company.company_name}'")
    except Exception as e:
        logger.debug(f"Wikipedia enrichment failed for '{company.company_name}': {e}")


# ── Targeted Product Search ───────────────────────────────────────────────

async def _search_products_services(company: EnrichedCompany) -> None:
    """Targeted Tavily search for products/services when structured extraction didn't find them."""
    if company.products_services:
        return  # Already populated

    try:
        from .web.tavily_tool import TavilyTool

        tavily = TavilyTool()
        if not tavily.available:
            return

        result = await asyncio.wait_for(
            tavily.search(
                query=f'"{company.company_name}" products services offerings solutions',
                max_results=3,
                include_answer=True,
                search_depth="basic",
            ),
            timeout=8.0,
        )

        answer = result.get("answer", "")
        if not answer:
            return

        from .llm.llm_service import LLMService
        llm = LLMService(lite=True)
        prompt = (
            f"From this text about {company.company_name}, extract a list of their "
            f"main products, services, or solutions. Return as a JSON array of strings.\n\n"
            f"Text: {answer[:1500]}\n\nJSON array:"
        )
        response = await asyncio.wait_for(
            llm.generate_json(prompt=prompt, system_prompt="Return valid JSON only."),
            timeout=6.0,
        )

        if isinstance(response, list) and response:
            company.products_services = [str(p).strip() for p in response if p and str(p).strip()][:10]
        elif isinstance(response, dict):
            for v in response.values():
                if isinstance(v, list):
                    company.products_services = [str(p).strip() for p in v if p and str(p).strip()][:10]
                    break

    except Exception as e:
        logger.debug(f"Product search failed for '{company.company_name}': {e}")


# ── Background Deep Intelligence (ScrapeGraphAI) ─────────────────────────

async def _background_deep_enrichment(
    company_name: str,
    domain: str,
    industry: str,
) -> None:
    """Background orchestrator for ALL deep company intelligence.

    Runs AFTER the initial fast enrichment completes. Fires all deep analysis
    in parallel (ScrapeGraphAI SearchGraph, SmartScraper on website, job
    posting analysis, tech/IP intelligence), then merges results into DB.

    This function never raises — all failures silently logged.
    """
    try:
        from app.config import get_settings
        settings = get_settings()

        if not settings.deep_enrichment_enabled:
            return

        tasks = []

        # Existing SearchGraph deep search (products, competitors, etc.)
        from .web.web_intel import deep_company_search
        tasks.append(("search_graph", deep_company_search(company_name)))

        # SmartScraper on company website pages
        if domain and settings.website_scrape_enabled:
            tasks.append(("website", _deep_website_scrape(domain, company_name)))

        # Job posting analysis
        if settings.hiring_signals_enabled:
            tasks.append(("hiring", _analyze_hiring_signals(company_name, domain, industry)))

        # Tech & IP intelligence
        if settings.tech_ip_analysis_enabled:
            tasks.append(("tech_ip", _analyze_tech_ip(company_name, industry)))

        if not tasks:
            return

        labels = [t[0] for t in tasks]
        results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        # Merge all results into the DB
        merged: dict = {}
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                logger.debug(f"Background {label} failed for {company_name}: {result}")
                continue
            if not result:
                continue

            if label == "search_graph" and hasattr(result, "products_services"):
                if result.products_services:
                    merged.setdefault("products_services", result.products_services)
                if result.competitors:
                    merged.setdefault("competitors", result.competitors)
                if result.revenue:
                    merged.setdefault("revenue", result.revenue)
                if result.tech_stack:
                    merged.setdefault("tech_stack", result.tech_stack)
                if result.investors:
                    merged.setdefault("investors", result.investors)
                if result.subsidiaries:
                    merged.setdefault("subsidiaries", result.subsidiaries)
            elif label == "website" and isinstance(result, dict):
                for k, v in result.items():
                    if v and k not in merged:
                        merged[k] = v
            elif label == "hiring" and isinstance(result, dict):
                if result.get("hiring_signals"):
                    merged["hiring_signals"] = result["hiring_signals"]
            elif label == "tech_ip" and isinstance(result, dict):
                if result.get("tech_stack") and "tech_stack" not in merged:
                    merged["tech_stack"] = result["tech_stack"]
                if result.get("patents"):
                    merged["patents"] = result["patents"]
                if result.get("partnerships"):
                    merged["partnerships"] = result["partnerships"]

        # Persist merged deep intel to DB
        if merged:
            try:
                import json
                from ..database import get_database
                db = get_database()
                # Convert lists to JSON strings for storage
                profile_update = {}
                for k, v in merged.items():
                    if isinstance(v, list):
                        profile_update[k] = json.dumps(v)
                    else:
                        profile_update[k] = str(v) if v else ""
                db.upsert_company_profile(company_name, profile_update)
                logger.info(
                    f"Background deep intel: {company_name} -- "
                    f"updated {list(merged.keys())}"
                )
            except Exception as e:
                logger.debug(f"Failed to persist deep intel for {company_name}: {e}")

    except Exception as e:
        logger.debug(f"Background deep enrichment failed for {company_name}: {e}")


# ── ScrapeGraphAI availability check (once at import time) ────────────────
try:
    from scrapegraphai.graphs import SmartScraperGraph as _SmartScraperGraph  # noqa: F401
    _SCRAPEGRAPH_AVAILABLE = True
except ImportError:
    _SCRAPEGRAPH_AVAILABLE = False
    logger.warning(
        "scrapegraphai not installed — _deep_website_scrape will be unavailable. "
        "Install with: pip install scrapegraphai"
    )


async def _deep_website_scrape(domain: str, company_name: str) -> dict:
    """Scrape company website pages using SmartScraperGraph for structured data.

    Targets homepage + common pages (/about, /products, /careers) for:
    products, leadership, job postings, tech signals.
    """
    if not _SCRAPEGRAPH_AVAILABLE:
        logger.debug("_deep_website_scrape skipped: scrapegraphai not installed")
        return {}

    from app.config import get_settings
    settings = get_settings()

    openai_key = settings.openai_api_key
    if not openai_key:
        return {}

    results: dict = {}

    pages = [
        (
            f"https://{domain}",
            f"Extract {company_name}'s main value proposition, key products/services, "
            f"and any featured solutions from the homepage. Return JSON with: "
            f"tagline, products (list), value_proposition."
        ),
        (
            f"https://{domain}/about",
            f"Extract {company_name}'s leadership team, company mission, founding story, "
            f"and office locations. Return JSON with: leaders (list of name+title), "
            f"mission, founded, locations (list)."
        ),
    ]

    for url, prompt in pages:
        try:
            from scrapegraphai.graphs import SmartScraperGraph

            config = {
                "llm": {
                    "api_key": openai_key,
                    "model": settings.scrapegraph_model,
                    "temperature": 0.1,
                },
                "verbose": False,
                "headless": True,
            }

            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=config,
            )

            data = await asyncio.wait_for(
                asyncio.to_thread(scraper.run),
                timeout=settings.scrapegraph_timeout,
            )

            if isinstance(data, dict):
                # Merge products from website
                if data.get("products") and not results.get("products_services"):
                    products = data["products"]
                    if isinstance(products, list):
                        results["products_services"] = [str(p).strip() for p in products if p][:10]
                # Merge leaders
                if data.get("leaders"):
                    results.setdefault("key_people", data["leaders"])

        except asyncio.TimeoutError:
            logger.debug(f"SmartScraper timed out for {url} ({settings.scrapegraph_timeout}s)")
            continue
        except Exception as e:
            logger.debug(f"SmartScraper failed for {url}: {e}")
            continue

    return results


async def _analyze_hiring_signals(
    company_name: str,
    domain: str,
    industry: str,
) -> dict:
    """Analyze job postings as buying signals using web search.

    Uses web search (not ScrapeGraphAI) — faster and more reliable for job boards.
    """
    try:
        from .web.web_intel import search as web_search
        from .llm.llm_service import LLMService

        query = f'"{company_name}" jobs OR careers OR hiring 2026'
        results = await asyncio.wait_for(
            web_search(query, max_results=5),
            timeout=10.0,
        )

        if not results:
            return {}

        snippets = []
        for r in results:
            if r.title:
                snippets.append(f"{r.title}: {r.snippet[:200] if r.snippet else ''}")

        if not snippets:
            return {}

        llm = LLMService(lite=True)
        prompt = (
            f"From these job posting search results for {company_name}, extract:\n"
            + "\n".join(f"- {s}" for s in snippets[:5])
            + "\n\nReturn JSON with: hiring_signals (list of strings describing key roles "
            "and departments they're hiring for, e.g. 'Expanding VP-level engineering team', "
            "'Hiring 5+ data scientists', 'New CISO role posted')"
        )

        result = await asyncio.wait_for(
            llm.generate_json(prompt=prompt, system_prompt="Return valid JSON only."),
            timeout=8.0,
        )

        if isinstance(result, dict):
            return result
        return {}

    except Exception as e:
        logger.debug(f"Hiring signal analysis failed for {company_name}: {e}")
        return {}


async def _analyze_tech_ip(company_name: str, industry: str) -> dict:
    """Analyze technology stack and IP/patent signals via web search."""
    try:
        from .web.web_intel import search as web_search
        from .llm.llm_service import LLMService

        # Parallel searches for tech and IP
        tech_query = f'"{company_name}" technology stack OR "built with" OR "powered by"'
        ip_query = f'"{company_name}" patent OR innovation OR "R&D" OR research'

        tech_results, ip_results = await asyncio.gather(
            web_search(tech_query, max_results=3),
            web_search(ip_query, max_results=3),
            return_exceptions=True,
        )

        snippets = []
        if not isinstance(tech_results, Exception):
            for r in tech_results:
                if r.snippet:
                    snippets.append(f"Tech: {r.snippet[:200]}")
        if not isinstance(ip_results, Exception):
            for r in ip_results:
                if r.snippet:
                    snippets.append(f"IP: {r.snippet[:200]}")

        if not snippets:
            return {}

        llm = LLMService(lite=True)
        prompt = (
            f"From these search results about {company_name}'s technology and IP:\n"
            + "\n".join(f"- {s}" for s in snippets[:6])
            + "\n\nReturn JSON with:\n"
            "- tech_stack: list of technologies/platforms they use\n"
            "- patents: list of brief patent/R&D descriptions\n"
            "- partnerships: list of technology partnerships or integrations"
        )

        result = await asyncio.wait_for(
            llm.generate_json(prompt=prompt, system_prompt="Return valid JSON only."),
            timeout=8.0,
        )

        if isinstance(result, dict):
            return result
        return {}

    except Exception as e:
        logger.debug(f"Tech/IP analysis failed for {company_name}: {e}")
        return {}


# ── Main Enrichment Function ──────────────────────────────────────────────

async def enrich(
    company_name: str,
    domain: str = "",
    skip_validation: bool = False,
    background_deep: bool = False,
    fast_mode: bool = False,
) -> Optional[EnrichedCompany]:
    """Enrich a company name into a fully validated profile.

    Returns None if validation fails (i.e. not a real business entity).

    Steps:
      1. Check Company KB (SQLite) — return cached if fresh
      2. Validate entity (unless skip_validation)
      3. Parallel enrichment: Tavily deep research, Apollo org data
      4. LLM structured field extraction from Tavily answer
      5. Domain resolution (web search -> name guess)
      6. LLM gap-fill (only if description still empty)
      7. Optional deep enrichment via ScrapeGraphAI
      8. Save to Company KB

    Args:
        company_name: Company name to enrich
        domain: Known domain (speeds up Apollo lookup, skips domain resolution)
        skip_validation: Skip entity validation (for pipeline-internal calls where
                         the company was already validated upstream)
        background_deep: If True, also run ScrapeGraphAI deep enrichment
        fast_mode: If True, return after KB cache + Clearbit domain only (< 3s).
                   Use for interactive search; fires background full enrichment.

    Returns:
        EnrichedCompany with merged data from all sources, or None if invalid.
    """
    if not company_name or not company_name.strip():
        return None

    company_name = company_name.strip()
    company = EnrichedCompany(company_name=company_name)

    if domain:
        company.domain = domain

    # ── Step 1: Check Company KB (SQLite cache) ──────────────────────
    try:
        from ..database import get_database

        db = get_database()
        cached = db.get_or_enrich_company(company_name, max_age_days=7)
        if cached:
            _merge_kb_cache(company, cached)
            has_enrichment = bool(company.description or company.headquarters or company.employee_count)
            # Return cached data if we have enrichment AND either:
            #   - company has a domain (strong identity), or
            #   - caller already validated upstream (skip_validation=True)
            if has_enrichment and (company.domain or skip_validation):
                company.validation = ValidationResult(
                    is_valid_company=True,
                    confidence=0.85,
                    validation_source="kb_cache",
                )
                return company
    except Exception as e:
        logger.debug(f"KB cache lookup failed for '{company_name}': {e}")

    # ── Step 2: Validate entity ──────────────────────────────────────
    if not skip_validation:
        validation = await validate_entity(company_name)
        company.validation = validation
        if not validation.is_valid_company:
            logger.info(
                f"Entity validation REJECTED '{company_name}': {validation.rejection_reason}"
            )
            return None
    else:
        company.validation = ValidationResult(
            is_valid_company=True,
            confidence=1.0,
            validation_source="skip_validation",
        )

    # ── Step 3: Parallel enrichment from multiple sources ────────────
    async def _enrich_tavily():
        try:
            from .web.tavily_tool import TavilyTool

            tavily = TavilyTool()
            if not tavily.available:
                return

            research = await asyncio.wait_for(
                tavily.deep_company_research(company_name),
                timeout=15.0,
            )
            _merge_tavily(company, research)

            # Extract structured fields from the AI answer
            if research.get("answer"):
                await _extract_structured_fields(company, research["answer"])

            # Fallback: if key fields still empty, extract from search result
            # snippets. Covers cases where Apollo returns 403 and Tavily's AI
            # answer was too brief for structured extraction.
            key_empty = not company.industry and not company.headquarters and not company.employee_count
            if key_empty and research.get("results"):
                snippets = "\n".join(
                    f"{r.get('title', '')}: {r.get('content', '')}"
                    for r in research["results"][:5]
                    if r.get("content")
                )
                if snippets and len(snippets) > 100:
                    await _extract_structured_fields(company, snippets)

        except asyncio.TimeoutError:
            logger.debug(f"Tavily enrichment timed out for '{company_name}'")
        except Exception as e:
            logger.debug(f"Tavily enrichment failed for '{company_name}': {e}")

    async def _enrich_apollo():
        """Try to get Apollo org data if we have a domain and it's not cached."""
        if not company.domain:
            return
        try:
            from .crm.apollo_tool import ApolloTool

            cached_org = ApolloTool.get_cached_org(company.domain)
            if cached_org:
                _merge_apollo_org(company, cached_org)
                return

            from ..config import get_settings
            settings = get_settings()
            if not settings.apollo_api_key:
                return

            apollo = ApolloTool()
            result = await asyncio.wait_for(
                apollo.search_people_at_company(company.domain, limit=1),
                timeout=10.0,
            )
            org_data = result.get("company", {})
            if org_data:
                _merge_apollo_org(company, org_data)
        except asyncio.TimeoutError:
            logger.debug(f"Apollo enrichment timed out for '{company_name}'")
        except Exception as e:
            logger.debug(f"Apollo enrichment failed for '{company_name}': {e}")

    await asyncio.gather(
        _enrich_tavily(),
        _enrich_apollo(),
        _enrich_from_yfinance(company),
        _enrich_from_wikipedia(company),
        return_exceptions=True,
    )

    # ── Step 4: Domain resolution ────────────────────────────────────
    if not company.domain:
        await _resolve_domain(company)

    # After domain resolution, try Apollo if we now have a domain but didn't before
    if company.domain and "apollo" not in company.data_sources:
        await _enrich_apollo()

    # ── Step 5: LLM gap-fill (description only, last resort) ────────
    await _llm_gap_fill(company)

    # ── Step 5b: Targeted product search if still missing ─────────
    await _search_products_services(company)

    # ── Step 6: Save to Company KB (ALL fields) ──────────────────
    import json as _json
    try:
        from ..database import get_database

        db = get_database()
        profile_dict = {
            "domain": company.domain,
            "website": company.website,
            "description": company.description,
            "industry": company.industry,
            "founded_year": company.founded_year,
            "headquarters": company.headquarters,
            "ceo": company.ceo,
            "employee_count": company.employee_count,
            "stock_ticker": company.stock_ticker,
            "funding_stage": company.funding_stage,
            "revenue": company.revenue,
            "total_funding": company.total_funding,
            # List fields persisted as JSON strings
            "products_services": _json.dumps(company.products_services) if company.products_services else "",
            "competitors": _json.dumps(company.competitors) if company.competitors else "",
            "sub_industries": _json.dumps(company.sub_industries) if company.sub_industries else "",
            "tech_stack": _json.dumps(company.tech_stack) if company.tech_stack else "",
            "investors": _json.dumps(company.investors) if company.investors else "",
        }
        db.upsert_company_profile(company_name, profile_dict)
    except Exception as e:
        logger.debug(f"Failed to save enriched profile for '{company_name}': {e}")

    # ── Step 7: Fire background deep enrichment (non-blocking) ────
    from ..config import get_settings
    settings = get_settings()
    if settings.deep_enrichment_enabled:
        asyncio.create_task(
            _background_deep_enrichment(company_name, company.domain, company.industry)
        )

    return company


# ── Batch Enrichment ──────────────────────────────────────────────────────

async def enrich_batch(
    names: list[str],
    max_concurrent: int = 5,
    skip_validation: bool = False,
) -> list[EnrichedCompany]:
    """Enrich multiple company names in parallel with bounded concurrency.

    Validation failures (None results) are silently skipped.

    Args:
        names: List of company names to enrich
        max_concurrent: Max parallel enrichments (default 5, respects API rate limits)
        skip_validation: Skip entity validation for all companies

    Returns:
        List of successfully enriched companies (validation failures excluded).
    """
    sem = asyncio.Semaphore(max_concurrent)
    results: list[Optional[EnrichedCompany]] = []

    async def _do_one(name: str) -> Optional[EnrichedCompany]:
        async with sem:
            try:
                return await enrich(name, skip_validation=skip_validation)
            except Exception as e:
                logger.warning(f"Batch enrichment failed for '{name}': {e}")
                return None

    raw = await asyncio.gather(
        *[_do_one(n) for n in names],
        return_exceptions=True,
    )

    for item in raw:
        if isinstance(item, EnrichedCompany):
            results.append(item)
        elif isinstance(item, Exception):
            logger.debug(f"Batch enrichment exception: {item}")

    return [r for r in results if r is not None]
