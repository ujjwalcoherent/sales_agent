"""
Company Finder Agent.
Finds relevant companies for each trend and impacted sector.

V8: AI Council Stage C — validates each company-trend pairing for genuine
relevance. Improves pitch angles with service-specific recommendations.

V7: NER-based company hallucination guard — cross-validates LLM-generated
company names against spaCy NER entities from source articles. Companies
not found in sources get verified via Wikipedia API (free, no key needed).
Verification status stored in CompanyData.ner_verified / verification_source.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from ...schemas import CompanyData, CompanySize, ImpactAnalysis, AgentState
from app.tools.llm.llm_service import LLMService
from app.tools.domain_utils import (
    extract_clean_domain,
    is_valid_company_domain,
    extract_domain_from_company_name,
)
from app.agents.leads import is_company_description
from ...config import get_settings, CMI_SERVICES

logger = logging.getLogger(__name__)

# Safe mapping from LLM size strings → CompanySize enum
_SIZE_MAP = {"startup": CompanySize.STARTUP, "smb": CompanySize.SMB,
             "mid": CompanySize.MID, "mid_market": CompanySize.MID_MARKET,
             "large": CompanySize.ENTERPRISE,
             "enterprise": CompanySize.ENTERPRISE,
             "large_enterprise": CompanySize.LARGE_ENTERPRISE}

def _safe_company_size(raw: str) -> CompanySize:
    """Convert any LLM size string to CompanySize without crashing."""
    return _SIZE_MAP.get(raw.lower().strip() if raw else "mid", CompanySize.MID)


class CompanyDiscovery:
    """
    Finds companies affected by trends via search + LLM extraction.

    For each impacted sector:
    - Searches for relevant companies in the configured country
    - Classifies by size (startup/mid/enterprise)
    - Extracts company websites and domains
    - NER-based hallucination guard (V7)

    NOTE: This is a deterministic pipeline stage, not an autonomous agent.
    Renamed from CompanyAgent for honest naming.
    """
    
    def __init__(self, mock_mode: bool = False, deps=None, log_callback=None):
        """Initialize company agent."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self._log_callback = log_callback
        self._deps = deps
        # Session-level cache: skip re-enrichment for companies already processed
        # in this pipeline run. Keyed by normalized company name.
        self._enriched_cache: Dict[str, CompanyData] = {}
        if deps:
            self.llm_service = deps.llm_service
        else:
            self.llm_service = LLMService(mock_mode=self.mock_mode)

    def _log(self, msg: str, level: str = "info"):
        """Log to both Python logger and optional UI callback."""
        if level == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg, level)
            except Exception:
                pass
    
    async def find_companies(self, state: AgentState) -> AgentState:
        """
        Find companies for all trends and impacts — runs all impacts concurrently.
        """
        self._log(f"Starting company search for {len(state.impacts)} impacts (parallel)...")

        if not state.impacts:
            self._log("No impacts to process for company search", "warning")
            return state

        max_per_trend = self.settings.max_companies_per_trend
        n = len(state.impacts)

        # Run ALL impacts concurrently
        async def _search_one(i: int, impact: ImpactAnalysis) -> List[CompanyData]:
            self._log(f"[{i}/{n}] Searching: {impact.trend_title[:60]}...")
            try:
                companies = await self._find_companies_for_impact(impact, max_per_trend)
                self._log(f"[{i}/{n}] Found {len(companies)} companies for: {impact.trend_title[:50]}")
                return companies
            except Exception as e:
                self._log(f"[{i}/{n}] Failed: {e}", "warning")
                return []

        results = await asyncio.gather(
            *[_search_one(i, imp) for i, imp in enumerate(state.impacts, 1)],
            return_exceptions=True,
        )

        all_companies = []
        for r in results:
            if isinstance(r, list):
                all_companies.extend(r)

        # Deduplicate by company name
        unique_companies = self._deduplicate_companies(all_companies)
        self._log(f"Deduplication: {len(all_companies)} → {len(unique_companies)} unique companies")

        # Phase 3B: filter out enterprise conglomerates — we target mid-size, not giants
        from app.config import get_settings
        blocklist = get_settings().enterprise_blocklist_set
        before_filter = len(unique_companies)
        unique_companies = [
            c for c in unique_companies
            if not any(
                blk in (c.company_name or "").lower()
                for blk in blocklist
            )
        ]
        filtered = before_filter - len(unique_companies)
        if filtered:
            self._log(f"Enterprise filter: removed {filtered} large conglomerates")

        # Size prioritization: mid-size companies first (CMI's target market)
        unique_companies = self._prioritize_by_size(unique_companies)

        # V7: Verify companies against NER entities + company_enricher
        self._log(f"V7 Verification: Checking {len(unique_companies)} companies against source articles...")
        self._log(f"  Step 1: Cross-referencing with NER entities from trend sources")
        self._log(f"  Step 2: Enrichment via company_enricher (Wikidata + Apollo + KB)")
        unique_companies = await self._verify_all_companies(unique_companies, state)

        # V8: Stage C — AI council validates company-trend relevance
        self._log(f"Stage C: AI validating {len(unique_companies)} leads for genuine trend relevance...")
        self._log(f"  Checking: Is each company truly affected? Is the pitch defensible?")
        unique_companies = await self._validate_leads(unique_companies, state)

        state.companies = unique_companies
        state.current_step = "companies_found"
        self._log(f"Company search complete: {len(unique_companies)} verified companies", "success")

        return state
    
    async def _find_companies_for_impact(
        self,
        impact: ImpactAnalysis,
        limit: int
    ) -> List[CompanyData]:
        """
        Find companies with INTENT SIGNALS from actual news.
        Searches for companies mentioned in news as struggling, expanding, or affected by the trend.
        
        Args:
            impact: Impact analysis with mid-size company types
            limit: Maximum companies to find
            
        Returns:
            List of CompanyData objects with intent signals
        """
        companies = []
        
        current_year = datetime.now().year
        search_queries = []

        # Build queries that return NEWS with specific company names
        trend_short = impact.trend_title[:60]

        country = self.settings.country

        # Q1: Trend-specific company news (most likely to mention real companies)
        search_queries.append(f'{trend_short} {country} companies')
        search_queries.append(f'{trend_short} {country} company funding investment news {current_year}')

        # Q2: Sector-specific company news
        if impact.direct_impact:
            for company_type in impact.direct_impact[:3]:
                search_queries.append(
                    f'{company_type} {country} company news {current_year}'
                )
                search_queries.append(
                    f'{country} {company_type} startup funding acquisition {current_year}'
                )

        # Q3: Company list queries (high yield for company names)
        if impact.direct_impact:
            for sector in impact.direct_impact[:2]:
                search_queries.append(f'top {sector} companies {country} {current_year}')
                search_queries.append(f'emerging {sector} companies {country} startups {current_year}')

        # Q4: Pain-point-driven company search
        if impact.midsize_pain_points:
            for pain_point in impact.midsize_pain_points[:2]:
                keywords = self._extract_key_terms(pain_point)
                if keywords:
                    search_queries.append(f'{country} company {keywords} news {current_year}')

        # Q5: Phase 3B — use who_needs_help verbatim as highest-signal query
        if getattr(impact, 'who_needs_help', ''):
            who = impact.who_needs_help[:120]  # cap length
            search_queries.insert(0, f'{who} {country} company news {current_year}')
            search_queries.insert(1, f'{who} {country} startup SME {current_year}')
        
        max_queries = self.settings.max_search_queries_per_impact
        queries_to_run = search_queries[:max_queries]

        self._log(f"  Running {len(queries_to_run)} web searches (parallel)...")

        # Step 1: Run all searches concurrently via web_intel
        async def _do_search(query: str):
            try:
                if self.mock_mode:
                    return await self._get_mock_companies_with_intent(query, impact, limit)
                from app.tools.web.web_intel import search
                results = await search(query, max_results=8)
                # Convert SearchResult objects to dicts for downstream compatibility
                return [{"title": r.title, "url": r.url, "content": r.snippet} for r in results]
            except Exception as e:
                logger.warning(f"Search failed: {e}")
                return []

        search_sem = asyncio.Semaphore(10)
        async def _do_search_bounded(query):
            async with search_sem:
                return await _do_search(query)

        search_results_per_query = await asyncio.gather(
            *[_do_search_bounded(q) for q in queries_to_run]
        )

        # Step 2: Collect all search results and extract companies concurrently
        if self.mock_mode:
            for result_list in search_results_per_query:
                if isinstance(result_list, list):
                    companies.extend(result_list)
        else:
            all_results = []
            for results in search_results_per_query:
                if isinstance(results, list):
                    all_results.extend(results)

            if all_results:
                # Cap results to avoid excessive LLM calls (each is an API call)
                _max_results = 10
                if len(all_results) > _max_results:
                    self._log(f"  Capping {len(all_results)} search results to {_max_results} (rate limit safety)")
                    all_results = all_results[:_max_results]
                self._log(f"  Extracting companies from {len(all_results)} search results...")
                # OpenAI primary — 500+ RPM allows higher concurrency
                sem = asyncio.Semaphore(5)
                async def _limited_extract(result):
                    async with sem:
                        return await self._extract_companies_with_intent(result, impact, queries_to_run[0])

                extracted_results = await asyncio.gather(
                    *[_limited_extract(r) for r in all_results],
                    return_exceptions=True,
                )
                ok, fail = 0, 0
                for r in extracted_results:
                    if isinstance(r, list):
                        companies.extend(r)
                        ok += 1
                    elif isinstance(r, Exception):
                        fail += 1
                        logger.warning(f"Extraction error: {type(r).__name__}: {str(r)[:150]}")
                if fail:
                    self._log(f"  Extraction: {ok} succeeded, {fail} failed", "warning")
            else:
                self._log("  No search results returned", "warning")

        # Fallback chain when search returns 0: web_intel (Tavily → DDG)
        if not companies and self._deps:
            companies = await self._fallback_search(impact, queries_to_run[:3], limit)

        companies = self._deduplicate_companies(companies)
        self._log(f"  {len(companies)} unique companies found for this trend")
        return companies[:limit]
    
    async def _fallback_search(self, impact, queries: list, limit: int = 10) -> list:
        """Fallback: web search (Tavily -> DDG) when primary search returns empty."""
        from app.schemas.sales import CompanyData, CompanySize
        companies = []
        from app.tools.web.web_intel import search
        self._log("  Fallback: web search (Tavily -> DDG)...")
        for query in queries[:3]:
            try:
                results = await search(query, max_results=5)
                for r in results:
                    result_dict = {"title": r.title, "url": r.url, "content": r.snippet}
                    if r.title and self.llm_service.has_available_provider():
                        try:
                            extracted = await self._extract_companies_with_intent(result_dict, impact, query)
                            companies.extend(extracted)
                        except Exception:
                            pass
                if len(companies) >= limit:
                    break
            except Exception as e:
                logger.debug(f"  Fallback search failed: {e}")
        if companies:
            self._log(f"  Web fallback: {len(companies)} companies found")
        return companies[:limit]

    _STOPWORDS = frozenset({
        'need', 'to', 'the', 'a', 'an', 'but', 'lack', 'dont', 'know',
        'want', 'have', 'is', 'are', 'for', 'of', 'in', 'on', 'with',
    })

    def _extract_key_terms(self, text: str) -> str:
        """Extract key business terms from pain point text."""
        words = text.lower().replace("'", "").split()
        key_terms = [w for w in words if w not in self._STOPWORDS and len(w) > 3]
        return ' '.join(key_terms[:4])
    
    async def _get_mock_companies_with_intent(
        self,
        query: str,
        impact: ImpactAnalysis,
        limit: int
    ) -> List[CompanyData]:
        """Generate mock companies with INTENT SIGNALS for testing."""
        # Get pain points and company types for context
        pain_points = getattr(impact, 'midsize_pain_points', [])[:2]
        company_types = getattr(impact, 'direct_impact', [])[:2]
        
        country = self.settings.country
        prompt = f"""Find REAL mid-size {country} companies (50-300 employees) that would be affected by this trend.

TREND: {impact.trend_title}
COMPANY TYPES AFFECTED: {', '.join(company_types) if company_types else 'General mid-size companies'}
PAIN POINTS THEY FACE: {', '.join(pain_points) if pain_points else 'Market challenges'}

Return a JSON array of 3-4 REAL {country} companies with:
- company_name: REAL company name (not fictional)
- website: Their actual website
- industry: Their industry
- company_size: "mid" (50-300 employees)
- intent_signal: Why they need consulting NOW (e.g., "Recently announced expansion", "Facing regulatory pressure", "Hiring for new division")
- reason_relevant: Why this trend affects them specifically
- description: One line about what they do

Focus on:
1. Companies actually mentioned in business news
2. Companies showing signs of expansion, struggle, or change
3. Mid-size companies, NOT Tata/Reliance/Infosys
4. Companies with clear consulting needs"""

        try:
            from app.schemas.llm_outputs import CompanyListLLM
            llm_result = await self.llm_service.run_structured(
                prompt=prompt,
                output_type=CompanyListLLM,
            )
            companies_data = [c.model_dump() for c in llm_result.companies]

            companies = []
            for item in companies_data[:limit]:
                company_name = item.get("company_name", "")
                if not company_name:
                    continue
                    
                website = item.get("website", "")
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                    # Reject domains that don't relate to the company (LLM may use article source URL)
                    if domain and is_valid_company_domain(domain):
                        name_norm = company_name.lower().replace(" ", "")
                        dom_base = domain.split(".")[0].lower()
                        if dom_base not in name_norm and name_norm not in dom_base and len(name_norm) > 4:
                            domain = ""
                            website = ""
                if not domain:
                    domain = extract_domain_from_company_name(company_name)

                company_id = hashlib.md5(company_name.lower().encode()).hexdigest()[:12]

                # Build reason with intent signal
                intent = item.get("intent_signal", "")
                reason = item.get("reason_relevant", f"Affected by {impact.trend_title[:30]}")
                full_reason = f"{intent}. {reason}" if intent else reason
                
                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=_safe_company_size(item.get("company_size", "mid")),
                    industry=item.get("industry", "Technology"),
                    website=website or (f"https://{domain}" if domain else ""),
                    domain=domain or "",
                    description=item.get("description", ""),
                    reason_relevant=full_reason,
                    trend_id=impact.trend_id
                ))

            logger.info(f"Generated {len(companies)} companies with intent for trend")
            return companies
            
        except Exception as e:
            logger.error(f"Failed to generate mock companies: {e}")
            return []
    
    async def _extract_companies_with_intent(
        self,
        search_result: Dict,
        impact: ImpactAnalysis,
        query: str
    ) -> List[CompanyData]:
        """
        Extract companies with INTENT SIGNALS from search results.
        Focuses on finding companies mentioned in news with buying signals.
        """
        snippet = search_result.get("content", search_result.get("snippet", ""))
        source_url = search_result.get("url", search_result.get("source_url", ""))
        title = search_result.get("title", "")

        if not snippet or len(snippet.strip()) < 50:
            return []  # Skip empty/tiny snippets — nothing to extract from

        # Skip if all LLM providers are exhausted (avoids 30s wait per call)
        if not self.llm_service.has_available_provider():
            logger.debug("Skipping extraction — all LLM providers in cooldown")
            return []

        # Use LLM to extract companies — broad extraction, filter later
        prompt = f"""Extract ALL companies mentioned in this article that could need consulting help.

TITLE: {title}
TEXT: {snippet[:1500]}
SOURCE: {source_url}

CONTEXT: We are looking for companies related to: {impact.trend_title}

For each company mentioned, provide:
- company_name: Official company name
- website: Company website if found
- industry: Their industry/sector
- company_size: Estimate as "startup" (<50 employees), "mid" (50-500), "large" (500-5000), or "enterprise" (5000+)
- intent_signal: What they are doing RIGHT NOW that creates a consulting need (e.g., "raised funding and scaling operations", "facing new regulatory requirements", "restructuring after layoffs", "expanding into new market", "struggling with supply chain issues")
- reason_relevant: Why this company specifically needs external consulting help for this trend
- description: One line about what the company does

RULES:
1. Include ANY real company mentioned by name — any size, from startups to large enterprises
2. Be accurate about company_size — a publicly listed company with 10,000+ employees is "enterprise", a 200-person firm is "mid"
3. Focus on companies that are AFFECTED by the trend and could benefit from consulting
4. Skip pure investors/VCs unless they are also operating companies
5. If no companies are named in the text, return empty list"""

        try:
            from app.schemas.llm_outputs import CompanyListLLM

            # Try structured output first, fall back to generate_json
            try:
                llm_result = await self.llm_service.run_structured(
                    prompt=prompt,
                    output_type=CompanyListLLM,
                )
                result = [c.model_dump() for c in llm_result.companies]
            except Exception as struct_err:
                logger.warning(f"Structured extraction failed: {struct_err}, trying generate_json...")
                raw = await self.llm_service.generate_json(prompt=prompt)
                if isinstance(raw, dict) and "error" not in raw:
                    result = raw.get("companies", [])
                    if not isinstance(result, list):
                        result = []
                elif isinstance(raw, list):
                    result = raw
                else:
                    raise RuntimeError(f"generate_json returned error: {raw}")

            # Load company relevance bandit for contextual scoring
            relevance_bandit = None
            try:
                from app.learning.company_bandit import CompanyRelevanceBandit
                relevance_bandit = CompanyRelevanceBandit()
                logger.debug("Relevance bandit loaded for contextual scoring")
            except Exception:
                logger.debug("Relevance bandit unavailable, using all companies")

            # Get event type and severity from impact for contextual scoring
            impact_event_type = getattr(impact, "validated_event_type", "") or "general"
            impact_severity = getattr(impact, "severity", "medium")
            if hasattr(impact_severity, "value"):
                impact_severity = impact_severity.value

            companies = []
            for item in result:
                company_name = item.get("company_name", "")
                if not company_name:
                    continue

                # Guard: reject names that are actually segment descriptions
                if is_company_description(company_name):
                    logger.debug(f"Skipping description-as-name in extraction: '{company_name[:80]}'")
                    continue

                raw_size = item.get("company_size", "mid").lower().strip()

                # Contextual relevance scoring (replaces static blocklist)
                if relevance_bandit:
                    # Compute industry match: check if company industry overlaps
                    # with trend's affected sectors
                    company_industry = item.get("industry", "").lower()
                    industry_match = 0.5  # default
                    affected_sectors = (
                        [s.lower() for s in impact.direct_impact[:5]]
                        + [s.lower() for s in impact.positive_sectors[:3]]
                    )
                    if affected_sectors and company_industry:
                        # Check if any affected sector words appear in company industry
                        for sector in affected_sectors:
                            sector_words = set(sector.split())
                            industry_words = set(company_industry.split())
                            if sector_words & industry_words:
                                industry_match = 0.85
                                break

                    # Compute intent signal strength from LLM extraction
                    intent = item.get("intent_signal", "")
                    intent_strength = 0.3  # baseline
                    strong_intents = {"raised funding", "expanding", "acquired",
                                      "launching", "restructuring", "struggling",
                                      "regulatory pressure", "compliance"}
                    if any(si in intent.lower() for si in strong_intents):
                        intent_strength = 0.8

                    relevance = relevance_bandit.compute_relevance(
                        company_size=raw_size,
                        event_type=impact_event_type,
                        industry_match=industry_match,
                        trend_severity=str(impact_severity),
                        intent_signal_strength=intent_strength,
                    )
                    min_relevance = self.settings.company_min_relevance
                    if relevance < min_relevance:
                        logger.debug(
                            f"Low relevance ({relevance:.2f}): {company_name} "
                            f"({raw_size}|{impact_event_type})"
                        )
                        continue

                website = item.get("website", "")
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                    # Reject domains that don't relate to the company name
                    # (LLM may extract the article source URL, e.g. prnewswire.com)
                    if domain and is_valid_company_domain(domain):
                        name_norm = company_name.lower().replace(" ", "")
                        dom_base = domain.split(".")[0].lower()
                        if dom_base not in name_norm and name_norm not in dom_base and len(name_norm) > 4:
                            domain = ""
                            website = ""
                if not domain:
                    domain = extract_domain_from_company_name(company_name)

                if domain and not is_valid_company_domain(domain):
                    domain = ""

                company_id = hashlib.md5(company_name.lower().encode()).hexdigest()[:12]

                intent = item.get("intent_signal", "")
                reason = item.get("reason_relevant", "")
                full_reason = f"\U0001f4cc {intent}. {reason}" if intent else reason

                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=_safe_company_size(raw_size),
                    industry=item.get("industry", "Technology"),
                    website=website or (f"https://{domain}" if domain else ""),
                    domain=domain or "",
                    description=item.get("description", ""),
                    reason_relevant=full_reason,
                    trend_id=impact.trend_id
                ))
            
            if companies:
                logger.info(f"Extracted {len(companies)} companies with intent from: {title[:50]}")
            return companies
            
        except Exception as e:
            logger.warning(f"LLM extraction failed for '{title[:40]}': {e}")
            self._log(f"  ⚠ Extraction failed: {type(e).__name__}: {str(e)[:120]}", "warning")
            return []

    def _deduplicate_companies(self, companies: List[CompanyData]) -> List[CompanyData]:
        """Remove duplicate companies by name and filter out description-as-name entries."""
        seen = set()
        unique = []

        for company in companies:
            name = company.company_name
            # Guard: reject names that are segment descriptions, not real company names
            if is_company_description(name):
                logger.debug(f"Dedup filter: description-as-name '{name[:80]}'")
                continue
            key = name.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(company)

        return unique

    # Size priority: lower = better (CMI targets mid-size companies)
    _SIZE_PRIORITY = {
        CompanySize.SMB: 0,
        CompanySize.MID: 1,
        CompanySize.MID_MARKET: 1,
        CompanySize.STARTUP: 2,
        CompanySize.ENTERPRISE: 3,
        CompanySize.LARGE_ENTERPRISE: 4,
    }

    def _prioritize_by_size(
        self, companies: List[CompanyData], max_enterprise: int = 2
    ) -> List[CompanyData]:
        """Sort companies by size priority (mid-size first) and cap enterprise."""
        enterprise_sizes = {CompanySize.ENTERPRISE, CompanySize.LARGE_ENTERPRISE}

        # Partition into target (startup/smb/mid) and enterprise
        target = [c for c in companies if c.company_size not in enterprise_sizes]
        enterprise = [c for c in companies if c.company_size in enterprise_sizes]

        # Sort each group by priority
        target.sort(key=lambda c: self._SIZE_PRIORITY.get(c.company_size, 2))
        enterprise.sort(key=lambda c: self._SIZE_PRIORITY.get(c.company_size, 3))

        # Cap enterprise companies
        capped = enterprise[:max_enterprise]
        result = target + capped

        if enterprise and len(enterprise) > max_enterprise:
            dropped = [c.company_name for c in enterprise[max_enterprise:]]
            self._log(f"  Size filter: kept {len(target)} target + {len(capped)} enterprise, "
                      f"dropped {len(dropped)} enterprise ({', '.join(dropped[:3])})")
        else:
            sizes = {}
            for c in result:
                sz = c.company_size.value if hasattr(c.company_size, 'value') else str(c.company_size)
                sizes[sz] = sizes.get(sz, 0) + 1
            size_str = ", ".join(f"{k}={v}" for k, v in sorted(sizes.items()))
            self._log(f"  Size distribution: {size_str}")

        return result

    async def _validate_leads(
        self,
        companies: List[CompanyData],
        state: AgentState,
    ) -> List[CompanyData]:
        """Stage C — Lead validation via SPOC verify_lead_crystallization.

        Legacy LLM-based lead_validator removed (was never deployed).
        Validation now happens in orchestrator via pipeline_validator.verify_lead_crystallization().
        """
        return companies

    # ══════════════════════════════════════════════════════════════════════════
    # V7: NER-BASED COMPANY HALLUCINATION GUARD
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize company name for fuzzy matching."""
        import re
        name = name.lower().strip()
        # Remove common suffixes
        for suffix in (" pvt ltd", " pvt. ltd.", " private limited", " limited",
                       " ltd", " ltd.", " inc", " inc.", " corp", " corp.",
                       " llp", " llc", " co.", " company"):
            name = name.replace(suffix, "")
        # Remove punctuation
        name = re.sub(r'[^a-z0-9\s]', '', name).strip()
        return name

    @staticmethod
    def _fuzzy_entity_match(
        company_name: str,
        source_entities: Set[str],
    ) -> bool:
        """
        V7: Check if a company name fuzzy-matches any NER-extracted entity.

        Uses bidirectional substring matching:
        - "Tata Motors" matches NER entity "Tata Motors Limited"
        - "RBI" matches "Reserve Bank of India" (not really, but "RBI" would be in entities too)
        - "Bajaj Finance" matches "Bajaj Finance Ltd"
        """
        norm_name = CompanyDiscovery._normalize_name(company_name)
        if not norm_name or len(norm_name) < 2:
            return False

        for entity in source_entities:
            norm_entity = CompanyDiscovery._normalize_name(entity)
            if not norm_entity:
                continue
            # Exact match
            if norm_name == norm_entity:
                return True
            # Company name is substring of entity
            if norm_name in norm_entity and len(norm_name) >= 3:
                return True
            # Entity is substring of company name
            if norm_entity in norm_name and len(norm_entity) >= 3:
                return True

        return False

    async def _verify_company(
        self,
        company: CompanyData,
        source_entities: Set[str],
    ) -> CompanyData:
        """V8: Verify and enrich via NER + company_enricher (unified pipeline).

        Confidence scoring:
        - NER match + Wikidata: 0.95 (verified in both sources)
        - NER match only: 0.9 (entity found in source articles)
        - Wikidata match: 0.85 (structured KB match, rich data)
        - Enriched (description only): 0.6
        - No match: 0.2 (potentially hallucinated)
        """
        # Step 1: NER entity matching (keep — this is article-specific)
        ner_match = self._fuzzy_entity_match(company.company_name, source_entities)
        if ner_match:
            company.ner_verified = True
            company.verification_source = "ner_match"
            company.verification_confidence = 0.9

        # Step 2: Unified enrichment via company_enricher
        try:
            from app.tools.company_enricher import validate_entity, enrich

            # Validate first (fast, cached)
            validation = await asyncio.wait_for(
                validate_entity(company.company_name), timeout=5.0
            )

            if not validation.is_valid_company and not ner_match:
                company.verification_source = "invalid_entity"
                company.verification_confidence = 0.1
                return company

            # Enrich (uses Wikidata + Apollo + KB cache)
            enriched = await asyncio.wait_for(
                enrich(company.company_name, domain=company.domain, skip_validation=True),
                timeout=8.0
            )

            if enriched:
                # Apply enrichment data (only fill blanks)
                company.description = enriched.description or company.description
                company.industry = enriched.industry or company.industry
                company.headquarters = enriched.headquarters or getattr(company, "headquarters", "")
                company.employee_count = enriched.employee_count or getattr(company, "employee_count", "")
                company.stock_ticker = enriched.stock_ticker or getattr(company, "stock_ticker", "")
                company.ceo = enriched.ceo or getattr(company, "ceo", "")
                company.founded_year = enriched.founded_year or getattr(company, "founded_year", None)
                if enriched.website and not company.website:
                    company.website = enriched.website
                if enriched.domain and not company.domain:
                    company.domain = enriched.domain

                # Set confidence based on validation + enrichment
                validation_src = enriched.validation.validation_source if enriched.validation else ""
                if validation_src in ("tavily", "kb_cache"):
                    if ner_match:
                        company.verification_confidence = 0.95
                        company.verification_source = f"ner_match+{validation_src}"
                    else:
                        company.ner_verified = True
                        company.verification_confidence = 0.85
                        company.verification_source = validation_src
                elif enriched.description:
                    company.verification_confidence = max(company.verification_confidence, 0.6)
                    company.verification_source = company.verification_source or "enriched"
                else:
                    # Enrichment returned but without validation data or description.
                    # Use the initial validate_entity() result — we know it passed
                    # (otherwise we'd have returned at line 810).
                    conf = getattr(validation, "confidence", 0.0) or 0.0
                    company.verification_confidence = max(company.verification_confidence, conf, 0.5)
                    company.verification_source = company.verification_source or getattr(validation, "validation_source", "") or "validated"
                    if conf >= 0.7:
                        company.ner_verified = True

                return company
        except Exception as e:
            logger.debug(f"Enrichment failed for '{company.company_name}': {e}")

        # NER-only result
        if ner_match:
            return company

        # Step 3: Unverified
        company.ner_verified = False
        company.verification_source = "unverified"
        company.verification_confidence = 0.2
        return company

    async def _verify_all_companies(
        self,
        companies: List[CompanyData],
        state: AgentState,
    ) -> List[CompanyData]:
        """
        V7: Verify all companies against source article NER entities.

        Collects all NER entities from trends/articles in the state, then
        verifies each company. Filters out companies below the minimum
        confidence threshold (COMPANY_MIN_VERIFICATION_CONFIDENCE).
        """
        # Collect all NER entities from the pipeline state
        source_entities: Set[str] = set()

        # From trends (which carry key_entities from synthesis)
        for trend in state.trends:
            for kw in getattr(trend, 'keywords', []):
                source_entities.add(self._normalize_name(kw))
            for entity in getattr(trend, 'key_entities', []):
                source_entities.add(self._normalize_name(entity))

        # From impact analysis (which carries sector/company info)
        for impact in state.impacts:
            for sector in impact.positive_sectors:
                source_entities.add(self._normalize_name(sector))
            for sector in impact.negative_sectors:
                source_entities.add(self._normalize_name(sector))

        self._log(
            f"  V7: {len(source_entities)} source entities extracted from trends"
        )

        # Verify all companies concurrently
        verified: List[CompanyData] = []
        min_confidence = self.settings.company_min_verification_confidence

        self._log(f"  V7: Running NER match + enrichment verification (concurrent)...")
        # Bounded concurrency: avoid hammering Tavily/Apollo with 50+ parallel calls
        verify_sem = asyncio.Semaphore(8)

        async def _verify_bounded(c: CompanyData) -> CompanyData:
            # Session cache: skip re-enrichment if already processed this run
            cache_key = self._normalize_name(c.company_name)
            if cache_key in self._enriched_cache:
                cached = self._enriched_cache[cache_key]
                # Copy enrichment data to this instance (may have different trend_id)
                c.description = cached.description or c.description
                c.industry = cached.industry or c.industry
                c.domain = cached.domain or c.domain
                c.website = cached.website or c.website
                c.ner_verified = cached.ner_verified
                c.verification_source = cached.verification_source
                c.verification_confidence = cached.verification_confidence
                return c
            async with verify_sem:
                try:
                    # Per-company timeout: 8s max to prevent pipeline hangs
                    result = await asyncio.wait_for(
                        self._verify_company(c, source_entities), timeout=8.0,
                    )
                    self._enriched_cache[cache_key] = result
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"  V7: Timeout verifying '{c.company_name}' (8s)")
                    c.verification_source = "timeout"
                    c.verification_confidence = 0.3
                    return c

        verified_companies = await asyncio.gather(
            *[_verify_bounded(c) for c in companies]
        )
        for company in verified_companies:
            if company.verification_confidence >= min_confidence:
                verified.append(company)
            else:
                self._log(
                    f"  V7: Filtered '{company.company_name}' "
                    f"(confidence={company.verification_confidence:.2f} < {min_confidence})"
                )

        # Log verification summary
        ner_count = sum(1 for c in verified if c.verification_source == "ner_match")
        tavily_count = sum(1 for c in verified if "tavily" in (c.verification_source or ""))
        enriched_count = sum(1 for c in verified if c.verification_source == "enriched")
        unverified_count = sum(1 for c in verified if c.verification_source == "unverified")
        filtered_count = len(companies) - len(verified)

        self._log(
            f"  V7 Results: {len(verified)}/{len(companies)} passed — "
            f"{ner_count} NER, {tavily_count} Tavily, {enriched_count} enriched, "
            f"{unverified_count} unverified, {filtered_count} filtered out"
        )

        return verified


# Backward compatibility alias
CompanyAgent = CompanyDiscovery
