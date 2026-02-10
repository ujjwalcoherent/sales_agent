"""
Company Finder Agent.
Finds relevant Indian companies for each trend and impacted sector.

V7: NER-based company hallucination guard — cross-validates LLM-generated
company names against spaCy NER entities from source articles. Companies
not found in sources get verified via Wikipedia API (free, no key needed).
Verification status stored in CompanyData.ner_verified / verification_source.
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import quote_plus

import aiohttp

from ..schemas import CompanyData, CompanySize, ImpactAnalysis, AgentState
from ..tools.tavily_tool import TavilyTool
from ..tools.llm_tool import LLMTool
from ..tools.domain_utils import (
    extract_clean_domain,
    is_valid_company_domain,
    extract_domain_from_company_name,
    extract_domains_from_text
)
from ..config import get_settings, COMPANY_SIZE_KEYWORDS

logger = logging.getLogger(__name__)


class CompanyAgent:
    """
    Agent responsible for finding companies affected by trends.
    
    For each impacted sector:
    - Searches for relevant Indian companies
    - Classifies by size (startup/mid/enterprise)
    - Extracts company websites and domains
    """
    
    def __init__(self, mock_mode: bool = False):
        """Initialize company agent."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.tavily_tool = TavilyTool(mock_mode=self.mock_mode)
        self.llm_tool = LLMTool(mock_mode=self.mock_mode)
    
    async def find_companies(self, state: AgentState) -> AgentState:
        """
        Find companies for all trends and impacts.
        
        Args:
            state: Current agent state with trends and impacts
            
        Returns:
            Updated state with companies
        """
        logger.info("🏢 Starting company search...")
        
        if not state.impacts:
            logger.warning("No impacts to process for company search")
            return state
        
        all_companies = []
        max_per_trend = self.settings.max_companies_per_trend
        
        for impact in state.impacts:
            try:
                companies = await self._find_companies_for_impact(impact, max_per_trend)
                all_companies.extend(companies)
                logger.info(f"✅ Found {len(companies)} companies for: {impact.trend_title[:40]}...")
            except Exception as e:
                logger.warning(f"Failed to find companies for impact: {e}")
        
        # Deduplicate by company name
        unique_companies = self._deduplicate_companies(all_companies)

        # V7: Verify companies against NER entities + Wikipedia
        unique_companies = await self._verify_all_companies(unique_companies, state)

        state.companies = unique_companies
        state.current_step = "companies_found"
        logger.info(f"Found {len(unique_companies)} unique verified companies")
        
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
        
        # Intent signal keywords to find companies with buying signals
        intent_keywords = [
            "expanding", "struggling", "hiring", "entering market",
            "facing challenges", "seeking partners", "launching",
            "restructuring", "investing in", "scaling", "growing"
        ]
        
        # Build search queries from impact analysis
        search_queries = []
        
        # Priority 1: Search for companies in directly impacted categories with intent
        if impact.direct_impact:
            for company_type in impact.direct_impact[:3]:
                # Query 1: Companies struggling/facing challenges
                current_year = datetime.now().year
                search_queries.append(f'"{company_type}" India company struggling OR challenges OR facing issues {current_year - 1} {current_year}')
                # Query 2: Companies expanding/growing
                search_queries.append(f'"{company_type}" India mid-size company expanding OR growing OR entering market')
        
        # Priority 2: Search based on pain points
        if impact.midsize_pain_points:
            for pain_point in impact.midsize_pain_points[:2]:
                # Extract key terms from pain point
                keywords = self._extract_key_terms(pain_point)
                if keywords:
                    search_queries.append(f'India mid-size company {keywords} news')
        
        # Priority 3: Search for companies affected by the trend
        search_queries.append(f'"{impact.trend_title}" affected companies India mid-size')
        
        # Fallback: Direct impact sectors
        if impact.direct_impact:
            for sector in impact.direct_impact[:2]:
                search_queries.append(f'top mid-size {sector} companies India 50-300 employees')
        
        logger.info(f"Intent-based search queries: {len(search_queries)}")
        
        for query in search_queries[:5]:  # Limit to 5 searches
            try:
                if self.mock_mode:
                    # In mock mode, generate companies with intent
                    mock_companies = await self._get_mock_companies_with_intent(
                        query, impact, limit
                    )
                    companies.extend(mock_companies)
                else:
                    # Real search for companies with intent signals
                    search_response = await self.tavily_tool.search(query=query, max_results=5)
                    if isinstance(search_response, dict) and "error" in search_response:
                        logger.warning(f"Tavily search error: {search_response['error']}")
                        continue
                    search_results = search_response.get("results", [])
                    
                    # Extract companies from results
                    for result in search_results:
                        try:
                            extracted = await self._extract_companies_with_intent(
                                result, impact, query
                            )
                            companies.extend(extracted)
                        except Exception as e:
                            logger.warning(f"Failed to extract company: {e}")
                
                if len(companies) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Search failed for query '{query[:50]}': {e}")
        
        # Deduplicate
        companies = self._deduplicate_companies(companies)
        
        logger.info(f"Total companies with intent found: {len(companies)}")
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
        
        prompt = f"""Find REAL mid-size Indian companies (50-300 employees) that would be affected by this trend.

TREND: {impact.trend_title}
COMPANY TYPES AFFECTED: {', '.join(company_types) if company_types else 'General mid-size companies'}
PAIN POINTS THEY FACE: {', '.join(pain_points) if pain_points else 'Market challenges'}

Return a JSON array of 3-4 REAL Indian companies with:
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
            companies_data = await self.llm_tool.generate_list(prompt=prompt)
            
            companies = []
            for item in companies_data[:limit]:
                company_name = item.get("company_name", "")
                if not company_name:
                    continue
                    
                website = item.get("website", "")
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                if not domain:
                    domain = extract_domain_from_company_name(company_name)
                
                company_id = hashlib.md5(company_name.encode()).hexdigest()[:12]
                
                # Build reason with intent signal
                intent = item.get("intent_signal", "")
                reason = item.get("reason_relevant", f"Affected by {impact.trend_title[:30]}")
                full_reason = f"{intent}. {reason}" if intent else reason
                
                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=CompanySize(item.get("company_size", "mid")),
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
        
        # Use LLM to extract companies with intent
        prompt = f"""Extract mid-size Indian companies (50-300 employees) from this news/article.

ARTICLE TITLE: {title}
ARTICLE TEXT: {snippet}
SOURCE: {source_url}

CONTEXT - We are looking for companies affected by: {impact.trend_title}

For each company mentioned, extract as JSON array:
- company_name: Official company name
- website: Company website if found
- industry: Their industry
- intent_signal: What action/change they are taking (e.g., "expanding operations", "facing regulatory pressure", "hiring for new division", "restructuring", "seeking partnerships")
- reason_relevant: Why this company would need consulting services based on the news
- description: What the company does

RULES:
1. Only include REAL companies clearly mentioned by name
2. Focus on mid-size companies (NOT Tata, Reliance, Infosys, Wipro)
3. Must have clear INTENT SIGNAL (action they are taking or challenge they face)
4. Return empty array [] if no qualifying companies found

Return JSON array only."""

        try:
            result = await self.llm_tool.generate_list(prompt=prompt)
            
            companies = []
            for item in result:
                company_name = item.get("company_name", "")
                if not company_name:
                    continue
                
                # Skip large enterprises
                large_companies = ['tata', 'reliance', 'infosys', 'wipro', 'hcl', 'hdfc', 'icici', 'bajaj', 'mahindra', 'adani', 'vedanta']
                if any(lc in company_name.lower() for lc in large_companies):
                    continue
                
                website = item.get("website", "")
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                if not domain:
                    domain = await self._find_company_domain(company_name)
                if not domain:
                    domain = extract_domain_from_company_name(company_name)
                
                if domain and not is_valid_company_domain(domain):
                    domain = ""
                
                company_id = hashlib.md5(company_name.encode()).hexdigest()[:12]
                
                # Build reason with intent
                intent = item.get("intent_signal", "")
                reason = item.get("reason_relevant", "")
                full_reason = f"📌 {intent}. {reason}" if intent else reason
                
                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=CompanySize.MID,
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
            logger.warning(f"LLM extraction failed: {e}")
            return []
    
    async def _find_company_domain(self, company_name: str) -> Optional[str]:
        """Find company domain via search."""
        try:
            return await self.tavily_tool.find_company_domain(company_name)
        except Exception:
            return None
    
    def _classify_company_size(self, text: str, company_name: str) -> CompanySize:
        """Classify company size based on context."""
        text_lower = text.lower() + " " + company_name.lower()
        
        for size, keywords in COMPANY_SIZE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return CompanySize(size)
        
        # Default to mid-size
        return CompanySize.MID
    
    def _deduplicate_companies(self, companies: List[CompanyData]) -> List[CompanyData]:
        """Remove duplicate companies by name."""
        seen = set()
        unique = []

        for company in companies:
            key = company.company_name.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(company)

        return unique

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
        norm_name = CompanyAgent._normalize_name(company_name)
        if not norm_name or len(norm_name) < 2:
            return False

        for entity in source_entities:
            norm_entity = CompanyAgent._normalize_name(entity)
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

    @staticmethod
    async def _verify_via_wikipedia(company_name: str) -> bool:
        """
        V7: Verify company existence via Wikipedia API (free, no key needed).

        Uses the Wikipedia search API to check if a page exists for the company.
        Returns True if a relevant page is found, False otherwise.
        """
        try:
            encoded = quote_plus(f"{company_name} company")
            url = (
                f"https://en.wikipedia.org/w/api.php?"
                f"action=query&list=search&srsearch={encoded}"
                f"&srlimit=3&format=json"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    results = data.get("query", {}).get("search", [])
                    if not results:
                        return False
                    # Check if any result title contains the company name
                    norm = CompanyAgent._normalize_name(company_name)
                    for result in results:
                        result_norm = CompanyAgent._normalize_name(result.get("title", ""))
                        if norm in result_norm or result_norm in norm:
                            return True
                    return False
        except Exception as e:
            logger.debug(f"Wikipedia verification failed for '{company_name}': {e}")
            return False

    async def _verify_company(
        self,
        company: CompanyData,
        source_entities: Set[str],
    ) -> CompanyData:
        """
        V7: Verify a single company against NER entities and Wikipedia.

        Sets company.ner_verified, verification_source, verification_confidence.

        Confidence scoring:
        - NER match: 0.9 (entity found in source articles)
        - Wikipedia match: 0.6 (company exists but not in article context)
        - No match: 0.2 (potentially hallucinated)
        """
        min_confidence = self.settings.company_min_verification_confidence

        # Step 1: Check against NER entities from source articles
        if self._fuzzy_entity_match(company.company_name, source_entities):
            company.ner_verified = True
            company.verification_source = "ner_match"
            company.verification_confidence = 0.9
            logger.debug(f"V7: '{company.company_name}' verified via NER match")
            return company

        # Step 2: Wikipedia fallback
        wiki_verified = await self._verify_via_wikipedia(company.company_name)
        if wiki_verified:
            company.ner_verified = True
            company.verification_source = "wikipedia"
            company.verification_confidence = 0.6
            logger.debug(f"V7: '{company.company_name}' verified via Wikipedia")
            return company

        # Step 3: No verification — flag as potentially hallucinated
        company.ner_verified = False
        company.verification_source = "unverified"
        company.verification_confidence = 0.2
        logger.warning(
            f"V7: '{company.company_name}' could NOT be verified "
            f"(not in NER entities, not on Wikipedia)"
        )

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

        logger.info(
            f"V7: Verifying {len(companies)} companies against "
            f"{len(source_entities)} source entities"
        )

        # Verify each company
        verified: List[CompanyData] = []
        min_confidence = self.settings.company_min_verification_confidence

        for company in companies:
            company = await self._verify_company(company, source_entities)
            if company.verification_confidence >= min_confidence:
                verified.append(company)
            else:
                logger.info(
                    f"V7: Filtered out '{company.company_name}' "
                    f"(confidence={company.verification_confidence:.2f} "
                    f"< threshold={min_confidence})"
                )

        # Log verification summary
        ner_count = sum(1 for c in verified if c.verification_source == "ner_match")
        wiki_count = sum(1 for c in verified if c.verification_source == "wikipedia")
        unverified_count = sum(1 for c in verified if c.verification_source == "unverified")
        filtered_count = len(companies) - len(verified)

        logger.info(
            f"V7: Verification complete: {len(verified)}/{len(companies)} passed "
            f"(NER={ner_count}, Wikipedia={wiki_count}, unverified={unverified_count}, "
            f"filtered={filtered_count})"
        )

        return verified


async def run_company_agent(state: AgentState) -> AgentState:
    """Wrapper function for LangGraph."""
    agent = CompanyAgent()
    return await agent.find_companies(state)
