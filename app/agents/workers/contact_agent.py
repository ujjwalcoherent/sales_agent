"""
Contact Finder Agent.
Finds decision-makers at companies based on trend type.
"""

import logging
import hashlib
from typing import List, Dict, Optional

from ...schemas import CompanyData, ContactData, ImpactAnalysis, AgentState
from ...tools.llm_service import LLMService
from ...tools.apollo_tool import ApolloTool
from ...config import get_settings, TREND_ROLE_MAPPING

logger = logging.getLogger(__name__)


class ContactFinder:
    """
    Finds decision-makers and influencers at target companies.

    Smart tiered approach:
    - Phase 1: Find 2-3 decision-makers (C-suite, VP, Director)
    - Phase 2: Find 2-3 influencers (Manager, Lead, Sr. Engineer)
    - Assigns seniority tier, reach score, and outreach tone per person

    NOTE: This is a deterministic pipeline stage, not an autonomous agent.
    """

    # ── Seniority classification keywords ──
    _DECISION_MAKER_KEYWORDS = frozenset({
        "ceo", "cfo", "cto", "coo", "cmo", "cio", "ciso", "cpo",
        "founder", "co-founder", "cofounder", "managing director",
        "president", "vice president", "vp", "svp", "evp",
        "director", "head of", "chief", "partner", "owner",
        "general manager", "gm",
    })
    _GATEKEEPER_KEYWORDS = frozenset({
        "assistant", "secretary", "receptionist", "office manager",
        "executive assistant", "admin", "coordinator",
    })

    @staticmethod
    def classify_tier(role: str) -> str:
        """Classify a job title into seniority tier."""
        role_lower = role.lower().strip()
        for kw in ContactFinder._DECISION_MAKER_KEYWORDS:
            if kw in role_lower:
                return "decision_maker"
        for kw in ContactFinder._GATEKEEPER_KEYWORDS:
            if kw in role_lower:
                return "gatekeeper"
        return "influencer"

    @staticmethod
    def compute_reach_score(
        email: str,
        email_confidence: int,
        verified: bool,
        linkedin_url: str,
        seniority_tier: str,
        role_relevance: float = 0.5,
    ) -> int:
        """Compute reach score (0-100) from multiple signals.

        Weights:
          - Email deliverability (40%): verified=40, confidence-based otherwise
          - LinkedIn presence (15%): URL exists = 15
          - Seniority match (25%): decision_maker=25, influencer=18, gatekeeper=8
          - Role relevance (20%): from impact target_roles match (0.0-1.0 * 20)
        """
        score = 0.0
        # Email deliverability (0-40)
        if verified:
            score += 40
        elif email:
            score += min(email_confidence * 0.4, 35)
        # LinkedIn (0-15)
        if linkedin_url:
            score += 15
        # Seniority (0-25)
        tier_scores = {"decision_maker": 25, "influencer": 18, "gatekeeper": 8}
        score += tier_scores.get(seniority_tier, 12)
        # Role relevance (0-20)
        score += role_relevance * 20
        return max(0, min(100, int(round(score))))

    @staticmethod
    def get_outreach_tone(seniority_tier: str) -> str:
        """Map seniority tier to outreach email tone."""
        return {
            "decision_maker": "executive",     # Brief, ROI-focused, strategic
            "influencer": "consultative",      # Helpful, insight-driven, problem-solving
            "gatekeeper": "professional",      # Formal, request-based, respectful
        }.get(seniority_tier, "consultative")

    def __init__(self, mock_mode: bool = False, deps=None):
        """Initialize contact agent."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        if deps:
            self.search_manager = deps.search_manager
            self.llm_service = deps.llm_service
            self.apollo_tool = deps.apollo_tool
        else:
            from ...search.manager import SearchManager
            self.search_manager = SearchManager()
            self.llm_service = LLMService(mock_mode=self.mock_mode, lite=True)
            self.apollo_tool = ApolloTool(mock_mode=self.mock_mode)
    
    async def find_contacts(self, state: AgentState) -> AgentState:
        """
        Find contacts for all companies.
        
        Args:
            state: Current agent state with companies
            
        Returns:
            Updated state with contacts
        """
        logger.info("👤 Starting contact search...")
        
        if not state.companies:
            logger.warning("No companies to find contacts for")
            return state
        
        # Build a map of trend_id to impact for role lookup
        impact_map = {imp.trend_id: imp for imp in state.impacts}
        
        all_contacts = []
        max_contacts = self.settings.max_contacts_per_company
        search_errors = []

        # Parallel contact search — semaphore limits concurrent API calls
        import asyncio
        semaphore = asyncio.Semaphore(4)

        async def _find_one(company):
            async with semaphore:
                try:
                    impact = impact_map.get(company.trend_id)
                    target_roles = (impact.target_roles if impact and impact.target_roles
                                    else ["CEO", "Founder", "CTO", "VP Operations"])
                    contacts = await self._find_contacts_for_company(
                        company, target_roles, max_contacts
                    )
                    logger.info(f"Found {len(contacts)} contacts at: {company.company_name}")
                    return contacts
                except Exception as e:
                    search_errors.append(f"{company.company_name}: {str(e)[:100]}")
                    logger.warning(f"Failed to find contacts for {company.company_name}: {e}")
                    return []

        results = await asyncio.gather(*[_find_one(c) for c in state.companies])
        for contacts in results:
            all_contacts.extend(contacts)

        state.contacts = all_contacts
        state.current_step = "contacts_found"

        if search_errors:
            state.errors.extend([f"contact_search: {e}" for e in search_errors])
            logger.warning(f"Contact search: {len(search_errors)}/{len(state.companies)} companies failed")
        logger.info(f"Found {len(all_contacts)} total contacts across {len(state.companies)} companies")
        
        return state
    
    async def _find_contacts_for_company(
        self,
        company: CompanyData,
        target_roles: List[str],
        limit: int
    ) -> List[ContactData]:
        """Find contacts at a company — tiered: decision-makers first, then influencers."""
        contacts = []

        # Split roles into tiers
        dm_roles = [r for r in target_roles if self.classify_tier(r) == "decision_maker"]
        other_roles = [r for r in target_roles if self.classify_tier(r) != "decision_maker"]

        # Fallback defaults if impact analysis didn't provide tier-specific roles
        if not dm_roles:
            dm_roles = ["CEO", "Founder", "CTO", "VP Operations"]
        if not other_roles:
            other_roles = ["Engineering Manager", "Product Manager", "Senior Engineer"]

        # Phase 1: Decision makers (up to half the limit)
        dm_limit = min(3, (limit + 1) // 2)
        if company.domain:
            try:
                apollo_dms = await self._find_via_apollo(company, dm_roles, dm_limit)
                contacts.extend(apollo_dms)
            except Exception as e:
                logger.debug(f"Apollo DM search failed for {company.company_name}: {e}")

        if len(contacts) < dm_limit:
            for role in dm_roles[:dm_limit - len(contacts)]:
                try:
                    contact = await self._find_via_search(company, role)
                    if contact and not self._contact_exists(contact, contacts):
                        contacts.append(contact)
                except Exception as e:
                    logger.debug(f"Search DM failed for {role} at {company.company_name}: {e}")

        # Phase 2: Influencers (remaining quota)
        inf_limit = limit - len(contacts)
        if inf_limit > 0 and company.domain:
            try:
                inf_contacts = await self._find_via_apollo(company, other_roles, inf_limit)
                for c in inf_contacts:
                    if not self._contact_exists(c, contacts):
                        contacts.append(c)
            except Exception as e:
                logger.debug(f"Apollo influencer search failed for {company.company_name}: {e}")

        if len(contacts) < limit:
            for role in other_roles[:limit - len(contacts)]:
                try:
                    contact = await self._find_via_search(company, role)
                    if contact and not self._contact_exists(contact, contacts):
                        contacts.append(contact)
                        if len(contacts) >= limit:
                            break
                except Exception as e:
                    logger.debug(f"Search influencer failed for {role} at {company.company_name}: {e}")

        return contacts[:limit]
    
    async def _find_via_apollo(
        self,
        company: CompanyData,
        roles: List[str],
        limit: int
    ) -> List[ContactData]:
        """Find contacts using Apollo.io API."""
        contacts = []
        
        # Search for people at the company with target roles
        people = await self.apollo_tool.search_people_at_company(
            domain=company.domain,
            roles=roles,
            limit=limit
        )
        
        for person in people:
            if not person.get("name"):
                continue
            
            contact_id = hashlib.md5(
                f"{company.id}_{person.get('name')}".encode()
            ).hexdigest()[:12]
            
            contacts.append(ContactData(
                id=contact_id,
                company_id=company.id,
                company_name=company.company_name,
                person_name=person.get("name", ""),
                role=person.get("title", ""),
                linkedin_url=person.get("linkedin_url", ""),
                email=person.get("email", ""),
                email_confidence=85 if person.get("email_verified") else 60,
                email_source="apollo" if person.get("email") else "",
                verified=person.get("email_verified", False)
            ))
        
        return contacts
    
    async def _find_via_search(
        self,
        company: CompanyData,
        role: str
    ) -> Optional[ContactData]:
        """Find a contact via SearXNG web search."""
        query = f"{role} {company.company_name} India LinkedIn"
        try:
            data = await self.search_manager.web_search(query, max_results=3)
        except Exception as e:
            logger.debug(f"Web search for {role} at {company.company_name} failed: {e}")
            data = {"results": [], "answer": ""}

        # Use LLM to extract person info from search results
        search_result = {
            "search_answer": data.get("answer", ""),
            "results": data.get("results", []),
        }
        contact_info = await self._extract_contact_from_search(
            search_result, company.company_name, role
        )
        
        if not contact_info.get("person_name"):
            return None
        
        contact_id = hashlib.md5(
            f"{company.id}_{contact_info.get('person_name')}".encode()
        ).hexdigest()[:12]
        
        return ContactData(
            id=contact_id,
            company_id=company.id,
            company_name=company.company_name,
            person_name=contact_info.get("person_name", ""),
            role=contact_info.get("role", role),
            linkedin_url=contact_info.get("linkedin_url", ""),
            email="",  # Will be found by email agent
            email_confidence=0,
            email_source=""
        )
    
    async def _extract_contact_from_search(
        self,
        search_result: Dict,
        company_name: str,
        target_role: str
    ) -> Dict:
        """Extract contact information from search results using LLM."""
        answer = search_result.get("search_answer", "")
        results = search_result.get("results", [])
        
        result_text = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')}"
            for r in results
        ])
        
        prompt = f"""Extract contact information for the {target_role} at {company_name}.

SEARCH ANSWER:
{answer}

SEARCH RESULTS:
{result_text}

Extract the following information as JSON:
- person_name: Full name of the person (or empty string if not found)
- role: Their exact job title
- linkedin_url: LinkedIn profile URL if found

Only include information you are confident about.
If no relevant person is found, return {{"person_name": ""}}"""

        try:
            result = await self.llm_service.generate_json(prompt=prompt)

            # Handle list response (NVIDIA sometimes returns array instead of object)
            if isinstance(result, list) and result:
                result = result[0]
            if not isinstance(result, dict):
                return {}

            # Clean up LinkedIn URL
            linkedin = result.get("linkedin_url", "")
            if linkedin and not linkedin.startswith("http"):
                if "linkedin.com" in linkedin:
                    linkedin = "https://" + linkedin
                else:
                    linkedin = ""
            
            return {
                "person_name": result.get("person_name", ""),
                "role": result.get("role", target_role),
                "linkedin_url": linkedin
            }
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return {}
    
    def _contact_exists(self, new_contact: ContactData, existing: List[ContactData]) -> bool:
        """Check if contact already exists in list."""
        new_name = new_contact.person_name.lower().strip()
        for contact in existing:
            if contact.person_name.lower().strip() == new_name:
                return True
        return False


# Backward compatibility alias
ContactAgent = ContactFinder


async def run_contact_agent(state: AgentState, deps=None) -> AgentState:
    """Wrapper function for LangGraph."""
    finder = ContactFinder(deps=deps) if deps else ContactFinder()
    return await finder.find_contacts(state)
