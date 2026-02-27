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
    Finds decision-makers at target companies.

    For each company:
    - Determines the best role to target based on trend
    - Searches for contact information (Apollo + web search)
    - Extracts name, role, and LinkedIn URL

    NOTE: This is a deterministic pipeline stage, not an autonomous agent.
    Renamed from ContactAgent for honest naming.
    """
    
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
                    logger.warning(f"Failed to find contacts for {company.company_name}: {e}")
                    return []

        results = await asyncio.gather(*[_find_one(c) for c in state.companies])
        for contacts in results:
            all_contacts.extend(contacts)

        state.contacts = all_contacts
        state.current_step = "contacts_found"
        logger.info(f"Found {len(all_contacts)} total contacts")
        
        return state
    
    async def _find_contacts_for_company(
        self,
        company: CompanyData,
        target_roles: List[str],
        limit: int
    ) -> List[ContactData]:
        """
        Find contacts at a specific company.
        
        Args:
            company: Company to search
            target_roles: List of roles to target
            limit: Maximum contacts to find
            
        Returns:
            List of ContactData objects
        """
        contacts = []
        
        # Method 1: Try Apollo first if we have a valid domain
        if company.domain:
            try:
                apollo_contacts = await self._find_via_apollo(
                    company, target_roles, limit
                )
                contacts.extend(apollo_contacts)
            except Exception as e:
                logger.debug(f"Apollo search failed for {company.company_name}: {e}")
        
        # Method 2: Web search for additional contacts
        if len(contacts) < limit:
            remaining = limit - len(contacts)
            for role in target_roles[:remaining]:
                try:
                    contact = await self._find_via_search(company, role)
                    if contact and not self._contact_exists(contact, contacts):
                        contacts.append(contact)
                        if len(contacts) >= limit:
                            break
                except Exception as e:
                    logger.debug(f"Search failed for {role} at {company.company_name}: {e}")
        
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
