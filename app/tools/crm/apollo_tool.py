"""
Apollo.io API Tool for email finding.
Primary email finder with 600 credits/month free tier.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import httpx

from app.config import get_settings
from app.schemas import EmailFinderResult
from app.tools.domain_utils import extract_clean_domain, is_valid_company_domain

logger = logging.getLogger(__name__)

_APOLLO_SEM = asyncio.Semaphore(3)


class ApolloTool:
    """
    Apollo.io API wrapper for finding professional emails.

    Endpoints used:
    - People Search: Find people at a company
    - People Enrichment: Get email for a specific person
    """

    APOLLO_BASE_URL = "https://api.apollo.io/v1"

    # Class-level org data cache keyed by domain — avoids re-querying Apollo
    # for the same company within a session. Populated by search_people_at_company().
    _org_cache: dict[str, dict] = {}
    _ORG_CACHE_MAX = 500

    def __init__(self, mock_mode: bool = False):
        """Initialize Apollo tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.api_key = self.settings.apollo_api_key

    @classmethod
    def get_cached_org(cls, domain: str) -> dict | None:
        """Get cached Apollo organization data for a domain.

        Returns the org enrichment dict if previously fetched via
        search_people_at_company(), or None if not cached.
        """
        if not domain:
            return None
        return cls._org_cache.get(domain.lower().strip())
    
    async def find_email(
        self,
        domain: str,
        full_name: str,
        role: Optional[str] = None
    ) -> EmailFinderResult:
        """
        Find email for a person at a company.
        
        Args:
            domain: Company domain (e.g., "zoho.com")
            full_name: Person's full name
            role: Optional job title for better matching
            
        Returns:
            EmailFinderResult with email, confidence, and source
        """
        if self.mock_mode:
            return self._get_mock_result(domain, full_name)
        
        if not self.api_key:
            logger.warning("Apollo API key not configured")
            return EmailFinderResult(error="API key not configured")
        
        # Validate domain
        clean_domain = extract_clean_domain(domain)
        if not clean_domain or not is_valid_company_domain(clean_domain):
            logger.warning(f"Invalid domain for Apollo: {domain}")
            return EmailFinderResult(error=f"Invalid domain: {domain}")
        
        try:
            # Try people match first (more accurate)
            result = await self._people_match(clean_domain, full_name, role)
            if result.email:
                return result
            
            # Fallback to people search
            result = await self._people_search(clean_domain, full_name, role)
            return result
            
        except Exception as e:
            logger.error(f"Apollo API error: {e}")
            return EmailFinderResult(error=str(e))
    
    async def _people_match(
        self,
        domain: str,
        full_name: str,
        role: Optional[str]
    ) -> EmailFinderResult:
        """Use Apollo people match endpoint."""
        url = f"{self.APOLLO_BASE_URL}/people/match"
        
        # Parse name
        name_parts = full_name.strip().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        
        payload = {
            "first_name": first_name,
            "last_name": last_name,
            "organization_name": domain.split(".")[0] if "." in domain else domain,
            "domain": domain
        }

        if role:
            payload["title"] = role

        headers = {"X-Api-Key": self.api_key}
        async with _APOLLO_SEM:
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    person = data.get("person", {})

                    email = person.get("email")
                    if email:
                        return EmailFinderResult(
                            email=email,
                            confidence=85,
                            source="apollo",
                            verified=person.get("email_status") == "verified"
                        )

                elif response.status_code == 422:
                    logger.debug(f"Apollo: Person not found for {full_name} at {domain}")
                else:
                    logger.warning(f"Apollo API returned {response.status_code}")

        return EmailFinderResult(source="apollo")

    async def _people_search(
        self,
        domain: str,
        full_name: str,
        role: Optional[str]
    ) -> EmailFinderResult:
        """Use Apollo people search endpoint."""
        url = f"{self.APOLLO_BASE_URL}/mixed_people/search"
        
        # Parse name
        name_parts = full_name.strip().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        
        payload = {
            "q_organization_domains": domain,
            "page": 1,
            "per_page": 5
        }

        if first_name:
            payload["q_keywords"] = full_name

        if role:
            payload["person_titles"] = [role]

        headers = {"X-Api-Key": self.api_key}
        async with _APOLLO_SEM:
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    people = data.get("people", [])

                    # Find best match
                    for person in people:
                        person_name = f"{person.get('first_name', '')} {person.get('last_name', '')}".lower()
                        if self._name_match(full_name.lower(), person_name):
                            email = person.get("email")
                            if email:
                                return EmailFinderResult(
                                    email=email,
                                    confidence=80,
                                    source="apollo",
                                    verified=person.get("email_status") == "verified"
                                )

                    # If no exact match, return first person with email
                    for person in people:
                        email = person.get("email")
                        if email:
                            return EmailFinderResult(
                                email=email,
                                confidence=60,
                                source="apollo",
                                verified=person.get("email_status") == "verified"
                            )
                else:
                    logger.warning(f"Apollo search returned {response.status_code}")

        return EmailFinderResult(source="apollo")

    async def search_people_at_company(
        self,
        domain: str,
        roles: Optional[list] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search for people at a company by role.

        Also extracts company-level organization data (industry, employees,
        funding, tech stack) from Apollo's response — free side-effect.

        Args:
            domain: Company domain
            roles: List of target roles
            limit: Maximum results

        Returns:
            {"people": [...], "company": {...}} where company contains org enrichment
        """
        if self.mock_mode:
            return {"people": self._get_mock_people(domain), "company": {}}

        if not self.api_key:
            return {"people": [], "company": {}}

        url = f"{self.APOLLO_BASE_URL}/mixed_people/search"

        payload = {
            "q_organization_domains": domain,
            "page": 1,
            "per_page": limit
        }

        if roles:
            payload["person_titles"] = roles

        headers = {"X-Api-Key": self.api_key}
        try:
            async with _APOLLO_SEM:
                async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                    response = await client.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        people = []

                        for person in data.get("people", []):
                            people.append({
                                "name": f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                                "title": person.get("title", ""),
                                "email": person.get("email"),
                                "linkedin_url": person.get("linkedin_url", ""),
                                "email_verified": person.get("email_status") == "verified"
                            })

                        # Extract company-level org data (free side-effect, no extra API call)
                        company_enrichment = self._extract_org_data(data.get("people", []))

                        if company_enrichment and domain:
                            if len(ApolloTool._org_cache) >= ApolloTool._ORG_CACHE_MAX:
                                oldest = next(iter(ApolloTool._org_cache))
                                del ApolloTool._org_cache[oldest]
                            ApolloTool._org_cache[domain.lower().strip()] = company_enrichment

                        return {"people": people, "company": company_enrichment}
                    else:
                        logger.warning(
                            f"Apollo people search HTTP {response.status_code} for {domain} "
                            f"(roles={roles}). Response: {response.text[:200]}"
                        )
        except Exception as e:
            logger.error(f"Apollo people search error for {domain}: {e}")

        return {"people": [], "company": {}}

    @staticmethod
    def _extract_org_data(people: list) -> Dict[str, Any]:
        """Extract organization data from Apollo people search response.

        Apollo's /mixed_people/search includes organization data in each person
        record. We extract it from the first result that has it.
        """
        for person in people:
            org = person.get("organization", {})
            if not org:
                continue
            hq_parts = [org.get("city", ""), org.get("state", ""), org.get("country", "")]
            hq = ", ".join(p for p in hq_parts if p).strip(", ")
            return {
                "industry": org.get("industry", ""),
                "employee_count": str(org.get("estimated_num_employees", "")) if org.get("estimated_num_employees") else "",
                "founded_year": org.get("founded_year"),
                "description": org.get("short_description", ""),
                "headquarters": hq,
                "funding_stage": org.get("latest_funding_stage", ""),
                "funding_total": org.get("funding_total"),
                "tech_stack": org.get("technologies", []) or [],
                "website": org.get("website_url", ""),
            }
        return {}
    
    def _name_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (fuzzy)."""
        # Simple matching - first name or last name match
        parts1 = set(name1.split())
        parts2 = set(name2.split())
        
        # At least one part should match
        return bool(parts1 & parts2)
    
    def _get_mock_result(self, domain: str, full_name: str) -> EmailFinderResult:
        """Return mock result for testing."""
        # Generate a plausible email
        name_parts = full_name.lower().split()
        if len(name_parts) >= 2:
            email = f"{name_parts[0]}.{name_parts[-1]}@{domain}"
        else:
            email = f"{name_parts[0]}@{domain}"
        
        return EmailFinderResult(
            email=email,
            confidence=85,
            source="apollo",
            verified=True
        )
    
    def _get_mock_people(self, domain: str) -> list:
        """Return mock people for testing."""
        return [
            {
                "name": "Rahul Kumar",
                "title": "Chief Technology Officer",
                "email": f"rahul.kumar@{domain}",
                "linkedin_url": "https://linkedin.com/in/rahulkumar",
                "email_verified": True
            },
            {
                "name": "Priya Sharma",
                "title": "VP Engineering",
                "email": f"priya.sharma@{domain}",
                "linkedin_url": "https://linkedin.com/in/priyasharma",
                "email_verified": True
            }
        ]
