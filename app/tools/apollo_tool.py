"""
Apollo.io API Tool for email finding.
Primary email finder with 600 credits/month free tier.
"""

import logging
from typing import Optional, Dict, Any
import httpx

from ..config import get_settings
from ..schemas import EmailFinderResult
from .domain_utils import extract_clean_domain, is_valid_company_domain

logger = logging.getLogger(__name__)


class ApolloTool:
    """
    Apollo.io API wrapper for finding professional emails.
    
    Endpoints used:
    - People Search: Find people at a company
    - People Enrichment: Get email for a specific person
    """
    
    APOLLO_BASE_URL = "https://api.apollo.io/v1"
    
    def __init__(self, mock_mode: bool = False):
        """Initialize Apollo tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.api_key = self.settings.apollo_api_key
    
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
            "api_key": self.api_key,
            "first_name": first_name,
            "last_name": last_name,
            "organization_name": domain.replace(".com", "").replace(".in", ""),
            "domain": domain
        }
        
        if role:
            payload["title"] = role
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                person = data.get("person", {})
                
                email = person.get("email")
                if email:
                    return EmailFinderResult(
                        email=email,
                        confidence=85,  # Apollo generally has high accuracy
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
            "api_key": self.api_key,
            "q_organization_domains": domain,
            "page": 1,
            "per_page": 5
        }
        
        if first_name:
            payload["q_keywords"] = full_name
        
        if role:
            payload["person_titles"] = [role]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
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
                            confidence=60,  # Lower confidence for non-exact match
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
    ) -> list:
        """
        Search for people at a company by role.
        
        Args:
            domain: Company domain
            roles: List of target roles
            limit: Maximum results
            
        Returns:
            List of people with their info
        """
        if self.mock_mode:
            return self._get_mock_people(domain)
        
        if not self.api_key:
            return []
        
        url = f"{self.APOLLO_BASE_URL}/mixed_people/search"
        
        payload = {
            "api_key": self.api_key,
            "q_organization_domains": domain,
            "page": 1,
            "per_page": limit
        }
        
        if roles:
            payload["person_titles"] = roles
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                    
                    return people
        except Exception as e:
            logger.error(f"Apollo people search error: {e}")
        
        return []
    
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
