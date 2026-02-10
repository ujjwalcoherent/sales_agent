"""
Hunter.io API Tool for email finding.
Fallback email finder when Apollo fails.
"""

import logging
from typing import Optional
import httpx

from ..config import get_settings
from ..schemas import EmailFinderResult
from .domain_utils import extract_clean_domain, is_valid_company_domain

logger = logging.getLogger(__name__)


class HunterTool:
    """
    Hunter.io API wrapper for finding and verifying emails.
    
    Free tier: 25 searches/month (use as fallback)
    """
    
    HUNTER_BASE_URL = "https://api.hunter.io/v2"
    
    def __init__(self, mock_mode: bool = False):
        """Initialize Hunter tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.api_key = self.settings.hunter_api_key
    
    async def find_email(
        self,
        domain: str,
        full_name: str
    ) -> EmailFinderResult:
        """
        Find email for a person at a company using Hunter.io.
        
        Args:
            domain: Company domain (e.g., "zoho.com")
            full_name: Person's full name
            
        Returns:
            EmailFinderResult with email, confidence, and source
        """
        if self.mock_mode:
            return self._get_mock_result(domain, full_name)
        
        if not self.api_key:
            logger.warning("Hunter API key not configured")
            return EmailFinderResult(error="API key not configured")
        
        # Validate and clean domain
        clean_domain = extract_clean_domain(domain)
        if not clean_domain or not is_valid_company_domain(clean_domain):
            logger.warning(f"Invalid domain for Hunter: {domain}")
            return EmailFinderResult(error=f"Invalid domain: {domain}")
        
        # Parse name
        name_parts = full_name.strip().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        
        if not first_name:
            return EmailFinderResult(error="Name required")
        
        try:
            url = f"{self.HUNTER_BASE_URL}/email-finder"
            params = {
                "domain": clean_domain,
                "first_name": first_name,
                "last_name": last_name,
                "api_key": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    email_data = data.get("data", {})
                    
                    email = email_data.get("email")
                    score = email_data.get("score", 0)
                    
                    if email and score >= self.settings.email_confidence_threshold:
                        return EmailFinderResult(
                            email=email,
                            confidence=score,
                            source="hunter",
                            verified=email_data.get("verification", {}).get("status") == "valid"
                        )
                    elif email:
                        # Email found but low confidence
                        logger.info(f"Hunter found email with low confidence ({score}): {email}")
                        return EmailFinderResult(
                            email=email,
                            confidence=score,
                            source="hunter",
                            verified=False
                        )
                    else:
                        return EmailFinderResult(
                            error="Email not found",
                            source="hunter"
                        )
                
                elif response.status_code == 400:
                    error_data = response.json()
                    error_msg = error_data.get("errors", [{}])[0].get("details", "Bad request")
                    logger.warning(f"Hunter API error: {error_msg}")
                    return EmailFinderResult(error=error_msg, source="hunter")
                
                elif response.status_code == 401:
                    logger.error("Hunter API: Invalid API key")
                    return EmailFinderResult(error="Invalid API key", source="hunter")
                
                elif response.status_code == 429:
                    logger.warning("Hunter API: Rate limit exceeded")
                    return EmailFinderResult(error="Rate limit exceeded", source="hunter")
                
                else:
                    logger.warning(f"Hunter API returned {response.status_code}")
                    return EmailFinderResult(
                        error=f"HTTP {response.status_code}",
                        source="hunter"
                    )
                    
        except httpx.TimeoutException:
            logger.error("Hunter API timeout")
            return EmailFinderResult(error="Request timeout", source="hunter")
        except Exception as e:
            logger.error(f"Hunter API error: {e}")
            return EmailFinderResult(error=str(e), source="hunter")
    
    async def verify_email(self, email: str) -> dict:
        """
        Verify if an email address is valid.
        
        Args:
            email: Email address to verify
            
        Returns:
            Verification result with status and details
        """
        if self.mock_mode:
            return {"status": "valid", "score": 90}
        
        if not self.api_key:
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.HUNTER_BASE_URL}/email-verifier"
            params = {
                "email": email,
                "api_key": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    verification = data.get("data", {})
                    
                    return {
                        "status": verification.get("status", "unknown"),
                        "score": verification.get("score", 0),
                        "result": verification.get("result", "unknown"),
                        "smtp_check": verification.get("smtp_check", False)
                    }
                else:
                    return {"error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Hunter verify error: {e}")
            return {"error": str(e)}
    
    async def domain_search(self, domain: str, limit: int = 5) -> list:
        """
        Find all emails at a domain.
        
        Args:
            domain: Company domain
            limit: Maximum results
            
        Returns:
            List of emails found at domain
        """
        if self.mock_mode:
            return self._get_mock_domain_emails(domain)
        
        if not self.api_key:
            return []
        
        clean_domain = extract_clean_domain(domain)
        if not clean_domain:
            return []
        
        try:
            url = f"{self.HUNTER_BASE_URL}/domain-search"
            params = {
                "domain": clean_domain,
                "limit": limit,
                "api_key": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    emails = []
                    
                    for item in data.get("data", {}).get("emails", []):
                        emails.append({
                            "email": item.get("value"),
                            "first_name": item.get("first_name", ""),
                            "last_name": item.get("last_name", ""),
                            "position": item.get("position", ""),
                            "confidence": item.get("confidence", 0)
                        })
                    
                    return emails
        except Exception as e:
            logger.error(f"Hunter domain search error: {e}")
        
        return []
    
    async def generate_email_pattern(
        self,
        domain: str,
        first_name: str,
        last_name: str
    ) -> Optional[str]:
        """
        Generate likely email using domain's email pattern.
        
        Args:
            domain: Company domain
            first_name: Person's first name
            last_name: Person's last name
            
        Returns:
            Generated email based on domain pattern
        """
        # Common email patterns
        patterns = [
            "{first}.{last}@{domain}",
            "{first}{last}@{domain}",
            "{f}{last}@{domain}",
            "{first}@{domain}",
            "{first}_{last}@{domain}"
        ]
        
        first = first_name.lower().strip()
        last = last_name.lower().strip() if last_name else ""
        f = first[0] if first else ""
        
        clean_domain = extract_clean_domain(domain)
        if not clean_domain:
            return None
        
        # Default to first.last pattern
        if last:
            return f"{first}.{last}@{clean_domain}"
        else:
            return f"{first}@{clean_domain}"
    
    def _get_mock_result(self, domain: str, full_name: str) -> EmailFinderResult:
        """Return mock result for testing."""
        name_parts = full_name.lower().split()
        if len(name_parts) >= 2:
            email = f"{name_parts[0]}.{name_parts[-1]}@{domain}"
        else:
            email = f"{name_parts[0]}@{domain}"
        
        return EmailFinderResult(
            email=email,
            confidence=85,
            source="hunter",
            verified=True
        )
    
    def _get_mock_domain_emails(self, domain: str) -> list:
        """Return mock domain emails for testing."""
        return [
            {
                "email": f"contact@{domain}",
                "first_name": "",
                "last_name": "",
                "position": "General",
                "confidence": 95
            },
            {
                "email": f"info@{domain}",
                "first_name": "",
                "last_name": "",
                "position": "General",
                "confidence": 90
            }
        ]
