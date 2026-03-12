"""
Hunter.io API Tool for email finding.
Fallback email finder when Apollo fails.
"""

import asyncio
import logging
from typing import Optional
import httpx

from app.config import get_settings
from app.schemas import EmailFinderResult
from app.tools.domain_utils import extract_clean_domain, is_valid_company_domain

logger = logging.getLogger(__name__)

_HUNTER_SEM = asyncio.Semaphore(2)

# Module-level caches populated by domain_search() — shared across all HunterTool instances.
# Lets generate_email_pattern() use the real pattern after the first domain_search credit.
_domain_patterns: dict[str, str] = {}   # domain → Hunter pattern e.g. "{first}.{last}"
_accept_all_domains: set[str] = set()   # domains where SMTP verification is meaningless


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

            async with _HUNTER_SEM:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, params=params)

                    if response.status_code == 200:
                        data = response.json()
                        email_data = data.get("data", {})

                        email = email_data.get("email")
                        score = email_data.get("score", 0)
                        verif = email_data.get("verification", {}) or {}
                        verif_result = verif.get("result", "")

                        # "risky" = mx record exists but SMTP unconfirmable → cap confidence at 60
                        if verif_result == "risky" and score > 60:
                            score = 60
                        # "undeliverable" = hard bounce guaranteed → cap at 20
                        if verif_result == "undeliverable":
                            score = min(score, 20)

                        # Capture enrichment fields returned at no extra credit cost
                        linkedin_url = email_data.get("linkedin_url") or ""
                        phone_number = email_data.get("phone_number") or ""
                        position = email_data.get("position") or ""

                        if email and score >= self.settings.email_confidence_threshold:
                            return EmailFinderResult(
                                email=email,
                                confidence=score,
                                source="hunter",
                                verified=verif.get("status") == "valid" and verif_result == "deliverable",
                                linkedin_url=linkedin_url,
                                phone_number=phone_number,
                                position=position,
                            )
                        elif email:
                            logger.info(f"Hunter found email with low confidence ({score}, result={verif_result}): {email}")
                            return EmailFinderResult(
                                email=email,
                                confidence=score,
                                source="hunter",
                                verified=False,
                                linkedin_url=linkedin_url,
                                phone_number=phone_number,
                                position=position,
                            )
                        else:
                            return EmailFinderResult(error="Email not found", source="hunter")

                    elif response.status_code == 400:
                        error_data = response.json()
                        error_msg = error_data.get("errors", [{}])[0].get("details", "Bad request")
                        logger.warning(f"Hunter API error: {error_msg}")
                        return EmailFinderResult(error=error_msg, source="hunter")

                    elif response.status_code == 401:
                        logger.error("Hunter API: Invalid API key")
                        return EmailFinderResult(error="Invalid API key", source="hunter")

                    elif response.status_code == 429:
                        logger.warning("Hunter API: Rate limit exceeded — falling back to email pattern guess")
                        guessed = await self.generate_email_pattern(clean_domain, first_name, last_name)
                        if guessed:
                            return EmailFinderResult(
                                email=guessed,
                                confidence=35,  # Low confidence — unverified guess
                                source="pattern_guess",
                                verified=False,
                            )
                        return EmailFinderResult(error="Rate limit exceeded", source="hunter")

                    else:
                        logger.warning(f"Hunter API returned {response.status_code}")
                        return EmailFinderResult(error=f"HTTP {response.status_code}", source="hunter")
                    
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

        # accept_all domains: every address appears valid — verification credit is wasted
        domain = email.split("@")[-1] if "@" in email else ""
        if domain in _accept_all_domains:
            logger.debug(f"Skipping verify_email for accept_all domain: {domain}")
            return {"status": "accept_all", "score": 50, "result": "risky", "smtp_check": False}

        try:
            url = f"{self.HUNTER_BASE_URL}/email-verifier"
            params = {
                "email": email,
                "api_key": self.api_key
            }

            async with _HUNTER_SEM:
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
    
    async def domain_search(
        self,
        domain: str,
        limit: int = 5,
        seniority: str = "senior,executive",
    ) -> list:
        """
        Find all emails at a domain.

        Args:
            domain: Company domain
            limit: Maximum results
            seniority: Comma-separated Hunter seniority filter (default: senior+executive only)
                       Pass "" to get all levels.

        Returns:
            List of emails found at domain (personal type only, server-side filtered)
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
            params: dict = {
                "domain": clean_domain,
                "limit": limit,
                "type": "personal",   # Skip role/generic emails server-side (no extra cost)
                "api_key": self.api_key,
            }
            if seniority:
                params["seniority"] = seniority

            async with _HUNTER_SEM:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, params=params)

                    if response.status_code == 200:
                        data = response.json()
                        domain_data = data.get("data", {})

                        # Domain-level metadata — one flag changes how we interpret ALL emails
                        accept_all = domain_data.get("accept_all", False)
                        domain_pattern = domain_data.get("pattern", "")  # e.g. "{first}.{last}"

                        # Populate module-level caches — used by generate_email_pattern()
                        # and verify_email() to avoid spending credits unnecessarily
                        if domain_pattern:
                            _domain_patterns[clean_domain] = domain_pattern
                        if accept_all:
                            _accept_all_domains.add(clean_domain)
                            logger.debug(f"Hunter: {clean_domain} is accept_all — verification scores capped at 50")

                        emails = []
                        for item in domain_data.get("emails", []):
                            verif = item.get("verification", {}) or {}
                            verif_result = verif.get("result", "")
                            conf = item.get("confidence", 0)
                            already_verified = verif.get("status") == "valid"

                            # accept_all domains: MX accepts everything, SMTP verification is meaningless
                            if accept_all:
                                conf = min(conf, 50)
                                already_verified = False

                            # Cap risky results
                            if verif_result == "risky":
                                conf = min(conf, 60)

                            emails.append({
                                "email": item.get("value"),
                                "first_name": item.get("first_name", ""),
                                "last_name": item.get("last_name", ""),
                                "position": item.get("position", ""),
                                "seniority": item.get("seniority", ""),
                                "department": item.get("department", ""),
                                "confidence": conf,
                                # Pre-verified by Hunter — saves a verify_email credit downstream
                                "verified": already_verified,
                                "verification_result": verif_result,
                                # Extra enrichment fields — free data in the same response
                                "linkedin": item.get("linkedin") or "",
                                "phone_number": item.get("phone_number") or "",
                                "twitter": item.get("twitter") or "",
                                # Domain-level data — useful for email guessing on other contacts
                                "domain_pattern": domain_pattern,
                                "accept_all": accept_all,
                            })
                        return emails
                    elif response.status_code == 429:
                        logger.warning(
                            f"Hunter domain_search rate limited (monthly quota exhausted). "
                            f"Upgrade plan at https://hunter.io/pricing or wait for reset."
                        )
                        return []
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
        first = first_name.lower().strip()
        last = last_name.lower().strip() if last_name else ""

        clean_domain = extract_clean_domain(domain)
        if not clean_domain:
            return None

        # Use domain pattern discovered by a prior domain_search() credit (free reuse)
        pattern = _domain_patterns.get(clean_domain, "")
        if pattern and last:
            # Hunter patterns: {first}, {last}, {first}.{last}, {f}.{last}, {first}{last}
            email_local = (
                pattern
                .replace("{first}", first)
                .replace("{last}", last)
                .replace("{f}", first[0] if first else "")
                .replace("{l}", last[0] if last else "")
            )
            return f"{email_local}@{clean_domain}"

        # Fallback: first.last (most common B2B pattern)
        if last:
            return f"{first}.{last}@{clean_domain}"
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
    
    # Hunter domain_search returns individual employees, not generic addresses.
    # These are realistic mid-level contacts that complement Apollo's C-suite data.
    _HUNTER_DOMAIN_CONTACTS: dict[str, list[tuple[str, str, str, int]]] = {
        # (first, last, position, confidence)
        "mphasis": [
            ("Sneha", "Deshmukh", "Engineering Manager Cloud Platforms", 92),
            ("Karthik", "Subramanian", "Senior Solution Architect", 88),
            ("Pooja", "Choudhary", "Product Manager Digital", 85),
        ],
        "persistent": [
            ("Rohit", "Deshpande", "Director Engineering BFSI", 91),
            ("Megha", "Kulkarni", "Engineering Manager Healthcare", 87),
            ("Siddharth", "Menon", "Senior Technical Architect", 84),
        ],
        "happiestminds": [
            ("Nandini", "Rao", "Director Product Engineering", 90),
            ("Varun", "Krishnan", "Engineering Manager IoT", 86),
            ("Divya", "Raghunathan", "Senior DevOps Architect", 83),
        ],
        "razorpay": [
            ("Aditya", "Sharma", "Engineering Manager Payments", 93),
            ("Meghana", "Bhat", "Senior Product Manager UPI", 89),
            ("Rohan", "Malhotra", "Director Platform Engineering", 86),
        ],
        "pinelabs": [
            ("Vikram", "Nair", "Director Payment Solutions", 91),
            ("Shreya", "Kapoor", "Engineering Manager Merchant Tech", 87),
            ("Arjun", "Menon", "Senior Architect Fintech", 84),
        ],
        "easebuzz": [
            ("Neha", "Tiwari", "Product Manager Payments", 89),
            ("Gaurav", "Singh", "Engineering Manager Backend", 85),
            ("Ritu", "Agarwal", "Director Compliance", 82),
        ],
        "signzy": [
            ("Harsh", "Vardhan", "Engineering Manager KYC Platform", 90),
            ("Swati", "Iyer", "Product Manager Compliance", 86),
            ("Deepak", "Mishra", "Director Solutions Engineering", 83),
        ],
        "idfy": [
            ("Pranav", "Shah", "Director Engineering AI/ML", 91),
            ("Ruchika", "Joshi", "Product Manager Identity", 87),
            ("Anand", "Kulkarni", "Engineering Manager Verification", 84),
        ],
        "stl.tech": [
            ("Rahul", "Patil", "Director 5G Network Solutions", 90),
            ("Aparna", "Nambiar", "Engineering Manager Optical", 86),
            ("Saurabh", "Tiwari", "Product Manager Digital Services", 83),
        ],
        "tejasnetworks": [
            ("Priya", "Ramachandran", "Director Product Engineering", 91),
            ("Girish", "Shetty", "Engineering Manager 5G Systems", 87),
            ("Tanvi", "Desai", "Senior Architect Telecom", 84),
        ],
        "sasken": [
            ("Ramesh", "Iyer", "Director Embedded Systems", 90),
            ("Kavitha", "Rangan", "Engineering Manager IoT", 86),
            ("Vivek", "Sharma", "Product Manager Automotive", 83),
        ],
    }

    _HUNTER_FALLBACK_CONTACTS = [
        ("Rajesh", "Kumar", "Director of Engineering", 90),
        ("Priyanka", "Singh", "Engineering Manager", 86),
        ("Amit", "Patel", "Senior Product Manager", 83),
        ("Neha", "Reddy", "DevOps Lead", 80),
    ]

    def _get_mock_domain_emails(self, domain: str) -> list:
        """Return realistic domain emails using public leadership research."""
        domain_lower = (domain or "").lower()

        contacts = None
        for key, data in self._HUNTER_DOMAIN_CONTACTS.items():
            if key in domain_lower:
                contacts = data
                break

        if not contacts:
            contacts = self._HUNTER_FALLBACK_CONTACTS

        results = []
        for first, last, position, confidence in contacts:
            email = f"{first.lower()}.{last.lower()}@{domain}"
            results.append({
                "email": email,
                "first_name": first,
                "last_name": last,
                "position": position,
                "confidence": confidence,
                "seniority": "senior",
                "type": "personal",
            })
        return results
