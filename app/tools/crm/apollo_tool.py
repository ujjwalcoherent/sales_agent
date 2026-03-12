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
            return {"people": self._get_mock_people(domain, roles, limit), "company": {}}

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
    
    # Real company contact databases built from public leadership pages.
    # Keyed by domain fragment → list of (name, title, linkedin_slug).
    _MOCK_COMPANY_CONTACTS: dict[str, list[tuple[str, str, str]]] = {
        # ── IT Services (Company-First use case) ──
        "mphasis": [
            ("Nitin Rakesh", "Chief Executive Officer", "nitin-rakesh"),
            ("Anup Nair", "SVP & Chief Technology Officer", "anupnair-cto"),
            ("Mahesh Lalwani", "VP Data & Artificial Intelligence", "maheshlalwani"),
            ("Aditya Chati", "VP Design Innovation & Prototyping", "adityachati"),
            ("Vivek Agarwal", "VP Sales", "vivekagarwal-mphasis"),
            ("Christopher Fernandes", "Vice President Delivery", "christopherfernandes"),
            ("Jayaprakash Bandu", "Vice President Engineering", "jayaprakashbandu"),
            ("Aziz Shaikhali", "VP & Client Partner", "azizshaikhali"),
        ],
        "persistent": [
            ("Sandeep Kalra", "Chief Executive Officer", "sandeepkalra"),
            ("Shriram N", "Chief Technology Officer", "shriramn-persistent"),
            ("Jaideep Dhok", "COO Technology", "jaideepdhok"),
            ("Vinit Teredesai", "Chief Financial Officer", "vinitteredesai"),
            ("Nitha Puthran", "Executive Vice President", "nithaputhran"),
            ("Tom Klein", "General Counsel & SVP Corporate Development", "tomklein-persistent"),
        ],
        "happiestminds": [
            ("Joseph Anantharaju", "Co-Chairman & CEO", "josephanantharaju"),
            ("Anand Balakrishnan", "Chief Financial Officer", "anandbalakrishnan-hm"),
            ("Prathamesh Kulkarni", "CEO BFSI & Healthcare", "prathameshkulkarni"),
            ("Jaganath Ram Shankar", "Head of Cloud & Infrastructure", "jaganathram"),
            ("Suresh Chettur", "Head Digital Process Automation", "sureshchettur"),
            ("Subhasis Bandyopadhyay", "Head of BFSI Industry Group", "subhasisbandyopadhyay"),
        ],
        "zensar": [
            ("Manish Tandon", "CEO & Managing Director", "manishtandon-zensar"),
            ("Prameela Kalive", "Chief Operating Officer", "prameelakalive"),
            ("Venky Ramanan", "EVP Hi-Tech & Manufacturing", "venkyramanan"),
            ("Ajay Bhanushali", "VP Engineering", "ajaybhanushali"),
            ("Rohit Kedia", "Director Cloud Services", "rohitkedia-zensar"),
        ],
        "birlasoft": [
            ("Angan Guha", "Chief Executive Officer", "anganguha"),
            ("Ganesan Karuppanaicker", "Chief Technology Officer", "ganesanktech"),
            ("Anand Sinha", "CIO & Global Head IT", "anandsinha-birlasoft"),
            ("Gurunath Muthu", "VP APAC & GCC", "gurunathmuthu"),
            ("Dharmender Kapoor", "Managing Director", "dharmenderkapoor"),
        ],
        "coforge": [
            ("Sudhir Singh", "CEO & Executive Director", "sudhirsingh19"),
            ("Saurabh Goel", "Chief Financial Officer", "saurabhgoel-coforge"),
            ("Pankaj Khanna", "EVP & Business Unit Head", "pankajkhanna-coforge"),
            ("Gautam Samanta", "COO", "gautamsamanta"),
            ("Arun Vasudev", "VP Engineering & Delivery", "arunvasudev-coforge"),
        ],
        "ltts": [
            ("Amit Chadha", "CEO & Managing Director", "amitchadha-ltts"),
            ("Abhishek Sinha", "Executive Director & President", "abhisheksinha-ltts"),
            ("Ashish Khushu", "Chief Technology Officer", "ashishkhushu"),
            ("Padmanabhan Iyer", "VP Engineering Services", "padmanabhaniyer"),
            ("Rajeev Gupta", "Director Innovation Lab", "rajeevgupta-ltts"),
        ],
        "cyient": [
            ("Sukamal Banerjee", "CEO & Executive Director", "sukamalbanerjee"),
            ("K A Prabhakaran", "Chief Technology Officer", "kaprabhakaran"),
            ("Anand Parameswaran", "President Global Delivery", "anandparameswaran"),
            ("Ajay Aggarwal", "Chief Financial Officer", "ajayaggarwal-cyient"),
            ("Katie Cook", "SVP Engineering Services", "katiecook-cyient"),
        ],
        "tataelxsi": [
            ("Manoj Raghavan", "CEO & Managing Director", "manojraghavan-elxsi"),
            ("Nitin Pai", "CMO & Chief Strategy Officer", "nitinpai-elxsi"),
            ("Gaurav Bajaj", "Chief Financial Officer", "gauravbajaj-elxsi"),
            ("Philip Mammen", "VP Human Resources", "philipmammen"),
            ("Anand Sahay", "Director Transportation Engineering", "anandsahay-elxsi"),
        ],
        # ── Fintech/BFSI (Industry-First use case) ──
        "razorpay": [
            ("Harshil Mathur", "CEO & Co-Founder", "harshilmathur"),
            ("Prabu Rambadran", "SVP Engineering", "praburambadran"),
            ("Khilan Haria", "Chief Product Officer", "khilanharia"),
            ("Rahul Kothari", "Chief Operating Officer", "rahulkothari-rz"),
            ("Arpit Chugh", "Chief Financial Officer", "arpitchugh-rz"),
            ("Shashank Kumar", "MD & Co-Founder", "shashankkumar-rz"),
        ],
        "pinelabs": [
            ("Amrish Rau", "Chairman & Managing Director", "amrishrau"),
            ("Sanjeev Kumar", "Chief Technology Officer", "sanjeevkumar-pinelabs"),
            ("Nitish Asthana", "President & COO", "nitishasthana"),
            ("Kush Mehra", "President Digital Infrastructure", "kushmehra"),
            ("Amrita Gangotra", "Independent Director", "amritagangotra"),
        ],
        "easebuzz": [
            ("Rohit Prasad", "MD & CEO", "rohitprasad-easebuzz"),
            ("Amit Kumar", "CTO & Director", "amitkumar-easebuzz"),
            ("Rohan Sharma", "Director & SVP", "rohansharma-easebuzz"),
            ("Ravikant Gour", "SVP Product Management", "ravikantgour"),
            ("Parimal Kumar Shivendu", "Group Head Operations", "parimalshivendu"),
        ],
        "lendingkart": [
            ("Prashant Joshi", "MD & CEO", "prashantjoshi-lk"),
            ("Harshvardhan Lunia", "Founder & Director", "harshvardhanlunia"),
            ("Rashmi Sharma", "Director Legal & Compliance", "rashmisharma-lk"),
            ("Naveen Gupta", "VP Product", "naveengupta-lk"),
            ("Raman Bhatia", "VP Risk & Analytics", "ramanbhatia-lk"),
        ],
        "signzy": [
            ("Ankit Ratan", "CEO & Co-Founder", "ankitratan"),
            ("Ankur Pandey", "CTO & Co-Founder", "ankurpandey-signzy"),
            ("Arpit Ratan", "CBO & Co-Founder", "arpitratan-signzy"),
            ("Vishal Sharma", "VP Engineering", "vishalsharma-signzy"),
            ("Priyanka Agarwal", "Head of Compliance Solutions", "priyankaagarwal-signzy"),
        ],
        "idfy": [
            ("Ashok Hariharan", "CEO & Co-Founder", "ashokhariharan"),
            ("Ashish Sahni", "Chief Technology Officer", "ashishsahni-idfy"),
            ("Paritosh Desai", "Chief Product Officer", "paritoshdesai"),
            ("Tridib Mukherjee", "Chief AI Officer", "tridibmukherjee"),
            ("Vineet Jawa", "Co-Founder & Director", "vineetjawa"),
        ],
        # ── 5G / IoT / Telecom (Report-Driven use case) ──
        "stl.tech": [
            ("Badri Gomatam", "Group CTO", "badrigomatam"),
            ("Spandan Mahapatra", "CTO Digital Services", "spandanmahapatra"),
            ("Pankaj Malik", "CEO Global Services", "pankajmalik-stl"),
            ("Ajay Jhanjhari", "Chief Financial Officer", "ajayjhanjhari"),
            ("Ayush Sharma", "Head Programmable Networking", "ayushsharma-stl"),
            ("Rajesh Gangadhar", "Head Wireless Broadband", "rajeshgangadhar"),
        ],
        "tejasnetworks": [
            ("Arnob Roy", "CEO & Executive Director", "arnobroy"),
            ("Kumar N", "CTO & Co-Founder", "kumarn-tejas"),
            ("Sumit Dhingra", "Chief Financial Officer", "sumitdhingra-tejas"),
            ("Sanjay Malik", "EVP Chief Strategy Officer", "sanjaymalik-tejas"),
            ("Ravi Shankar", "VP Engineering 5G", "ravishankar-tejas"),
        ],
        "sasken": [
            ("Rajiv C Mody", "CEO & Chairman", "rajivcmody"),
            ("Jaimir Sanghvi", "VP Sales India & North America", "jaimirsanghvi"),
            ("Alwyn Joseph Premkumar", "Deputy CEO Americas", "alwynjoseph"),
            ("Neelu Sinha", "VP Automotive & Industrials", "neelusinha-sasken"),
            ("Aravind Srinivas", "Director IoT Solutions", "aravindsrinivas-sasken"),
        ],
        "endurancegroup": [
            ("Anurang Jain", "Managing Director", "anurangjain"),
            ("Rajendra Abhange", "Director & COO", "rajendraabhange"),
            ("Ramesh Gehaney", "Director Manufacturing", "rameshgehaney"),
            ("Satish Patel", "VP Engineering", "satishpatel-endurance"),
            ("Priya Deshmukh", "Head Quality & IoT Integration", "priyadeshmukh-endurance"),
        ],
    }

    # Fallback names when domain doesn't match any known company
    _MOCK_FALLBACK_PEOPLE = [
        ("Vikram Mehta", "vikrammehta-tech"),
        ("Anjali Gupta", "anjaligupta-eng"),
        ("Rajesh Iyer", "rajeshiyer-ops"),
        ("Kavita Reddy", "kavitareddy-strat"),
        ("Suresh Nair", "sureshnair-fin"),
        ("Deepika Joshi", "deepikajoshi-mkt"),
        ("Amit Saxena", "amitsaxena-data"),
        ("Sneha Deshmukh", "snehadeshmukh-prod"),
        ("Siddharth Menon", "sidmenon-arch"),
        ("Nandini Rao", "nandinirao-bd"),
    ]

    def _get_mock_people(self, domain: str, roles: list | None = None, limit: int = 5) -> list:
        """Return mock contacts using real leadership data from public sources.

        Matches domain against known company databases. Falls back to
        role-aware generated contacts for unknown domains.
        """
        domain_lower = (domain or "").lower()

        # Try to match domain against known companies
        matched_contacts = None
        for key, contacts in self._MOCK_COMPANY_CONTACTS.items():
            if key in domain_lower:
                matched_contacts = contacts
                break

        if matched_contacts:
            results = []
            for name, title, slug in matched_contacts[:limit]:
                name_parts = name.lower().split()
                email = f"{name_parts[0]}.{name_parts[-1]}@{domain}"
                results.append({
                    "name": name,
                    "title": title,
                    "email": email,
                    "linkedin_url": f"https://linkedin.com/in/{slug}",
                    "email_verified": True,
                })
            return results

        # Unknown domain: use fallback pool with requested roles
        seed = sum(ord(c) for c in domain_lower) if domain_lower else 0
        pool = list(self._MOCK_FALLBACK_PEOPLE)
        offset = seed % len(pool)
        pool = pool[offset:] + pool[:offset]

        fallback_titles = [
            "Chief Technology Officer", "VP Engineering", "Director of Engineering",
            "Engineering Manager", "Product Manager", "Head of Strategy",
            "Senior Architect", "Director Operations", "Data Science Manager",
            "DevOps Manager",
        ]
        titles = list(roles) if roles else fallback_titles
        count = min(limit, len(pool))
        results = []
        for i in range(count):
            name, slug = pool[i]
            title = titles[i % len(titles)] if titles else "Manager"
            name_parts = name.lower().split()
            email = f"{name_parts[0]}.{name_parts[-1]}@{domain}" if domain else f"{name_parts[0]}@example.com"
            results.append({
                "name": name,
                "title": title,
                "email": email,
                "linkedin_url": f"https://linkedin.com/in/{slug}",
                "email_verified": i < (count * 3 // 4),
            })
        return results
