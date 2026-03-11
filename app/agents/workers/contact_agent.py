"""
Contact Finder Agent.
Finds decision-makers at companies based on trend type.
"""

import logging
import hashlib
import re
from typing import List, Dict, Optional

from ...schemas import CompanyData, ContactData, ImpactAnalysis, AgentState
from app.tools.llm.llm_service import LLMService
from app.tools.crm.apollo_tool import ApolloTool
from ...config import get_settings, TREND_ROLE_MAPPING

logger = logging.getLogger(__name__)


def match_roles_to_trend(
    trend_type: str,
    pain_point: str = "",
    who_needs_help: str = "",
    trend_title: str = "",
) -> list[str]:
    """Determine which roles to target based on the trend.

    Uses trend_type as primary signal, then scans trend_title, pain_point,
    and who_needs_help for role keywords. Returns ordered list of target roles.
    """
    roles = list(TREND_ROLE_MAPPING.get(trend_type, []))

    # Also try scanning trend_type itself for partial matches when it's a
    # free-form LLM string like "AI security acquisition"
    if not roles and trend_type and trend_type != "general":
        tt_lower = trend_type.lower()
        for key in TREND_ROLE_MAPPING:
            if key != "default" and key in tt_lower:
                roles = list(TREND_ROLE_MAPPING[key])
                break

    # Scan trend_title, pain_point, and who_needs_help for additional role signals
    text = f"{trend_title} {pain_point} {who_needs_help}".lower()
    if "security" in text or "cyber" in text or "breach" in text:
        roles = list(TREND_ROLE_MAPPING.get("cybersecurity", [])) + roles
    if "cost" in text or "budget" in text or "expense" in text or "savings" in text:
        roles = list(TREND_ROLE_MAPPING.get("cost_reduction", [])) + roles
    if " ai " in f" {text} " or "artificial intelligence" in text or "machine learning" in text:
        roles = list(TREND_ROLE_MAPPING.get("ai_adoption", [])) + roles
    if "cloud" in text or "infrastructure" in text or "migration" in text:
        roles = list(TREND_ROLE_MAPPING.get("cloud_migration", [])) + roles
    if "compliance" in text or "regulatory" in text or "gdpr" in text:
        roles = list(TREND_ROLE_MAPPING.get("compliance", [])) + roles
    if "sustainability" in text or "esg" in text or "carbon" in text or "green" in text:
        roles = list(TREND_ROLE_MAPPING.get("sustainability", [])) + roles
    if "talent" in text or "hiring" in text or "workforce" in text or "retention" in text:
        roles = list(TREND_ROLE_MAPPING.get("talent", [])) + roles
    if "privacy" in text or "data protection" in text:
        roles = list(TREND_ROLE_MAPPING.get("data_privacy", [])) + roles
    if "acqui" in text or "merger" in text or "m&a" in text or "buyout" in text:
        roles = list(TREND_ROLE_MAPPING.get("funding", [])) + roles
    if "expansion" in text or "expand" in text or "growth" in text or "market entry" in text:
        roles = list(TREND_ROLE_MAPPING.get("expansion", [])) + roles
    if "funding" in text or "raise" in text or "series " in text or "ipo" in text:
        roles = list(TREND_ROLE_MAPPING.get("funding", [])) + roles

    # If no roles found, use default
    if not roles:
        roles = list(TREND_ROLE_MAPPING.get("default", []))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for r in roles:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:8]  # Cap at 8 roles


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
            self.hunter_tool = getattr(deps, "hunter_tool", None)
        else:
            from app.tools.search import SearchManager
            from app.tools.crm.hunter_tool import HunterTool
            self.search_manager = SearchManager()
            self.llm_service = LLMService(mock_mode=self.mock_mode, lite=True)
            self.apollo_tool = ApolloTool(mock_mode=self.mock_mode)
            self.hunter_tool = HunterTool(mock_mode=self.mock_mode)
    
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
        
        # Build maps for role lookup
        impact_map = {imp.trend_id: imp for imp in state.impacts}
        trend_map = {t.id: t for t in state.trends} if state.trends else {}
        
        all_contacts = []
        max_contacts = self.settings.max_contacts_per_company
        search_errors = []

        # Parallel contact search — semaphore limits concurrent API calls
        import asyncio
        semaphore = asyncio.Semaphore(4)

        async def _find_one(company):
            async with semaphore:
                try:
                    # Priority: company.target_roles (LLM-inferred) > trend match > defaults
                    if getattr(company, "target_roles", None):
                        target_roles = list(company.target_roles)
                    else:
                        impact = impact_map.get(company.trend_id)
                        trend = trend_map.get(company.trend_id)
                        trend_type = getattr(trend, "trend_type", "") or "" if trend else ""
                        trend_title = getattr(trend, "trend_title", "") or "" if trend else ""
                        pain_point = " ".join(getattr(impact, "midsize_pain_points", []) or []) if impact else ""
                        who_needs_help = getattr(impact, "who_needs_help", "") or "" if impact else ""
                        matched_roles = match_roles_to_trend(
                            trend_type, pain_point, who_needs_help, trend_title=trend_title,
                        )
                        if matched_roles:
                            target_roles = matched_roles
                        elif impact and impact.target_roles:
                            target_roles = impact.target_roles
                        else:
                            target_roles = list(TREND_ROLE_MAPPING.get("default", []))

                        # ContactBandit re-ranking: prioritize roles that historically
                        # produce high-value contacts for this event type
                        try:
                            from app.learning.contact_bandit import ContactBandit
                            _cb = ContactBandit.load()
                            _event = trend_type or "general"
                            _size = getattr(company, "company_size_band", "mid") or "mid"
                            _ranked = _cb.rank_roles(
                                roles=target_roles, event_type=_event, company_size=_size,
                            )
                            # rank_roles returns List[Tuple[str, float]] — extract role names
                            target_roles = [r for r, _ in _ranked]
                        except Exception:
                            pass  # Graceful degradation: use original role order
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

        # Fallback defaults from settings (not hardcoded)
        if not dm_roles:
            dm_roles = [r.strip() for r in self.settings.default_dm_roles.split(",") if r.strip()]
        if not other_roles:
            other_roles = [r.strip() for r in self.settings.default_influencer_roles.split(",") if r.strip()]

        # Phase 1: Decision makers (up to half the limit, at least 3)
        dm_limit = min(max(3, limit // 2), limit)
        if company.domain:
            try:
                apollo_dms = await self._find_via_apollo(company, dm_roles, dm_limit)
                contacts.extend(apollo_dms)
            except Exception as e:
                logger.debug(f"Apollo DM search failed for {company.company_name}: {e}")

        if len(contacts) < dm_limit:
            import asyncio as _aio
            needed = dm_limit - len(contacts)
            dm_search_results = await _aio.gather(
                *[self._find_via_search(company, role) for role in dm_roles[:needed]],
                return_exceptions=True,
            )
            for i, result in enumerate(dm_search_results):
                if isinstance(result, Exception):
                    logger.debug(f"Search DM failed for {dm_roles[i]} at {company.company_name}: {result}")
                    continue
                if result and not self._contact_exists(result, contacts):
                    contacts.append(result)

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
            import asyncio as _aio
            needed = limit - len(contacts)
            inf_search_results = await _aio.gather(
                *[self._find_via_search(company, role) for role in other_roles[:needed]],
                return_exceptions=True,
            )
            for i, result in enumerate(inf_search_results):
                if isinstance(result, Exception):
                    logger.debug(f"Search influencer failed for {other_roles[i]} at {company.company_name}: {result}")
                    continue
                if result and not self._contact_exists(result, contacts):
                    contacts.append(result)
                    if len(contacts) >= limit:
                        break

        # Phase 3: Hunter domain_search fallback — bulk contact discovery
        if len(contacts) < limit and company.domain and self.hunter_tool:
            try:
                hunter_results = await self.hunter_tool.domain_search(
                    company.domain, limit=limit - len(contacts)
                )
                for r in hunter_results:
                    name = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip()
                    if not name or name == " ":
                        continue
                    contact_id = hashlib.md5(
                        f"{company.id}_{name}".encode()
                    ).hexdigest()[:12]
                    contact = ContactData(
                        id=contact_id,
                        company_id=company.id,
                        company_name=company.company_name,
                        person_name=name,
                        role=r.get("position", ""),
                        email=r.get("email", ""),
                        email_confidence=r.get("confidence", 50),
                        email_source="hunter_domain",
                        verified=False,
                    )
                    if not self._contact_exists(contact, contacts):
                        contacts.append(contact)
                        if len(contacts) >= limit:
                            break
                if hunter_results:
                    logger.info(f"Hunter domain_search added {len(hunter_results)} contacts for {company.company_name}")
            except Exception as e:
                logger.debug(f"Hunter domain_search failed for {company.company_name}: {e}")

        # Phase 4: Hunter find_email for contacts missing emails.
        # Web search (Phases 1-2 fallback) finds names but no emails.
        # Hunter find_email resolves name+domain → verified email address.
        if self.hunter_tool and company.domain:
            no_email = [c for c in contacts if not c.email and c.person_name]
            if no_email:
                import asyncio as _aio
                email_results = await _aio.gather(
                    *[
                        self.hunter_tool.find_email(company.domain, c.person_name)
                        for c in no_email[:5]  # Cap at 5 to stay within Hunter rate limits
                    ],
                    return_exceptions=True,
                )
                resolved = 0
                for contact, result in zip(no_email[:5], email_results):
                    if isinstance(result, Exception):
                        logger.debug(f"Hunter find_email failed for {contact.person_name}: {result}")
                        continue
                    if result and result.email:
                        contact.email = result.email
                        contact.email_confidence = result.confidence or 70
                        contact.email_source = "hunter"
                        contact.verified = result.verified or False
                        resolved += 1
                if resolved:
                    logger.info(
                        f"Hunter find_email resolved {resolved}/{len(no_email)} "
                        f"emails for {company.company_name}"
                    )

        # Post-process: validate emails and clean up
        for c in contacts:
            if c.email and not self.is_valid_email(c.email):
                logger.debug(f"Invalid email format stripped: {c.email}")
                c.email = ""
                c.email_confidence = 0
                c.verified = False

        return contacts[:limit]
    
    async def _find_via_apollo(
        self,
        company: CompanyData,
        roles: List[str],
        limit: int
    ) -> List[ContactData]:
        """Find contacts using Apollo.io API. Also caches org data for enrichment."""
        contacts = []

        # search_people_at_company now returns {"people": [...], "company": {...}}
        result = await self.apollo_tool.search_people_at_company(
            domain=company.domain,
            roles=roles,
            limit=limit
        )
        people = result.get("people", []) if isinstance(result, dict) else result

        # Cache Apollo org data on the company for downstream enrichment
        apollo_org = result.get("company", {}) if isinstance(result, dict) else {}
        if apollo_org:
            if not company.description and apollo_org.get("description"):
                company.description = apollo_org["description"]
            if not company.industry and apollo_org.get("industry"):
                company.industry = apollo_org["industry"]

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
        """Find a contact via web search.

        Uses a broad leadership query (avoids LinkedIn keyword which DDG blocks),
        then LLM-extracts the specific role from the results.
        """
        country = getattr(self, '_country', 'India')
        # Single combined query — avoid "LinkedIn" keyword (DDG blocks it)
        query = f"{company.company_name} {role} {country} executive"
        try:
            data = await self.search_manager.web_search(query, max_results=5)
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
            f"- {r.get('title', '')}: {r.get('content', '')}"
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
    
    _EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format."""
        if not email:
            return False
        return bool(ContactFinder._EMAIL_REGEX.match(email.strip()))

    def _contact_exists(self, new_contact: ContactData, existing: List[ContactData]) -> bool:
        """Check if contact already exists by name OR email."""
        new_name = new_contact.person_name.lower().strip()
        new_email = (new_contact.email or "").lower().strip()
        for contact in existing:
            if contact.person_name.lower().strip() == new_name:
                return True
            # Also deduplicate by email (different name variants, same person)
            if new_email and (contact.email or "").lower().strip() == new_email:
                return True
        return False


# Backward compatibility alias
ContactAgent = ContactFinder
