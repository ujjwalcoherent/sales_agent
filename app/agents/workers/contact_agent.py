"""
Contact Finder Agent.
Finds decision-makers at companies based on trend type.
"""

import asyncio
import logging
import hashlib
import re
from typing import List

from ...schemas import CompanyData, ContactData, AgentState
from app.tools.llm.llm_service import LLMService
from app.tools.crm.apollo_tool import ApolloTool
from ...config import get_settings, TREND_ROLE_MAPPING

logger = logging.getLogger(__name__)

_RE_DIGITS = re.compile(r"\d+")

_C_SUITE = frozenset({
    "ceo", "cfo", "cto", "coo", "cmo", "cio", "ciso", "founder", "co-founder",
    "cofounder", "managing director", "president", "owner",
})
# For large companies (>900 employees), C-suite doesn't read cold emails.
# Target VP/Director/Head instead.
_LARGE_COMPANY_THRESHOLD = 900


def _filter_roles_by_size(roles: list[str], employee_count: int | None) -> list[str]:
    """Re-order roles based on company size.

    For companies >900 employees: demote pure C-suite titles to end of list
    (keep if no alternatives exist). VP/Director/Head are primary targets.
    For small companies (<100 employees): CEO/Founder IS the right person.
    """
    if not employee_count or employee_count <= _LARGE_COMPANY_THRESHOLD:
        return roles  # Small/mid: C-suite is correct

    # Large company: separate C-suite from VP/Director level
    c_suite = [r for r in roles if any(kw in r.lower() for kw in _C_SUITE)]
    non_c_suite = [r for r in roles if r not in c_suite]

    if non_c_suite:
        # VP/Director roles first, then C-suite as last resort
        return non_c_suite + c_suite
    return roles  # No alternatives — keep C-suite


def match_roles_to_trend(
    trend_type: str,
    pain_point: str = "",
    who_needs_help: str = "",
    trend_title: str = "",
    employee_count: int | None = None,
    bandit=None,
) -> list[str]:
    """Determine which roles to target based on the trend.

    Uses trend_type as primary signal, then scans trend_title, pain_point,
    and who_needs_help for role keywords. Returns ordered list of target roles.

    bandit: pre-loaded ContactBandit instance; if None, loads from disk.
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

    filtered = _filter_roles_by_size(unique, employee_count)

    # Re-rank using contact bandit (Thompson Sampling on historical email engagement).
    # Bandit learns which (role × event_type × company_size) combos get replies.
    # REF: Chapelle & Li (2011) Thompson Sampling for CTR — arXiv:1111.1797
    try:
        if bandit is None:
            from app.learning.contact_bandit import ContactBandit
            bandit = ContactBandit.load()
        if bandit.total_updates > 0:
            size = "enterprise" if (employee_count or 0) > 900 else (
                "smb" if (employee_count or 0) < 100 else "mid_market"
            )
            ranked = bandit.rank_roles(
                roles=filtered,
                event_type=trend_type or "general",
                company_size=size,
            )
            filtered = [role for role, _ in ranked]
    except Exception:
        pass  # Graceful fallback to static ordering

    return filtered[:8]  # Cap at 8 roles


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
        hunter_seniority: str = "",
        phone_number: str = "",
    ) -> int:
        """Compute reach score (0-100). Higher = more likely to respond to cold outreach.

        Weights (sum to 100):
          Email (0-40)  — primary channel. verified=40, else confidence×0.4 capped at 32.
          Phone (0-10)  — direct-dial from Hunter.
          LinkedIn(0-10)— secondary channel / identity signal.
          Tier  (0-20)  — decision_maker=20, influencer=15, gatekeeper=5.
          Fit   (0-20)  — role_relevance (0.0-1.0) × 20 from trend target_roles match.

        Caps:
          Gatekeeper: max 40 — filters mail, doesn't act on it.
          No email + no phone: max 35 — no known way to reach them.
        """
        # Hunter's ML seniority cross-validates our keyword-based tier
        if hunter_seniority:
            h = hunter_seniority.lower()
            if h == "executive" and seniority_tier != "decision_maker":
                seniority_tier = "decision_maker"
            elif h == "junior" and seniority_tier == "decision_maker":
                seniority_tier = "influencer"

        score = 0.0

        # Email deliverability (0-40)
        if verified:
            score += 40
        elif email:
            score += min(email_confidence * 0.4, 32)

        # Phone (0-10)
        if phone_number:
            score += 10

        # LinkedIn (0-10)
        if linkedin_url:
            score += 10

        # Seniority tier (0-20)
        score += {"decision_maker": 20, "influencer": 15, "gatekeeper": 5}.get(seniority_tier, 10)

        # Role relevance (0-20)
        score += role_relevance * 20

        if seniority_tier == "gatekeeper":
            score = min(score, 40)
        if not email and not phone_number:
            score = min(score, 35)

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
        semaphore = asyncio.Semaphore(4)

        # Load ContactBandit once — avoid N redundant file reads (one per company)
        _contact_bandit = None
        try:
            from app.learning.contact_bandit import ContactBandit
            _contact_bandit = ContactBandit.load()
        except Exception:
            pass  # Graceful degradation: rank_roles won't be called

        async def _find_one(company):
            async with semaphore:
                try:
                    # Priority: company.target_roles (LLM-inferred) > trend match > defaults
                    trend_title = ""
                    who_needs_help = ""
                    pain_point = ""
                    if getattr(company, "target_roles", None):
                        target_roles = list(company.target_roles)
                        impact = impact_map.get(company.trend_id)
                        trend = trend_map.get(company.trend_id)
                        trend_title = getattr(trend, "trend_title", "") if trend else ""
                        who_needs_help = getattr(impact, "who_needs_help", "") if impact else ""
                    else:
                        impact = impact_map.get(company.trend_id)
                        trend = trend_map.get(company.trend_id)
                        trend_type = getattr(trend, "trend_type", "") if trend else ""
                        trend_title = getattr(trend, "trend_title", "") if trend else ""
                        pain_point = " ".join(getattr(impact, "midsize_pain_points", []) or []) if impact else ""
                        who_needs_help = getattr(impact, "who_needs_help", "") if impact else ""
                        emp_count = getattr(company, "employee_count", None)
                        if isinstance(emp_count, str):
                            # Parse "500-1000" → take lower bound
                            m = _RE_DIGITS.search(emp_count)
                            emp_count = int(m.group()) if m else None
                        # Pass pre-loaded bandit — avoids N disk reads (one per company)
                        matched_roles = match_roles_to_trend(
                            trend_type, pain_point, who_needs_help, trend_title=trend_title,
                            employee_count=emp_count, bandit=_contact_bandit,
                        )
                        if matched_roles:
                            target_roles = matched_roles
                        elif impact and impact.target_roles:
                            target_roles = impact.target_roles
                        else:
                            target_roles = list(TREND_ROLE_MAPPING.get("default", []))
                    contacts = await self._find_contacts_for_company(
                        company, target_roles, max_contacts,
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
        limit: int,
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

        # Phase 1: Decision makers (30% of budget, at least 2)
        dm_limit = min(max(2, limit * 3 // 10), limit)
        if company.domain:
            try:
                apollo_dms = await self._find_via_apollo(company, dm_roles, dm_limit)
                contacts.extend(apollo_dms)
            except Exception as e:
                logger.debug(f"Apollo DM search failed for {company.company_name}: {e}")

        # Phase 2: Influencers + evaluators (remaining quota)
        inf_limit = limit - len(contacts)
        if inf_limit > 0 and company.domain:
            try:
                inf_contacts = await self._find_via_apollo(company, other_roles, inf_limit)
                for c in inf_contacts:
                    if not self._contact_exists(c, contacts):
                        contacts.append(c)
            except Exception as e:
                logger.debug(f"Apollo influencer search failed for {company.company_name}: {e}")

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
                        verified=r.get("verified", False),
                        # Hunter ML-inferred seniority — cross-validates keyword-based tier
                        hunter_seniority=r.get("seniority", ""),
                        # LinkedIn + phone from Hunter — no extra credit, avoids separate search
                        linkedin_url=r.get("linkedin", ""),
                        phone_number=r.get("phone_number", ""),
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
                email_results = await asyncio.gather(
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
                        # Enrich contact with free fields from email-finder response
                        if result.linkedin_url and not contact.linkedin_url:
                            contact.linkedin_url = result.linkedin_url
                        if result.phone_number and not contact.phone_number:
                            contact.phone_number = result.phone_number
                        if result.position and not contact.role:
                            contact.role = result.position
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
