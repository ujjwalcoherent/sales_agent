"""
Email Finder and Outreach Writer Agent - Coherent Market Insights Edition.
Finds verified emails and generates personalized consulting pitch emails.
"""

import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

from ...schemas import ContactData, TrendData, CompanyData, OutreachEmail, AgentState
from ...tools.apollo_tool import ApolloTool
from ...tools.hunter_tool import HunterTool
from ...tools.llm_service import LLMService
from ...tools.domain_utils import extract_clean_domain
from ...config import get_settings, CMI_SERVICES

logger = logging.getLogger(__name__)


class EmailGenerator:
    """
    Finds verified emails and generates personalized outreach.

    1. Finding verified emails using Apollo (primary) and Hunter (fallback)
    2. Generating personalized consulting pitch emails via LLM

    NOTE: This is a deterministic pipeline stage, not an autonomous agent.
    Renamed from EmailAgent for honest naming.
    """
    
    def __init__(self, mock_mode: bool = False, deps=None):
        """Initialize email agent."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        if deps:
            self.apollo_tool = deps.apollo_tool
            self.hunter_tool = deps.hunter_tool
            self.llm_service = deps.llm_service
        else:
            self.apollo_tool = ApolloTool(mock_mode=self.mock_mode)
            self.hunter_tool = HunterTool(mock_mode=self.mock_mode)
            self.llm_service = LLMService(mock_mode=self.mock_mode)
    
    async def process_emails(self, state: AgentState) -> AgentState:
        """
        Find emails and generate outreach for all contacts.
        
        Args:
            state: Current agent state with contacts
            
        Returns:
            Updated state with emails and outreach
        """
        logger.info("📧 Starting email finding and outreach generation...")
        
        if not state.contacts:
            logger.warning("No contacts to process")
            return state
        
        # Build maps for quick lookup
        trend_map = {t.id: t for t in state.trends}
        company_map = {c.id: c for c in state.companies}
        impact_map = {i.trend_id: i for i in state.impacts}
        
        # Parallel email lookup + outreach generation (semaphore limits API calls)
        import asyncio
        semaphore = asyncio.Semaphore(4)
        email_failures = []

        async def _process_one(contact) -> OutreachEmail | None:
            async with semaphore:
                try:
                    company = company_map.get(contact.company_id)
                    if not company:
                        return None

                    trend = trend_map.get(company.trend_id)
                    impact = impact_map.get(company.trend_id)

                    # Find email if not already present
                    if not contact.email and company.domain:
                        email_result = await self._find_email(
                            domain=company.domain,
                            full_name=contact.person_name,
                        )
                        if email_result.get("email"):
                            contact.email = email_result["email"]
                            contact.email_confidence = email_result.get("confidence", 0)
                            contact.email_source = email_result.get("source", "")
                            contact.verified = email_result.get("verified", False)

                    # Generate outreach if we have an email
                    if contact.email:
                        outreach = await self._generate_outreach(
                            contact=contact,
                            company=company,
                            trend=trend,
                            impact=impact,
                        )
                        conf = contact.email_confidence
                        threshold = self.settings.email_confidence_threshold
                        logger.log(
                            logging.INFO if conf >= threshold else logging.WARNING,
                            f"{'Generated' if conf >= threshold else 'Low-confidence'} "
                            f"outreach for: {contact.person_name}",
                        )
                        return outreach
                    return None
                except Exception as e:
                    email_failures.append(f"{contact.person_name}: {str(e)[:100]}")
                    logger.warning(f"Failed to process email for {contact.person_name}: {e}")
                    return None

        results = await asyncio.gather(*[_process_one(c) for c in state.contacts])
        outreach_emails = [r for r in results if r is not None]
        emails_found = sum(1 for c in state.contacts if c.email)

        if email_failures:
            state.errors.extend([f"email_gen: {e}" for e in email_failures])
            logger.warning(f"Email gen: {len(email_failures)}/{len(state.contacts)} contacts failed")

        state.outreach_emails = outreach_emails
        state.current_step = "emails_generated"
        logger.info(f"Found {emails_found} emails, generated {len(outreach_emails)} outreach emails")
        
        return state
    
    async def _find_email(self, domain: str, full_name: str) -> Dict:
        """
        Find email using cascading strategy: Apollo → Hunter → Pattern
        
        Args:
            domain: Company domain
            full_name: Person's full name
            
        Returns:
            Dict with email, confidence, source, verified
        """
        # Clean domain
        clean_domain = extract_clean_domain(domain)
        if not clean_domain:
            return {}
        
        # Method 1: Try Apollo (primary)
        logger.debug(f"Trying Apollo for {full_name}@{clean_domain}")
        apollo_result = await self.apollo_tool.find_email(
            domain=clean_domain,
            full_name=full_name
        )
        
        if apollo_result.email and apollo_result.confidence >= 60:
            return {
                "email": apollo_result.email,
                "confidence": apollo_result.confidence,
                "source": "apollo",
                "verified": apollo_result.verified
            }
        
        # Method 2: Try Hunter (fallback)
        logger.debug(f"Trying Hunter for {full_name}@{clean_domain}")
        hunter_result = await self.hunter_tool.find_email(
            domain=clean_domain,
            full_name=full_name
        )
        
        if hunter_result.email and hunter_result.confidence >= 50:
            return {
                "email": hunter_result.email,
                "confidence": hunter_result.confidence,
                "source": "hunter",
                "verified": hunter_result.verified
            }
        
        # Method 3: Generate pattern-based email
        logger.debug(f"Generating pattern email for {full_name}@{clean_domain}")
        name_parts = full_name.strip().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""
        
        pattern_email = await self.hunter_tool.generate_email_pattern(
            domain=clean_domain,
            first_name=first_name,
            last_name=last_name
        )
        
        if pattern_email:
            return {
                "email": pattern_email,
                "confidence": 40,  # Low confidence for pattern
                "source": "pattern",
                "verified": False
            }
        
        return {}
    
    # ── Role-specific personalization frames ──
    # Each tier sees a fundamentally different value proposition, not just tone.
    _ROLE_FRAMES = {
        "decision_maker": {
            "lens": "revenue impact, competitive positioning, and strategic risk",
            "hook": "how this trend directly affects your bottom line and market position",
            "cta": "a brief 15-minute strategic discussion",
            "tone": (
                "TONE: Executive-level. Under 100 words. Lead with business impact "
                "(revenue, margin, market share). No jargon, no fluff. One clear ask. "
                "Every sentence must earn its place. Address them as a peer executive."
            ),
        },
        "influencer": {
            "lens": "operational challenges, technology implications, and team readiness",
            "hook": "what this means for your team's day-to-day operations and roadmap",
            "cta": "a 15-minute call to share what we're seeing across similar teams",
            "tone": (
                "TONE: Consultative. Under 150 words. Show you understand their specific "
                "technical/operational challenge. Position as a helpful peer who's seen this "
                "pattern before. Reference how the trend impacts their function specifically."
            ),
        },
        "gatekeeper": {
            "lens": "organizational alignment and how this trend affects multiple departments",
            "hook": "a development that may be relevant to your leadership team",
            "cta": "a brief introduction — happy to share a one-page summary your team can review",
            "tone": (
                "TONE: Professional. Under 120 words. Formal and respectful. Be clear about "
                "who you are and why this is relevant. Make it easy for them to forward "
                "internally. Offer something tangible (a summary, a briefing)."
            ),
        },
    }

    async def _generate_outreach(
        self,
        contact: ContactData,
        company: CompanyData,
        trend: Optional[TrendData],
        impact: Optional[Dict]
    ) -> OutreachEmail:
        """Generate a role-personalized outreach email.

        Personalization axes:
          1. Seniority tier → different value proposition and CTA
          2. Role title → LLM frames the trend through their function
          3. Trend + impact → company-specific context
          4. Pitch angle → matched services
        """
        trend_title = trend.trend_title if trend else "current market developments"
        trend_summary = trend.summary if trend else "significant market changes"

        opportunities = []
        relevant_services = []
        pitch_angle = ""
        if impact:
            opportunities = getattr(impact, "business_opportunities", [])
            relevant_services = getattr(impact, "relevant_services", [])
            pitch_angle = getattr(impact, "pitch_angle", "")

        # Get relevant CMI service details
        service_offerings = []
        for svc_key, svc_data in CMI_SERVICES.items():
            if any(s.lower() in svc_key.lower() or svc_key.lower() in s.lower()
                   for s in relevant_services):
                service_offerings.extend(svc_data["offerings"][:2])
        if not service_offerings:
            service_offerings = ["Market sizing and competitive analysis", "Strategic advisory services"]

        first_name = contact.person_name.split()[0] if contact.person_name else "there"

        # Determine seniority tier and role-specific frame
        from app.agents.workers.contact_agent import ContactFinder
        tier = ContactFinder.classify_tier(contact.role)
        tone = ContactFinder.get_outreach_tone(tier)
        frame = self._ROLE_FRAMES.get(tier, self._ROLE_FRAMES["influencer"])

        prompt = f"""Write a personalized outreach email from Coherent Market Insights.

SENDER (Coherent Market Insights):
- Global market research and consulting firm
- Expertise: market intelligence, competitive analysis, strategic advisory
- Services: {', '.join(service_offerings[:3])}

RECIPIENT PROFILE:
- Name: {contact.person_name}
- Role: {contact.role}
- Seniority: {tier.replace('_', ' ').title()}
- Company: {company.company_name}
- Industry: {company.industry}

MARKET CONTEXT (why we're reaching out):
- Trend: {trend_title}
- Summary: {trend_summary}
- Impact on {company.company_name}: {company.reason_relevant}
{f'- Pitch angle: {pitch_angle}' if pitch_angle else ''}

OPPORTUNITIES:
{chr(10).join(['- ' + opp for opp in opportunities[:3]]) if opportunities else '- Market impact assessment and strategic advisory'}

PERSONALIZATION INSTRUCTIONS:
Frame this email through the lens of {frame['lens']}.
The hook should be about {frame['hook']}.
The call-to-action should be: {frame['cta']}.

{frame['tone']}

RULES:
- Address as "{contact.person_name}" (use "Dear" for executives, first name for others)
- Reference their SPECIFIC ROLE — explain why this trend matters to a {contact.role}
- Do NOT use generic phrases like "I came across your profile" or "Hope this finds you well"
- Do NOT mention "pain point", "opportunity", or any sales framework language
- Sign off as: "Best regards,\\nCoherent Market Insights Team"

Respond as JSON:
- subject: Email subject line (max 60 chars, reference the trend, not the person's name)
- body: Email body text (use real newlines, not \\n)"""

        system_prompt = (
            "You are a senior business development consultant at Coherent Market Insights. "
            "You write concise, insight-driven emails that demonstrate genuine understanding "
            "of the recipient's role-specific challenges. Your emails read like a knowledgeable "
            "peer reaching out — never like a cold sales pitch. Always respond with valid JSON only."
        )

        try:
            from app.schemas.llm_outputs import OutreachDraftLLM

            # Try structured output first, fall back to generate_json
            try:
                result = await self.llm_service.run_structured(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    output_type=OutreachDraftLLM,
                )
                subject = result.subject or f"Regarding {trend_title}"
                body = result.body.replace("\\n", "\n") if result.body else ""
            except Exception as struct_err:
                logger.warning(f"Structured email gen failed: {struct_err}, trying generate_json...")
                raw = await self.llm_service.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                if isinstance(raw, dict) and "error" not in raw:
                    subject = raw.get("subject", f"Regarding {trend_title}")
                    body = raw.get("body", "").replace("\\n", "\n")
                else:
                    raise RuntimeError(f"generate_json failed: {raw}")

        except Exception as e:
            logger.warning(f"LLM email generation failed: {e}")
            # Role-aware fallback template
            greeting = f"Dear {contact.person_name}" if tier == "decision_maker" else f"Hi {first_name}"
            cta = frame["cta"]
            subject = f"{trend_title[:45]} — implications for {company.company_name}"[:60]
            body = f"""{greeting},

Recent developments around {trend_title} have significant implications for {company.company_name}, particularly from the perspective of {frame['lens']}.

At Coherent Market Insights, we've been tracking this closely and have insights that may be relevant to your work as {contact.role}.

Would you be open to {cta}?

Best regards,
Coherent Market Insights Team"""

        email_id = hashlib.md5(
            f"{contact.id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        return OutreachEmail(
            id=email_id,
            contact_id=contact.id,
            trend_title=trend_title,
            company_name=company.company_name,
            person_name=contact.person_name,
            role=contact.role,
            email=contact.email,
            subject=subject,
            body=body,
            email_confidence=contact.email_confidence,
            generated_at=datetime.utcnow()
        )


# Backward compatibility alias
EmailAgent = EmailGenerator


async def run_email_agent(state: AgentState, deps=None) -> AgentState:
    """Wrapper function for LangGraph."""
    generator = EmailGenerator(deps=deps) if deps else EmailGenerator()
    return await generator.process_emails(state)
