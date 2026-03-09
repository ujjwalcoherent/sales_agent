"""
Email Finder and Outreach Writer Agent - Coherent Market Insights Edition.
Finds verified emails and generates personalized consulting pitch emails.
"""

import asyncio
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

from ...schemas import ContactData, TrendData, CompanyData, ImpactAnalysis, OutreachEmail, AgentState
from app.tools.crm.apollo_tool import ApolloTool
from app.tools.crm.hunter_tool import HunterTool
from app.tools.llm.llm_service import LLMService
from app.tools.domain_utils import extract_clean_domain
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
    
    def __init__(self, mock_mode: bool = False, deps=None, campaign_mode: bool = False):
        """Initialize email agent.

        Args:
            campaign_mode: When True, uses faster person intel (8s timeout, no deep)
                          and higher concurrency for campaign throughput.
        """
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.campaign_mode = campaign_mode
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

                    # Verify email via Hunter if found but not yet verified
                    if contact.email and not contact.verified and self.hunter_tool:
                        try:
                            verification = await self.hunter_tool.verify_email(contact.email)
                            if verification.get("status") == "valid":
                                contact.verified = True
                                contact.email_confidence = max(contact.email_confidence, 85)
                                logger.debug(f"Hunter verified email for {contact.person_name}")
                            elif verification.get("result") == "deliverable":
                                contact.email_confidence = max(contact.email_confidence, 75)
                        except Exception as e:
                            logger.debug(f"Hunter verify_email failed for {contact.person_name}: {e}")

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

    @staticmethod
    def _build_person_context_block(person_ctx) -> str:
        """Build the person-specific section for the LLM prompt.

        Returns an empty string if no meaningful intel was gathered,
        so the prompt degrades gracefully to role-only personalization.

        Includes deep intel (blog posts, speaking topics, GitHub, career
        history, content themes) when Stage 2 enrichment has completed.
        """
        if not person_ctx.has_context:
            return ""

        lines = ["\nPERSON INTELLIGENCE (use this to personalize):"]
        if person_ctx.background_summary:
            lines.append(f"- Background: {person_ctx.background_summary}")
        if person_ctx.recent_focus:
            lines.append(f"- Recent focus: {person_ctx.recent_focus}")
        if person_ctx.notable_achievements:
            lines.append(f"- Notable: {', '.join(person_ctx.notable_achievements[:3])}")
        if person_ctx.linkedin_headline:
            lines.append(f"- LinkedIn: {person_ctx.linkedin_headline[:150]}")
        if person_ctx.news_mentions:
            lines.append(f"- Recent news: {'; '.join(person_ctx.news_mentions[:2])}")

        # Deep intel fields (from Stage 2 background enrichment)
        if getattr(person_ctx, "recent_posts", None):
            lines.append(f"\nRECENT CONTENT (blog/articles):")
            for post in person_ctx.recent_posts[:3]:
                lines.append(f"  - {post[:150]}")
        if getattr(person_ctx, "speaking_topics", None):
            lines.append(f"\nSPEAKING/CONFERENCES:")
            for topic in person_ctx.speaking_topics[:2]:
                lines.append(f"  - {topic[:150]}")
        if getattr(person_ctx, "github_profile", "") and person_ctx.github_profile:
            lines.append(f"- GitHub: {person_ctx.github_profile[:200]}")
        if getattr(person_ctx, "career_history", None):
            lines.append(f"- Career: {'; '.join(person_ctx.career_history[:3])}")
        if getattr(person_ctx, "content_themes", None):
            lines.append(f"- Recurring themes: {', '.join(person_ctx.content_themes[:4])}")

        if person_ctx.talking_points:
            lines.append("\nTALKING POINTS TO REFERENCE:")
            for tp in person_ctx.talking_points[:3]:
                lines.append(f"  - {tp}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    @staticmethod
    def _person_context_rules(person_ctx) -> str:
        """Return extra prompt rules when person intel is available."""
        rules = []
        if person_ctx.has_context:
            rules.append(
                "- Reference something SPECIFIC about them (their background, "
                "recent work, or achievement from the PERSON INTELLIGENCE section)"
            )
            rules.append(
                "- Do NOT just restate their job title -- show you know who they are"
            )
            # Deep intel-specific rules
            if getattr(person_ctx, "recent_posts", None):
                rules.append(
                    "- Reference a specific recent article or blog post they wrote"
                )
            if getattr(person_ctx, "speaking_topics", None):
                rules.append(
                    "- Mention a conference talk or podcast they appeared in"
                )
            if getattr(person_ctx, "content_themes", None):
                rules.append(
                    "- Align the pitch angle with their recurring professional themes"
                )
            return "\n".join(rules)
        return "- Show you understand their role and its typical challenges"

    @staticmethod
    def _format_evidence(trend: Optional["TrendData"]) -> str:
        """Format evidence snippets for email prompt (if available)."""
        if not trend:
            return ""
        snippets = getattr(trend, "evidence_snippets", [])
        if not snippets:
            return ""
        lines = ["EVIDENCE (cite these sources in the email):"]
        for s in snippets[:3]:
            lines.append(f"  - {s}")
        return "\n".join(lines)

    async def _generate_outreach(
        self,
        contact: ContactData,
        company: CompanyData,
        trend: Optional[TrendData],
        impact: Optional[ImpactAnalysis]
    ) -> OutreachEmail:
        """Generate a hyper-personalized outreach email.

        Personalization axes:
          1. Person intel → background, recent work, talking points (NEW)
          2. Seniority tier → different value proposition and CTA
          3. Role title → LLM frames the trend through their function
          4. Trend + impact → company-specific context
          5. Pitch angle → matched services
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

        # ── Gather person intelligence (never crashes) ────────
        from app.tools.person_intel import gather_person_context, PersonContext

        if self.campaign_mode:
            # Campaign mode: fast person intel (8s timeout, no deep scraping)
            try:
                person_ctx = await asyncio.wait_for(
                    gather_person_context(
                        person_name=contact.person_name,
                        company_name=company.company_name,
                        role=contact.role,
                        trend_context=pitch_angle or trend_title,
                        deep=False,
                    ),
                    timeout=10.0,
                )
            except (asyncio.TimeoutError, Exception):
                person_ctx = PersonContext(
                    person_name=contact.person_name,
                    company_name=company.company_name,
                    role=contact.role,
                )
        else:
            person_ctx = await gather_person_context(
                person_name=contact.person_name,
                company_name=company.company_name,
                role=contact.role,
                trend_context=pitch_angle or trend_title,
                deep=self.settings.email_personalization_depth == "deep",
                linkedin_url=getattr(contact, "linkedin_url", ""),
            )

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

        # ── Build person context block for the prompt ─────────
        person_block = self._build_person_context_block(person_ctx)

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
{person_block}
MARKET CONTEXT (why we're reaching out):
- Trend: {trend_title}
- Summary: {trend_summary}
- Impact on {company.company_name}: {company.reason_relevant}
{f'- Pitch angle: {pitch_angle}' if pitch_angle else ''}
{self._format_evidence(trend)}

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
{self._person_context_rules(person_ctx)}
- Do NOT use generic phrases like "I came across your profile" or "Hope this finds you well"
- Do NOT use placeholders like [Your Name], [Your Title], [Your Contact Information] — use real info
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

        # Strip any LLM placeholder artifacts
        import re as _re
        _org = "Coherent Market Insights"
        body = _re.sub(r'\[Your Name\]', "Coherent Market Insights Team", body, flags=_re.IGNORECASE)
        body = _re.sub(r'\[Your (?:Title|Position|Role)\]', "", body, flags=_re.IGNORECASE)
        body = _re.sub(r'\[Your (?:Contact Information|Phone|Email|Company)\]', "", body, flags=_re.IGNORECASE)
        body = _re.sub(r'\[(?:Company Name|Your Company|Our Company)\]', _org, body, flags=_re.IGNORECASE)
        body = _re.sub(r'\n{3,}', '\n\n', body).strip()

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
