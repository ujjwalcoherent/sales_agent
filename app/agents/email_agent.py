"""
Email Finder and Outreach Writer Agent - Coherent Market Insights Edition.
Finds verified emails and generates personalized consulting pitch emails.
"""

import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

from ..schemas import ContactData, TrendData, CompanyData, OutreachEmail, AgentState
from ..tools.apollo_tool import ApolloTool
from ..tools.hunter_tool import HunterTool
from ..tools.llm_tool import LLMTool
from ..tools.domain_utils import extract_clean_domain
from ..config import get_settings, CMI_SERVICES

logger = logging.getLogger(__name__)


class EmailAgent:
    """
    Agent responsible for:
    1. Finding verified emails using Apollo (primary) and Hunter (fallback)
    2. Generating personalized outreach emails
    """
    
    def __init__(self, mock_mode: bool = False):
        """Initialize email agent."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.apollo_tool = ApolloTool(mock_mode=self.mock_mode)
        self.hunter_tool = HunterTool(mock_mode=self.mock_mode)
        self.llm_tool = LLMTool(mock_mode=self.mock_mode)
    
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
        
        outreach_emails = []
        emails_found = 0
        
        for contact in state.contacts:
            try:
                # Get associated company and trend
                company = company_map.get(contact.company_id)
                if not company:
                    continue
                
                trend = trend_map.get(company.trend_id)
                impact = impact_map.get(company.trend_id)
                
                # Find email if not already present
                if not contact.email and company.domain:
                    email_result = await self._find_email(
                        domain=company.domain,
                        full_name=contact.person_name
                    )
                    
                    if email_result.get("email"):
                        contact.email = email_result["email"]
                        contact.email_confidence = email_result.get("confidence", 0)
                        contact.email_source = email_result.get("source", "")
                        contact.verified = email_result.get("verified", False)
                        emails_found += 1
                
                # Generate outreach if we have an email with sufficient confidence
                if contact.email and contact.email_confidence >= self.settings.email_confidence_threshold:
                    outreach = await self._generate_outreach(
                        contact=contact,
                        company=company,
                        trend=trend,
                        impact=impact
                    )
                    outreach_emails.append(outreach)
                    logger.info(f"✅ Generated outreach for: {contact.person_name}")
                    
                elif contact.email:
                    # Low confidence email - still generate outreach but note it
                    outreach = await self._generate_outreach(
                        contact=contact,
                        company=company,
                        trend=trend,
                        impact=impact
                    )
                    outreach_emails.append(outreach)
                    logger.info(f"⚠️ Low confidence email for: {contact.person_name}")
                
            except Exception as e:
                logger.warning(f"Failed to process email for {contact.person_name}: {e}")
        
        state.outreach_emails = outreach_emails
        state.current_step = "emails_generated"
        logger.info(f"🎯 Found {emails_found} emails, generated {len(outreach_emails)} outreach emails")
        
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
    
    async def _generate_outreach(
        self,
        contact: ContactData,
        company: CompanyData,
        trend: Optional[TrendData],
        impact: Optional[Dict]
    ) -> OutreachEmail:
        """
        Generate personalized CONSULTING PITCH email for Coherent Market Insights.
        
        Args:
            contact: Contact information
            company: Company data
            trend: Associated trend
            impact: Impact analysis
            
        Returns:
            OutreachEmail with subject and body
        """
        # Build context for email
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
        
        prompt = f"""Write a CONSULTING PITCH email from Coherent Market Insights to a potential client.

ABOUT COHERENT MARKET INSIGHTS:
- We are a global market research and consulting firm
- We help mid-size companies with market intelligence, competitive analysis, and strategic advisory
- Our expertise: Market research, procurement intelligence, industry analysis, technology research

RECIPIENT:
- Name: {contact.person_name}
- Role: {contact.role}
- Company: {company.company_name}
- Industry: {company.industry}
- Company Size: Mid-size (50-300 employees)

THE NEWS/TREND THAT TRIGGERED THIS OUTREACH:
- Headline: {trend_title}
- Summary: {trend_summary}
- Why this matters to {company.company_name}: {company.reason_relevant}

CONSULTING OPPORTUNITIES WE IDENTIFIED:
{chr(10).join(['- ' + opp for opp in opportunities[:3]]) if opportunities else '- Market impact assessment and strategic advisory'}

OUR RELEVANT SERVICES:
{chr(10).join(['- ' + svc for svc in service_offerings[:3]])}

EMAIL REQUIREMENTS:
1. Subject line: Reference the news/trend + value proposition (max 60 chars)
2. Opening: "I noticed [the news/trend] and thought of {company.company_name}..."
3. Show understanding of their challenge: What problem does this news create for them?
4. Position CMI as the solution: How can our research/consulting help them navigate this?
5. Mention 1-2 specific services we can offer
6. CTA: Offer a 15-minute discovery call
7. Keep it under 150 words
8. Tone: Helpful consultant, not salesy
9. Sign off as: "Best regards, [Coherent Market Insights Team]"

Respond as JSON with:
- subject: Email subject line (max 60 chars)
- body: Email body (use actual newlines, not \\n)"""

        system_prompt = """You are a business development consultant at Coherent Market Insights.
You write helpful, insight-driven emails that show you understand the prospect's challenges.
Your emails should feel like a consultant reaching out to help, not a salesperson pitching.
Always respond with valid JSON only."""

        try:
            result = await self.llm_tool.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            subject = result.get("subject", f"Regarding {trend_title}")
            body = result.get("body", "")
            
            # Clean up body - ensure proper newlines
            body = body.replace("\\n", "\n")
            
        except Exception as e:
            logger.warning(f"LLM email generation failed: {e}")
            # Fallback template
            subject = f"{first_name}, thoughts on {trend_title[:30]}..."
            body = f"""Hi {first_name},

I noticed {company.company_name}'s work in {company.industry} and wanted to reach out regarding {trend_title}.

Given the current market dynamics, I believe there are some strategic opportunities worth exploring together.

Would you be open to a brief 15-minute call this week to discuss?

Best regards"""

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


async def run_email_agent(state: AgentState) -> AgentState:
    """Wrapper function for LangGraph."""
    agent = EmailAgent()
    return await agent.process_emails(state)
