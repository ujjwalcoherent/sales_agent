"""
Mock LLM responses for testing and development.

Provides deterministic responses based on prompt content hashing.
Designed to work with pydantic-ai's FunctionModel.
"""

import hashlib
import json
import logging
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart

logger = logging.getLogger(__name__)


def get_mock_response(prompt: str, json_mode: bool = False) -> str:
    """Return mock response for testing.

    Args:
        prompt: The user prompt text.
        json_mode: Whether JSON output is expected.

    Returns:
        Deterministic mock response string.
    """
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 5
    prompt_lower = prompt.lower()

    if json_mode or "json" in prompt_lower:
        # Email/outreach FIRST — email prompts contain "impact" as a substring
        # ("Impact on {company_name}:") which would falsely match the impact branch.
        # Detect email by the leading "Write a personalized outreach email" pattern.
        if ("write" in prompt_lower and ("outreach email" in prompt_lower or "personalized" in prompt_lower)):
            return json.dumps(_MOCK_EMAILS[prompt_hash % 3])
        # Cross-trend synthesis (compound impacts)
        if "compound" in prompt_lower and "simultaneously" in prompt_lower:
            return json.dumps(_MOCK_CROSS_TREND_SYNTHESIS)
        # Impact/council prompts — check BEFORE companies because impact prompts
        # naturally contain "compan" (first_order_companies) and "find" (find cheaper)
        # which would falsely match the company extraction branch.
        if "direct_impact" in prompt_lower or "first_order_companies" in prompt_lower or \
                ("impact" in prompt_lower and "consultant" in prompt_lower):
            sector = _detect_sector(prompt_lower)
            idx = {"fintech": 0, "telecom": 2, "space": 3}.get(sector, prompt_hash % 3)
            return json.dumps(_MOCK_IMPACTS[idx % len(_MOCK_IMPACTS)])
        # Company extraction — specific phrases from company extraction prompts
        if ("compan" in prompt_lower and ("find" in prompt_lower or "json array" in prompt_lower)) or "extract companies" in prompt_lower:
            return _get_mock_company_response(prompt_lower)
        if "synthesize" in prompt_lower or "cluster" in prompt_lower:
            sector = _detect_sector(prompt_lower)
            idx = {"space": 3}.get(sector, prompt_hash % len(_MOCK_SYNTH_TRENDS))
            return json.dumps(_MOCK_SYNTH_TRENDS[idx % len(_MOCK_SYNTH_TRENDS)])
        # Generic impact prompts (after company check to avoid false matches)
        elif "impact" in prompt_lower or "consultant" in prompt_lower:
            sector = _detect_sector(prompt_lower)
            idx = {"fintech": 0, "telecom": 2, "space": 3}.get(sector, prompt_hash % 3)
            return json.dumps(_MOCK_IMPACTS[idx])
        elif "trend" in prompt_lower or "news" in prompt_lower:
            return json.dumps(_MOCK_TRENDS[prompt_hash])
        elif "contact" in prompt_lower or "person" in prompt_lower:
            return json.dumps({
                "person_name": "Rahul Sharma",
                "role": "CTO",
                "linkedin_url": "https://linkedin.com/in/rahul-sharma"
            })
        elif "email" in prompt_lower or "outreach" in prompt_lower or "pitch" in prompt_lower:
            return json.dumps(_MOCK_EMAILS[prompt_hash % 3])

    return "Mock LLM response for testing purposes."


def _build_structured_mock(system_prompt: str, user_prompt: str = "") -> str:
    """Return a valid JSON string matching the expected pydantic-ai output_type
    based on keywords found in the agent's system prompt."""
    sp = system_prompt.lower()

    if "source intel" in sp:
        return json.dumps({
            "articles": [],
            "total_fetched": 5,
            "total_after_dedup": 5,
            "sources_used": ["mock_rss"],
            "web_searches_performed": 0,
            "event_distribution": {"regulation": 2, "funding": 3},
            "reasoning": "Mock: fetched 5 articles from RSS sources."
        })

    if "analysis agent" in sp or "cluster" in sp:
        return json.dumps({
            "num_clusters": 3,
            "noise_ratio": 0.15,
            "mean_coherence": 0.62,
            "num_trends_passed": 3,
            "num_trends_rejected": 1,
            "params_used": {"semantic_dedup_threshold": 0.88, "coherence_min": 0.48},
            "retries_performed": 0,
            "reasoning": "Mock: 3 clusters with good coherence."
        })

    # Council uses UnifiedImpactAnalysisLLM — check BEFORE "quality" and "market impact"
    # because council's system prompt contains "quality" (evidence quality) and "impact"
    if "mid-size" in sp and ("first-order" in sp or "consultant" in sp):
        sector = _detect_sector(user_prompt)
        return json.dumps(_MOCK_COUNCIL_RESPONSES[sector])

    if "market impact" in sp:
        # ImpactResult for the orchestrator-level impact node
        return json.dumps({
            "total_trends_analyzed": 3,
            "high_confidence_count": 2,
            "low_confidence_count": 1,
            "precedent_searches": 0,
            "reasoning": "Mock: analyzed 3 trends with high confidence."
        })

    if "quality agent" in sp or "quality" in sp:
        return json.dumps({
            "stage": "mock",
            "passed": True,
            "should_retry": False,
            "items_passed": 3,
            "items_filtered": 0,
            "quality_score": 0.75,
            "issues": [],
            "reasoning": "Mock: all quality checks passed."
        })

    if "lead gen" in sp or "lead" in sp:
        return json.dumps({
            "companies_found": 2,
            "contacts_found": 2,
            "emails_generated": 2,
            "outreach_generated": 2,
            "low_relevance_filtered": 0,
            "reasoning": "Mock: found 2 companies and generated outreach."
        })

    if "business development consultant" in sp:
        # email_agent: OutreachDraftLLM {subject, body}
        # Pick trend-specific email based on user prompt content
        up = user_prompt.lower()
        if any(kw in up for kw in ["kyc", "rbi", "compliance", "nbfc", "regtech", "lending"]):
            return json.dumps(_MOCK_EMAILS[0])  # RBI KYC
        elif any(kw in up for kw in ["genai", "tcs", "infosys", "wipro", "it services", "llm", "topaz"]):
            return json.dumps(_MOCK_EMAILS[1])  # IT GenAI
        elif any(kw in up for kw in ["5g", "iot", "telecom", "manufacturing", "private network"]):
            return json.dumps(_MOCK_EMAILS[2])  # 5G IoT
        elif any(kw in up for kw in ["rocket", "launch", "satellite", "spacex", "aerospace", "orbital", "reusable"]):
            return json.dumps(_MOCK_EMAILS[3])  # Space / carrier rocket
        else:
            # Hash-based selection for other prompts
            idx = int(hashlib.md5(user_prompt.encode()).hexdigest()[:8], 16) % 3
            return json.dumps(_MOCK_EMAILS[idx])

    # Generic fallback — return a plain text response
    return "Mock LLM response for testing purposes."


def get_mock_response_for_function_model(messages: list[Any], info: Any) -> ModelResponse:
    """Adapter for pydantic-ai FunctionModel.

    FunctionModel passes ModelMessage objects. We extract the user prompt
    text and delegate to the main mock function. Returns a ModelResponse
    (required by pydantic-ai >= 1.0).

    When the agent has a structured output_type (SourceIntelResult,
    AnalysisResult, ImpactResult, QualityVerdict, LeadGenResult), we detect
    which agent is calling via its system prompt and return a valid JSON
    payload so pydantic-ai can parse it without retrying.
    """
    # Extract user prompt and system prompt from messages
    prompt = ""
    system_prompt = ""
    all_text_parts = []
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if not hasattr(part, 'content') or not isinstance(part.content, str):
                    continue
                all_text_parts.append(part.content)
                part_type = type(part).__name__
                if "User" in part_type:
                    prompt = part.content
                elif "System" in part_type:
                    system_prompt = part.content
                elif not prompt:
                    prompt = part.content
    if not prompt and messages:
        prompt = str(messages[-1])
    # Combined text for keyword matching when individual extraction fails
    if not prompt:
        prompt = " ".join(all_text_parts)

    # If this is a known pydantic-ai agent (has a recognisable system prompt),
    # return structured JSON so validation passes on the first try.
    sp_lower = system_prompt.lower()
    is_known_agent = any(kw in sp_lower for kw in (
        "source intel", "analysis agent", "cluster",
        "market impact", "mid-size", "business development analyst",
        "quality agent", "quality",
        "lead gen", "lead",
        "business development consultant",  # email_agent
    ))

    if is_known_agent:
        text = _build_structured_mock(system_prompt, user_prompt=prompt)
        return ModelResponse(parts=[TextPart(content=text)])

    # Fallback: use the original prompt-based routing
    all_text = (prompt + " " + system_prompt).lower()
    json_mode = "json" in all_text
    text = get_mock_response(prompt, json_mode)
    return ModelResponse(parts=[TextPart(content=text)])


def _detect_sector(text: str) -> str:
    """Detect sector from prompt text for routing mock responses.

    Extracts the TREND portion from council prompts to avoid false matches
    on keywords in the CMI services catalog (e.g., "compliance" appears in
    industry_analysis service but isn't fintech-specific).
    """
    # If this is a council prompt, extract just the trend title + summary
    tl = text.lower()
    if "trend:" in tl:
        # Extract between "TREND:" and "SOURCE DATA:" or "OUR SERVICE CATALOG:"
        start = tl.index("trend:")
        end = len(tl)
        for marker in ["source data:", "our service catalog:", "key topics:"]:
            pos = tl.find(marker, start)
            if pos > start:
                end = min(end, pos)
        tl = tl[start:end]

    # Check IT FIRST — IT keywords are most specific (company names)
    if any(kw in tl for kw in ["tcs", "infosys", "wipro", "hcl", "genai", "it services",
                                "mphasis", "persistent", "happiest minds", "tech mahindra"]):
        return "it"
    if any(kw in tl for kw in ["kyc", "rbi", "nbfc", "fintech", "lending", "regtech",
                                "lendingkart", "capital float", "zestmoney", "easebuzz"]):
        return "fintech"
    # Space/aerospace — carrier rocket report-driven mock theme
    if any(kw in tl for kw in ["spacex", "rocket", "launch", "satellite", "starship",
                                "blue origin", "rocket lab", "relativity space", "isro",
                                "carrier rocket", "reusable", "leo", "constellation",
                                "orbital", "aerospace"]):
        return "space"
    # "manufacturing" alone is too broad (appears in IT article summaries).
    # Require 5g/iot/telecom-specific terms for telecom detection.
    if any(kw in tl for kw in ["5g", "iot", "telecom", "private network", "sensor",
                                "airtel", "jio", "endurance", "sundram", "edge computing"]):
        return "telecom"
    return "it"


def _get_mock_company_response(prompt_lower: str) -> str:
    """Return mock company data based on sector keywords in prompt."""
    sector_companies = {
        "oil_energy": [
            {"company_name": "Petrosol Energy Services", "company_size": "mid", "industry": "Oil Equipment", "website": "https://petrosol.in", "description": "Oil field equipment supplier, 180 employees", "intent_signal": "Facing margin pressure", "reason_relevant": "Needs procurement intelligence"},
            {"company_name": "Gujarat Oilfield Services", "company_size": "mid", "industry": "Oil Services", "website": "https://gosindia.com", "description": "Oilfield services, 220 employees", "intent_signal": "Restructuring operations", "reason_relevant": "Needs strategic consulting"},
        ],
        "fintech": [
            {"company_name": "Lendingkart", "company_size": "mid", "industry": "Fintech", "website": "https://lendingkart.com", "description": "SME lending platform, 150 employees", "intent_signal": "Compliance overhaul needed", "reason_relevant": "Affected by RBI KYC norms"},
            {"company_name": "Capital Float", "company_size": "mid", "industry": "Fintech", "website": "https://capitalfloat.com", "description": "Digital lending, 200 employees", "intent_signal": "Seeking regulatory guidance", "reason_relevant": "Must comply with regulations"},
        ],
        "logistics": [
            {"company_name": "Country Delight", "company_size": "mid", "industry": "Quick Commerce", "website": "https://countrydelight.in", "description": "Farm-fresh delivery, 250 employees", "intent_signal": "Expanding to new cities", "reason_relevant": "Quick delivery competitor"},
            {"company_name": "Delhivery Express", "company_size": "mid", "industry": "Logistics", "website": "https://delhivery.com", "description": "Last-mile logistics, 280 employees", "intent_signal": "Supply chain optimization", "reason_relevant": "Logistics opportunity"},
        ],
        "electronics": [
            {"company_name": "VVDN Technologies", "company_size": "mid", "industry": "Electronics Manufacturing", "website": "https://vvdntech.com", "description": "Electronics design, 250 employees", "intent_signal": "Entering semiconductor supply chain", "reason_relevant": "Semiconductor ecosystem"},
            {"company_name": "Syrma SGS Technology", "company_size": "mid", "industry": "Electronics", "website": "https://syrmasgs.com", "description": "EMS provider, 200 employees", "intent_signal": "Diversifying components", "reason_relevant": "Sourcing opportunity"},
        ],
        "hr": [
            {"company_name": "Xpheno", "company_size": "mid", "industry": "HR Tech", "website": "https://xpheno.com", "description": "Specialist staffing, 120 employees", "intent_signal": "Growing talent acquisition", "reason_relevant": "Talent opportunity"},
            {"company_name": "PeopleStrong", "company_size": "mid", "industry": "HR Tech", "website": "https://peoplestrong.com", "description": "HR technology, 250 employees", "intent_signal": "Workforce optimization", "reason_relevant": "HR tech needs"},
        ],
        "regtech": [
            {"company_name": "Signzy", "company_size": "mid", "industry": "RegTech", "website": "https://signzy.com", "description": "Digital KYC solutions, 150 employees", "intent_signal": "KYC demand surge", "reason_relevant": "Compliance solutions"},
            {"company_name": "IDfy", "company_size": "mid", "industry": "RegTech", "website": "https://idfy.com", "description": "Identity verification, 180 employees", "intent_signal": "Expanding verification services", "reason_relevant": "Compliance provider"},
        ],
        "space": [
            {"company_name": "Skyroot Aerospace", "company_size": "mid", "industry": "Space Launch", "website": "https://skyroot.in", "description": "India's first private orbital launch company, 200 employees", "intent_signal": "Expanding launch cadence", "reason_relevant": "Needs supply chain intelligence for Vikram-2 reusable rocket development"},
            {"company_name": "Agnikul Cosmos", "company_size": "mid", "industry": "Space Launch", "website": "https://agnikul.in", "description": "3D-printed rocket engine maker, 150 employees", "intent_signal": "Scaling manufacturing", "reason_relevant": "Additive manufacturing for rocket components — sourcing and qualification data needed"},
        ],
        "default": [
            {"company_name": "Moglix", "company_size": "mid", "industry": "B2B Commerce", "website": "https://moglix.com", "description": "Industrial B2B marketplace, 280 employees", "intent_signal": "Seeking procurement intelligence", "reason_relevant": "Supply chain needs"},
            {"company_name": "OfBusiness", "company_size": "mid", "industry": "B2B Commerce", "website": "https://ofbusiness.com", "description": "B2B raw materials, 250 employees", "intent_signal": "Expanding supplier network", "reason_relevant": "Procurement opportunity"},
        ],
    }

    if any(kw in prompt_lower for kw in ["rocket", "launch", "satellite", "spacex", "aerospace", "orbital", "constellation"]):
        companies = sector_companies.get("space", sector_companies["default"])
    elif any(kw in prompt_lower for kw in ["oil", "energy", "fuel", "petro"]):
        companies = sector_companies["oil_energy"]
    elif any(kw in prompt_lower for kw in ["fintech", "lending", "nbfc"]):
        companies = sector_companies["fintech"]
    elif any(kw in prompt_lower for kw in ["logistics", "delivery", "commerce"]):
        companies = sector_companies["logistics"]
    elif any(kw in prompt_lower for kw in ["semiconductor", "electronics", "manufacturing"]):
        companies = sector_companies["electronics"]
    elif any(kw in prompt_lower for kw in ["hr", "recruitment", "talent"]):
        companies = sector_companies["hr"]
    elif any(kw in prompt_lower for kw in ["regtech", "compliance"]):
        companies = sector_companies["regtech"]
    else:
        companies = sector_companies["default"]

    return json.dumps(companies)


# --- Mock data constants ---

_MOCK_SYNTH_TRENDS = [
    {
        "trend_title": "RBI Mandates Stricter KYC for Digital Lenders",
        "trend_summary": "The Reserve Bank of India has announced comprehensive new KYC requirements affecting over 500 fintech lenders across India. Companies must achieve full compliance within 90 days or face regulatory penalties.",
        "trend_type": "regulation", "severity": "high",
        "key_entities": ["RBI", "Lendingkart", "Capital Float", "ZestMoney", "Paytm Lending"],
        "key_facts": ["90-day compliance deadline", "Affects 500+ fintech lenders", "New video KYC requirements"],
        "key_numbers": ["500+ lenders affected", "90 days deadline", "Rs 50 lakh penalty"],
        "primary_sectors": ["fintech", "banking", "nbfc"],
        "secondary_sectors": ["regtech", "identity_verification"],
        "affected_regions": ["Mumbai", "Bangalore", "Delhi NCR"],
        "is_national": True, "lifecycle_stage": "emerging",
        "confidence_explanation": "Multiple tier-1 sources reporting with official RBI circular reference"
    },
    {
        "trend_title": "Zepto Raises $200M at $5B Valuation",
        "trend_summary": "Quick commerce startup Zepto has closed a massive $200 million funding round, valuing the company at $5 billion. The funds will be used to expand dark store network to 50 new cities.",
        "trend_type": "funding", "severity": "high",
        "key_entities": ["Zepto", "Blinkit", "Swiggy Instamart", "BigBasket"],
        "key_facts": ["$200M Series F round", "Expansion to 50 cities", "10-minute delivery focus"],
        "key_numbers": ["$200 million raised", "$5 billion valuation", "50 new cities"],
        "primary_sectors": ["retail", "logistics", "ecommerce"],
        "secondary_sectors": ["cold_chain", "warehousing"],
        "affected_regions": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
        "is_national": True, "lifecycle_stage": "growing",
        "confidence_explanation": "Official company announcement with investor confirmation"
    },
    {
        "trend_title": "Government Approves 3 New Semiconductor Fabs",
        "trend_summary": "The Union Cabinet has approved Rs 1.26 lakh crore investment for setting up three new semiconductor fabrication plants under the India Semiconductor Mission.",
        "trend_type": "policy", "severity": "high",
        "key_entities": ["Tata Electronics", "Vedanta", "Ministry of Electronics", "India Semiconductor Mission"],
        "key_facts": ["3 new fabs approved", "Rs 1.26 lakh crore investment", "Part of Make in India initiative"],
        "key_numbers": ["Rs 1.26 lakh crore", "3 fabs", "100,000 jobs expected"],
        "primary_sectors": ["manufacturing", "electronics", "semiconductors"],
        "secondary_sectors": ["chemicals", "equipment"],
        "affected_regions": ["Gujarat", "Karnataka", "Tamil Nadu"],
        "is_national": True, "lifecycle_stage": "emerging",
        "confidence_explanation": "Official government announcement with cabinet approval"
    },
    {
        "trend_title": "Reusable Rocket Development Slashes Launch Costs by 40-60%",
        "trend_summary": "SpaceX's Starship full reusability milestone and Relativity Space's 3D-printed Terran R are driving launch costs below $20M per flight. Rocket Lab's Neutron and Blue Origin's New Glenn are entering commercial service, intensifying competition in the $18.7B carrier rocket market projected to reach $35.6B by 2033.",
        "trend_type": "technology", "severity": "high",
        "key_entities": ["SpaceX", "Relativity Space", "Rocket Lab", "Blue Origin", "ULA", "Boeing", "Lockheed Martin"],
        "key_facts": ["Starship launch cost dropped to $15M", "Terran R 85% 3D-printed", "Neutron targets medium-lift reusable segment", "New Glenn opens commercial bookings"],
        "key_numbers": ["$15M per Starship launch", "9.6% market CAGR", "$35.58B market by 2033", "62% SpaceX commercial launch share"],
        "primary_sectors": ["aerospace", "space_launch", "satellite"],
        "secondary_sectors": ["defence", "telecommunications", "earth_observation"],
        "affected_regions": ["North America", "Asia Pacific", "Europe"],
        "is_national": False, "lifecycle_stage": "growing",
        "confidence_explanation": "Multiple tier-1 sources confirming commercial milestones and contract awards"
    }
]

_MOCK_TRENDS = [
    {"trend_title": "RBI Mandates Stricter KYC for Digital Lenders", "summary": "Reserve Bank of India announced new KYC requirements affecting 500+ fintech lenders.", "severity": "high", "industries_affected": ["Fintech", "Digital Lending", "NBFC", "Banking"], "keywords": ["RBI", "KYC", "digital lending", "compliance"], "trend_type": "regulation", "urgency": "Immediate compliance required - 90 day deadline"},
    {"trend_title": "Zepto Raises $200M, Valued at $5B", "summary": "Quick commerce startup Zepto closed massive funding round, plans expansion to 50 new cities.", "severity": "high", "industries_affected": ["Quick Commerce", "E-commerce", "Logistics", "Retail"], "keywords": ["Zepto", "quick commerce", "funding", "dark stores"], "trend_type": "funding", "urgency": "Competitors need to respond to market pressure"},
    {"trend_title": "Swiggy Announces 400 Employee Layoffs", "summary": "Food delivery giant Swiggy cuts 400 jobs in restructuring ahead of IPO.", "severity": "medium", "industries_affected": ["Food Tech", "Gig Economy", "HR Tech", "Recruitment"], "keywords": ["Swiggy", "layoffs", "IPO", "restructuring"], "trend_type": "layoffs", "urgency": "Talent available in market, HR solutions needed"},
    {"trend_title": "Government Approves 3 New Semiconductor Fabs", "summary": "Cabinet clears Rs 1.26 lakh crore investment for semiconductor manufacturing.", "severity": "high", "industries_affected": ["Semiconductors", "Electronics", "Manufacturing", "IT Hardware"], "keywords": ["semiconductor", "PLI scheme", "electronics manufacturing"], "trend_type": "policy", "urgency": "First-mover advantage in emerging ecosystem"},
    {"trend_title": "Reliance Jio Partners with NVIDIA for AI Cloud", "summary": "Jio and NVIDIA announce strategic partnership for enterprise AI infrastructure in India.", "severity": "high", "industries_affected": ["Cloud Computing", "AI/ML", "Enterprise IT", "Data Centers"], "keywords": ["Jio", "NVIDIA", "AI cloud", "enterprise AI"], "trend_type": "partnership", "urgency": "Enterprises need AI strategy now"},
]

_MOCK_IMPACTS = [
    {
        "direct_impact": ["Digital Lending NBFCs", "Fintech Payment Apps", "P2P Lending Platforms", "Buy Now Pay Later Companies"],
        "direct_impact_reasoning": "RBI's new KYC norms directly mandate these companies to overhaul their customer verification processes. Mid-size lenders (50-200 employees) face a 90-day compliance deadline with penalties for non-compliance. Most lack dedicated compliance teams and will need external help to assess gaps, implement new processes, and train staff.",
        "indirect_impact": ["RegTech Software Providers", "Identity Verification Services", "Compliance Consulting Firms"],
        "indirect_impact_reasoning": "As lenders scramble to comply, they'll need technology and consulting support. RegTech demand will surge. KYC verification vendors will see increased volumes. Compliance consultants will be hired for gap assessments.",
        "additional_verticals": ["Banking Software Vendors", "Data Analytics Firms", "Cybersecurity Companies", "Legal Advisory Firms", "API Integration Specialists"],
        "additional_verticals_reasoning": "The KYC overhaul requires system upgrades (software vendors), data handling changes (analytics), security enhancements (cybersecurity), legal review (law firms), and Aadhaar/PAN API integration.",
        "midsize_pain_points": [
            "Need to overhaul KYC processes within 90 days but lack dedicated compliance team",
            "Facing penalties up to 2% revenue for non-compliance but unsure of exact requirements",
            "Board demanding impact assessment but no internal research capability",
            "Competitors already engaging Big 4 consultants — mid-size lenders can't afford them",
            "Customer onboarding will slow down during transition, risking market share loss"
        ],
        "consulting_projects": [
            "KYC compliance gap assessment for mid-size NBFCs",
            "Regulatory landscape mapping and timeline analysis",
            "Cost-benefit analysis of compliance technology options",
            "Benchmarking study of KYC best practices across peer lenders",
            "Vendor evaluation for identity verification solutions"
        ],
        "positive_sectors": ["Fintech Lenders", "RegTech", "Compliance Consulting", "Identity Verification"],
        "negative_sectors": ["Unorganized Moneylenders"],
        "business_opportunities": ["KYC compliance gap assessment", "Regulatory landscape mapping", "Cost-benefit analysis of compliance technology"],
        "relevant_services": ["Market Monitoring", "Consulting and Advisory Services", "Technology Research"],
        "target_roles": ["CEO", "Chief Compliance Officer", "VP Operations", "Director Risk Management", "CTO", "VP Engineering", "Head of Product", "Compliance Manager"],
        "pitch_angle": "Navigate RBI's KYC mandate with expert compliance guidance",
        "reasoning": "90-day deadline creates urgency. Mid-size lenders need external expertise to avoid penalties and operational disruption."
    },
    {
        "direct_impact": ["Quick Commerce Startups", "Grocery Delivery Apps", "Dark Store Operators", "Last-Mile Logistics"],
        "direct_impact_reasoning": "Zepto's $200M funding directly threatens competitors like Blinkit, Instamart, and BigBasket. Mid-size players in 15-25 city range face existential pressure — they can't match Zepto's expansion speed without strategic data on which cities to prioritize and which to abandon.",
        "indirect_impact": ["Cold Chain Infrastructure", "Warehouse Real Estate", "Gig Economy Platforms"],
        "indirect_impact_reasoning": "More dark stores = more cold storage demand. Warehouse rentals in urban areas will spike 15-20%. Gig workers will shift to highest-paying platform, creating retention crisis.",
        "additional_verticals": ["FMCG Brands", "Local Kirana Tech", "Retail Analytics", "Micro-fulfillment Equipment"],
        "additional_verticals_reasoning": "FMCG brands need quick commerce channel strategy. Kirana stores need tech to compete. Analytics firms see demand for hyperlocal consumer data.",
        "midsize_pain_points": [
            "Need competitive intelligence on Zepto's city expansion plans but can't afford Meltwater/Feedly",
            "Dark store unit economics unclear — should they expand or consolidate?",
            "Board asking for market share projections but lack consumer research capability",
            "Delivery partners being poached — need compensation benchmarking data"
        ],
        "consulting_projects": [
            "Competitive intelligence report on quick commerce landscape",
            "Dark store location strategy and unit economics analysis",
            "Supply chain optimization for 10-minute delivery model",
            "Consumer behavior study: quick commerce adoption by city tier"
        ],
        "positive_sectors": ["Quick Commerce", "Cold Chain", "Logistics Tech"],
        "negative_sectors": ["Traditional Retail", "Kirana Stores without Tech"],
        "business_opportunities": ["Competitive intelligence on quick commerce", "Dark store location strategy", "Supply chain optimization"],
        "relevant_services": ["Competitive Intelligence", "Market Intelligence", "Industry Analysis"],
        "target_roles": ["CEO", "Chief Strategy Officer", "VP Supply Chain", "Director Operations", "Product Manager", "Engineering Manager", "Head of Logistics Technology", "Operations Manager"],
        "pitch_angle": "Win the quick commerce battle with strategic intelligence",
        "reasoning": "Funding war intensifies. Mid-size players without data-driven strategy will be acquired or shut down within 18 months."
    },
    {
        "direct_impact": ["Semiconductor Manufacturing", "Electronics Assembly", "PCB Manufacturers", "Chip Design Companies"],
        "direct_impact_reasoning": "Rs 1.26 lakh crore investment directly benefits semiconductor fabs and local component suppliers. Mid-size EMS companies (100-300 employees) need to position themselves in the supply chain NOW or miss the window.",
        "indirect_impact": ["Electronics Contract Manufacturing", "Consumer Electronics Brands", "Automotive Electronics"],
        "indirect_impact_reasoning": "Local chip supply enables contract manufacturers to reduce import dependency. Consumer electronics brands will need to evaluate make-vs-buy decisions. Auto electronics suppliers face component sourcing shifts.",
        "additional_verticals": ["Specialty Chemicals for Semiconductors", "Cleanroom Equipment", "Industrial Gases", "Semiconductor Testing Services"],
        "additional_verticals_reasoning": "Chip fabs need ultra-pure chemicals, cleanroom infrastructure, specialty gases. Testing/validation services will see 10x demand as local production scales up.",
        "midsize_pain_points": [
            "Want to enter semiconductor supply chain but lack knowledge of qualification requirements",
            "Unsure which fab partnerships to pursue — Tata vs Vedanta vs ISMC",
            "Need workforce skill gap analysis for semiconductor manufacturing roles",
            "Board wants ROI projection for semiconductor-adjacent investment but no internal research team"
        ],
        "consulting_projects": [
            "Semiconductor ecosystem supplier identification and profiling",
            "Market entry feasibility study for chip component manufacturing",
            "Workforce skill gap analysis for semiconductor roles",
            "Supply chain mapping: India semiconductor value chain opportunities"
        ],
        "positive_sectors": ["Semiconductors", "Electronics Manufacturing", "Chemical Suppliers"],
        "negative_sectors": ["Chip Importers", "Trading Companies"],
        "business_opportunities": ["Semiconductor ecosystem supplier identification", "Market entry feasibility", "Workforce skill gap analysis"],
        "relevant_services": ["Industry Analysis", "Procurement Intelligence", "Cross Border Expansion"],
        "target_roles": ["CEO", "VP Manufacturing", "Chief Procurement Officer", "Director Strategy", "CTO", "Engineering Manager", "Head of IoT", "Supply Chain Manager", "Product Manager"],
        "pitch_angle": "Capitalize on India's semiconductor revolution",
        "reasoning": "Historic Rs 1.26L crore investment creates once-in-a-generation opportunity. First movers in the supply chain will lock in contracts for decades."
    },
    {
        "direct_impact": ["Small Satellite Launch Providers", "Rocket Component Manufacturers", "Satellite Constellation Operators", "Launch Service Brokers"],
        "direct_impact_reasoning": "SpaceX's reusable Starship at $15M/launch and Relativity Space's 3D-printed Terran R are compressing launch economics. Mid-size launch providers and component suppliers must adapt or face margin erosion as prices fall 40-60%.",
        "indirect_impact": ["Space Insurance Underwriters", "Ground Station Operators", "Satellite Manufacturing Firms"],
        "indirect_impact_reasoning": "Cheaper launches accelerate constellation deployment, driving demand for ground infrastructure and satellite production. Insurance premiums adjust as reusable rockets demonstrate reliability records.",
        "additional_verticals": ["Additive Manufacturing for Aerospace", "AI Flight Software", "Propulsion Test Facilities", "Composite Materials Suppliers", "Space Debris Tracking"],
        "additional_verticals_reasoning": "3D-printed rockets create demand for aerospace-grade additive manufacturing. AI trajectory optimization needs specialized software. Increased launch cadence requires propulsion testing capacity and debris monitoring.",
        "midsize_pain_points": [
            "Need competitive intelligence on SpaceX/Rocket Lab pricing but lack market research team",
            "Evaluating reusable vs expendable architecture but no independent feasibility analysis",
            "Board asking for satellite constellation deployment cost projections but no internal modeling capability",
            "Supply chain for aerospace-grade composites and propulsion components is opaque — no qualified vendor directory",
            "Regulatory landscape for commercial launches varies by jurisdiction — need compliance mapping"
        ],
        "consulting_projects": [
            "Launch cost benchmarking — reusable vs expendable economics by payload class",
            "Satellite constellation deployment strategy — launch provider evaluation and manifest planning",
            "Space supply chain mapping — qualified aerospace component suppliers by region",
            "Market entry feasibility for commercial launch services in Asia-Pacific",
            "Competitive intelligence on SpaceX, Rocket Lab, and Blue Origin commercial strategies"
        ],
        "positive_sectors": ["Commercial Launch Providers", "Satellite Operators", "Ground Systems", "Additive Manufacturing"],
        "negative_sectors": ["Expendable Rocket Manufacturers", "Legacy Launch Service Brokers"],
        "business_opportunities": ["Launch provider benchmarking and selection", "Constellation deployment strategy", "Aerospace supply chain intelligence"],
        "relevant_services": ["Industry Analysis", "Procurement Intelligence", "Competitive Intelligence"],
        "target_roles": ["CEO", "VP Engineering", "Chief Technology Officer", "Director of Launch Operations", "Head of Business Development", "VP Manufacturing", "Chief Procurement Officer", "Programme Manager"],
        "pitch_angle": "Navigate the reusable rocket revolution with independent launch market intelligence",
        "reasoning": "Launch costs falling 40-60% is reshaping the $18.7B carrier rocket market. Companies without independent market data risk overpaying for launches or missing strategic windows."
    }
]

_MOCK_CROSS_TREND_SYNTHESIS = {
    "cross_trend_insight": "India's fintech sector faces a perfect storm: RBI's KYC mandate forces compliance overhaul while aggressive funding rounds (Zepto $200M) are reshaping the competitive landscape. Mid-size companies serving both regulated lenders AND fast-growing commerce platforms are caught in a vice — needing to invest in compliance AND growth simultaneously.",
    "compound_impacts": [
        {
            "company_type": "Mid-size Payment Gateway Providers",
            "affected_by_trends": [1, 2],
            "compound_challenge": "Must upgrade KYC processes (Trend 1) while also scaling infrastructure to handle quick commerce transaction volumes (Trend 2). Dual investment pressure with limited resources.",
            "consulting_opportunity": "Cost-benefit analysis: KYC compliance technology that also scales for high-volume quick commerce transactions"
        },
        {
            "company_type": "Regional Logistics Tech Companies",
            "affected_by_trends": [2, 3],
            "compound_challenge": "Quick commerce expansion (Trend 2) demands dark store logistics while semiconductor policy (Trend 3) shifts electronics supply chains. Their warehouse management systems need to handle both.",
            "consulting_opportunity": "Supply chain reconfiguration study covering both quick commerce fulfillment and electronics component logistics"
        },
        {
            "company_type": "Mid-size NBFC-Fintech Hybrids",
            "affected_by_trends": [1, 2],
            "compound_challenge": "KYC compliance costs (Trend 1) squeeze margins right when they need capital to compete with well-funded quick commerce players offering BNPL (Trend 2).",
            "consulting_opportunity": "Strategic options analysis: compliance-first vs growth-first resource allocation for constrained mid-size lenders"
        }
    ],
    "mega_opportunity": "Help mid-size fintech companies navigate simultaneous regulatory pressure and competitive disruption with integrated compliance-plus-growth strategy consulting."
}

_MOCK_COUNCIL_RESPONSES = {
    "it": {
        "reasoning": "FIRST-ORDER ANALYSIS: The trend directly affects mid-size companies in the target sector. Companies like Mphasis (30K employees), Persistent Systems (23K), Happiest Minds (5K), and Zensar Technologies (15K) face immediate competitive pressure as TCS and Infosys lock up $3B+ in GenAI contracts. These firms must differentiate via vertical specialisation or risk losing enterprise clients who increasingly demand integrated AI delivery. SECOND-ORDER ANALYSIS: HR Tech platforms serving mid-tier IT firms (Xpheno, PeopleStrong) face demand shifts as restructuring at Wipro and Infosys floods the market with experienced talent, depressing placement margins.",
        "first_order_companies": ["Mphasis (30K employees, hyperscaler partnership strategy)", "Persistent Systems (23K, BFSI vertical focus)", "Happiest Minds Technologies (5K, digital-first positioning)", "Zensar Technologies (15K, hi-tech & manufacturing verticals)"],
        "first_order_mechanism": "TCS and Infosys have collectively secured $3B+ in GenAI transformation contracts with proprietary AI platforms (Topaz, WiproCo). Mid-size IT firms cannot match this R&D investment and face client attrition as enterprises consolidate AI vendor relationships with larger players who offer end-to-end delivery.",
        "second_order_companies": ["HR Tech platforms (Xpheno, PeopleStrong) serving mid-tier IT firms", "Cloud infrastructure resellers dependent on mid-size IT channel", "Niche AI training providers (UpGrad, Scaler) serving reskilling demand"],
        "second_order_mechanism": "As mid-size IT firms lose GenAI mandates to larger players, they reduce hiring through HR platforms. Simultaneously, workforce restructuring at Wipro (6K roles) and Infosys (10K roles) creates talent surplus that depresses placement margins for staffing firms.",
        "affected_company_types": ["Mid-size IT Services (5K-30K employees)", "Regional Software Consultancies", "HR Tech Platforms", "Cloud Infrastructure Resellers", "Niche AI Training Providers"],
        "affected_sectors": ["IT Services", "HR Technology", "Cloud Infrastructure", "EdTech"],
        "pain_points": [
            "Need competitive intelligence on TCS/Infosys GenAI deal pipeline but lack internal research team — Mphasis CEO Nitin Rakesh acknowledged reliance on hyperscaler partnerships",
            "Facing margin pressure from clients demanding AI-augmented delivery at legacy billing rates — Persistent Systems BFSI vertical particularly exposed",
            "Board asking for market positioning strategy vs. proprietary AI platforms but no strategic planning capability — Happiest Minds spending Rs 500Cr on AI",
            "Key AI/ML talent being poached by TCS (375K trained) and Infosys (280K trained) offering better upskilling",
            "VP Engineering and Director-level leaders need competitive benchmarking data to justify AI investment to board"
        ],
        "consulting_projects": [
            "Competitive benchmarking: GenAI capabilities across mid-size IT services landscape",
            "Market positioning strategy: differentiation via vertical AI vs. horizontal platform play",
            "Cost structure analysis: margin impact of AI-augmented delivery model adoption",
            "Talent retention benchmarking: compensation and upskilling programmes vs. TCS/Infosys",
            "Client portfolio risk assessment: which accounts are most vulnerable to consolidation"
        ],
        "business_opportunities": ["Competitive intelligence on GenAI deal pipeline", "Strategic positioning study for mid-tier IT", "Cost optimization analysis for AI delivery transformation"],
        "detailed_reasoning": "The Indian IT services landscape is undergoing structural disruption driven by GenAI adoption. TCS ($3B+ GenAI wins), Infosys (Topaz platform, 190 active engagements), and Wipro (WiproCo LLM stack) are rapidly building proprietary AI capabilities that create switching costs for enterprise clients. Mid-size firms like Mphasis, Persistent, Happiest Minds, and Zensar face a strategic inflection point: invest heavily in proprietary AI (capital-intensive, risky) or specialise vertically (defensible but smaller TAM). Engineering managers, product directors, and CTOs at these companies urgently need competitive intelligence to make informed bets. The restructuring wave — Wipro cutting 6K roles, Infosys restructuring 10K — is also creating talent market disruption that affects hiring strategies at mid-size firms.",
        "evidence_citations": ["TCS Q3 FY26: 18% of new TCV from GenAI deals (Economic Times)", "Infosys Topaz: JPMorgan $400M contract (Livemint)", "Wipro restructuring 6K roles (Livemint)", "Mphasis CEO CII speech on hyperscaler strategy (Business Standard)"],
        "service_recommendations": [
            {"service_name": "Competitive Intelligence", "offering": "Competitor profiling and benchmarking — GenAI capabilities across Indian IT majors", "relevance": "Direct need: CTOs and VP Engineering need data on rival AI platforms to justify internal investment"},
            {"service_name": "Market Intelligence", "offering": "Market sizing — enterprise AI services TAM by vertical", "relevance": "Board presentations require credible market data for AI strategy decisions"},
            {"service_name": "Procurement Intelligence", "offering": "Vendor evaluation — AI platform build vs. buy analysis", "relevance": "Engineering managers evaluating hyperscaler vs. proprietary model trade-offs"}
        ],
        "pitch_angle": "Get the competitive intelligence on TCS and Infosys GenAI strategies that your board is asking for — before your next quarterly review",
        "target_roles": ["CEO", "Chief Technology Officer", "VP Engineering", "Director of Engineering", "Chief Strategy Officer", "Head of AI/ML", "Product Manager", "Engineering Manager"],
        "confidence": 0.82,
    },
    "fintech": {
        "reasoning": "FIRST-ORDER ANALYSIS: RBI's new KYC mandate directly impacts mid-size digital lending NBFCs (100-500 employees) that lack dedicated compliance teams. Companies like Lendingkart (150 employees), Capital Float (200 employees), and Easebuzz (120 employees) face a 90-day compliance deadline with penalties up to 2% of revenue for non-compliance. Most rely on manual onboarding and lack automated KYC infrastructure. SECOND-ORDER ANALYSIS: RegTech vendors (Signzy, IDfy, Perfios) see 3x pipeline growth as lenders scramble to implement video KYC. Compliance consulting firms face demand surge but mid-size lenders can't afford Big 4 rates — creating opportunity for CMI-sized firms.",
        "first_order_companies": ["Lendingkart (150 employees, SME lending platform)", "Capital Float (200 employees, digital lending)", "Easebuzz (120 employees, payment gateway + lending)", "ZestMoney (180 employees, BNPL provider)"],
        "first_order_mechanism": "RBI circular mandates video KYC, enhanced due diligence for digital lenders, and real-time PAN-Aadhaar verification within 90 days. Mid-size NBFCs processing 10K-50K loans/month face onboarding slowdown during transition. Non-compliance triggers penalties up to 2% revenue and potential license suspension.",
        "second_order_companies": ["RegTech solutions providers (Signzy, IDfy) — demand surge for KYC APIs", "Compliance consulting firms — gap assessment demand from 500+ lenders", "API integration specialists — Aadhaar/PAN/CKYC integration work"],
        "second_order_mechanism": "As 500+ digital lenders simultaneously seek compliance solutions, RegTech vendors face capacity constraints. Integration timelines stretch from 2 weeks to 6-8 weeks. Lenders that can't integrate fast enough lose customers to compliant competitors, creating cascading revenue pressure.",
        "affected_company_types": ["Mid-size Digital Lending NBFCs (100-500 employees)", "P2P Lending Platforms", "BNPL Providers", "Payment Gateway Companies with lending arms", "RegTech Software Providers"],
        "affected_sectors": ["Digital Lending", "Fintech", "RegTech", "Identity Verification", "Compliance Consulting"],
        "pain_points": [
            "Need to overhaul KYC processes within 90 days but lack dedicated compliance team — Lendingkart has only 2 compliance staff for 10K monthly applications",
            "Facing penalties up to 2% revenue for non-compliance but unsure which exact clauses apply to their lending vertical",
            "Board demanding impact assessment but no internal research capability — Capital Float CEO needs data for investor update",
            "Competitors already engaging Big 4 consultants at Rs 50L+ — mid-size lenders can't afford that but need equivalent guidance",
            "Customer onboarding will slow 40-60% during KYC transition, risking market share to compliant competitors like PhonePe Lending"
        ],
        "consulting_projects": [
            "KYC compliance gap assessment benchmarked against 50+ peer NBFCs",
            "Regulatory landscape mapping — which RBI clauses affect which lending verticals",
            "Vendor evaluation — RegTech solutions (Signzy, IDfy, Perfios) compared on cost and integration time",
            "Cost-benefit analysis — build vs buy for KYC infrastructure",
            "Competitive intelligence — how peer lenders are responding to the mandate"
        ],
        "business_opportunities": ["KYC compliance gap assessment for mid-size NBFCs", "RegTech vendor evaluation and selection", "Regulatory landscape mapping for digital lending"],
        "detailed_reasoning": "India's digital lending sector faces a regulatory inflection point. The RBI's comprehensive KYC mandate affects 500+ fintech lenders who must overhaul customer verification within 90 days. Mid-size NBFCs like Lendingkart, Capital Float, and ZestMoney are particularly vulnerable — they process thousands of loans monthly but lack the compliance infrastructure of banks. The mandate requires video KYC, real-time PAN-Aadhaar verification, and enhanced due diligence that most mid-size lenders haven't implemented. CEOs and compliance officers at these companies urgently need gap assessments, vendor evaluations, and implementation roadmaps. The 90-day deadline creates natural urgency — companies that don't comply face penalties and potential license suspension. Meanwhile, the compliance rush is creating a seller's market for RegTech vendors, making independent vendor evaluation even more valuable.",
        "evidence_citations": ["RBI circular on digital lending KYC — 90 day compliance deadline (RBI website)", "500+ registered digital lenders affected (Fintech Association of India)", "Penalties up to 2% revenue for non-compliance (Economic Times)", "Lendingkart processes 10K+ SME loans monthly (Company filing)"],
        "service_recommendations": [
            {"service_name": "Market Monitoring", "offering": "Regulatory tracking — RBI digital lending compliance updates and peer benchmarking", "relevance": "Compliance officers need real-time regulatory intelligence to avoid penalties"},
            {"service_name": "Consulting and Advisory", "offering": "KYC gap assessment — compliance readiness audit benchmarked against 50+ NBFCs", "relevance": "CEOs need actionable gap analysis before the 90-day deadline"},
            {"service_name": "Technology Research", "offering": "RegTech vendor evaluation — KYC solution comparison on cost, speed, and coverage", "relevance": "CTOs evaluating build-vs-buy for KYC infrastructure"}
        ],
        "pitch_angle": "Navigate RBI's KYC mandate with expert compliance guidance — before the 90-day deadline hits",
        "target_roles": ["CEO", "Chief Compliance Officer", "VP Operations", "CTO", "Director Risk Management", "Head of Product", "Compliance Manager", "VP Engineering"],
        "confidence": 0.85,
    },
    "telecom": {
        "reasoning": "FIRST-ORDER ANALYSIS: India's enterprise 5G rollout across 80 cities creates immediate demand for private network deployments in manufacturing. Mid-size manufacturers like Endurance Technologies (3K employees), Sundram Fasteners (5K), and Sona BLW Precision (2K) are evaluating factory-floor IoT but lack internal telecom expertise. Airtel's Rs 2,400Cr Maruti contract sets the benchmark but mid-size firms can't negotiate similar terms without market intelligence. SECOND-ORDER ANALYSIS: IoT platform companies (STL, Tejas Networks) need to position as system integrators. Edge computing vendors face demand for on-premise inference hardware. Industrial sensor manufacturers see 5x order pipeline growth.",
        "first_order_companies": ["Endurance Technologies (3K employees, auto components manufacturer)", "Sundram Fasteners (5K employees, precision engineering)", "Sona BLW Precision (2K employees, auto drivetrain)", "Bharat Forge (4K employees, forging and manufacturing)"],
        "first_order_mechanism": "Enterprise 5G enables real-time IoT monitoring, predictive maintenance, and automated quality inspection on factory floors. Mid-size manufacturers investing Rs 50-200Cr in smart factory upgrades need supply chain mapping to identify qualified IoT vendors, sensor suppliers, and system integrators — information that's fragmented across 100+ vendors.",
        "second_order_companies": ["IoT platform companies (STL, Tejas Networks) — need positioning as 5G system integrators", "Industrial sensor manufacturers — 5x order pipeline from smart factory deployments", "Edge computing hardware vendors — on-premise inference demand from latency-sensitive manufacturing"],
        "second_order_mechanism": "As mid-size manufacturers adopt private 5G, they create cascading demand through the IoT supply chain. Sensor vendors need to qualify for automotive/manufacturing standards. System integrators face capacity constraints as 100+ factories start pilots simultaneously. Edge computing vendors must adapt data center products for harsh factory environments.",
        "affected_company_types": ["Mid-size Auto Component Manufacturers (1K-5K employees)", "Precision Engineering Companies", "IoT Platform and System Integrators", "Industrial Sensor Manufacturers", "Edge Computing Hardware Vendors"],
        "affected_sectors": ["Manufacturing", "Telecom Equipment", "IoT Platforms", "Industrial Sensors", "Edge Computing"],
        "pain_points": [
            "Need supply chain mapping for 5G IoT ecosystem but no internal telecom expertise — Endurance Technologies CTO evaluating 50+ IoT vendors without benchmarking data",
            "Airtel's Rs 2,400Cr Maruti deal sets pricing benchmark but mid-size firms can't negotiate without competitive intelligence on private 5G costs",
            "Board asking for ROI projection on smart factory investment but lack feasibility analysis — Sundram Fasteners wants to pilot in 2 plants before full rollout",
            "Facing vendor lock-in risk from single IoT platform provider but no independent evaluation framework",
            "Manufacturing engineers need sensor qualification data for automotive standards (IATF 16949) but information is scattered across vendor datasheets"
        ],
        "consulting_projects": [
            "Supply chain mapping — India's 5G IoT ecosystem from sensors to platforms to integrators",
            "Market entry feasibility — unit economics for private 5G network deployment in manufacturing",
            "Vendor evaluation — IoT platform comparison (cost, scalability, manufacturing-specific features)",
            "ROI analysis — smart factory investment payback for mid-size auto component manufacturers",
            "Competitive intelligence — how peer manufacturers are adopting Industry 4.0 technologies"
        ],
        "business_opportunities": ["5G IoT supply chain mapping for manufacturers", "Smart factory feasibility and ROI analysis", "IoT vendor evaluation for manufacturing"],
        "detailed_reasoning": "India's enterprise 5G rollout is creating a once-in-a-decade opportunity for mid-size manufacturers. With coverage now spanning 80 cities and anchor deals like Airtel-Maruti (Rs 2,400Cr) demonstrating ROI, the question for mid-size manufacturers has shifted from 'should we adopt 5G IoT?' to 'how fast can we deploy?'. Companies like Endurance Technologies, Sundram Fasteners, and Sona BLW are evaluating smart factory investments of Rs 50-200Cr but face a fragmented vendor ecosystem with 100+ IoT platform providers, sensor manufacturers, and system integrators. They lack the internal telecom expertise to evaluate options and the benchmarking data to negotiate fair pricing. The first movers who build smart factories will gain 15-20% efficiency advantages, creating competitive pressure on laggards. Manufacturing CTOs and plant managers urgently need supply chain intelligence, vendor evaluations, and feasibility studies.",
        "evidence_citations": ["Airtel-Maruti Rs 2,400Cr private 5G contract (Economic Times)", "Enterprise 5G coverage in 80 Indian cities (TRAI report)", "Smart factory ROI: 15-20% efficiency gain in pilot deployments (Frost & Sullivan)", "100+ IoT platform vendors in India (NASSCOM IoT report)"],
        "service_recommendations": [
            {"service_name": "Procurement Intelligence", "offering": "5G IoT vendor evaluation — platform comparison for manufacturing use cases", "relevance": "CTOs need independent vendor assessment to avoid lock-in and negotiate fair pricing"},
            {"service_name": "Industry Analysis", "offering": "Smart factory feasibility — ROI analysis for private 5G deployment in auto manufacturing", "relevance": "Board-level decision support for Rs 50-200Cr capex investments"},
            {"service_name": "Supply Chain Analysis", "offering": "5G IoT ecosystem mapping — sensors, platforms, integrators qualified for manufacturing", "relevance": "Procurement teams need qualified vendor shortlists for IATF 16949 compliance"}
        ],
        "pitch_angle": "Map the 5G IoT supply chain for your smart factory investment — before competitors lock in the best vendors",
        "target_roles": ["CEO", "Chief Technology Officer", "VP Manufacturing", "Plant Manager", "Head of IoT", "Director Strategy", "Chief Procurement Officer", "Engineering Manager"],
        "confidence": 0.80,
    },
    "space": {
        "reasoning": "FIRST-ORDER ANALYSIS: The carrier rocket market's shift to reusability is creating a two-tier industry. SpaceX ($15M/launch) and Relativity Space (85% 3D-printed) set new cost benchmarks. Mid-size launch providers like Skyroot Aerospace (200 employees), Agnikul Cosmos (150 employees), and Isar Aerospace (300 employees) must achieve competitive unit economics or risk being priced out. SECOND-ORDER ANALYSIS: Satellite constellation operators (OneWeb-Eutelsat, AST SpaceMobile) benefit from cheaper launches but face launch manifest congestion. Ground systems and satellite manufacturing firms see cascading demand as deployment cadence accelerates.",
        "first_order_companies": ["Skyroot Aerospace (200 employees, India's first private orbital launch)", "Agnikul Cosmos (150 employees, 3D-printed rocket engines)", "Isar Aerospace (300 employees, European sovereign launch)", "Space One (180 employees, Japanese small orbital rocket)"],
        "first_order_mechanism": "Reusable rockets from SpaceX and Rocket Lab compress launch pricing 40-60%, forcing mid-size launch providers to differentiate on responsiveness, dedicated orbits, or regional access. Companies that cannot demonstrate competitive unit economics within 2-3 years face acquisition or shutdown.",
        "second_order_companies": ["Satellite constellation operators accelerating deployment timelines", "Ground station network providers scaling capacity", "Aerospace composite material suppliers seeing 5x order growth"],
        "second_order_mechanism": "Cheaper launches unlock constellation deployments that were previously cost-prohibitive. This cascading demand stresses ground infrastructure, satellite manufacturing, and spectrum allocation. Insurance underwriters must reprice as reusable rockets build flight heritage.",
        "affected_company_types": ["Mid-size Launch Providers (100-500 employees)", "Rocket Component Manufacturers", "Satellite Constellation Operators", "Ground Systems Companies", "Aerospace Material Suppliers"],
        "affected_sectors": ["Space Launch Services", "Satellite Manufacturing", "Ground Systems", "Aerospace Materials", "Space Insurance"],
        "pain_points": [
            "Need launch cost benchmarking data to negotiate with SpaceX/Rocket Lab but lack internal market research — Skyroot CEO needs data for Series C investor presentations",
            "Evaluating reusable vs expendable rocket architecture but no independent techno-economic analysis available at mid-size budget",
            "Board asking for 5-year market projections but lack access to launch manifest data and pricing intelligence",
            "Supplier qualification for aerospace-grade composites and propulsion components is opaque — no centralized directory",
            "Launch insurance premiums are 5-15% of mission cost but no independent actuarial benchmarking available for new vehicles"
        ],
        "consulting_projects": [
            "Launch cost benchmarking — reusable vs expendable economics by payload class and orbit",
            "Competitive intelligence — SpaceX, Rocket Lab, Blue Origin commercial strategies and pricing trends",
            "Aerospace supply chain mapping — qualified component suppliers by region and certification",
            "Market entry feasibility — commercial launch services market opportunity by geography",
            "Constellation deployment strategy — launch provider evaluation, manifest optimization, and risk diversification"
        ],
        "business_opportunities": ["Launch market competitive intelligence", "Aerospace supply chain mapping", "Constellation deployment strategy consulting"],
        "detailed_reasoning": "The global carrier rocket market ($18.7B in 2026, projected $35.6B by 2033 at 9.6% CAGR) is undergoing a structural shift driven by reusability. SpaceX controls 62% of commercial launch revenue with Falcon 9/Starship. Relativity Space's 3D-printed Terran R reduces production from 18 months to 60 days. Rocket Lab's Neutron and Blue Origin's New Glenn are entering commercial service. Mid-size launch providers — Skyroot, Agnikul, Isar Aerospace, Space One — face a strategic inflection point: they must find defensible niches (dedicated small-sat launches, regional access, sovereign launch capacity) or be priced out. Their engineering and business development teams urgently need competitive intelligence, supply chain mapping, and market feasibility studies to make informed strategic decisions.",
        "evidence_citations": ["SpaceX Starship $15M/launch achieved (Reuters)", "Relativity Space Terran R 85% 3D-printed first orbital flight (SpaceNews)", "Carrier rocket market $18.7B to $35.6B by 2033 at 9.6% CAGR (CMI)", "Rocket Lab Neutron $2.1B launch contracts (Ars Technica)"],
        "service_recommendations": [
            {"service_name": "Competitive Intelligence", "offering": "Launch market benchmarking — pricing, cadence, and capability comparison across 10+ providers", "relevance": "CEOs and BD teams need data to position against SpaceX and Rocket Lab in customer negotiations"},
            {"service_name": "Procurement Intelligence", "offering": "Aerospace supply chain mapping — qualified propulsion, composite, and avionics suppliers by region", "relevance": "VP Engineering teams need qualified vendor shortlists for AS9100-certified components"},
            {"service_name": "Industry Analysis", "offering": "Constellation deployment market sizing — demand forecast by orbit type and application", "relevance": "Strategy teams need market sizing for investor presentations and go-to-market planning"}
        ],
        "pitch_angle": "Get independent launch market intelligence to position your rocket company in a market dominated by SpaceX",
        "target_roles": ["CEO", "Chief Technology Officer", "VP Engineering", "Head of Business Development", "Director of Launch Operations", "VP Manufacturing", "Chief Procurement Officer", "Programme Manager"],
        "confidence": 0.82,
    },
}

_MOCK_EMAILS = [
    {"subject": "RBI KYC Mandate — Compliance Gap Assessment for Mid-Size Lenders", "body": "Hi {first_name},\n\nWith the RBI's 90-day KYC compliance deadline now in effect, many mid-size NBFCs like Lendingkart and Capital Float are racing to overhaul onboarding flows while competitors engage Big 4 firms.\n\nAt Coherent Market Insights, we've been tracking this regulatory shift closely. For companies in your position, we offer:\n\n\u2022 Regulatory landscape mapping — exactly which clauses affect your lending vertical\n\u2022 KYC compliance gap assessment — benchmarked against 50+ peer NBFCs\n\u2022 Vendor evaluation — RegTech solutions (Signzy, IDfy, Perfios) compared on cost, integration time, and compliance coverage\n\nWe recently completed a similar assessment for a Series B fintech lender that reduced their compliance timeline from 90 to 45 days.\n\nWould 15 minutes work to discuss how this applies to your team?\n\nBest regards,\nCoherent Market Insights"},
    {"subject": "GenAI Competitive Intelligence — Where Mid-Size IT Stands vs TCS & Infosys", "body": "Hi {first_name},\n\nTCS and Infosys have locked up $3B+ in GenAI transformation contracts this quarter alone. For mid-size IT services firms, the question isn't whether to invest in AI — it's where to place the bet.\n\nAt Coherent Market Insights, we help technology leaders make data-driven decisions:\n\n\u2022 Competitive benchmarking — GenAI capabilities across 15 Indian IT firms (deal pipeline, platform maturity, talent density)\n\u2022 Market positioning strategy — vertical AI specialisation vs horizontal platform play\n\u2022 Client portfolio risk assessment — which enterprise accounts are most vulnerable to consolidation by larger players\n\nOur latest report covers how Mphasis, Persistent, and Happiest Minds are positioning against the top-tier — I'd be happy to share a preview.\n\n15 minutes?\n\nBest regards,\nCoherent Market Insights"},
    {"subject": "5G Enterprise IoT — Supply Chain Opportunity Map for Mid-Size Manufacturers", "body": "Hi {first_name},\n\nWith India's enterprise 5G coverage now spanning 80 cities and Airtel's Rs 2,400Cr Maruti contract setting the benchmark, mid-size manufacturers and IoT firms have a narrow window to position in the supply chain.\n\nAt Coherent Market Insights, we specialise in:\n\n\u2022 Supply chain mapping — India's 5G IoT ecosystem from sensors to platforms\n\u2022 Supplier identification — qualified component and system integration partners\n\u2022 Market entry feasibility — unit economics for private 5G network deployments\n\nEndurance Technologies, Sundram Fasteners and similar firms are already piloting — the question is where to invest first.\n\nWorth a 15-minute call to discuss your sector?\n\nBest regards,\nCoherent Market Insights"},
    {"subject": "Carrier Rocket Market Intelligence — Launch Cost Benchmarking for Mid-Size Providers", "body": "Hi {first_name},\n\nWith SpaceX's Starship achieving full reusability at $15M/launch and Relativity Space's 3D-printed Terran R completing its first orbital flight, the carrier rocket market is restructuring fast.\n\nAt Coherent Market Insights, we help aerospace companies navigate this shift:\n\n\u2022 Launch cost benchmarking — reusable vs expendable economics across payload classes\n\u2022 Competitive intelligence — SpaceX, Rocket Lab, and Blue Origin pricing and strategy\n\u2022 Aerospace supply chain mapping — qualified propulsion, composite, and avionics suppliers\n\nOur latest carrier rocket market analysis covers the $18.7B-to-$35.6B market trajectory and identifies where mid-size launch providers can find defensible niches.\n\nWould 15 minutes work to discuss how this applies to your launch programme?\n\nBest regards,\nCoherent Market Insights"},
]
