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
        # Check for companies FIRST (company prompts may also contain "trend")
        # Use specific phrases from CompanyAgent prompts, not broad substrings
        if ("compan" in prompt_lower and ("find" in prompt_lower or "json array" in prompt_lower)) or "extract companies" in prompt_lower:
            return _get_mock_company_response(prompt_lower)
        if "synthesize" in prompt_lower or "cluster" in prompt_lower:
            return json.dumps(_MOCK_SYNTH_TRENDS[prompt_hash % 3])
        # Cross-trend synthesis (compound impacts) — must be before impact check
        elif "compound" in prompt_lower and "simultaneously" in prompt_lower:
            return json.dumps(_MOCK_CROSS_TREND_SYNTHESIS)
        # Impact/consultant prompts contain "trend" too (the trend title),
        # so check impact keywords BEFORE the generic trend branch
        elif "impact" in prompt_lower or "consultant" in prompt_lower or "direct_impact" in prompt_lower:
            return json.dumps(_MOCK_IMPACTS[prompt_hash % 3])
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


def get_mock_response_for_function_model(messages: list[Any], info: Any) -> ModelResponse:
    """Adapter for pydantic-ai FunctionModel.

    FunctionModel passes ModelMessage objects. We extract the user prompt
    text and delegate to the main mock function. Returns a ModelResponse
    (required by pydantic-ai >= 1.0).
    """
    # Extract user prompt from messages (skip system prompts)
    prompt = ""
    system_prompt = ""
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if not hasattr(part, 'content') or not isinstance(part.content, str):
                    continue
                # UserPromptPart → the actual prompt
                part_type = type(part).__name__
                if "User" in part_type:
                    prompt = part.content
                elif "System" in part_type:
                    system_prompt = part.content
                elif not prompt:
                    prompt = part.content
    if not prompt and messages:
        prompt = str(messages[-1])

    # Detect JSON mode from either prompt or system prompt
    all_text = (prompt + " " + system_prompt).lower()
    json_mode = "json" in all_text

    text = get_mock_response(prompt, json_mode)
    return ModelResponse(parts=[TextPart(content=text)])


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
        "default": [
            {"company_name": "Moglix", "company_size": "mid", "industry": "B2B Commerce", "website": "https://moglix.com", "description": "Industrial B2B marketplace, 280 employees", "intent_signal": "Seeking procurement intelligence", "reason_relevant": "Supply chain needs"},
            {"company_name": "OfBusiness", "company_size": "mid", "industry": "B2B Commerce", "website": "https://ofbusiness.com", "description": "B2B raw materials, 250 employees", "intent_signal": "Expanding supplier network", "reason_relevant": "Procurement opportunity"},
        ],
    }

    if any(kw in prompt_lower for kw in ["oil", "energy", "fuel", "petro"]):
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
        "target_roles": ["CEO", "Chief Compliance Officer", "VP Operations", "Director Risk Management"],
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
        "target_roles": ["CEO", "Chief Strategy Officer", "VP Supply Chain", "Director Operations"],
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
        "target_roles": ["CEO", "VP Manufacturing", "Chief Procurement Officer", "Director Strategy"],
        "pitch_angle": "Capitalize on India's semiconductor revolution",
        "reasoning": "Historic Rs 1.26L crore investment creates once-in-a-generation opportunity. First movers in the supply chain will lock in contracts for decades."
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

_MOCK_EMAILS = [
    {"subject": "RBI's New KYC Norms - Impact Assessment for Lenders", "body": "Hi there,\n\nI noticed the RBI's new KYC mandate and thought of your company. With the 90-day compliance deadline, many fintech lenders are scrambling to understand the full impact.\n\nAt Coherent Market Insights, we can help with:\n\n\u2022 Regulatory compliance landscape assessment\n\u2022 Cost-impact analysis of new KYC requirements\n\u2022 Benchmarking against industry best practices\n\nWould you be open to a 15-minute call?\n\nBest regards,\nCoherent Market Insights Team"},
    {"subject": "Quick Commerce Battle - Competitive Intelligence Opportunity", "body": "Hi there,\n\nWith Zepto's $200M raise and aggressive expansion plans, the quick commerce landscape is shifting rapidly.\n\nAt Coherent Market Insights, we help companies navigate competitive disruptions through:\n\n\u2022 Competitor profiling and strategy analysis\n\u2022 Market share tracking and benchmarking\n\u2022 Go-to-market strategy recommendations\n\nWould a 15-minute call be useful?\n\nBest regards,\nCoherent Market Insights Team"},
    {"subject": "Semiconductor Policy - Supply Chain Opportunity Analysis", "body": "Hi there,\n\nThe government's Rs 1.26 lakh crore semiconductor investment opens significant opportunities.\n\nAt Coherent Market Insights, we specialize in:\n\n\u2022 Supply chain opportunity mapping\n\u2022 Supplier identification and profiling\n\u2022 Market entry feasibility studies\n\nWould you be interested in a 15-minute discussion?\n\nBest regards,\nCoherent Market Insights Team"},
]
