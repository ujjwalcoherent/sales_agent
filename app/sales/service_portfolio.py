"""
Service portfolio — defines what we sell. That's it.

The LLM handles all matching logic (which trends → which services).
No hardcoded keyword patterns, no regex, no if/else matching.
This is just the source of truth for our service offerings.
"""

SERVICE_VERTICALS = {
    "procurement_intelligence": {
        "name": "Procurement Intelligence",
        "sub_services": [
            "Supplier identification and profiling",
            "Cost structure and should-cost analysis",
            "Commodity and category market analysis",
            "Supply base risk assessment and mitigation",
            "Benchmarking of procurement practices",
            "Supplier performance evaluation",
            "Procurement process optimization",
            "Contract and negotiation support",
            "Spend analysis and savings opportunity identification",
        ],
    },
    "market_intelligence": {
        "name": "Market Intelligence",
        "sub_services": [
            "Market sizing and segmentation",
            "Market trends and growth forecasts",
            "Customer needs and behavior analysis",
            "Regulatory and policy landscape assessment",
            "Channel and distribution analysis",
            "Opportunity and threat identification",
            "Product and service landscape mapping",
            "Market entry and expansion feasibility",
            "Trade analysis (export-import)",
            "Pricing analysis",
        ],
    },
    "competitive_intelligence": {
        "name": "Competitive Intelligence",
        "sub_services": [
            "Competitor profiling and benchmarking",
            "Analysis of competitor strategies, strengths, and weaknesses",
            "Product and service comparisons",
            "Pricing and go-to-market analysis",
            "Tracking competitor marketing and sales activities",
            "Monitoring of new product launches and innovations",
            "Mergers, acquisitions, and partnership tracking",
            "Sector and industry trend analysis",
        ],
    },
    "market_monitoring": {
        "name": "Market Monitoring",
        "sub_services": [
            "Ongoing tracking of market trends and developments",
            "Real-time updates on regulatory and economic changes",
            "Monitoring competitor and supplier activities",
            "Periodic market and industry reports",
            "Alerts on key market events and disruptions",
            "Tracking customer sentiment and feedback",
            "Early warning systems for emerging risks",
        ],
    },
    "industry_analysis": {
        "name": "Industry Analysis",
        "sub_services": [
            "Industry structure and value chain mapping",
            "Key industry drivers and challenges",
            "Regulatory and compliance environment review",
            "Analysis of technological advancements and disruptions",
            "Industry benchmarking and best practices",
            "Demand and supply dynamics assessment",
            "Identification of key players and market shares",
        ],
    },
    "technology_research": {
        "name": "Technology Research",
        "sub_services": [
            "Technology landscape and trends analysis",
            "Assessment of emerging and disruptive technologies",
            "Technology adoption and impact studies",
            "Patent and intellectual property analysis",
            "Vendor and solution evaluation",
            "R&D pipeline and innovation tracking",
            "Technology feasibility and ROI assessment",
        ],
    },
    "cross_border_expansion": {
        "name": "Cross-Border Expansion",
        "sub_services": [
            "Market entry strategy and feasibility studies",
            "Regulatory and compliance advisory for new markets",
            "Local partner and supplier identification",
            "Cultural and consumer behavior analysis",
            "Go-to-market planning and localization",
            "Competitive landscape in target geographies",
            "Risk assessment and mitigation for international operations",
        ],
    },
    "consumer_insights": {
        "name": "Consumer Insights",
        "sub_services": [
            "Consumer behavior and attitude analysis",
            "Segmentation and persona development",
            "Customer journey mapping",
            "Brand perception and loyalty studies",
            "Product and service usage analysis",
            "Voice of customer (VoC) research",
            "Socio-demographic and psychographic profiling",
            "Customer satisfaction and NPS tracking",
        ],
    },
    "consulting_advisory": {
        "name": "Consulting & Advisory Services",
        "sub_services": [
            "Strategic planning and business transformation",
            "Financial advisory and performance improvement",
            "Operational efficiency and process optimization",
            "Technology and digital transformation advisory",
            "Change management and organizational development",
            "Market entry and growth strategy",
        ],
    },
}


def get_portfolio_for_prompt() -> str:
    """Compact service list for LLM prompts."""
    lines = []
    for i, (vid, v) in enumerate(SERVICE_VERTICALS.items(), 1):
        subs = ", ".join(v["sub_services"][:4])
        lines.append(f"{i}. {v['name']}: {subs}")
    return "\n".join(lines)
