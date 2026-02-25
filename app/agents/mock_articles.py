"""
Mock article data for pipeline testing (mock_mode=True).

Kept in a separate file to avoid cluttering source_intel.py.
Three tight clusters so clustering/synthesis/impact stages all get realistic input.
"""

MOCK_ARTICLES_RAW = [
    # ── Cluster 1: RBI KYC regulation (6 articles) ──────────────────────────
    (
        "RBI Mandates New KYC Norms for Digital Lenders",
        "The Reserve Bank of India has issued a circular mandating enhanced KYC verification "
        "for all digital lenders within 90 days. Over 500 fintech NBFCs must upgrade their "
        "video-KYC systems or face penalties up to Rs 50 lakh.",
        "Economic Times", "regulation",
    ),
    (
        "Digital Lenders Face 90-Day Compliance Crunch on RBI KYC",
        "Fintech lenders are racing to comply with RBI's new KYC rules. The central bank's "
        "February circular requires biometric verification for loans above Rs 50,000, "
        "affecting Lendingkart, Capital Float and ZestMoney.",
        "Livemint", "regulation",
    ),
    (
        "RBI Tightens KYC: NBFCs Must Submit Compliance Plans by March",
        "The regulator expects all NBFCs to file a remediation plan by March 31. Failure to "
        "comply risks suspension of new loan disbursal. RegTech vendors report a 3x spike in inquiries.",
        "Business Standard", "regulation",
    ),
    (
        "RBI Governor Warns on KYC Compliance Gaps at Digital Lenders",
        "Sanjay Malhotra flagged compliance gaps at the Annual NBFC Summit. RBI has issued "
        "show-cause notices to 18 lenders. Mid-size companies are most exposed given limited "
        "internal compliance teams.",
        "Hindu BusinessLine", "regulation",
    ),
    (
        "Fintech Lenders Rush to Hire Compliance Heads After RBI Circular",
        "Job postings for Chief Compliance Officers surged 45% on LinkedIn after the RBI mandate. "
        "The 90-day window is too short for companies without compliance infrastructure, analysts warn.",
        "Inc42", "regulation",
    ),
    (
        "RegTech Startups See 3x Demand as RBI KYC Deadline Looms",
        "Identity verification firms Signzy and IDfy are seeing record inbound interest. "
        "Signzy CEO says their pipeline has tripled since the RBI circular. "
        "Compliance tech investment is surging.",
        "YourStory", "regulation",
    ),

    # ── Cluster 2: Quick commerce funding (6 articles) ───────────────────────
    (
        "Zepto Raises $200M at $5B Valuation for Dark Store Expansion",
        "Quick commerce startup Zepto closed a $200 million Series F at $5 billion valuation. "
        "Founders plan to add 200 dark stores in Tier-2 cities within six months, "
        "challenging Blinkit and Swiggy Instamart.",
        "Economic Times", "funding",
    ),
    (
        "Zepto Plans 50 New Cities, Challenging Blinkit and Instamart",
        "Armed with fresh capital, Zepto is entering Nagpur, Coimbatore and Kochi. "
        "CEO says they are hiring 5,000 dark store staff. Blinkit has responded by announcing "
        "300 new stores to defend market share.",
        "Livemint", "funding",
    ),
    (
        "Quick Commerce Heats Up: Funding Rounds Total $1B in 2025",
        "Aggregate VC investment into Indian quick commerce surpassed $1 billion this year. "
        "Companies without scale will face consolidation pressure by 2026 as "
        "10-minute delivery becomes table stakes.",
        "Mint", "funding",
    ),
    (
        "Dark Store Landlords Demand 40% Premium as Quick Commerce Expands",
        "Real estate owners near metro areas are commanding 35-40% rental premium for "
        "dark store spaces. Cold storage and EV-charging capable facilities see highest demand.",
        "Hindu BusinessLine", "funding",
    ),
    (
        "Zepto Eyes B2B Deliveries to Restaurants After Series F Close",
        "Following the funding round, Zepto is piloting B2B supply deliveries to 300 cloud "
        "kitchen operators in Mumbai, positioning against Swiggy's Genie and Dunzo Daily.",
        "Business Standard", "funding",
    ),
    (
        "Quick Commerce Giants Locked in Price War on Grocery Margins",
        "Blinkit, Zepto and Swiggy Instamart are offering 20-30% discount events that compress "
        "category margins to -8% on average. Suppliers and cold-chain operators are alarmed "
        "by the unsustainability.",
        "Economic Times", "funding",
    ),

    # ── Cluster 3: Semiconductor policy (5 articles) ─────────────────────────
    (
        "Cabinet Approves Rs 1.26 Lakh Crore for 3 Semiconductor Fabs",
        "The Union Cabinet cleared Rs 1,26,000 crore for three chip plants. Tata Electronics "
        "gets Dholera (28nm), Vedanta-Foxconn gets Pune, and ISMC sets up an analogue fab "
        "in Karnataka.",
        "PIB", "policy",
    ),
    (
        "India Semiconductor Mission Awards First Fab Contracts",
        "ISM CEO confirmed Tata and CG Power have signed final investment agreements. "
        "The fabs will produce 50,000 wafer starts per month at peak capacity by 2027.",
        "Economic Times", "policy",
    ),
    (
        "EMS Companies Scramble for Semiconductor Supply Chain Position",
        "Mid-size EMS firms are holding emergency board meetings to assess chip ecosystem entry. "
        "Syrma SGS and VVDN Technologies have announced dedicated semiconductor sub-assembly lines.",
        "Financial Express", "policy",
    ),
    (
        "India's PLI Scheme Attracts 18 Chipmakers: MEITY Report",
        "MEITY's report shows 18 foreign chipmakers have expressed intent to set up design "
        "centres or ATMP facilities under PLI. Assembly, test, mark, pack capacity is seen "
        "as the near-term opportunity.",
        "Business Standard", "policy",
    ),
    (
        "Semiconductor Workforce Gap: India Needs 85,000 Engineers by 2026",
        "NASSCOM estimates India's semiconductor sector needs 85,000 additional engineers. "
        "Only 12,000 graduate annually with chip-relevant skills. IITs and bootcamp providers "
        "are launching crash programmes.",
        "Mint", "policy",
    ),
]
