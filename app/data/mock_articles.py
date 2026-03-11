"""
Mock article data for pipeline testing (mock_mode=True).

Used by agents/source_intel.py when mock_mode is enabled.
Three tight clusters per use-case so clustering/synthesis/impact stages
get realistic, mode-appropriate input.

Datasets:
  MOCK_ARTICLES_RAW            — default / industry_first (fintech + regulation + quick-commerce)
  MOCK_ARTICLES_COMPANY_FIRST  — IT services companies (TCS, Infosys, Wipro, HCL, etc.)
  MOCK_ARTICLES_REPORT_DRIVEN  — 5G / IoT / telecom theme
"""

# ──────────────────────────────────────────────────────────────────────────────
# Default dataset — used for industry_first (fintech / regulation / e-commerce)
# ──────────────────────────────────────────────────────────────────────────────

MOCK_ARTICLES_RAW = [
    # ── Cluster 1: RBI KYC regulation (6 articles) ──────────────────────────
    (
        "RBI Mandates New KYC Norms for Digital Lenders",
        "The Reserve Bank of India has issued a circular mandating enhanced KYC verification "
        "for all digital lenders within 90 days. Over 500 fintech NBFCs must upgrade their "
        "video-KYC systems or face penalties up to Rs 50 lakh. The RBI directive specifically "
        "targets Lendingkart, Capital Float, ZestMoney and other mid-size digital lending "
        "platforms that have grown rapidly without commensurate compliance infrastructure. "
        "The regulator has flagged that many lenders rely on legacy identity checks that do not "
        "meet the updated Prevention of Money Laundering Act requirements.",
        "Economic Times", "regulation",
    ),
    (
        "Digital Lenders Face 90-Day Compliance Crunch on RBI KYC",
        "Fintech lenders are racing to comply with RBI's new KYC rules. The central bank's "
        "February circular requires biometric verification for loans above Rs 50,000, "
        "affecting Lendingkart, Capital Float and ZestMoney. Compliance teams at these companies "
        "report they must overhaul core onboarding flows, retrain staff, and integrate new "
        "RegTech APIs within the tight deadline. Industry bodies FIDC and DLAI have requested "
        "a 180-day extension but RBI has so far not responded. The regulator indicated that "
        "lenders which miss the deadline face suspension of new loan origination.",
        "Livemint", "regulation",
    ),
    (
        "RBI Tightens KYC: NBFCs Must Submit Compliance Plans by March",
        "The regulator expects all NBFCs to file a remediation plan by March 31. Failure to "
        "comply risks suspension of new loan disbursal. RegTech vendors report a 3x spike in "
        "inquiries since the RBI circular landed. Companies such as Signzy and IDfy are seeing "
        "their inbound pipeline triple as digital lenders scramble to partner with compliant "
        "identity verification providers. Mid-size NBFCs like Mahindra Finance and Manappuram "
        "Finance are also reviewing whether their existing video-KYC setups meet the new bar. "
        "Compliance technology investment is expected to surpass Rs 200 crore industry-wide.",
        "Business Standard", "regulation",
    ),
    (
        "RBI Governor Warns on KYC Compliance Gaps at Digital Lenders",
        "Sanjay Malhotra flagged compliance gaps at the Annual NBFC Summit. RBI has issued "
        "show-cause notices to 18 lenders. Mid-size companies are most exposed given limited "
        "internal compliance teams. The governor cited specific failures at three unnamed "
        "digital lenders where facial-recognition liveness checks were being bypassed. "
        "KFin Technologies and Karvy Fintech, which provide back-office processing for many "
        "NBFCs, have begun offering bundled compliance audit services. The RBI circular also "
        "affects buy-now-pay-later providers such as LazyPay and Simpl.",
        "Hindu BusinessLine", "regulation",
    ),
    (
        "Fintech Lenders Rush to Hire Compliance Heads After RBI Circular",
        "Job postings for Chief Compliance Officers surged 45% on LinkedIn after the RBI mandate. "
        "The 90-day window is too short for companies without compliance infrastructure, "
        "analysts warn. Lendingkart has already posted for a VP Compliance role at a "
        "Rs 60-80 lakh CTC band — a salary level previously unusual for mid-size NBFCs. "
        "Consulting firms Deloitte, EY and PwC have created dedicated NBFC compliance "
        "practices to service the sudden demand. The talent crunch is particularly acute "
        "outside Mumbai, where smaller regional digital lenders operate.",
        "Inc42", "regulation",
    ),
    (
        "RegTech Startups See 3x Demand as RBI KYC Deadline Looms",
        "Identity verification firms Signzy and IDfy are seeing record inbound interest. "
        "Signzy CEO says their pipeline has tripled since the RBI circular. "
        "Compliance tech investment is surging. IDfy's video-KYC product now serves over "
        "120 NBFC clients, up from 40 a year ago. Meanwhile, Perfios, which provides "
        "bank-statement analytics for lenders, reports that KYC workflow integrations now "
        "account for 35% of new client requests. The RegTech sector is expected to attract "
        "Rs 1,500 crore in VC funding in 2026, double last year's figure.",
        "YourStory", "regulation",
    ),

    # ── Cluster 2: Quick commerce funding (6 articles) ───────────────────────
    (
        "Zepto Raises $200M at $5B Valuation for Dark Store Expansion",
        "Quick commerce startup Zepto closed a $200 million Series F at $5 billion valuation. "
        "Founders plan to add 200 dark stores in Tier-2 cities within six months, "
        "challenging Blinkit and Swiggy Instamart. The round was co-led by General Atlantic "
        "and Nexus Venture Partners. Zepto CEO Aadit Palicha said the company targets "
        "profitability at the contribution margin level by Q3 2026. The fresh capital will also "
        "fund a B2B grocery delivery pilot targeting restaurant chains and cloud kitchens "
        "in Mumbai and Delhi NCR.",
        "Economic Times", "funding",
    ),
    (
        "Zepto Plans 50 New Cities, Challenging Blinkit and Instamart",
        "Armed with fresh capital, Zepto is entering Nagpur, Coimbatore and Kochi. "
        "CEO says they are hiring 5,000 dark store staff. Blinkit has responded by announcing "
        "300 new stores to defend market share. Swiggy Instamart is accelerating its own "
        "dark store rollout in Tier-2 markets. Logistics providers Delhivery and Xpressbees "
        "are competing for last-mile delivery partnerships with all three platforms. "
        "Cold-chain operator Snowman Logistics reports a 40% jump in dark store inquiries "
        "as quick commerce companies seek to expand their perishable inventory range.",
        "Livemint", "funding",
    ),
    (
        "Quick Commerce Heats Up: Funding Rounds Total $1B in 2025",
        "Aggregate VC investment into Indian quick commerce surpassed $1 billion this year. "
        "Companies without scale will face consolidation pressure by 2026 as "
        "10-minute delivery becomes table stakes. Dunzo, which struggled to maintain capital, "
        "is now partnering with Reliance Retail to survive. Magicpin and Ola Dash have "
        "retreated from quick commerce. Analysts at Bernstein estimate Blinkit, Zepto and "
        "Swiggy Instamart collectively control 87% of the market, leaving little room for "
        "new entrants without differentiated supply chains or proprietary dark store tech.",
        "Mint", "funding",
    ),
    (
        "Dark Store Landlords Demand 40% Premium as Quick Commerce Expands",
        "Real estate owners near metro areas are commanding 35-40% rental premium for "
        "dark store spaces. Cold storage and EV-charging capable facilities see highest demand. "
        "Anarock Property Consultants notes that quick commerce has displaced traditional "
        "retail as the primary driver of last-mile warehouse demand in cities like Bengaluru "
        "and Hyderabad. Zepto and Blinkit are also signing long-term 5-year leases to lock "
        "in prime locations, pushing up rental costs for traditional FMCG distributors who "
        "rely on the same micro-warehousing footprint.",
        "Hindu BusinessLine", "funding",
    ),
    (
        "Zepto Eyes B2B Deliveries to Restaurants After Series F Close",
        "Following the funding round, Zepto is piloting B2B supply deliveries to 300 cloud "
        "kitchen operators in Mumbai, positioning against Swiggy's Genie and Dunzo Daily. "
        "The company has hired Rohan Malhotra, formerly of BigBasket, to lead the B2B vertical. "
        "Swiggy has filed a counter with its own bulk-order delivery product targeting "
        "corporate cafeterias. Both platforms are courting ITC, HUL and Nestle as anchor FMCG "
        "partners for guaranteed SKU availability in dark stores.",
        "Business Standard", "funding",
    ),
    (
        "Quick Commerce Giants Locked in Price War on Grocery Margins",
        "Blinkit, Zepto and Swiggy Instamart are offering 20-30% discount events that compress "
        "category margins to -8% on average. Suppliers and cold-chain operators are alarmed "
        "by the unsustainability. Marico and Dabur have publicly warned that quick commerce "
        "platforms are forcing unsustainable trade margins that hurt brand equity. "
        "Industry body FICCI has urged DPIIT to frame guidelines for platform discounting "
        "practices. Meanwhile, Blinkit parent Zomato reported Rs 40 crore EBITDA loss from "
        "quick commerce operations in Q3 FY26.",
        "Economic Times", "funding",
    ),

    # ── Cluster 3: Semiconductor policy (5 articles) ─────────────────────────
    (
        "Cabinet Approves Rs 1.26 Lakh Crore for 3 Semiconductor Fabs",
        "The Union Cabinet cleared Rs 1,26,000 crore for three chip plants. Tata Electronics "
        "gets Dholera (28nm), Vedanta-Foxconn gets Pune, and ISMC sets up an analogue fab "
        "in Karnataka. The approvals mark a decisive shift in India's electronics manufacturing "
        "strategy, reducing dependence on Taiwan and China for critical semiconductor supply. "
        "MEITY secretary confirmed that the first wafers from Tata's Dholera plant are "
        "targeted for Q2 2027. CG Power, which holds a stake in the Sanand ATMP facility, "
        "saw its stock rise 12% on the announcement.",
        "PIB", "policy",
    ),
    (
        "India Semiconductor Mission Awards First Fab Contracts",
        "ISM CEO confirmed Tata and CG Power have signed final investment agreements. "
        "The fabs will produce 50,000 wafer starts per month at peak capacity by 2027. "
        "Renesas Electronics and NXP Semiconductors have expressed interest in anchor "
        "customer agreements with Tata's Dholera fab. The India Semiconductor Mission is also "
        "in talks with Applied Materials and ASML to establish equipment service centres "
        "in Gujarat to support the fab cluster. The government expects the chip ecosystem "
        "to directly employ 40,000 engineers by 2030.",
        "Economic Times", "policy",
    ),
    (
        "EMS Companies Scramble for Semiconductor Supply Chain Position",
        "Mid-size EMS firms are holding emergency board meetings to assess chip ecosystem entry. "
        "Syrma SGS and VVDN Technologies have announced dedicated semiconductor sub-assembly lines. "
        "Dixon Technologies, the largest Indian EMS player, has signed an MoU with Tata "
        "Electronics to assemble smartphone chipsets at its Noida facility. Amber Enterprises "
        "is similarly exploring ATMP tie-ups. Industry analysts note that EMS companies with "
        "existing surface-mount technology lines are best positioned to pivot into chip "
        "packaging and test work as India's semiconductor ecosystem matures.",
        "Financial Express", "policy",
    ),
    (
        "India's PLI Scheme Attracts 18 Chipmakers: MEITY Report",
        "MEITY's report shows 18 foreign chipmakers have expressed intent to set up design "
        "centres or ATMP facilities under PLI. Assembly, test, mark, pack capacity is seen "
        "as the near-term opportunity. US-based Microchip Technology has confirmed plans for "
        "a design centre in Bengaluru. Qualcomm and MediaTek are evaluating ATMP investments "
        "in partnership with Indian EMS companies. The PLI outlay for semiconductors was "
        "increased by Rs 10,000 crore in the Union Budget, signalling sustained policy commitment.",
        "Business Standard", "policy",
    ),
    (
        "Semiconductor Workforce Gap: India Needs 85,000 Engineers by 2026",
        "NASSCOM estimates India's semiconductor sector needs 85,000 additional engineers. "
        "Only 12,000 graduate annually with chip-relevant skills. IITs and bootcamp providers "
        "are launching crash programmes. Imec India, the research arm of Belgium's Imec, "
        "has partnered with IIT Bombay and IIT Madras to offer 6-month chip design "
        "certification courses. Tata Electronics has committed Rs 500 crore to fund 10,000 "
        "VLSI training seats at partner engineering colleges by 2027. NASSCOM warns that "
        "without workforce scale-up the fab ramp will be bottlenecked by talent shortages.",
        "Mint", "policy",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Company-First dataset — IT services / consulting companies
# Cluster 1: TCS / Infosys / Wipro AI transformation
# Cluster 2: HCL Tech / Tech Mahindra cloud migration
# Cluster 3: Infosys / Wipro / Cognizant restructuring 2026
# ──────────────────────────────────────────────────────────────────────────────

MOCK_ARTICLES_COMPANY_FIRST = [
    # ── Cluster 1: IT services companies adopting AI (6 articles) ────────────
    (
        "TCS and Infosys Win Combined $3B in GenAI Transformation Contracts",
        "Tata Consultancy Services and Infosys have collectively secured over $3 billion in "
        "generative AI transformation contracts in the first quarter of 2026, according to "
        "earnings disclosures from both companies. TCS's GenAI-linked deal wins accounted for "
        "18% of total new contract value, up from 9% a year ago. Infosys's Topaz AI platform "
        "contributed to 22 large enterprise engagements worth over $50 million each. "
        "Both companies are rapidly reskilling workforces — TCS has trained 375,000 employees "
        "on AI tools, while Infosys has upskilled 280,000 through its internal Lex platform. "
        "Analysts at Kotak expect GenAI to add 200-300 basis points to revenue growth for "
        "top-tier IT firms over the next two years.",
        "Economic Times", "technology",
    ),
    (
        "Infosys Topaz Platform Lands JPMorgan and BP as Anchor Clients",
        "Infosys announced that JPMorgan Chase and BP have signed multi-year contracts to "
        "deploy the Topaz AI platform for risk analytics and energy trading optimisation "
        "respectively. The JPMorgan engagement, valued at approximately $400 million over "
        "five years, involves deploying large language models on proprietary financial data. "
        "BP will use Infosys Topaz to improve predictive maintenance across 12 refineries. "
        "Wipro has responded by acquiring AI startup Aggne to bolster its own financial "
        "services AI practice. TCS has separately announced a joint innovation lab with "
        "Microsoft Azure to co-develop industry-specific LLM solutions for the banking, "
        "insurance and manufacturing verticals. Cognizant is also accelerating Topaz "
        "competitor investments following the announcement.",
        "Livemint", "technology",
    ),
    (
        "Wipro and TCS Race to Build Proprietary LLM Stacks for Enterprises",
        "Wipro has launched WiproCo, a proprietary large language model fine-tuned on "
        "manufacturing, pharma and financial services data. TCS is testing its AI companion "
        "platform internally before a planned enterprise rollout in Q2 2026. Infosys, which "
        "moved earlier with Topaz, now claims 190 active GenAI engagements. Industry analysts "
        "at NASSCOM note that Indian IT majors are shifting from staff-augmentation to "
        "outcome-based AI contracts, compressing headcount growth but expanding margins. "
        "HCL Technologies has partnered with Google DeepMind to adapt Gemini models for "
        "enterprise workflows in banking and insurance. Mphasis and Hexaware are watching "
        "closely and considering similar proprietary model investments to stay competitive "
        "with the larger players in GenAI enterprise services.",
        "Hindu BusinessLine", "technology",
    ),
    (
        "Mid-Size IT Firms Struggle to Match TCS and Infosys AI Investments",
        "While Infosys and TCS pour billions into AI capability building, mid-size Indian IT "
        "companies such as Mphasis, Hexaware and NIIT Technologies are finding it harder to "
        "compete for large GenAI transformation mandates. Mphasis CEO Nitin Rakesh acknowledged "
        "at a CII event that the company is partnering with hyperscalers rather than building "
        "proprietary stacks, a strategy that limits margin upside but reduces capital risk. "
        "Hexaware, which went public in February 2026, has allocated Rs 500 crore to AI "
        "capability development but analysts question whether it is sufficient to remain "
        "competitive with Infosys and Wipro on large banking transformation deals. "
        "Cognizant, sitting between the two size tiers, is betting its vertical AI strategy "
        "will allow it to compete against larger peers without matching their absolute "
        "investment levels.",
        "Business Standard", "technology",
    ),
    (
        "Cognizant Bets on Vertical AI to Challenge TCS and Infosys",
        "Cognizant has reorganised its delivery model around five vertical AI centres of "
        "excellence — banking, healthcare, retail, manufacturing and communications — in a "
        "direct challenge to TCS and Infosys's horizontal GenAI offerings. CEO Ravi Kumar S "
        "said vertical specialisation allows Cognizant to reach faster time-to-value for "
        "clients. The company has acquired two healthcare AI startups in the past six months "
        "and is hiring 8,000 AI engineers in India and the US. TCS and Infosys are watching "
        "the strategy closely; both are exploring similar vertical focus areas within their "
        "existing delivery units. Wipro's vertical industry strategy predates Cognizant's "
        "reorganisation by two years, suggesting the model has gained validation across "
        "multiple large Indian IT services companies.",
        "Inc42", "technology",
    ),
    (
        "AI Adoption Forces TCS, Wipro and Infosys to Rethink Bench Strategies",
        "The shift toward GenAI-driven project delivery is creating new workforce pressures "
        "at India's largest IT services firms. TCS, Wipro and Infosys are all reducing "
        "their traditional bench strength — the pool of undeployed employees that historically "
        "buffered project ramp-ups. Instead, AI-augmented delivery teams complete work faster "
        "with fewer people. Infosys's attrition rate fell to 12.1% as the company made fewer "
        "lateral hires. Wipro reported its bench as a percentage of headcount dropped from "
        "14% to 9% year-on-year. Industry observers warn that this efficiency gain is "
        "fundamentally reshaping how the $250B Indian IT services industry prices contracts. "
        "HCL Technologies and Cognizant are similarly reporting lower bench ratios, "
        "confirming the trend extends beyond the top two and represents a structural shift "
        "in how Indian IT companies manage their human capital.",
        "YourStory", "technology",
    ),

    # ── Cluster 2: HCL Tech / Tech Mahindra cloud migration (5 articles) ─────
    (
        "HCL Technologies Wins $1.2B Cloud Migration Mandate from Deutsche Bank",
        "HCL Technologies has been awarded a $1.2 billion, seven-year contract to migrate "
        "Deutsche Bank's core banking workloads from on-premises mainframes to a hybrid "
        "cloud architecture on Microsoft Azure. The deal is HCL's largest single cloud "
        "migration engagement to date and will involve over 3,500 HCL engineers in India, "
        "Germany and the UK. Tech Mahindra, which was also shortlisted, confirmed it is "
        "in advanced negotiations for a similar mandate with a European insurance group. "
        "Both companies are positioning cloud migration as a counter-cycle revenue driver "
        "as discretionary IT spending softens in North America. Wipro and Infosys have "
        "also signalled intent to grow their cloud migration practices in Europe to offset "
        "slower growth in the US financial services vertical.",
        "Economic Times", "technology",
    ),
    (
        "Tech Mahindra and HCL Battle for $800M European Cloud Outsourcing Market",
        "Tech Mahindra and HCL Technologies are competing for a cluster of European "
        "cloud-outsourcing contracts worth an estimated $800 million, according to people "
        "familiar with the matter. Tech Mahindra has strengthened its SAP S/4HANA migration "
        "practice by acquiring Germany-based Thinksoft Global and is leveraging this to "
        "pitch to mid-size European manufacturers. HCL has countered by forming a strategic "
        "alliance with AWS to offer pre-validated cloud migration blueprints for regulated "
        "industries including financial services and pharmaceuticals. Wipro has separately "
        "announced a cloud migration centre of excellence in Amsterdam. Infosys is also "
        "targeting European financial services cloud migration as part of its next growth "
        "phase, making the European market highly contested among Indian IT majors.",
        "Livemint", "technology",
    ),
    (
        "Enterprise Cloud Migration Spend to Hit $18B in India by 2028: Nasscom",
        "A new Nasscom report projects enterprise cloud migration spending in India will "
        "reach $18 billion by 2028 as large banks, insurers and manufacturers modernise "
        "legacy systems. HCL Technologies, Infosys and Tech Mahindra are the top three "
        "preferred partners for cloud migration among Indian enterprise CIOs surveyed. "
        "The report identifies public sector banks and state-owned manufacturing companies "
        "as the next major wave of cloud migration clients, creating opportunities for "
        "mid-size IT services firms that have deep government relationships. "
        "Capacity constraints — particularly for certified cloud architects — are flagged "
        "as the primary risk to delivery timelines. Wipro and Cognizant are both expanding "
        "their certified cloud architect training programmes specifically to address this "
        "capacity bottleneck as they compete for the next wave of large enterprise "
        "migration mandates.",
        "Hindu BusinessLine", "technology",
    ),
    (
        "HCL Tech Q3 Results: Cloud Revenue Grows 34% as Migration Deals Scale",
        "HCL Technologies reported a 34% year-on-year growth in cloud services revenue for "
        "Q3 FY26, driven by ramp-up of large migration deals including Deutsche Bank and "
        "Boeing. CEO C Vijayakumar said the company added 12 new cloud transformation "
        "clients this quarter, with average deal sizes increasing to $85 million from "
        "$60 million a year ago. Tech Mahindra's cloud revenue grew 19% in the same period, "
        "slower than HCL but accelerating from 11% in Q2. Infosys also beat cloud guidance "
        "as its Cobalt cloud platform reached $1.5 billion in annual revenue run-rate. "
        "Wipro trailed behind with 14% cloud revenue growth, prompting analysts to "
        "question whether its cloud practice restructuring under the new CEO is translating "
        "into competitive deal wins.",
        "Business Standard", "technology",
    ),
    (
        "SMB Cloud Migration Surge Creates Opening for Mid-Tier IT Vendors",
        "Small and medium businesses are increasingly turning to cloud migration to reduce "
        "server maintenance costs, creating a market that large IT firms like TCS and "
        "Infosys find too fragmented to serve efficiently. Mid-size players including "
        "Persistent Systems, Mphasis and L&T Technology Services are positioning for this "
        "segment. Persistent Systems has launched a Rs 100 crore fund specifically to "
        "subsidise cloud migration assessments for SMBs in manufacturing and pharma. "
        "The company estimates 40,000 Indian SMBs will migrate core workloads in 2026, "
        "representing a Rs 12,000 crore addressable market for IT services vendors. "
        "Hexaware has also targeted the SMB segment with fixed-price cloud migration "
        "packages, a model that removes the open-ended billing risk that many smaller "
        "companies cite as a barrier to engaging IT services firms.",
        "Mint", "technology",
    ),

    # ── Cluster 3: Infosys / Wipro / Cognizant restructuring 2026 (6 articles)
    (
        "Infosys Announces 10,000-Role Restructuring, Shifts Mix Toward AI Engineers",
        "Infosys has announced a workforce restructuring programme that will affect up to "
        "10,000 roles over the next 12 months as the company rebalances its workforce toward "
        "AI engineering and away from traditional application maintenance. The company will "
        "reduce headcount in low-value testing and maintenance roles while adding 15,000 "
        "AI engineers and data scientists through a combination of hiring and reskilling. "
        "Wipro is expected to announce a similar programme before the end of the fiscal year. "
        "Cognizant, which underwent a large restructuring in 2023, is now expanding headcount "
        "selectively in high-value AI and cloud practices after completing its earlier cuts. "
        "TCS and HCL Technologies have so far avoided formal restructuring announcements, "
        "preferring to manage headcount through natural attrition and redeployment.",
        "Economic Times", "technology",
    ),
    (
        "Wipro Cuts 6,000 Roles in Consulting Pyramid Reshape",
        "Wipro has laid off approximately 6,000 employees across its consulting and project "
        "management layers as part of CEO Srinivas Pallia's effort to flatten the delivery "
        "pyramid and improve margins. The cuts are concentrated in mid-level project manager "
        "and account management roles where AI-driven project tracking tools have reduced "
        "the need for human oversight. Infosys and TCS are closely watching the experiment; "
        "if Wipro achieves its target of 200 basis points of margin improvement, it will "
        "accelerate similar restructuring at competitors. The affected employees will receive "
        "a 6-month severance and access to an internal reskilling programme in cloud and AI. "
        "HCL Technologies and Cognizant are also evaluating restructuring options in their "
        "own management layers, though neither has made a public announcement.",
        "Livemint", "technology",
    ),
    (
        "Cognizant's Workforce Overhaul: 8,000 Hired in AI, 5,000 Exit in Legacy Roles",
        "Cognizant reported that it hired 8,000 employees in AI, data engineering and "
        "cloud specialisations in 2025 while allowing 5,000 legacy application support roles "
        "to run off through natural attrition and selective exits. CEO Ravi Kumar S said the "
        "company's 'invest and divest' talent strategy is designed to change its revenue "
        "mix from 60% legacy IT to 60% digital within three years. TCS has taken a different "
        "approach, using internal reskilling programmes to convert legacy staff to AI roles "
        "rather than net reductions. Infosys sits between the two — cutting some roles but "
        "prioritising retraining for its largest delivery workforce. Wipro's more aggressive "
        "restructuring has put pressure on Cognizant to accelerate its own transition "
        "timeline to avoid being perceived as slower to modernise than its closest peer.",
        "Hindu BusinessLine", "technology",
    ),
    (
        "IT Sector Restructuring Sends Talent to Product Companies and Startups",
        "The wave of restructuring at Infosys, Wipro and Cognizant is creating a talent "
        "windfall for Indian product startups and global technology companies. Bengaluru-based "
        "B2B SaaS startups report a surge in applications from mid-level IT services "
        "professionals with 8-15 years of experience. Razorpay, Chargebee and Freshworks "
        "each added over 200 experienced engineers from services firms in Q4 2025. "
        "TCS and HCL Technologies, which have not announced formal restructuring programmes, "
        "are absorbing some of the talent but at lower volumes than in previous hiring cycles. "
        "Talent management consultants note that lateral salary expectations from IT services "
        "veterans are calibrated to large-company pay scales, creating negotiation friction "
        "with venture-backed startups that prefer equity-heavy compensation packages.",
        "Business Standard", "technology",
    ),
    (
        "Automation Tools Accelerate Restructuring Pace at Wipro and Infosys",
        "Internal productivity data at Wipro and Infosys shows that AI-powered code review, "
        "test automation and documentation tools have reduced the engineering hours needed "
        "per project by 20-35% compared to 2023 baselines. This productivity gain is the "
        "direct driver behind the workforce restructuring both companies are undertaking. "
        "Wipro's internal Bengaluru Centre of Excellence for Automation has benchmarked that "
        "a team of 100 engineers with AI tooling delivers output equivalent to a traditional "
        "team of 130-140. The implication for clients is faster delivery; the implication "
        "for the industry is structural headcount pressure that will persist for several years. "
        "TCS and HCL Technologies report similar productivity gains but are absorbing the "
        "efficiency dividend through higher margins rather than headcount cuts, a strategic "
        "choice that diverges from Wipro and Infosys.",
        "Inc42", "technology",
    ),
    (
        "Nasscom Warns Restructuring Wave Could Suppress IT Freshers Hiring in 2026",
        "Nasscom's latest workforce report warns that the simultaneous restructuring at "
        "Infosys, Wipro and Cognizant will suppress fresher hiring across the Indian IT "
        "industry in 2026 for the first time since 2009. TCS has already signalled it will "
        "hire 40,000 freshers in FY27, down from 60,000 in FY26. Wipro and Infosys have "
        "not committed to fresher targets for the year. Engineering colleges that rely on "
        "IT services placements to post high placement statistics are scrambling to "
        "develop product company partnerships. Nasscom is lobbying the government to create "
        "a transition fund for retrenched IT workers. HCL Technologies and Tech Mahindra "
        "have indicated they will maintain fresher intake at existing levels, providing "
        "partial relief to campuses heavily exposed to the IT services recruitment pipeline.",
        "YourStory", "technology",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Report-Driven dataset — 5G / IoT / telecom theme
# Cluster 1: 5G rollout and enterprise IoT adoption in India
# Cluster 2: Airtel / Jio / BSNL 5G B2B contracts
# Cluster 3: IoT device management platform funding
# ──────────────────────────────────────────────────────────────────────────────

MOCK_ARTICLES_REPORT_DRIVEN = [
    # ── Cluster 1: 5G rollout and enterprise IoT adoption (6 articles) ────────
    (
        "India's 5G Enterprise Rollout Reaches 80 Cities, IoT Pilots Scale Up",
        "India's 5G enterprise coverage has expanded to 80 cities in early 2026, with Jio "
        "and Airtel both announcing accelerated rollouts targeting industrial corridors in "
        "Gujarat, Maharashtra and Tamil Nadu. Enterprise IoT adoption on 5G networks is "
        "accelerating in manufacturing, logistics and healthcare verticals. Siemens India "
        "has deployed a 5G-connected robotic assembly line at its Aurangabad plant that "
        "processes sensor data from 2,000 IoT endpoints in real time. Bosch India and "
        "Mahindra Logistics are separately piloting 5G-enabled asset tracking across "
        "their respective supply chains. The TRAI estimates enterprise 5G services "
        "will generate Rs 45,000 crore in annual revenue by 2028. HCL Technologies and "
        "Wipro have both launched dedicated 5G IoT practice groups targeting this market.",
        "Economic Times", "technology",
    ),
    (
        "Manufacturing Sector Drives 60% of Enterprise 5G Demand in India",
        "A Deloitte survey of 250 Indian enterprises found that the manufacturing sector "
        "accounts for 60% of enterprise 5G pilot demand, followed by logistics at 18% and "
        "healthcare at 11%. The survey found that Jio and Airtel are the dominant 5G "
        "enterprise providers, with BSNL gaining ground in government and defence verticals. "
        "Key use cases driving adoption include: automated guided vehicles in warehouses, "
        "predictive maintenance on factory floors, and drone-based inventory inspection. "
        "Tata Steel, Jindal Steel and JSW Steel are among the large manufacturers "
        "running advanced 5G IoT pilots, while mid-size auto-component makers are "
        "beginning to explore smaller-scale 5G private network deployments.",
        "Hindu BusinessLine", "technology",
    ),
    (
        "5G Private Networks Enable Real-Time IoT on Factory Floors",
        "Private 5G networks are emerging as a transformative technology for Indian "
        "manufacturers seeking ultra-low latency IoT connectivity on factory floors. "
        "Nokia and Ericsson are competing with Jio and Airtel's enterprise 5G offerings "
        "to provide private network infrastructure to large industrials. Larsen and Toubro's "
        "precision engineering division has deployed a Nokia private 5G network at its "
        "Pune facility, connecting 500 CNC machines and 300 robotic arms to a central "
        "AI analytics platform. Honeywell India is marketing its enterprise IoT middleware "
        "stack specifically for Indian manufacturers transitioning to private 5G. "
        "The private 5G network market in India is expected to reach Rs 8,000 crore by 2027.",
        "Business Standard", "technology",
    ),
    (
        "Healthcare IoT Gains Momentum as Apollo, Fortis Deploy 5G-Connected Devices",
        "Apollo Hospitals and Fortis Healthcare are deploying 5G-connected IoT devices "
        "across their hospital networks to enable real-time patient monitoring and "
        "predictive equipment maintenance. Apollo's partnership with Jio involves connecting "
        "over 15,000 IoT-enabled medical devices across 40 hospitals to a cloud analytics "
        "platform. Fortis has chosen Airtel's enterprise 5G solution for a similar deployment "
        "at its 22 hospitals. Philips India and GE HealthCare India are supplying the "
        "connected medical devices and associated cloud analytics. The healthcare IoT segment "
        "is growing at 38% annually in India, driven by regulatory pressure for real-time "
        "equipment calibration records and post-pandemic interest in remote patient monitoring.",
        "Livemint", "technology",
    ),
    (
        "5G-Enabled Smart Logistics Hubs Proliferate Along Dedicated Freight Corridors",
        "Logistics parks along India's Dedicated Freight Corridors are being upgraded with "
        "5G-enabled IoT infrastructure, creating smart logistics hubs that can track "
        "shipments in real time and automate warehouse operations. Mahindra Logistics, "
        "Delhivery and BlueDart are the most active deployers of 5G IoT at their mega "
        "logistics hubs. Each company is using Jio's enterprise 5G connectivity to link "
        "warehouse management systems, automated conveyor belts and fleet telematics into "
        "a single operational dashboard. The Ministry of Commerce estimates that 5G-enabled "
        "logistics optimisation could reduce India's logistics costs from 13% of GDP to "
        "below 10% by 2030. Airtel is competing with Jio for the next tranche of logistics "
        "hub 5G contracts by bundling managed security and SLA-backed uptime guarantees "
        "that Jio's standard enterprise plans currently do not include.",
        "Mint", "technology",
    ),
    (
        "IoT Sensor Costs Fall 40%, Triggering Broad 5G Adoption Among SME Manufacturers",
        "The average cost of an industrial IoT sensor has fallen 40% over the past two years, "
        "bringing 5G-connected factory automation within reach of small and medium "
        "manufacturers for the first time. Mid-size auto-component makers such as Endurance "
        "Technologies and Sundram Fasteners are deploying 5G IoT pilots at their plants. "
        "Airtel Business reports that SME manufacturers now account for 25% of its enterprise "
        "5G enquiries, up from under 5% in 2024. Technology solution providers like Tata "
        "Elxsi and Wipro are packaging 5G IoT deployments as managed services for mid-size "
        "manufacturers, reducing the upfront capital requirement. Jio is responding with "
        "a subsidised device programme that provides free IoT sensor hardware in exchange "
        "for multi-year 5G connectivity contracts, targeting the same SME manufacturing "
        "segment.",
        "Inc42", "technology",
    ),

    # ── Cluster 2: Airtel / Jio / BSNL 5G B2B contracts (5 articles) ─────────
    (
        "Airtel Wins Rs 2,400 Crore B2B 5G Contract from Maruti Suzuki",
        "Airtel Business has signed a five-year Rs 2,400 crore contract to deploy private "
        "5G connectivity and IoT management services across Maruti Suzuki's three "
        "manufacturing plants in Haryana and Gujarat. The contract covers private network "
        "infrastructure, SIM management for over 30,000 connected machines and a managed "
        "security layer to protect operational technology systems from cyber threats. "
        "Jio had also been shortlisted but Maruti chose Airtel for its track record in "
        "large enterprise network management. This win positions Airtel as the leading "
        "5G B2B provider for automotive manufacturing, a sector where it already holds "
        "contracts with Tata Motors and Bajaj Auto. Infosys is the system integration "
        "partner for the Airtel 5G deployment at Maruti's Gurugram facility.",
        "Economic Times", "technology",
    ),
    (
        "Jio Secures 5G IoT Deals with Indian Railways and Port Trust",
        "Reliance Jio has announced 5G IoT contracts with Indian Railways and the Jawaharlal "
        "Nehru Port Trust to modernise tracking and operations systems. The Indian Railways "
        "contract, valued at Rs 1,800 crore over seven years, will deploy Jio's 5G network "
        "across 50 railway yards for real-time freight wagon tracking and predictive "
        "maintenance of locomotives. The JNPT contract involves 5G-connected autonomous "
        "straddle carriers and IoT sensors across the port's container handling equipment. "
        "Airtel and Nokia are watching these wins closely as indicators that Jio is "
        "targeting government infrastructure as a 5G B2B anchor segment. TCS and HCL "
        "Technologies, which serve as IT partners to Indian Railways, are expected to "
        "integrate Jio's 5G IoT layer into their existing railway management system "
        "contracts.",
        "Hindu BusinessLine", "technology",
    ),
    (
        "BSNL Pivots to Enterprise 5G After Consumer Market Setbacks",
        "State-owned BSNL, which has struggled to compete with Jio and Airtel in the "
        "consumer mobile market, is pivoting to enterprise 5G as a revenue recovery strategy. "
        "BSNL has signed MoUs with DRDO, HAL and other defence-linked organisations to "
        "provide private 5G connectivity for sensitive facilities that cannot use foreign "
        "network equipment. The company is deploying TCS-manufactured 5G base stations "
        "under the Atma Nirbhar programme to create an end-to-end indigenous 5G network. "
        "BSNL's enterprise 5G chief Shyam Sunda Taneja told reporters that the defence "
        "and smart city verticals will account for 60% of BSNL's 5G revenue in FY27. "
        "HCL Technologies is supporting BSNL's OSS/BSS modernisation as part of the "
        "same transformation programme, providing network operations software to manage "
        "the new 5G infrastructure.",
        "Business Standard", "technology",
    ),
    (
        "Airtel, Jio Clash Over Rs 5,000 Crore Smart City 5G IoT Tenders",
        "Airtel and Jio are competing aggressively for a cluster of smart city 5G IoT "
        "tenders floated by state governments in Uttar Pradesh, Rajasthan and Karnataka, "
        "collectively valued at Rs 5,000 crore. The tenders cover street lighting automation, "
        "traffic management, CCTV surveillance and municipal utility monitoring using 5G "
        "connected IoT sensors. Both operators have formed consortia with IT services "
        "companies — Airtel with HCL Technologies and Jio with TCS — to offer integrated "
        "5G network plus software platform solutions. BSNL is also bidding for the UP "
        "smart city contract on price and indigenisation grounds. Wipro and Infosys are "
        "offering independent software platform bids for the Karnataka smart city tender, "
        "targeting the application layer without committing to full network infrastructure "
        "ownership.",
        "Livemint", "technology",
    ),
    (
        "Vodafone Idea's 5G Delay Cedes Enterprise Contracts to Jio and Airtel",
        "Vodafone Idea's inability to launch 5G commercially in most markets is costing it "
        "significant enterprise contracts that are going to Jio and Airtel instead. "
        "Industry estimates suggest Vi has lost Rs 800-1,200 crore in potential 5G "
        "enterprise annual recurring revenue in the past 12 months. Tata Communications, "
        "which resells capacity from Jio and Airtel, is picking up some displaced Vi "
        "enterprise customers but is constrained by its own 5G infrastructure investments. "
        "Analysts at ICICI Securities warn that if Vi cannot close its 5G gap within "
        "18 months it may be forced to merge with BSNL or exit enterprise 5G entirely. "
        "Ericsson, which supplies Airtel's 5G equipment and is also a candidate for Vi's "
        "delayed network modernisation, confirmed it is in discussions with Vi about "
        "an accelerated deployment timeline.",
        "Mint", "technology",
    ),

    # ── Cluster 3: IoT device management platform funding (6 articles) ────────
    (
        "IoT Platform Startup Datoin Raises Rs 180 Crore Series B for Industrial Deployments",
        "Datoin, a Pune-based industrial IoT platform company, has raised Rs 180 crore in "
        "a Series B round led by Sequoia India, with participation from Qualcomm Ventures. "
        "The company's platform manages over 2 million IoT device endpoints for clients "
        "including Tata Steel, Mahindra and Larsen and Toubro. Datoin plans to use the "
        "capital to expand its 5G-native device management capabilities and enter the "
        "Southeast Asian market. The funding follows a period of rapid revenue growth — "
        "Datoin's ARR tripled in 2025 to Rs 75 crore as enterprise IoT adoption accelerated. "
        "Competitors Exotel and Mlytics are also raising capital to capture the same market.",
        "YourStory", "funding",
    ),
    (
        "Sternum IoT Raises $45M to Secure Connected Devices on 5G Networks",
        "Israeli-Indian cybersecurity startup Sternum IoT has raised $45 million in a "
        "Series C round to expand its embedded security platform for industrial IoT devices. "
        "The platform prevents firmware tampering and rogue device injection on 5G-connected "
        "factory networks. Jio and Airtel have both integrated Sternum into their enterprise "
        "IoT management offerings as a value-added security layer. Customers including "
        "Honeywell India, Siemens India and Bosch India are deploying Sternum to meet "
        "new CERT-In requirements for OT network security. The Series C was led by "
        "Accel India with strategic investment from Qualcomm Ventures. The company plans "
        "to double its engineering headcount in Bengaluru and expand into Southeast Asian "
        "markets where 5G IoT deployments are similarly accelerating.",
        "Inc42", "funding",
    ),
    (
        "Altizon Systems Secures Strategic Investment from Rockwell Automation",
        "Pune-based industrial IoT platform company Altizon Systems has received a "
        "strategic equity investment from Rockwell Automation, valuing the company at "
        "approximately Rs 400 crore. Altizon's Datonis platform connects over 500,000 "
        "industrial assets across 200 manufacturing clients in India, Germany and the US. "
        "The Rockwell investment will deepen integration between Altizon's cloud IoT "
        "layer and Rockwell's PLC and edge computing hardware. Mahindra and Mahindra "
        "and Hindustan Unilever are among Altizon's Indian enterprise anchor clients. "
        "The investment signals growing interest from global industrial automation majors "
        "in Indian IoT platform companies as India's manufacturing sector modernises. "
        "Siemens and ABB, which also have large industrial automation businesses, are "
        "evaluating partnership arrangements with Indian IoT startups as a faster path "
        "to 5G-native product offerings than internal development.",
        "Economic Times", "funding",
    ),
    (
        "TATA Elxsi and Bosch Partner on Next-Gen IoT Edge Computing Platform",
        "Tata Elxsi and Bosch have announced a joint development programme to create an "
        "edge computing platform specifically designed for 5G-connected industrial IoT "
        "deployments. The platform will run AI inference workloads at the factory edge, "
        "reducing latency compared to cloud-only architectures. Initial deployments are "
        "targeted at automotive manufacturing plants, where real-time quality inspection "
        "requires sub-10ms decision latency that cloud-based systems cannot guarantee. "
        "Siemens India and ABB India are evaluating the joint platform as part of their "
        "own 5G IoT roadmaps. TCS has filed a response noting its own edge AI platform "
        "for manufacturing IoT, escalating competition in the segment. Airtel has expressed "
        "interest in bundling the Tata Elxsi-Bosch platform into its private 5G managed "
        "service for automotive clients, which would accelerate deployments at Maruti "
        "Suzuki and Tata Motors.",
        "Hindu BusinessLine", "technology",
    ),
    (
        "Entrib Raises Rs 60 Crore to Build 5G-First IoT Management Platform",
        "Bengaluru-based IoT startup Entrib has raised Rs 60 crore in a Pre-Series B round "
        "led by 3one4 Capital to build a 5G-native device management platform for Indian "
        "enterprises. Unlike legacy IoT platforms designed for 4G networks, Entrib's "
        "architecture is optimised for the network slicing and ultra-low latency features "
        "of 5G, enabling use cases like real-time robotic control and autonomous vehicle "
        "coordination. The company counts Delhivery, Flipkart's logistics arm and "
        "Mahindra Logistics as early customers. Jio has expressed interest in integrating "
        "Entrib's platform into its enterprise 5G IoT suite. Airtel is evaluating a similar "
        "white-label arrangement that would let Entrib power Airtel's managed IoT offering "
        "for the automotive and logistics verticals.",
        "Business Standard", "funding",
    ),
    (
        "Connx IoT Raises $12M Seed to Standardise Multi-Vendor 5G IoT Deployments",
        "Connx IoT, a Mumbai-based startup, has raised a $12 million seed round from "
        "Lightspeed India and Qualcomm Ventures to build a vendor-neutral middleware layer "
        "that connects IoT devices from different manufacturers into a unified 5G management "
        "plane. The platform addresses a key pain point for enterprise customers who deploy "
        "IoT sensors from Bosch, Siemens, Honeywell and others but lack a unified dashboard. "
        "Airtel and Tata Communications are evaluating Connx as a platform to bundle into "
        "their managed IoT service offerings. The company targets profitability within "
        "18 months as it scales to 1 million connected endpoints. Jio, which competes "
        "with Airtel for enterprise IoT market share, is also reported to be in early "
        "conversations with Connx about potential technology licensing arrangements.",
        "Mint", "funding",
    ),
]
