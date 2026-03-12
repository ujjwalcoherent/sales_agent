"""
Mock article data for pipeline testing (mock_mode=True).

Used by agents/source_intel.py when mock_mode is enabled.
Three tight clusters per use-case so clustering/synthesis/impact stages
get realistic, mode-appropriate input.

Datasets:
  MOCK_ARTICLES_RAW            — default / industry_first (fintech regulation [:12] + quick-commerce + semiconductor)
  MOCK_ARTICLES_COMPANY_FIRST  — IT services companies (TCS, Infosys, Wipro, HCL, etc.)
  MOCK_ARTICLES_REPORT_DRIVEN  — carrier rocket / space launch theme
  MOCK_REPORT_SUMMARY          — default report text for report_driven mock mode
"""

# ──────────────────────────────────────────────────────────────────────────────
# Default dataset — used for industry_first
# First 12 articles = fintech (KYC + UPI/BNPL) — sliced for fintech_bfsi mode
# Full 23 articles = mixed (fintech + quick-commerce + semiconductor)
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

    # ── Cluster 2: UPI / Digital Payments regulation (6 articles) ────────────
    (
        "NPCI Caps UPI Market Share at 30%, Forcing PhonePe and Google Pay to Slow",
        "The National Payments Corporation of India has enforced its 30% UPI market share cap, "
        "directly impacting PhonePe (48% share) and Google Pay (36% share). Mid-size fintech "
        "payment apps like Paytm Payments Bank, BharatPe and CRED must now decide whether to "
        "aggressively acquire the diverted transaction volume. NPCI's circular gives PhonePe and "
        "Google Pay until December 2026 to comply. Industry estimates suggest Rs 12,000 crore in "
        "annual transaction volume will shift to smaller UPI apps. The cap also creates an opening "
        "for bank-owned UPI apps like SBI YONO and HDFC PayZapp to regain market share.",
        "Economic Times", "regulation",
    ),
    (
        "Mid-Size Fintech Apps Race to Capture UPI Volume from Market Share Cap",
        "BharatPe, CRED and Freecharge are investing Rs 500 crore combined in merchant acquisition "
        "to capture the UPI volume that PhonePe and Google Pay must shed under NPCI's 30% cap. "
        "BharatPe CEO Suhail Sameer confirmed a target of 10 million new merchants by March 2027. "
        "Meanwhile, Razorpay and Cashfree are positioning their payment gateway services to help "
        "mid-size NBFCs and digital lenders build their own UPI collection infrastructure. The "
        "RBI has also signalled that UPI credit line products will be subject to the same KYC "
        "norms that currently apply to digital lending.",
        "Livemint", "regulation",
    ),
    (
        "RBI Digital Lending Guidelines Force Fintech Lenders to Restructure BNPL",
        "The Reserve Bank's updated digital lending guidelines now classify buy-now-pay-later "
        "products as credit instruments, requiring full NBFC registration. Mid-size BNPL providers "
        "like LazyPay, Simpl, and Uni Cards must restructure their lending partnerships or face "
        "enforcement action. Simpl has already pivoted to a co-lending model with Federal Bank. "
        "The guidelines also mandate that loan service providers (LSPs) display all-in interest "
        "rates prominently, affecting customer acquisition costs for digital lenders.",
        "Business Standard", "regulation",
    ),
    (
        "Fintech Lending Consolidation Accelerates as RBI Tightens Oversight",
        "Three mid-size digital lenders — ZestMoney, MoneyTap and EarlySalary — are in advanced "
        "merger talks following RBI's regulatory tightening. The combined entity would serve 8 "
        "million borrowers with an AUM of Rs 6,000 crore. Industry sources say compliance costs "
        "have risen 40% since the new KYC and BNPL regulations, making standalone operations "
        "unviable for lenders below Rs 2,000 crore AUM. Private equity firm Warburg Pincus is "
        "reportedly evaluating a controlling stake in the merged entity.",
        "Mint", "regulation",
    ),
    (
        "NBFC Microfinance Institutions Face Double Squeeze from RBI and RBI KYC",
        "Small-finance banks and NBFC-MFIs face simultaneous pressure from the RBI's revised "
        "microfinance pricing guidelines and the new KYC mandate. Institutions like CreditAccess "
        "Grameen, Spandana Sphoorty, and Arohan Financial must cap interest rates at cost-plus "
        "model while also upgrading KYC infrastructure. CreditAccess Grameen CEO noted that "
        "compliance costs now consume 2.8% of their AUM, up from 1.5% a year ago. The double "
        "regulatory burden has triggered a wave of branch rationalization among mid-size MFIs, "
        "with 200+ branches closed industry-wide in Q4 FY26.",
        "Hindu BusinessLine", "regulation",
    ),
    (
        "Digital Payment Fraud Rises 28%, Forcing Fintech Investment in AI Detection",
        "RBI's annual fraud report shows digital payment fraud rose 28% YoY to Rs 1,457 crore. "
        "UPI fraud alone accounted for Rs 573 crore. Mid-size fintech companies including "
        "Razorpay, Cashfree, and Juspay are investing Rs 300 crore combined in AI-based fraud "
        "detection systems. The RBI has mandated that all payment aggregators implement device "
        "fingerprinting and behavioral biometrics by September 2026. Signzy and Bureau (formerly "
        "Bureau.id) are seeing 4x demand for real-time transaction monitoring APIs from NBFCs.",
        "Inc42", "regulation",
    ),

    # ── Cluster 3: Quick commerce funding (6 articles) ───────────────────────
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
# Report-Driven dataset — Carrier Rocket / Space Launch theme
# Based on Coherent Market Insights "Carrier Rocket Market" report.
# Cluster 1: Reusable rocket development and cost reduction
# Cluster 2: Small satellite constellation launches
# Cluster 3: Space launch startup funding and partnerships
# ──────────────────────────────────────────────────────────────────────────────

# Default report summary for mock mode and frontend demo.
# Condensed from the Coherent Market Insights Carrier Rocket Market report.
MOCK_REPORT_SUMMARY = (
    "The Global Carrier Rocket Market is estimated to be valued at USD 18.73 Bn in 2026 "
    "and is expected to reach USD 35.58 Bn by 2033, exhibiting a CAGR of 9.6%. Liquid rocket "
    "engines dominate with 37.5% share due to superior throttling and restart capabilities. "
    "Satellite deployment accounts for 60.6% of launch demand, driven by LEO communication "
    "constellations and earth observation networks. North America leads with 37.8% share, "
    "anchored by SpaceX, Boeing, and Lockheed Martin, while Asia Pacific (18.1%) is the "
    "fastest-growing region with CNSA and ISRO expanding launch cadence. Key market trends "
    "include reusable rocket development reducing launch costs by 40-60%, AI-driven trajectory "
    "optimization improving fuel efficiency, and 3D-printed rocket components cutting "
    "production time. Relativity Space is 3D-printing over 85% of its Terran R vehicle. "
    "SpaceX's Falcon 9 and Starship use AI for real-time landing decisions. Rocket Lab's "
    "Neutron targets the medium-lift reusable segment. The rise of private-public partnerships "
    "and the growing small satellite market are creating opportunities for new entrants "
    "in both launch services and ground systems infrastructure."
)

MOCK_ARTICLES_REPORT_DRIVEN = [
    # ── Cluster 1: Reusable rocket development and cost reduction (6 articles) ─
    (
        "SpaceX Starship Achieves Full Reusability Milestone, Cuts Launch Cost to $15M",
        "SpaceX has successfully recovered both the Super Heavy booster and Starship upper "
        "stage for the third consecutive flight, marking full operational reusability. CEO "
        "Elon Musk confirmed the cost per Starship launch has dropped to approximately "
        "$15 million, down from $90 million for expendable variants. The milestone positions "
        "SpaceX to dominate the heavy-lift market for satellite constellation deployment "
        "and NASA's Artemis lunar cargo missions. Boeing and Lockheed Martin's ULA joint "
        "venture is accelerating development of Vulcan Centaur's reusable engine pod in "
        "response. Analysts at Morgan Stanley estimate SpaceX now controls 62% of global "
        "commercial launch revenue, up from 48% in 2024.",
        "Reuters", "technology",
    ),
    (
        "Relativity Space Completes First Terran R Flight with 85% 3D-Printed Components",
        "Relativity Space has completed the inaugural orbital flight of its Terran R rocket, "
        "which uses additive manufacturing for over 85% of its structural components. The "
        "Long Beach-based company demonstrated that 3D-printed rocket engines and fuel tanks "
        "can withstand the thermal and mechanical stresses of orbital insertion. Production "
        "time for Terran R is 60 days from raw material to launch-ready vehicle, compared "
        "to 18-24 months for traditionally manufactured rockets. The U.S. Space Force has "
        "awarded Relativity a $1.2 billion contract for national security launches through "
        "2030. Boeing's space division is evaluating Relativity's Stargate 3D printer "
        "technology for potential use in its own satellite manufacturing processes.",
        "SpaceNews", "technology",
    ),
    (
        "Rocket Lab Unveils Neutron Medium-Lift Vehicle with Reusable First Stage",
        "Rocket Lab has unveiled the production version of its Neutron medium-lift launch "
        "vehicle, designed to carry 13,000 kg to low Earth orbit with a reusable first stage. "
        "CEO Peter Beck confirmed that Neutron targets the gap between SpaceX's Falcon 9 and "
        "smaller dedicated smallsat launchers. The vehicle uses Archimedes engines running on "
        "liquid oxygen and methane, enabling rapid turnaround between flights. Rocket Lab has "
        "secured $2.1 billion in launch contracts for Neutron from constellation operators "
        "including Globalstar and MDA. The company's Electron small launcher has already "
        "completed 55 missions, establishing Rocket Lab as the second most frequent Western "
        "launch provider after SpaceX.",
        "Ars Technica", "product_launch",
    ),
    (
        "ULA Vulcan Centaur Wins $3.5B Pentagon Contract, Competing with SpaceX Falcon Heavy",
        "United Launch Alliance has secured a $3.5 billion contract from the U.S. Department "
        "of Defense for national security space launches using its Vulcan Centaur rocket. The "
        "award covers 25 launches through 2029 for classified military satellite deployments. "
        "Vulcan Centaur uses Blue Origin's BE-4 liquid methane engines and an innovative "
        "mid-air booster recovery system to partially reduce launch costs. Boeing and Lockheed "
        "Martin, ULA's parent companies, have committed an additional $800 million in capital "
        "to accelerate Vulcan production at ULA's Decatur, Alabama factory. SpaceX challenged "
        "the contract award, arguing that its Falcon Heavy offers lower cost per kilogram.",
        "Defense News", "contract",
    ),
    (
        "Blue Origin New Glenn Completes Qualification Flights, Opens Commercial Bookings",
        "Blue Origin has successfully completed three qualification flights of its New Glenn "
        "heavy-lift rocket and is now accepting commercial launch bookings for 2027. New Glenn "
        "features a reusable first stage powered by seven BE-4 engines and can deliver 45,000 "
        "kg to LEO. Amazon's Project Kuiper has contracted 12 New Glenn flights for its "
        "broadband satellite constellation deployment. Telesat and AST SpaceMobile have also "
        "signed multi-launch agreements. Blue Origin founder Jeff Bezos announced a $2 billion "
        "expansion of the company's Cape Canaveral manufacturing facility to support a target "
        "launch cadence of 24 flights per year by 2029.",
        "CNBC", "product_launch",
    ),
    (
        "AI-Driven Trajectory Optimization Reduces Rocket Fuel Consumption by 12%",
        "A joint study by NASA JPL and SpaceX demonstrates that machine learning-based "
        "trajectory optimization reduces fuel consumption by 8-12% across typical LEO "
        "insertion profiles. SpaceX has integrated the AI system into Falcon 9 and Starship "
        "flight computers for real-time adjustment during ascent. The technology also enables "
        "autonomous abort decisions within 50 milliseconds, improving crew safety for future "
        "manned missions. Rocket Lab and Relativity Space are developing competing AI flight "
        "software, while Boeing's Starliner programme is licensing NASA JPL's trajectory "
        "algorithms for its CST-100 capsule missions. The European Space Agency has funded "
        "a parallel research programme with ArianeGroup to apply similar techniques to "
        "Ariane 6 launches.",
        "MIT Technology Review", "technology",
    ),

    # ── Cluster 2: Small satellite constellation launches (6 articles) ─────────
    (
        "Amazon Kuiper Constellation Deploys First 600 Satellites via SpaceX and Blue Origin",
        "Amazon's Project Kuiper has deployed its first batch of 600 broadband satellites "
        "across three launch providers: SpaceX Falcon 9, Blue Origin New Glenn, and "
        "Arianespace Ariane 6. The deployment marks the beginning of Amazon's plan to "
        "launch 3,236 satellites by 2029 to provide global low-latency internet. SpaceX "
        "handled 8 of the 14 initial launches despite being a direct competitor to Kuiper "
        "through its Starlink network. ULA's Vulcan Centaur is scheduled to join the Kuiper "
        "launch manifest in mid-2027. Amazon has invested $10 billion in the Kuiper programme, "
        "making it the largest commercial satellite constellation project after Starlink.",
        "Wall Street Journal", "expansion",
    ),
    (
        "ISRO's Small Satellite Launch Vehicle Wins Export Orders from European Operators",
        "The Indian Space Research Organisation's Small Satellite Launch Vehicle (SSLV) has "
        "secured launch contracts worth $180 million from European satellite operators OneWeb "
        "and Eutelsat. The SSLV can deliver 500 kg to LEO at one-fifth the cost of European "
        "alternatives, making it attractive for operators seeking affordable dedicated launches. "
        "ISRO plans to ramp up SSLV production to 12 launches per year by 2028 through its "
        "commercialisation arm NewSpace India Limited (NSIL). L&T and Godrej Aerospace are "
        "manufacturing SSLV components under ISRO's outsourcing programme. The export wins "
        "position India as a credible alternative to SpaceX and Rocket Lab for the growing "
        "small satellite launch market.",
        "Hindu BusinessLine", "contract",
    ),
    (
        "Starlink Reaches 5 Million Subscribers as SpaceX Launches 60 Satellites Per Week",
        "SpaceX's Starlink broadband network has reached 5 million active subscribers globally, "
        "supported by an unprecedented launch cadence of approximately 60 satellites per week "
        "on Falcon 9 rockets. The constellation now includes over 6,800 operational satellites "
        "in LEO. SpaceX is transitioning to Starship for Starlink V3 satellite deployment, "
        "which will carry 100+ larger satellites per launch compared to Falcon 9's 23. "
        "Amazon's Kuiper and Telesat's Lightspeed are the primary competitors, though both "
        "lag significantly in deployment scale. Analysts at Quilty Space estimate Starlink's "
        "2026 revenue will exceed $8.5 billion, driven by enterprise maritime and aviation "
        "connectivity contracts with Delta Air Lines and Maersk.",
        "Bloomberg", "expansion",
    ),
    (
        "China's Long March 8 Reusable Variant Targets Commercial Constellation Market",
        "China Aerospace Science and Technology Corporation (CASC) has unveiled a reusable "
        "variant of its Long March 8 rocket designed to compete for commercial constellation "
        "deployment contracts. The Long March 8R features grid fins and landing legs for "
        "first-stage recovery, directly modelled on SpaceX's Falcon 9 approach. China's "
        "Guowang broadband constellation requires 13,000 satellites, creating domestic "
        "demand for over 300 launches. CASC is also marketing the Long March 8R to "
        "international customers at $25 million per launch, undercutting Rocket Lab's "
        "Neutron and Arianespace's Ariane 6. The European Commission has raised concerns "
        "about Chinese state subsidies enabling below-market launch pricing.",
        "South China Morning Post", "product_launch",
    ),
    (
        "AST SpaceMobile Launches First Commercial Direct-to-Cell Satellites on Falcon 9",
        "AST SpaceMobile has launched five BlueBird satellites on a SpaceX Falcon 9 rocket, "
        "initiating the first commercial direct-to-smartphone satellite broadband service. "
        "The satellites connect standard unmodified smartphones to broadband internet from "
        "LEO orbit without specialized hardware. AT&T, Vodafone, and Rakuten are the anchor "
        "mobile network partners providing ground-network integration. AST's constellation "
        "plan calls for 168 satellites to cover the equatorial and mid-latitude regions. "
        "Rocket Lab's Neutron and Blue Origin's New Glenn are being evaluated for future "
        "AST launches to diversify launch provider risk away from sole dependence on SpaceX.",
        "TechCrunch", "product_launch",
    ),
    (
        "OneWeb-Eutelsat Merger Creates $8B LEO Constellation Operator, Orders 600 New Satellites",
        "The merged OneWeb-Eutelsat entity has ordered 600 next-generation LEO satellites from "
        "Airbus Defence and Space, valued at approximately $3.2 billion. The new satellites "
        "will expand OneWeb's constellation from 648 to over 1,200 spacecraft by 2029. Launch "
        "contracts have been awarded to Arianespace, SpaceX, and ISRO's NSIL. Boeing's "
        "satellite manufacturing division lost the contract to Airbus despite offering a lower "
        "per-unit price, due to delivery timeline concerns. The merger positions OneWeb-Eutelsat "
        "as the third-largest LEO constellation operator behind Starlink and Amazon Kuiper, "
        "with a combined enterprise value of $8 billion.",
        "Financial Times", "acquisition",
    ),

    # ── Cluster 3: Space launch startup funding and partnerships (5 articles) ──
    (
        "Stoke Space Raises $200M Series C for Fully Reusable Two-Stage Rocket",
        "Seattle-based Stoke Space has raised $200 million in Series C funding to develop "
        "the first fully reusable two-stage orbital rocket. The company's Nova vehicle uses "
        "a novel differential throttling system that eliminates the need for grid fins during "
        "upper stage re-entry. Breakthrough Energy Ventures led the round with participation "
        "from Spark Capital and Industrious Ventures. Stoke aims for an initial orbital test "
        "in late 2027 and operational launches by 2028. If successful, the Nova rocket would "
        "be the first competitor to SpaceX's Starship to achieve full two-stage reusability. "
        "Boeing Ventures and Lockheed Martin Ventures both passed on the round, preferring "
        "to invest in their existing ULA partnership.",
        "Bloomberg", "funding",
    ),
    (
        "Skyroot Aerospace Becomes India's First Private Orbital Launch Company",
        "Hyderabad-based Skyroot Aerospace has achieved orbit with its Vikram-1 rocket, "
        "becoming the first Indian private company to reach orbital velocity. The three-stage "
        "solid-fuelled rocket delivered a 300 kg earth observation satellite for Planet Labs "
        "to a 500 km sun-synchronous orbit. ISRO provided tracking and telemetry support "
        "through its ground station network. Skyroot has raised a cumulative $150 million "
        "in venture funding from GIC Singapore, Nexus Venture Partners, and LNT Technology "
        "Services. The company plans to develop Vikram-2, a liquid-fuelled reusable variant "
        "targeting the 1,000 kg to LEO segment. Tata Advanced Systems is supplying composite "
        "structures for the Vikram-2 programme under a multi-year manufacturing agreement.",
        "Economic Times", "product_launch",
    ),
    (
        "Isar Aerospace Secures ESA Contract for European Sovereign Launch Access",
        "German launch startup Isar Aerospace has signed a contract with the European Space "
        "Agency worth EUR 150 million to provide sovereign European access to orbit using "
        "its Spectrum rocket. The contract guarantees 10 launches from the Andoya Space "
        "Centre in Norway between 2028 and 2032. Isar's Spectrum is a two-stage liquid "
        "rocket carrying 1,000 kg to LEO, positioned as a European alternative to Rocket Lab's "
        "Electron. Airbus Defence and Space and OHB SE are the anchor satellite customers. "
        "The ESA contract follows growing European concern about strategic dependence on "
        "American launch providers, particularly SpaceX, for defence and institutional payloads.",
        "Reuters", "contract",
    ),
    (
        "Firefly Aerospace Wins NASA CLPS Lunar Delivery Contract, Partners with Northrop Grumman",
        "Firefly Aerospace has been awarded a $280 million NASA CLPS contract to deliver "
        "science payloads to the lunar surface using its Blue Ghost lander, launched atop "
        "a SpaceX Falcon 9 rocket. The contract covers three lunar delivery missions between "
        "2027 and 2029. Firefly has also signed a strategic partnership with Northrop Grumman "
        "to develop a medium-lift launch vehicle combining Firefly's engine technology with "
        "Northrop's solid rocket motor expertise. The partnership positions both companies to "
        "compete for the next round of U.S. Space Force launch contracts. Lockheed Martin's "
        "space division views the Firefly-Northrop alliance as a direct competitive threat "
        "to its cislunar transportation business.",
        "SpaceNews", "contract",
    ),
    (
        "Japan's Space One Raises $180M to Commercialise Kairos Orbital Rocket",
        "Japanese launch startup Space One has raised $180 million in a Series D round led by "
        "Canon Electronics and IHI Aerospace to scale production of its Kairos small orbital "
        "rocket. The Kairos is a four-stage solid-fuelled vehicle capable of delivering 250 kg "
        "to LEO from Space One's dedicated launch site at Kii in Wakayama Prefecture. The "
        "company targets 20 launches per year by 2029, primarily serving Japanese government "
        "constellation programmes and Asian commercial earth observation operators. Toyota "
        "Motor's Woven Planet subsidiary has invested in Space One to explore satellite-based "
        "autonomous driving support systems. Mitsubishi Heavy Industries, which operates "
        "Japan's H3 rocket, views Space One as complementary rather than competitive.",
        "Nikkei Asia", "funding",
    ),
]
