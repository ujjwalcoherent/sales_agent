"""
Two-tier event classification for news articles.

TIER 1 (Fast, CPU): Embedding-based classification using REAL article title
anchors (20+ per event type). 80% of articles classified here with no LLM cost.

TIER 2 (LLM validation): For ambiguous cases (confidence < threshold or top-2
within margin), send to LLM for classification with reasoning. Catches the
~20% that embeddings alone misclassify.

V7 IMPROVEMENTS over V6:
  - Real article title anchors instead of hand-written descriptions
  - LLM Tier-2 validation for ambiguous classifications
  - Every classification now carries reasoning
  - Configurable LLM trigger threshold

PERFORMANCE: ~1-2s for 500 articles (batched embedding + vector math)
             + ~5s for 20% ambiguous LLM calls
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TAXONOMY_CANDIDATES_PATH = Path("./data/taxonomy_candidates.jsonl")

# V7: Event types with urgency scores, REAL article title anchors, and keyword boosters.
# Using real article titles as anchors gives much better embedding coverage than
# hand-written descriptions — incoming articles land in the same embedding space.
EVENTS = {
    "regulation": {
        "urgency": 0.95,
        "descriptions": [
            "SEBI tightens norms for algo trading platforms in India",
            "RBI mandates banks to adopt new KYC framework by March 2026",
            "TRAI releases new tariff order for telecom operators",
            "Government notifies new EPR rules for electronics manufacturers",
            "FSSAI issues new food safety compliance standards for packaged goods",
            "Data protection authority issues first enforcement notice under DPDP Act",
            "MCA mandates ESG reporting for companies with turnover above Rs 1000 crore",
            "IRDAI introduces new guidelines for health insurance claim settlement",
            "BIS makes quality certification mandatory for imported steel products",
            "Labour ministry notifies new wage code implementation timeline",
            "Government regulation, compliance mandate, regulatory change, central bank policy",
            "Regulatory crackdown, policy tightening, mandatory disclosure, data protection law",
        ],
        "keyword_boost": ["regulation", "regulatory", "compliance", "mandate", "sebi", "rbi", "trai", "fssai", "norms", "guidelines", "circular", "directive"],
    },
    "funding": {
        "urgency": 0.70,
        "descriptions": [
            "Razorpay raises $500M in Series G round at $7.5B valuation",
            "Zerodha-backed fintech startup secures $25M seed funding",
            "PhysicsWallah raises $100M Series B from WestBridge Capital",
            "Healthtech startup Pristyn Care raises $100M in Series E",
            "Indian SaaS company Freshworks announces secondary share sale",
            "Ola Electric files for IPO after raising $500M pre-IPO round",
            "Early stage VC fund announces $200M India-focused vehicle",
            "Private equity firm KKR invests $200M in Indian pharma company",
            "AgriTech startup DeHaat raises $60M Series D from Sofina",
            "B2B commerce startup Udaan raises debt funding of $250M",
            "Peak XV Partners raises $1.3 billion for first independent India fund after Sequoia split",
            "Accel raises new $650 million fund focused on Indian startups and growth stage companies",
            "Venture capital firm closes billion-dollar fund for India and Southeast Asia investments",
            "Weekly funding roundup: Indian startups raised $500 million across 20 deals this week",
            "Startup funding round, venture capital investment, Series A B C D",
            "Angel investment deal closed, growth equity round completed",
        ],
        "keyword_boost": ["funding", "raises", "raised", "investment", "series", "venture", "capital", "investor", "valuation", "unicorn", "seed", "fund", "billion", "million", "secures", "secured"],
    },
    "expansion": {
        "urgency": 0.75,
        "descriptions": [
            "Tata Motors opens new EV manufacturing plant in Tamil Nadu",
            "Flipkart expands quick commerce to 30 new cities across India",
            "Infosys opens new development centre in Tier-2 city Nagpur",
            "Reliance Retail adds 2500 new stores in FY25 expansion drive",
            "Indian IT company HCLTech expands operations into Saudi Arabia",
            "Zomato launches services in 500 new towns under hyperlocal push",
            "Indian pharma company sets up API manufacturing unit in Gujarat",
            "Adani Group breaks ground on new port facility in Kerala",
            "DMart announces 50 new store openings in South India",
            "Tech Mahindra sets up AI centre of excellence in Hyderabad",
            "Business expansion, opening new office plant factory, entering new market",
            "Capacity expansion, production ramp-up, market penetration strategy",
        ],
        "keyword_boost": ["expansion", "expands", "opens", "new office", "new plant", "new facility", "scaling", "entering"],
    },
    "acquisition": {
        "urgency": 0.80,
        "descriptions": [
            "Adani Group acquires Ambuja Cements in $6.5B deal",
            "Byju's acquires Aakash Educational Services for $1 billion",
            "Tata Digital acquires majority stake in BigBasket for $1.3B",
            "Piramal Group completes acquisition of DHFL for Rs 34,250 crore",
            "Zomato acquires Blinkit in all-stock deal valued at $568M",
            "Reliance acquires majority stake in Dunzo for $200M",
            "HCL Technologies to acquire German IT firm for EUR 700M",
            "PhonePe acquires WealthDesk and OpenQ to expand wealth management",
            "HDFC Bank-HDFC merger creates India's largest financial entity",
            "JSW Steel acquires Bhushan Power and Steel for Rs 19,700 crore",
            "ASM Technologies to acquire 20% stake in AI startup Myelin Foundry",
            "HDFC Mutual Fund raises stake in healthcare company to over 7 percent",
            "ChrysCapital-led group buys 70% stake in Novartis India arm for Rs 1,446 crore",
            "Company acquisition, corporate merger, buyout, takeover bid, M&A deal",
        ],
        "keyword_boost": ["acquisition", "acquires", "merger", "buyout", "takeover", "M&A", "merged", "acquire", "stake", "buys stake", "exits"],
    },
    "leadership_change": {
        "urgency": 0.65,
        "descriptions": [
            "Infosys appoints new CEO Salil Parekh replacing Vishal Sikka",
            "Wipro CEO Thierry Delaporte steps down after 4 years",
            "Paytm founder Vijay Shekhar Sharma relinquishes CEO role",
            "HDFC Bank announces new managing director after Aditya Puri retirement",
            "TCS appoints K Krithivasan as new CEO and MD",
            "Byju's CEO Raveendran steps down amid governance concerns",
            "New RBI governor takes charge amid policy transition",
            "Biocon founder Kiran Mazumdar Shaw appoints new CEO",
            "Board reshuffle at Yes Bank as new directors appointed",
            "Senior leadership change at Ola as CTO and CFO exit",
            "New CEO CTO CFO appointed, executive resignation, leadership transition",
        ],
        "keyword_boost": ["appoints", "appointed", "CEO", "CTO", "CFO", "COO", "resignation", "steps down", "new head"],
    },
    "layoffs": {
        "urgency": 0.85,
        "descriptions": [
            "Byju's lays off 2500 employees in cost-cutting drive",
            "Ola fires 1000 workers as electric scooter sales slow down",
            "Swiggy cuts 380 jobs ahead of IPO to reduce costs",
            "ShareChat parent lays off 500 employees across divisions",
            "PharmEasy fires 600 employees as funding winter continues",
            "Meesho lays off 250 employees in restructuring exercise",
            "Vedanta announces workforce optimization affecting 1200 jobs",
            "Unacademy fires 350 employees in third round of layoffs",
            "Tech startup lays off entire India team amid global restructuring",
            "Indian IT firm announces voluntary separation scheme for 2000 employees",
            "Company layoffs, job cuts, workforce reduction, restructuring, downsizing",
        ],
        "keyword_boost": ["layoffs", "layoff", "fires", "fired", "job cuts", "workforce reduction", "downsizing", "restructuring"],
    },
    "ipo": {
        "urgency": 0.80,
        "descriptions": [
            "Swiggy files DRHP for Rs 10,000 crore IPO with SEBI",
            "Ola Electric IPO subscribed 4.3x on final day of bidding",
            "FirstCry parent Brainbees Solutions lists at 40% premium on BSE",
            "MobiKwik IPO opens for subscription at Rs 279 per share",
            "Hyundai India files for largest auto IPO in Indian history",
            "Navi Technologies withdraws IPO application amid market uncertainty",
            "LIC IPO becomes India's largest ever public offering at Rs 21,000 crore",
            "Go Digit Insurance gets SEBI approval for Rs 6,000 crore IPO",
            "PhonePe considers IPO in India after moving domicile from Singapore",
            "Boat files draft papers for Rs 2000 crore IPO",
            "IPO initial public offering, stock market listing, DRHP filing",
        ],
        "keyword_boost": ["IPO", "listing", "DRHP", "going public", "public offering", "stock market debut", "listed"],
    },
    "technology": {
        "urgency": 0.60,
        "descriptions": [
            "Indian government launches AI mission with Rs 10,000 crore outlay",
            "TCS rolls out enterprise generative AI platform for banking clients",
            "Jio launches 5G cloud gaming service across 50 cities",
            "ISRO and private companies collaborate on small satellite launch",
            "Indian startups adopt GPT-4 and Claude for customer service automation",
            "Cybersecurity breach at Indian hospital chain exposes 3M patient records",
            "Government mandates cloud-first policy for all central ministries",
            "Indian semiconductor fab gets environmental clearance in Gujarat",
            "UPI crosses 10 billion monthly transactions milestone",
            "Infosys launches quantum computing research lab in Bangalore",
            "AI artificial intelligence adoption, digital transformation, cloud migration",
            "Technology disruption, automation, SaaS platform, tech innovation",
        ],
        "keyword_boost": ["AI", "artificial intelligence", "digital", "cloud", "cybersecurity", "automation", "SaaS", "machine learning", "generative"],
    },
    "partnership": {
        "urgency": 0.50,
        "descriptions": [
            "Reliance and Disney merge India media operations in $8.5B deal",
            "Tata and Airbus sign MoU to manufacture military helicopters in India",
            "Mahindra and Volkswagen explore EV component supply partnership",
            "Google Cloud partners with Indian government for digital infrastructure",
            "NPCI and Singapore's PayNow link UPI for cross-border payments",
            "Indian Oil and NTPC form JV for green hydrogen production",
            "Amazon and Future Group enter into strategic alliance for retail",
            "Wipro and Microsoft announce expanded cloud partnership",
            "Ola and banks collaborate on vehicle financing platform",
            "HDFC and Paytm partner for instant personal loan distribution",
            "Strategic business partnership, corporate alliance, joint venture",
        ],
        "keyword_boost": ["partnership", "partners", "alliance", "joint venture", "MOU", "collaboration", "tie-up"],
    },
    "crisis": {
        "urgency": 0.90,
        "descriptions": [
            "Adani Group faces short-seller attack as Hindenburg Research publishes report",
            "Yes Bank placed under moratorium by RBI amid liquidity crisis",
            "DHFL promoters arrested in Rs 34,000 crore bank fraud case",
            "Byju's faces insolvency proceedings as NCLT admits creditor petition",
            "Go First airline files for voluntary insolvency resolution",
            "PNB detects Rs 11,400 crore fraud involving Nirav Modi companies",
            "Paytm Payments Bank ordered to stop new account openings by RBI",
            "Major data breach at Domino's India exposes 180M customer records",
            "IL&FS default triggers NBFC liquidity crisis across India",
            "Wipro discovers accounting fraud in subsidiary operations",
            "Corporate fraud, financial scam, data breach, bankruptcy filing",
        ],
        "keyword_boost": ["fraud", "scam", "scandal", "breach", "bankruptcy", "default", "violation", "crisis", "investigation"],
    },
    "supply_chain": {
        "urgency": 0.75,
        "descriptions": [
            "Red Sea shipping crisis forces Indian exporters to reroute via Cape of Good Hope",
            "Semiconductor shortage forces Indian auto makers to cut production by 20%",
            "Steel prices surge 40% putting pressure on Indian construction companies",
            "Lithium shortage threatens Indian EV battery manufacturing plans",
            "Container freight rates triple as global shipping lanes disrupted",
            "Indian pharma companies face API supply disruption from China lockdowns",
            "Cotton prices surge affecting textile manufacturers in Tirupur and Surat",
            "Coal shortage forces power plants to operate at 50% capacity",
            "Palm oil import restrictions create supply gap for FMCG companies",
            "Rare earth supply concerns impact Indian electronics manufacturers",
            "Supply chain disruption, procurement challenge, raw material shortage",
        ],
        "keyword_boost": ["supply chain", "procurement", "shortage", "logistics", "bottleneck", "raw material", "sourcing"],
    },
    "price_change": {
        "urgency": 0.65,
        "descriptions": [
            "Crude oil prices cross $100 per barrel impacting Indian fuel costs",
            "RBI hikes repo rate by 25 bps to combat persistent inflation",
            "Indian government raises import duty on gold to 15 percent",
            "Steel companies announce 8-12% price hike effective next month",
            "FMCG companies shrink pack sizes as input costs rise sharply",
            "India-EU FTA reduces auto tariffs by 30% benefiting exporters",
            "Cement prices fall 10% amid oversupply in southern India",
            "Electricity tariffs revised upward by state regulators across 5 states",
            "Wheat prices surge as India restricts exports amid heat wave",
            "Telecom operators announce tariff hikes of 15-25% across plans",
            "Sharp surge in memory chip prices triggers shrinkflation in consumer electronics",
            "Commodity price surge forces manufacturers to cut features and shrink pack sizes",
            "Price hike, inflation impact, commodity price surge, margin pressure",
        ],
        "keyword_boost": ["price", "prices", "hike", "inflation", "cost increase", "tariff", "price war", "margin", "shrinkflation", "surge"],
    },
    "consumer_shift": {
        "urgency": 0.55,
        "descriptions": [
            "Quick commerce grows 70% in India as consumers demand 10-minute delivery",
            "D2C beauty brands capture 25% market share from traditional FMCG",
            "Rural India drives smartphone adoption with 200M new internet users",
            "Premium segment grows fastest as Indian middle class trades up",
            "UPI transactions in Tier-3 cities grow 150% year-over-year",
            "Electric two-wheeler sales overtake petrol scooters in urban India",
            "Indian consumers shift to plant-based protein alternatives",
            "Festival season e-commerce sales cross $10 billion for first time",
            "Subscription economy grows as Indians adopt streaming and SaaS tools",
            "Gen Z spending patterns reshape Indian retail landscape",
            "Consumer behavior change, D2C growth, e-commerce boom, quick commerce",
        ],
        "keyword_boost": ["consumer", "D2C", "e-commerce", "spending", "demand", "shopping", "brand", "quick commerce"],
    },
    "market_entry": {
        "urgency": 0.70,
        "descriptions": [
            "Apple begins manufacturing iPhone 15 Pro in India under PLI scheme",
            "Saudi Aramco explores $50B refinery investment in Maharashtra",
            "Tesla files for trademark registration in India ahead of market entry",
            "IKEA opens largest store in India in Bangalore amid expansion push",
            "Chinese EV maker BYD doubles India production capacity",
            "Indian IT companies expand into African markets for growth",
            "Japan's SoftBank increases India allocation to $10B for new fund",
            "PLI scheme attracts $30B in committed investments across 14 sectors",
            "Indian pharma companies register products in 50 new African markets",
            "Foxconn announces $1.5B investment in new Karnataka electronics factory",
            "Cross-border market entry, foreign direct investment, FDI inflow",
        ],
        "keyword_boost": ["market entry", "FDI", "PLI", "Make in India", "export", "trade agreement", "international", "cross-border"],
    },

    # ─── NEW CATEGORIES: Cover the 30% blind spots ────────────────────────
    "earnings": {
        "urgency": 0.60,
        "descriptions": [
            "Reliance Industries Q3 results: Net profit rises 12% to Rs 18,540 crore",
            "TCS reports 8.4% revenue growth in Q2 FY26, beats street estimates",
            "Infosys raises revenue guidance after strong Q3 earnings beat",
            "Zomato turns profitable for first time with Rs 138 crore net income",
            "HDFC Bank reports 20% jump in net profit for Q3 at Rs 16,372 crore",
            "Adani Enterprises misses profit estimates as commodity prices fall",
            "Indian IT companies report mixed Q2 results amid demand slowdown",
            "BSE-listed companies report aggregate 15% profit growth in Q3 FY26",
            "Tata Motors posts record profit on strong JLR and EV demand",
            "Paytm narrows losses as revenue from financial services grows 40%",
            "Quarterly earnings results, profit loss statement, revenue growth, EPS beat miss",
            "Corporate financial results, annual report, profit margins, EBITDA",
        ],
        "keyword_boost": ["profit", "revenue", "earnings", "results", "quarterly", "Q1", "Q2", "Q3", "Q4", "FY", "EPS", "EBITDA", "net income", "topline", "bottomline"],
    },
    "market_movement": {
        "urgency": 0.50,
        "descriptions": [
            "Sensex surges 800 points as FII inflows boost market sentiment",
            "Nifty 50 hits all-time high of 25,000 amid broad-based rally",
            "Indian stock market crashes 3% on global recession fears",
            "FII outflows cross Rs 50,000 crore in January as dollar strengthens",
            "Midcap and smallcap stocks rally 5% after budget announcements",
            "Indian rupee falls to record low of 85.5 against US dollar",
            "Gold prices hit Rs 75,000 per 10 grams on safe-haven demand",
            "Bond yields rise as RBI signals tighter monetary policy stance",
            "Commodity markets rally on supply concerns and strong China demand",
            "Crypto market in India grows as Bitcoin crosses $100,000 milestone",
            "GIFT Nifty soars as US Supreme Court rules against Trump tariffs",
            "Cybersecurity stocks hit sharply after new AI tool disrupts legacy vendors",
            "What does the Supreme Court tariff decision mean for the Indian stock market",
            "$134 billion at stake as brokerages flag refund risks after tariff ruling",
            "S&P 500 rises to day's high as Trump overrules Supreme Court on tariffs",
            "Silver rate outlook: White metal historically consolidates 3 to 8 years",
            "Gold vs silver vs Nifty 50: Which asset to prefer in your portfolio",
            "AI stocks down big from 52-week highs — are they worth buying now",
            "Oil jitters cap gains as markets end the week flat amid sectoral churn",
            "Stock market rally crash correction, index movement, FII DII flows",
            "Multibagger penny stock, bull bear market, portfolio returns",
        ],
        "keyword_boost": ["sensex", "nifty", "stock", "stocks", "market", "rally", "crash", "FII", "DII", "rupee", "gold", "silver", "share price", "returns", "multibagger", "bull", "bear", "soars", "surges", "plunges", "shares", "portfolio"],
    },
    "infrastructure": {
        "urgency": 0.65,
        "descriptions": [
            "Government approves Rs 1.5 lakh crore for new expressway projects",
            "Mumbai-Ahmedabad bullet train project reaches 50% completion milestone",
            "National Investment Infrastructure Fund raises $5 billion for roads",
            "Navi Mumbai international airport to be operational by December 2026",
            "Indian Railways announces Rs 75,000 crore capex for track doubling",
            "PM inaugurates new metro line connecting Delhi to Greater Noida",
            "Adani Ports wins contract for new container terminal in Sri Lanka",
            "Rural road connectivity programme covers 95% of habitations",
            "Smart cities mission delivers 500 completed projects across India",
            "India's largest solar park of 5 GW capacity announced in Rajasthan",
            "Government infrastructure investment, highway road port airport project",
            "Urban development, smart city, public transport, connectivity",
        ],
        "keyword_boost": ["infrastructure", "highway", "metro", "airport", "port", "railway", "road", "bridge", "project", "capex", "construction", "smart city"],
    },
    "geopolitical": {
        "urgency": 0.75,
        "descriptions": [
            "India-US trade tensions escalate as tariff talks stall",
            "PM Modi meets Trump to discuss bilateral trade and defense ties",
            "India-China border standoff continues to impact business sentiment",
            "India bans Chinese apps citing national security concerns",
            "BRICS nations agree on alternative payment system to bypass dollar",
            "India-EU free trade agreement talks enter final stage",
            "Sanctions on Russia create opportunity for Indian oil refiners",
            "India-Middle East trade corridor announced as alternative to Suez Canal",
            "US immigration policy changes impact Indian IT workforce deployment",
            "India-Canada diplomatic row affects bilateral business ties",
            "How will companies get refunds after Supreme Court strikes down Trump tariffs",
            "Trump imposes new 10% tariff on all imports after Supreme Court ruling",
            "International diplomacy, trade war, sanctions, bilateral relations",
            "Geopolitical tensions, cross-border conflict, foreign policy shift",
        ],
        "keyword_boost": ["trade war", "tariff", "tariffs", "sanctions", "bilateral", "diplomatic", "foreign policy", "geopolitical", "BRICS", "G20", "summit", "supreme court"],
    },
    "policy": {
        "urgency": 0.70,
        "descriptions": [
            "Union Budget 2026 allocates Rs 11 lakh crore for capital expenditure",
            "Government announces new PLI scheme for semiconductor manufacturing",
            "India's new industrial policy aims to boost manufacturing to 25% of GDP",
            "Cabinet approves new national education policy implementation roadmap",
            "PM inaugurates AI mission with Rs 10,000 crore investment plan",
            "RBI monetary policy committee holds repo rate steady at 6.5%",
            "Government announces new startup policy with tax incentives for founders",
            "Green hydrogen policy offers subsidies for electrolyzer manufacturing",
            "Digital India programme enters next phase with rural broadband push",
            "Government revises income tax slabs in budget benefiting middle class",
            "Government policy initiative, budget allocation, scheme announcement",
            "Economic policy reform, fiscal stimulus, incentive programme",
        ],
        "keyword_boost": ["budget", "policy", "government", "scheme", "subsidy", "incentive", "reform", "allocation", "fiscal", "ministry"],
    },
    "sustainability": {
        "urgency": 0.55,
        "descriptions": [
            "India commits to net zero carbon emissions by 2070 at COP summit",
            "Tata Power commissions 1 GW solar capacity in Rajasthan",
            "SEBI mandates ESG reporting for top 1000 listed companies",
            "Indian companies face carbon border tax on EU exports from 2026",
            "Adani Green Energy becomes world's largest solar company",
            "Green bond issuances by Indian companies cross $10 billion milestone",
            "EV battery recycling industry set to become Rs 50,000 crore market",
            "India's renewable energy capacity crosses 200 GW landmark",
            "Corporate India accelerates ESG adoption amid investor pressure",
            "Steel companies invest in green hydrogen to decarbonize production",
            "ESG environmental social governance, carbon neutral, climate action",
            "Renewable energy, solar wind, sustainability transition, green finance",
        ],
        "keyword_boost": ["ESG", "sustainability", "renewable", "solar", "wind", "carbon", "green", "climate", "net zero", "EV", "clean energy"],
    },

    # ─── NOISE ATTRACTOR CATEGORIES: Pull junk AWAY from business types ──
    # These exist to attract entertainment/sports/lifestyle articles that
    # otherwise get misclassified as "earnings" or "consumer_shift" due to
    # shared vocabulary ("collection", "revenue", "brand", "trend").
    # Articles tagged with these are filtered before clustering.
    "entertainment": {
        "urgency": 0.0,  # Zero urgency = noise
        "descriptions": [
            "Do Deewane Seher Mein Box Office Collection Day 1: Siddhant Chaturvedi",
            "O Romeo Box Office Collection Day 8: Shahid Kapoor Film Sees Major Slide",
            "Nia Sharma says her words are as sharp as her eyeliner",
            "Alia Bhatt airport look turns into a fashion moment with Rs 3 lakh Gucci coat",
            "Shah Rukh Khan upcoming movie Pathaan 2 release date announced",
            "Netflix India announces new original series lineup for 2026",
            "Pushpa 2 breaks all-time box office records in first week",
            "Bigg Boss 20 elimination: contestant evicted in shocking twist",
            "Bollywood celebrity wedding photos go viral on social media",
            "Jawan crosses Rs 1000 crore worldwide collection milestone",
            "Film review: Animal starring Ranbir Kapoor is a violent mess",
            "Grey's Anatomy star Eric Dane dies at 53 after long battle with ALS diagnosis",
            "Ambani family hosts grand dinner party with global celebrities at Antilla Mumbai",
            "Priyanka Chopra shares adorable photos of daughter Malti on Instagram",
            "Bollywood box office collection, movie review, celebrity gossip, OTT release",
            "Film star actor actress award ceremony entertainment industry drama",
            "Celebrity death obituary funeral tribute social media viral photos pics",
            "Pokemon FireRed and LeafGreen on Nintendo Switch release date and gameplay",
            "New video game console launch and gaming industry news updates",
        ],
        "keyword_boost": ["bollywood", "box office", "collection day", "movie", "film", "actress", "actor", "celebrity", "netflix", "ott", "bigg boss", "trailer", "blockbuster", "flop", "see pics", "photos", "viral", "star dies", "obituary", "pokemon", "nintendo", "playstation", "xbox", "gaming", "gameplay"],
    },
    "sports": {
        "urgency": 0.0,
        "descriptions": [
            "India vs Australia 3rd Test Day 2: Virat Kohli smashes century at MCG",
            "IPL 2026 Auction: Chennai Super Kings buy Pat Cummins for Rs 20 crore",
            "Jasprit Bumrah named ICC Cricketer of the Year 2025",
            "Pro Kabaddi League Season 12 final: Patna Pirates clinch title",
            "Neeraj Chopra wins gold medal at World Athletics Championship",
            "Indian football team qualifies for AFC Asian Cup knockout stage",
            "PV Sindhu reaches semi-finals at All England Badminton Open",
            "Indian Premier League viewership breaks records with 50 crore fans",
            "FIFA World Cup 2026 qualifiers: India draws 1-1 against Qatar",
            "Indian cricket team selection for upcoming England tour announced",
            "Cricket match score IPL T20 test series ODI world cup",
            "Sports tournament league championship medal trophy fixture result",
        ],
        "keyword_boost": ["cricket", "ipl", "match", "wicket", "innings", "football", "kabaddi", "badminton", "hockey", "medal", "tournament", "league", "championship", "score", "olympic"],
    },
    "lifestyle": {
        "urgency": 0.0,
        "descriptions": [
            "Assamese poems of displacement published in new anthology",
            "Food News Today: Your Source for Delicious Food Insights and Tasteful Trends",
            "Best restaurants to visit in Goa for a weekend getaway in 2026",
            "Horoscope today February 20 2026: What stars say for your zodiac sign",
            "How to Find Your Skills and Talents Through Patterns Not Passion",
            "Top 10 yoga poses for beginners to improve flexibility and strength",
            "Readers comments: On Indias reading culture, a thank you for a mirror",
            "New book review: A memoir of growing up in small-town India",
            "Best skincare routine for Indian summers according to dermatologists",
            "Recipe: How to make perfect butter chicken at home in 30 minutes",
            "Travel guide: Hidden gems of Northeast India you must visit",
            "Are banks open today? Check bank holiday calendar for your state",
            "Bank holidays in February 2026: Complete list of closed dates by state",
            "Can you open multiple PPF accounts? Here is the answer explained simply",
            "Monthly savings you need to accumulate Rs 10 crore in 15 years explained",
            "5 Morning habits to make your days more productive and successful",
            "Wellness tips, self-help advice, recipe cooking, travel destination guide",
            "Book poetry literature review, horoscope zodiac astrology prediction",
            "Personal finance tips, bank holiday list, savings calculator, PPF NPS",
            "Want share in father's property? Inheritance rights and family law explained",
            "How to claim share in ancestral property: legal rights guide for heirs",
        ],
        "keyword_boost": ["recipe", "horoscope", "zodiac", "yoga", "wellness", "travel", "destination", "book review", "poetry", "self-help", "skincare", "fitness", "cooking", "astrology", "bank holiday", "morning habits", "productive", "ppf", "savings calculator"],
    },
    # ─── CRIME / VIOLENCE: Pull crime stories away from business events ──
    # Crime articles often get misclassified as market_movement (robbery amounts),
    # supply_chain (property disputes), or crisis (violence involving companies).
    # This category attracts them into noise.
    "crime": {
        "urgency": 0.0,  # Zero urgency = noise
        "descriptions": [
            "Mother and infant burnt to death in Indian state over witchcraft allegations",
            "Barpeta showroom robbery: Rs 12 crore worth of jewellery looted by armed gang",
            "Man arrested for stabbing wife over property dispute in Delhi NCR",
            "Police bust interstate drug trafficking ring operating from Mumbai warehouse",
            "Three killed in road accident on national highway in Rajasthan",
            "Mob lynching reported in Bihar village over cattle theft allegation",
            "Woman found murdered in apartment, husband absconding say police",
            "Gold smuggling racket busted at Hyderabad airport, customs seize Rs 5 crore",
            "Child kidnapping gang arrested, five children rescued in UP operation",
            "Cyber fraud: Retired army officer loses Rs 1.2 crore to online scam",
            "Serial chain snatcher arrested after CCTV footage analysis in Bengaluru",
            "Property dealer shot dead by business rival in broad daylight in Lucknow",
            "Murder arrest robbery kidnapping theft crime violence assault police FIR",
            "Accused absconding stolen looted mob attack lynch gang rape victim death",
            "School shooting suspect arrested after mass shooting incident in Canada",
            "Suspect flagged by AI company before carrying out deadly shooting attack",
        ],
        "keyword_boost": ["murder", "murdered", "robbery", "robbed", "looted", "arrested", "stabbing", "kidnapping", "theft", "assault", "lynching", "FIR", "accused", "crime", "violence", "police arrest", "smuggling", "shot dead", "burnt to death", "witchcraft", "shooting", "homicide", "suspect", "rape", "trafficking"],
    },
    # ─── POLITICS / ELECTIONS: Pull political news away from business types ──
    # Political articles about elections, court cases, party politics get
    # misclassified as policy, regulation, or crisis due to shared vocabulary.
    "politics": {
        "urgency": 0.0,  # Zero urgency = noise
        "descriptions": [
            "Rahul Gandhi appears in court in RSS defamation case hearing",
            "BJP sweeps municipal elections in Gujarat, Congress calls for recount",
            "AAP announces free electricity promise ahead of Delhi assembly elections",
            "Congress leader attacks PM Modi over handling of farmers protest",
            "Mamata Banerjee challenges BJP at rally in Kolkata ahead of state polls",
            "Assam evicted voters find names struck off electoral rolls ahead of bypolls",
            "Supreme Court refuses to stay CAA implementation despite opposition plea",
            "Former Prince Andrew to be removed from line of succession to throne",
            "Opposition parties form alliance to challenge ruling government in 2027 elections",
            "Voter list controversy erupts in West Bengal ahead of panchayat elections",
            "Election campaign rally, party politics, opposition attack, BJP Congress AAP",
            "Court hearing verdict defamation case political leader MLA MP minister",
        ],
        "keyword_boost": ["election", "elections", "BJP", "Congress", "AAP", "vote", "voter", "rally", "opposition", "ruling party", "campaign", "defamation", "court case", "MLA", "MP", "minister", "panchayat", "electoral", "assembly polls"],
    },

    # ─── V11: New event types to reduce "general" rate ──────────────────
    # These 4 types cover the most common gaps in Indian business news
    # classification where articles fall into "general" for lack of a match.

    "economic_data": {
        "urgency": 0.60,
        "descriptions": [
            "India GDP growth slows to 6.3% in Q3 FY26 as manufacturing weakens",
            "CPI inflation rises to 5.2% in January driven by vegetable prices",
            "India's current account deficit widens to $23 billion in Q3 2025",
            "RBI holds repo rate at 6.5% for sixth consecutive policy meeting",
            "India's unemployment rate drops to 6.8% in December CMIE data shows",
            "India's fiscal deficit reaches 58% of full year target in first half",
            "Foreign exchange reserves hit record $700 billion says RBI data",
            "India's manufacturing PMI rises to 57.5 in January signalling expansion",
            "Wholesale price index WPI inflation turns positive at 1.3% in January",
            "India's trade deficit narrows to $15 billion in January as exports rise",
            "GST collection crosses Rs 1.72 lakh crore in January 2026",
            "India foreign direct investment FDI inflows rise 12% to $45 billion",
        ],
        "keyword_boost": ["GDP", "inflation", "CPI", "WPI", "PMI", "unemployment", "fiscal deficit", "current account", "trade deficit", "forex reserves", "repo rate", "GST collection", "FDI", "economic growth", "quarterly data", "economic survey"],
    },
    "legal": {
        "urgency": 0.65,
        "descriptions": [
            "Supreme Court strikes down IEPFA rules on unclaimed dividends as unconstitutional",
            "NCLT approves Tata Steel's merger with seven group subsidiaries",
            "Delhi High Court stays CCI order imposing Rs 1337 crore penalty on Google",
            "Bombay High Court quashes retrospective tax demand on Vodafone Idea",
            "Supreme Court upholds SEBI's power to regulate mutual fund distributors",
            "NCLAT orders reinstatement of suspended directors at Byju's parent company",
            "Arbitration tribunal awards $1.2 billion compensation to Cairn Energy",
            "Competition Commission approves Reliance-Disney merger with conditions",
            "Consumer court orders insurance company to pay Rs 50 lakh for claim rejection",
            "Tax tribunal rules in favour of IT companies on transfer pricing adjustments",
            "Court ruling, legal challenge, tribunal verdict, arbitration, lawsuit settlement",
        ],
        "keyword_boost": ["court", "tribunal", "ruling", "verdict", "lawsuit", "arbitration", "NCLT", "NCLAT", "CCI", "appeal", "bench", "judgment", "constitution", "petition", "writ"],
    },
    "trade": {
        "urgency": 0.70,
        "descriptions": [
            "India imposes anti-dumping duty on Chinese steel imports for five years",
            "US announces 25% tariff on Indian electronics effective March 2026",
            "India-UK free trade agreement talks enter final round in London",
            "India's merchandise exports cross $40 billion mark in January 2026",
            "PLI scheme drives Rs 12000 crore in electronics exports from India",
            "India bans import of palm oil from Malaysia over diplomatic row",
            "India and Australia sign interim trade deal covering 85% of tariff lines",
            "DGFT notifies new export incentive scheme RODTEP rates for textiles",
            "India's services exports surge 18% to $30 billion in December quarter",
            "EU carbon border tax CBAM to impact Indian steel exports by $2 billion",
            "Trump tariff reciprocal trade levy imports duties customs bilateral agreement",
        ],
        "keyword_boost": ["tariff", "tariffs", "export", "exports", "import", "imports", "trade deal", "trade agreement", "FTA", "customs duty", "anti-dumping", "DGFT", "PLI", "RODTEP", "bilateral", "reciprocal", "levy", "duties"],
    },
    "product_launch": {
        "urgency": 0.55,
        "descriptions": [
            "Tata Motors launches Punch EV at Rs 10.99 lakh in Indian market",
            "Samsung Galaxy S26 Ultra launched in India starting at Rs 1,29,999",
            "Reliance Jio launches JioAirFiber 5G home broadband service across 100 cities",
            "Apple introduces new MacBook Air M4 with 18-hour battery life in India",
            "Maruti Suzuki unveils electric SUV eVitara at Rs 17.49 lakh",
            "OnePlus 14 Pro launched in India with Snapdragon 8 Gen 4 processor",
            "Ola Electric launches S1 Air scooter at Rs 1.10 lakh for mass market",
            "Flipkart launches Flipkart Minutes 10-minute delivery in 15 cities",
            "Mahindra XEV 9e electric SUV launched at Rs 21.90 lakh in India",
            "Hyundai Creta Electric launched at Rs 17.99 lakh challenging Tata Nexon EV",
            "New product launch, release date, price Rs lakh, introduced unveiled, starting at",
        ],
        "keyword_boost": ["launch", "launched", "launches", "unveil", "unveiled", "introduces", "starting at", "priced at", "release", "new model", "product launch"],
    },
}

# Noise types: articles classified as these are filtered before clustering.
# They exist only to attract non-business content away from real event types.
NOISE_EVENT_TYPES = {"entertainment", "sports", "lifestyle", "crime", "politics"}

# V12: Hierarchical 2-level domain taxonomy.
# Level 0 = DOMAIN (7 business + 1 noise). k-NN discriminates within domain only.
# Level 1 = EVENT TYPE (existing 24 business types fit here).
#
# Why hierarchy matters for scaling:
#   - Adding a new event type = adding to one domain, not competing with all 29
#   - Within-domain k-NN only compares 3-7 types, not 29 → more discriminative
#   - Level 0 filter replaces noise attractors (separate binary decision)
#   - Each domain can have independent threshold tuning
#
# REF: NewsAPI.ai (136 types, 2-level), PLOVER compressed CAMEO 310→16-18,
#      AYLIEN 3000 topics + 307 events (2-3 levels).
EVENT_DOMAINS = {
    "corporate_action": {
        "types": ["acquisition", "funding", "ipo", "expansion", "market_entry",
                  "partnership", "product_launch"],
        "description": "Things companies DO — deals, launches, market moves",
    },
    "market_data": {
        "types": ["earnings", "market_movement", "economic_data", "price_change"],
        "description": "Financial and market information — prices, GDP, earnings",
    },
    "macro_environment": {
        "types": ["policy", "geopolitical", "trade", "infrastructure"],
        "description": "Economy-wide forces — government policy, trade, geopolitics",
    },
    "technology_shift": {
        "types": ["technology"],
        "description": "Tech adoption, AI, digital transformation, cybersecurity",
    },
    "supply_demand": {
        "types": ["supply_chain", "consumer_shift"],
        "description": "Supply chain disruptions, consumer behavior changes",
    },
    "workforce": {
        "types": ["leadership_change", "layoffs"],
        "description": "People changes — C-suite moves, hiring, layoffs",
    },
    "legal_regulatory": {
        "types": ["regulation", "legal", "sustainability"],
        "description": "Courts, compliance, ESG mandates, regulatory changes",
    },
    "noise": {
        "types": ["entertainment", "sports", "lifestyle", "crime", "politics"],
        "description": "Non-business content — filtered before clustering",
    },
}

# Reverse mapping: event_type → domain
EVENT_TYPE_TO_DOMAIN = {}
for _domain, _info in EVENT_DOMAINS.items():
    for _etype in _info["types"]:
        EVENT_TYPE_TO_DOMAIN[_etype] = _domain
EVENT_TYPE_TO_DOMAIN["general"] = "unknown"
EVENT_TYPE_TO_DOMAIN["crisis"] = "corporate_action"  # crises are corporate events

# Sales intelligence buyer intent mapping (Bombora/6sense methodology).
# Event → buyer intent stage → recommended sales action.
EVENT_BUYER_INTENT = {
    "acquisition": {"intent": "growth_mode", "action": "capacity_sell"},
    "funding": {"intent": "new_budget", "action": "vendor_pitch"},
    "expansion": {"intent": "scaling", "action": "infrastructure_pitch"},
    "leadership_change": {"intent": "new_decision_maker", "action": "relationship_build"},
    "layoffs": {"intent": "cost_cutting", "action": "efficiency_pitch"},
    "technology": {"intent": "evaluating_solutions", "action": "demo_offer"},
    "regulation": {"intent": "compliance_mandate", "action": "compliance_solution"},
    "product_launch": {"intent": "competitive_response", "action": "competitive_intel"},
    "ipo": {"intent": "market_visibility", "action": "advisory_pitch"},
    "crisis": {"intent": "damage_control", "action": "crisis_management"},
    "supply_chain": {"intent": "resilience_building", "action": "supply_solution"},
    "partnership": {"intent": "ecosystem_building", "action": "partnership_pitch"},
    "market_entry": {"intent": "market_access", "action": "market_intelligence"},
    "earnings": {"intent": "performance_pressure", "action": "optimization_pitch"},
    "trade": {"intent": "trade_adaptation", "action": "cross_border_advisory"},
    "policy": {"intent": "policy_adaptation", "action": "consulting_advisory"},
    "legal": {"intent": "legal_exposure", "action": "compliance_advisory"},
}

# V10: Strong keyword patterns that override "general" classification.
# When k-NN fails (e.g., "Canada school shooting: OpenAI..." → "general" because
# OpenAI pushes toward technology anchors), these keyword sets catch obvious noise.
# Requires >= 1 match from "strong" OR >= 2 matches from "weak" to reclassify.
NOISE_KEYWORD_OVERRIDE = {
    "crime": {
        "strong": {"shooting", "murder case", "homicide", "robbery",
                   "kidnapping case", "serial killer", "rape case",
                   "mob lynching", "gang rape", "shot dead", "burnt to death"},
        "weak": {"suspect", "arrested", "police", "crime", "shooting",
                 "murder", "assault", "stabbing", "theft", "looted",
                 "smuggling", "trafficking", "accused", "victim"},
    },
    "politics": {
        "strong": {"succession to the throne", "line of succession",
                   "election results", "assembly elections", "panchayat election",
                   "political party", "opposition leader"},
        "weak": {"throne", "succession", "election", "elections", "ballot",
                 "voted", "ruling party", "opposition", "prince andrew",
                 "bjp", "congress party", "aap", "political", "mp minister"},
    },
    "entertainment": {
        "strong": {"box office collection", "movie review", "bollywood",
                   "bigg boss", "celebrity gossip", "trailer release",
                   "pokemon", "nintendo", "playstation", "xbox"},
        "weak": {"box office", "celebrity", "actor", "actress", "movie",
                 "film review", "ott release", "netflix series", "trailer",
                 "gaming", "gameplay", "eShop", "console", "album release"},
    },
    "lifestyle": {
        "strong": {"horoscope today", "zodiac sign", "morning habits",
                   "yoga poses", "skincare routine", "recipe"},
        "weak": {"horoscope", "zodiac", "wellness", "self-help", "yoga",
                 "recipe", "cooking", "astrology", "fitness routine"},
    },
    "sports": {
        "strong": {"ipl auction", "cricket match", "world cup qualifier",
                   "olympic medal", "premier league"},
        "weak": {"ipl", "cricket", "wicket", "innings", "football match",
                 "tournament", "championship", "medal", "kabaddi"},
    },
}


def reclassify_general_as_noise(article) -> Optional[str]:
    """Check if a 'general' article should be reclassified as noise.

    Returns the noise type if reclassified, None if it stays general.
    Uses NOISE_KEYWORD_OVERRIDE: 1 strong match OR 2+ weak matches.
    """
    text = f"{article.title}. {getattr(article, 'summary', '') or ''}".lower()
    for noise_type, keywords in NOISE_KEYWORD_OVERRIDE.items():
        strong_hits = sum(1 for kw in keywords["strong"] if kw in text)
        if strong_hits >= 1:
            return noise_type
        weak_hits = sum(1 for kw in keywords["weak"] if kw in text)
        if weak_hits >= 2:
            return noise_type
    return None


# Event type → CMI services mapping: which CMI services are relevant for each event type.
# Articles with event types mapping to 0 services are candidates for filtering.
EVENT_CMI_SERVICE_MAP = {
    "regulation": ["market_monitoring", "market_intelligence", "consulting_advisory"],
    "acquisition": ["competitive_intelligence", "market_intelligence"],
    "funding": ["market_intelligence", "competitive_intelligence"],
    "expansion": ["cross_border_expansion", "market_intelligence"],
    "crisis": ["market_monitoring", "consulting_advisory"],
    "supply_chain": ["procurement_intelligence", "industry_analysis"],
    "price_change": ["procurement_intelligence", "market_monitoring"],
    "technology": ["technology_research", "industry_analysis"],
    "partnership": ["competitive_intelligence", "market_intelligence"],
    "ipo": ["market_intelligence", "competitive_intelligence"],
    "consumer_shift": ["consumer_insights", "market_intelligence"],
    "market_entry": ["cross_border_expansion", "competitive_intelligence"],
    "leadership_change": ["competitive_intelligence"],
    "layoffs": ["industry_analysis", "market_monitoring"],
    "earnings": ["competitive_intelligence", "market_intelligence"],
    "market_movement": ["market_monitoring", "market_intelligence"],
    "infrastructure": ["market_intelligence", "industry_analysis"],
    "geopolitical": ["market_monitoring", "consulting_advisory", "cross_border_expansion"],
    "policy": ["market_monitoring", "consulting_advisory", "market_intelligence"],
    "sustainability": ["industry_analysis", "consulting_advisory"],
    "entertainment": [],  # Noise attractor — filtered before clustering
    "sports": [],         # Noise attractor — filtered before clustering
    "lifestyle": [],      # Noise attractor — filtered before clustering
    "crime": [],          # Noise attractor — filtered before clustering
    "politics": [],       # Noise attractor — filtered before clustering
    "general": [],  # No CMI service → candidate for filtering
    "economic_data": ["market_monitoring", "market_intelligence", "industry_analysis"],
    "legal": ["consulting_advisory", "market_monitoring", "competitive_intelligence"],
    "trade": ["cross_border_expansion", "market_intelligence", "procurement_intelligence"],
    "product_launch": ["competitive_intelligence", "market_intelligence", "technology_research"],
}

# Pre-extracted for quick access
EVENT_URGENCY = {etype: info["urgency"] for etype, info in EVENTS.items()}

# V6: Flatten all descriptions for embedding (used by classifier)
# Each event type → list of description strings
EVENT_DESCRIPTION_VARIANTS = {
    etype: info["descriptions"] for etype, info in EVENTS.items()
}

# V6: Keyword boost sets (lowercased for matching)
EVENT_KEYWORD_BOOST = {
    etype: {kw.lower() for kw in info.get("keyword_boost", [])}
    for etype, info in EVENTS.items()
}


class EmbeddingEventClassifier:
    """
    Semantic event classifier using k-NN anchor voting.

    V9: k-NN over ALL individual anchor embeddings instead of averaging
    per event type. Each anchor description stays as a separate reference
    point. Classification = majority vote among top-K nearest anchors.

    This avoids the centroid-averaging problem where diverse event types
    (e.g., "regulation" covering SEBI + data protection + labor codes)
    produce centroids that sit between subspaces and match nothing well.

    Architecture:
      1. Embed all anchor descriptions (10-12 per event type × 20 types = ~230)
      2. For each article, find top-K nearest anchor embeddings
      3. Weighted vote: closer anchors get more weight (similarity score)
      4. Keyword boost for disambiguation when top-2 event types are close
      5. LLM Tier 2 for ambiguous cases

    REF: SetFit (Tunstall et al. 2022) — same principle, without fine-tuning.
    """

    def __init__(self, embedding_tool, k: int = 7):
        self.embedding_tool = embedding_tool
        self.k = k  # Number of nearest anchors for voting
        # k-NN index: all individual anchor embeddings + their labels
        self._anchor_embeddings_norm = None  # (n_anchors, dim)
        self._anchor_labels = None           # [event_type_str] per anchor
        self._event_types = None
        # Legacy: centroid embeddings for backward compat (used by classify_batch)
        self._event_embeddings_norm = None
        # Config
        self._threshold: Optional[float] = None
        self._ambiguity_margin: Optional[float] = None
        # V8: Unmatched article buffer for taxonomy evolution
        self._unmatched_buffer: List[Any] = []
        self._unmatched_embeddings: List[np.ndarray] = []

    def _get_config(self) -> Tuple[float, float]:
        """Load classifier config from env (cached after first call)."""
        if self._threshold is not None:
            return self._threshold, self._ambiguity_margin
        try:
            from app.config import get_settings
            s = get_settings()
            self._threshold = s.event_classifier_threshold
            self._ambiguity_margin = s.event_classifier_ambiguity_margin
        except Exception:
            self._threshold = 0.35
            self._ambiguity_margin = 0.05
        return self._threshold, self._ambiguity_margin

    def _ensure_event_embeddings(self):
        """
        Compute and cache anchor embeddings for k-NN voting (once).

        V9: Keeps ALL individual anchor embeddings instead of averaging.
        V12: Merges static EVENTS anchors with learned anchors from previous runs.
        Also builds centroid embeddings for backward compatibility.
        """
        if self._anchor_embeddings_norm is not None:
            return

        self._event_types = list(EVENT_DESCRIPTION_VARIANTS.keys())

        # V12: Load learned anchors and merge with static ones
        learned_anchors = self.load_learned_anchor_titles()
        n_learned = sum(len(v) for v in learned_anchors.values())
        if n_learned:
            logger.info(
                f"Loaded {n_learned} learned anchors across "
                f"{len(learned_anchors)} event types"
            )

        # Embed all variants for all event types (static + learned)
        all_descs: List[str] = []
        anchor_labels: List[str] = []
        variant_counts: List[int] = []
        for etype in self._event_types:
            # Static anchors from EVENTS dict
            variants = list(EVENT_DESCRIPTION_VARIANTS[etype])
            # Merge learned anchors (avoid duplicates)
            if etype in learned_anchors:
                existing_set = set(v.lower()[:100] for v in variants)
                for learned_title in learned_anchors[etype]:
                    if learned_title.lower()[:100] not in existing_set:
                        variants.append(learned_title)
                        existing_set.add(learned_title.lower()[:100])
            all_descs.extend(variants)
            anchor_labels.extend([etype] * len(variants))
            variant_counts.append(len(variants))

        raw_all = np.array(self.embedding_tool.embed_batch(all_descs))

        # k-NN index: all individual anchors, normalized
        norms_all = np.linalg.norm(raw_all, axis=1, keepdims=True)
        norms_all[norms_all == 0] = 1
        self._anchor_embeddings_norm = raw_all / norms_all
        self._anchor_labels = anchor_labels

        # Also build centroids for backward compatibility (LLM tier 2, etc.)
        averaged = []
        idx = 0
        for count in variant_counts:
            chunk = raw_all[idx:idx + count]
            avg = np.mean(chunk, axis=0)
            averaged.append(avg)
            idx += count
        event_embs = np.array(averaged)
        norms = np.linalg.norm(event_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._event_embeddings_norm = event_embs / norms

        total_variants = sum(variant_counts)

        # V12: Build domain centroids for 2-stage classification
        # Domain centroid = average of all anchor embeddings for types in that domain
        self._domain_names = list(EVENT_DOMAINS.keys())
        domain_centroids = []
        self._domain_type_indices = {}  # domain → list of anchor indices
        for domain_name, domain_info in EVENT_DOMAINS.items():
            domain_types = set(domain_info["types"])
            # Find all anchor indices belonging to this domain
            indices = [
                i for i, label in enumerate(self._anchor_labels)
                if label in domain_types
            ]
            self._domain_type_indices[domain_name] = indices
            if indices:
                domain_embs = self._anchor_embeddings_norm[indices]
                centroid = domain_embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                domain_centroids.append(centroid)
            else:
                # Domain has no types in EVENTS dict (shouldn't happen)
                domain_centroids.append(np.zeros(event_embs.shape[1]))
        self._domain_centroids_norm = np.array(domain_centroids)

        logger.debug(
            f"Event embeddings computed: {len(self._event_types)} types, "
            f"{total_variants} total variants, "
            f"{len(self._domain_names)} domains, dim={event_embs.shape[1]}"
        )

    def _keyword_boost_score(self, text_lower: str, event_type: str) -> float:
        """
        V6: Check if article text contains keywords for an event type.

        Returns a small boost (0.0 to 0.08) based on keyword matches.
        This helps disambiguate when embedding scores are close.
        """
        keywords = EVENT_KEYWORD_BOOST.get(event_type, set())
        if not keywords:
            return 0.0
        matches = sum(1 for kw in keywords if kw in text_lower)
        # Cap at 0.12 boost (3+ keyword matches = max boost)
        return min(matches * 0.04, 0.12)

    def classify_batch(
        self, articles: list, threshold: float = None,
    ) -> Dict[str, int]:
        """
        V9: k-NN anchor voting + keyword boost + LLM Tier 2.

        Instead of comparing articles to centroid averages, finds the K nearest
        individual anchor descriptions and uses weighted majority voting.
        This preserves the full diversity of each event type's anchor space.

        Architecture:
          1. Embed all article titles+summaries (batched, single API call)
          2. Compute cosine sim against ALL ~230 individual anchors
          3. For each article, find top-K nearest anchors
          4. Weighted vote: sum similarity scores per event type among top-K
          5. Keyword boost for disambiguation when top-2 types are close
          6. LLM Tier 2 for low-confidence classifications

        Args:
            articles: List of NewsArticle objects.
            threshold: Minimum weighted vote score to assign a specific event type.
                       If None, uses env EVENT_CLASSIFIER_THRESHOLD (default 0.40).
        """
        self._ensure_event_embeddings()
        env_threshold, ambiguity_margin = self._get_config()
        if threshold is None:
            threshold = env_threshold

        # Tier 2 LLM threshold: articles below this get LLM validation
        llm_threshold = threshold + 0.12

        # Embed title + summary for richer context
        texts = [
            f"{a.title}. {a.summary or ''}"[:500] for a in articles
        ]
        title_embs = np.array(self.embedding_tool.embed_batch(texts))
        norms = np.linalg.norm(title_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        title_embs_norm = title_embs / norms

        # Cosine similarity: (n_articles, n_anchors) — ALL individual anchors
        anchor_sims = np.dot(title_embs_norm, self._anchor_embeddings_norm.T)

        distribution: Dict[str, int] = {}
        confidences: List[float] = []
        disambiguated_count = 0
        ambiguous_indices: List[int] = []

        for i, article in enumerate(articles):
            # k-NN: find top-K nearest anchors
            top_k_indices = np.argsort(anchor_sims[i])[-self.k:][::-1]
            top_k_sims = anchor_sims[i][top_k_indices]
            top_k_labels = [self._anchor_labels[idx] for idx in top_k_indices]

            # Weighted vote: sum similarity scores per event type
            vote_scores: Dict[str, float] = {}
            vote_counts: Dict[str, int] = {}
            for label, sim in zip(top_k_labels, top_k_sims):
                vote_scores[label] = vote_scores.get(label, 0.0) + float(sim)
                vote_counts[label] = vote_counts.get(label, 0) + 1

            # Normalize by count to get average similarity per type
            avg_scores = {
                label: vote_scores[label] / vote_counts[label]
                for label in vote_scores
            }

            # Sort by total vote score (not average — gives weight to types with more matches)
            sorted_types = sorted(vote_scores.keys(), key=lambda t: vote_scores[t], reverse=True)
            best_type = sorted_types[0]
            best_score = avg_scores[best_type]
            best_vote = vote_scores[best_type]

            second_type = sorted_types[1] if len(sorted_types) > 1 else None
            second_score = avg_scores.get(second_type, 0.0) if second_type else 0.0

            text_lower = f"{article.title}. {getattr(article, 'summary', '') or ''}".lower()

            if best_score >= threshold:
                etype = best_type

                # Disambiguation: if top-2 types are close, use keyword boost
                if second_type and (best_score - second_score) < ambiguity_margin:
                    boost_best = self._keyword_boost_score(text_lower, etype)
                    boost_second = self._keyword_boost_score(text_lower, second_type)
                    if boost_second > boost_best:
                        logger.debug(
                            f"Disambiguated '{article.title[:50]}': "
                            f"{etype}({best_score:.3f}+{boost_best:.3f}) → "
                            f"{second_type}({second_score:.3f}+{boost_second:.3f})"
                        )
                        etype = second_type
                        best_score = second_score + boost_second
                        disambiguated_count += 1
                    else:
                        best_score += boost_best

                urgency = EVENT_URGENCY[etype]

                # Flag for LLM validation if confidence is borderline
                is_ambiguous = (
                    best_score < llm_threshold
                    or (second_type and (best_score - second_score) < ambiguity_margin * 2)
                )
                if is_ambiguous:
                    ambiguous_indices.append(i)

                reasoning = (
                    f"k-NN vote: {etype} (avg_sim={best_score:.2f}, "
                    f"votes={vote_counts.get(etype, 0)}/{self.k})"
                )
            else:
                etype = "general"
                urgency = 0.30
                reasoning = f"Below threshold ({best_score:.2f} < {threshold})"
                ambiguous_indices.append(i)
                self._unmatched_buffer.append(article)
                self._unmatched_embeddings.append(title_embs_norm[i])

            confidences.append(best_score)

            # Store classification with reasoning for downstream use
            article._trigger_event = etype
            article._trigger_urgency = urgency
            article._trigger_confidence = best_score
            article._trigger_reasoning = reasoning
            article._cmi_services = EVENT_CMI_SERVICE_MAP.get(etype, [])

            # V12: Add domain and buyer intent metadata
            domain = EVENT_TYPE_TO_DOMAIN.get(etype, "unknown")
            article._trigger_domain = domain
            buyer_intent = EVENT_BUYER_INTENT.get(etype, {})
            article._trigger_buyer_intent = buyer_intent.get("intent", "")
            article._trigger_sales_action = buyer_intent.get("action", "")
            article._trigger_intent = (
                f"{etype} event detected (confidence: {best_score:.2f}, "
                f"domain: {domain})"
                if etype != "general"
                else "General business news"
            )

            if hasattr(article, 'trend_types') and isinstance(article.trend_types, list):
                if etype not in [str(t) for t in article.trend_types]:
                    article.trend_types.append(etype)

            distribution[etype] = distribution.get(etype, 0) + 1

        # V10: Post-classification noise keyword sweep
        # Articles stuck as "general" get re-checked against strong noise keywords.
        # This catches cases where k-NN fails (e.g., "school shooting: OpenAI..."
        # → "general" because OpenAI pushes toward technology anchors).
        noise_reclassified = 0
        for article in articles:
            if getattr(article, '_trigger_event', '') != 'general':
                continue
            new_type = reclassify_general_as_noise(article)
            if new_type:
                old_count = distribution.get("general", 0)
                if old_count > 0:
                    distribution["general"] = old_count - 1
                distribution[new_type] = distribution.get(new_type, 0) + 1
                article._trigger_event = new_type
                article._trigger_urgency = 0.0
                article._trigger_confidence = getattr(article, '_trigger_confidence', 0.3)
                article._trigger_reasoning = (
                    f"Keyword noise sweep: reclassified general → {new_type}"
                )
                article._cmi_services = []
                article._trigger_intent = f"{new_type} noise (keyword override)"
                noise_reclassified += 1
                logger.debug(
                    f"Noise sweep: '{(article.title or '')[:60]}' → {new_type}"
                )
        if noise_reclassified:
            logger.info(
                f"Noise keyword sweep: {noise_reclassified} articles "
                f"reclassified from general → noise"
            )

        # V9: Structured logging with stats
        avg_conf = np.mean(confidences) if confidences else 0.0
        general_pct = distribution.get("general", 0) / max(len(articles), 1) * 100
        logger.info(
            f"Event classification (V10 k-NN+sweep, k={self.k}): {distribution} | "
            f"threshold={threshold:.2f}, avg_confidence={avg_conf:.3f}, "
            f"general={general_pct:.0f}%, disambiguated={disambiguated_count}, "
            f"noise_swept={noise_reclassified}, "
            f"ambiguous={len(ambiguous_indices)} (queued for LLM Tier 2)"
        )

        # Store ambiguous indices for async LLM validation (called separately)
        self._ambiguous_indices = ambiguous_indices
        self._ambiguous_articles = [articles[i] for i in ambiguous_indices] if ambiguous_indices else []

        return distribution

    async def classify_ambiguous_with_llm(self, articles: list = None) -> int:
        """
        V7 Tier 2: LLM-validate ambiguous classifications.

        Called AFTER classify_batch() to refine articles that Tier 1 couldn't
        classify confidently. Uses LLM with event type definitions to get
        classification + reasoning.

        Args:
            articles: Articles to validate. If None, uses self._ambiguous_articles
                      from the last classify_batch() call.

        Returns:
            Number of articles reclassified by LLM.
        """
        if articles is None:
            articles = getattr(self, '_ambiguous_articles', [])

        if not articles:
            return 0

        # Cap LLM calls to avoid cost explosion
        from app.config import get_settings
        max_llm_calls = min(len(articles), get_settings().event_max_llm_calls)

        # Sort by confidence ascending — most ambiguous first (closest to decision
        # boundary). These benefit most from LLM validation. High-confidence
        # ambiguous articles likely have correct k-NN labels already.
        articles = sorted(
            articles,
            key=lambda a: getattr(a, '_trigger_confidence', 0.5),
        )
        articles = articles[:max_llm_calls]

        try:
            from app.tools.llm_service import LLMService
            llm = LLMService(lite=True)
        except Exception as e:
            logger.warning(f"Tier 2 LLM unavailable: {e}")
            return 0

        event_types_desc = "\n".join(
            f"- {etype}: urgency={info['urgency']}, keywords: {', '.join(info.get('keyword_boost', [])[:5])}"
            for etype, info in EVENTS.items()
        )

        import asyncio
        semaphore = asyncio.Semaphore(1)  # Sequential: prevents 429 cascade
        reclassified = 0
        consecutive_failures = 0
        max_consecutive_failures = 5  # Early exit if providers are exhausted

        async def _validate_one(article) -> bool:
            nonlocal reclassified, consecutive_failures
            if consecutive_failures >= max_consecutive_failures:
                return False  # All providers likely exhausted, stop wasting calls
            async with semaphore:
                # Rate limit protection: 1s delay between calls
                await asyncio.sleep(1.0)
                prompt = f"""Classify this news article into ONE event type.

ARTICLE TITLE: {article.title}
ARTICLE SUMMARY: {(getattr(article, 'summary', '') or '')[:300]}

EVENT TYPES:
{event_types_desc}

If none fit well, use "general".

Respond with JSON:
{{
    "event_type": "<type>",
    "confidence": <0.0-1.0>,
    "reasoning": "<1-2 sentences why>"
}}"""

                try:
                    result = await llm.generate_json(prompt=prompt)
                    new_etype = result.get("event_type", "general")
                    new_confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "")

                    if new_etype in EVENT_URGENCY and new_confidence >= 0.4:
                        old_etype = getattr(article, '_trigger_event', 'general')
                        if old_etype != new_etype:
                            logger.debug(
                                f"Tier 2 reclassified '{article.title[:50]}': "
                                f"{old_etype} → {new_etype} ({new_confidence:.2f})"
                            )
                            reclassified += 1

                        article._trigger_event = new_etype
                        article._trigger_urgency = EVENT_URGENCY[new_etype]
                        article._trigger_confidence = max(
                            getattr(article, '_trigger_confidence', 0), new_confidence
                        )
                        article._trigger_reasoning = f"LLM validated: {reasoning}"
                        article._trigger_intent = f"{new_etype} event (LLM: {new_confidence:.2f})"
                        consecutive_failures = 0  # Reset on success
                        return True
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"Tier 2: {consecutive_failures} consecutive failures — "
                            f"providers likely exhausted, skipping remaining {len(articles)} articles"
                        )
                    else:
                        logger.debug(f"Tier 2 failed for '{article.title[:40]}': {e}")
                return False

        tasks = [_validate_one(a) for a in articles]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            f"Event classification (Tier 2 LLM): {len(articles)} validated, "
            f"{reclassified} reclassified"
        )
        return reclassified

    # ═══════════════════════════════════════════════════════════════════
    # V8: EVOLVING TAXONOMY — discover new event categories from data
    # ═══════════════════════════════════════════════════════════════════

    def discover_taxonomy_candidates(
        self,
        min_buffer_size: int = 15,
        min_cluster_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """Cluster unmatched articles to discover candidate new event types.

        Called after classify_batch(). If enough articles fell below the
        classification threshold ("general"), clusters them with mini-Leiden
        to find coherent groups that might represent new event types.

        Args:
            min_buffer_size: Minimum unmatched articles before attempting.
            min_cluster_size: Minimum articles per discovered cluster.

        Returns:
            List of candidate dicts with centroid, article titles, count.
        """
        if len(self._unmatched_buffer) < min_buffer_size:
            return []

        if not self._unmatched_embeddings:
            return []

        emb_array = np.array(self._unmatched_embeddings)

        # Mini-Leiden clustering on unmatched articles
        try:
            from app.trends.clustering import cluster_leiden
            labels, _, metrics = cluster_leiden(
                emb_array,
                k=min(10, len(emb_array) - 1),
                resolution=1.5,  # Higher resolution = more, smaller clusters
                seed=42,
                min_community_size=min_cluster_size,
            )
        except Exception as e:
            logger.debug(f"Taxonomy discovery clustering failed: {e}")
            return []

        n_clusters = metrics.get("n_clusters", 0)
        if n_clusters == 0:
            return []

        # Build candidate clusters
        candidates = []
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() < min_cluster_size:
                continue

            cluster_articles = [
                self._unmatched_buffer[i]
                for i in range(len(labels))
                if labels[i] == cid
            ]
            cluster_embs = emb_array[mask]
            centroid = cluster_embs.mean(axis=0)

            candidates.append({
                "article_titles": [a.title for a in cluster_articles[:10]],
                "article_count": len(cluster_articles),
                "centroid": centroid.tolist(),
                "proposed_name": None,  # Filled by LLM naming
            })

        logger.info(
            f"Taxonomy discovery: {len(self._unmatched_buffer)} unmatched → "
            f"{n_clusters} clusters, {len(candidates)} candidates"
        )
        return candidates

    async def name_taxonomy_candidates(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to name discovered taxonomy candidates (1 call total).

        Args:
            candidates: Output from discover_taxonomy_candidates().

        Returns:
            Same candidates list with proposed_name filled in.
        """
        if not candidates:
            return candidates

        try:
            from app.tools.llm_service import LLMService
            llm = LLMService(lite=True)
        except Exception as e:
            logger.warning(f"LLM unavailable for taxonomy naming: {e}")
            return candidates

        # Build a single prompt with all clusters
        cluster_descriptions = []
        for i, c in enumerate(candidates):
            titles = "\n".join(f"  - {t}" for t in c["article_titles"][:5])
            cluster_descriptions.append(
                f"Cluster {i+1} ({c['article_count']} articles):\n{titles}"
            )

        prompt = f"""These article clusters don't fit existing event categories.
Propose a short event type name (1-2 words, snake_case) for each cluster.

Existing event types: {', '.join(EVENTS.keys())}

{chr(10).join(cluster_descriptions)}

Respond with JSON:
{{
    "clusters": [
        {{"cluster": 1, "name": "proposed_name", "urgency": 0.5, "reasoning": "why"}}
    ]
}}"""

        try:
            result = await llm.generate_json(prompt=prompt)
            cluster_names = result.get("clusters", [])
            for entry in cluster_names:
                idx = entry.get("cluster", 0) - 1
                if 0 <= idx < len(candidates):
                    candidates[idx]["proposed_name"] = entry.get("name", "unknown")
                    candidates[idx]["proposed_urgency"] = entry.get("urgency", 0.5)
                    candidates[idx]["reasoning"] = entry.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Taxonomy naming failed: {e}")

        return candidates

    def save_taxonomy_candidates(
        self,
        candidates: List[Dict[str, Any]],
        path: Path = TAXONOMY_CANDIDATES_PATH,
    ) -> int:
        """Persist taxonomy candidates to JSONL for cross-run tracking.

        Returns number of candidates saved.
        """
        if not candidates:
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone

        saved = 0
        try:
            with open(path, "a", encoding="utf-8") as f:
                for c in candidates:
                    if not c.get("proposed_name"):
                        continue
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "proposed_name": c["proposed_name"],
                        "proposed_urgency": c.get("proposed_urgency", 0.5),
                        "article_count": c["article_count"],
                        "article_titles": c["article_titles"][:5],
                        "reasoning": c.get("reasoning", ""),
                        "centroid": c.get("centroid", [])[:10],  # Truncated for storage
                    }
                    f.write(json.dumps(record, default=str) + "\n")
                    saved += 1
            logger.info(f"Saved {saved} taxonomy candidates to {path}")
        except Exception as e:
            logger.warning(f"Failed to save taxonomy candidates: {e}")

        return saved

    @staticmethod
    def load_taxonomy_candidates(
        path: Path = TAXONOMY_CANDIDATES_PATH,
    ) -> List[Dict[str, Any]]:
        """Load taxonomy candidates from JSONL."""
        if not path.exists():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass
        return records

    @staticmethod
    def evolve_taxonomy(
        min_persistence: int = 3,
        path: Path = TAXONOMY_CANDIDATES_PATH,
    ) -> List[Dict[str, Any]]:
        """Promote taxonomy candidates seen in min_persistence+ consecutive runs.

        Scans candidate history to find proposed event types that appear
        repeatedly. These are strong signals for genuinely missing categories.

        Args:
            min_persistence: Minimum consecutive runs a candidate must appear.
            path: Path to taxonomy candidates JSONL.

        Returns:
            List of promoted candidates ready for review.
        """
        candidates = EmbeddingEventClassifier.load_taxonomy_candidates(path)
        if not candidates:
            return []

        # Count occurrences of each proposed name
        from collections import Counter
        name_counts = Counter(c.get("proposed_name", "") for c in candidates)

        promoted = []
        seen_names = set()
        for name, count in name_counts.items():
            if not name or name in seen_names:
                continue
            if count >= min_persistence:
                # Get the most recent record for this name
                latest = next(
                    c for c in reversed(candidates)
                    if c.get("proposed_name") == name
                )
                promoted.append({
                    "name": name,
                    "persistence": count,
                    "urgency": latest.get("proposed_urgency", 0.5),
                    "reasoning": latest.get("reasoning", ""),
                    "article_titles": latest.get("article_titles", []),
                })
                seen_names.add(name)

        if promoted:
            logger.info(
                f"Taxonomy evolution: {len(promoted)} candidates promoted "
                f"(seen {min_persistence}+ times): "
                f"{[p['name'] for p in promoted]}"
            )

        return promoted

    # ════════════════════════════════════════════════════════════════════
    # V12: Self-Evolving Anchor System
    # ════════════════════════════════════════════════════════════════════
    #
    # After each pipeline run, high-confidence classified articles become
    # new anchors for future classification. This creates a self-reinforcing
    # loop: more articles → better anchors → better classification.
    #
    # Architecture:
    #   1. After classify_batch(), collect articles with confidence > 0.65
    #   2. For each event type, keep the top-20 highest-confidence article titles
    #   3. Save to data/learned_anchors.json (persistent across runs)
    #   4. On next startup, merge learned anchors with static EVENTS anchors
    #   5. Decay: learned anchors older than 14 days lose priority
    #
    # REF: SetFit continual learning, NewsAPI.ai sentence-level training data
    # ════════════════════════════════════════════════════════════════════

    LEARNED_ANCHORS_PATH = Path("./data/learned_anchors.json")

    def collect_learned_anchors(
        self,
        articles: list,
        min_confidence: float = 0.65,
        max_per_type: int = 20,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect high-confidence classifications as future anchors.

        Called after classify_batch(). Saves the best-classified articles
        per event type so future runs have richer anchor coverage.

        Self-learning loop:
            classify_batch() → collect_learned_anchors() → save → load on next run
            → richer anchors → better classify_batch() → ...

        Args:
            articles: Articles that were just classified by classify_batch().
            min_confidence: Minimum classification confidence to qualify.
            max_per_type: Maximum learned anchors per event type.

        Returns:
            Dict of event_type → list of learned anchor records.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        learned: Dict[str, List[Dict[str, Any]]] = {}

        for art in articles:
            event_type = getattr(art, "_trigger_event", None)
            confidence = getattr(art, "_trigger_confidence", 0.0)

            # Skip noise types and low-confidence classifications
            if not event_type or event_type in NOISE_EVENT_TYPES or event_type == "general":
                continue
            if confidence < min_confidence:
                continue

            title = getattr(art, "title", "")
            if not title or len(title) < 15:
                continue

            if event_type not in learned:
                learned[event_type] = []

            learned[event_type].append({
                "title": title[:200],
                "confidence": round(float(confidence), 4),
                "timestamp": now,
                "source": getattr(art, "source", "unknown"),
            })

        # Keep only top-N per type by confidence
        for etype in learned:
            learned[etype] = sorted(
                learned[etype], key=lambda x: -x["confidence"]
            )[:max_per_type]

        return learned

    def save_learned_anchors(
        self,
        new_anchors: Dict[str, List[Dict[str, Any]]],
        path: Path = None,
        max_per_type: int = 20,
        max_age_days: int = 14,
    ) -> int:
        """Merge new learned anchors with existing ones and save.

        Implements:
        - Dedup by title (no duplicate anchor descriptions)
        - Recency decay: anchors older than max_age_days are pruned
        - Capacity: max_per_type anchors per event type (newest + highest confidence first)

        Returns:
            Total number of learned anchors saved.
        """
        from datetime import datetime, timezone, timedelta

        path = path or self.LEARNED_ANCHORS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing
        existing = self._load_learned_anchors_raw(path)

        # Merge
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        for etype, anchors in new_anchors.items():
            if etype not in existing:
                existing[etype] = []
            existing[etype].extend(anchors)

        # Dedup + prune + cap
        total_saved = 0
        for etype in list(existing.keys()):
            seen_titles = set()
            deduped = []
            for a in existing[etype]:
                title = a.get("title", "")
                ts = a.get("timestamp", "")
                if title in seen_titles:
                    continue
                # Prune old anchors
                if ts and ts < cutoff:
                    continue
                seen_titles.add(title)
                deduped.append(a)

            # Sort by confidence desc, cap at max_per_type
            existing[etype] = sorted(
                deduped, key=lambda x: -x.get("confidence", 0)
            )[:max_per_type]
            total_saved += len(existing[etype])

        # Save
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, default=str)
            logger.info(
                f"Learned anchors saved: {total_saved} across "
                f"{len(existing)} event types → {path}"
            )
        except Exception as e:
            logger.warning(f"Failed to save learned anchors: {e}")

        return total_saved

    @staticmethod
    def _load_learned_anchors_raw(path: Path = None) -> Dict[str, List[Dict[str, Any]]]:
        """Load raw learned anchors from JSON file."""
        path = path or EmbeddingEventClassifier.LEARNED_ANCHORS_PATH
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def load_learned_anchor_titles(
        path: Path = None,
        max_per_type: int = 10,
    ) -> Dict[str, List[str]]:
        """Load learned anchor titles for merging with static EVENTS anchors.

        Returns dict of event_type → list of title strings (ready to embed).
        These get merged with EVENT_DESCRIPTION_VARIANTS at classification time.
        """
        raw = EmbeddingEventClassifier._load_learned_anchors_raw(path)
        result = {}
        for etype, anchors in raw.items():
            titles = [a["title"] for a in anchors[:max_per_type] if a.get("title")]
            if titles:
                result[etype] = titles
        return result

    def get_event_effectiveness(
        self,
        articles: list,
        labels: "np.ndarray",
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-event-type effectiveness: what % end up in non-noise clusters?

        Args:
            articles: All articles (with _trigger_event set).
            labels: Cluster labels (-1 = noise).

        Returns:
            {event_type: {total: N, clustered: N, effectiveness: float}}
        """
        from collections import defaultdict
        stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "clustered": 0}
        )

        for article, label in zip(articles, labels):
            etype = getattr(article, '_trigger_event', 'general')
            stats[etype]["total"] += 1
            if int(label) >= 0:
                stats[etype]["clustered"] += 1

        result = {}
        for etype, s in stats.items():
            result[etype] = {
                "total": s["total"],
                "clustered": s["clustered"],
                "effectiveness": round(
                    s["clustered"] / max(s["total"], 1), 3
                ),
            }

        return result
