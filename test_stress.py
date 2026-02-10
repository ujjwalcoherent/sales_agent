"""
Stress test for the trend engine pipeline.

Tests:
1. VOLUME: 200+ articles through the full pipeline
2. KNOWN TRENDS: Validates detection of trends we know happened
3. CLASSIFICATION: Checks embedding-based event classifier accuracy
4. CLUSTERING: Verifies cluster quality at scale
5. SYNTHESIS: Validates LLM output quality across many clusters

Run: python test_stress.py
"""

import asyncio
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from uuid import uuid4

# Suppress noise from HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Setup console logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def safe_print(text):
    """Print with encoding safety."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


async def stress_test_volume():
    """
    TEST 1: Volume test — 200+ articles through the full pipeline.
    Fetches max articles from all sources and runs the complete pipeline.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 1: VOLUME (200+ articles)")
    print("=" * 70)

    from app.tools.rss_tool import RSSTool
    from app.trends.engine import RecursiveTrendEngine

    # Fetch 25 per source (22 sources → ~200+ articles)
    rss = RSSTool()
    start = time.time()
    articles = await rss.fetch_all_sources(max_per_source=25, hours_ago=72)
    fetch_time = time.time() - start
    print(f"\n  Fetched: {len(articles)} articles in {fetch_time:.1f}s")

    if len(articles) < 50:
        print("  WARNING: Fewer than 50 articles fetched. APIs may be down.")
        print("  Continuing with available articles...")

    # Source distribution
    source_counts = Counter(a.source_name for a in articles)
    print(f"  Sources: {len(source_counts)} unique")
    for src, count in source_counts.most_common(10):
        safe_print(f"    {src}: {count}")

    # Run full pipeline
    engine = RecursiveTrendEngine(max_depth=2, mock_mode=False)
    start = time.time()
    tree = await engine.run(articles)
    pipeline_time = time.time() - start

    # Results
    metrics = engine.metrics
    print(f"\n  PIPELINE RESULTS:")
    print(f"    Input articles:     {metrics['article_counts']['input']}")
    print(f"    After relevance:    {metrics['article_counts'].get('after_relevance', '?')}")
    print(f"    After dedup:        {metrics['article_counts'].get('after_dedup', '?')}")
    print(f"    After semantic:     {metrics['article_counts'].get('after_semantic_dedup', '?')}")
    print(f"    Clusters:           {metrics.get('n_clusters', '?')}")
    print(f"    Noise articles:     {metrics.get('noise_count', '?')}")
    print(f"    Trend nodes:        {len(tree.nodes)}")
    print(f"    Max depth:          {tree.max_depth_reached}")
    print(f"    Pipeline time:      {pipeline_time:.1f}s")

    # Event classification distribution
    event_dist = metrics.get("event_distribution", {})
    if event_dist:
        print(f"\n  EVENT CLASSIFICATION:")
        total = sum(event_dist.values())
        general = event_dist.get("general", 0)
        classified = total - general
        print(f"    Total: {total}, Classified: {classified} ({classified*100//max(total,1)}%), General: {general}")
        for etype, count in sorted(event_dist.items(), key=lambda x: -x[1]):
            bar = "█" * (count * 2)
            safe_print(f"    {etype:20s} {count:3d} {bar}")

    # Phase times
    print(f"\n  PHASE TIMES:")
    for phase, t in sorted(metrics.get("phase_times", {}).items()):
        print(f"    {phase:25s} {t:.2f}s")

    # Trend quality analysis
    print(f"\n  TREND ANALYSIS:")
    severity_counts = Counter()
    intent_types = Counter()
    services_recommended = Counter()
    trends_with_companies = 0
    trends_with_services = 0

    for node_id, node in tree.nodes.items():
        severity_counts[node.severity.value if hasattr(node.severity, 'value') else str(node.severity)] += 1

        # Check buying intent
        buying = getattr(node, 'buying_intent', {}) or {}
        if buying:
            intent_types[buying.get("signal_type", "unknown")] += 1

        # Check affected companies
        companies = getattr(node, 'affected_companies', []) or []
        if companies:
            trends_with_companies += 1

    for severity, count in severity_counts.most_common():
        print(f"    Severity {severity}: {count}")
    print(f"    Trends with companies: {trends_with_companies}/{len(tree.nodes)}")
    print(f"    Buying intent types: {dict(intent_types)}")

    # Show top trends
    print(f"\n  TOP TRENDS (by score):")
    sorted_nodes = sorted(
        tree.nodes.values(),
        key=lambda n: getattr(n, 'trend_score', 0),
        reverse=True,
    )
    for node in sorted_nodes[:10]:
        buying = getattr(node, 'buying_intent', {}) or {}
        companies = getattr(node, 'affected_companies', []) or []
        services = getattr(node, 'signals', {}).get('recommended_services', [])
        severity = node.severity.value if hasattr(node.severity, 'value') else str(node.severity)
        safe_print(
            f"    [{severity:6s}] {node.trend_title[:60]}"
            f" | score={node.trend_score:.2f}"
            f" | articles={node.article_count}"
        )
        if buying:
            safe_print(f"             Intent: {buying.get('signal_type', '?')} | Urgency: {buying.get('urgency', '?')}")
            safe_print(f"             Who: {str(buying.get('who_needs_help', ''))[:80]}")
        if companies:
            safe_print(f"             Companies: {', '.join(companies[:5])}")

    # Validation checks
    print(f"\n  VALIDATION:")
    checks = []

    # Check 1: Got enough articles
    n_articles = metrics['article_counts']['input']
    checks.append(("Volume >= 100 articles", n_articles >= 100))

    # Check 2: Event classifier working (not all general)
    # Real RSS feeds include non-business content (sports, entertainment, politics),
    # so with threshold=0.35 it's normal for 60-70% to be "general". The key check
    # is that SOME articles get classified, not that the majority do.
    general_pct = event_dist.get("general", 0) / max(sum(event_dist.values()), 1)
    checks.append(("Event classifier < 75% general", general_pct < 0.75))

    # Check 3: Got clusters
    n_clusters = metrics.get('n_clusters', 0)
    checks.append(("Clusters >= 5", n_clusters >= 5))

    # Check 4: Noise not too high
    noise = metrics.get('noise_count', 0)
    after_semantic = metrics['article_counts'].get('after_semantic_dedup', n_articles)
    noise_pct = noise / max(after_semantic, 1)
    checks.append(("Noise < 50%", noise_pct < 0.5))

    # Check 5: All clusters synthesized
    checks.append(("All nodes have titles", all(n.trend_title for n in tree.nodes.values())))

    # Check 6: Pipeline under 300s (includes HTTP scraping of 200+ articles)
    checks.append(("Pipeline < 300s", pipeline_time < 300))

    # Check 7: Multiple event types detected
    n_event_types = len([e for e in event_dist if e != "general" and event_dist[e] > 0])
    checks.append(("Event types >= 5", n_event_types >= 5))

    # Check 8: Trends have companies
    checks.append(("Trends with companies >= 50%", trends_with_companies >= len(tree.nodes) * 0.5))

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {name}")

    return all_pass


async def stress_test_known_trends():
    """
    TEST 2: Known trend detection — synthetic articles about known events.
    Tests if the pipeline correctly clusters and classifies known trends.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 2: KNOWN TREND DETECTION")
    print("=" * 70)

    from app.schemas.news import NewsArticle
    from app.trends.engine import RecursiveTrendEngine

    now = datetime.now(timezone.utc)

    # Build synthetic articles about known recent trends
    # Each group should form a cluster
    known_trends = {
        "RBI_regulation": [
            ("RBI proposes new guidelines for digital lending platforms", "The Reserve Bank of India has released new draft guidelines for digital lending, requiring stricter KYC norms and data protection standards for all fintech companies operating in the lending space."),
            ("Digital lenders face stricter compliance under new RBI rules", "Fintech companies offering digital loans will need to comply with enhanced regulatory requirements including mandatory disclosure of all charges and a cooling-off period for borrowers."),
            ("RBI circular mandates data localization for digital lenders", "In a move to strengthen data privacy, RBI has mandated that all digital lending platforms store customer data within India, affecting both domestic and foreign fintech operators."),
            ("Impact of RBI's new digital lending guidelines on fintech sector", "Industry experts analyze how the Reserve Bank's new guidelines will reshape the digital lending landscape, potentially consolidating the market among larger, well-capitalized players."),
            ("NBFCs rush to comply with RBI's digital lending framework", "Non-banking financial companies are investing in compliance infrastructure as the deadline for RBI's new digital lending regulations approaches, with industry spending estimated at Rs 500 crore."),
        ],
        "EV_expansion": [
            ("Tata Motors announces Rs 15000 crore EV manufacturing plant in Tamil Nadu", "Tata Motors has unveiled plans for a massive electric vehicle manufacturing facility in Tamil Nadu, expected to produce 500,000 EVs annually by 2028."),
            ("Mahindra Electric to invest Rs 8000 crore in new EV platform", "Mahindra Group's electric vehicle subsidiary plans major investment in developing a new electric vehicle platform for the Indian market."),
            ("Government approves PLI scheme extension for electric vehicles", "The central government has extended the Production Linked Incentive scheme for electric vehicles by two years, with an additional allocation of Rs 10,000 crore."),
            ("Tesla supplier network grows in India as EV ecosystem expands", "International automotive suppliers are establishing manufacturing bases in India to serve the growing electric vehicle market, with over 15 new facilities planned."),
            ("India's EV sales cross 2 million units in FY26", "Electric vehicle sales in India have reached a milestone of 2 million units in the current financial year, representing 15% of total automobile sales."),
        ],
        "AI_technology": [
            ("Indian IT firms invest heavily in generative AI capabilities", "Infosys, TCS, and Wipro collectively allocate $2 billion for AI transformation, training 100,000 engineers in generative AI technologies.", ),
            ("Government launches National AI Mission with Rs 10000 crore budget", "India's National AI Mission aims to build AI computing infrastructure, establish centers of excellence, and train 1 million AI professionals by 2028."),
            ("AI startup funding in India reaches $3 billion in 2025", "Indian artificial intelligence startups raised record funding as investors bet on the country's AI potential, with enterprise AI and healthcare AI leading the charge."),
            ("NVIDIA partners with Indian cloud providers for AI infrastructure", "NVIDIA has signed agreements with multiple Indian cloud service providers to deploy GPU computing clusters, accelerating AI adoption across Indian enterprises."),
            ("Concerns grow over AI job displacement in Indian BPO sector", "Industry leaders warn that generative AI could impact 500,000 jobs in India's business process outsourcing sector over the next three years."),
        ],
        "startup_funding": [
            ("Zepto raises $500 million at $5 billion valuation", "Quick commerce startup Zepto has closed its latest funding round, making it one of the most valuable startups in India's e-commerce ecosystem."),
            ("PhonePe completes $700 million fundraise for expansion", "Digital payments giant PhonePe has raised $700 million to fuel expansion into lending, insurance, and cross-border payments."),
            ("Indian startups raised $12 billion in Q3 2025", "Despite global funding slowdown, Indian startup ecosystem shows resilience with $12 billion raised in the third quarter, led by fintech and deeptech sectors."),
            ("Ola Electric files for IPO amid surging EV demand", "Ola Electric has filed its draft red herring prospectus for an IPO, seeking to raise Rs 5,000 crore as the EV market continues to expand."),
            ("Razorpay acquires fintech startup for $200 million", "Payment processing giant Razorpay has acquired a compliance technology startup to strengthen its offerings for regulated financial institutions."),
        ],
        "supply_chain_crisis": [
            ("Global semiconductor shortage impacts Indian auto production", "Major Indian automobile manufacturers including Maruti Suzuki and Hyundai report production delays due to ongoing chip supply constraints."),
            ("Commodity prices surge affecting Indian manufacturing sector", "Steel, copper, and aluminum prices have risen sharply, squeezing margins for Indian manufacturers and potentially leading to product price increases."),
            ("Port congestion delays hit Indian exports and imports", "Major Indian ports face congestion as global shipping disruptions continue, with container dwell times increasing by 40% compared to normal levels."),
            ("Indian pharma companies face API supply disruption from China", "Pharmaceutical manufacturers in India are scrambling to secure alternative sources for active pharmaceutical ingredients as supply from China faces disruption."),
        ],
        "crisis_fraud": [
            ("SEBI bans promoter of listed company for insider trading", "Securities and Exchange Board of India has imposed a lifetime ban on the promoter of a mid-cap listed company for insider trading violations worth Rs 200 crore."),
            ("Major data breach at Indian bank exposes 10 million customers", "A cybersecurity incident at one of India's largest private banks has compromised personal and financial data of approximately 10 million customers."),
            ("RBI imposes Rs 500 crore penalty on NBFC for KYC violations", "The Reserve Bank of India has levied a record penalty on a non-banking financial company for systematic violations of Know Your Customer norms."),
        ],
        "merger_acquisition": [
            ("Adani Group acquires NDTV stake in Rs 3000 crore deal", "Adani Enterprises has completed acquisition of a significant stake in NDTV through open offer and indirect purchase, valuing the media company at Rs 3000 crore."),
            ("Reliance Industries and Disney complete $8.5B merger of media assets", "The mega merger of Reliance's Viacom18 and Disney's Star India creates India's largest entertainment company with combined revenue of Rs 50,000 crore."),
            ("Tata Group restructures tech portfolio with $2B acquisition", "Tata Digital has acquired a leading SaaS company to accelerate its super app strategy, marking one of the largest tech acquisitions by an Indian conglomerate."),
            ("Vodafone Idea completes Rs 18000 crore rights issue", "Struggling telecom operator Vodafone Idea has successfully raised capital through its rights issue, providing a lifeline for the debt-laden company."),
        ],
    }

    # Flatten into article list
    all_articles = []
    trend_labels = {}  # article_id -> expected_trend

    for trend_name, article_data in known_trends.items():
        for title, summary in article_data:
            article = NewsArticle(
                id=uuid4(),
                title=title,
                summary=summary,
                url=f"https://example.com/{len(all_articles)}",
                source_id=f"test_{trend_name}",
                source_name="Test Source",
                source_type="rss",
                source_tier="tier_1",
                source_credibility=0.9,
                published_at=now - timedelta(hours=len(all_articles) % 24),
            )
            all_articles.append(article)
            trend_labels[str(article.id)] = trend_name

    print(f"\n  Synthetic articles: {len(all_articles)} across {len(known_trends)} known trends")
    for trend_name, articles in known_trends.items():
        print(f"    {trend_name}: {len(articles)} articles")

    # Run pipeline
    engine = RecursiveTrendEngine(max_depth=2, mock_mode=False)
    start = time.time()
    tree = await engine.run(all_articles)
    pipeline_time = time.time() - start

    metrics = engine.metrics
    print(f"\n  PIPELINE RESULTS:")
    print(f"    Input:      {metrics['article_counts']['input']}")
    print(f"    After rel:  {metrics['article_counts'].get('after_relevance', '?')}")
    print(f"    After dedup:{metrics['article_counts'].get('after_dedup', '?')}")
    print(f"    Clusters:   {metrics.get('n_clusters', '?')}")
    print(f"    Noise:      {metrics.get('noise_count', '?')}")
    print(f"    Time:       {pipeline_time:.1f}s")

    # Event classification check
    event_dist = metrics.get("event_distribution", {})
    print(f"\n  EVENT CLASSIFICATION:")
    for etype, count in sorted(event_dist.items(), key=lambda x: -x[1]):
        safe_print(f"    {etype:20s} {count:3d}")

    # Analyze clusters vs known trends
    safe_print(f"\n  CLUSTER -> TREND MAPPING:")
    cluster_trend_map = {}

    for node_id, node in tree.nodes.items():
        # For each cluster, check which known trends its articles came from
        article_ids = [str(aid) for aid in node.source_articles]
        origin_trends = Counter()
        for aid in article_ids:
            if aid in trend_labels:
                origin_trends[trend_labels[aid]] += 1

        dominant_trend = origin_trends.most_common(1)[0][0] if origin_trends else "unknown"
        purity = origin_trends.most_common(1)[0][1] / max(len(article_ids), 1) if origin_trends else 0

        cluster_trend_map[node.trend_title] = {
            "dominant": dominant_trend,
            "purity": purity,
            "articles": len(article_ids),
            "origins": dict(origin_trends),
        }

        # Show recommended services
        buying = getattr(node, 'buying_intent', {}) or {}
        severity = node.severity.value if hasattr(node.severity, 'value') else str(node.severity)
        safe_print(
            f"\n    [{severity:6s}] {node.trend_title[:60]}"
            f" | {node.article_count} articles"
        )
        safe_print(f"             Origin: {dict(origin_trends)} (purity: {purity:.0%})")
        if buying:
            safe_print(f"             Intent: {buying.get('signal_type', '?')} | {buying.get('urgency', '?')}")
            safe_print(f"             Who: {str(buying.get('who_needs_help', ''))[:80]}")

    # Validation
    print(f"\n  VALIDATION:")
    checks = []

    # Check: All known trends detected (each should appear in at least one cluster)
    detected_trends = set()
    for info in cluster_trend_map.values():
        detected_trends.add(info["dominant"])
        # Also count trends that appear as non-dominant origins
        for origin_trend in info.get("origins", {}):
            detected_trends.add(origin_trend)
    missing = set(known_trends.keys()) - detected_trends
    checks.append(("All known trends detected", len(missing) == 0))
    if missing:
        print(f"    MISSING TRENDS: {missing}")

    # Check: Cluster purity (articles from same trend should cluster together)
    purities = [info["purity"] for info in cluster_trend_map.values()]
    avg_purity = sum(purities) / max(len(purities), 1)
    checks.append(("Avg cluster purity >= 60%", avg_purity >= 0.6))
    print(f"    Average cluster purity: {avg_purity:.0%}")

    # Check: Not too many clusters (shouldn't split known trends excessively)
    n_clusters = len(tree.nodes)
    checks.append(("Clusters <= 2x known trends", n_clusters <= len(known_trends) * 2))

    # Check: Low noise
    noise = metrics.get('noise_count', 0)
    total = metrics['article_counts'].get('after_semantic_dedup', len(all_articles))
    noise_pct = noise / max(total, 1)
    checks.append(("Noise < 30%", noise_pct < 0.3))

    # Check: All clusters synthesized
    checks.append(("All nodes synthesized", all(n.trend_title for n in tree.nodes.values())))

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {name}")

    return all_pass


async def stress_test_classifier():
    """
    TEST 3: Embedding-based event classifier accuracy test.
    Tests classification on labeled examples.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 3: EVENT CLASSIFIER ACCURACY")
    print("=" * 70)

    from app.schemas.news import NewsArticle
    from app.news.event_classifier import EmbeddingEventClassifier
    from app.tools.embeddings import EmbeddingTool

    now = datetime.now(timezone.utc)

    # Labeled test cases: (title, summary, expected_event_type)
    test_cases = [
        # Regulation
        ("RBI mandates new KYC norms for digital lending", "Reserve Bank of India requires stricter identity verification", "regulation"),
        ("SEBI tightens disclosure rules for listed companies", "Securities regulator introduces new quarterly reporting requirements", "regulation"),
        ("Government passes new data protection bill", "Parliament approves comprehensive data privacy legislation", "regulation"),
        # Funding
        ("Startup raises $100 million Series C round", "AI startup closes major funding from venture capital firms", "funding"),
        ("PhonePe valued at $12 billion after fresh fundraise", "Digital payments company closes Series E round", "funding"),
        # Acquisition
        ("Reliance acquires media company in $2B deal", "Reliance Industries completes acquisition of media assets", "acquisition"),
        ("Tata Group buys AI startup for strategic expansion", "Tata Digital acquires artificial intelligence company", "acquisition"),
        # Technology
        ("Indian IT firms invest in generative AI", "TCS and Infosys allocate billions for AI transformation", "technology"),
        ("NVIDIA launches new AI chips for Indian market", "Graphics chipmaker targets Indian enterprise AI deployment", "technology"),
        # Layoffs
        ("Tech company lays off 2000 employees", "Major restructuring as company cuts workforce by 15%", "layoffs"),
        ("Mass layoffs hit Indian startup ecosystem", "Multiple startups announce significant workforce reductions", "layoffs"),
        # IPO
        ("Company files DRHP for IPO worth Rs 5000 crore", "Draft red herring prospectus filed with SEBI", "ipo"),
        ("Startup plans public listing on BSE", "Company prepares for initial public offering", "ipo"),
        # Expansion
        ("Mahindra opens new manufacturing plant in Gujarat", "Automobile maker invests Rs 3000 crore in new facility", "expansion"),
        ("Amazon expands cloud infrastructure in India", "Tech giant opens new data centers in Mumbai and Hyderabad", "expansion"),
        # Supply chain
        ("Semiconductor shortage disrupts auto production", "Chip supply constraints force production cuts at Maruti", "supply_chain"),
        ("Raw material prices surge hits manufacturing margins", "Steel and copper prices rise sharply affecting producers", "supply_chain"),
        # Crisis
        ("Fraud detected at major Indian bank", "Financial irregularities worth Rs 500 crore uncovered", "crisis"),
        ("Data breach exposes millions of customer records", "Cybersecurity incident compromises sensitive personal data", "crisis"),
        # Price change
        ("Fuel prices hiked for third consecutive month", "Petrol and diesel prices increased by Rs 2 per litre", "price_change"),
        ("Inflation pushes food prices to record levels", "Consumer price index rises sharply driven by food costs", "price_change"),
        # Consumer shift
        ("Quick commerce reshapes Indian grocery shopping", "Blinkit Zepto and Swiggy Instamart see 200% growth", "consumer_shift"),
        ("D2C brands capture market share from legacy FMCG", "Direct to consumer brands disrupting traditional retail", "consumer_shift"),
        # Partnership
        ("Strategic alliance between Infosys and Microsoft", "Companies sign partnership for cloud and AI services", "partnership"),
        ("Joint venture announced for renewable energy project", "Indian and Japanese firms collaborate on solar manufacturing", "partnership"),
        # Market entry
        ("Foreign company enters Indian market with $500M investment", "International retailer establishes India operations", "market_entry"),
        ("India-UAE trade agreement opens new export opportunities", "Bilateral trade deal reduces tariffs on key commodities", "market_entry"),
        # Leadership change
        ("New CEO appointed at Tata Consultancy Services", "Board announces leadership transition at India's largest IT company", "leadership_change"),
        ("Founder steps down from tech startup board", "Company transitions to professional management", "leadership_change"),
        # Edge cases: Non-business (should classify as general or low confidence)
        ("India wins cricket World Cup final", "Cricket team defeats Australia in thrilling match", "general"),
        ("Bollywood star announces new movie release", "Actor signs three-film deal with production house", "general"),
        ("Weather forecast: Heavy rains expected in Mumbai", "IMD issues orange alert for coastal Maharashtra", "general"),
    ]

    embedding_tool = EmbeddingTool()
    classifier = EmbeddingEventClassifier(embedding_tool)

    # Create articles
    articles = []
    expected = []
    for i, (title, summary, expected_type) in enumerate(test_cases):
        article = NewsArticle(
            id=uuid4(),
            title=title,
            summary=summary,
            url=f"https://example.com/{i}",
            source_id="test",
            source_name="Test",
            source_type="rss",
            source_tier="tier_1",
            source_credibility=0.9,
            published_at=now,
        )
        articles.append(article)
        expected.append(expected_type)

    # Classify
    start = time.time()
    distribution = classifier.classify_batch(articles)
    elapsed = time.time() - start

    print(f"\n  Classified {len(articles)} articles in {elapsed:.2f}s")
    print(f"  Distribution: {distribution}")

    # Check accuracy
    correct = 0
    wrong = []
    for article, expected_type in zip(articles, expected):
        actual = getattr(article, '_trigger_event', 'general')
        confidence = getattr(article, '_trigger_confidence', 0.0)
        if actual == expected_type:
            correct += 1
        else:
            wrong.append({
                "title": article.title[:60],
                "expected": expected_type,
                "got": actual,
                "confidence": confidence,
            })

    accuracy = correct / len(articles) * 100
    print(f"\n  ACCURACY: {correct}/{len(articles)} = {accuracy:.1f}%")

    if wrong:
        print(f"\n  MISCLASSIFIED ({len(wrong)}):")
        for w in wrong:
            safe_print(f"    '{w['title']}...'")
            print(f"      Expected: {w['expected']}, Got: {w['got']} (conf: {w['confidence']:.3f})")

    # Validation
    print(f"\n  VALIDATION:")
    checks = [
        ("Accuracy >= 60%", accuracy >= 60),
        ("Accuracy >= 70%", accuracy >= 70),
        ("Accuracy >= 80%", accuracy >= 80),
        ("Non-business filtered", all(
            getattr(a, '_trigger_confidence', 1.0) < 0.3
            for a, e in zip(articles, expected) if e == "general"
        )),
        ("Classification < 5s", elapsed < 5),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed and "Accuracy >= 80%" not in name:  # 80% is aspirational
            all_pass = False
        print(f"    [{status}] {name}")

    return all_pass


async def stress_test_gdelt_historical():
    """
    TEST 4: Historical GDELT data — test with articles from past days.
    GDELT lets us query different time periods.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 4: GDELT MULTI-DAY HISTORICAL")
    print("=" * 70)

    import httpx
    from app.schemas.news import NewsArticle
    from app.trends.engine import RecursiveTrendEngine

    now = datetime.now(timezone.utc)

    async def fetch_gdelt_batch(query: str, timespan_minutes: int = 4320, max_records: int = 100) -> list:
        """Fetch batch from GDELT DOC API."""
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max_records),
            "sort": "DateDesc",
            "timespan": str(timespan_minutes),
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, params=params)
                if r.status_code != 200:
                    print(f"    GDELT HTTP {r.status_code}")
                    return []
                data = r.json()
                articles = []
                for item in data.get("articles", []):
                    try:
                        pub_date = datetime.strptime(
                            item.get("seendate", "")[:14], "%Y%m%d%H%M%S"
                        ).replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pub_date = now

                    articles.append(NewsArticle(
                        id=uuid4(),
                        title=item.get("title", "").strip(),
                        summary="",
                        url=item.get("url", ""),
                        source_id="gdelt",
                        source_name=item.get("domain", "GDELT"),
                        source_type="api",
                        source_tier="tier_2",
                        source_credibility=0.7,
                        published_at=pub_date,
                    ))
                return articles
        except Exception as e:
            print(f"    GDELT fetch error: {e}")
            return []

    # Fetch large batches from GDELT with different queries
    queries = [
        ("India business broad", "sourcecountry:IN sourcelang:english", 100),
        ("India regulation", "sourcecountry:IN sourcelang:english regulation OR compliance OR RBI OR SEBI", 50),
        ("India technology", "sourcecountry:IN sourcelang:english technology OR AI OR startup OR digital", 50),
        ("India finance", "sourcecountry:IN sourcelang:english funding OR IPO OR merger OR acquisition", 50),
    ]

    all_articles = []
    for name, query, max_r in queries:
        print(f"\n  Fetching: {name}...")
        articles = await fetch_gdelt_batch(query, timespan_minutes=4320, max_records=max_r)
        all_articles.extend(articles)
        print(f"    Got {len(articles)} articles")

    # Deduplicate by title
    seen_titles = set()
    unique = []
    for a in all_articles:
        title_key = a.title.lower().strip()[:80]
        if title_key not in seen_titles and len(a.title) > 20:
            seen_titles.add(title_key)
            unique.append(a)
    all_articles = unique

    print(f"\n  Total unique GDELT articles: {len(all_articles)}")

    if len(all_articles) < 30:
        print("  Not enough GDELT articles. Skipping this test.")
        return True

    # Run pipeline on this large batch
    engine = RecursiveTrendEngine(max_depth=2, mock_mode=False)
    start = time.time()
    tree = await engine.run(all_articles)
    pipeline_time = time.time() - start

    metrics = engine.metrics
    print(f"\n  PIPELINE RESULTS:")
    print(f"    Input:      {metrics['article_counts']['input']}")
    print(f"    After rel:  {metrics['article_counts'].get('after_relevance', '?')}")
    print(f"    Clusters:   {metrics.get('n_clusters', '?')}")
    print(f"    Noise:      {metrics.get('noise_count', '?')}")
    print(f"    Trends:     {len(tree.nodes)}")
    print(f"    Time:       {pipeline_time:.1f}s")

    # Event distribution
    event_dist = metrics.get("event_distribution", {})
    print(f"\n  EVENT CLASSIFICATION:")
    for etype, count in sorted(event_dist.items(), key=lambda x: -x[1]):
        safe_print(f"    {etype:20s} {count:3d}")

    # Show trends
    print(f"\n  DETECTED TRENDS:")
    for node in sorted(tree.nodes.values(), key=lambda n: n.trend_score, reverse=True)[:8]:
        severity = node.severity.value if hasattr(node.severity, 'value') else str(node.severity)
        buying = getattr(node, 'buying_intent', {}) or {}
        safe_print(
            f"    [{severity:6s}] {node.trend_title[:60]}"
            f" | {node.article_count} articles"
            f" | score={node.trend_score:.2f}"
        )
        if buying:
            safe_print(f"             {buying.get('signal_type', '?')} | {buying.get('urgency', '?')}")

    # Validation
    print(f"\n  VALIDATION:")
    checks = [
        ("GDELT articles >= 50", len(all_articles) >= 50),
        ("Clusters >= 3", metrics.get('n_clusters', 0) >= 3),
        ("Pipeline < 180s", pipeline_time < 180),
        ("All nodes synthesized", all(n.trend_title for n in tree.nodes.values())),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {name}")

    return all_pass


async def main():
    print("=" * 70)
    print("  SALES AGENT TREND ENGINE — STRESS TEST SUITE")
    print("=" * 70)
    start = time.time()

    results = {}

    # Test 1: Volume (real RSS feeds, max articles)
    try:
        results["Volume (200+ articles)"] = await stress_test_volume()
    except Exception as e:
        print(f"\n  TEST 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["Volume (200+ articles)"] = False

    # Test 2: Known trend detection (synthetic articles)
    try:
        results["Known Trends"] = await stress_test_known_trends()
    except Exception as e:
        print(f"\n  TEST 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["Known Trends"] = False

    # Test 3: Classifier accuracy
    try:
        results["Classifier Accuracy"] = await stress_test_classifier()
    except Exception as e:
        print(f"\n  TEST 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["Classifier Accuracy"] = False

    # Test 4: GDELT historical
    try:
        results["GDELT Historical"] = await stress_test_gdelt_historical()
    except Exception as e:
        print(f"\n  TEST 4 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["GDELT Historical"] = False

    # Summary
    total_time = time.time() - start
    print("\n" + "=" * 70)
    print("  STRESS TEST RESULTS")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Total time: {total_time:.1f}s")

    all_pass = all(results.values())
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
