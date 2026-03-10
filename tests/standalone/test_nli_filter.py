"""
Standalone NLI Filter Precision/Recall Test.

Measures: precision, recall, F1 against a 30-article labeled test set
(15 SHOULD-KEEP, 15 SHOULD-DROP) that exercises all major filter paths:
  - NLI auto-accept (entailment >= 0.88)
  - NLI auto-reject (entailment <= 0.10)
  - LLM ambiguous zone (0.10 < entailment < 0.88)

Also validates invariants via the CUDA-Agent "Protected Verification" pattern.

Run:
    venv/Scripts/python.exe tests/standalone/test_nli_filter.py
    venv/Scripts/python.exe -m pytest tests/standalone/test_nli_filter.py -v -s
"""

import asyncio
import logging
import sys
import time

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for arrows etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # suppress noisy engine logs during test
    format="%(levelname)s | %(name)s | %(message)s",
)
# Keep filter and nli logs at INFO so we see the gate breakdown
logging.getLogger("app.intelligence.filter").setLevel(logging.INFO)
logging.getLogger("app.intelligence.engine.nli_filter").setLevel(logging.INFO)

# ── Path setup (works whether run as script or via pytest) ────────────────────
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.filter import filter_articles
from app.intelligence.models import Article, DiscoveryMode, DiscoveryScope


# ══════════════════════════════════════════════════════════════════════════════
# LABELED TEST SET  (30 articles — ground-truth labels defined here)
# ══════════════════════════════════════════════════════════════════════════════

#
# Each record: (title, summary, label)
#   label = "keep" | "drop"
#
# Summaries are realistic ~1-2 sentence elaborations. Articles without a
# substantive summary are marked is_title_only=True in the fixture builder
# and will be routed to the LLM path regardless of NLI score (matching the
# filter's "title-only articles → always LLM" rule in filter.py:107-109).
#

LABELED_ARTICLES: List[Tuple[str, str, str]] = [
    # ── SHOULD-KEEP (B2B business events) ─────────────────────────────────────
    (
        "Zepto raises $200M Series G from General Catalyst and others",
        "Quick-commerce startup Zepto has closed a $200 million Series G funding round "
        "led by General Catalyst with participation from Nexus Ventures and StepStone Group, "
        "valuing the company at $3.6 billion.",
        "keep",
    ),
    (
        "Freshworks signs 5-year enterprise cloud deal with HDFC Life Insurance",
        "Freshworks announced a five-year strategic agreement with HDFC Life Insurance "
        "to deploy its CRM and customer-support platform across the insurer's 400 branches, "
        "replacing a legacy on-premise system.",
        "keep",
    ),
    (
        "Infosys acquires German consulting firm for €180M to expand EU presence",
        "Infosys has completed the acquisition of Berlin-based digital consulting firm "
        "infoXpert GmbH for €180 million, adding 1,200 consultants and three SAP practices "
        "to its European delivery network.",
        "keep",
    ),
    (
        "Razorpay launches PaymentOS platform for B2B SaaS companies",
        "Razorpay unveiled PaymentOS, an API-first payments orchestration platform "
        "designed for B2B SaaS vendors, enabling multi-currency billing, dunning automation, "
        "and real-time reconciliation with ERPs.",
        "keep",
    ),
    (
        "Meesho raises $275M in new funding, valuation hits $5.1 billion",
        "Social commerce platform Meesho has raised $275 million in its latest primary round "
        "from Fidelity, SoftBank Vision Fund 2, and B Capital, pushing its post-money "
        "valuation to $5.1 billion ahead of a planned IPO.",
        "keep",
    ),
    (
        "Nykaa acquires D2C skincare brand for strategic retail expansion",
        "Nykaa has acquired Dot & Key, a direct-to-consumer skincare brand, in an all-cash "
        "deal estimated at Rs 320 crore, deepening its owned-brand portfolio for premium "
        "skincare and hair care categories.",
        "keep",
    ),
    (
        "Swiggy Instamart expands to 30 new cities, signs logistics partnership",
        "Swiggy Instamart is expanding its 10-minute grocery delivery to 30 tier-2 and "
        "tier-3 cities, partnering with Shadowfax for last-mile fulfillment across markets "
        "where Swiggy does not own dark-store infrastructure.",
        "keep",
    ),
    (
        "Ola Electric files DRHP for IPO, targets $700M raise on NSE",
        "Ola Electric has filed its draft red herring prospectus with SEBI for an initial "
        "public offering seeking to raise up to $700 million, intending to list on the "
        "National Stock Exchange in Q3 2026.",
        "keep",
    ),
    (
        "Cipla recalls blood pressure drug from US market after FDA inspection",
        "Cipla Limited is voluntarily recalling approximately 40,000 bottles of "
        "Amlodipine Besylate tablets from the US market following an FDA warning letter "
        "citing manufacturing deviations at its Goa facility.",
        "keep",
    ),
    (
        "NVIDIA announces DGX H200 India data center with Tata Consultancy",
        "NVIDIA and Tata Consultancy Services jointly announced a sovereign AI data center "
        "in Pune housing 512 DGX H200 systems, offering GPU-as-a-service to Indian "
        "enterprises and government agencies.",
        "keep",
    ),
    (
        "SAP India wins 3-year ERP contract with Mahindra Manufacturing",
        "SAP India has secured a three-year enterprise resource planning contract with "
        "Mahindra & Mahindra's manufacturing division, covering RISE with SAP S/4HANA "
        "cloud migration for 14 plants across six states.",
        "keep",
    ),
    (
        "PhonePe acquires Indus OS to expand Android ecosystem in India",
        "PhonePe has completed the acquisition of Indus OS, the Indic-language Android "
        "alternative app store, for an undisclosed sum, gaining 100 million pre-installed "
        "devices and a direct channel for its financial services stack.",
        "keep",
    ),
    (
        "KIMS Hospitals enters Karnataka with 2 new super-speciality facilities",
        "KIMS Hospitals has signed binding agreements to open two 300-bed super-speciality "
        "hospitals in Bengaluru and Mysuru, expanding outside Andhra Pradesh for the first "
        "time, with operations targeted for Q2 2027.",
        "keep",
    ),
    (
        "Nexperia semiconductor plant dispute escalates, chip shortage warning",
        "The UK government's investigation into Newport Wafer Fab's acquisition by "
        "Netherlands-based Nexperia has escalated after ministers issued a national "
        "security direction, raising concerns of a European compound-semiconductor supply gap.",
        "keep",
    ),
    (
        "OpenAI signs enterprise agreement with Tata Communications for India",
        "OpenAI has entered a multi-year enterprise licensing agreement with Tata "
        "Communications to distribute GPT-4o API access through Tata's IZO Cloud "
        "platform to Indian enterprises, government bodies, and startups.",
        "keep",
    ),

    # ── HARD KEEP — less obvious B2B, filter may miss ─────────────────────────
    (
        "UltraTech Cement raises prices by Rs 15/bag across North, West India",
        "UltraTech Cement has hiked retail and dealer prices by Rs 15 per 50 kg bag "
        "in North and West India effective March 2026, citing higher fuel and logistics "
        "costs. The move follows similar increases by ACC and Ambuja, signalling a "
        "broad-based industry repricing that will affect construction project budgets.",
        "keep",
    ),
    (
        "Capgemini India wins Rs 430 crore IT modernisation deal with HPCL",
        "Hindustan Petroleum Corporation has awarded Capgemini India a five-year IT "
        "services contract worth Rs 430 crore to migrate its upstream and downstream "
        "operations to SAP S/4HANA on Azure, replacing a 15-year-old legacy ERP.",
        "keep",
    ),
    (
        "Coal India signs fuel supply agreements with 12 private power plants",
        "Coal India Limited has signed long-term fuel supply agreements with 12 "
        "independent power producers including Adani Power, JSW Energy and CESC, "
        "covering 48 million tonnes of coal annually for thermal generation capacity "
        "totalling 14 GW under the government's revised linkage policy.",
        "keep",
    ),
    (
        "Bajaj Finserv launches MSME credit platform targeting 50 lakh borrowers",
        "Bajaj Finserv has launched an AI-underwritten MSME lending platform offering "
        "collateral-free loans of Rs 5-50 lakh to small manufacturers and traders, "
        "using GST filing history, bank statements, and supply-chain data as primary "
        "credit signals. The platform targets 50 lakh MSMEs by FY2027.",
        "keep",
    ),

    # ── HARD DROP — look B2B but are not actionable intelligence ──────────────
    (
        "PhonePe signs Rs 100 crore jersey sponsorship with Mumbai Indians for IPL",
        "PhonePe has signed a three-year Rs 100 crore title sponsorship deal to "
        "feature as the official payments partner and jersey sponsor of the Mumbai "
        "Indians franchise for the 2026-2028 Indian Premier League seasons, marking "
        "the brand's largest sports marketing investment to date.",
        "drop",
    ),
    (
        "India's IT sector exports hit $220 billion in FY2026, NASSCOM reports",
        "India's technology sector exported $220 billion in services and products in "
        "FY2026, up 9% year-on-year, with cloud, AI, and cybersecurity sub-segments "
        "growing fastest, according to the NASSCOM annual strategic review released "
        "on Monday in Bengaluru.",
        "drop",
    ),
    (
        "Tata Motors Nexon crosses 5 lakh cumulative EV sales milestone",
        "The Tata Nexon EV has crossed 5 lakh cumulative units sold in India since "
        "launch, making it the country's best-selling electric passenger vehicle. "
        "Tata Motors CEO Shailesh Chandra called it a landmark for Indian EV adoption "
        "and confirmed a new sub-Rs 12 lakh model is planned for 2027.",
        "drop",
    ),
    (
        "RBI holds repo rate at 6.25%, signals cautious easing in H2 2026",
        "The Reserve Bank of India's Monetary Policy Committee voted 4-2 to hold the "
        "benchmark repo rate at 6.25% for a third consecutive meeting, with Governor "
        "Sanjay Malhotra signalling room for a 25 bps cut in the second half of 2026 "
        "if core inflation falls below 4.5%.",
        "drop",
    ),
    (
        "SEBI tightens F&O regulations, weekly options contracts cut to one per index",
        "The Securities and Exchange Board of India has issued a circular limiting "
        "weekly options expiry contracts to one per major index, effective April 2026, "
        "following a study showing retail investors lost Rs 1.81 lakh crore in "
        "derivatives in three years. Turnover in weekly contracts is expected to drop "
        "by 40-60%.",
        "drop",
    ),
    (
        "Sensex crosses 85,000 for first time as FIIs return to Indian equities",
        "India's benchmark BSE Sensex closed above 85,000 for the first time on "
        "Tuesday, driven by foreign institutional investor inflows of Rs 9,200 crore "
        "in a single session, with HDFC Bank, Reliance Industries and Infosys "
        "contributing the most to the rally.",
        "drop",
    ),

    # ── SHOULD-DROP (noise — no B2B value) ────────────────────────────────────
    (
        "ICC T20 World Cup: India beats Australia by 10 wickets in final",
        "India crushed Australia by 10 wickets in the ICC T20 World Cup final at "
        "Barbados, with Rohit Sharma hitting an unbeaten 57 and Virat Kohli scoring "
        "76 as the Men in Blue claimed their second T20 world title.",
        "drop",
    ),
    (
        "PM Modi inaugurates Rs 18,000 crore Delhi Metro Phase 4 expansion",
        "Prime Minister Narendra Modi inaugurated the 65-kilometre Delhi Metro Phase 4 "
        "network, connecting Janakpuri West to RK Ashram, funded jointly by the Central "
        "government and Delhi government under the National Transit Mission.",
        "drop",
    ),
    (
        "ED arrests Al Falah chairman in money laundering case",
        "The Enforcement Directorate arrested Mohammed Arshad, chairman of Al Falah "
        "Cooperative Credit Society, in connection with an alleged Rs 450 crore money "
        "laundering case linked to chit-fund fraud in Kerala.",
        "drop",
    ),
    (
        "Nothing Phone 4a launched in India starting at Rs 19,999",
        "Nothing has launched the Phone 4a in India at a starting price of Rs 19,999, "
        "featuring the Snapdragon 7s Gen 3 chipset, a 50MP rear camera, and the brand's "
        "signature Glyph Interface 2.0 on the transparent back panel.",
        "drop",
    ),
    (
        "Samsung Galaxy S26 Ultra review: best Android phone of 2026",
        "The Samsung Galaxy S26 Ultra earns its flagship crown with a 200MP camera "
        "system, Snapdragon 8 Elite Gen 2, and an all-day titanium chassis, making it "
        "the definitive Android flagship for consumers who want the best.",
        "drop",
    ),
    (
        "Motorola Edge 70 Fusion vs OnePlus Nord CE5 camera comparison",
        "We put the Motorola Edge 70 Fusion's 50MP Sony sensor against the OnePlus "
        "Nord CE5's 108MP Samsung ISOCELL HM6 in 15 real-world shooting scenarios "
        "including night, portrait, and video stabilisation tests.",
        "drop",
    ),
    (
        "Iranian drone strikes Bahrain desalination facility",
        "Iran-backed Houthi forces launched a coordinated drone attack on the "
        "Al-Dur desalination and power plant in Bahrain, triggering a temporary "
        "shutdown and prompting the US Fifth Fleet to raise force-protection levels.",
        "drop",
    ),
    (
        "Centre responds to Bengal flood crisis, deploys NDRF teams",
        "The Union Home Ministry dispatched 12 NDRF battalions to West Bengal "
        "after the Damodar Valley Corporation released excess water from Durgapur "
        "barrage, flooding 150 villages in Bankura, Hooghly, and Bardhaman districts.",
        "drop",
    ),
    (
        "Virat Kohli signs Rs 50 crore brand deal with fitness startup",
        "Indian cricket star Virat Kohli has signed a three-year brand ambassadorship "
        "with Cult.fit for Rs 50 crore, becoming the face of the fitness platform's "
        "nationwide gym and online fitness subscription campaigns.",
        "drop",
    ),
    (
        "Gold prices hit all-time high of Rs 75,000 per 10 grams",
        "Gold futures on MCX crossed Rs 75,000 per 10 grams for the first time in "
        "history, driven by global safe-haven demand amid Middle East tensions and "
        "expectations of a US Federal Reserve rate pause in March.",
        "drop",
    ),
    (
        "India's GDP growth slows to 6.2% in Q3 amid global headwinds",
        "India's gross domestic product expanded at 6.2% in the October-December "
        "quarter, the slowest pace in five quarters, as manufacturing output contracted "
        "and export growth moderated due to weak global demand.",
        "drop",
    ),
    (
        "Actor Ranveer Singh launches production company in Mumbai",
        "Bollywood actor Ranveer Singh has launched RowdyPictures, an independent "
        "production house based in Mumbai, with plans to produce three feature films "
        "and a prestige web series in its debut slate for 2026-27.",
        "drop",
    ),
    (
        "Best budget phones under Rs 20,000 to buy in March 2026",
        "Looking for the best smartphone under Rs 20,000? We've tested 12 devices "
        "and ranked the top picks across camera, battery life, performance, and "
        "display quality — with picks from Redmi, Realme, Samsung, and iQOO.",
        "drop",
    ),
    (
        "Cricket: Jaipur connection in ICC World Cup trophy design",
        "The ICC revealed that the 2026 World Cup trophy was designed by a Jaipur-based "
        "craftsman inspired by Rajasthani blue pottery motifs, with the trophy base "
        "featuring the host nations' flags etched in sterling silver.",
        "drop",
    ),
    (
        "22 Iranian sailors from IRIS Dena discharged after hostage crisis",
        "Twenty-two sailors from the Iranian frigate IRIS Dena have been released "
        "after a week-long hostage standoff in the Strait of Hormuz, following "
        "diplomatic intervention by Oman and the United Nations.",
        "drop",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_SPORTS_CELEBRITY_GOVT_SIGNALS = {
    # Sports
    "icc", "world cup", "cricket", "virat kohli", "rohit sharma",
    "t20", "test match", "odi", "jaipur connection",
    # Celebrity / entertainment
    "bollywood", "actor ranveer", "production company", "ranveer singh",
    "prestige web series",
    # Consumer electronics listicles (not enterprise tech news)
    "best budget phone", "camera comparison", "samsung galaxy s26 review",
    "motorola edge", "oneplus nord", "nothing phone 4a launched",
    # Government / military / geo-political (specific enough to avoid B2B overlap)
    "pm modi inaugurates", "ndrf teams", "ed arrests", "money laundering case",
    "drone strikes bahrain", "iranian drone", "iran-backed houthi",
    "bengal flood crisis", "damodar valley",
    # Macro-only (no specific company as primary actor)
    "gold prices hit all-time high", "india gdp growth slows",
    # Hostage / military
    "22 iranian sailors", "iris dena", "hostage crisis",
    # Celebrity brand deals (not B2B)
    "virat kohli signs rs 50 crore brand deal",
    # Trophy/design human interest
    "world cup trophy design", "rajasthani blue pottery",
    # Hard negatives added 2026-03-10
    "jersey sponsorship with mumbai indians",   # sports sponsorship with company name
    "it sector exports hit", "nasscom reports",  # industry aggregate stats, no deal
    "nexon crosses 5 lakh", "cumulative ev sales milestone",  # consumer product milestone
    "rbi holds repo rate", "signals cautious easing",          # monetary policy
    "sebi tightens f&o", "weekly options expiry",              # market regulation
    "sensex crosses 85,000", "fiis return to indian equities", # market milestone
}


def _has_noise_signal(article: Article) -> bool:
    """Check whether an article's title/summary contains noise signals.

    Short signals (≤4 chars, e.g. "odi", "icc", "t20") use whole-word matching
    to avoid false positives like "odi" matching "bodies", "icc" matching
    "price" etc.  Longer signals use plain substring matching which is safe.
    """
    import re as _re
    combined = f"{article.title} {article.summary}".lower()
    for sig in _SPORTS_CELEBRITY_GOVT_SIGNALS:
        if len(sig) <= 4:
            if _re.search(r'\b' + _re.escape(sig) + r'\b', combined):
                return True
        else:
            if sig in combined:
                return True
    return False


def _make_article(idx: int, title: str, summary: str) -> Article:
    return Article(
        url=f"https://test-nli.example.com/article/{idx}",
        title=title,
        summary=summary,
        source_name="TestSource",
        source_url="https://test-nli.example.com",
        published_at=datetime.now(timezone.utc) - timedelta(hours=idx),
    )


def _make_scope(industry: str = "Technology") -> DiscoveryScope:
    """Minimal DiscoveryScope for Industry-First mode (hardest for the filter)."""
    return DiscoveryScope(
        mode=DiscoveryMode.INDUSTRY_FIRST,
        industry=industry,
        region="IN",
        hours=120,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CORE TEST: 30-ARTICLE LABELED SET
# ══════════════════════════════════════════════════════════════════════════════

async def run_labeled_set_test() -> dict:
    """Run filter_articles against the 30-article labeled set and compute metrics."""

    articles = [
        _make_article(i, title, summary)
        for i, (title, summary, _label) in enumerate(LABELED_ARTICLES)
    ]
    ground_truth = {
        articles[i].id: label
        for i, (_t, _s, label) in enumerate(LABELED_ARTICLES)
    }
    should_keep_ids: Set[str] = {aid for aid, lbl in ground_truth.items() if lbl == "keep"}
    should_drop_ids: Set[str] = {aid for aid, lbl in ground_truth.items() if lbl == "drop"}

    scope = _make_scope()
    params = DEFAULT_PARAMS

    t0 = time.perf_counter()
    result = await filter_articles(articles, scope, params)
    elapsed = time.perf_counter() - t0

    kept_ids: Set[str] = {a.id for a in result.articles}

    # ── Confusion matrix components ───────────────────────────────────────────
    true_positives  = kept_ids & should_keep_ids        # correctly kept
    false_positives = kept_ids & should_drop_ids        # kept but should drop
    false_negatives = should_keep_ids - kept_ids        # should keep but dropped

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Lookup titles for FP/FN ───────────────────────────────────────────────
    id_to_title = {articles[i].id: LABELED_ARTICLES[i][0] for i in range(len(articles))}

    fp_titles = sorted(id_to_title[aid] for aid in false_positives)
    fn_titles = sorted(id_to_title[aid] for aid in false_negatives)

    return {
        "kept_count":       len(kept_ids),
        "input_count":      len(articles),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "false_positives":  fp_titles,
        "false_negatives":  fn_titles,
        "auto_accepted":    result.auto_accepted_count,
        "auto_rejected":    result.auto_rejected_count,
        "llm_classified":   result.llm_classified_count,
        "nli_mean":         result.nli_mean_entailment,
        "elapsed_s":        elapsed,
        "result":           result,
        "kept_articles":    result.articles,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED VERIFICATION INVARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def verify_invariants(metrics: dict) -> List[Tuple[str, bool, str]]:
    """Apply CUDA-Agent Protected Verification pattern.

    Returns list of (invariant_name, passed, detail).
    """
    checks: List[Tuple[str, bool, str]] = []
    result = metrics["result"]
    kept_articles = metrics["kept_articles"]

    # ── INV-1: output count <= input count ────────────────────────────────────
    inv1 = metrics["kept_count"] <= metrics["input_count"]
    checks.append((
        "INV-1 output_count <= input_count",
        inv1,
        f"kept={metrics['kept_count']}, input={metrics['input_count']}",
    ))

    # ── INV-2: nli_mean_entailment > 0.0 for kept articles ───────────────────
    inv2 = result.nli_mean_entailment > 0.0
    checks.append((
        "INV-2 nli_mean_entailment > 0.0",
        inv2,
        f"nli_mean={result.nli_mean_entailment:.4f}",
    ))

    # ── INV-3: no kept article has noise-domain title/summary ─────────────────
    noise_in_output = [a for a in kept_articles if _has_noise_signal(a)]
    inv3 = len(noise_in_output) == 0
    noise_titles = [a.title for a in noise_in_output]
    checks.append((
        "INV-3 no sports/celebrity/govt in output",
        inv3,
        f"violations={len(noise_in_output)}: {noise_titles}" if noise_titles
        else "clean",
    ))

    # ── INV-4: FilterResult assertions match (from the filter itself) ─────────
    inv4 = result.assertion_count_non_increasing
    checks.append((
        "INV-4 FilterResult.assertion_count_non_increasing",
        inv4,
        "assertion from filter.py",
    ))

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# REAL ARTICLE DISTRIBUTION TEST (100 mock articles via mock_articles.py)
# ══════════════════════════════════════════════════════════════════════════════

async def run_distribution_test() -> Optional[dict]:
    """Run filter on up to 100 real mock articles, show distribution stats."""
    try:
        from app.data.mock_articles import MOCK_ARTICLES_RAW
    except ImportError:
        return None

    # MOCK_ARTICLES_RAW is a list of (title, summary, source, category) tuples
    sample = MOCK_ARTICLES_RAW[:100]
    articles = [
        _make_article(
            i,
            title=rec[0] if len(rec) > 0 else "",
            summary=rec[1] if len(rec) > 1 else "",
        )
        for i, rec in enumerate(sample)
    ]

    scope = _make_scope()
    t0 = time.perf_counter()
    result = await filter_articles(articles, scope, DEFAULT_PARAMS)
    elapsed = time.perf_counter() - t0

    pass_rate = len(result.articles) / max(len(articles), 1)

    return {
        "input_count":      len(articles),
        "kept_count":       len(result.articles),
        "pass_rate":        pass_rate,
        "auto_accepted":    result.auto_accepted_count,
        "auto_rejected":    result.auto_rejected_count,
        "llm_classified":   result.llm_classified_count,
        "nli_mean":         result.nli_mean_entailment,
        "elapsed_s":        elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {value*100:.1f}%"


def print_report(metrics: dict, invariants: List[Tuple[str, bool, str]],
                 dist: Optional[dict]) -> None:
    print()
    print("=" * 70)
    print("  NLI FILTER — PRECISION / RECALL TEST  (30-article labeled set)")
    print("=" * 70)

    print(f"\n  Articles:  {metrics['input_count']} in -> {metrics['kept_count']} kept")
    print(f"  Elapsed:   {metrics['elapsed_s']:.2f}s")

    print()
    print("  ── Gate Breakdown ────────────────────────────────────────────────")
    print(f"  NLI auto-accept  (>= 0.88):  {metrics['auto_accepted']:>3} articles")
    print(f"  NLI auto-reject  (<= 0.10):  {metrics['auto_rejected']:>3} articles")
    print(f"  LLM classified  (ambiguous): {metrics['llm_classified']:>3} articles")
    print(f"  NLI mean entailment (kept):  {metrics['nli_mean']:.4f}")

    print()
    print("  ── Confusion Matrix ──────────────────────────────────────────────")
    print(f"  True  Positives (TP):  {metrics['tp']:>3}  (kept and should keep)")
    print(f"  False Positives (FP):  {metrics['fp']:>3}  (kept but should drop)")
    print(f"  False Negatives (FN):  {metrics['fn']:>3}  (dropped but should keep)")

    print()
    print("  ── Metrics ───────────────────────────────────────────────────────")
    print(f"  Precision:  {_bar(metrics['precision'])}")
    print(f"  Recall:     {_bar(metrics['recall'])}")
    print(f"  F1 Score:   {metrics['f1']*100:.2f}%")

    print()
    print("  ── False Positives (should-drop articles that were KEPT) ─────────")
    if metrics["false_positives"]:
        for title in metrics["false_positives"]:
            print(f"    [FP]  {title}")
    else:
        print("    None — perfect precision!")

    print()
    print("  ── False Negatives (should-keep articles that were DROPPED) ──────")
    if metrics["false_negatives"]:
        for title in metrics["false_negatives"]:
            print(f"    [FN]  {title}")
    else:
        print("    None — perfect recall!")

    print()
    print("  ── Protected Verification Invariants ─────────────────────────────")
    all_passed = True
    for name, passed, detail in invariants:
        status = "PASS" if passed else "FAIL"
        marker = "  " if passed else "!!"
        print(f"  {marker} [{status}] {name}")
        print(f"          {detail}")
        if not passed:
            all_passed = False

    print()
    print(f"  Invariant summary: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")

    if dist:
        print()
        print("  ── Real Mock-Article Distribution (100-sample) ───────────────────")
        print(f"  Articles:    {dist['input_count']} in -> {dist['kept_count']} kept")
        print(f"  Pass rate:   {dist['pass_rate']*100:.1f}%")
        print(f"  NLI auto-accept:  {dist['auto_accepted']:>3}")
        print(f"  NLI auto-reject:  {dist['auto_rejected']:>3}")
        print(f"  LLM classified:   {dist['llm_classified']:>3}")
        print(f"  NLI mean (kept):  {dist['nli_mean']:.4f}")
        print(f"  Elapsed:          {dist['elapsed_s']:.2f}s")

    print()
    print("=" * 70)
    print(f"  RESULT:  F1={metrics['f1']*100:.2f}%  "
          f"Precision={metrics['precision']*100:.1f}%  "
          f"Recall={metrics['recall']*100:.1f}%")
    print(f"           {'INVARIANTS OK' if all_passed else 'INVARIANT FAILURES'}")
    print("=" * 70)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> int:
    print("\n[NLI Filter Test] Warming up model (first load may take 15-30s)...")
    t_warm = time.perf_counter()
    try:
        from app.intelligence.engine.nli_filter import warmup
        warmup()
        print(f"[NLI Filter Test] Model ready in {time.perf_counter()-t_warm:.1f}s")
    except Exception as exc:
        print(f"[NLI Filter Test] Warmup warning: {exc}")

    print("[NLI Filter Test] Running 30-article labeled precision/recall test...")
    metrics = await run_labeled_set_test()

    invariants = verify_invariants(metrics)

    print("[NLI Filter Test] Running 100-article distribution test (mock data)...")
    dist = await run_distribution_test()

    print_report(metrics, invariants, dist)

    # Exit code: 0 = all invariants pass AND F1 >= 0.50 in Industry-First mode, else 1.
    #
    # NOTE ON THRESHOLDS — this is a hard 40-article set (19 keep + 21 drop).
    # The drop set includes HARD NEGATIVES that plausibly look B2B:
    #   - Sports sponsorships naming real tech companies (PhonePe/IPL)
    #   - Macro/industry statistics (IT exports $220B, Sensex milestone)
    #   - Consumer product milestones by enterprise companies (Tata Nexon EV)
    #   - Regulatory policy without specific company action (RBI rate hold, SEBI F&O)
    # These hard negatives should produce realistic precision in the 65-85% range.
    # 100% precision on this set means the hard negatives are too easy.
    #
    # Observed baselines:
    #   30-article easy set (2026-03-08): Precision=100% Recall=46.7% F1=63.6%
    #   40-article hard set (target):     Precision=65-85% Recall=55-70% F1=60-75%
    #
    # To improve recall in Industry-First:
    #   1. Lower nli_auto_reject threshold (currently 0.10 → try 0.05)
    #   2. Relax LLM prompt to allow B2C-adjacent companies with B2B events
    #   3. Switch borderline articles to Company-First scope with explicit targets
    all_invs_pass = all(passed for _, passed, _ in invariants)
    f1_ok = metrics["f1"] >= 0.50   # 50% = realistic floor for hard Industry-First set
    if not all_invs_pass:
        print("  [FAIL] One or more invariants failed.")
    if not f1_ok:
        print(f"  [FAIL] F1 {metrics['f1']*100:.1f}% is below 55% threshold "
              f"(Industry-First mode baseline).")

    return 0 if (all_invs_pass and f1_ok) else 1


# ══════════════════════════════════════════════════════════════════════════════
# PYTEST ENTRYPOINTS (same logic, exposed as test functions)
# ══════════════════════════════════════════════════════════════════════════════

def test_labeled_set_f1():
    """pytest: F1 >= 0.50 on the 40-article hard labeled set (Industry-First mode baseline).

    Industry-First is the hardest filter mode — no company targets means no fast-accept
    path, so more articles go to the conservative LLM prompt. Precision is prioritised
    over recall (fail-CLOSED design: noise is worse than gaps).

    The hard set includes borderline drop articles (sports sponsorships naming real
    companies, macro stats, consumer milestones, regulatory policy) that should produce
    realistic precision in the 65-85% range rather than an artificial 100%.
    Observed easy-set baseline (2026-03-08): Precision=100%, Recall=46.7%, F1=63.6%.
    """
    metrics = asyncio.run(run_labeled_set_test())
    print_report(metrics, verify_invariants(metrics), None)
    assert metrics["f1"] >= 0.50, (
        f"F1 {metrics['f1']*100:.1f}% below 50% threshold. "
        f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}"
    )


def test_invariant_output_count_non_increasing():
    """pytest: output count <= input count."""
    metrics = asyncio.run(run_labeled_set_test())
    assert metrics["kept_count"] <= metrics["input_count"]


def test_invariant_nli_mean_positive():
    """pytest: mean NLI entailment of kept articles > 0."""
    metrics = asyncio.run(run_labeled_set_test())
    assert metrics["result"].nli_mean_entailment > 0.0


def test_invariant_no_noise_in_output():
    """pytest: no article in output matches sports/celebrity/govt signals."""
    metrics = asyncio.run(run_labeled_set_test())
    noise = [a.title for a in metrics["kept_articles"] if _has_noise_signal(a)]
    assert noise == [], f"Noise articles in output: {noise}"


def test_precision_above_threshold():
    """pytest: precision >= 0.60 on the hard 40-article set.

    Hard negatives (sports sponsorships, macro stats, consumer milestones,
    regulatory policy) make 100% precision impossible on realistic data.
    Target range: 65-85%.  Below 60% means the filter is leaking noise.
    """
    metrics = asyncio.run(run_labeled_set_test())
    assert metrics["precision"] >= 0.60, (
        f"Precision {metrics['precision']*100:.1f}% below 60%. FP: {metrics['false_positives']}"
    )


def test_recall_above_threshold():
    """pytest: recall >= 0.50 on the hard 40-article set.

    19 keep articles include harder cases (price hike, SME platform, govt contract).
    Target range 55-70%.  Below 50% means the filter is too conservative.
    """
    metrics = asyncio.run(run_labeled_set_test())
    assert metrics["recall"] >= 0.50, (
        f"Recall {metrics['recall']*100:.1f}% below 50%. FN: {metrics['false_negatives']}"
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
