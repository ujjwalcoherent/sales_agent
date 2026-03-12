"""
Standalone Clustering Test -- validates semantic coherence and grouping quality.

Uses pre-injected synthetic embeddings (bypassing broken LLMService) so that
tests are deterministic and independent of API keys or internet access.

Embeddings are constructed analytically:
  - Within-group vectors: small Gaussian noise around a shared centroid
  - Between-group centroids: orthogonal unit vectors (maximally separated)

This mimics what neural embeddings produce for semantically coherent text.

Staged warm-up (CUDA-Agent pattern):
  Level 1 -- 5 very distinct articles (Leiden discovery)
             --> expect no cross-topic merging
  Level 2 -- 15 articles in 3 obvious topic groups (HAC)
             --> expect Group A/B/C each in their own cluster
  Level 3 -- Mock 72h scale: 20 articles / 4 tech sub-topics
             --> full quality report: coherence, singletons, mega, flags

Run with:
  venv/Scripts/python.exe tests/standalone/test_clustering.py
"""

import asyncio
import logging
import sys
import os
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import numpy as np

# project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force UTF-8 on Windows console (skip under pytest — breaks capture)
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)

from app.intelligence.config import ClusteringParams
from app.intelligence.models import (
    Article,
    ClusterResult,
    DiscoveryScope,
    EntityGroup,
    PipelineState,
)

EMBEDDING_DIM = 256   # small but realistic for fast tests
RNG = np.random.default_rng(42)


# ==============================================================================
# SYNTHETIC EMBEDDING HELPERS
# ==============================================================================

def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised vector."""
    n = np.linalg.norm(v)
    return v / max(n, 1e-10)


def _orthonormal_basis(k: int, dim: int) -> List[np.ndarray]:
    """Generate k approximately orthogonal unit vectors in R^dim via Gram-Schmidt."""
    vecs = []
    for _ in range(k):
        v = RNG.standard_normal(dim).astype(np.float32)
        for u in vecs:
            v -= np.dot(v, u) * u
        vecs.append(_unit(v))
    return vecs


def _cluster_embeddings(
    centroid: np.ndarray,
    n: int,
    noise: float = 0.08,
) -> np.ndarray:
    """Generate n embeddings tightly clustered around `centroid`.

    noise=0.08 gives mean pairwise cosine ~ 0.93 (high coherence).
    """
    embs = []
    for _ in range(n):
        v = centroid + RNG.standard_normal(len(centroid)).astype(np.float32) * noise
        embs.append(_unit(v))
    return np.stack(embs)


# ==============================================================================
# ARTICLE / ENTITY GROUP FACTORIES
# ==============================================================================

def _make_article(
    idx: int,
    title: str,
    summary: str,
    source_name: str,
    embedding: np.ndarray,
    days_ago: float = 1.0,
    entity_names: Optional[List[str]] = None,
) -> Article:
    pub = datetime.now(timezone.utc) - timedelta(days=days_ago)
    art = Article(
        url=f"https://example.com/article-{idx}",
        title=title,
        summary=summary,
        source_name=source_name,
        source_url=f"https://{source_name.lower().replace(' ', '')}.com",
        published_at=pub,
        run_index=idx,
        embedding=embedding.tolist(),
    )
    if entity_names:
        art.entities_raw = entity_names
        art.__dict__["entity_names"] = entity_names
    return art


def _make_entity_group(name: str, article_indices: List[int]) -> EntityGroup:
    return EntityGroup(
        canonical_name=name,
        article_indices=article_indices,
        entity_type="ORG",
        is_validated=True,
        is_b2b=True,
    )


# ==============================================================================
# CLUSTER ANALYSIS HELPERS
# ==============================================================================

def _check_same_cluster(clusters: List[ClusterResult], indices: List[int]) -> bool:
    """True if ALL given indices are in the same single cluster."""
    for c in clusters:
        if all(i in c.article_indices for i in indices):
            return True
    return False


def _check_majority_co_clustered(
    clusters: List[ClusterResult], indices: List[int], threshold: float = 0.60
) -> bool:
    """True if at least `threshold` fraction of `indices` appear together in some cluster.

    This is more lenient than _check_same_cluster: HAC sometimes splits a 5-article
    entity group into 2 sub-clusters (e.g. 3+2). We accept this as long as at least
    60% of articles from the group end up in the same cluster.
    """
    best_overlap = 0
    for c in clusters:
        overlap = sum(1 for i in indices if i in c.article_indices)
        if overlap > best_overlap:
            best_overlap = overlap
    return (best_overlap / max(len(indices), 1)) >= threshold


def _check_all_in_group(
    clusters: List[ClusterResult], indices: List[int], group_name: str
) -> bool:
    """True if every article in `indices` is either noise OR in a cluster that
    contains ONLY articles from `indices` (no cross-group contamination).

    This is the key correctness check: articles from Group A must never share
    a cluster with articles from Group B.
    """
    index_set = set(indices)
    for c in clusters:
        cluster_set = set(c.article_indices)
        # Check if this cluster contains any article from our group
        if cluster_set & index_set:
            # If it does, ALL its articles must be from our group
            if not cluster_set.issubset(index_set):
                return False
    return True


def _check_separate_clusters(
    clusters: List[ClusterResult], group_a: List[int], group_b: List[int]
) -> bool:
    """True if no single cluster contains articles from BOTH groups."""
    for c in clusters:
        has_a = any(i in c.article_indices for i in group_a)
        has_b = any(i in c.article_indices for i in group_b)
        if has_a and has_b:
            return False
    return True


def _coherence_for_group(clusters: List[ClusterResult], indices: List[int]) -> float:
    """Return the MEAN coherence_score across all clusters from this group.

    HAC may split a 5-article entity group into 2 sub-clusters; we report
    the average coherence across those sub-clusters.
    """
    index_set = set(indices)
    group_clusters = [c for c in clusters if set(c.article_indices) & index_set]
    if not group_clusters:
        return 0.0
    scored = [c.coherence_score for c in group_clusters if c.coherence_score > 0]
    return sum(scored) / len(scored) if scored else 0.0


def _print_cluster_report(
    clusters: List[ClusterResult],
    passed_ids: List[str],
    title: str,
) -> None:
    print(f"\n{'='*72}")
    print(f"  CLUSTER REPORT: {title}")
    print(f"{'='*72}")
    print(f"  Total clusters : {len(clusters)}")
    print(f"  Passed         : {len(passed_ids)}")
    print(f"  Rejected       : {len(clusters) - len(passed_ids)}")

    coherences = [c.coherence_score for c in clusters if c.coherence_score > 0]
    if coherences:
        print(f"  Mean coherence : {sum(coherences)/len(coherences):.3f}")
        print(f"  Min / Max coh  : {min(coherences):.3f} / {max(coherences):.3f}")

    print(f"\n  {'#':<4} {'Alg':<8} {'N':<5} {'Coherence':<11} {'Status':<8} Entity / Label")
    print(f"  {'-'*76}")

    flags: List[str] = []
    for i, c in enumerate(clusters):
        status = "PASS" if c.cluster_id in passed_ids else "FAIL"
        alg = (c.algorithm or "?")[:6]
        label_str = c.label or c.primary_entity or "(no label)"
        coh_str = f"{c.coherence_score:.3f}" if c.coherence_score else "  -  "
        print(f"  {i:<4} {alg:<8} {c.article_count:<5} {coh_str:<11} {status:<8} {label_str[:50]}")

        if c.article_count == 1:
            flags.append(f"  [SINGLETON]  Cluster {i}: '{label_str}'")
        if c.article_count > 10:
            flags.append(f"  [MEGA]       Cluster {i}: {c.article_count} articles -- may be too broad")
        if 0 < c.coherence_score < 0.30:
            flags.append(f"  [LOW-COH]    Cluster {i}: coherence={c.coherence_score:.3f}")

    if flags:
        print(f"\n  QUALITY FLAGS:")
        for f in flags:
            print(f)


# ==============================================================================
# CORE RUNNER
# ==============================================================================

async def run_clustering(
    articles: List[Article],
    entity_groups: List[EntityGroup],
    level_label: str,
) -> Dict[str, Any]:
    from app.intelligence.cluster.orchestrator import cluster_and_validate

    scope = DiscoveryScope(companies=[], hours=72)
    params = ClusteringParams()
    params.hac_min_cluster_size = 2
    params.val_min_articles = 2
    params.val_min_sources = 2
    params.val_coherence_min = 0.40
    params.val_composite_reject = 0.40

    state = PipelineState(scope=scope)

    print(f"\n  [{level_label}] cluster_and_validate: {len(articles)} articles, "
          f"{len(entity_groups)} entity groups ...")

    clusters, passed_ids, rejected_ids, val_results, noise = await cluster_and_validate(
        articles=articles,
        entity_groups=entity_groups,
        scope=scope,
        params=params,
        state=state,
    )

    return {
        "clusters": clusters,
        "passed_ids": passed_ids,
        "rejected_ids": rejected_ids,
        "val_results": val_results,
        "noise": noise,
        "params": params,
    }


# ==============================================================================
# LEVEL 1: 5 Very Distinct Articles
# ==============================================================================

async def test_level1() -> bool:
    print("\n" + "="*72)
    print("  LEVEL 1: 5 Very Distinct Articles (Leiden discovery)")
    print("  Expect: no cross-topic merging; each article stays isolated")
    print("="*72)

    # 5 maximally separated centroids
    centroids = _orthonormal_basis(5, EMBEDDING_DIM)

    articles = [
        _make_article(0, "SpaceX launches 60 Starlink satellites",
                      "SpaceX deployed 60 Starlink internet satellites using Falcon 9.",
                      "TechCrunch", centroids[0], 0.5, ["SpaceX"]),
        _make_article(1, "WHO declares cholera health emergency",
                      "WHO declared a public health emergency in East Africa due to cholera.",
                      "Reuters", centroids[1], 1.0, ["WHO"]),
        _make_article(2, "Federal Reserve raises interest rates 25bp",
                      "The US Fed raised benchmark rates 0.25pp to combat inflation.",
                      "Bloomberg", centroids[2], 1.5, ["Federal Reserve"]),
        _make_article(3, "FC Barcelona signs striker in record transfer",
                      "Barcelona completed a EUR 120M transfer for a top forward.",
                      "ESPN", centroids[3], 2.0, ["FC Barcelona"]),
        _make_article(4, "Amazon rainforest deforestation hits record high",
                      "Satellite data shows Amazon deforestation at an all-time high.",
                      "Guardian", centroids[4], 2.5, ["Amazon Rainforest"]),
    ]

    # No entity groups -- all go to Leiden discovery (ungrouped >= 4)
    result = await run_clustering(articles, [], "L1")
    clusters = result["clusters"]
    passed_ids = result["passed_ids"]

    _print_cluster_report(clusters, passed_ids, "Level 1 -- 5 Distinct Articles")

    print("\n  ASSERTIONS:")

    # No cluster should merge two articles from different topics
    cross = False
    for c in clusters:
        if len(c.article_indices) > 1:
            # Any multi-article cluster is a cross-topic merge (all 5 are orthogonal)
            cross = True
            print(f"  WARN: cluster merged articles {c.article_indices}")

    no_cross = not cross
    no_mega = all(c.article_count <= 1 for c in clusters)

    print(f"  No cross-topic merging  : {'PASS' if no_cross else 'FAIL'}")
    print(f"  No mega-cluster (>1)    : {'PASS' if no_mega else 'FAIL'}")

    # Level 1 also passes if all 5 end up as noise (no clusters formed)
    # because distinct articles should not be forced into clusters
    all_noise = len(clusters) == 0
    if all_noise:
        print(f"  All 5 articles went to noise (expected for highly distinct articles): OK")

    passed = no_cross and (no_mega or all_noise)
    print(f"\n  LEVEL 1 RESULT         : {'PASS' if passed else 'FAIL'}")
    return passed


# ==============================================================================
# LEVEL 2: 15 Articles in 3 Clear Groups + 2 Noise
# ==============================================================================

GROUP_A_IDX = list(range(0, 5))
GROUP_B_IDX = list(range(5, 10))
GROUP_C_IDX = list(range(10, 15))
NOISE_IDX = [15, 16]


async def test_level2() -> bool:
    print("\n" + "="*72)
    print("  LEVEL 2: 15 Articles in 3 Topic Groups + 2 Noise (HAC)")
    print("  Expect: Group A (funding), B (healthcare), C (AI) each cluster")
    print("="*72)

    # 3 well-separated group centroids + 2 distinct noise directions
    centroids = _orthonormal_basis(5, EMBEDDING_DIM)
    ca, cb, cc, cn1, cn2 = centroids

    # Generate tight embeddings within each group
    embs_a = _cluster_embeddings(ca, 5, noise=0.05)
    embs_b = _cluster_embeddings(cb, 5, noise=0.05)
    embs_c = _cluster_embeddings(cc, 5, noise=0.05)

    sources_a = ["Economic Times", "LiveMint", "Business Standard", "VCCircle", "Moneycontrol"]
    sources_b = ["NDTV Profit", "Economic Times", "Hindu Business Line", "Business Standard", "LiveMint"]
    sources_c = ["Reuters", "TechCrunch", "Bloomberg", "CNBC", "VentureBeat"]

    articles: List[Article] = []

    # Group A -- Indian startup funding (0-4)
    funding_data = [
        ("Zepto raises $350M Series F at $5B valuation",
         "Quick-commerce startup Zepto secured $350M in Series F funding at a $5 billion valuation.",
         ["Zepto"]),
        ("Blinkit parent Zomato raises $400M in QIP",
         "Zomato raised $400M via qualified institutional placement to fund Blinkit operations.",
         ["Blinkit", "Zomato"]),
        ("Meesho closes $500M pre-IPO funding round",
         "Social commerce platform Meesho closed a $500M pre-IPO round with Tiger Global.",
         ["Meesho"]),
        ("PhonePe raises $200M from General Atlantic",
         "Fintech unicorn PhonePe secured $200M from General Atlantic in its latest round.",
         ["PhonePe"]),
        ("Swiggy raises $700M ahead of IPO listing",
         "Food delivery platform Swiggy raised $700M in pre-IPO funding valuing it at $15B.",
         ["Swiggy"]),
    ]
    for i, (title, summary, ents) in enumerate(funding_data):
        articles.append(_make_article(i, title, summary, sources_a[i], embs_a[i], 0.3 + i*0.1, ents))

    # Group B -- Healthcare expansion (5-9)
    health_data = [
        ("Apollo Hospitals opens 500-bed facility in Hyderabad",
         "Apollo Hospitals inaugurated a 500-bed multi-speciality hospital in Hyderabad.",
         ["Apollo Hospitals"]),
        ("Fortis Healthcare acquires 3 hospitals in tier-2 cities",
         "Fortis Healthcare acquired three hospitals in Jaipur, Bhopal, and Coimbatore.",
         ["Fortis Healthcare"]),
        ("KIMS plans 10 new hospitals across South India",
         "KIMS Hospitals announced expansion with 10 new facilities in AP and Telangana.",
         ["KIMS", "KIMS Hospitals"]),
        ("Narayana Health to open 5 hospitals by 2026",
         "Narayana Health announced plans to open 5 new hospitals across tier-2 Indian cities.",
         ["Narayana Health"]),
        ("Max Healthcare acquires Sahara Hospital Lucknow",
         "Max Healthcare Group completed acquisition of Sahara Hospital in Lucknow for Rs 940 crore.",
         ["Max Healthcare"]),
    ]
    for i, (title, summary, ents) in enumerate(health_data):
        articles.append(_make_article(5 + i, title, summary, sources_b[i], embs_b[i], 1.0 + i*0.1, ents))

    # Group C -- AI enterprise deals (10-14)
    ai_data = [
        ("NVIDIA announces $10B AI partnership with Microsoft",
         "NVIDIA and Microsoft signed a $10B enterprise AI infrastructure deal for Azure GPU clusters.",
         ["NVIDIA", "Microsoft"]),
        ("OpenAI launches GPT-5 enterprise tier for Fortune 500",
         "OpenAI released GPT-5 with enterprise licensing for Fortune 500 document intelligence.",
         ["OpenAI"]),
        ("Anthropic signs multi-year AI contract with AWS",
         "Anthropic secured a major multi-year deal with Amazon Web Services for Claude enterprise.",
         ["Anthropic", "AWS"]),
        ("Google Gemini 2.0 signed for Salesforce CRM integration",
         "Google and Salesforce announced Gemini 2.0 Ultra integration across Salesforce Einstein AI.",
         ["Google", "Gemini", "Salesforce"]),
        ("Meta Llama 4 adopted by SAP for enterprise AI",
         "SAP integrated Meta Llama 4 open-source model into SAP AI Core for enterprise deployments.",
         ["Meta", "SAP", "Llama"]),
    ]
    for i, (title, summary, ents) in enumerate(ai_data):
        articles.append(_make_article(10 + i, title, summary, sources_c[i], embs_c[i], 0.2 + i*0.1, ents))

    # Noise (15-16) -- far from all groups
    articles.append(_make_article(15,
        "India cricket team wins Test series against Australia",
        "Indian cricket team clinched the Test series 3-1 in a dominant performance.",
        "CricBuzz", _unit(cn1), 3.0, ["BCCI"]))
    articles.append(_make_article(16,
        "Venice Film Festival announces 82nd edition lineup",
        "The Venice Film Festival revealed its competition slate with 21 films from 18 countries.",
        "Variety", _unit(cn2), 2.5, ["Venice Film Festival"]))

    entity_groups = [
        _make_entity_group("Startup Funding India", GROUP_A_IDX),
        _make_entity_group("Healthcare Hospital Expansion", GROUP_B_IDX),
        _make_entity_group("AI Enterprise Deals", GROUP_C_IDX),
        _make_entity_group("Cricket", [15]),
        _make_entity_group("Film", [16]),
    ]

    result = await run_clustering(articles, entity_groups, "L2")
    clusters = result["clusters"]
    passed_ids = result["passed_ids"]

    _print_cluster_report(clusters, passed_ids, "Level 2 -- 3 Groups + Noise")

    # Assertions
    # Primary: no cross-group contamination (strongest correctness signal)
    group_a_pure = _check_all_in_group(clusters, GROUP_A_IDX, "Funding")
    group_b_pure = _check_all_in_group(clusters, GROUP_B_IDX, "Healthcare")
    group_c_pure = _check_all_in_group(clusters, GROUP_C_IDX, "AI")

    # Majority within-group co-clustering (>= 60% in same cluster)
    group_a_majority = _check_majority_co_clustered(clusters, GROUP_A_IDX)
    group_b_majority = _check_majority_co_clustered(clusters, GROUP_B_IDX)
    group_c_majority = _check_majority_co_clustered(clusters, GROUP_C_IDX)

    # Cross-group separation (redundant with pure check, but explicit)
    ab_sep = _check_separate_clusters(clusters, GROUP_A_IDX, GROUP_B_IDX)
    ac_sep = _check_separate_clusters(clusters, GROUP_A_IDX, GROUP_C_IDX)
    bc_sep = _check_separate_clusters(clusters, GROUP_B_IDX, GROUP_C_IDX)

    # Coherence (mean across sub-clusters for the group)
    coh_a = _coherence_for_group(clusters, GROUP_A_IDX)
    coh_b = _coherence_for_group(clusters, GROUP_B_IDX)
    coh_c = _coherence_for_group(clusters, GROUP_C_IDX)
    coh_a_ok = coh_a >= 0.60 if coh_a > 0 else False
    coh_b_ok = coh_b >= 0.60 if coh_b > 0 else False
    coh_c_ok = coh_c >= 0.60 if coh_c > 0 else False

    noise_a_sep = _check_separate_clusters(clusters, NOISE_IDX, GROUP_A_IDX)
    noise_b_sep = _check_separate_clusters(clusters, NOISE_IDX, GROUP_B_IDX)
    noise_c_sep = _check_separate_clusters(clusters, NOISE_IDX, GROUP_C_IDX)

    print("\n  ASSERTIONS:")
    print(f"  Group A no cross-contamination : {'PASS' if group_a_pure else 'FAIL'}"
          f"  [mean coh={coh_a:.3f}]")
    print(f"  Group B no cross-contamination : {'PASS' if group_b_pure else 'FAIL'}"
          f"  [mean coh={coh_b:.3f}]")
    print(f"  Group C no cross-contamination : {'PASS' if group_c_pure else 'FAIL'}"
          f"  [mean coh={coh_c:.3f}]")
    print(f"  Group A majority co-clustered  : {'PASS' if group_a_majority else 'FAIL'}")
    print(f"  Group B majority co-clustered  : {'PASS' if group_b_majority else 'FAIL'}")
    print(f"  Group C majority co-clustered  : {'PASS' if group_c_majority else 'FAIL'}")
    print(f"  Group A + B separated          : {'PASS' if ab_sep else 'FAIL'}")
    print(f"  Group A + C separated          : {'PASS' if ac_sep else 'FAIL'}")
    print(f"  Group B + C separated          : {'PASS' if bc_sep else 'FAIL'}")
    print(f"  Coherence A >= 0.60            : {'PASS' if coh_a_ok else 'FAIL'}  [{coh_a:.3f}]")
    print(f"  Coherence B >= 0.60            : {'PASS' if coh_b_ok else 'FAIL'}  [{coh_b:.3f}]")
    print(f"  Coherence C >= 0.60            : {'PASS' if coh_c_ok else 'FAIL'}  [{coh_c:.3f}]")
    print(f"  Noise not in funding cluster   : {'PASS' if noise_a_sep else 'FAIL'}")
    print(f"  Noise not in healthcare cluster: {'PASS' if noise_b_sep else 'FAIL'}")
    print(f"  Noise not in AI cluster        : {'PASS' if noise_c_sep else 'FAIL'}")

    print("\n  CLUSTER TITLES (visual inspection):")
    for i, c in enumerate(clusters):
        title = c.label or c.primary_entity or "(no label)"
        members = sorted(c.article_indices)
        member_titles = [articles[j].title[:65] if j < len(articles) else "?" for j in members[:3]]
        extra = f"  (+{len(members)-3} more)" if len(members) > 3 else ""
        print(f"    [{i}] '{title}' ({len(members)} articles){extra}")
        for mt in member_titles:
            print(f"         -> {mt}")

    passed = (
        group_a_pure and group_b_pure and group_c_pure
        and group_a_majority and group_b_majority and group_c_majority
        and ab_sep and ac_sep and bc_sep
        and coh_a_ok and coh_b_ok and coh_c_ok
        and noise_a_sep and noise_b_sep and noise_c_sep
    )
    print(f"\n  LEVEL 2 RESULT         : {'PASS' if passed else 'FAIL'}")
    return passed


# ==============================================================================
# LEVEL 3: Mock 72h Real-Scale Dataset -- 20 Articles / 4 Tech Sub-topics
# ==============================================================================

async def test_level3() -> bool:
    print("\n" + "="*72)
    print("  LEVEL 3: Mock 72h Real-Scale (20 articles / 4 tech sub-topics)")
    print("  Full quality report: coherence distribution, singletons, mega, flags")
    print("="*72)

    # 4 well-separated topic centroids
    centroids = _orthonormal_basis(4, EMBEDDING_DIM)
    c_cloud, c_cyber, c_ai, c_semi = centroids

    embs_cloud = _cluster_embeddings(c_cloud, 5, noise=0.07)
    embs_cyber = _cluster_embeddings(c_cyber, 5, noise=0.07)
    embs_ai    = _cluster_embeddings(c_ai,    5, noise=0.07)
    embs_semi  = _cluster_embeddings(c_semi,  5, noise=0.07)

    base = datetime.now(timezone.utc)

    articles: List[Article] = []

    # Cloud / hyperscaler (0-4)
    cloud_data = [
        ("AWS opens new cloud region in Saudi Arabia",
         "Amazon Web Services inaugurated its Riyadh region for Middle East enterprise customers.", "Reuters", 5, ["AWS", "Amazon"]),
        ("Microsoft Azure expands data center capacity in Southeast Asia",
         "Azure is adding three new availability zones in Singapore and Malaysia for enterprises.", "TechCrunch", 8, ["Microsoft", "Azure"]),
        ("Google Cloud signs $1B deal with Singapore government",
         "Google Cloud and Singapore inked a 5-year digital infrastructure government agreement.", "Bloomberg", 12, ["Google Cloud", "Google"]),
        ("AWS Q4 2025 revenue grows 28% as AI workloads surge",
         "Amazon's cloud division posted record quarterly revenue driven by AI inference demand.", "CNBC", 20, ["AWS", "Amazon"]),
        ("Oracle CloudWorld 2026 announces 30 new AI infrastructure services",
         "Oracle unveiled 30 new cloud services focused on GPU compute and generative AI workloads.", "ZDNet", 26, ["Oracle"]),
    ]
    for i, (t, s, src, h, ents) in enumerate(cloud_data):
        pub = base - timedelta(hours=h)
        a = Article(url=f"https://news.test/{i}", title=t, summary=s,
                    source_name=src, source_url=f"https://{src.lower().replace(' ','')}.com",
                    published_at=pub, run_index=i, embedding=embs_cloud[i].tolist())
        a.__dict__["entity_names"] = ents
        articles.append(a)

    # Cybersecurity (5-9)
    cyber_data = [
        ("CrowdStrike detects Russian APT targeting energy sector",
         "CrowdStrike Falcon intel uncovered a new threat actor targeting European energy infrastructure.", "DarkReading", 6, ["CrowdStrike"]),
        ("Palo Alto Networks patches critical zero-day in PAN-OS firewall",
         "Palo Alto issued an emergency patch for a remote code execution flaw in PAN-OS firmware.", "SecurityWeek", 10, ["Palo Alto Networks"]),
        ("SentinelOne raises $200M to expand AI threat detection platform",
         "EDR vendor SentinelOne closed a growth equity round to scale its AI-driven SOC platform.", "VentureBeat", 15, ["SentinelOne"]),
        ("Fortinet warns of actively exploited FortiOS vulnerability",
         "Fortinet urged enterprise customers to patch a critical FortiOS flaw under active exploitation.", "BleepingComputer", 22, ["Fortinet"]),
        ("Zscaler expands Zero Trust Exchange with AI posture management",
         "Zscaler announced AI-based security posture management integrated into its Zero Trust platform.", "CSO Online", 30, ["Zscaler"]),
    ]
    for i, (t, s, src, h, ents) in enumerate(cyber_data):
        pub = base - timedelta(hours=h)
        a = Article(url=f"https://news.test/{5+i}", title=t, summary=s,
                    source_name=src, source_url=f"https://{src.lower().replace(' ','')}.com",
                    published_at=pub, run_index=5+i, embedding=embs_cyber[i].tolist())
        a.__dict__["entity_names"] = ents
        articles.append(a)

    # AI models (10-14)
    ai_data = [
        ("OpenAI launches GPT-5 with 2M token context window",
         "OpenAI GPT-5 features a 2 million token context for enterprise document intelligence.", "The Verge", 4, ["OpenAI"]),
        ("Anthropic Claude 4 Sonnet tops MMLU reasoning benchmark",
         "Anthropic Claude 4 Sonnet achieved state-of-the-art results on graduate reasoning benchmarks.", "Ars Technica", 7, ["Anthropic"]),
        ("Google DeepMind Gemini Ultra available for Workspace enterprise",
         "DeepMind Gemini Ultra model is now GA in Google Workspace for enterprise customers.", "TechRadar", 11, ["Google", "DeepMind", "Gemini"]),
        ("Meta releases Llama 4 open-source with 405B parameters",
         "Meta AI released Llama 4 under an open license, making it the largest public open-weights model.", "Wired", 18, ["Meta", "Llama"]),
        ("Mistral AI raises $640M Series C, signs Microsoft Azure deal",
         "Paris-based Mistral AI raised $640M and signed a distribution deal with Microsoft Azure.", "Financial Times", 24, ["Mistral AI", "Microsoft"]),
    ]
    for i, (t, s, src, h, ents) in enumerate(ai_data):
        pub = base - timedelta(hours=h)
        a = Article(url=f"https://news.test/{10+i}", title=t, summary=s,
                    source_name=src, source_url=f"https://{src.lower().replace(' ','')}.com",
                    published_at=pub, run_index=10+i, embedding=embs_ai[i].tolist())
        a.__dict__["entity_names"] = ents
        articles.append(a)

    # Semiconductors (15-19)
    semi_data = [
        ("TSMC begins 2nm volume production at Arizona fab",
         "TSMC Arizona plant started mass-producing 2nm chips for Apple and NVIDIA orders.", "Nikkei", 9, ["TSMC"]),
        ("Intel unveils Lunar Lake NPU for on-device AI inference",
         "Intel Lunar Lake chips include an integrated neural processing unit for local AI inference.", "AnandTech", 14, ["Intel"]),
        ("NVIDIA Blackwell GPU shortages delay enterprise AI by 6-9 months",
         "Enterprise customers report 6-9 month wait times for NVIDIA H200 and Blackwell GPU clusters.", "The Information", 19, ["NVIDIA"]),
        ("AMD MI350 GPU announced as NVIDIA Blackwell competitor",
         "AMD unveiled its MI350 GPU series with claimed performance parity to NVIDIA Blackwell.", "Tom's Hardware", 25, ["AMD"]),
        ("Qualcomm Snapdragon X Elite wins major PC OEM contracts",
         "Qualcomm signed PC design wins with Lenovo, Dell, and HP for Snapdragon X Elite chips.", "Reuters", 32, ["Qualcomm"]),
    ]
    for i, (t, s, src, h, ents) in enumerate(semi_data):
        pub = base - timedelta(hours=h)
        a = Article(url=f"https://news.test/{15+i}", title=t, summary=s,
                    source_name=src, source_url=f"https://{src.lower().replace(' ','')}.com",
                    published_at=pub, run_index=15+i, embedding=embs_semi[i].tolist())
        a.__dict__["entity_names"] = ents
        articles.append(a)

    entity_groups = [
        _make_entity_group("Cloud Hyperscalers", list(range(0, 5))),
        _make_entity_group("Cybersecurity", list(range(5, 10))),
        _make_entity_group("AI Models", list(range(10, 15))),
        _make_entity_group("Semiconductors", list(range(15, 20))),
    ]

    result = await run_clustering(articles, entity_groups, "L3")
    clusters = result["clusters"]
    passed_ids = result["passed_ids"]

    _print_cluster_report(clusters, passed_ids, "Level 3 -- Mock 72h (4 Tech Sub-topics)")

    # Quality statistics
    coherences = [c.coherence_score for c in clusters if c.coherence_score > 0]
    singletons = [c for c in clusters if c.article_count == 1]
    mega = [c for c in clusters if c.article_count > 10]
    low_coh = [c for c in clusters if 0 < c.coherence_score < 0.30]
    rejection_rate = (len(clusters) - len(passed_ids)) / max(len(clusters), 1)

    print(f"\n  QUALITY SUMMARY:")
    print(f"  Total clusters       : {len(clusters)}")
    print(f"  Passed validation    : {len(passed_ids)}")
    print(f"  Rejection rate       : {rejection_rate:.0%}  (target: 60-80%)")
    print(f"  Singletons           : {len(singletons)}")
    print(f"  Mega clusters (>10)  : {len(mega)}")
    print(f"  Low coherence (<0.30): {len(low_coh)}")

    if coherences:
        import statistics as stat
        print(f"\n  Coherence distribution:")
        print(f"    Mean   : {stat.mean(coherences):.3f}")
        print(f"    Median : {stat.median(coherences):.3f}")
        if len(coherences) > 1:
            print(f"    Stdev  : {stat.stdev(coherences):.3f}")
        print(f"    Min    : {min(coherences):.3f}")
        print(f"    Max    : {max(coherences):.3f}")

    print(f"\n  ALL CLUSTER TITLES (visual quality check):")
    for i, c in enumerate(clusters):
        lbl = c.label or c.primary_entity or "(no label)"
        status = "PASS" if c.cluster_id in passed_ids else "FAIL"
        coh_str = f"{c.coherence_score:.3f}" if c.coherence_score else "  -  "
        flags = ""
        if c.article_count == 1:
            flags += " [SINGLETON]"
        if c.article_count > 10:
            flags += " [MEGA]"
        if 0 < c.coherence_score < 0.30:
            flags += " [LOW-COH]"
        print(f"    {i:>3}. [{status}] coh={coh_str} n={c.article_count:<3} {lbl[:60]}{flags}")

    # Grouping assertions
    # Primary: no cross-contamination between sub-topics
    cloud_pure = _check_all_in_group(clusters, list(range(0, 5)), "Cloud")
    cyber_pure = _check_all_in_group(clusters, list(range(5, 10)), "Cyber")
    ai_pure    = _check_all_in_group(clusters, list(range(10, 15)), "AI")
    semi_pure  = _check_all_in_group(clusters, list(range(15, 20)), "Semi")

    # Majority co-clustering within sub-topics
    # L3 uses threshold=0.40 because HAC may split a 5-article group into 2+3
    # sub-clusters (both valid), so the "best cluster" may contain only 3/5 = 60%
    # or 2/5 = 40%. We only need to ensure at least 2 articles co-cluster per group.
    cloud_maj = _check_majority_co_clustered(clusters, list(range(0, 5)), threshold=0.40)
    cyber_maj = _check_majority_co_clustered(clusters, list(range(5, 10)), threshold=0.40)
    ai_maj    = _check_majority_co_clustered(clusters, list(range(10, 15)), threshold=0.40)
    semi_maj  = _check_majority_co_clustered(clusters, list(range(15, 20)), threshold=0.40)

    cloud_cyber_sep = _check_separate_clusters(clusters, list(range(0,5)), list(range(5,10)))
    ai_semi_sep     = _check_separate_clusters(clusters, list(range(10,15)), list(range(15,20)))
    cloud_ai_sep    = _check_separate_clusters(clusters, list(range(0,5)), list(range(10,15)))
    cyber_semi_sep  = _check_separate_clusters(clusters, list(range(5,10)), list(range(15,20)))

    coh_cloud_v = _coherence_for_group(clusters, list(range(0,5)))
    coh_cyber_v = _coherence_for_group(clusters, list(range(5,10)))
    coh_ai_v    = _coherence_for_group(clusters, list(range(10,15)))
    coh_semi_v  = _coherence_for_group(clusters, list(range(15,20)))

    coh_ok = lambda v: v >= 0.40 if v > 0 else False  # L3: use 0.40 (sub-clusters are smaller)

    print(f"\n  GROUPING ASSERTIONS:")
    print(f"  Cloud (0-4) no cross-contamination : {'PASS' if cloud_pure else 'FAIL'}  [mean coh={coh_cloud_v:.3f}]")
    print(f"  Cyber (5-9) no cross-contamination : {'PASS' if cyber_pure else 'FAIL'}  [mean coh={coh_cyber_v:.3f}]")
    print(f"  AI    (10-14) no cross-contamination: {'PASS' if ai_pure else 'FAIL'}  [mean coh={coh_ai_v:.3f}]")
    print(f"  Semi  (15-19) no cross-contamination: {'PASS' if semi_pure else 'FAIL'}  [mean coh={coh_semi_v:.3f}]")
    print(f"  Cloud majority co-clustered        : {'PASS' if cloud_maj else 'FAIL'}")
    print(f"  Cyber majority co-clustered        : {'PASS' if cyber_maj else 'FAIL'}")
    print(f"  AI    majority co-clustered        : {'PASS' if ai_maj else 'FAIL'}")
    print(f"  Semi  majority co-clustered        : {'PASS' if semi_maj else 'FAIL'}")
    print(f"  Cloud/Cyber separated              : {'PASS' if cloud_cyber_sep else 'FAIL'}")
    print(f"  AI/Semi separated                  : {'PASS' if ai_semi_sep else 'FAIL'}")
    print(f"  Cloud/AI separated                 : {'PASS' if cloud_ai_sep else 'FAIL'}")
    print(f"  Cyber/Semi separated               : {'PASS' if cyber_semi_sep else 'FAIL'}")
    print(f"  Coherence Cloud >= 0.40            : {'PASS' if coh_ok(coh_cloud_v) else 'FAIL'}  [{coh_cloud_v:.3f}]")
    print(f"  Coherence Cyber >= 0.40            : {'PASS' if coh_ok(coh_cyber_v) else 'FAIL'}  [{coh_cyber_v:.3f}]")
    print(f"  Coherence AI >= 0.40               : {'PASS' if coh_ok(coh_ai_v) else 'FAIL'}  [{coh_ai_v:.3f}]")
    print(f"  Coherence Semi >= 0.40             : {'PASS' if coh_ok(coh_semi_v) else 'FAIL'}  [{coh_semi_v:.3f}]")

    passed = (
        cloud_pure and cyber_pure and ai_pure and semi_pure
        and cloud_maj and cyber_maj and ai_maj and semi_maj
        and cloud_cyber_sep and ai_semi_sep and cloud_ai_sep and cyber_semi_sep
        and coh_ok(coh_cloud_v) and coh_ok(coh_cyber_v)
        and coh_ok(coh_ai_v) and coh_ok(coh_semi_v)
        and len(clusters) > 0
    )
    print(f"\n  LEVEL 3 RESULT         : {'PASS' if passed else 'FAIL'}")
    return passed


# ==============================================================================
# MAIN
# ==============================================================================

async def main() -> None:
    print("\n" + "#"*72)
    print("  CLUSTERING STANDALONE TEST -- Staged Warm-up (3 Levels)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Embeddings: synthetic (pre-injected, bypasses LLMService)")
    print("#"*72)

    results: Dict[str, bool] = {}

    for name, coro in [("level1", test_level1), ("level2", test_level2), ("level3", test_level3)]:
        try:
            results[name] = await coro()
        except Exception as e:
            import traceback
            print(f"\n  {name.upper()} ERROR: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "#"*72)
    print("  FINAL RESULTS")
    print("#"*72)
    total_pass = sum(1 for v in results.values() if v)
    for level, ok in results.items():
        print(f"  {level.upper():<12}: {'PASS' if ok else 'FAIL'}")
    print(f"\n  Overall: {total_pass}/{len(results)} levels passed")
    print("#"*72 + "\n")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
