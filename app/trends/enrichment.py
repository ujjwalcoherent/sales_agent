"""
Post-clustering enrichment layer.

Runs AFTER Leiden clustering, BEFORE LLM synthesis. Deterministic, fast, no LLM.
Extracts structured intelligence from each cluster's articles:

1. Entity consolidation — merge entity lists across cluster members
2. Company activity scoring — mention frequency × recency × role weight
3. Subject vs mention classification — title presence + frequency
4. Cluster validation — entity coherence, source diversity, temporal spread

Architecture pattern from JRC/EMM (Piskorski & Tanev, 2008):
  "clustered news are heavily exploited at all stages of processing,
   with focus on cluster-level information fusion."

REF: data/research_enrichment_layer.md (full research findings)
"""

import asyncio
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Recency decay ────────────────────────────────────────────────────────
# Exponential decay with 5-day half-life (from Autobound research:
# "an intent signal from today is worth 10x more than one from a month ago")

def recency_weight(days_since: float, half_life: float = 5.0) -> float:
    """Exponential decay weight. Day 0=1.0, Day 5=0.5, Day 10=0.25."""
    if days_since < 0:
        days_since = 0
    return 2.0 ** (-days_since / half_life)


# ── Entity consolidation ────────────────────────────────────────────────

def consolidate_entities(
    articles: List[Any],
) -> Dict[str, Dict]:
    """Merge entity lists from all cluster articles into deduplicated,
    frequency-weighted canonical entities.

    Pattern from GDELT Global Entity Graph: identical entity+type combinations
    are aggregated with mention counts and average salience.

    Returns:
        Dict mapping normalized entity name → {
            "type": str,           # ORG, PERSON, GPE, etc.
            "mention_count": int,  # Total mentions across all articles
            "article_count": int,  # How many articles mention this entity
            "in_titles": int,      # How many article titles contain this entity
            "avg_salience": float, # Average salience across mentions
            "max_salience": float, # Peak salience in any article
        }
    """
    from app.news.entity_normalizer import normalize_entity

    entity_data: Dict[str, Dict] = {}
    seen_per_article: Dict[str, Set[str]] = {}  # article_id → set of normalized names

    for article in articles:
        article_id = str(getattr(article, 'id', id(article)))
        seen_per_article[article_id] = set()
        title_lower = (article.title or '').lower()

        for ent in getattr(article, 'entities', []) or []:
            raw_name = getattr(ent, 'text', str(ent))
            ent_type = getattr(ent, 'type', 'UNKNOWN')
            salience = getattr(ent, 'salience', 0.0)

            # Normalize: "Tata Motors Ltd." → "Tata Motors"
            name = normalize_entity(raw_name)
            if not name or len(name) < 2:
                continue

            if name not in entity_data:
                entity_data[name] = {
                    "type": ent_type,
                    "mention_count": 0,
                    "article_count": 0,
                    "in_titles": 0,
                    "salience_sum": 0.0,
                    "max_salience": 0.0,
                }

            entity_data[name]["mention_count"] += 1
            entity_data[name]["salience_sum"] += salience
            entity_data[name]["max_salience"] = max(
                entity_data[name]["max_salience"], salience
            )

            # Count unique articles (not double-count same entity in same article)
            if name not in seen_per_article[article_id]:
                seen_per_article[article_id].add(name)
                entity_data[name]["article_count"] += 1

                # Check title presence
                if name.lower() in title_lower:
                    entity_data[name]["in_titles"] += 1

    # Compute average salience
    for name, data in entity_data.items():
        count = data["mention_count"]
        data["avg_salience"] = round(data["salience_sum"] / max(count, 1), 3)
        del data["salience_sum"]  # Clean up intermediate field

    return entity_data


# ── Company activity scoring ─────────────────────────────────────────────

def score_company_activity(
    entities: Dict[str, Dict],
    articles: List[Any],
    activity_window_days: float = 5.0,
) -> List[Dict]:
    """Score ORG entities for lead potential using activity signals.

    Combines:
    - mention_frequency: How many articles mention this company
    - recency: Exponential decay from most recent mention
    - role: Subject (2x) vs mentioned (1x) vs peripheral (0.3x)
    - salience: How central the company is to the articles

    Returns list of scored companies, sorted by activity_score descending.
    """
    # Use cluster's latest article date as reference, not NOW.
    # This makes scores stable across reruns — a cluster from 3 days ago
    # shouldn't get penalized more just because we re-run enrichment later.
    cluster_end_date = None
    for article in articles:
        pub = getattr(article, 'published_at', None)
        if pub:
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if cluster_end_date is None or pub > cluster_end_date:
                cluster_end_date = pub
    now = cluster_end_date or datetime.now(timezone.utc)
    total_articles = len(articles)
    scored: List[Dict] = []

    for name, data in entities.items():
        # Only score organizations (companies, institutions)
        if data["type"] not in ("ORG", "ORGANIZATION"):
            continue

        article_count = data["article_count"]
        in_titles = data["in_titles"]

        # ── Role classification (SENTiVENT pattern) ──
        title_ratio = in_titles / max(total_articles, 1)
        mention_ratio = article_count / max(total_articles, 1)

        if title_ratio >= 0.4 or mention_ratio >= 0.7:
            role = "subject"
            role_weight = 2.0
        elif mention_ratio >= 0.2:
            role = "mentioned"
            role_weight = 1.0
        else:
            role = "peripheral"
            role_weight = 0.3

        # ── Recency score ──
        # Find the most recent article mentioning this entity
        best_recency = 0.0
        for article in articles:
            ent_names_lower = {
                n.lower() for n in (getattr(article, 'entity_names', []) or [])
            }
            if name.lower() in ent_names_lower or any(
                name.lower() in n.lower() for n in ent_names_lower
            ):
                pub = getattr(article, 'published_at', None)
                if pub:
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                    days = (now - pub).total_seconds() / 86400
                    best_recency = max(best_recency, recency_weight(days))

        # ── Composite activity score ──
        activity_score = (
            article_count * role_weight * best_recency
            * (1.0 + data["avg_salience"])  # Salience bonus
        )

        scored.append({
            "name": name,
            "type": data["type"],
            "article_count": article_count,
            "in_titles": in_titles,
            "mention_count": data["mention_count"],
            "avg_salience": data["avg_salience"],
            "max_salience": data["max_salience"],
            "role": role,
            "role_weight": role_weight,
            "recency_score": round(best_recency, 3),
            "activity_score": round(activity_score, 3),
        })

    # Sort by activity score, highest first
    scored.sort(key=lambda c: c["activity_score"], reverse=True)
    return scored


# ── Cluster validation ───────────────────────────────────────────────────

# Generic entities that appear in most articles from the same domain.
# These should NOT count toward entity coherence — they don't discriminate.
_GENERIC_ENTITIES = frozenset({
    "india", "indian", "us", "china", "uk", "eu", "new delhi", "delhi",
    "mumbai", "bangalore", "bengaluru", "government", "ministry",
    "artificial intelligence", "ai", "market", "markets", "digital",
    "technology", "business", "economy", "global", "world",
    "million", "billion", "crore", "lakh", "percent", "%",
    "fy25", "fy26", "fy2025", "fy2026", "q1", "q2", "q3", "q4",
})


def validate_cluster(
    articles: List[Any],
    entities: Dict[str, Dict],
    min_entity_overlap: float = 0.15,
    min_sources: int = 2,
    max_temporal_span_days: float = 14.0,
) -> Dict[str, Any]:
    """Validate cluster quality using 5 signals.

    Signals:
    1. Entity coherence — fraction of articles sharing ≥1 SPECIFIC top entity
    2. Source diversity — number of unique sources
    3. Temporal span — time range of articles
    4. Event-type concentration — dominant event type must be >50%
    5. Title vocabulary coherence — pairwise title word overlap

    A cluster is invalid if it fails ANY hard check (entity coherence, event
    concentration) or 2+ soft checks.
    """
    rejection_reasons = []
    n = len(articles)

    # ── Source diversity ──
    sources = set()
    for a in articles:
        src = getattr(a, 'source_id', '') or getattr(a, 'source_name', '')
        if src:
            sources.add(src)
    source_count = len(sources)
    # Source diversity is a soft check — single-source clusters with 5+ articles
    # are likely echo-chamber noise, but 3-4 article clusters from one source
    # can be valid niche events
    if source_count < min_sources and n >= 5:
        rejection_reasons.append(
            f"low_source_diversity ({source_count} < {min_sources})"
        )

    # ── Temporal span ──
    dates = []
    for a in articles:
        pub = getattr(a, 'published_at', None)
        if pub:
            dates.append(pub)
    if len(dates) >= 2:
        span = (max(dates) - min(dates)).total_seconds() / 86400
    else:
        span = 0.0
    if span > max_temporal_span_days:
        rejection_reasons.append(
            f"temporal_span_too_wide ({span:.1f} days > {max_temporal_span_days})"
        )

    # ── Entity coherence (excluding generic entities) ──
    # Top entities: those appearing in ≥ 30% of articles, not generic
    top_entities = {
        name for name, data in entities.items()
        if data["article_count"] >= max(2, n * 0.3)
        and name.lower() not in _GENERIC_ENTITIES
    }

    if not top_entities:
        entity_coherence = 0.0
    else:
        articles_with_top_entity = 0
        for a in articles:
            a_ents = {
                en.lower() for en in (getattr(a, 'entity_names', []) or [])
            }
            if any(te.lower() in a_ents or any(
                te.lower() in e for e in a_ents
            ) for te in top_entities):
                articles_with_top_entity += 1
        entity_coherence = articles_with_top_entity / max(n, 1)

    # ── Event-type concentration ──
    event_types = Counter()
    for a in articles:
        etype = getattr(a, '_trigger_event', 'general')
        event_types[etype] += 1
    dominant_event, dominant_count = event_types.most_common(1)[0] if event_types else ("general", 0)
    event_concentration = dominant_count / max(n, 1)
    n_event_types = len(event_types)

    # Entity coherence check — but high event concentration compensates.
    # Funding/acquisition clusters inherently mention different companies,
    # so 80%+ event concentration is a strong quality signal on its own.
    event_compensates = (
        event_concentration >= 0.80
        and dominant_event not in ("general", "unknown", "")
    )
    if entity_coherence < min_entity_overlap and not event_compensates:
        rejection_reasons.append(
            f"low_entity_coherence ({entity_coherence:.2f} < {min_entity_overlap})"
        )

    # Clusters with 4+ different event types and no dominant type are grab-bags
    if n_event_types >= 4 and event_concentration < 0.50:
        rejection_reasons.append(
            f"event_type_scatter ({n_event_types} types, "
            f"dominant={dominant_event} at {event_concentration:.0%})"
        )

    # ── Title vocabulary coherence ──
    # Extract stemmed title words, compute pairwise overlap
    _title_stop = frozenset({
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "are", "was", "were", "be", "been", "has", "have", "had",
        "with", "from", "by", "its", "it", "this", "that", "how", "what",
        "why", "who", "will", "can", "may", "could", "would", "should",
        "not", "no", "but", "if", "as", "up", "out", "about", "after",
        "new", "more", "most", "top", "big", "all", "over", "into",
        "says", "said", "set", "get", "gets", "here", "now", "also",
        "amid", "check", "know", "things", "need", "five", "three",
        "india", "indian", "market", "business", "economy",
    })
    title_vocabs = []
    for a in articles:
        words = set()
        for w in (a.title or '').lower().split():
            w = w.strip(".,;:!?'\"()-[]{}#@$%&*/\\|<>~`+=^_")
            if w and len(w) > 2 and w not in _title_stop:
                words.add(w)
        title_vocabs.append(words)

    # Pairwise Jaccard of title words
    if n >= 2:
        jaccard_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if title_vocabs[i] and title_vocabs[j]:
                    intersection = len(title_vocabs[i] & title_vocabs[j])
                    union = len(title_vocabs[i] | title_vocabs[j])
                    jaccard_sum += intersection / union if union > 0 else 0
                pair_count += 1
        avg_title_jaccard = jaccard_sum / max(pair_count, 1)
    else:
        avg_title_jaccard = 1.0

    return {
        "is_valid": len(rejection_reasons) == 0,
        "entity_coherence": round(entity_coherence, 3),
        "source_diversity": source_count,
        "temporal_span_days": round(span, 1),
        "top_entities": sorted(top_entities)[:10],
        "event_concentration": round(event_concentration, 3),
        "dominant_event": dominant_event,
        "n_event_types": n_event_types,
        "avg_title_jaccard": round(avg_title_jaccard, 4),
        "rejection_reasons": rejection_reasons,
    }


# ── LLM Cluster Validation (Layer 2.75) ──────────────────────────────
# Curriculum learning cascade:
#   Tier 1 (deterministic): Auto-approve high-confidence, auto-reject garbage
#   Tier 2 (LLM): Validate borderline clusters that passed Tier 1
# Cost: ~$0.0003/cluster with GPT-4o-mini / DeepSeek. ~$0.01/pipeline run.
# REF: NewsCatcher rejects 80% via LLM validation for precision.

_CLUSTER_VALIDATION_SYSTEM = (
    "You are a senior news editor validating article clusters for a business "
    "intelligence platform. Your job is to determine if a group of articles "
    "genuinely describes the SAME real-world event, development, or closely "
    "related chain of events. Be strict: a cluster about 'Indian tech startups' "
    "that mixes AI funding with EV launches is NOT coherent. A cluster about "
    "'Trump tariff impacts on Indian markets' where all articles discuss market "
    "reactions to the same tariff ruling IS coherent even if they mention "
    "different stocks. Always respond with valid JSON."
)

_CLUSTER_VALIDATION_PROMPT = """Evaluate whether these {n} articles form a coherent cluster about a single event or closely related developments.

ARTICLES:
{articles_text}

CLUSTER METADATA:
- Dominant event type: {dominant_event} ({event_pct}% of articles)
- Number of unique sources: {n_sources}
- Entity coherence (deterministic): {entity_coherence:.2f}
- Top shared entities: {top_entities}

INSTRUCTIONS:
1. Read all article titles and summaries carefully
2. Determine if they discuss the SAME real-world event or tightly related developments
3. Identify any outlier articles that clearly don't belong
4. If the cluster contains 2+ distinct sub-topics, flag should_split=True
5. Assign a coherence_score: 0.0=random grab-bag, 0.5=loosely related, 0.8=same topic, 1.0=same event
6. Suggest a concise label for the cluster

Respond with JSON containing: reasoning, is_coherent, coherence_score, suggested_label, outlier_indices, should_split, split_reason"""


async def validate_cluster_llm(
    cluster_id: int,
    articles: list,
    enrichment: Dict[str, Any],
    llm_service=None,
) -> Dict[str, Any]:
    """LLM-based cluster validation (Tier 2 in curriculum cascade).

    Only called for clusters that PASSED deterministic validation but need
    higher-confidence verification. Returns validation dict compatible with
    the enrichment validation format.

    Args:
        cluster_id: Cluster identifier
        articles: List of NewsArticle objects in the cluster
        enrichment: Enrichment dict from enrich_cluster()
        llm_service: LLMService instance (or created if None)

    Returns:
        Dict with: is_coherent, coherence_score, suggested_label,
        outlier_indices, should_split, reasoning
    """
    if llm_service is None:
        from app.tools.llm_service import LLMService
        llm_service = LLMService(lite=True)  # Use cheaper model

    # Build article text for prompt (title + first 150 chars of summary)
    articles_text_parts = []
    for i, a in enumerate(articles):
        title = getattr(a, 'title', '') or ''
        summary = getattr(a, 'summary', '') or getattr(a, 'content', '') or ''
        summary_short = summary[:200].strip()
        source = getattr(a, 'source_name', '') or getattr(a, 'source_id', '') or ''
        articles_text_parts.append(
            f"[{i}] ({source}) {title}\n    {summary_short}"
        )
    articles_text = "\n".join(articles_text_parts)

    # Extract metadata from enrichment
    validation = enrichment.get("validation", {})
    dominant_event = validation.get("dominant_event", "unknown")
    event_concentration = validation.get("event_concentration", 0.0)
    entity_coherence = validation.get("entity_coherence", 0.0)
    top_entities = validation.get("top_entities", [])[:5]
    source_diversity = validation.get("source_diversity", 0)

    prompt = _CLUSTER_VALIDATION_PROMPT.format(
        n=len(articles),
        articles_text=articles_text,
        dominant_event=dominant_event,
        event_pct=round(event_concentration * 100),
        n_sources=source_diversity,
        entity_coherence=entity_coherence,
        top_entities=", ".join(top_entities) if top_entities else "(none)",
    )

    try:
        from app.schemas.llm_outputs import ClusterValidationLLM

        # Track A: typed structured output
        result = await llm_service.run_structured(
            prompt=prompt,
            system_prompt=_CLUSTER_VALIDATION_SYSTEM,
            output_type=ClusterValidationLLM,
            temperature=0.1,  # Low temperature for consistent validation
        )

        llm_result = {
            "is_coherent": result.is_coherent,
            "coherence_score": result.coherence_score,
            "suggested_label": result.suggested_label,
            "outlier_indices": result.outlier_indices,
            "should_split": result.should_split,
            "split_reason": result.split_reason,
            "reasoning": result.reasoning,
            "method": "run_structured",
        }

    except Exception as e1:
        logger.debug(f"Cluster {cluster_id} LLM validation run_structured failed: {e1}")
        # Track B: generate_json fallback
        try:
            raw = await llm_service.generate_json(
                prompt=prompt,
                system_prompt=_CLUSTER_VALIDATION_SYSTEM,
            )
            llm_result = {
                "is_coherent": bool(raw.get("is_coherent", True)),
                "coherence_score": float(raw.get("coherence_score", 0.5)),
                "suggested_label": str(raw.get("suggested_label", "")),
                "outlier_indices": list(raw.get("outlier_indices", [])),
                "should_split": bool(raw.get("should_split", False)),
                "split_reason": str(raw.get("split_reason", "")),
                "reasoning": str(raw.get("reasoning", "")),
                "method": "generate_json",
            }
        except Exception as e2:
            logger.warning(
                f"Cluster {cluster_id} LLM validation failed (both tracks): {e2}"
            )
            # Return neutral result — don't reject on LLM failure
            llm_result = {
                "is_coherent": True,
                "coherence_score": 0.5,
                "suggested_label": "",
                "outlier_indices": [],
                "should_split": False,
                "split_reason": "",
                "reasoning": f"LLM validation failed: {e2}",
                "method": "fallback",
            }

    logger.info(
        f"Cluster {cluster_id} LLM validation: "
        f"coherent={llm_result['is_coherent']}, "
        f"score={llm_result['coherence_score']:.2f}, "
        f"label='{llm_result['suggested_label']}', "
        f"outliers={llm_result['outlier_indices']}, "
        f"split={llm_result['should_split']}, "
        f"method={llm_result['method']}"
    )

    return llm_result


async def validate_all_clusters_llm(
    enrichments: List[Dict],
    cluster_articles: Dict[int, list],
    llm_service=None,
    min_deterministic_score: float = 0.0,
    max_deterministic_score: float = 1.0,
) -> Dict[int, Dict]:
    """Run LLM validation on all clusters that passed deterministic checks.

    Curriculum learning cascade:
    - Clusters that failed deterministic validation → already rejected (skip)
    - Clusters with very high deterministic confidence → auto-approved (skip)
    - Borderline clusters → LLM validation (this function)

    For efficiency, runs all LLM calls concurrently with a semaphore.

    Returns:
        Dict mapping cluster_id → LLM validation result
    """
    if llm_service is None:
        from app.tools.llm_service import LLMService
        llm_service = LLMService(lite=True)

    # Select clusters for LLM validation
    candidates = []
    for enrichment in enrichments:
        if not enrichment["validation"]["is_valid"]:
            continue  # Already rejected by deterministic checks
        cid = enrichment["cluster_id"]
        articles = cluster_articles.get(cid, [])
        if len(articles) < 2:
            continue  # Single-article clusters don't need validation
        candidates.append((cid, articles, enrichment))

    if not candidates:
        return {}

    logger.info(
        f"LLM cluster validation: {len(candidates)} candidates "
        f"(out of {len(enrichments)} total clusters)"
    )

    # Run validations concurrently (semaphore limits parallel LLM calls)
    sem = asyncio.Semaphore(3)  # Max 3 concurrent LLM calls

    async def _validate_one(cid, arts, enr):
        async with sem:
            return cid, await validate_cluster_llm(cid, arts, enr, llm_service)

    tasks = [_validate_one(cid, arts, enr) for cid, arts, enr in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    llm_validations = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"LLM validation task failed: {r}")
            continue
        cid, validation = r
        llm_validations[cid] = validation

    # Summary
    rejected = sum(1 for v in llm_validations.values() if not v["is_coherent"])
    split_candidates = sum(1 for v in llm_validations.values() if v["should_split"])
    avg_score = (
        sum(v["coherence_score"] for v in llm_validations.values()) / len(llm_validations)
        if llm_validations else 0.0
    )
    logger.info(
        f"LLM validation complete: {len(llm_validations)} validated, "
        f"{rejected} rejected, {split_candidates} split candidates, "
        f"avg_score={avg_score:.2f}"
    )

    return llm_validations


# ── Cross-cluster entity linking ─────────────────────────────────────────

def link_entities_across_clusters(
    cluster_enrichments: List[Dict],
) -> Dict[str, Dict]:
    """Track ORG entities across all clusters. Companies appearing in
    multiple clusters are stronger lead signals (multi-trend involvement).

    Args:
        cluster_enrichments: List of enrichment dicts from enrich_cluster().

    Returns:
        Dict mapping company name → {
            "cluster_ids": list,
            "total_articles": int,
            "total_mentions": int,
            "cross_cluster_boost": float,  # 1.0 for single, 1.5+ for multi
        }
    """
    company_clusters: Dict[str, Dict] = {}

    for enrichment in cluster_enrichments:
        cluster_id = enrichment.get("cluster_id", -1)
        for company in enrichment.get("companies", []):
            name = company["name"]
            if name not in company_clusters:
                company_clusters[name] = {
                    "cluster_ids": [],
                    "total_articles": 0,
                    "total_mentions": 0,
                }
            company_clusters[name]["cluster_ids"].append(cluster_id)
            company_clusters[name]["total_articles"] += company["article_count"]
            company_clusters[name]["total_mentions"] += company["mention_count"]

    # Compute cross-cluster boost
    for name, data in company_clusters.items():
        n_clusters = len(set(data["cluster_ids"]))
        # Multi-cluster companies get a 50% boost per additional cluster
        data["cross_cluster_boost"] = 1.0 + 0.5 * max(0, n_clusters - 1)
        data["n_clusters"] = n_clusters

    return company_clusters


# ── Main enrichment function ─────────────────────────────────────────────

def enrich_cluster(
    cluster_id: int,
    articles: List[Any],
    activity_window_days: float = 5.0,
) -> Dict[str, Any]:
    """Full post-clustering enrichment for a single cluster.

    Called after Leiden clustering, before LLM synthesis. Produces structured
    intelligence that feeds into the synthesis prompt as context, replacing
    the current approach where the LLM must extract entities from raw text.

    Returns dict with:
    - cluster_id, article_count
    - entities: consolidated entity dict
    - companies: scored company list (sorted by activity_score)
    - validation: cluster quality signals
    - primary_companies: companies classified as "subject" role
    - event_types: Counter of event types in cluster
    """
    entities = consolidate_entities(articles)
    companies = score_company_activity(
        entities, articles, activity_window_days=activity_window_days
    )
    validation = validate_cluster(articles, entities)

    # Event type distribution
    event_types = Counter()
    for a in articles:
        etype = getattr(a, '_trigger_event', 'general')
        event_types[etype] += 1

    # Separate primary vs peripheral companies
    primary = [c for c in companies if c["role"] == "subject"]
    mentioned = [c for c in companies if c["role"] == "mentioned"]
    peripheral = [c for c in companies if c["role"] == "peripheral"]

    result = {
        "cluster_id": cluster_id,
        "article_count": len(articles),
        "entities": entities,
        "companies": companies,
        "validation": validation,
        "primary_companies": primary,
        "mentioned_companies": mentioned,
        "peripheral_companies": peripheral,
        "event_types": dict(event_types),
        "dominant_event": event_types.most_common(1)[0] if event_types else ("general", 0),
    }

    # Log summary
    logger.info(
        f"Cluster {cluster_id} enrichment: {len(articles)} articles, "
        f"{len(entities)} entities, {len(companies)} ORG companies "
        f"({len(primary)} subject, {len(mentioned)} mentioned), "
        f"valid={validation['is_valid']}"
    )

    return result


def enrich_all_clusters(
    labels,
    articles: List[Any],
    activity_window_days: float = 5.0,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Enrich all clusters from a Leiden clustering run.

    Args:
        labels: numpy array of cluster labels (-1 = noise)
        articles: full article list (same order as labels)
        activity_window_days: recency window

    Returns:
        (cluster_enrichments, cross_cluster_companies)
    """
    import numpy as np

    unique_labels = sorted(set(labels))
    cluster_enrichments = []

    for cl_id in unique_labels:
        if cl_id == -1:
            continue  # Skip noise
        members = [i for i, l in enumerate(labels) if l == cl_id]
        cluster_articles = [articles[i] for i in members]
        enrichment = enrich_cluster(
            cluster_id=cl_id,
            articles=cluster_articles,
            activity_window_days=activity_window_days,
        )
        cluster_enrichments.append(enrichment)

    # Cross-cluster entity linking
    cross_cluster = link_entities_across_clusters(cluster_enrichments)

    # Log cross-cluster summary
    multi_cluster = {
        name: data for name, data in cross_cluster.items()
        if data["n_clusters"] > 1
    }
    if multi_cluster:
        logger.info(
            f"Cross-cluster: {len(multi_cluster)} companies in multiple clusters: "
            + ", ".join(
                f"{name} ({data['n_clusters']} clusters)"
                for name, data in sorted(
                    multi_cluster.items(),
                    key=lambda x: x[1]["n_clusters"],
                    reverse=True,
                )[:5]
            )
        )

    return cluster_enrichments, cross_cluster
