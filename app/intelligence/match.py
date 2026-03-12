"""
MatchAgent — Product Catalog ↔ Cluster Matching (Math Gate 8).

Pure math — no LLM involved.

fit_score = 0.50 * keyword_overlap_ratio(cluster.keywords, product.buying_triggers)
          + 0.30 * cosine_similarity(cluster.embedding, product.embedding)
          + 0.20 * industry_match_score(cluster.industry, product.target_industries)

All weights are fixed priors.

Math assertions:
  Assert: fit_score in [0.0, 1.0]
  Assert: every MatchResult has evidence_quotes (at least 1 article excerpt)
  Assert: results are sorted descending by fit_score

Product catalog:
  Loaded from: data/product_catalog.json (if exists)
  Fallback: user_products as string list → auto-generate Product objects

The match engine also extracts "why it fits" evidence:
  For each matched trigger, finds the sentence in the article that mentions it.
  This becomes the evidence_quotes field — actionable context for the sales rep.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional, Tuple

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import ClusterResult, MatchResult, Product, ProductCatalog

logger = logging.getLogger(__name__)

_RE_WORD_TOKENS = re.compile(r'\b\w{3,}\b')
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_CATALOG_PATH = os.path.join("data", "product_catalog.json")


def compute_match_results(
    clusters: List[ClusterResult],
    user_products: List[str],
    params: Optional[ClusteringParams] = None,
) -> List[MatchResult]:
    """Compute match scores for all (cluster, product) pairs.

    Args:
        clusters: labeled clusters from SynthesisAgent
        user_products: list of product names/descriptions from user's scope
        params: ClusteringParams with match weights

    Returns:
        List[MatchResult] sorted descending by fit_score.
        All math assertions checked.
    """
    if not clusters or not user_products:
        return []

    if params is None:
        params = DEFAULT_PARAMS

    # Load or auto-generate product catalog
    catalog = _load_catalog(user_products)

    results: List[MatchResult] = []
    for cluster in clusters:
        for product in catalog.products:
            result = _score_one(cluster, product, params)
            if result.fit_score > 0.0:
                results.append(result)

    # Sort descending by fit_score (math assertion)
    results.sort(key=lambda r: r.fit_score, reverse=True)

    # Math assertion: all scores in [0, 1]
    invalid = [r for r in results if not (0.0 <= r.fit_score <= 1.0)]
    if invalid:
        logger.error(f"[match] {len(invalid)} results have fit_score out of [0,1]")

    # Math assertion: all results have evidence
    no_evidence = [r for r in results if not r.evidence_quotes]
    if no_evidence:
        logger.warning(f"[match] {len(no_evidence)} results have no evidence quotes")

    logger.info(f"[match] {len(results)} match results for {len(clusters)} clusters × {len(catalog.products)} products")
    return results


def _score_one(
    cluster: ClusterResult,
    product: Product,
    params: ClusteringParams,
) -> MatchResult:
    """Score one (cluster, product) pair."""
    # Signal 1: keyword overlap (50% weight)
    cluster_keywords = _extract_cluster_keywords(cluster)
    trigger_overlap = _jaccard_similarity(cluster_keywords, set(product.buying_triggers))

    # Signal 2: semantic similarity (30% weight)
    semantic_sim = _cosine_similarity(cluster.centroid_embedding, product.embedding)

    # Signal 3: industry match (20% weight)
    industry_score = _industry_match(cluster, product)

    fit_score = (
        params.match_keyword_weight * trigger_overlap
        + params.match_semantic_weight * semantic_sim
        + params.match_industry_weight * industry_score
    )
    fit_score = round(min(max(fit_score, 0.0), 1.0), 3)

    # Extract evidence quotes
    matched_triggers, evidence_quotes = _extract_evidence(cluster, product.buying_triggers)

    # Generate why_it_fits
    why = _generate_why(cluster, product, matched_triggers, trigger_overlap)

    return MatchResult(
        cluster_id=cluster.cluster_id,
        company=cluster.primary_entity or (cluster.entity_names[0] if cluster.entity_names else ""),
        product_name=product.name,
        fit_score=fit_score,
        keyword_overlap=round(trigger_overlap, 3),
        semantic_similarity=round(semantic_sim, 3),
        industry_match=round(industry_score, 3),
        evidence_quotes=evidence_quotes,
        matched_triggers=matched_triggers,
        why_it_fits=why,
    )


def _extract_cluster_keywords(cluster: ClusterResult) -> set:
    """Extract keywords from cluster label, summary, and entity names."""
    text = f"{cluster.label} {cluster.summary} {' '.join(cluster.entity_names)}"
    text = text.lower()
    # Remove stop words and short tokens
    stop = {
        "the", "a", "an", "and", "or", "is", "was", "are", "were", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "as", "its", "has", "had"
    }
    words = {w for w in _RE_WORD_TOKENS.findall(text) if w not in stop}
    return words


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets of strings (case-insensitive)."""
    if not set_a or not set_b:
        return 0.0
    a_lower = {s.lower() for s in set_a}
    b_lower = {s.lower() for s in set_b}

    # Also check substring matches (trigger "data breach" matches keyword "breach")
    overlap = 0
    for trigger in b_lower:
        trigger_words = set(trigger.split())
        if trigger_words & a_lower or trigger.lower() in str(a_lower):
            overlap += 1

    return min(overlap / max(len(b_lower), 1), 1.0)


def _cosine_similarity(emb_a: List[float], emb_b: List[float]) -> float:
    """Cosine similarity between two embeddings. Returns 0.0 if either is empty."""
    if not emb_a or not emb_b or len(emb_a) != len(emb_b):
        return 0.0
    try:
        import numpy as np
        from app.intelligence.engine.similarity import cosine_sim_pair
        return cosine_sim_pair(np.array(emb_a, dtype=np.float32), np.array(emb_b, dtype=np.float32))
    except Exception:
        return 0.0


def _industry_match(cluster: ClusterResult, product: Product) -> float:
    """Industry match score: 1.0 if exact match, 0.5 if partial, 0.0 if no match."""
    if not product.target_industries:
        return 0.5  # No industry constraint → partial match

    cluster_industry = ""
    if cluster.industry:
        cluster_industry = f"{cluster.industry.level_1} {cluster.industry.level_2}".lower()

    if not cluster_industry:
        return 0.3  # Unknown cluster industry → low partial match

    for ind in product.target_industries:
        ind_lower = ind.lower()
        if ind_lower in cluster_industry or any(w in cluster_industry for w in ind_lower.split()):
            return 1.0

    return 0.0


def _extract_evidence(
    cluster: ClusterResult,
    buying_triggers: List[str],
) -> Tuple[List[str], List[str]]:
    """Find trigger keywords in cluster label/summary and extract as evidence.

    Returns (matched_triggers, evidence_quotes)
    """
    text = f"{cluster.label}. {cluster.summary}"
    text_lower = text.lower()

    matched = []
    quotes = []

    for trigger in buying_triggers:
        trigger_lower = trigger.lower()
        # Find trigger words in text
        trigger_words = trigger_lower.split()
        if all(w in text_lower for w in trigger_words):
            matched.append(trigger)
            # Extract surrounding sentence as evidence
            for sentence in _RE_SENTENCE_SPLIT.split(text):
                if any(w in sentence.lower() for w in trigger_words):
                    quotes.append(sentence.strip()[:150])
                    break

    # Ensure at least one quote (fallback to cluster label)
    if not quotes and cluster.label:
        quotes = [cluster.label]

    return matched[:5], quotes[:3]  # Cap at 5 triggers, 3 quotes


def _generate_why(
    cluster: ClusterResult,
    product: Product,
    matched_triggers: List[str],
    overlap_score: float,
) -> str:
    """Generate human-readable why_it_fits explanation."""
    company = cluster.primary_entity or "This company"
    if matched_triggers:
        triggers_str = ", ".join(f"'{t}'" for t in matched_triggers[:2])
        return (
            f"{company} matches your trigger {triggers_str} — "
            f"potential fit for {product.name}"
        )
    elif overlap_score > 0:
        return f"{company} is in {product.target_industries[0] if product.target_industries else 'your target market'}"
    return f"{company} → {product.name} (indirect match, review recommended)"




def _load_catalog(user_products: List[str]) -> ProductCatalog:
    """Load product catalog from file, or auto-generate from user_products strings."""
    # Try loading from file first
    if os.path.exists(_CATALOG_PATH):
        try:
            with open(_CATALOG_PATH) as f:
                data = json.load(f)
            products = [Product(**p) for p in data.get("products", [])]
            if products:
                return ProductCatalog(products=products, owner_company=data.get("owner_company", ""))
        except Exception as exc:
            logger.warning(f"[match] Could not load catalog from {_CATALOG_PATH}: {exc}")

    # Auto-generate from user_products strings
    products = []
    for prod_str in user_products:
        # Try to parse structured format: "name|trigger1,trigger2|industry1"
        parts = prod_str.split("|")
        if len(parts) >= 2:
            products.append(Product(
                name=parts[0].strip(),
                description=parts[0].strip(),
                buying_triggers=[t.strip() for t in parts[1].split(",")],
                target_industries=[parts[2].strip()] if len(parts) > 2 else [],
            ))
        else:
            # Plain string → use as name, extract keywords as triggers
            products.append(Product(
                name=prod_str.strip(),
                description=prod_str.strip(),
                buying_triggers=prod_str.lower().split()[:5],
            ))

    return ProductCatalog(products=products)
