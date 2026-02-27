"""
Post-clustering coherence validator -- ensures clusters are semantically tight.

Operations in original embedding space (not UMAP-reduced):
  1. Validate coherence (mean pairwise cosine)
  2. Split incoherent clusters (agglomerative)
  3. Merge redundant clusters (centroid similarity)
  4. Reject very low coherence (demote to noise)
  5. Multi-signal outlier detection and vocabulary-based refinement
  6. Quality report with grade (A-F)
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.schemas.news import NewsArticle

logger = logging.getLogger(__name__)


def _simple_stem(w: str) -> str:
    """Lightweight suffix stemmer for vocabulary overlap (not full Porter)."""
    if len(w) <= 3:
        return w
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3:
        return w
    # -ies → -i (companies → compani)
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "i"
    # -ing → strip (trading → trad, jumping → jump)
    if w.endswith("ing") and len(w) > 6:
        return w[:-3]
    # -ment → strip (agreement → agree)
    if w.endswith("ment") and len(w) > 7:
        return w[:-4]
    # -ed → strip (launched → launch, surged → surg)
    if w.endswith("ed") and len(w) > 5:
        return w[:-2]
    # -ers → strip trailing s (traders → trader)
    if w.endswith("ers") and len(w) > 5:
        return w[:-1]
    # -es → strip (reserves → reserv, surges → surg)
    if w.endswith("es") and len(w) > 5:
        return w[:-2]
    # -s → strip (gains → gain, markets → market)
    if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
        return w[:-1]
    # trailing -e → strip (reserve → reserv, trade → trad)
    if w.endswith("e") and len(w) > 4:
        return w[:-1]
    return w


def compute_cluster_quality_report(
    articles: List[NewsArticle],
    embeddings: np.ndarray,
    article_indices: List[int],
    keywords: List[str] = None,
) -> Dict[str, Any]:
    """Generate a quality report for a single cluster.

    Returns dict with coherence_score, entity/keyword consistency,
    temporal_spread_hours, source_diversity, quality_grade (A-F), and reasoning.
    """
    report = {
        "coherence_score": 0.0,
        "entity_consistency": 0.0,
        "keyword_consistency": 0.0,
        "temporal_spread_hours": 0.0,
        "source_diversity": 0.0,
        "quality_grade": "F",
        "quality_reasoning": "",
    }

    if not articles or len(articles) < 2:
        report["coherence_score"] = 1.0
        report["quality_grade"] = "A"
        report["quality_reasoning"] = "Single-article cluster (trivially coherent)."
        return report

    from app.tools.embeddings import mean_pairwise_cosine
    coherence = mean_pairwise_cosine(embeddings[article_indices]) if len(article_indices) >= 2 else 1.0
    report["coherence_score"] = round(coherence, 3)

    # Entity consistency: % of articles sharing top 3 entities
    entity_counts = Counter()
    for a in articles:
        for name in getattr(a, 'entity_names', []):
            entity_counts[name.lower()] += 1
    top_entities = [e for e, _ in entity_counts.most_common(3)]
    if top_entities and len(articles) > 1:
        articles_with_top = 0
        for a in articles:
            a_entities = {n.lower() for n in getattr(a, 'entity_names', [])}
            if any(e in a_entities for e in top_entities):
                articles_with_top += 1
        entity_consistency = articles_with_top / len(articles)
    else:
        entity_consistency = 0.0
    report["entity_consistency"] = round(entity_consistency, 3)

    if keywords:
        top_kw = [k.lower() for k in keywords[:3]]
        articles_with_kw = 0
        for a in articles:
            text = f"{a.title} {a.summary or ''}".lower()
            if any(kw in text for kw in top_kw):
                articles_with_kw += 1
        keyword_consistency = articles_with_kw / len(articles)
    else:
        keyword_consistency = 0.0
    report["keyword_consistency"] = round(keyword_consistency, 3)

    # Temporal spread (std dev of published_at in hours)
    timestamps = []
    for a in articles:
        pub = getattr(a, 'published_at', None) or getattr(a, 'published_date', None)
        if pub:
            if isinstance(pub, datetime):
                timestamps.append(pub.timestamp())
            elif isinstance(pub, str):
                try:
                    timestamps.append(datetime.fromisoformat(pub).timestamp())
                except (ValueError, TypeError):
                    pass

    if len(timestamps) >= 2:
        spread_seconds = float(np.std(timestamps))
        spread_hours = spread_seconds / 3600
    else:
        spread_hours = 0.0
    report["temporal_spread_hours"] = round(spread_hours, 1)

    unique_sources = len({
        getattr(a, 'source_id', '') or getattr(a, 'source_name', '')
        for a in articles
    })
    source_diversity = unique_sources / len(articles)
    report["source_diversity"] = round(source_diversity, 3)

    # Composite grade: sales-actionability focus
    cmi_scores = [
        getattr(a, '_cmi_relevance_score', 0.5) for a in articles
    ]
    cmi_relevance = sum(cmi_scores) / max(len(cmi_scores), 1)

    # OSS placeholder (updated post-synthesis by update_quality_with_oss)
    specificity_score = 0.0

    # Second-order quality: employee ranges + geography + sub-segments in article text
    import re
    from app.shared.geo import SPECIFIC_GEO_PATTERN as _geo_pattern
    _size_pattern = re.compile(r'\d+[-–]\d+\s*(?:employee|staff|worker)', re.IGNORECASE)
    all_text = " ".join(
        (a.title or "") + " " + (a.summary or "")[:200]
        for a in articles
    )
    has_size = bool(_size_pattern.search(all_text))
    has_geo = bool(_geo_pattern.search(all_text))
    _subsegment = re.compile(
        r'\b(?:auto\s*parts?|jeweller|pharma|semiconductor|garment|textile|'
        r'nbfc|microfinance|cold\s*chain|ev\s*battery|solar|defense|mining)\b',
        re.IGNORECASE,
    )
    has_subsegment = bool(_subsegment.search(all_text))
    second_order_quality = (
        (0.4 if has_size else 0.0)
        + (0.3 if has_geo else 0.0)
        + (0.3 if has_subsegment else 0.0)
    )

    composite = (
        coherence * 0.15                           # Keep: basic quality
        + entity_consistency * 0.10                # Keep: cluster integrity
        + cmi_relevance * 0.25                     # Average CMI relevance
        + specificity_score * 0.25                 # OSS (updated post-synthesis)
        + second_order_quality * 0.15              # Size + geography + sub-segment
        + min(1.0, source_diversity * 2) * 0.10    # Multi-source confirmation
    )
    report["cmi_relevance"] = round(cmi_relevance, 3)
    report["second_order_quality"] = round(second_order_quality, 3)
    report["specificity_score"] = specificity_score

    if composite >= 0.70:
        grade = "A"
    elif composite >= 0.55:
        grade = "B"
    elif composite >= 0.40:
        grade = "C"
    elif composite >= 0.25:
        grade = "D"
    else:
        grade = "F"
    report["quality_grade"] = grade

    coherence_label = "strong" if coherence >= 0.5 else "moderate" if coherence >= 0.35 else "weak"
    coherence_detail = "articles share semantic similarity" if coherence >= 0.5 else "articles may cover different sub-topics"
    reasons = [
        f"Coherence is {coherence_label} ({coherence:.2f}) — {coherence_detail}",
    ]

    if top_entities:
        reasons.append(
            f"{int(entity_consistency * 100)}% of articles mention top entities "
            f"({', '.join(top_entities[:3])})"
        )

    reasons.append(
        f"Source diversity: {unique_sources} unique sources across {len(articles)} articles "
        f"({'diverse coverage' if source_diversity >= 0.4 else 'possible echo-chamber republishing'})"
    )

    if spread_hours > 0:
        if spread_hours < 24:
            reasons.append(f"All articles published within ~{spread_hours:.0f} hours (tight temporal window)")
        elif spread_hours < 168:
            reasons.append(f"Articles span ~{spread_hours / 24:.0f} days")
        else:
            reasons.append(f"Articles span ~{spread_hours / 24:.0f} days (wide temporal spread)")

    report["quality_reasoning"] = ". ".join(reasons) + "."
    return report


def update_quality_with_oss(
    report: Dict[str, Any],
    oss: float,
) -> Dict[str, Any]:
    """Recompute quality grade with actual OSS (replaces the 0.0 placeholder)."""
    report["specificity_score"] = round(oss, 4)
    composite = (
        report.get("coherence_score", 0.0) * 0.15
        + report.get("entity_consistency", 0.0) * 0.10
        + report.get("cmi_relevance", 0.5) * 0.25
        + oss * 0.25
        + report.get("second_order_quality", 0.0) * 0.15
        + min(1.0, report.get("source_diversity", 0.0) * 2) * 0.10
    )

    if composite >= 0.70:
        grade = "A"
    elif composite >= 0.55:
        grade = "B"
    elif composite >= 0.40:
        grade = "C"
    elif composite >= 0.25:
        grade = "D"
    else:
        grade = "F"

    report["quality_grade"] = grade
    report["composite_score"] = round(composite, 4)
    return report


def validate_and_refine_clusters(
    cluster_articles: Dict[int, List[NewsArticle]],
    embeddings: np.ndarray,
    articles: List[NewsArticle],
    labels: np.ndarray,
    min_coherence: float = 0.40,
    reject_threshold: float = 0.25,
    merge_threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> Tuple[Dict[int, List[NewsArticle]], np.ndarray, int]:
    """Validate cluster quality and refine: split incoherent, merge redundant, reject noise.

    Operates in original embedding space (not UMAP-reduced).

    Returns (refined_cluster_articles, refined_labels, noise_change).
    """
    if not cluster_articles:
        return cluster_articles, labels, 0

    aid_to_idx = {id(a): i for i, a in enumerate(articles)}
    emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings

    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb_array / norms

    # Phase 1: Compute coherence
    cluster_coherence: Dict[int, float] = {}
    cluster_centroids: Dict[int, np.ndarray] = {}

    for cid, arts in cluster_articles.items():
        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) < 2:
            cluster_coherence[cid] = 1.0  # Single article = trivially coherent
            if idxs:
                cluster_centroids[cid] = emb_norm[idxs[0]]
            continue

        cluster_embs = emb_norm[idxs]
        sim_matrix = np.dot(cluster_embs, cluster_embs.T)
        n = len(sim_matrix)
        mean_sim = (sim_matrix.sum() - n) / max(n * (n - 1), 1)
        cluster_coherence[cid] = float(mean_sim)
        cluster_centroids[cid] = cluster_embs.mean(axis=0)

    if cluster_coherence:
        coherences = list(cluster_coherence.values())
        logger.info(
            f"Cluster coherence: min={min(coherences):.3f}, "
            f"mean={sum(coherences)/len(coherences):.3f}, "
            f"max={max(coherences):.3f}"
        )

    # Phase 2: Split incoherent clusters
    new_labels = labels.copy()

    next_cluster_id = max(cluster_articles.keys()) + 1 if cluster_articles else 0
    splits_done = 0

    to_split = {
        cid: arts for cid, arts in cluster_articles.items()
        if cluster_coherence.get(cid, 1.0) < min_coherence
        and len(arts) >= min_cluster_size * 2  # Need enough to split
    }

    low_coherence_cids = {
        cid for cid, arts in cluster_articles.items()
        if cluster_coherence.get(cid, 1.0) < reject_threshold
    }
    if low_coherence_cids:
        rejected_articles = sum(len(cluster_articles[c]) for c in low_coherence_cids)
        logger.info(
            f"  Rejecting {len(low_coherence_cids)} low-coherence clusters "
            f"({rejected_articles} articles → noise): "
            f"{[round(cluster_coherence[c], 3) for c in low_coherence_cids]}"
        )
        for cid in low_coherence_cids:
            del cluster_articles[cid]

    # Flag bimodal clusters (two distinct sub-groups) even if coherence passes
    for cid, arts in list(cluster_articles.items()):
        if cid in to_split or cid in low_coherence_cids:
            continue
        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) < min_cluster_size * 2:
            continue
        cluster_embs = emb_norm[idxs]
        centroid = cluster_embs.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-10:
            sims = cluster_embs @ (centroid / c_norm)
            from scipy.stats import kurtosis as _kurtosis
            k_val = float(_kurtosis(sims))
            if k_val < -1.0:  # Bimodal: two distinct sub-groups
                to_split[cid] = arts
                logger.info(
                    f"  Bimodal cluster {cid}: kurtosis={k_val:.2f}, "
                    f"forcing split ({len(arts)} articles)"
                )

    for cid, arts in to_split.items():

        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) < min_cluster_size * 2:
            continue

        cluster_embs = emb_norm[idxs]
        best_sub_labels = None
        best_avg_coherence = cluster_coherence.get(cid, 0.0)

        for n_sub in range(2, min(6, len(idxs) // min_cluster_size + 1)):
            try:
                agg = AgglomerativeClustering(
                    n_clusters=n_sub,
                    metric='cosine',
                    linkage='average',
                )
                sub_labels = agg.fit_predict(cluster_embs)

                sub_coherences = []
                valid_split = True
                for sub_id in range(n_sub):
                    sub_mask = sub_labels == sub_id
                    if sub_mask.sum() < min_cluster_size:
                        valid_split = False
                        break
                    sub_embs = cluster_embs[sub_mask]
                    sim = np.dot(sub_embs, sub_embs.T)
                    n = len(sim)
                    if n < 2:
                        sub_coherences.append(1.0)
                        continue
                    sub_coh = (sim.sum() - n) / max(n * (n - 1), 1)
                    sub_coherences.append(sub_coh)

                if not valid_split or not sub_coherences:
                    continue

                sizes = [int((sub_labels == sid).sum()) for sid in range(n_sub)]
                total_size = sum(sizes)
                weighted_avg = sum(c * s for c, s in zip(sub_coherences, sizes)) / max(total_size, 1)

                if weighted_avg > best_avg_coherence:
                    best_sub_labels = sub_labels
                    best_avg_coherence = weighted_avg
            except Exception:
                continue

        if best_sub_labels is not None:
            # Apply the split
            for sub_id in range(best_sub_labels.max() + 1):
                sub_mask = best_sub_labels == sub_id
                sub_idxs = [idxs[i] for i, m in enumerate(sub_mask) if m]
                if len(sub_idxs) >= min_cluster_size:
                    for idx in sub_idxs:
                        new_labels[idx] = next_cluster_id
                    next_cluster_id += 1
                else:
                    # Too small → noise
                    for idx in sub_idxs:
                        new_labels[idx] = -1

            splits_done += 1
            logger.debug(
                f"  Split cluster {cid}: coherence {cluster_coherence[cid]:.3f} → "
                f"{best_sub_labels.max() + 1} sub-clusters (coherence ≥ {best_avg_coherence:.3f})"
            )

    # Phase 3: Merge redundant clusters
    refined_groups: Dict[int, List[NewsArticle]] = defaultdict(list)
    for article, label in zip(articles, new_labels):
        if label >= 0:
            refined_groups[int(label)].append(article)

    refined_centroids: Dict[int, np.ndarray] = {}
    for cid, arts in refined_groups.items():
        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if idxs:
            refined_centroids[cid] = emb_norm[idxs].mean(axis=0)

    merges_done = 0
    merge_map: Dict[int, int] = {}  # cid → merge_target_cid
    cids = sorted(refined_centroids.keys())

    if len(cids) > 1:
        centroid_matrix = np.array([refined_centroids[c] for c in cids])
        centroid_sims = np.dot(centroid_matrix, centroid_matrix.T)

        # Pre-compute top-5 entities and title words for merge checks
        cluster_top_entities: Dict[int, set] = {}
        cluster_title_words: Dict[int, set] = {}
        for cid_m in cids:
            arts = refined_groups.get(cid_m, [])
            ent_counts: Counter = Counter()
            for a in arts:
                for name in getattr(a, 'entity_names', []):
                    ent_counts[name.lower()] += 1
            cluster_top_entities[cid_m] = {e for e, _ in ent_counts.most_common(5)}
            title_words = set()
            for a in arts:
                for w in (a.title or "").lower().split():
                    w = w.strip(".,;:!?'\"()-")
                    if len(w) > 3 and w not in {"the", "and", "for", "with", "from", "that", "this", "india", "indian", "market", "company", "business", "sector"}:
                        title_words.add(w)
            cluster_title_words[cid_m] = title_words

        for i in range(len(cids)):
            if cids[i] in merge_map:
                continue
            for j in range(i + 1, len(cids)):
                if cids[j] in merge_map:
                    continue

                sim = centroid_sims[i, j]
                should_merge = False

                if sim >= merge_threshold:
                    should_merge = True

                # Entity-aware merge: lower threshold when >50% entity overlap
                elif sim >= 0.65:
                    ents_i = cluster_top_entities.get(cids[i], set())
                    ents_j = cluster_top_entities.get(cids[j], set())
                    if ents_i and ents_j:
                        overlap = len(ents_i & ents_j)
                        max_possible = min(len(ents_i), len(ents_j))
                        if max_possible > 0 and overlap / max_possible > 0.50:
                            should_merge = True
                            logger.debug(
                                f"  Entity-aware merge: {cids[j]}→{cids[i]} "
                                f"(sim={sim:.3f}, entity overlap={overlap}/{max_possible})"
                            )

                # Title-based merge: >60% content word overlap
                elif sim >= 0.55:
                    words_i = cluster_title_words.get(cids[i], set())
                    words_j = cluster_title_words.get(cids[j], set())
                    if words_i and words_j:
                        union = len(words_i | words_j)
                        inter = len(words_i & words_j)
                        if union > 0 and inter / union > 0.60:
                            should_merge = True
                            logger.debug(
                                f"  Title-based merge: {cids[j]}→{cids[i]} "
                                f"(sim={sim:.3f}, title overlap={inter}/{union})"
                            )

                if should_merge:
                    merge_map[cids[j]] = cids[i]
                    merges_done += 1
                    if sim >= merge_threshold:
                        logger.debug(
                            f"  Merging cluster {cids[j]} into {cids[i]}: "
                            f"similarity={sim:.3f}"
                        )

    # Resolve transitive merge chains
    for cid in list(merge_map.keys()):
        target = merge_map[cid]
        while target in merge_map:
            target = merge_map[target]
        merge_map[cid] = target

    if merge_map:
        for idx in range(len(new_labels)):
            label = int(new_labels[idx])
            if label in merge_map:
                new_labels[idx] = merge_map[label]

    final_groups: Dict[int, List[NewsArticle]] = defaultdict(list)
    for article, label in zip(articles, new_labels):
        if label >= 0:
            final_groups[int(label)].append(article)

    # Phase 4: Multi-signal cascade outlier detection
    # Articles must fail 2+ independent dimensions to be ejected.

    def _ent_name(ent) -> str:
        if hasattr(ent, "text"):
            return ent.text
        if isinstance(ent, dict):
            return ent.get("text", str(ent))
        return str(ent)

    def _ent_type(ent) -> str:
        if hasattr(ent, "type"):
            return ent.type
        if isinstance(ent, dict):
            return ent.get("type", "")
        return ""

    def _ent_salience(ent) -> float:
        if hasattr(ent, "salience"):
            return ent.salience
        if isinstance(ent, dict):
            return ent.get("salience", 0.0)
        return 0.0

    outliers_removed = 0
    entity_outliers_removed = 0
    event_outliers_removed = 0
    content_outliers_removed = 0

    for cid, arts in list(final_groups.items()):
        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) < 4:
            continue

        # Pass A: Embedding-based outlier detection
        cluster_embs = emb_norm[idxs]
        centroid = cluster_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        sims = cluster_embs @ centroid_norm

        q1, q3 = np.percentile(sims, [25, 75])
        iqr = q3 - q1
        lower_fence = max(q1 - 1.5 * iqr, 0.12)
        outlier_mask = sims < lower_fence

        # Pass B: Entity "WHO" check (token-level matching)
        _WHO_STOP_TOKENS = frozenset({
            "the", "and", "for", "its", "new", "all", "has", "was", "are", "not",
            "but", "from", "with", "will", "been", "have", "this", "that", "said",
            "over", "more", "than", "also", "into", "amid", "who", "how", "why",
            "may", "can", "now", "per", "via", "set", "get", "key", "top",
            "limited", "ltd", "inc", "corp", "group", "company", "pvt",
            "india", "indian", "global", "world", "national",
        })

        who_article_tokens: list[set] = []
        for a in arts:
            tokens = set()
            for ent in getattr(a, "entities", []) or []:
                etype = _ent_type(ent)
                name = _ent_name(ent).lower().strip()
                if etype in ("ORG", "PERSON", "PRODUCT") and name:
                    for tok in name.split():
                        tok = tok.strip(".,;:!?'\"()-")
                        if len(tok) > 2 and tok not in _WHO_STOP_TOKENS:
                            tokens.add(tok)
            who_article_tokens.append(tokens)

        who_token_freq: Counter = Counter()
        for wt in who_article_tokens:
            who_token_freq.update(wt)
        who_threshold = max(len(arts) * 0.15, 2)
        core_who = {t for t, c in who_token_freq.items() if c >= who_threshold}

        all_entity_counter: Counter = Counter()
        for a in arts:
            for ent in getattr(a, "entities", []) or []:
                name = _ent_name(ent).lower().strip()
                if name and len(name) > 2:
                    all_entity_counter[name] += 1
        all_threshold = max(len(arts) * 0.30, 2)
        core_entities = {e for e, c in all_entity_counter.items() if c >= all_threshold}

        # Pass C: Event type concentration
        event_counter: Counter = Counter()
        for a in arts:
            etype = getattr(a, "_trigger_event", "general")
            event_counter[etype] += 1
        dominant_event, dominant_count = event_counter.most_common(1)[0] if event_counter else ("general", 0)
        event_concentration = dominant_count / len(arts) if arts else 0

        # Pass D: Content keyword participation
        _CONTENT_STOP = frozenset({
            "the", "and", "for", "with", "from", "that", "this", "has", "was",
            "are", "have", "not", "but", "its", "will", "can", "may", "says",
            "said", "new", "after", "over", "into", "how", "why", "what",
            "also", "been", "set", "more", "than", "just", "like", "about",
            "here", "most", "some", "could", "would", "should", "being",
            "their", "them", "they", "your", "very", "does", "much", "many",
            "only", "well", "back", "even", "first", "next", "last", "know",
            "need", "take", "make", "come", "look", "want", "give", "good",
            "year", "years", "india", "indian", "amid", "market", "markets",
            "company", "companies", "business", "sector", "industry", "economy",
            "growth", "report", "news", "latest", "today", "live", "update",
            "crore", "lakh", "rupee", "rupees", "govt", "government",
            "global", "rise", "fell", "drop", "high", "firm", "share",
            "shares", "stock", "price", "percent", "rate", "rates",
            "which", "there", "these", "those", "when", "where", "while",
            "other", "such", "each", "every", "both", "through", "between",
        })

        content_word_sets = []
        for a in arts:
            text = (a.title or "") + " " + ((a.content or "")[:300])
            words = set(
                w for raw in text.split()
                if len(raw) > 2
                for w in [raw.lower().strip(".,;:!?'\"()-–—:/")]
                if len(w) > 2 and w not in _CONTENT_STOP and not w.isdigit()
            )
            content_word_sets.append(words)

        content_word_freq: Counter = Counter()
        for ws in content_word_sets:
            for w in ws:
                content_word_freq[w] += 1
        content_topic_thresh = max(len(arts) * 0.20, 2)
        content_topic_words = {w for w, c in content_word_freq.items() if c >= content_topic_thresh}

        kept_arts = []
        for i, (art, idx) in enumerate(zip(arts, idxs)):
            fail_signals = 0
            fail_reasons = []

            if outlier_mask[i]:
                fail_signals += 1
                fail_reasons.append("emb")

            entity_miss = False
            if core_who and len(arts) >= 6:
                art_who_tokens = set()
                for ent in getattr(art, "entities", []) or []:
                    if _ent_type(ent) in ("ORG", "PERSON", "PRODUCT"):
                        for tok in _ent_name(ent).lower().split():
                            tok = tok.strip(".,;:!?'\"()-")
                            if len(tok) > 2 and tok not in _WHO_STOP_TOKENS:
                                art_who_tokens.add(tok)
                if not art_who_tokens & core_who:
                    entity_miss = True

            if entity_miss and core_entities and len(arts) >= 8:
                art_ents = set()
                for ent in getattr(art, "entities", []) or []:
                    name = _ent_name(ent).lower().strip()
                    if name:
                        art_ents.add(name)
                if art_ents & core_entities:
                    entity_miss = False

            if entity_miss:
                fail_signals += 1
                fail_reasons.append("entity")

            if event_concentration >= 0.50 and len(arts) >= 6:
                art_event = getattr(art, "_trigger_event", "general")
                if art_event != dominant_event and art_event != "general":
                    fail_signals += 1
                    fail_reasons.append("event")

            if content_topic_words and len(arts) >= 6:
                if not content_word_sets[i] & content_topic_words:
                    fail_signals += 1
                    fail_reasons.append("content")

            if fail_signals >= 2:
                new_labels[idx] = -1
                if "emb" in fail_reasons:
                    outliers_removed += 1
                elif "entity" in fail_reasons:
                    entity_outliers_removed += 1
                elif "event" in fail_reasons:
                    event_outliers_removed += 1
                else:
                    content_outliers_removed += 1
                continue

            kept_arts.append(art)
        final_groups[cid] = kept_arts

    total_outliers = outliers_removed + entity_outliers_removed + event_outliers_removed + content_outliers_removed
    if total_outliers:
        logger.info(
            f"  Outlier detection: {total_outliers} articles removed "
            f"({outliers_removed} embedding, {entity_outliers_removed} entity, "
            f"{event_outliers_removed} event-type, {content_outliers_removed} content)"
        )

    # Phase 4.5: Title-kinship outlier removal + MACRO/MICRO classification
    _TITLE_STOP = frozenset({
        # English function words
        "the", "and", "for", "with", "from", "that", "this", "has", "was",
        "are", "have", "not", "but", "its", "will", "can", "may", "says",
        "said", "new", "after", "over", "into", "how", "why", "what",
        "also", "been", "set", "more", "than", "just", "like", "about",
        "here", "most", "some", "could", "would", "should", "being",
        "their", "them", "they", "your", "very", "does", "much", "many",
        "only", "well", "back", "even", "first", "next", "last", "know",
        "need", "take", "make", "come", "look", "want", "give", "good",
        "year", "years", "could", "says", "gets", "made", "gets", "among",
        # Domain-generic: common in India business/finance news
        "india", "indian", "amid", "market", "markets", "company",
        "companies", "business", "sector", "industry", "economy",
        "growth", "report", "news", "latest", "today", "live", "update",
        "updates", "crore", "lakh", "rupee", "rupees", "govt",
        "government", "minister", "ministry", "policy", "global",
        "rise", "rises", "fell", "falls", "drop", "drops", "high",
        "highs", "lows", "firm", "firms", "share", "shares", "stock",
        "stocks", "price", "prices", "percent", "rate", "rates",
    })

    _TITLE_STOP_STEMMED = frozenset(_simple_stem(w) for w in _TITLE_STOP)

    def _extract_title_words(title: str) -> set:
        result = set()
        for raw in (title or "").split():
            if len(raw) <= 2:
                continue
            w = raw.lower().strip(".,;:!?'\"()-–—:/")
            if len(w) > 2 and w not in _TITLE_STOP and not w.isdigit():
                stemmed = _simple_stem(w)
                if stemmed not in _TITLE_STOP_STEMMED:
                    result.add(stemmed)
        return result

    title_kinship_removed = 0
    title_splits_done = 0

    for cid, arts in list(final_groups.items()):
        if len(arts) < 5:
            continue

        title_word_sets = [_extract_title_words(a.title) for a in arts]

        kinship_scores = []
        for i in range(len(arts)):
            kin = sum(
                1 for j in range(len(arts))
                if i != j and title_word_sets[i] & title_word_sets[j]
            )
            kinship_scores.append(kin)

        # Adaptive kinship threshold (looser for larger clusters)
        if len(arts) >= 30:
            kin_pct = 0.05  # Very large: 5% (e.g., 44 articles → need 2 kin)
        elif len(arts) >= 15:
            kin_pct = 0.07  # Large: 7% (e.g., 20 articles → need 1 kin)
        elif len(arts) >= 10:
            kin_pct = 0.10  # Medium: 10%
        else:
            kin_pct = 0.15  # Small: 15%
        kin_threshold = max(1, int(len(arts) * kin_pct))

        core_count = sum(1 for k in kinship_scores if k >= kin_threshold)
        if core_count < min_cluster_size:
            continue
        kept_arts = []
        for a, kin_score in zip(arts, kinship_scores):
            if kin_score >= kin_threshold:
                kept_arts.append(a)
            else:
                idx = aid_to_idx.get(id(a))
                if idx is not None:
                    new_labels[idx] = -1
                    title_kinship_removed += 1

        if len(kept_arts) >= min_cluster_size:
            final_groups[cid] = kept_arts
        else:
            del final_groups[cid]
            for a in kept_arts:
                idx = aid_to_idx.get(id(a))
                if idx is not None:
                    new_labels[idx] = -1

    if title_kinship_removed:
        logger.info(
            f"  Title-kinship: {title_kinship_removed} articles removed "
            f"(no shared title words with cluster)"
        )

    # MACRO/MICRO classification + splitting
    for cid, arts in list(final_groups.items()):
        if len(arts) < 6:
            continue

        title_word_sets = [_extract_title_words(a.title) for a in arts]

        word_freq: Counter = Counter()
        for ws in title_word_sets:
            for w in ws:
                word_freq[w] += 1

        topic_threshold = max(len(arts) * 0.20, 2)
        topic_words = {w for w, c in word_freq.items() if c >= topic_threshold}

        if not topic_words:
            # No common vocabulary -- try embedding-based split
            idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
            if len(idxs) < 6:
                continue
            cluster_embs = emb_norm[idxs]
            emb_sims = np.dot(cluster_embs, cluster_embs.T)
            emb_dist = 1.0 - emb_sims
            np.fill_diagonal(emb_dist, 0.0)
            emb_dist = np.clip(emb_dist, 0.0, 2.0)

            best_n = 2
            best_score = -1.0
            for n_sub in range(2, min(5, len(idxs) // min_cluster_size + 1)):
                try:
                    agg = AgglomerativeClustering(
                        n_clusters=n_sub, metric="precomputed",
                        linkage="average",
                    )
                    sub_labels = agg.fit_predict(emb_dist)
                    # Score by average intra-cluster embedding similarity
                    scores = []
                    for s_id in range(n_sub):
                        members = np.where(sub_labels == s_id)[0]
                        if len(members) >= 2:
                            sub_sims = emb_sims[np.ix_(members, members)]
                            n_m = len(members)
                            avg = (sub_sims.sum() - n_m) / max(n_m * (n_m - 1), 1)
                            scores.append(avg)
                    score = np.mean(scores) if scores else 0.0
                    if score > best_score:
                        best_score = score
                        best_n = n_sub
                except Exception:
                    continue

            if best_score > 0.0:
                agg = AgglomerativeClustering(
                    n_clusters=best_n, metric="precomputed",
                    linkage="average",
                )
                sub_labels = agg.fit_predict(emb_dist)
                created = 0
                del final_groups[cid]
                for s_id in range(best_n):
                    member_idxs = np.where(sub_labels == s_id)[0]
                    sub_arts = [arts[member_idxs[k]] for k in range(len(member_idxs))]
                    if len(sub_arts) >= min_cluster_size:
                        final_groups[next_cluster_id] = sub_arts
                        for a in sub_arts:
                            idx = aid_to_idx.get(id(a))
                            if idx is not None:
                                new_labels[idx] = next_cluster_id
                        next_cluster_id += 1
                        created += 1
                    else:
                        for a in sub_arts:
                            idx = aid_to_idx.get(id(a))
                            if idx is not None:
                                new_labels[idx] = -1
                if created > 0:
                    title_splits_done += 1
                    logger.info(
                        f"  MACRO cluster {cid}: zero topic words, "
                        f"embedding-split into {created} sub-clusters"
                    )
            continue

        participating = [bool(ws & topic_words) for ws in title_word_sets]
        participation_rate = sum(participating) / len(arts)

        if participation_rate >= 0.60:
            if participation_rate < 1.0:
                kept = []
                removed = 0
                for a, p in zip(arts, participating):
                    if p:
                        kept.append(a)
                    else:
                        idx = aid_to_idx.get(id(a))
                        if idx is not None:
                            new_labels[idx] = -1
                            removed += 1
                if removed > 0 and len(kept) >= min_cluster_size:
                    final_groups[cid] = kept
                    title_kinship_removed += removed
                    logger.info(
                        f"  MICRO cluster {cid}: {removed} non-participating "
                        f"articles removed (participation={participation_rate:.0%})"
                    )
            continue

        # MACRO cluster: hybrid Jaccard + embedding split
        n = len(arts)
        jaccard_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                union = title_word_sets[i] | title_word_sets[j]
                if union:
                    jac = len(title_word_sets[i] & title_word_sets[j]) / len(union)
                else:
                    jac = 0.0
                jaccard_matrix[i, j] = jac
                jaccard_matrix[j, i] = jac

        idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
        if len(idxs) == n:
            cluster_embs = emb_norm[idxs]
            emb_sims = np.dot(cluster_embs, cluster_embs.T)
            hybrid_sim = 0.6 * jaccard_matrix + 0.4 * emb_sims
        else:
            hybrid_sim = jaccard_matrix

        hybrid_dist = 1.0 - hybrid_sim
        np.fill_diagonal(hybrid_dist, 0.0)
        hybrid_dist = np.clip(hybrid_dist, 0.0, 2.0)

        best_n = 2
        best_score = -1.0
        for n_sub in range(2, min(5, n // min_cluster_size + 1)):
            try:
                agg = AgglomerativeClustering(
                    n_clusters=n_sub, metric="precomputed",
                    linkage="average",
                )
                sub_labels = agg.fit_predict(hybrid_dist)
                scores = []
                for s in range(n_sub):
                    members = np.where(sub_labels == s)[0]
                    if len(members) >= 2:
                        sub_sims = hybrid_sim[np.ix_(members, members)]
                        n_m = len(members)
                        avg = (sub_sims.sum() - n_m) / max(n_m * (n_m - 1), 1)
                        scores.append(avg)
                score = np.mean(scores) if scores else 0.0
                if score > best_score:
                    best_score = score
                    best_n = n_sub
            except Exception:
                continue

        if best_score < 0.01:
            continue

        agg = AgglomerativeClustering(
            n_clusters=best_n, metric="precomputed",
            linkage="average",
        )
        sub_labels = agg.fit_predict(hybrid_dist)

        created = 0
        del final_groups[cid]
        for s in range(best_n):
            member_idxs = np.where(sub_labels == s)[0]
            sub_arts = [arts[i] for i in member_idxs]
            if len(sub_arts) >= min_cluster_size:
                final_groups[next_cluster_id] = sub_arts
                for a in sub_arts:
                    idx = aid_to_idx.get(id(a))
                    if idx is not None:
                        new_labels[idx] = next_cluster_id
                next_cluster_id += 1
                created += 1
            else:
                for a in sub_arts:
                    idx = aid_to_idx.get(id(a))
                    if idx is not None:
                        new_labels[idx] = -1

        if created > 0:
            title_splits_done += 1
            top_topic = sorted(topic_words, key=lambda w: word_freq[w], reverse=True)[:3]
            logger.info(
                f"  MACRO cluster {cid}: participation={participation_rate:.0%}, "
                f"topic_words={top_topic}, hybrid-split into {created} sub-clusters"
            )

    if title_splits_done:
        logger.info(
            f"  Title-coherence: {title_splits_done} MACRO clusters split"
        )

    # Phase 4.6: Pairwise vocabulary coherence
    pairwise_dissolved = 0
    articles_ejected = 0
    _PW_STOP = frozenset({
        # Basic English
        "the", "and", "for", "its", "new", "all", "has", "was", "are",
        "not", "but", "from", "with", "will", "been", "have", "this",
        "that", "said", "over", "more", "than", "also", "into", "amid",
        "who", "how", "why", "whom", "whose", "may", "can", "now",
        "per", "via", "set", "get", "key", "top", "just", "like",
        "some", "most", "much", "many", "even", "very", "what", "when",
        "where", "which", "while", "does", "says", "only", "such",
        "were", "them", "then", "each", "both", "here", "year", "years",
        "being", "would", "could", "should", "their", "they", "about",
        "other", "after", "before",
        # Generic number/quantity words
        "million", "billion", "trillion", "crore", "lakh",
        # Non-discriminating verbs/adjectives
        "first", "last", "next", "big", "look", "check", "among", "across",
        "major", "calls", "pushes", "faces", "makes", "takes", "shows",
        "launch", "launches", "launched", "impact", "expected", "likely",
        "rise", "need", "aims", "targets", "plans", "seeks", "moves",
        "hits", "sees", "set", "could", "would",
        # Domain stop words (India business news)
        "india", "indian", "global", "world", "national", "market", "markets",
        "company", "limited", "ltd", "inc", "corp", "group", "pvt",
        "business", "sector", "growth", "economy", "govt", "report",
        "news", "today",
        # Tech/digital false bridges — connect unrelated tech stories
        "digital", "technology", "tech", "system", "systems",
        "platform", "online", "cyber",
        # Generic government/institutional words (only truly non-discriminating)
        "government", "policy", "programme", "scheme",
        # Generic financial words — ONLY truly non-discriminating ones.
        # DO NOT add "fund", "share", "stock", "price" — these are
        # discriminating within specific clusters (e.g., "Peak XV fund").
        "percent", "quarterly", "annual", "fiscal",
        # Media/reporting noise
        "latest", "breaking", "update", "updates", "live", "watch",
        "read", "know", "things", "need", "five", "rush", "hour",
    })

    _PW_STOP_STEMMED = frozenset(_simple_stem(w) for w in _PW_STOP)

    def _build_vocab(a, include_content=False):
        """Build stemmed vocabulary set from title words, entity tokens, and optionally content."""
        vocab = set()
        for w in (a.title or "").lower().split():
            w = w.strip(".,;:!?'\"()-\u2013\u2014:/[]")
            if len(w) > 2 and w not in _PW_STOP and not w.isdigit():
                stemmed = _simple_stem(w)
                if stemmed not in _PW_STOP_STEMMED:
                    vocab.add(stemmed)
        for name in getattr(a, 'entity_names', []) or []:
            for tok in name.lower().split():
                tok = tok.strip(".,;:!?'\"()-")
                if len(tok) > 2 and tok not in _PW_STOP:
                    stemmed = _simple_stem(tok)
                    if stemmed not in _PW_STOP_STEMMED:
                        vocab.add(stemmed)
        if include_content:
            body = getattr(a, 'content', None) or getattr(a, 'summary', None) or ""
            if body:
                words_seen = 0
                for w in body.lower().split():
                    if words_seen >= 150:
                        break
                    w = w.strip(".,;:!?'\"()-\u2013\u2014:/[]{}|")
                    if len(w) > 3 and w not in _PW_STOP and not w.isdigit():
                        stemmed = _simple_stem(w)
                        if stemmed not in _PW_STOP_STEMMED:
                            vocab.add(stemmed)
                    words_seen += 1
        return vocab

    # Step 1: Dissolve fully incoherent clusters
    # Use include_content=True so summaries boost Jaccard — headlines alone are
    # too sparse: "IPO market heats up" and "Companies rush to list on exchanges"
    # share zero title words yet cover the same story. Summary expands the vocab.
    for cid, arts in list(final_groups.items()):
        if len(arts) < 2:
            continue

        art_vocab = [_build_vocab(a, include_content=True) for a in arts]

        sims = []
        for i in range(len(art_vocab)):
            for j in range(i + 1, len(art_vocab)):
                vi, vj = art_vocab[i], art_vocab[j]
                union = len(vi | vj)
                inter = len(vi & vj)
                sims.append(inter / union if union > 0 else 0.0)
        avg_jaccard = sum(sims) / len(sims) if sims else 0.0

        # Threshold 0.01 (not 0.02): with content included, vocab sets are larger
        # so Jaccard scores are naturally lower. 0.01 dissolves only truly chaotic
        # mixed-topic clusters (< 1% word overlap across all pairs).
        if avg_jaccard < 0.01:
            del final_groups[cid]
            for a in arts:
                idx = aid_to_idx.get(id(a))
                if idx is not None:
                    new_labels[idx] = -1
            pairwise_dissolved += 1

    # Step 1.5: Split intermediate-coherence clusters
    PW_SPLIT_THRESHOLD = 0.06
    pairwise_splits = 0
    for cid, arts in list(final_groups.items()):
        if len(arts) < 6:
            continue

        art_vocab = [_build_vocab(a) for a in arts]
        n = len(arts)

        # Compute pairwise Jaccard
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = art_vocab[i], art_vocab[j]
                union = len(vi | vj)
                inter = len(vi & vj)
                sims.append(inter / union if union > 0 else 0.0)
        avg_jaccard = sum(sims) / len(sims) if sims else 0.0

        if avg_jaccard >= PW_SPLIT_THRESHOLD or avg_jaccard < 0.02:
            continue

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = art_vocab[i], art_vocab[j]
                union = len(vi | vj)
                inter = len(vi & vj)
                sim = inter / union if union > 0 else 0.0
                dist_matrix[i, j] = 1.0 - sim
                dist_matrix[j, i] = 1.0 - sim

        best_split = None
        best_score = -1.0
        for n_clusters in [2, 3]:
            if n < n_clusters * 2:
                continue
            try:
                agg = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="complete",
                )
                sub_labels = agg.fit_predict(dist_matrix)
            except Exception:
                continue

            sub_groups: Dict[int, List[int]] = defaultdict(list)
            for i, sl in enumerate(sub_labels):
                sub_groups[sl].append(i)

            if any(len(idxs) < 3 for idxs in sub_groups.values()):
                continue

            sub_jaccards = []
            for idxs in sub_groups.values():
                sub_sims = []
                for ii in range(len(idxs)):
                    for jj in range(ii + 1, len(idxs)):
                        vi = art_vocab[idxs[ii]]
                        vj = art_vocab[idxs[jj]]
                        union = len(vi | vj)
                        inter = len(vi & vj)
                        sub_sims.append(inter / union if union > 0 else 0.0)
                sub_avg = sum(sub_sims) / len(sub_sims) if sub_sims else 0.0
                sub_jaccards.append(sub_avg)

            if not all(sj > avg_jaccard for sj in sub_jaccards):
                continue

            split_score = min(sub_jaccards)
            if split_score > best_score:
                best_score = split_score
                best_split = sub_groups

        if best_split:
            next_cid = max(final_groups.keys()) + 1
            del final_groups[cid]
            for sub_idxs in best_split.values():
                sub_arts = [arts[i] for i in sub_idxs]
                final_groups[next_cid] = sub_arts
                for a in sub_arts:
                    idx = aid_to_idx.get(id(a))
                    if idx is not None:
                        new_labels[idx] = next_cid
                next_cid += 1
            pairwise_splits += 1
            logger.info(
                f"Phase 4.6 Step 1.5: Split cluster (avg_jaccard={avg_jaccard:.3f}) "
                f"into {len(best_split)} sub-clusters"
            )

    # Step 2: Per-article connectivity ejection (MAD-based)
    for cid, arts in list(final_groups.items()):
        if len(arts) < 3:
            continue

        art_vocab = [_build_vocab(a, include_content=True) for a in arts]
        n = len(arts)

        connectivity = []
        for i in range(n):
            pair_sims = []
            for j in range(n):
                if i == j:
                    continue
                vi, vj = art_vocab[i], art_vocab[j]
                union = len(vi | vj)
                inter = len(vi & vj)
                pair_sims.append(inter / union if union > 0 else 0.0)
            connectivity.append(sum(pair_sims) / len(pair_sims) if pair_sims else 0.0)

        conn_arr = np.array(connectivity)
        median_conn = float(np.median(conn_arr))
        mad = float(np.median(np.abs(conn_arr - median_conn)))

        eject_thresh = max(median_conn - 1.5 * max(mad, 1e-6), 0.0)

        to_eject = [i for i, c in enumerate(connectivity) if c < eject_thresh]

        if len(to_eject) > 0.4 * n:
            continue

        if to_eject:
            for i in sorted(to_eject, reverse=True):
                a = arts[i]
                idx = aid_to_idx.get(id(a))
                if idx is not None:
                    new_labels[idx] = -1
                arts.pop(i)
            articles_ejected += len(to_eject)

    # Step 3: Connected-component ejection (keep largest component)
    component_ejected = 0
    for cid, arts in list(final_groups.items()):
        if len(arts) < 3:
            continue

        art_vocab = [_build_vocab(a, include_content=True) for a in arts]
        n = len(arts)

        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = art_vocab[i], art_vocab[j]
                if vi & vj:  # any shared vocabulary
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        components: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(comp)

        if len(components) <= 1:
            continue

        largest = max(components, key=len)
        if len(largest) < min_cluster_size:
            del final_groups[cid]
            for a in arts:
                idx = aid_to_idx.get(id(a))
                if idx is not None:
                    new_labels[idx] = -1
            pairwise_dissolved += 1
            continue

        to_eject = set()
        for comp in components:
            if comp is not largest:
                to_eject.update(comp)

        if to_eject:
            for i in sorted(to_eject, reverse=True):
                a = arts[i]
                idx = aid_to_idx.get(id(a))
                if idx is not None:
                    new_labels[idx] = -1
                arts.pop(i)
            component_ejected += len(to_eject)

    if pairwise_dissolved or pairwise_splits or articles_ejected or component_ejected:
        logger.info(
            f"  Pairwise coherence: {pairwise_dissolved} dissolved, "
            f"{pairwise_splits} split, "
            f"{articles_ejected} ejected (MAD), "
            f"{component_ejected} ejected (components)"
        )

    noise_additions = 0
    final_clean: Dict[int, List[NewsArticle]] = {}
    for cid, arts in final_groups.items():
        if len(arts) >= min_cluster_size:
            final_clean[cid] = arts
        else:
            for a in arts:
                if id(a) in aid_to_idx:
                    new_labels[aid_to_idx[id(a)]] = -1
                    noise_additions += 1

    original_noise = int(np.sum(labels == -1))
    new_noise = int(np.sum(new_labels == -1))
    noise_change = new_noise - original_noise

    logger.info(
        f"Coherence validation: {splits_done} splits, {merges_done} merges, "
        f"{outliers_removed} outliers, noise {original_noise}→{new_noise} "
        f"({'+' if noise_change >= 0 else ''}{noise_change}), "
        f"{len(final_clean)} final clusters"
    )

    return final_clean, new_labels, noise_change
