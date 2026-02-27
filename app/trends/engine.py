"""
TrendPipeline — layered trend detection pipeline.

Architecture: 6 composable layers with typed contracts.

  Layer 1 (Ingest):      RSS → scrape → dedup → NER → embed → filter
  Layer 2 (Cluster):     Leiden community detection → coherence → keywords
  Layer 3 (Relate):      Causal graph — entity bridges, sector chains (stub)
  Layer 4 (Temporalize): Trend memory — novelty vs. continuity (stub)
  Layer 5 (Enrich):      Signal scoring → LLM synthesis → quality gate → tree

Clustering: Leiden on k-NN graph (1024-dim embeddings, no dim reduction).
  Deterministic, production-validated (NewsCatcher, GraphRAG).

REF: Traag et al. 2019 (Leiden), BERTopic (Grootendorst 2022),
     BERTrend (Boutaleb et al. 2024).
"""

import asyncio
import logging
import math
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.news import NewsArticle
from app.schemas.trends import TrendTree
from app.news.dedup import ArticleDeduplicator
from app.news.entity_extractor import EntityExtractor
from app.news.scraper import scrape_articles
from app.news.event_classifier import EmbeddingEventClassifier
from app.tools.embeddings import EmbeddingTool
from app.trends.keywords import KeywordExtractor
from app.trends.signals import compute_all_signals
from app.trends.tree import build_trend_tree
from app.trends.synthesis import synthesize_clusters
from app.trends.subclustering import recursive_subcluster
from app.trends.coherence import validate_and_refine_clusters, compute_cluster_quality_report

# ── Logging: file + console ─────────────────────────────────────────────
_LOG_FILE = Path("trend_engine_debug.log")


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record for crash-safe debugging."""
    def emit(self, record):
        super().emit(record)
        self.flush()


_file_handler = FlushingFileHandler(_LOG_FILE, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'
))
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Attach handlers to the parent 'app.trends' logger so ALL sub-modules
# (subclustering, synthesis, tree_builder, etc.) write to the same log file.
_parent_logger = logging.getLogger("app.trends")
_parent_logger.setLevel(logging.DEBUG)
_parent_logger.addHandler(_file_handler)
_parent_logger.addHandler(_console_handler)
_parent_logger.propagate = False

logger = logging.getLogger(__name__)



def _country_matches(country_lower: str, entity: str) -> bool:
    """Word-boundary match for geo filter. Prevents false positives.

    - "india" matches "india", "indian" (demonym, len diff <=3)
    - "india" does NOT match "indonesian" (len diff >3)
    - Short entities like "in" don't match anything (len < 4)
    """
    if country_lower == entity:
        return True
    # Covers demonyms: india→indian, brazil→brazilian (len diff ≤3)
    if entity.startswith(country_lower) and len(entity) <= len(country_lower) + 3:
        return True
    # Short forms: only if entity is long enough to be meaningful
    if len(entity) >= 4 and country_lower.startswith(entity):
        return True
    return False


# ── Event-type embedding augmentation ────────────────────────────────────
# Breaks the "India business news gravity well" where all articles are close
# in embedding space.  Adding scaled one-hot event-type dimensions makes
# articles with DIFFERENT event types further apart before Leiden builds its
# k-NN graph.  "general" / unknown types get zero vectors (neutral — they
# cluster purely by content similarity).
#
# Math: after L2-norm, same-type pairs get a cosine boost of α²/(1+α²).
#   α=0.50 → +20pp boost;  α=0.35 → +11pp.  Articles in the compressed
#   0.6-0.9 cosine range benefit significantly from even 10-20pp separation.

_EVENT_NEUTRAL_TYPES = frozenset({"general", "unknown", ""})


def _augment_with_event_type(
    embeddings: np.ndarray,
    event_types: list,
    scale: float = 0.50,
) -> np.ndarray:
    """Concatenate scaled event-type one-hot vectors to article embeddings.

    Args:
        embeddings: (N, D) float32 array — original article embeddings.
        event_types: length-N list of event type strings per article.
        scale: Magnitude of one-hot dimensions.  Higher = more weight on
               event type vs. semantic similarity.

    Returns:
        (N, D + n_types) float32 array.  Articles with neutral event types
        get all-zero extra dimensions (pure embedding distance).
    """
    specific_types = sorted(set(
        t for t in event_types if t not in _EVENT_NEUTRAL_TYPES
    ))
    if not specific_types:
        return embeddings  # nothing to augment

    type_to_idx = {t: i for i, t in enumerate(specific_types)}
    n_types = len(specific_types)
    n_articles = embeddings.shape[0]

    onehot = np.zeros((n_articles, n_types), dtype=np.float32)
    for i, t in enumerate(event_types):
        idx = type_to_idx.get(t)
        if idx is not None:
            onehot[i, idx] = scale

    augmented = np.concatenate([embeddings, onehot], axis=1)
    logger.info(
        f"Event-type augmentation: {n_types} types, scale={scale}, "
        f"dim {embeddings.shape[1]}→{augmented.shape[1]}, "
        f"neutral={sum(1 for t in event_types if t in _EVENT_NEUTRAL_TYPES)}"
    )
    return augmented


# ── Entity fingerprint augmentation ─────────────────────────────────────
# Discriminates WITHIN the same event type.  Event-type augmentation
# separates "regulation" from "funding", but can't separate two different
# "crisis" stories (robbery vs corporate fraud).  Entity fingerprinting
# gives articles with shared ORG/PERSON/PRODUCT entities higher cosine
# similarity.  Uses feature hashing to fixed 32 dimensions.

from app.shared.stopwords import ENTITY_STOP as _ENTITY_FP_STOP


def _augment_with_entity_fingerprint(
    embeddings: np.ndarray,
    articles: list,
    n_buckets: int = 32,
    scale: float = 0.30,
) -> np.ndarray:
    """Add entity fingerprint dimensions for within-event-type discrimination.

    For each article, extracts ORG/PERSON/PRODUCT entity tokens, hashes them
    to `n_buckets` dimensions, and concatenates.  Articles sharing named
    entities get closer in augmented space.

    Args:
        embeddings: (N, D) array.
        articles: list of NewsArticle with entities.
        n_buckets: Size of hash fingerprint vector.
        scale: Magnitude of fingerprint dimensions.

    Returns:
        (N, D + n_buckets) array.
    """
    import hashlib
    n = embeddings.shape[0]
    fp = np.zeros((n, n_buckets), dtype=np.float32)

    # Compute IDF for entity tokens (rare entities matter more)
    token_doc_freq: Counter = Counter()
    article_tokens: list = []

    for a in articles:
        tokens = set()
        for ent in getattr(a, "entities", []) or []:
            etype = ent.type if hasattr(ent, "type") else (
                ent.get("type", "") if isinstance(ent, dict) else ""
            )
            name = ent.text if hasattr(ent, "text") else (
                ent.get("text", "") if isinstance(ent, dict) else str(ent)
            )
            if etype in ("ORG", "PERSON", "PRODUCT") and name:
                for tok in name.lower().split():
                    tok = tok.strip(".,;:!?'\"()-")
                    if len(tok) > 2 and tok not in _ENTITY_FP_STOP:
                        tokens.add(tok)
        article_tokens.append(tokens)
        token_doc_freq.update(tokens)

    if not token_doc_freq:
        return embeddings  # no entity tokens found

    # IDF: log(N / df).  Rare entities get high weight.
    idf = {}
    for tok, df in token_doc_freq.items():
        idf[tok] = math.log(max(n, 1) / max(df, 1))

    # Build fingerprint: hash each token to a bucket, weighted by IDF.
    for i, tokens in enumerate(article_tokens):
        for tok in tokens:
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % n_buckets
            fp[i, bucket] += idf.get(tok, 1.0)

    # Normalize fingerprints to unit length, then scale
    norms = np.linalg.norm(fp, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fp = fp / norms * scale

    augmented = np.concatenate([embeddings, fp], axis=1)
    entity_articles = sum(1 for t in article_tokens if t)
    logger.info(
        f"Entity fingerprint augmentation: {n_buckets} buckets, scale={scale}, "
        f"dim {embeddings.shape[1]}→{augmented.shape[1]}, "
        f"{entity_articles}/{n} articles have entity tokens, "
        f"{len(token_doc_freq)} unique tokens"
    )
    return augmented


# ── Hybrid similarity matrix ─────────────────────────────────────────
# Replaces augmentation-based approach for Leiden graph construction.
# Instead of concatenating sparse dimensions to 1024-dim embeddings
# (where they get drowned out), compute an explicit similarity matrix
# that blends semantic, lexical, categorical, and temporal signals.
#
# Research: BERTopic + Gibbs-BERTopic find 60% semantic + 40% lexical
# is optimal. We add event-type (10%) and temporal (5%) for news.

# Stop words for title vocabulary (same list as coherence.py _PW_STOP subset)
_HYBRID_TITLE_STOP = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "it", "by", "as", "be", "do", "so", "up", "if", "my", "no",
    "we", "he", "us", "am", "how", "who", "all", "its", "not", "but",
    "new", "has", "was", "are", "been", "will", "from", "with", "what",
    "that", "this", "have", "into", "also", "more", "than", "over",
    "says", "said", "amid", "may", "can", "now", "per", "via", "set",
    "get", "key", "top", "why", "after", "about", "could", "would",
    "india", "indian", "global", "world", "national",
    "digital", "technology", "tech", "system", "systems", "platform",
    "government", "policy", "programme", "scheme",
    "percent", "quarterly", "annual", "fiscal",
    "latest", "breaking", "update", "updates", "live", "watch", "read",
    "know", "things", "need", "five", "rush", "hour",
    "launch", "launches", "launched", "impact", "expected", "likely",
    "rise", "aims", "targets", "plans", "seeks", "moves", "hits", "sees",
    "business", "economy", "markets", "today", "limited", "ltd", "inc",
    "corp", "group", "pvt", "company", "report",
})


def _title_vocab(article) -> frozenset:
    """Extract stemmed, stop-filtered title vocabulary for an article."""
    title = article.title if hasattr(article, "title") else ""
    words = set()
    for w in title.lower().split():
        w = w.strip(".,;:!?'\"()-[]{}#@&*+=/\\|<>~`")
        if len(w) < 3 or w in _HYBRID_TITLE_STOP:
            continue
        # Lightweight suffix stemming (same as coherence.py)
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "i"
        elif w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ment") and len(w) > 6:
            w = w[:-4]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("ers") and len(w) > 4:
            w = w[:-1]
        elif w.endswith("es") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        if len(w) >= 3 and w not in _HYBRID_TITLE_STOP:
            words.add(w)

    # Also include entity names (ORG/PERSON/PRODUCT tokens)
    for ent_name in getattr(article, "entity_names", []) or []:
        for tok in ent_name.lower().split():
            tok = tok.strip(".,;:!?'\"()-")
            if len(tok) > 2 and tok not in _ENTITY_FP_STOP:
                words.add(tok)
    return frozenset(words)


def compute_hybrid_similarity(
    articles: list,
    embeddings: np.ndarray,
    w_semantic: float = 0.50,
    w_lexical: float = 0.30,
    w_event: float = 0.15,
    w_temporal: float = 0.05,
) -> np.ndarray:
    """Compute blended similarity matrix for Leiden graph construction.

    Combines four signals into a single N×N similarity matrix:
      - Semantic: MEAN-CENTERED embedding cosine (removes domain bias)
      - Lexical: TF-IDF cosine on title+entity words (NOT binary Jaccard)
      - Event-type: binary match for non-general event types (categorical)
      - Temporal: exponential decay based on publication time gap

    KEY FIX (Iteration 8): Mean-centering removes the "India news gravity well"
    where all same-region articles have 0.55-0.85 cosine similarity regardless
    of topic.  After centering, baseline drops to ~0.15-0.25, making topical
    differences actually visible to the clustering algorithm.

    TF-IDF cosine replaces binary Jaccard (which was avg=0.004 — essentially
    zero — because titles have few overlapping words).  TF-IDF properly
    weights rare discriminative terms (company names, product names) and
    downweights common domain vocabulary ("market", "sector", "growth").

    Args:
        articles: List of NewsArticle objects.
        embeddings: (N, D) array of article embeddings (NOT augmented).
        w_semantic: Weight for mean-centered embedding cosine similarity.
        w_lexical: Weight for TF-IDF cosine on title+entity words.
        w_event: Weight for event-type match.
        w_temporal: Weight for temporal proximity.

    Returns:
        (N, N) float32 symmetric similarity matrix in [0, 1].
    """
    from collections import Counter as _Counter

    n = len(articles)

    # ── 1. SEMANTIC: Mean-centered embedding cosine ──────────────────────
    # Subtract corpus mean to remove dominant "Indian business news" direction.
    # This is the single highest-ROI fix for the anisotropy problem.
    # REF: "All-but-the-Top" (Mu et al., ICLR 2018) — removing top principal
    # components from word embeddings improves isotropy and downstream tasks.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = embeddings / norms

    corpus_mean = emb_normed.mean(axis=0)
    emb_centered = emb_normed - corpus_mean
    center_norms = np.linalg.norm(emb_centered, axis=1, keepdims=True)
    center_norms[center_norms == 0] = 1.0
    emb_centered = emb_centered / center_norms

    sem_sim = emb_centered @ emb_centered.T
    np.clip(sem_sim, 0.0, 1.0, out=sem_sim)

    # ── 2. LEXICAL: TF-IDF cosine on title + entity words ───────────────
    # Binary Jaccard was avg=0.004 (broken). TF-IDF weights rare terms
    # (company names) high and common terms ("market") low.
    vocabs = [_title_vocab(a) for a in articles]

    # Build document-frequency counts
    df = _Counter()
    for vocab in vocabs:
        for word in vocab:
            df[word] += 1

    # TF-IDF vectors (sparse, then cosine)
    all_terms = sorted(df.keys())
    term_idx = {t: i for i, t in enumerate(all_terms)}
    n_terms = len(all_terms)

    if n_terms > 0:
        # Build TF-IDF matrix (N x n_terms)
        tfidf = np.zeros((n, n_terms), dtype=np.float32)
        for i, vocab in enumerate(vocabs):
            for word in vocab:
                if word in term_idx:
                    tf = 1.0  # binary TF (word present or not in short titles)
                    idf = math.log(n / (1 + df[word]))  # smooth IDF
                    tfidf[i, term_idx[word]] = tf * idf

        # L2 normalize for cosine similarity
        tfidf_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf_norms[tfidf_norms == 0] = 1.0
        tfidf_normed = tfidf / tfidf_norms

        lex_sim = tfidf_normed @ tfidf_normed.T
        np.clip(lex_sim, 0.0, 1.0, out=lex_sim)
    else:
        lex_sim = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(lex_sim, 1.0)

    # ── 3. EVENT-TYPE: Match + domain coherence (V12) ────────────────────
    # Full match (same specific type): 1.0
    # Same domain, different type: 0.4 (partial coherence)
    # Different domain: 0.0
    from app.news.event_classifier import EVENT_TYPE_TO_DOMAIN
    event_types = [
        getattr(a, "_trigger_event", "general") or "general" for a in articles
    ]
    domains = [EVENT_TYPE_TO_DOMAIN.get(et, "unknown") for et in event_types]
    evt_sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = event_types[i], event_types[j]
            d1, d2 = domains[i], domains[j]
            if e1 == e2 and e1 not in _EVENT_NEUTRAL_TYPES:
                evt_sim[i, j] = 1.0
                evt_sim[j, i] = 1.0
            elif d1 == d2 and d1 not in ("unknown", "noise"):
                # Same domain but different type: partial similarity
                evt_sim[i, j] = 0.4
                evt_sim[j, i] = 0.4
    np.fill_diagonal(evt_sim, 1.0)

    # ── 4. TEMPORAL: Exponential decay (48h half-life) ───────────────────
    timestamps = []
    for a in articles:
        pub = getattr(a, "published_at", None)
        if pub is not None:
            timestamps.append(pub.timestamp() if hasattr(pub, "timestamp") else 0.0)
        else:
            timestamps.append(0.0)
    timestamps = np.array(timestamps, dtype=np.float64)

    temp_sim = np.ones((n, n), dtype=np.float32)
    if timestamps.max() > 0:
        for i in range(n):
            for j in range(i + 1, n):
                hours_apart = abs(timestamps[i] - timestamps[j]) / 3600.0
                decay = math.exp(-hours_apart / 48.0)
                temp_sim[i, j] = decay
                temp_sim[j, i] = decay

    # ── 5. SAME-SOURCE PENALTY ────────────────────────────────────────────
    # Inspired by nyan (NyanNyanovich/nyan, 285★): articles from the same
    # RSS feed share writing style, vocabulary, and editorial perspective,
    # which inflates their similarity regardless of topical relevance.
    # Applying a multiplicative penalty prevents same-source clustering.
    #
    # This is one of the highest-ROI fixes identified in the Iteration 9
    # open-source research: nyan uses it in production for their Telegram
    # news channel aggregator.
    same_source_penalty = 0.70  # Multiply same-source similarity by 0.70
    source_ids = [getattr(a, "source_id", "") or getattr(a, "source_name", "") or "" for a in articles]
    source_sim = np.ones((n, n), dtype=np.float32)
    same_source_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if source_ids[i] and source_ids[j] and source_ids[i] == source_ids[j]:
                source_sim[i, j] = same_source_penalty
                source_sim[j, i] = same_source_penalty
                same_source_pairs += 1

    # ── Blend all signals ────────────────────────────────────────────────
    hybrid = (
        w_semantic * sem_sim.astype(np.float32)
        + w_lexical * lex_sim
        + w_event * evt_sim
        + w_temporal * temp_sim
    )
    # Apply same-source penalty as a multiplicative factor (not additive)
    # This preserves the relative ordering of signals but reduces absolute
    # similarity for same-source pairs.
    hybrid *= source_sim
    np.fill_diagonal(hybrid, 1.0)

    triu = np.triu_indices(n, k=1)
    logger.info(
        f"Hybrid similarity: {n}×{n} matrix, "
        f"weights=(sem={w_semantic}, lex={w_lexical}, evt={w_event}, temp={w_temporal}), "
        f"avg_sem={sem_sim[triu].mean():.3f} (mean-centered), "
        f"avg_lex={lex_sim[triu].mean():.3f} (TF-IDF cosine, {n_terms} terms), "
        f"avg_evt={evt_sim[triu].mean():.3f}, "
        f"same_source_pairs={same_source_pairs} (penalty={same_source_penalty}), "
        f"avg_hybrid={hybrid[triu].mean():.3f}"
    )

    return hybrid


class TrendPipeline:
    """Layered trend detection pipeline: articles → TrendTree.

    Layers:
      1. ingest:      Scrape, dedup, NER, embed, filter → (articles, embeddings)
      2. cluster:     Leiden + coherence + keywords → cluster data
      3. relate:      Causal graph construction (stub — Phase 3)
      4. temporalize: Trend memory comparison (stub — Phase 4)
      5. enrich:      Signals, synthesis, tree assembly → TrendTree
    """

    def __init__(
        self,
        dedup_threshold: float = 0.25,
        dedup_num_perm: int = 128,
        dedup_shingle_size: int = 2,
        semantic_dedup_threshold: float = 0.88,
        spacy_model: str = "en_core_web_sm",
        min_cluster_size: int = 5,
        max_depth: int = 3,
        max_concurrent_llm: int = 6,
        subcluster_min_coherence: float = 0.40,
        subcluster_min_differentiation: float = 0.15,
        subcluster_min_articles: int = 4,
        llm_tool=None,
        mock_mode: bool = False,
        country: str = "",
        domestic_source_ids: Optional[set] = None,
    ):
        self.deduplicator = ArticleDeduplicator(
            threshold=dedup_threshold, num_perm=dedup_num_perm,
            shingle_size=dedup_shingle_size,
        )
        self.entity_extractor = EntityExtractor(model_name=spacy_model)
        self.embedding_tool = EmbeddingTool()
        self.keyword_extractor = KeywordExtractor()

        self.min_cluster_size = min_cluster_size
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.max_depth = max_depth
        self.max_concurrent_llm = max_concurrent_llm
        self.min_subcluster_size = max(min_cluster_size, 8)
        self.subcluster_min_coherence = subcluster_min_coherence
        self.subcluster_min_differentiation = subcluster_min_differentiation
        self.subcluster_min_articles = subcluster_min_articles

        self._llm_tool = llm_tool
        self.mock_mode = mock_mode
        self.metrics: Dict[str, Any] = {"phase_times": {}, "article_counts": {}}

        # Load config (overridable via .env), then overlay adaptive EMA thresholds
        from app.config import get_settings
        self.settings = get_settings()
        _s = self.settings
        self.coherence_min = _s.coherence_min
        self.coherence_reject = _s.coherence_reject
        self.merge_threshold = _s.merge_threshold
        self.cmi_relevance_threshold = _s.cmi_relevance_threshold
        self.cmi_hard_floor = _s.cmi_hard_floor

        # Phase 5: Adaptive EMA thresholds override config defaults
        try:
            from app.learning.pipeline_metrics import (
                compute_adaptive_thresholds, load_history,
            )
            adapted = compute_adaptive_thresholds()
            if adapted:
                if "coherence_min" in adapted:
                    self.coherence_min = adapted["coherence_min"]
                if "coherence_reject" in adapted:
                    self.coherence_reject = adapted["coherence_reject"]
                if "merge_threshold" in adapted:
                    self.merge_threshold = adapted["merge_threshold"]
                self.metrics["adaptive_thresholds"] = adapted

            # CSI-driven stabilization: if cluster stability < 0.5,
            # loosen thresholds to reduce parameter oscillation
            recent = load_history(last_n=1)
            if recent:
                csi = recent[-1].get("cluster_stability_index", 1.0)
                if 0 <= csi < 0.5:
                    loosen_factor = 0.90  # 10% more permissive
                    self.coherence_min *= loosen_factor
                    self.coherence_reject *= loosen_factor
                    logger.warning(
                        f"CSI={csi:.3f} (unstable) — loosening coherence "
                        f"thresholds by 10% to reduce oscillation"
                    )
                    self.metrics["csi_stabilization_applied"] = True
        except Exception as e:
            logger.debug(f"Adaptive threshold loading skipped: {e}")

        # Log active weights (may be learned from feedback)
        try:
            from app.trends.signals.composite import _get_weights
            self.metrics["active_actionability_weights"] = _get_weights()
        except Exception:
            pass

        # Leiden clustering params
        self.leiden_k = _s.leiden_k
        self.leiden_resolution = _s.leiden_resolution
        self.leiden_auto_resolution = _s.leiden_auto_resolution
        self.leiden_min_community_size = _s.leiden_min_community_size
        self.leiden_seed = _s.leiden_seed

        # Geographic relevance filter (dynamic, entity-based)
        self._target_country = country
        self._domestic_source_ids = domestic_source_ids or set()

    @property
    def llm_tool(self):
        if self._llm_tool is None:
            from app.tools.llm_service import LLMService
            self._llm_tool = LLMService(mock_mode=self.mock_mode)
        return self._llm_tool

    # ════════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ════════════════════════════════════════════════════════════════════

    async def run(
        self,
        articles: List[NewsArticle],
        use_cache: bool = False,
        cache_path: str = "./data/article_cache",
    ) -> TrendTree:
        """Execute the full layered pipeline: articles → TrendTree.

        Layers:
          1. ingest  → (articles, embeddings)
          2. cluster → (cluster_articles, cluster_keywords, cluster_signals, ...)
          3. relate  → correlation edges, cascades, entity bridges
          4. temporalize → trend memory, novelty/continuity scores
          5. enrich  → TrendTree (synthesis, validation, sub-clustering)
          6. causal  → multi-agent causal reasoning (Stage D Council)

        Args:
            articles: Input articles (ignored if use_cache=True and cache exists).
            use_cache: If True, load articles + embeddings from ChromaDB cache.
            cache_path: Path to ChromaDB persistent directory.
        """
        total_start = time.time()
        self.metrics = {"phase_times": {}, "article_counts": {"input": len(articles)}}
        offline = getattr(self.settings, 'offline_mode', False)
        llm_label = f"Ollama/{self.settings.ollama_model}" if offline else "cloud (Gemini/NVIDIA)"
        embed_label = f"local/{self.settings.local_embedding_model}" if self.settings.embedding_provider == 'local' else self.settings.embedding_provider
        logger.info(f"=== Pipeline START | {len(articles)} articles | LLM={llm_label} | Embed={embed_label} | offline={offline} ===")

        if not articles and not use_cache:
            logger.warning("No articles provided")
            return TrendTree(root_ids=[], nodes={})

        # All timeouts from config — no hardcoded values
        _t = self.settings

        # ── Layer 1: Ingest ──────────────────────────────────────────────
        try:
            articles, embeddings = await asyncio.wait_for(
                self._layer_ingest(
                    articles, use_cache=use_cache, cache_path=cache_path,
                ),
                timeout=_t.engine_event_class_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("[TIMEOUT] Layer 1 (ingest) timeout")
            raise TimeoutError("Embedding generation timeout")

        # ── Layer 2: Cluster ─────────────────────────────────────────────
        try:
            cluster_data = asyncio.wait_for(
                asyncio.to_thread(self._layer_cluster, articles, embeddings),
                timeout=_t.engine_clustering_timeout,
            ) if asyncio.iscoroutinefunction(self._layer_cluster) else self._layer_cluster(articles, embeddings)
        except (asyncio.TimeoutError, TypeError):
            cluster_data = self._layer_cluster(articles, embeddings)

        logger.info(f"Clustering complete: {len(cluster_data.get('cluster_articles', {}))} clusters")

        # ── Layer 2.5: Post-Cluster Enrichment ─────────────────────────
        cluster_data = self._layer_post_cluster_enrich(cluster_data, articles)

        # ── Layer 2.75: LLM Cluster Validation ───────────────────────
        try:
            cluster_data = await asyncio.wait_for(
                self._layer_llm_cluster_validation(cluster_data, articles),
                timeout=_t.engine_validation_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[TIMEOUT] Layer 2.75 (LLM validation) — skipping")

        # ── Layer 3: Relate ───────────────────────────────────────────────
        cluster_data = self._layer_relate(cluster_data)

        # ── Layer 4: Temporalize ──────────────────────────────────────────
        cluster_data = self._layer_temporalize(cluster_data)

        # ── Layer 5: Enrich ──────────────────────────────────────────────
        try:
            tree = await asyncio.wait_for(
                self._layer_enrich(cluster_data, articles),
                timeout=_t.engine_synthesis_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[TIMEOUT] Layer 5 (enrich) — creating minimal tree")
            from app.schemas.trends import TrendTree
            minimal_nodes = {
                cid: {
                    "id": cid,
                    "trend_title": f"Cluster {cid[:8]}",
                    "articles": cluster_data.get("cluster_articles", {}).get(cid, []),
                    "severity": 0.5,
                }
                for cid in list(cluster_data.get("cluster_articles", {}).keys())[:5]
            }
            tree = TrendTree(root_ids=list(minimal_nodes.keys()), nodes=minimal_nodes)

        # ── Layer 6: Cross-trend causal council (nice-to-have) ────────
        try:
            tree = await asyncio.wait_for(
                self._layer_causal_reasoning(tree, cluster_data, articles),
                timeout=_t.engine_causal_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[TIMEOUT] Layer 6 (causal) — continuing without causal edges")

        total_time = time.time() - total_start
        self.metrics["total_seconds"] = round(total_time, 2)
        self.metrics["clustering_method"] = "leiden"
        logger.info(
            f"Pipeline complete: {len(articles)} articles → "
            f"{len(tree.root_ids)} trends, {len(tree.nodes)} nodes "
            f"(depth {tree.max_depth_reached}) in {total_time:.1f}s"
        )

        # V4: Post-synthesis OSS update to trend memory
        # (trend memory stores centroids BEFORE synthesis, OSS comes AFTER)
        self._update_memory_oss(cluster_data)

        # Source quality bandit: update posteriors from this run
        self._update_source_bandit(cluster_data, articles)

        # Taxonomy evolution: discover new event categories from unmatched articles
        self._discover_taxonomy(cluster_data, articles)

        # Auto-generate feedback for weight learner (closes self-learning loop)
        self._auto_generate_feedback(cluster_data)

        # Record metrics for calibration + drift detection
        # (after auto-feedback so auto_feedback counts are captured)
        self._record_pipeline_metrics(cluster_data)

        # V12: Self-evolving anchors — save high-confidence classifications
        # as future anchor descriptions for the event classifier
        self._evolve_classifier_anchors(articles)

        return tree

    # ════════════════════════════════════════════════════════════════════
    # LAYER IMPLEMENTATIONS
    # ════════════════════════════════════════════════════════════════════

    async def _layer_ingest(
        self,
        articles: List[NewsArticle],
        use_cache: bool = False,
        cache_path: str = "./data/article_cache",
    ) -> tuple:
        """Layer 1: Ingest — fetch, scrape, dedup, NER, embed, filter.

        Contract: List[NewsArticle] → (List[NewsArticle], embeddings)
        """
        embeddings = None

        # Cache fast path: skip fetch/scrape/event/dedup/NER/embed
        if use_cache:
            from app.tools.article_cache import ArticleCache
            cache = ArticleCache(cache_path)
            stats = cache.get_stats()
            if stats["count"] > 0:
                articles, embeddings = cache.load_articles()
                logger.info(
                    f"Cache loaded: {len(articles)} articles with embeddings "
                    f"(skipped fetch/scrape/embed)"
                )
                self.metrics["article_counts"]["input"] = len(articles)
                self.metrics["article_counts"]["from_cache"] = len(articles)
                for article, emb in zip(articles, embeddings):
                    article.title_embedding = emb

                # Event classification + NER on cached articles (needed for
                # enrichment layer: entity coherence, company scoring, event
                # concentration validation, cross-cluster linking).
                # These are fast (~2-5s total) and essential for quality.
                if articles:
                    needs_events = not getattr(articles[0], '_trigger_event', None)
                    needs_ner = not getattr(articles[0], 'entity_names', None)

                    if needs_events:
                        await asyncio.to_thread(
                            self._phase_classify_events, articles
                        )
                        # Tier 2 LLM reclassification for ambiguous articles
                        await self._phase_classify_events_tier2()
                    if needs_ner:
                        articles = self._phase_ner(articles)

                # Title-exact dedup: cache may have duplicate entries
                seen_titles: dict[str, int] = {}
                title_dedup_articles = []
                title_dedup_embeddings = []
                for a, e in zip(articles, embeddings):
                    key = (a.title.strip().lower(), a.source_id or "")
                    if key not in seen_titles:
                        seen_titles[key] = 0
                        title_dedup_articles.append(a)
                        title_dedup_embeddings.append(e)
                    else:
                        seen_titles[key] += 1
                title_dupes = len(articles) - len(title_dedup_articles)
                if title_dupes:
                    logger.info(
                        f"Title-exact dedup: {len(articles)} → {len(title_dedup_articles)} "
                        f"({title_dupes} duplicates removed)"
                    )
                articles = title_dedup_articles
                embeddings = title_dedup_embeddings

                # Apply quality filters even on cached articles
                articles = self._phase_noise_filter(articles)
                articles = await self._phase_article_triage(articles)
                articles = self._phase_language_filter(articles)
                # Rebuild embeddings from surviving articles' title_embedding
                embeddings = [a.title_embedding for a in articles]

                articles, embeddings = self._phase_semantic_dedup(
                    articles, embeddings, threshold=self.semantic_dedup_threshold
                )
                self.metrics["article_counts"]["after_semantic_dedup"] = len(articles)
                articles, embeddings = self._phase_cmi_relevance(articles, embeddings)

        if embeddings is None:
            # Source bandit: prioritize articles from high-quality sources
            articles = self._apply_source_bandit_priority(articles)

            # Normal path: full ingest pipeline
            # Scrape FIRST — classify needs article.content to be populated
            articles = await self._phase_scrape(articles)
            await asyncio.to_thread(self._phase_classify_events, articles)

            await self._phase_classify_events_tier2()

            articles = self._phase_noise_filter(articles)
            articles = await self._phase_article_triage(articles)
            articles = self._phase_language_filter(articles)
            articles = self._phase_dedup(articles)
            articles = self._phase_ner(articles)
            articles = self._phase_geo_filter(articles)

            embeddings = self._phase_embed(articles)
            articles, embeddings = self._phase_semantic_dedup(
                articles, embeddings, threshold=self.semantic_dedup_threshold
            )
            self.metrics["article_counts"]["after_semantic_dedup"] = len(articles)

            articles, embeddings = self._phase_cmi_relevance(articles, embeddings)

            if use_cache:
                from app.tools.article_cache import ArticleCache
                cache = ArticleCache(cache_path)
                stored = cache.store_articles(articles, embeddings)
                logger.info(f"Cached {stored} articles with embeddings for future runs")

        self._all_articles = articles
        return articles, embeddings

    def _layer_cluster(
        self,
        articles: List[NewsArticle],
        embeddings,
    ) -> Dict[str, Any]:
        """Layer 2: Cluster — Leiden + coherence validation + keywords + signals.

        Contract: (articles, embeddings) → cluster_data dict
        """
        labels, noise_count = self._phase_cluster_leiden(articles, embeddings)

        # Coherence validation (original 1024-dim space)
        cluster_articles = self._group_by_cluster(articles, labels)
        cluster_articles, labels, noise_delta = self._phase_coherence_validation(
            cluster_articles, embeddings, articles, labels
        )
        noise_count += noise_delta

        cluster_keywords = self._phase_keywords(cluster_articles)

        # Signal computation + cluster quality
        cluster_signals = self._phase_signals(cluster_articles)
        self._compute_cluster_quality(
            cluster_articles, cluster_signals, embeddings, articles, labels
        )

        # Recompute cluster quality with real coherence values
        from app.trends.signals.composite import (
            compute_cluster_quality_score, compute_confidence_score,
        )
        for cid, sigs in cluster_signals.items():
            sigs["cluster_quality_score"] = compute_cluster_quality_score(sigs)
            sigs["confidence"] = compute_confidence_score(sigs)

        return {
            "cluster_articles": cluster_articles,
            "cluster_keywords": cluster_keywords,
            "cluster_signals": cluster_signals,
            "labels": labels,
            "noise_count": noise_count,
            "embeddings": embeddings,
        }

    def _layer_post_cluster_enrich(
        self,
        cluster_data: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> Dict[str, Any]:
        """Layer 2.5: Post-Cluster Enrichment — entity consolidation + company scoring.

        Deterministic, fast (no LLM). Produces structured intelligence per cluster:
        - Entity consolidation across cluster members
        - Company activity scoring (mention frequency × recency × role weight)
        - Subject vs mention classification for each ORG entity
        - Cluster validation (entity coherence, source diversity)
        - Cross-cluster entity linking

        This data feeds into synthesis prompt as structured context, replacing
        the current approach where the LLM must extract entities from raw text.

        Architecture: JRC/EMM "cluster-level information fusion" pattern.
        REF: data/research_enrichment_layer.md
        """
        t = time.time()

        try:
            from app.trends.enrichment import (
                enrich_all_clusters, enrich_cluster,
            )

            labels = cluster_data["labels"]
            cluster_articles = cluster_data["cluster_articles"]

            # Enrich each cluster
            enrichments, cross_cluster = enrich_all_clusters(
                labels, articles, activity_window_days=5.0,
            )

            # Store enrichment data in cluster_data for downstream use
            cluster_data["enrichments"] = enrichments
            cluster_data["cross_cluster_companies"] = cross_cluster

            # Inject enrichment summaries into cluster signals
            enrichment_by_id = {e["cluster_id"]: e for e in enrichments}
            for cid, signals in cluster_data["cluster_signals"].items():
                enrichment = enrichment_by_id.get(cid)
                if enrichment:
                    signals["primary_companies"] = [
                        c["name"] for c in enrichment["primary_companies"][:5]
                    ]
                    signals["mentioned_companies"] = [
                        c["name"] for c in enrichment["mentioned_companies"][:5]
                    ]
                    signals["entity_coherence"] = enrichment["validation"]["entity_coherence"]
                    signals["cluster_valid"] = enrichment["validation"]["is_valid"]

            # ── Reject invalid clusters (precision-first approach) ──
            # NewsCatcher rejects 80% of clusters for precision. We reject
            # only those that fail validation — entity scatter + event scatter.
            rejected_ids = set()
            for e in enrichments:
                if not e["validation"]["is_valid"]:
                    cid = e["cluster_id"]
                    rejected_ids.add(cid)
                    reasons = ", ".join(e["validation"]["rejection_reasons"])
                    logger.info(
                        f"Cluster {cid} REJECTED by enrichment: {reasons} "
                        f"({e['article_count']} articles)"
                    )

            if rejected_ids:
                # Remove rejected clusters from cluster_signals and
                # cluster_articles. Articles go back to noise.
                for cid in rejected_ids:
                    cluster_data["cluster_signals"].pop(cid, None)
                    cluster_data["cluster_articles"].pop(cid, None)
                logger.info(
                    f"Enrichment rejected {len(rejected_ids)} clusters: "
                    f"{sorted(rejected_ids)}"
                )

            cluster_data["rejected_cluster_ids"] = sorted(rejected_ids)

            # Metrics
            total_companies = sum(
                len(e["companies"]) for e in enrichments
            )
            total_primary = sum(
                len(e["primary_companies"]) for e in enrichments
            )
            valid_clusters = sum(
                1 for e in enrichments if e["validation"]["is_valid"]
            )
            multi_cluster = sum(
                1 for v in cross_cluster.values() if v["n_clusters"] > 1
            )

            elapsed = time.time() - t
            self.metrics["phase_times"]["post_enrichment"] = round(elapsed, 3)
            self.metrics["enrichment"] = {
                "total_entities": sum(len(e["entities"]) for e in enrichments),
                "total_companies": total_companies,
                "primary_companies": total_primary,
                "valid_clusters": valid_clusters,
                "rejected_clusters": len(rejected_ids),
                "total_clusters": len(enrichments),
                "multi_cluster_companies": multi_cluster,
            }

            logger.info(
                f"Post-cluster enrichment: {len(enrichments)} clusters, "
                f"{total_companies} companies ({total_primary} primary), "
                f"{valid_clusters}/{len(enrichments)} valid, "
                f"{multi_cluster} multi-cluster companies, "
                f"{elapsed:.2f}s"
            )

        except Exception as e:
            logger.warning(f"Post-cluster enrichment skipped: {e}")
            cluster_data["enrichments"] = []
            cluster_data["cross_cluster_companies"] = {}
            elapsed = time.time() - t
            self.metrics["phase_times"]["post_enrichment"] = round(elapsed, 3)

        return cluster_data

    async def _layer_llm_cluster_validation(
        self,
        cluster_data: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> Dict[str, Any]:
        """Layer 2.75: LLM Cluster Validation — curriculum learning cascade.

        Tier 1 (deterministic, Layer 2.5) already auto-rejected obvious garbage.
        This layer sends surviving clusters to LLM for higher-confidence validation.

        Clusters rejected by LLM (is_coherent=False AND score < 0.35) are removed.
        Clusters flagged for splitting (should_split=True) are noted for future use.
        Outlier articles identified by LLM are ejected from their clusters.

        Cost: ~$0.0003/cluster with DeepSeek/GPT-4o-mini. ~$0.01/pipeline run.
        REF: NewsCatcher rejects 80% of clusters via LLM validation.
        """
        t = time.time()

        try:
            from app.trends.enrichment import validate_all_clusters_llm
            from app.tools.llm_service import LLMService

            enrichments = cluster_data.get("enrichments", [])
            cluster_articles = cluster_data.get("cluster_articles", {})

            if not enrichments or not cluster_articles:
                logger.info("LLM cluster validation skipped: no enrichments")
                return cluster_data

            llm_service = LLMService(lite=True)
            llm_validations = await validate_all_clusters_llm(
                enrichments=enrichments,
                cluster_articles=cluster_articles,
                llm_service=llm_service,
            )

            if not llm_validations:
                logger.info("LLM cluster validation: no clusters validated")
                return cluster_data

            # ── Apply LLM decisions ──────────────────────────────────────
            llm_rejected_ids = set()
            outlier_ejections = 0

            for cid, val in llm_validations.items():
                c_arts = cluster_articles.get(cid, [])

                # Eject outlier articles identified by LLM FIRST
                # (even for incoherent clusters — salvage the core)
                if val["outlier_indices"]:
                    valid_outliers = [
                        idx for idx in val["outlier_indices"]
                        if 0 <= idx < len(c_arts)
                    ]
                    if valid_outliers and len(c_arts) - len(valid_outliers) >= 2:
                        new_arts = [
                            a for i, a in enumerate(c_arts)
                            if i not in set(valid_outliers)
                        ]
                        cluster_data["cluster_articles"][cid] = new_arts
                        c_arts = new_arts  # Update for size check below
                        outlier_ejections += len(valid_outliers)
                        logger.info(
                            f"Cluster {cid}: ejected {len(valid_outliers)} "
                            f"outlier articles (LLM), {len(new_arts)} remain"
                        )

                # Reject only truly incoherent clusters (score < 0.20)
                # Clusters with 0.20-0.40 that have outliers removed may now be clean
                if not val["is_coherent"] and val["coherence_score"] < 0.20 and len(c_arts) >= 2:
                    llm_rejected_ids.add(cid)
                    logger.info(
                        f"Cluster {cid} REJECTED by LLM: "
                        f"score={val['coherence_score']:.2f}, "
                        f"reason='{val['reasoning'][:120]}'"
                    )
                    continue

                # Store LLM label for downstream use
                if val.get("suggested_label"):
                    signals = cluster_data["cluster_signals"].get(cid, {})
                    signals["llm_label"] = val["suggested_label"]
                    signals["llm_coherence_score"] = val["coherence_score"]

            # Remove LLM-rejected clusters
            if llm_rejected_ids:
                for cid in llm_rejected_ids:
                    cluster_data["cluster_signals"].pop(cid, None)
                    cluster_data["cluster_articles"].pop(cid, None)
                logger.info(
                    f"LLM validation rejected {len(llm_rejected_ids)} clusters: "
                    f"{sorted(llm_rejected_ids)}"
                )

            # Merge with existing rejections
            prev_rejected = set(cluster_data.get("rejected_cluster_ids", []))
            cluster_data["rejected_cluster_ids"] = sorted(
                prev_rejected | llm_rejected_ids
            )

            # Metrics
            elapsed = time.time() - t
            self.metrics["phase_times"]["llm_cluster_validation"] = round(elapsed, 3)
            self.metrics["llm_validation"] = {
                "clusters_validated": len(llm_validations),
                "clusters_rejected": len(llm_rejected_ids),
                "outlier_ejections": outlier_ejections,
                "avg_coherence_score": round(
                    sum(v["coherence_score"] for v in llm_validations.values())
                    / max(len(llm_validations), 1), 3
                ),
                "split_candidates": sum(
                    1 for v in llm_validations.values() if v["should_split"]
                ),
            }

            logger.info(
                f"LLM cluster validation: {len(llm_validations)} validated, "
                f"{len(llm_rejected_ids)} rejected, "
                f"{outlier_ejections} outliers ejected, "
                f"{elapsed:.1f}s"
            )

        except Exception as e:
            logger.warning(f"LLM cluster validation skipped: {e}")
            elapsed = time.time() - t
            self.metrics["phase_times"]["llm_cluster_validation"] = round(elapsed, 3)
            self.metrics["llm_validation"] = {"error": str(e)}

        return cluster_data

    def _layer_relate(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 3: Relate — cross-trend correlation via entity bridges.

        Detects entity bridges (IDF-weighted), sector chains, temporal lag,
        and cascade paths between clusters.

        Contract: cluster_data → cluster_data (with edges, cascades added)
        """
        try:
            from app.trends.correlation import find_correlations

            # Compute cluster centroids for semantic similarity correlation
            _centroids = {}
            _embs = cluster_data.get("embeddings")
            _labels = cluster_data.get("labels")
            if _embs is not None and _labels is not None:
                _emb_arr = np.array(_embs, dtype=np.float32) if not isinstance(_embs, np.ndarray) else _embs
                for cid in cluster_data["cluster_articles"]:
                    mask = _labels == cid
                    if mask.any():
                        _centroids[cid] = _emb_arr[mask].mean(axis=0)

            edges, cascades, bridges = find_correlations(
                cluster_articles=cluster_data["cluster_articles"],
                cluster_signals=cluster_data["cluster_signals"],
                cluster_centroids=_centroids if _centroids else None,
            )

            cluster_data["trend_edges"] = edges
            cluster_data["cascades"] = cascades
            cluster_data["bridge_entities"] = bridges

            # Tag cascade clusters for higher priority in scoring
            cascade_clusters = set()
            for cascade in cascades:
                cascade_clusters.update(cascade)
            for cid in cascade_clusters:
                sigs = cluster_data["cluster_signals"].get(cid)
                if sigs:
                    sigs["cascade_count"] = sum(
                        1 for c in cascades if cid in c
                    )

            self.metrics["correlation"] = {
                "edges": len(edges),
                "cascades": len(cascades),
                "bridge_entity_pairs": len(bridges),
            }

        except Exception as e:
            logger.warning(f"Correlation layer skipped: {e}")
            cluster_data["trend_edges"] = []
            cluster_data["cascades"] = []
            cluster_data["bridge_entities"] = {}

        return cluster_data

    def _layer_temporalize(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 4: Temporalize — trend memory comparison.

        Compares current cluster centroids against stored centroids in
        ChromaDB. Computes novelty, continuity, and lifecycle stage
        for each cluster.

        Contract: cluster_data → cluster_data (with temporal scores added)
        """
        try:
            from app.trends.memory import TrendMemory

            # Compute centroids for each cluster
            embeddings = cluster_data["embeddings"]
            labels = cluster_data["labels"]
            emb_array = np.array(embeddings, dtype=np.float32) if not isinstance(embeddings, np.ndarray) else embeddings

            cluster_centroids = {}
            cluster_article_counts = {}
            for cid, arts in cluster_data["cluster_articles"].items():
                mask = labels == cid
                if mask.any():
                    cluster_centroids[cid] = emb_array[mask].mean(axis=0)
                cluster_article_counts[cid] = len(arts)

            if not cluster_centroids:
                cluster_data["novelty_scores"] = {}
                cluster_data["continuity_scores"] = {}
                cluster_data["lifecycle_stages"] = {}
                return cluster_data

            memory = TrendMemory()

            # Prune stale centroids
            pruned = memory.prune_stale()
            if pruned:
                logger.info(f"Pruned {pruned} stale trend centroids")

            novelty, continuity, lifecycle = memory.compute_novelty(
                cluster_centroids,
                cluster_keywords=cluster_data.get("cluster_keywords"),
                cluster_article_counts=cluster_article_counts,
            )

            # Inject temporal scores into cluster signals
            for cid, signals in cluster_data["cluster_signals"].items():
                signals["novelty_score"] = novelty.get(cid, 1.0)
                signals["continuity_score"] = continuity.get(cid, 0.0)
                signals["lifecycle_stage"] = lifecycle.get(cid, "birth")

            cluster_data["novelty_scores"] = novelty
            cluster_data["continuity_scores"] = continuity
            cluster_data["lifecycle_stages"] = lifecycle

            self.metrics["memory"] = {
                "stored_centroids": memory.stored_count,
                "avg_novelty": round(
                    sum(novelty.values()) / max(len(novelty), 1), 3
                ),
                "lifecycle_counts": dict(Counter(lifecycle.values())),
            }

            logger.info(
                f"Temporal memory: {len(novelty)} clusters scored, "
                f"avg_novelty={self.metrics['memory']['avg_novelty']:.3f}, "
                f"lifecycle={dict(Counter(lifecycle.values()))}, "
                f"{memory.stored_count} stored centroids"
            )

        except Exception as e:
            logger.warning(f"Temporal memory layer skipped: {e}")
            cluster_data["novelty_scores"] = {}
            cluster_data["continuity_scores"] = {}
            cluster_data["lifecycle_stages"] = {}

        return cluster_data

    async def _layer_enrich(
        self,
        cluster_data: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> TrendTree:
        """Layer 5: Enrich — synthesis, quality gate, validation, tree assembly.

        Contract: cluster_data → TrendTree
        """
        cluster_articles = cluster_data["cluster_articles"]
        cluster_keywords = cluster_data["cluster_keywords"]
        cluster_signals = cluster_data["cluster_signals"]
        noise_count = cluster_data["noise_count"]

        # V4: Cluster-level CMI relevance check — dissolve low-relevance clusters
        cmi_dissolved = 0
        for cid in list(cluster_articles.keys()):
            arts = cluster_articles[cid]
            cmi_scores = [
                getattr(a, '_cmi_relevance_score', 0.5) for a in arts
            ]
            avg_cmi = sum(cmi_scores) / max(len(cmi_scores), 1)
            if avg_cmi < 0.30 and len(arts) > 0:
                logger.info(
                    f"  V4: Dissolving cluster {cid} — avg CMI={avg_cmi:.3f} < 0.30 "
                    f"({len(arts)} articles)"
                )
                del cluster_articles[cid]
                if cid in cluster_keywords:
                    del cluster_keywords[cid]
                if cid in cluster_signals:
                    del cluster_signals[cid]
                cmi_dissolved += 1
        if cmi_dissolved:
            self.metrics["cmi_dissolved_clusters"] = cmi_dissolved
            logger.info(f"  V4: {cmi_dissolved} clusters dissolved (avg CMI < 0.30)")

        # LLM synthesis + quality gate
        cluster_summaries_raw = await self._phase_synthesize(
            cluster_articles, cluster_keywords
        )
        cluster_summaries = self._phase_quality_gate(
            cluster_summaries_raw, cluster_signals
        )

        # Store summaries in cluster_data for post-run hooks
        # (auto-feedback, taxonomy discovery, etc.)
        cluster_data["summaries"] = cluster_summaries

        # V4: Update quality reports with OSS from synthesis
        try:
            from app.trends.coherence import update_quality_with_oss
            oss_updated = 0
            for cid, synth in cluster_summaries.items():
                if not synth:
                    continue
                oss = synth.get("_oss")
                if oss is None:
                    continue
                report = cluster_signals.get(cid, {}).get("cluster_quality")
                if report:
                    updated_report = update_quality_with_oss(report, oss)
                    cluster_signals[cid]["cluster_quality"] = updated_report
                    oss_updated += 1
            if oss_updated:
                logger.info(f"  V4: Updated quality grades with OSS for {oss_updated} clusters")
        except Exception as e:
            logger.debug(f"OSS quality update skipped: {e}")

        # Log synthesis outcomes (which clusters failed and why)
        self._log_synthesis_outcomes(
            cluster_summaries_raw, cluster_summaries, cluster_signals
        )

        # V4: Post-synthesis grade gate — drop grade-F and very-low-OSS clusters
        # This is a SECOND validation pass that uses OSS-updated grades.
        # Runs AFTER OSS quality update so grades reflect synthesis specificity.
        pre_gate_count = len(cluster_summaries)
        post_synth_dropped = []
        for cid in list(cluster_summaries.keys()):
            report = cluster_signals.get(cid, {}).get("cluster_quality")
            if not report:
                continue
            grade = report.get("quality_grade", "C")
            oss = cluster_summaries[cid].get("_oss", 1.0) if cluster_summaries[cid] else 1.0
            # Drop grade F (composite < 0.25) — universally poor quality
            # Drop grade D (composite < 0.40) only when OSS also very low (< 0.15)
            if grade == "F" or (grade == "D" and oss < 0.15):
                reason = f"grade={grade}, OSS={oss:.2f}"
                title = cluster_summaries[cid].get("trend_title", "?")[:50] if cluster_summaries[cid] else "?"
                post_synth_dropped.append(f"[{reason}] {title}")
                del cluster_summaries[cid]
                cluster_articles.pop(cid, None)
                cluster_keywords.pop(cid, None)
                cluster_signals.pop(cid, None)
        if post_synth_dropped:
            self.metrics["post_synth_grade_dropped"] = len(post_synth_dropped)
            logger.info(
                f"  V4 post-synthesis gate: {len(post_synth_dropped)}/{pre_gate_count} clusters dropped "
                f"({pre_gate_count - len(post_synth_dropped)} remain)"
            )
            for dropped in post_synth_dropped[:5]:
                logger.debug(f"    Dropped: {dropped}")

        # AI validation (trend importance + hierarchy)
        cluster_validations = await self._phase_validate_trends(
            cluster_articles, cluster_keywords, cluster_signals, cluster_summaries
        )

        tree = self._phase_tree(
            cluster_articles, cluster_keywords, cluster_signals,
            cluster_summaries, noise_count, len(articles),
            cluster_validations=cluster_validations,
        )

        # Recursive sub-clustering
        if self.max_depth > 1:
            tree = await recursive_subcluster(self, tree, cluster_summaries)

            # Sub-trend cross-correlation: entity bridges between sub-trends
            # across different parent clusters
            try:
                from app.trends.correlation import find_subtopic_correlations
                article_map = {str(a.id): a for a in getattr(self, '_all_articles', [])}
                subtopic_edges = find_subtopic_correlations(tree, article_map)
                if subtopic_edges:
                    self.metrics["subtopic_correlations"] = len(subtopic_edges)
                    # Store on tree for downstream use
                    for node_id_str, node in tree.nodes.items():
                        related = [
                            e for e in subtopic_edges
                            if e["source_node_id"] == node_id_str or e["target_node_id"] == node_id_str
                        ]
                        if related:
                            node.signals["cross_trend_bridges"] = related
            except Exception as e:
                logger.debug(f"Sub-trend correlation skipped: {e}")

        return tree

    async def _layer_causal_reasoning(
        self,
        tree: TrendTree,
        cluster_data: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> TrendTree:
        """Layer 6: Causal Council — multi-agent cross-trend causal reasoning.

        Runs a 4-agent pipeline:
          Agent 1: Pre-filter candidate pairs (0 LLM, statistical scoring)
          Agent 2: Evaluate causal mechanisms (1 LLM per candidate)
          Agent 3: Build cascade narratives (1 LLM per cascade chain)
          Agent 4: Validate evidence (0 LLM, NER + keyword matching)

        Contract: TrendTree → TrendTree (annotated with causal edges + cascades)
        """
        t = time.time()
        trend_edges = cluster_data.get("trend_edges", [])
        cascades = cluster_data.get("cascades", [])

        if not trend_edges and not cascades:
            logger.info("Layer 6 (causal): Skipped — no correlation edges or cascades")
            self.metrics["phase_times"]["causal_council"] = 0.0
            self.metrics["causal_council"] = {"skipped": True, "reason": "no_edges"}
            return tree

        # Build cluster_id → node_id mapping
        # Match by comparing article sets (cluster_articles vs node.source_articles)
        cluster_articles = cluster_data.get("cluster_articles", {})
        cluster_id_to_node_id: Dict[int, str] = {}
        for cid, carts in cluster_articles.items():
            cart_ids = {str(a.id) for a in carts}
            for nid, node in tree.nodes.items():
                node_art_ids = {str(aid) for aid in node.source_articles}
                if cart_ids and node_art_ids and cart_ids == node_art_ids:
                    cluster_id_to_node_id[cid] = nid
                    break

        # Log unmapped clusters (helps diagnose silent failures)
        edge_cids = set()
        for edge in trend_edges:
            edge_cids.add(edge.get("source"))
            edge_cids.add(edge.get("target"))
        unmapped = edge_cids - set(cluster_id_to_node_id.keys()) - {None}
        if unmapped:
            logger.warning(
                f"Layer 6 (causal): {len(unmapped)} cluster IDs unmapped "
                f"(edges reference clusters not in tree): {sorted(unmapped)[:5]}"
            )

        if not cluster_id_to_node_id:
            logger.warning("Layer 6 (causal): Could not map cluster IDs to node IDs")
            self.metrics["phase_times"]["causal_council"] = 0.0
            return tree

        try:
            from app.agents.workers.council.causal_council import run_causal_council, apply_causal_results

            council_result = await run_causal_council(
                tree=tree,
                trend_edges=trend_edges,
                cascades=cascades,
                cluster_id_to_node_id=cluster_id_to_node_id,
                all_articles=articles,
                llm_service=self.llm_tool,
                max_candidates=min(len(trend_edges), 10),
                max_cascades=5,
                concurrency=self.max_concurrent_llm,
            )

            # Apply results to tree (in-place)
            apply_causal_results(tree, council_result)

            self.metrics["causal_council"] = {
                "edges_confirmed": council_result.edges_confirmed,
                "cascades_found": council_result.cascades_found,
                "pairs_evaluated": council_result.pairs_evaluated,
                "llm_calls": council_result.llm_calls_made,
            }

        except Exception as e:
            logger.warning(f"Layer 6 (causal): Failed — {e}", exc_info=True)
            self.metrics["causal_council"] = {"error": str(e)}

        elapsed = time.time() - t
        self.metrics["phase_times"]["causal_council"] = round(elapsed, 2)
        logger.info(f"Layer 6 (causal): {elapsed:.1f}s")

        return tree

    def _record_pipeline_metrics(self, cluster_data: Dict[str, Any]) -> None:
        """Record score distributions for calibration and drift detection."""
        try:
            from app.learning.pipeline_metrics import (
                record_run, detect_drift, detect_drift_ewma,
                compute_distributions, compute_source_quality,
                compute_cluster_stability, save_cluster_assignments,
                record_cluster_signals, compute_run_quality,
                load_history,
            )
            # Assign a single run_id that will be used for BOTH cluster signal
            # logging AND the pipeline run log entry (record_run uses _flatten_metrics
            # which spreads self.metrics, so pre-setting run_id here ensures both
            # logs share the same identifier for cross-referencing).
            if "run_id" not in self.metrics:
                self.metrics["run_id"] = str(uuid.uuid4())[:8]

            cluster_signals = cluster_data["cluster_signals"]
            coherences = np.array([
                s.get("intra_cluster_cosine", 0.0) for s in cluster_signals.values()
            ])
            trend_scores = np.array([
                s.get("trend_score", 0.0) for s in cluster_signals.values()
            ])
            confidence_scores = np.array([
                s.get("confidence_score", 0.0) for s in cluster_signals.values()
            ])
            self.metrics.update(compute_distributions(
                coherences=coherences, trend_scores=trend_scores,
                confidence_scores=confidence_scores,
            ))

            # Log general_ratio for EWMA monitoring
            event_dist = self.metrics.get("event_distribution", {})
            total_events = sum(event_dist.values()) if event_dist else 0
            if total_events > 0:
                self.metrics["general_ratio"] = round(
                    event_dist.get("general", 0) / total_events, 4
                )

            # Per-source quality stats (feeds source bandit + future improvements)
            articles = getattr(self, '_all_articles', [])
            labels = cluster_data.get("labels")
            if articles and labels is not None:
                cluster_quality_scores = {
                    cid: sigs.get("cluster_quality_score", 0.5)
                    for cid, sigs in cluster_signals.items()
                }
                self.metrics["source_quality"] = compute_source_quality(
                    articles, labels, cluster_quality_scores,
                )

            # Cluster stability index (detect parameter oscillation)
            if articles and labels is not None:
                csi = compute_cluster_stability(articles, labels)
                if csi >= 0:
                    self.metrics["cluster_stability_index"] = csi
                # Save assignments for next run's stability check
                save_cluster_assignments(articles, labels, self.metrics)

            # OSS + cluster signal logging for auto-learning (Phase 4F)
            summaries = cluster_data.get("summaries", {})
            if summaries and cluster_signals:
                # Extract per-cluster OSS scores from synthesis results
                cluster_oss = {}
                for cid, synth in summaries.items():
                    if synth:
                        oss_val = synth.get("_oss")
                        if oss_val is not None:
                            cluster_oss[cid] = float(oss_val)

                # Log per-cluster signals + OSS for weight auto-learning
                run_id = self.metrics["run_id"]
                if cluster_oss:
                    record_cluster_signals(
                        run_id=run_id,
                        cluster_signals=cluster_signals,
                        cluster_oss=cluster_oss,
                    )

                # Compute run quality fingerprint
                prev_history = load_history(last_n=1)
                prev_mean_oss = None
                if prev_history:
                    prev_rq = prev_history[-1].get("run_quality", {})
                    prev_mean_oss = prev_rq.get("mean_oss")

                run_quality = compute_run_quality(
                    syntheses=summaries,
                    previous_mean_oss=prev_mean_oss,
                )
                self.metrics["run_quality"] = run_quality

            record_run(self.metrics)

            # Emit LearningSignals — one per synthesized trend, logged to JSONL.
            # Consumed by weight_learner (OSS auto-learning) and source_bandit.
            if summaries:
                try:
                    self._emit_learning_signals(
                        summaries=summaries,
                        run_id=self.metrics.get("run_id", ""),
                    )
                except Exception as _ls_err:
                    logger.debug(f"LearningSignal emit skipped: {_ls_err}")

            # Z-score drift detection (catches sudden jumps)
            drift_alerts = detect_drift(self.metrics)
            if drift_alerts:
                self.metrics["drift_alerts"] = drift_alerts

            # EWMA drift detection (catches slow persistent degradation)
            ewma_alerts = detect_drift_ewma(self.metrics)
            if ewma_alerts:
                self.metrics["ewma_drift_alerts"] = ewma_alerts
                # Signal Optuna to double trials on next run
                self.metrics["ewma_drift_detected"] = True
                logger.warning(
                    f"EWMA drift detected — Optuna will use extra trials "
                    f"on next run: {ewma_alerts}"
                )
        except Exception as e:
            logger.debug(f"Metric logging skipped: {e}")

    def _log_synthesis_outcomes(
        self,
        raw_summaries: Dict[int, Dict[str, Any]],
        filtered_summaries: Dict[int, Dict[str, Any]],
        cluster_signals: Dict[int, Dict[str, Any]],
    ) -> None:
        """Log which clusters failed synthesis and why (for feedback loop)."""
        try:
            dropped = set(raw_summaries.keys()) - set(filtered_summaries.keys())
            outcomes = {
                "total_clusters": len(raw_summaries),
                "passed": len(filtered_summaries),
                "failed": len(dropped),
                "failures": [],
            }
            for cid in dropped:
                summary = raw_summaries.get(cid, {})
                sigs = cluster_signals.get(cid, {})
                reason = "unknown"
                if not summary:
                    reason = "empty_synthesis"
                elif not summary.get("trend_title") or len(str(summary.get("trend_title", "")).strip()) < 5:
                    reason = "no_title"
                elif not summary.get("trend_summary") or len(str(summary.get("trend_summary", "")).strip()) < 20:
                    reason = "short_summary"
                else:
                    reason = "low_confidence"

                outcomes["failures"].append({
                    "cluster_id": cid,
                    "reason": reason,
                    "article_count": sigs.get("article_count", 0),
                    "coherence": sigs.get("intra_cluster_cosine", 0),
                    "confidence": sigs.get("confidence", 0),
                })

            self.metrics["synthesis_outcomes"] = outcomes
            if dropped:
                logger.info(
                    f"Synthesis outcomes: {len(filtered_summaries)}/{len(raw_summaries)} "
                    f"passed, {len(dropped)} dropped"
                )
        except Exception as e:
            logger.debug(f"Synthesis outcome logging failed: {e}")

    def _discover_taxonomy(
        self,
        cluster_data: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> None:
        """Discover new event taxonomy candidates from unmatched articles."""
        try:
            if not hasattr(self, '_event_classifier'):
                return

            classifier = self._event_classifier
            labels = cluster_data["labels"]

            # Log per-event-type effectiveness
            effectiveness = classifier.get_event_effectiveness(articles, labels)
            self.metrics["event_effectiveness"] = effectiveness

            # Discover taxonomy candidates from unmatched buffer
            candidates = classifier.discover_taxonomy_candidates()
            if candidates:
                self.metrics["taxonomy_candidates_count"] = len(candidates)
                # Save candidates (LLM naming is async, skip in sync context)
                classifier.save_taxonomy_candidates(candidates)
                logger.info(
                    f"Taxonomy: {len(candidates)} new event type candidates "
                    f"discovered from {len(classifier._unmatched_buffer)} "
                    f"unmatched articles"
                )
        except Exception as e:
            logger.debug(f"Taxonomy discovery skipped: {e}")

    def _update_memory_oss(self, cluster_data: Dict[str, Any]) -> None:
        """Post-synthesis: write OSS scores back to trend memory centroids.

        Since _layer_temporalize runs BEFORE synthesis, centroids are stored
        with oss=0.0. After synthesis produces OSS, we update stored centroids.
        """
        try:
            summaries = cluster_data.get("summaries", {})
            if not summaries:
                return

            cluster_oss = {}
            for cid, synth in summaries.items():
                if synth:
                    oss_val = synth.get("_oss")
                    if oss_val is not None:
                        cluster_oss[cid] = float(oss_val)

            if not cluster_oss:
                return

            # Rebuild centroids from embeddings + labels
            embeddings = cluster_data.get("embeddings")
            labels = cluster_data.get("labels")
            if embeddings is None or labels is None:
                return

            emb_array = np.array(embeddings, dtype=np.float32) if not isinstance(embeddings, np.ndarray) else embeddings
            cluster_centroids = {}
            for cid in cluster_oss:
                mask = labels == cid
                if mask.any():
                    cluster_centroids[cid] = emb_array[mask].mean(axis=0)

            if not cluster_centroids:
                return

            from app.trends.memory import TrendMemory
            memory = TrendMemory()
            updated = memory.update_oss_scores(cluster_centroids, cluster_oss)
            if updated:
                logger.info(f"Trend memory: updated OSS for {updated} centroids")
        except Exception as e:
            logger.debug(f"Trend memory OSS update skipped: {e}")

    def _update_source_bandit(
        self,
        cluster_data: Dict[str, Any],
        articles: List[NewsArticle],
    ) -> None:
        """Update source quality bandit posteriors from this run's results."""
        try:
            from app.learning.source_bandit import SourceBandit

            bandit = SourceBandit()
            labels = cluster_data["labels"]
            cluster_signals = cluster_data["cluster_signals"]

            # Build article_id → label mapping
            article_labels = {}
            for i, (article, label) in enumerate(zip(articles, labels)):
                article_labels[str(id(article))] = int(label)

            # Build source_id → [article_ids]
            source_articles: Dict[str, list] = defaultdict(list)
            for article in articles:
                src = getattr(article, 'source_id', '') or ''
                if src:
                    source_articles[src].append(str(id(article)))

            # Build cluster quality scores
            cluster_quality = {
                cid: sigs.get("cluster_quality_score", 0.5)
                for cid, sigs in cluster_signals.items()
            }

            # Entity richness per source
            entity_richness: Dict[str, float] = {}
            for src, aids in source_articles.items():
                ent_counts = []
                for article in articles:
                    if str(id(article)) in aids:
                        ents = getattr(article, 'entities', [])
                        ent_counts.append(len(ents) if ents else 0)
                if ent_counts:
                    entity_richness[src] = sum(ent_counts) / len(ent_counts)

            # Pre-clustering content quality per source (independent signal).
            # Measured BEFORE clustering — not self-referential.
            # Factors: content length, scrape success, title quality.
            content_quality: Dict[str, float] = {}
            for src, aids in source_articles.items():
                scores = []
                for article in articles:
                    if str(id(article)) not in aids:
                        continue
                    # Content length: 0-500 chars = 0.0, 2000+ = 1.0
                    content = getattr(article, 'content', '') or ''
                    len_score = min(1.0, len(content) / 2000)
                    # Title quality: has numbers or proper nouns = better
                    title = getattr(article, 'title', '') or ''
                    title_score = 0.5
                    if any(c.isdigit() for c in title):
                        title_score += 0.25  # Contains specific numbers
                    if len(title.split()) >= 5:
                        title_score += 0.25  # Not too short
                    title_score = min(1.0, title_score)
                    scores.append(0.6 * len_score + 0.4 * title_score)
                if scores:
                    content_quality[src] = sum(scores) / len(scores)

            # V4: Extract per-cluster OSS for cross-level feedback
            cluster_oss = {}
            summaries = cluster_data.get("summaries", {})
            for cid, synth in summaries.items():
                if synth:
                    oss_val = synth.get("_oss")
                    if oss_val is not None:
                        cluster_oss[cid] = float(oss_val)

            estimates = bandit.update_from_run(
                source_articles=dict(source_articles),
                article_labels=article_labels,
                cluster_quality=cluster_quality,
                entity_richness=entity_richness,
                content_quality=content_quality,
                cluster_oss=cluster_oss,
            )

            self.metrics["source_bandit"] = {
                "sources_tracked": bandit.source_count,
                "top_5": dict(
                    sorted(estimates.items(), key=lambda x: x[1], reverse=True)[:5]
                ),
                "bottom_5": dict(
                    sorted(estimates.items(), key=lambda x: x[1])[:5]
                ),
            }
            logger.info(
                f"Source bandit updated: {bandit.source_count} sources tracked"
            )
        except Exception as e:
            logger.debug(f"Source bandit update skipped: {e}")

    def _emit_learning_signals(
        self,
        summaries: Dict,
        run_id: str,
    ) -> None:
        """Emit one LearningSignal per synthesized trend to the signal log.

        Signals are consumed by weight_learner (OSS→weights) and source_bandit.
        Written to data/learning_signals.jsonl.
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        log_path = Path("./data/learning_signals.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        signals = []
        now = datetime.now(timezone.utc).isoformat()
        for cid, synth in summaries.items():
            if not synth:
                continue
            signals.append({
                "run_id": run_id,
                "timestamp": now,
                "trend_title": synth.get("trend_title", ""),
                "event_type": synth.get("event_type", ""),
                "oss_score": float(synth.get("_oss", 0.0)),
                "synthesis_retries": int(synth.get("_retries", 0)),
                "cluster_id": str(cid),
            })

        if signals:
            with open(log_path, "a", encoding="utf-8") as f:
                for s in signals:
                    f.write(json.dumps(s) + "\n")
            logger.debug(f"LearningSignals: {len(signals)} emitted to {log_path}")

    def _auto_generate_feedback(self, cluster_data: Dict[str, Any]) -> None:
        """Auto-generate trend feedback for weight learner from cluster quality.

        Closes the self-learning loop: instead of requiring human feedback,
        automatically classify synthesized trends as "good_trend" or "bad_trend"
        based on multi-signal quality assessment.

        NOTE: Auto-feedback is tagged with metadata.auto=True and is used
        ONLY for pipeline health monitoring. The weight_learner explicitly
        filters out auto-feedback to prevent a circular validation loop
        (pipeline rating itself → learning from those ratings → tautology).

        Real learning requires human feedback via the Streamlit UI.

        Criteria for good_trend:
          - Coherence >= 0.45
          - Article count >= 3
          - Synthesis has specific title (named entities + numbers)
          - Event type is specific (not "general")

        Criteria for bad_trend:
          - Coherence < 0.35 OR
          - Synthesis failed/generic OR
          - All articles are "general" event type
        """
        try:
            from app.tools.feedback import save_feedback

            cluster_signals = cluster_data.get("cluster_signals", {})
            cluster_articles = cluster_data.get("cluster_articles", {})
            summaries = cluster_data.get("summaries", {})
            good_count = 0
            bad_count = 0

            for cid, sigs in cluster_signals.items():
                coherence = sigs.get("intra_cluster_cosine", 0)
                n_articles = sigs.get("article_count", 0)
                trend_score = sigs.get("trend_score", 0)
                confidence = sigs.get("confidence_score", 0)
                cluster_quality = sigs.get("cluster_quality_score", 0)

                # Get synthesis info
                summary = summaries.get(cid, {})
                title = summary.get("trend_title", "") if isinstance(summary, dict) else ""
                synth_text = summary.get("trend_summary", "") if isinstance(summary, dict) else ""

                # Get event type concentration
                arts = cluster_articles.get(cid, [])
                event_types = [getattr(a, '_trigger_event', 'general') for a in arts]
                general_ratio = event_types.count('general') / max(len(event_types), 1)

                # Build signal breakdown for learning
                signals = {
                    "actionability_breakdown": {
                        "coherence": {"raw": coherence},
                        "article_count": {"raw": min(n_articles / 10, 1.0)},
                        "confidence": {"raw": confidence},
                        "trend_score": {"raw": trend_score},
                    },
                    "trend_score_breakdown": {
                        "coherence": {"raw": coherence},
                        "source_diversity": {"raw": sigs.get("source_diversity", 0)},
                        "entity_richness": {"raw": sigs.get("entity_richness", 0)},
                    },
                    "cluster_quality_breakdown": {
                        "coherence": {"raw": coherence},
                        "entity_overlap": {"raw": sigs.get("entity_overlap", 0)},
                        "source_diversity": {"raw": sigs.get("source_diversity", 0)},
                    },
                }

                # Classify: good or bad
                has_synthesis = bool(title and len(title) > 10 and synth_text and len(synth_text) > 30)
                is_specific = general_ratio < 0.5

                if coherence >= 0.45 and n_articles >= 3 and has_synthesis and is_specific:
                    save_feedback(
                        feedback_type="trend",
                        item_id=f"auto_{self.metrics.get('run_id', 'unknown')}_{cid}",
                        rating="good_trend",
                        signals=signals,
                        metadata={"title": title[:100], "auto": True, "coherence": coherence},
                    )
                    good_count += 1
                elif coherence < 0.35 or not has_synthesis or general_ratio >= 0.8:
                    save_feedback(
                        feedback_type="trend",
                        item_id=f"auto_{self.metrics.get('run_id', 'unknown')}_{cid}",
                        rating="bad_trend",
                        signals=signals,
                        metadata={"title": title[:100], "auto": True, "coherence": coherence},
                    )
                    bad_count += 1

            # Also generate bad feedback from LLM-rejected clusters
            rejected_ids = set(cluster_data.get("rejected_cluster_ids", []))
            logger.info(
                f"Auto-feedback: checking {len(rejected_ids)} rejected clusters "
                f"for bad_trend feedback: {sorted(rejected_ids)}"
            )
            for cid in rejected_ids:
                sigs = cluster_signals.get(cid, {})
                coherence = sigs.get("intra_cluster_cosine", 0)
                signals = {
                    "actionability_breakdown": {
                        "coherence": {"raw": coherence},
                        "article_count": {"raw": min(sigs.get("article_count", 0) / 10, 1.0)},
                        "confidence": {"raw": sigs.get("confidence_score", 0)},
                        "trend_score": {"raw": sigs.get("trend_score", 0)},
                    },
                    "trend_score_breakdown": {
                        "coherence": {"raw": coherence},
                        "source_diversity": {"raw": sigs.get("source_diversity", 0)},
                        "entity_richness": {"raw": sigs.get("entity_richness", 0)},
                    },
                    "cluster_quality_breakdown": {
                        "coherence": {"raw": coherence},
                        "entity_overlap": {"raw": sigs.get("entity_overlap", 0)},
                        "source_diversity": {"raw": sigs.get("source_diversity", 0)},
                    },
                }
                save_feedback(
                    feedback_type="trend",
                    item_id=f"auto_{self.metrics.get('run_id', 'unknown')}_rejected_{cid}",
                    rating="bad_trend",
                    signals=signals,
                    metadata={"rejected": True, "auto": True, "coherence": coherence},
                )
                bad_count += 1

            self.metrics["auto_feedback"] = {
                "good_trends": good_count,
                "bad_trends": bad_count,
            }
            if good_count or bad_count:
                logger.info(
                    f"Auto-feedback: {good_count} good + {bad_count} bad trends "
                    f"saved for weight learner"
                )
        except Exception as e:
            logger.debug(f"Auto-feedback generation skipped: {e}")

    def _evolve_classifier_anchors(self, articles: list) -> None:
        """V12: Save high-confidence classifications as future anchors.

        Self-learning loop for event classifier:
          classify_batch() → collect_learned_anchors() → save
          → next run loads them → richer k-NN index → better classification

        This is the key mechanism that makes the event taxonomy self-evolving:
        as more articles are processed, the classifier automatically builds
        richer anchor sets from its own successful classifications.
        """
        try:
            if not hasattr(self, "_event_classifier") or not self._event_classifier:
                return

            # Collect high-confidence classifications
            learned = self._event_classifier.collect_learned_anchors(
                articles, min_confidence=0.65, max_per_type=20
            )

            if not learned:
                return

            # Save (merges with existing, dedupes, prunes old)
            total = self._event_classifier.save_learned_anchors(learned)
            n_new = sum(len(v) for v in learned.values())
            self.metrics["learned_anchors"] = {
                "new_collected": n_new,
                "total_saved": total,
                "event_types": list(learned.keys()),
            }
            logger.info(
                f"Self-evolving anchors: {n_new} new from {len(learned)} types, "
                f"{total} total saved"
            )
        except Exception as e:
            logger.debug(f"Anchor evolution skipped: {e}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE IMPLEMENTATIONS
    # ════════════════════════════════════════════════════════════════════

    def _apply_source_bandit_priority(
        self, articles: List[NewsArticle],
    ) -> List[NewsArticle]:
        """Apply source bandit Thompson Sampling to prioritize articles.

        Ranks sources by Thompson Sample from Beta posteriors, then sorts
        articles so high-quality-source articles come first. This closes
        the source bandit feedback loop: posteriors are updated after each
        run (in _update_source_bandit), and here they are CONSUMED to
        prioritize articles before clustering.

        If the total article count is large (>200), aggressively trims
        low-quality sources to keep the pipeline focused.
        """
        try:
            from app.learning.source_bandit import SourceBandit
            bandit = SourceBandit()

            # Get all source IDs present in articles
            source_ids = list({
                getattr(a, 'source_id', 'unknown') or 'unknown'
                for a in articles
            })

            if not bandit.source_count or len(source_ids) < 2:
                return articles  # No bandit data yet, use all

            # Thompson Sampling: rank sources
            ranked = bandit.select_sources(source_ids)
            rank_map = {sid: i for i, sid in enumerate(ranked)}

            # Sort articles: high-quality sources first
            articles.sort(
                key=lambda a: rank_map.get(
                    getattr(a, 'source_id', 'unknown') or 'unknown', 999
                )
            )

            # If many articles, trim bottom 20% of sources
            if len(articles) > 200 and len(ranked) > 5:
                cutoff_idx = max(3, int(len(ranked) * 0.80))
                low_quality_sources = set(ranked[cutoff_idx:])
                before = len(articles)
                articles = [
                    a for a in articles
                    if (getattr(a, 'source_id', '') or '') not in low_quality_sources
                ]
                trimmed = before - len(articles)
                if trimmed > 0:
                    logger.info(
                        f"Source bandit trimmed {trimmed} articles from "
                        f"{len(low_quality_sources)} low-quality sources"
                    )

            self.metrics["source_bandit_applied"] = True
            self.metrics["source_bandit_ranking"] = ranked[:10]
            return articles

        except Exception as e:
            logger.debug(f"Source bandit priority skipped: {e}")
            return articles

    async def _phase_scrape(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 0: Optionally scrape full article content (disabled by default — RSS summary used)."""
        from app.config import get_settings as _gs2
        cfg = _gs2()
        if not cfg.scrape_enabled:
            logger.info(f"Phase 0 (scrape): disabled — using RSS summaries for {len(articles)} articles")
            self.metrics["phase_times"]["scrape"] = 0.0
            self.metrics["article_counts"]["scraped"] = 0
            return articles

        t = time.time()
        try:
            enriched = await scrape_articles(
                articles,
                max_concurrent=cfg.scrape_max_concurrent,
                max_articles=cfg.scrape_max_articles,
            )
            elapsed = time.time() - t
            self.metrics["phase_times"]["scrape"] = round(elapsed, 2)
            self.metrics["article_counts"]["scraped"] = enriched
            has_content = sum(1 for a in articles if a.content)
            logger.info(f"Phase 0 (scrape): {enriched} enriched, {has_content}/{len(articles)} have content in {elapsed:.1f}s")
        except Exception as e:
            self.metrics["phase_times"]["scrape"] = round(time.time() - t, 2)
            logger.warning(f"Phase 0 (scrape) failed: {e}")
        return articles

    def _phase_classify_events(self, articles: List[NewsArticle]) -> None:
        """Phase 0.5: Classify events using embedding similarity (no regex)."""
        t = time.time()
        try:
            if not hasattr(self, '_event_classifier'):
                self._event_classifier = EmbeddingEventClassifier(self.embedding_tool)
            distribution = self._event_classifier.classify_batch(articles)
        except Exception as e:
            logger.warning(f"Phase 0.5 (events) failed: {e}")
            self.metrics["phase_times"]["event_classify"] = round(time.time() - t, 2)
            self.metrics["event_distribution"] = {}
            return
        elapsed = time.time() - t
        self.metrics["phase_times"]["event_classify"] = round(elapsed, 2)
        self.metrics["event_distribution"] = distribution
        logger.info(f"Phase 0.5 (events, embedding-based): {distribution} in {elapsed:.2f}s")

    async def _phase_classify_events_tier2(self) -> None:
        """Phase 0.6: LLM-validate ambiguous event classifications from Tier 1."""
        if not hasattr(self, '_event_classifier'):
            return
        ambiguous = getattr(self._event_classifier, '_ambiguous_articles', [])
        if not ambiguous:
            return
        if getattr(self.settings, 'offline_mode', False):
            logger.info(f"Phase 0.6 (events, LLM Tier 2): skipped (offline mode, {len(ambiguous)} ambiguous kept as-is)")
            return
        if self.mock_mode:
            logger.info("Phase 0.6 (events, LLM Tier 2): skipped (mock mode)")
            return
        t = time.time()
        try:
            reclassified = await self._event_classifier.classify_ambiguous_with_llm()
            elapsed = time.time() - t
            self.metrics["phase_times"]["event_classify_tier2"] = round(elapsed, 2)
            logger.info(
                f"Phase 0.6 (events, LLM Tier 2): {len(ambiguous)} ambiguous → "
                f"{reclassified} reclassified in {elapsed:.2f}s"
            )
        except Exception as e:
            logger.warning(f"Phase 0.6 (events, LLM Tier 2) failed: {e}")
            self.metrics["phase_times"]["event_classify_tier2"] = round(time.time() - t, 2)

    def _phase_noise_filter(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Filter articles classified as noise types (entertainment, sports, lifestyle).

        These event types exist as attractors — they pull non-business content
        AWAY from real event categories (e.g., "box office collection" away from
        "earnings"). Articles tagged as noise are removed before clustering.
        """
        from app.news.event_classifier import NOISE_EVENT_TYPES
        t = time.time()
        kept = []
        filtered = 0
        noise_dist = Counter()
        for a in articles:
            etype = getattr(a, '_trigger_event', 'general')
            if etype in NOISE_EVENT_TYPES:
                filtered += 1
                noise_dist[etype] += 1
            else:
                kept.append(a)
        elapsed = time.time() - t
        self.metrics["phase_times"]["noise_filter"] = round(elapsed, 3)
        self.metrics["article_counts"]["noise_filtered"] = filtered
        if filtered:
            breakdown = ", ".join(f"{t}={c}" for t, c in noise_dist.most_common())
            logger.info(
                f"Noise filter: {filtered} articles removed "
                f"({len(articles)} → {len(kept)}) [{breakdown}]"
            )
        return kept

    async def _phase_article_triage(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """LLM-based triage: classify ambiguous articles as business or noise.

        Uses the Article Triage Agent (council pattern) to batch-classify
        articles that the embedding classifier couldn't confidently classify.
        Clear business articles skip the LLM call entirely.

        Architecture:
          1. select_triage_candidates() splits articles into clear vs ambiguous
          2. triage_articles() batch-classifies ambiguous articles via LLM
          3. Returns clear + LLM-approved articles (noise filtered out)

        Cost: ~2-3 LLM calls per pipeline run (10-15 articles per call).
        """
        t = time.time()

        if getattr(self.settings, 'offline_mode', False):
            logger.info("Article triage: skipped (offline mode)")
            return articles

        from app.agents.workers.council.article_triage import (
            select_triage_candidates, triage_articles,
        )

        # Split: clear business articles skip LLM, ambiguous ones get triaged
        candidates, clear = select_triage_candidates(articles)

        if not candidates:
            elapsed = time.time() - t
            self.metrics["phase_times"]["article_triage"] = round(elapsed, 3)
            self.metrics["article_counts"]["triage_candidates"] = 0
            self.metrics["article_counts"]["triage_filtered"] = 0
            return articles

        # LLM batch triage
        try:
            kept, filtered = await triage_articles(
                candidates,
                llm_service=self.llm_tool,
                batch_size=10,
            )
        except Exception as e:
            logger.warning(f"Article triage failed, keeping all articles: {e}")
            kept = candidates
            filtered = []

        # Combine clear + LLM-approved
        result = clear + kept

        elapsed = time.time() - t
        self.metrics["phase_times"]["article_triage"] = round(elapsed, 3)
        self.metrics["article_counts"]["triage_candidates"] = len(candidates)
        self.metrics["article_counts"]["triage_filtered"] = len(filtered)

        if filtered:
            noise_categories = Counter(
                getattr(a, '_triage_category', 'unknown') for a in filtered
            )
            breakdown = ", ".join(f"{c}={n}" for c, n in noise_categories.most_common())
            logger.info(
                f"Article triage: {len(filtered)} noise articles removed "
                f"({len(articles)} → {len(result)}) [{breakdown}]"
            )
            # Log specific filtered articles for debugging
            for a in filtered[:5]:
                logger.debug(
                    f"  Triaged out: '{(a.title or '')[:60]}' "
                    f"({getattr(a, '_triage_category', '?')}, "
                    f"conf={getattr(a, '_triage_confidence', 0):.2f})"
                )

        return result

    def _phase_language_filter(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Filter non-English articles using langdetect (Google n-gram detector).

        Checks title + summary + first 300 chars of scraped content.
        Scales globally — works for any target language, no hardcoded ranges.
        """
        from langdetect import detect, LangDetectException

        t = time.time()
        target_lang = "en"  # Configurable per-pipeline later

        kept = []
        filtered = 0
        filtered_langs = Counter()
        for a in articles:
            check = f"{a.title} {a.summary or ''} {(a.content or '')[:300]}"
            if len(check.strip()) < 20:
                kept.append(a)
                continue
            try:
                detected = detect(check[:500])
                if detected == target_lang:
                    kept.append(a)
                else:
                    filtered += 1
                    filtered_langs[detected] += 1
            except LangDetectException:
                kept.append(a)  # Ambiguous — let through

        elapsed = time.time() - t
        self.metrics["phase_times"]["language_filter"] = round(elapsed, 2)
        self.metrics["article_counts"]["language_filtered"] = filtered
        if filtered:
            top_langs = ", ".join(f"{l}={c}" for l, c in filtered_langs.most_common(5))
            logger.info(
                f"Language filter: {filtered} non-English articles removed "
                f"({len(articles)} → {len(kept)}) in {elapsed:.2f}s [{top_langs}]"
            )
        return kept

    def _phase_dedup(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 1: Remove near-duplicate articles (MinHash LSH)."""
        logger.info(f">> Phase 1 (dedup): {len(articles)} articles...")
        t = time.time()
        result = self.deduplicator.deduplicate(articles)
        elapsed = time.time() - t
        self.metrics["phase_times"]["dedup"] = round(elapsed, 2)
        self.metrics["article_counts"]["after_dedup"] = len(result)
        logger.info(f"Phase 1 (dedup): {len(articles)} -> {len(result)} in {elapsed:.2f}s")
        return result

    def _phase_ner(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 2: Extract named entities (spaCy)."""
        logger.info(f">> Phase 2 (NER): {len(articles)} articles...")
        t = time.time()
        result = self.entity_extractor.extract_batch(articles)
        elapsed = time.time() - t
        self.metrics["phase_times"]["ner"] = round(elapsed, 2)
        logger.info(f"Phase 2 (NER): {len(articles)} articles in {elapsed:.2f}s")
        return result

    def _phase_geo_filter(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 2.3: Geographic relevance filter using NER entities.

        Dynamic, entity-based — no hardcoded keyword lists. Uses GPE/NORP/LOC
        entities extracted by Phase 2 (NER) to check if each article is relevant
        to the configured target country.

        Works for any country: set COUNTRY env var to "Brazil", "Germany", etc.
        The filter adapts automatically — no code changes needed.

        Articles pass if ANY of:
        - A GPE/NORP/LOC entity contains the country name (substring match)
        - The article has no geographic entities (can't determine → keep)
        - The article's source is in the domestic_source_ids set
        """
        if not self._target_country:
            return articles  # No country configured → skip

        t = time.time()
        country_lower = self._target_country.lower()

        relevant = []
        filtered_titles = []

        for article in articles:
            # Fast path: domestic source → auto-pass
            if article.source_id in self._domestic_source_ids:
                relevant.append(article)
                continue

            # Collect geographic + nationality entities from NER
            geo_entities = [
                e.text.lower() for e in article.entities
                if e.type in ("GPE", "NORP", "LOC")
            ]

            if not geo_entities:
                # No geographic entities → can't determine, keep it
                relevant.append(article)
                continue

            # Word-boundary matching: prevents "indonesian" matching "india",
            # prevents short entity "in" matching anything.
            is_relevant = any(
                _country_matches(country_lower, entity)
                for entity in geo_entities
            )

            if is_relevant:
                relevant.append(article)
            else:
                filtered_titles.append(article.title[:80])

        elapsed = time.time() - t
        n_removed = len(articles) - len(relevant)
        self.metrics["phase_times"]["geo_filter"] = round(elapsed, 2)
        self.metrics["article_counts"]["after_geo_filter"] = len(relevant)

        if n_removed:
            logger.info(
                f"Phase 2.3 (geo filter, country={self._target_country}): "
                f"{len(articles)} → {len(relevant)} ({n_removed} non-{self._target_country} removed)"
            )
            if filtered_titles:
                logger.debug(f"  Filtered: {filtered_titles[:10]}")
        else:
            logger.info(f"Phase 2.3 (geo filter): all {len(articles)} articles passed")

        return relevant


    def _phase_embed(self, articles: List[NewsArticle]) -> List[List[float]]:
        """Phase 3: Content-aware embeddings (title + body + entities + event)."""
        logger.info(f">> Phase 3 (embed): {len(articles)} articles — local model, may take a moment...")
        t = time.time()

        def _embed_text(a):
            title = a.title or ""
            entities_str = ""
            if hasattr(a, 'entities') and a.entities:
                ent_names = [getattr(e, 'name', str(e)) for e in a.entities[:10]]
                entities_str = " ".join(ent_names)
            # ITERATION 8 FIX: Embed title + entities ONLY, no body text.
            #
            # WHY: Body text is the primary source of "India business news gravity
            # well" — all Indian news articles share similar intro patterns
            # ("In a significant development...", "According to sources..."),
            # domain vocabulary, and structural conventions.  This shared body
            # vocabulary dominates the 1024-dim embedding space and makes
            # unrelated articles (e.g., "Tata Punch EV" and "Barpeta robbery")
            # cluster together at 0.55-0.85 cosine similarity.
            #
            # REF: AWS Financial News clustering (110K articles) confirmed
            # headline + entity names substantially outperforms full-text
            # embeddings for same-domain clustering.
            #
            # Body text is still available for post-clustering summarization
            # (synthesis, keyword extraction) — we just don't embed it.
            #
            # Title is repeated twice to emphasize it in the embedding.
            # First sentence of body is kept as minimal context.
            body = a.content or a.summary or ""
            first_sentence = ""
            if body:
                # Extract first sentence (up to 150 chars) for minimal context
                for sep in ['. ', '.\n', '.\r']:
                    idx = body.find(sep)
                    if idx > 0:
                        first_sentence = body[:idx + 1]
                        break
                if not first_sentence and len(body) > 20:
                    first_sentence = body[:150]
                first_sentence = first_sentence[:150]
            return f"{title}. {title}. {entities_str} {first_sentence}"

        texts = [_embed_text(a) for a in articles]
        embeddings = self.embedding_tool.embed_batch(texts)

        for article, emb in zip(articles, embeddings):
            article.title_embedding = emb

        elapsed = time.time() - t
        self.metrics["phase_times"]["embed"] = round(elapsed, 2)

        # Sanity check for identical embeddings
        if len(embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sample = min(5, len(embeddings))
            sims = cosine_similarity(embeddings[:sample])
            avg = sims[np.triu_indices(sample, k=1)].mean()
            if avg > 0.99:
                logger.warning("Embeddings near-identical — check embedding tool!")

        logger.info(f"Phase 3 (embed): {len(articles)} articles in {elapsed:.2f}s")
        return embeddings

    def _phase_semantic_dedup(
        self, articles: List[NewsArticle], embeddings: List[List[float]],
        threshold: float = 0.78,
    ) -> tuple:
        """Phase 3.5: Semantic dedup using content-aware embeddings.

        Removes articles whose embedding cosine similarity exceeds the threshold.
        Includes diagnostics showing the similarity distribution (P50-P99) so
        threshold tuning is data-driven, not guesswork.

        NOTE: Phase 1 (MinHash + title fingerprint + entity fingerprint) already
        removes ~30% of articles via 3 text-level stages. This phase is a FINAL
        pass for articles that slip through — so low removal counts are expected
        when Phase 1 is aggressive. Check the distribution log to verify.
        """
        t = time.time()
        n = len(articles)
        if n <= 1:
            return articles, embeddings

        # ── Safety gate: cap N to avoid O(N^2) OOM ──────────────────────────
        from app.config import get_settings as _gs
        MAX_DEDUP_ARTICLES = _gs().semantic_dedup_max_articles
        overflow_articles = []
        overflow_embeddings = []
        if n > MAX_DEDUP_ARTICLES:
            logger.warning(
                f"Phase 3.5: {n} articles exceeds {MAX_DEDUP_ARTICLES} cap for NxN dedup. "
                f"Deduping first {MAX_DEDUP_ARTICLES}; keeping remaining {n - MAX_DEDUP_ARTICLES} as-is."
            )
            overflow_articles = articles[MAX_DEDUP_ARTICLES:]
            overflow_embeddings = embeddings[MAX_DEDUP_ARTICLES:]
            articles = articles[:MAX_DEDUP_ARTICLES]
            embeddings = embeddings[:MAX_DEDUP_ARTICLES]
            n = MAX_DEDUP_ARTICLES

        emb_array = np.array(embeddings)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_norm = emb_array / norms

        # ── Safety gate: detect degenerate embeddings ──────────────────────
        # If a random sample of articles all have >0.95 similarity, the embedding
        # model is broken (collapsed space). Skip dedup to avoid mass deletion.
        sample_size = min(20, n)
        sample_sims = np.dot(emb_norm[:sample_size], emb_norm[:sample_size].T)
        upper_tri = sample_sims[np.triu_indices(sample_size, k=1)]
        if len(upper_tri) > 0:
            avg_sample_sim = float(upper_tri.mean())
            if avg_sample_sim > 0.95:
                elapsed = time.time() - t
                logger.error(
                    f"Phase 3.5 SKIPPED: embeddings are degenerate (avg sample sim={avg_sample_sim:.3f}). "
                    f"The embedding model may have failed or produced collapsed vectors. "
                    f"Keeping all {n} articles to prevent data loss."
                )
                self.metrics["semantic_dedup_skipped"] = True
                return articles, embeddings

        sim_matrix = np.dot(emb_norm, emb_norm.T)

        # Find duplicates (keep first/earliest article)
        keep_mask = np.ones(n, dtype=bool)
        for i in range(1, n):
            if not keep_mask[i]:
                continue
            earlier = np.where(keep_mask[:i])[0]
            if len(earlier) == 0:
                continue
            max_sim = sim_matrix[i, earlier].max()
            if max_sim >= threshold:
                keep_mask[i] = False

        # ── Diagnostics: similarity distribution ────────────────────────────
        max_sims = []
        for i in range(1, n):
            earlier = np.arange(i)
            if len(earlier) > 0:
                max_sims.append(float(sim_matrix[i, earlier].max()))

        if max_sims:
            pcts = [50, 75, 90, 95, 99]
            p_vals = np.percentile(max_sims, pcts)
            logger.info(
                f"  Similarity distribution: "
                f"P50={p_vals[0]:.3f} P75={p_vals[1]:.3f} P90={p_vals[2]:.3f} "
                f"P95={p_vals[3]:.3f} P99={p_vals[4]:.3f} (threshold={threshold})"
            )
            arr = np.array(max_sims)
            for t_val in [0.70, 0.75, 0.78, 0.82, 0.85]:
                count = int((arr >= t_val).sum())
                logger.debug(f"    At threshold {t_val}: {count} would be removed")

        keep_idx = np.where(keep_mask)[0]
        dedup_articles = [articles[i] for i in keep_idx]
        dedup_embeddings = [embeddings[i] for i in keep_idx]

        # ── Safety gate 2: cap max removal at 50% ──────────────────────────
        total_removed = n - len(dedup_articles)
        removal_pct = total_removed / n if n > 0 else 0
        if removal_pct > 0.50:
            logger.warning(
                f"Phase 3.5: removing {removal_pct:.0%} of articles is excessive. "
                f"Raising threshold to reduce false positives."
            )
            # Re-run with a higher threshold
            higher_threshold = min(threshold + 0.07, 0.92)
            keep_mask = np.ones(n, dtype=bool)
            for i in range(1, n):
                if not keep_mask[i]:
                    continue
                earlier = np.where(keep_mask[:i])[0]
                if len(earlier) == 0:
                    continue
                max_sim = sim_matrix[i, earlier].max()
                if max_sim >= higher_threshold:
                    keep_mask[i] = False
            keep_idx = np.where(keep_mask)[0]
            dedup_articles = [articles[i] for i in keep_idx]
            dedup_embeddings = [embeddings[i] for i in keep_idx]
            total_removed = n - len(dedup_articles)
            logger.info(f"  Re-run with threshold={higher_threshold}: {total_removed} removed")

        for article, emb in zip(dedup_articles, dedup_embeddings):
            article.title_embedding = emb

        # Re-attach overflow articles that weren't part of NxN dedup
        if overflow_articles:
            dedup_articles.extend(overflow_articles)
            dedup_embeddings.extend(overflow_embeddings)

        elapsed = time.time() - t
        logger.info(
            f"Phase 3.5 (semantic dedup): {n + len(overflow_articles)} → {len(dedup_articles)} "
            f"({total_removed} removed) in {elapsed:.2f}s"
        )
        return dedup_articles, dedup_embeddings

    def _phase_cmi_relevance(
        self, articles: List[NewsArticle], embeddings: List[List[float]],
        threshold: float = None,
    ) -> tuple:
        """Phase 3.7: CMI Service Relevance SCORER (not a gate).

        Scores each article's relevance to CMI services via cosine similarity.
        ALL articles are kept — low-relevance clusters score lower in composite
        signals instead of being dropped before clustering.

        Previously this was a filter that dropped 30-50% of articles, fragmenting
        trends before Leiden even got a chance. Now it's a scorer that feeds into
        actionability via the cmi_relevance signal factor.

        Cost: ~4s (embedding CMI descriptions via NVIDIA API, then matrix multiply).
        """
        if threshold is None:
            threshold = self.cmi_relevance_threshold
        t = time.time()
        from app.config import CMI_SERVICES

        # Build CMI service embeddings (cached on engine instance)
        if not hasattr(self, '_cmi_service_embeddings'):
            provider_before = getattr(self.embedding_tool, '_active_provider', None)

            service_texts = []
            service_labels = []
            for svc_key, svc_info in CMI_SERVICES.items():
                name = svc_info["name"]
                for offering in svc_info["offerings"]:
                    service_texts.append(f"{name}: {offering}")
                    service_labels.append(svc_key)
                kws = svc_info.get("keywords", [])
                if kws:
                    service_texts.append(f"{name}: {', '.join(kws)}")
                    service_labels.append(svc_key)

            svc_embs = np.array(self.embedding_tool.embed_batch(service_texts))

            provider_after = getattr(self.embedding_tool, '_active_provider', None)
            if provider_before and provider_after and provider_before != provider_after:
                logger.error(
                    f"Phase 3.7 PROVIDER MISMATCH: articles used '{provider_before}', "
                    f"CMI used '{provider_after}'. Skipping CMI scoring."
                )
                return articles, embeddings

            norms = np.linalg.norm(svc_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._cmi_service_embeddings = svc_embs / norms
            self._cmi_service_labels = service_labels
            logger.info(
                f"Phase 3.7: Embedded {len(service_texts)} CMI service descriptions "
                f"({len(CMI_SERVICES)} services, provider={provider_after})"
            )

        # Compute cosine similarity: each article vs all service embeddings
        emb_array = np.array(embeddings)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_norm = emb_array / norms

        sims = np.dot(emb_norm, self._cmi_service_embeddings.T)
        max_sims = sims.max(axis=1)
        best_service_idx = sims.argmax(axis=1)

        # Score all articles AND gate low-CMI articles.
        # V4: Apply hard floor to ALL event types (not just "general").
        # Also blocklist non-business event types entirely.
        cmi_hard_floor = getattr(self, 'cmi_hard_floor', 0.25)

        # Blocklist: event types that are never relevant to B2B sales
        _CMI_BLOCKLIST = frozenset({
            "entertainment", "sports", "celebrity", "gossip",
            "us_media_m_and_a", "investment_portfolio",
            "lifestyle", "weather", "obituary",
        })

        low_relevance_count = 0
        dropped_count = 0
        dropped_titles = []
        keep_mask = []

        for i, article in enumerate(articles):
            score = float(max_sims[i])
            best_svc = self._cmi_service_labels[int(best_service_idx[i])]
            article._cmi_relevance_score = score
            article._best_cmi_service = best_svc
            if score < threshold:
                low_relevance_count += 1

            # V4: blocklist check — always drop non-business content
            etype = getattr(article, '_trigger_event', 'general')
            if etype in _CMI_BLOCKLIST:
                keep_mask.append(False)
                dropped_count += 1
                dropped_titles.append(f"[blocklist:{etype}] {article.title[:50] if article.title else '?'}")
            # V4: hard floor for ALL event types (not just "general")
            elif score < cmi_hard_floor:
                keep_mask.append(False)
                dropped_count += 1
                dropped_titles.append(f"[cmi<{cmi_hard_floor}:{etype}] {article.title[:50] if article.title else '?'}")
            else:
                keep_mask.append(True)

        # Apply gate
        if dropped_count > 0:
            articles = [a for a, keep in zip(articles, keep_mask) if keep]
            embeddings = [e for e, keep in zip(embeddings, keep_mask) if keep]
            max_sims = max_sims[np.array(keep_mask)]

        elapsed = time.time() - t
        self.metrics["phase_times"]["cmi_relevance"] = round(elapsed, 3)
        self.metrics["article_counts"]["after_cmi_scorer"] = len(articles)
        self.metrics["cmi_low_relevance"] = low_relevance_count
        self.metrics["article_counts"]["cmi_hard_floor_dropped"] = dropped_count

        logger.info(
            f"Phase 3.7 (CMI scorer): {len(articles) + dropped_count} articles scored, "
            f"{low_relevance_count} below {threshold}, "
            f"{dropped_count} dropped (general + CMI<{cmi_hard_floor}) "
            f"in {elapsed:.3f}s"
        )

        if dropped_count and dropped_titles:
            for dt in dropped_titles[:5]:
                logger.debug(f"  CMI gate dropped: {dt}")

        if len(max_sims) > 0:
            pcts = np.percentile(max_sims, [10, 25, 50, 75, 90])
            logger.info(
                f"  CMI similarity: P10={pcts[0]:.3f} P25={pcts[1]:.3f} "
                f"P50={pcts[2]:.3f} P75={pcts[3]:.3f} P90={pcts[4]:.3f}"
            )

        return articles, embeddings

    def _phase_cluster_leiden(
        self,
        articles: List[NewsArticle],
        embeddings: List[List[float]],
    ):
        """Phase 4+5: Leiden clustering on k-NN graph.

        Optuna path (default): Bayesian optimization of (k, resolution,
        min_community_size) jointly. Warm-starts from last run's best params.
        Fallback path: Binary search on resolution only.

        Hybrid similarity: instead of augmenting embeddings with one-hot
        event-type + entity fingerprint dimensions (which get drowned in
        1024-dim space), we compute an explicit N×N blended similarity
        matrix:  55% embedding cosine + 25% title/entity Jaccard + 15%
        event-type match + 5% temporal proximity.  This breaks the "India
        news gravity well" where stylistically similar articles (e.g.,
        "Tata Punch EV" and "Barpeta robbery") end up in the same cluster.
        """
        t = time.time()
        emb_array = np.array(embeddings, dtype=np.float32)
        n_articles = len(emb_array)

        if n_articles < 3:
            return np.full(n_articles, -1, dtype=int), n_articles

        # ── Hybrid similarity matrix ─────────────────────────────────────
        # Blends 4 signals into an N×N matrix that Leiden uses to build
        # its k-NN graph.  Original 1024-dim embeddings are preserved for
        # coherence validation (quality metrics operate in embedding space).
        event_types = [
            getattr(a, '_trigger_event', 'general') or 'general'
            for a in articles
        ]
        self.metrics["event_type_augmentation"] = {
            "n_specific": sum(
                1 for et in event_types if et not in _EVENT_NEUTRAL_TYPES
            ),
            "n_neutral": sum(
                1 for et in event_types if et in _EVENT_NEUTRAL_TYPES
            ),
            "unique_types": sorted(set(event_types)),
        }

        hybrid_sim = compute_hybrid_similarity(articles, emb_array)
        self.metrics["hybrid_similarity"] = {
            "shape": list(hybrid_sim.shape),
            "avg": round(float(hybrid_sim[np.triu_indices(n_articles, k=1)].mean()), 4),
            "std": round(float(hybrid_sim[np.triu_indices(n_articles, k=1)].std()), 4),
        }

        from app.trends.clustering import (
            cluster_leiden, auto_resolve_resolution, compute_leiden_quality,
            optimize_leiden, load_last_best_params, load_best_params_for_data,
            compute_meta_features,
        )
        from app.config import get_settings
        _s = get_settings()

        # Compute meta-features for similarity-based warm-starting
        # (use original embeddings — meta-features characterize the dataset)
        meta_features = compute_meta_features(emb_array)
        self.metrics["meta_features"] = meta_features

        # Optuna path: jointly optimize k, resolution, min_community_size
        if _s.leiden_optuna_enabled:
            # Try meta-feature matching first, fall back to most recent
            warm_params = load_best_params_for_data(meta_features)
            if warm_params:
                logger.info(f"Optuna warm-start (meta-feature match): {warm_params}")
            else:
                warm_params = load_last_best_params()
                if warm_params:
                    logger.info(f"Optuna warm-start from previous best: {warm_params}")

            # Double Optuna trials if EWMA drift was detected last run
            n_trials = _s.leiden_optuna_trials
            try:
                from app.learning.pipeline_metrics import load_history as _lh
                last_runs = _lh(last_n=1)
                if last_runs and last_runs[-1].get("ewma_drift_detected"):
                    n_trials = min(n_trials * 2, 50)
                    logger.info(
                        f"EWMA drift on last run — doubling Optuna trials "
                        f"to {n_trials}"
                    )
            except Exception:
                pass

            best = optimize_leiden(
                emb_array,
                n_trials=n_trials,
                seed=self.leiden_seed,
                warm_start_params=warm_params,
                timeout=_s.leiden_optuna_timeout,
                precomputed_sim=hybrid_sim,
            )

            # Use optimized params for final clustering
            k = best["k"]
            resolution = best["resolution"]
            min_community = best["min_community_size"]

            # Store for metrics logging
            self.metrics["optuna_best_params"] = {
                "k": k, "resolution": round(resolution, 4),
                "min_community_size": min_community,
                "score": best["score"],
            }

        else:
            # Fallback: binary search on resolution only
            k = self.leiden_k
            min_community = self.leiden_min_community_size
            resolution = self.leiden_resolution
            if self.leiden_auto_resolution:
                resolution = auto_resolve_resolution(
                    emb_array, k=k, seed=self.leiden_seed,
                    precomputed_sim=hybrid_sim,
                )
                logger.info(f"Auto-resolved Leiden resolution={resolution:.2f}")

        labels, noise_count, leiden_metrics = cluster_leiden(
            emb_array,
            k=k,
            resolution=resolution,
            seed=self.leiden_seed,
            min_community_size=min_community,
            precomputed_sim=hybrid_sim,
        )

        # Store for coherence validation
        self._active_min_cluster_size = min_community

        # Quality metrics — use ORIGINAL embeddings (1024-dim) for coherence
        # measurement, not the hybrid space.
        quality = compute_leiden_quality(emb_array, labels)

        # ── CROSS-SOURCE VALIDATION ────────────────────────────────────────
        # Inspired by nyan (production news aggregator): a valid news cluster
        # should contain articles from at least 2 different sources. A cluster
        # with articles only from one RSS feed is likely an editorial series or
        # section grouping, not a genuine multi-reported event.
        #
        # Demote single-source clusters to noise (label = -1).
        min_sources = 2
        demoted_clusters = 0
        demoted_articles = 0
        for cl_id in range(leiden_metrics["n_clusters"]):
            members = np.where(labels == cl_id)[0]
            sources = set()
            for idx in members:
                src = getattr(articles[idx], "source_id", "") or getattr(articles[idx], "source_name", "")
                if src:
                    sources.add(src)
            if len(sources) < min_sources and len(sources) > 0:
                for idx in members:
                    labels[idx] = -1
                    noise_count += 1
                demoted_clusters += 1
                demoted_articles += len(members)

        if demoted_clusters > 0:
            # Recount clusters after demotion
            valid_labels = set(labels[labels >= 0].tolist())
            # Relabel to contiguous 0..n_clusters-1
            if valid_labels:
                label_map = {old: new for new, old in enumerate(sorted(valid_labels))}
                new_labels = np.full_like(labels, -1)
                for i, l in enumerate(labels):
                    if l >= 0:
                        new_labels[i] = label_map[l]
                labels = new_labels
            leiden_metrics["n_clusters"] = len(valid_labels)
            logger.info(
                f"Cross-source validation: demoted {demoted_clusters} single-source "
                f"clusters ({demoted_articles} articles) to noise"
            )

        elapsed = time.time() - t
        n_clusters = leiden_metrics["n_clusters"]

        self.metrics["phase_times"]["cluster"] = round(elapsed, 2)
        self.metrics["n_clusters"] = n_clusters
        self.metrics["noise_count"] = noise_count
        self.metrics["demoted_single_source"] = demoted_clusters
        self.metrics["leiden"] = leiden_metrics
        self.metrics["cluster_quality"] = quality

        if quality.get("avg_coherence"):
            logger.info(
                f"Leiden: {n_clusters} clusters, "
                f"coherence={quality['avg_coherence']:.3f}, "
                f"modularity={leiden_metrics['modularity']:.3f}, "
                f"k={k}, res={resolution:.3f}, "
                f"{noise_count} noise in {elapsed:.2f}s"
            )

        return labels, noise_count

    def _group_by_cluster(
        self, articles: List[NewsArticle], labels: np.ndarray
    ) -> Dict[int, List[NewsArticle]]:
        """Group articles by cluster label. Noise (label=-1) excluded."""
        groups: Dict[int, List[NewsArticle]] = defaultdict(list)
        for article, label in zip(articles, labels):
            if label >= 0:
                groups[int(label)].append(article)
        return dict(groups)

    def _phase_keywords(
        self, cluster_articles: Dict[int, List[NewsArticle]]
    ) -> Dict[int, List[str]]:
        """Phase 6: Extract per-cluster keywords (c-TF-IDF + NER + MMR)."""
        t = time.time()
        result = self.keyword_extractor.extract_from_articles(cluster_articles)
        elapsed = time.time() - t
        self.metrics["phase_times"]["keywords"] = round(elapsed, 2)
        logger.info(f"Phase 6 (keywords): {len(result)} clusters in {elapsed:.2f}s")
        return result

    def _phase_signals(
        self, cluster_articles: Dict[int, List[NewsArticle]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Phase 7: Compute trend signals with two-pass percentile classification (T2).

        Pass 1: Compute all signals for each cluster (including initial classification).
        Pass 2: Compute actual P10/P50 from the distribution of trend_scores,
                 then re-classify signal strength using BERTrend true percentiles.

        This fixes the hardcoded P10=0.2, P50=0.5 constants. Now the thresholds
        adapt to the actual data distribution each run.
        """
        t = time.time()

        # Pass 1: Compute all signals (includes initial classification with defaults)
        # V7: Pass all_articles for dynamic credibility scoring (cross-citation + originality)
        all_arts = getattr(self, '_all_articles', None)
        cluster_signals = {cid: compute_all_signals(arts, all_articles=all_arts) for cid, arts in cluster_articles.items()}

        # Pass 2: True percentile classification (T2 — BERTrend P10/P50)
        # Uses EMA-smoothed P10/P50 from pipeline history when available.
        # This run's percentiles are blended 30/70 with historical EMA.
        from app.trends.signals.composite import compute_percentiles, classify_signal_strength
        trend_scores = [s.get("trend_score", 0.0) for s in cluster_signals.values()]

        percentiles = compute_percentiles(trend_scores)
        p10_raw = percentiles["p10"]
        p50_raw = percentiles["p50"]

        # Blend with EMA history (loaded during __init__ via adaptive thresholds)
        adaptive = self.metrics.get("adaptive_thresholds", {})
        p10_ema = adaptive.get("signal_p10")
        p50_ema = adaptive.get("signal_p50")
        alpha = 0.3  # 30% this run, 70% history

        if p10_ema is not None and p50_ema is not None:
            p10 = alpha * p10_raw + (1 - alpha) * p10_ema
            p50 = alpha * p50_raw + (1 - alpha) * p50_ema
            logger.info(f"Signal P10/P50: raw={p10_raw:.3f}/{p50_raw:.3f} -> EMA-blended={p10:.3f}/{p50:.3f}")
        else:
            p10, p50 = p10_raw, p50_raw

        self.metrics["percentile_p10"] = p10
        self.metrics["percentile_p50"] = p50
        self.metrics["percentile_distribution"] = percentiles

        # Re-classify each cluster with (EMA-smoothed) percentiles
        reclassified = 0
        for cid, sigs in cluster_signals.items():
            old_strength = sigs.get("signal_strength", "noise")
            new_strength = classify_signal_strength(
                popularity_score=sigs.get("trend_score", 0.0),
                acceleration=sigs.get("acceleration", 0.0),
                p10=p10,
                p50=p50,
            )
            if new_strength.value != old_strength:
                reclassified += 1
            sigs["signal_strength"] = new_strength.value

        elapsed = time.time() - t
        self.metrics["phase_times"]["signals"] = round(elapsed, 2)
        logger.info(
            f"Phase 7 (signals): {len(cluster_signals)} clusters in {elapsed:.2f}s "
            f"| P10={p10:.3f}, P50={p50:.3f}, {reclassified} reclassified"
        )
        return cluster_signals

    def _compute_cluster_quality(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_signals: Dict[int, Dict[str, Any]],
        embeddings,
        articles: List[NewsArticle],
        labels: np.ndarray,
    ) -> None:
        """Compute rich cluster quality metrics and inject into per-cluster signals.

        V7: Now generates full ClusterQualityReport with:
        - Coherence score (mean pairwise cosine)
        - Entity consistency (% sharing top entities)
        - Keyword consistency
        - Temporal spread
        - Source diversity
        - Quality grade (A/B/C/D/F)
        - Human-readable quality reasoning
        """
        try:
            emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
            aid_to_idx = {id(a): i for i, a in enumerate(articles)}

            cosine_sims = []
            grades = []
            for cid, arts in cluster_articles.items():
                idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]

                # Get keywords for this cluster (from keyword extraction phase)
                kws = cluster_signals.get(cid, {}).get("keywords", [])

                # Generate full quality report
                report = compute_cluster_quality_report(
                    articles=arts,
                    embeddings=emb_array,
                    article_indices=idxs,
                    keywords=kws,
                )

                # Inject into cluster signals
                if cid in cluster_signals:
                    cluster_signals[cid]["intra_cluster_cosine"] = report["coherence_score"]
                    cluster_signals[cid]["cluster_quality"] = report

                cosine_sims.append(report["coherence_score"])
                grades.append(report["quality_grade"])

            if cosine_sims:
                avg_sim = sum(cosine_sims) / len(cosine_sims)
                self.metrics["avg_intra_cluster_cosine"] = round(avg_sim, 4)
                self.metrics["min_intra_cluster_cosine"] = round(min(cosine_sims), 4)
                grade_counts = Counter(grades)
                logger.info(
                    f"Cluster quality: avg coherence={avg_sim:.3f}, "
                    f"min={min(cosine_sims):.3f}, "
                    f"grades={dict(grade_counts)}, "
                    f"DBCV={self.metrics.get('dbcv_score', 'N/A')}"
                )
        except Exception as e:
            logger.warning(f"Cluster quality computation failed: {e}")

    def _phase_coherence_validation(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        embeddings: np.ndarray,
        articles: List[NewsArticle],
        labels: np.ndarray,
    ) -> tuple:
        """Phase 5.5: Validate and refine clusters using original-space coherence.

        Splits incoherent clusters (UMAP artifacts), merges redundant ones,
        rejects noise masquerading as clusters. All in original 1024-dim space.
        """
        t = time.time()
        try:
            # Phase 5B: Scale coherence thresholds by dataset size.
            # The adaptive EMA thresholds are calibrated on large datasets (700+ articles).
            # On fresh/small datasets (< 300 articles), these thresholds over-filter.
            n_arts = len(articles)
            if n_arts < 150:
                size_scale = 0.70   # 30% relaxation: very small datasets (< 150)
            elif n_arts < 350:
                size_scale = 0.78   # 22% relaxation: fresh daily data (EMA calibrated on 700+)
            elif n_arts < 500:
                size_scale = 0.88   # 12% relaxation: medium datasets
            elif n_arts < 700:
                size_scale = 0.94   # 6% relaxation: large-ish datasets
            else:
                size_scale = 1.0    # No relaxation for full cached runs (700+)

            scaled_coherence_min = self.coherence_min * size_scale
            scaled_reject = self.coherence_reject * size_scale

            if size_scale < 1.0:
                logger.info(
                    f"Phase 5B: {n_arts} articles → threshold scale={size_scale:.2f} "
                    f"(coherence_min: {self.coherence_min:.3f}→{scaled_coherence_min:.3f}, "
                    f"reject: {self.coherence_reject:.3f}→{scaled_reject:.3f})"
                )

            refined_articles, refined_labels, noise_delta = validate_and_refine_clusters(
                cluster_articles=cluster_articles,
                embeddings=embeddings,
                articles=articles,
                labels=labels,
                min_coherence=scaled_coherence_min,
                reject_threshold=scaled_reject,
                merge_threshold=self.merge_threshold,
                min_cluster_size=getattr(self, '_active_min_cluster_size', 3),
            )
            elapsed = time.time() - t
            self.metrics["phase_times"]["coherence_validation"] = round(elapsed, 2)
            self.metrics["coherence_splits"] = len(refined_articles) - len(cluster_articles)
            self.metrics["coherence_noise_delta"] = noise_delta
            logger.info(
                f"Phase 5.5 (coherence): {len(cluster_articles)} → {len(refined_articles)} clusters, "
                f"noise delta={noise_delta:+d} in {elapsed:.2f}s"
            )
            return refined_articles, refined_labels, noise_delta
        except Exception as e:
            elapsed = time.time() - t
            self.metrics["phase_times"]["coherence_validation"] = round(elapsed, 2)
            logger.warning(f"Phase 5.5 (coherence) failed: {e}")
            return cluster_articles, labels, 0

    async def _phase_synthesize(
        self, cluster_articles: Dict[int, List[NewsArticle]],
        cluster_keywords: Dict[int, List[str]],
    ) -> Dict[int, Dict[str, Any]]:
        """Phase 8: LLM synthesis (delegated to synthesis module)."""
        logger.info(f">> Phase 8 (synthesis): {len(cluster_articles)} clusters — LLM calls, slowest phase...")
        t = time.time()
        result = await synthesize_clusters(
            cluster_articles, cluster_keywords, self.llm_tool, self.max_concurrent_llm,
            mock_mode=self.mock_mode,
        )
        elapsed = time.time() - t
        self.metrics["phase_times"]["synthesize"] = round(elapsed, 2)
        logger.info(f"Phase 8 (synthesis): {len(result)} clusters in {elapsed:.2f}s")
        return result

    def _phase_quality_gate(
        self,
        cluster_summaries: Dict[int, Dict[str, Any]],
        cluster_signals: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        V9: Quality gate — drop empty/low-quality synthesis results before tree assembly.

        Drops clusters where:
        1. Synthesis returned empty dict (LLM failure or V10 rejection)
        2. Synthesis has no title or summary (critical fields missing)
        3. Overall confidence is below MIN_SYNTHESIS_CONFIDENCE threshold

        This prevents garbage data from propagating to impact/company agents.
        """
        t = time.time()

        # Load thresholds from config
        min_confidence = 0.3
        try:
            from app.config import get_settings as _gs_qg
            min_confidence = _gs_qg().min_synthesis_confidence
        except Exception:
            pass

        original_count = len(cluster_summaries)
        filtered = {}
        dropped_empty = 0
        dropped_no_title = 0
        dropped_low_confidence = 0

        for cid, summary in cluster_summaries.items():
            # Gate 1: Empty synthesis (LLM failure or V10 REJECT)
            if not summary:
                dropped_empty += 1
                continue

            # Gate 2: Missing critical fields
            title = summary.get("trend_title", "")
            text_summary = summary.get("trend_summary", "")
            if not title or len(str(title).strip()) < 5:
                dropped_no_title += 1
                logger.debug(f"V9: Dropped cluster {cid} — no title")
                continue
            if not text_summary or len(str(text_summary).strip()) < 20:
                dropped_no_title += 1
                logger.debug(f"V9: Dropped cluster {cid} — summary too short")
                continue

            # Gate 3: Sigmoid confidence gate (replaces binary cutoff)
            # Instead of "reject if < 0.40", uses sigmoid for smooth transition.
            # Clusters below the adaptive P25 are strongly discounted but not
            # hard-rejected — they still appear with reduced scores.
            signals = cluster_signals.get(cid, {})
            confidence = signals.get("confidence", 1.0)

            # Also check V10 validation score if present
            val_meta = summary.get("_validation", {})
            val_score = val_meta.get("score", 1.0)
            effective_confidence = min(confidence, val_score) if val_meta else confidence

            # Adaptive center: use EMA'd P25 of confidence if available
            adaptive = self.metrics.get("adaptive_thresholds", {})
            gate_center = adaptive.get("confidence_p25", min_confidence)

            # Sigmoid: smooth transition around gate_center
            # temperature controls sharpness (smaller = sharper)
            temperature = 0.08
            gate_weight = 1.0 / (1.0 + math.exp(-(effective_confidence - gate_center) / temperature))

            # Hard floor: if gate_weight < 0.10, drop entirely (truly garbage)
            if gate_weight < 0.10:
                dropped_low_confidence += 1
                logger.debug(
                    f"V9: Dropped cluster {cid} — confidence {effective_confidence:.2f} "
                    f"(gate_weight={gate_weight:.3f} < 0.10)"
                )
                continue

            # Apply soft gate: discount actionability/trend scores
            if gate_weight < 0.95:
                signals["gate_weight"] = round(gate_weight, 3)
                if "actionability_score" in signals:
                    signals["actionability_score"] *= gate_weight
                if "trend_score" in signals:
                    signals["trend_score"] *= gate_weight

            filtered[cid] = summary

        elapsed = time.time() - t
        dropped_total = original_count - len(filtered)
        self.metrics["phase_times"]["quality_gate"] = round(elapsed, 4)

        if dropped_total > 0:
            logger.info(
                f"V9 quality gate: {len(filtered)}/{original_count} clusters passed "
                f"(dropped: {dropped_empty} empty, {dropped_no_title} no-title, "
                f"{dropped_low_confidence} low-confidence) in {elapsed:.3f}s"
            )
        else:
            logger.debug(f"V9 quality gate: all {original_count} clusters passed")

        return filtered

    async def _phase_validate_trends(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_keywords: Dict[int, List[str]],
        cluster_signals: Dict[int, Dict[str, Any]],
        cluster_summaries: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Phase 8.5: AI Council validates trend importance and hierarchy."""
        logger.info(f">> Phase 8.5 (validate): {len(cluster_summaries)} trends — AI council LLM calls...")
        t = time.time()
        try:
            from app.agents.workers.council.trend_validator import validate_trends

            # Build input data for the validator
            trends_data = []
            for cid in sorted(cluster_articles.keys()):
                arts = cluster_articles[cid]
                sigs = cluster_signals.get(cid, {})
                summary = cluster_summaries.get(cid, {})
                kws = cluster_keywords.get(cid, [])

                trends_data.append({
                    "trend_id": str(cid),
                    "trend_title": summary.get("trend_title") or (f"{kws[0].capitalize()}, {kws[1].capitalize()} and {kws[2].capitalize()} Developments" if len(kws) >= 3 else ", ".join(kws[:3])),
                    "summary": summary.get("trend_summary", "")[:500],
                    "article_count": len(arts),
                    "source_diversity": sigs.get("source_diversity", 0.0),
                    "keywords": kws[:10],
                    "entities": summary.get("key_entities", [])[:10],
                    "coherence_score": sigs.get("intra_cluster_cosine", 0.5),
                    "signal_strength": sigs.get("signal_strength", "noise"),
                    "article_titles": [a.title for a in arts[:8]],
                })

            validations = await validate_trends(trends_data, llm_service=self.llm_tool)

            # Map back to cluster IDs
            result = {}
            cids = sorted(cluster_articles.keys())
            for i, val in enumerate(validations):
                if i < len(cids):
                    result[cids[i]] = val.model_dump()

            elapsed = time.time() - t
            self.metrics["phase_times"]["trend_validation"] = round(elapsed, 2)

            # Log summary
            depth_counts = {}
            for v in result.values():
                d = v.get("validated_depth", "?")
                depth_counts[d] = depth_counts.get(d, 0) + 1
            logger.info(
                f"Phase 8.5 (AI trend validation): {len(result)} trends validated "
                f"in {elapsed:.1f}s — {depth_counts}"
            )
            return result

        except Exception as e:
            elapsed = time.time() - t
            self.metrics["phase_times"]["trend_validation"] = round(elapsed, 2)
            logger.warning(f"Phase 8.5 (trend validation) failed: {e}. Using volume-based fallback.")
            return {}

    def _phase_tree(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_keywords: Dict[int, List[str]],
        cluster_signals: Dict[int, Dict[str, Any]],
        cluster_summaries: Dict[int, Dict[str, Any]],
        noise_count: int,
        total_articles: int,
        cluster_validations: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> TrendTree:
        """Phase 9: Assemble TrendTree (delegated to tree_builder module)."""
        t = time.time()
        tree = build_trend_tree(
            cluster_articles, cluster_keywords, cluster_signals,
            cluster_summaries, noise_count, total_articles,
            cluster_validations=cluster_validations,
        )
        elapsed = time.time() - t
        self.metrics["phase_times"]["tree"] = round(elapsed, 2)
        logger.info(f"Phase 9 (tree): {len(tree.root_ids)} trends, {noise_count} noise in {elapsed:.2f}s")
        return tree


# ── Backward compatibility alias ─────────────────────────────────────────
RecursiveTrendEngine = TrendPipeline


# ── Convenience function ─────────────────────────────────────────────────


async def detect_trend_tree(
    articles: List[NewsArticle],
    min_cluster_size: int = 5,
    max_depth: int = 3,
    mock_mode: bool = False,
    country: str = "",
    domestic_source_ids: Optional[set] = None,
    use_cache: bool = False,
    cache_path: str = "./data/article_cache",
) -> TrendTree:
    """Quick function to run the full trend detection pipeline.

    All thresholds are loaded from config.py / .env by the engine constructor.
    """
    from app.config import get_settings
    s = get_settings()
    pipeline = TrendPipeline(
        min_cluster_size=min_cluster_size,
        max_depth=max_depth,
        semantic_dedup_threshold=s.semantic_dedup_threshold,
        max_concurrent_llm=s.engine_max_concurrent_llm,
        mock_mode=mock_mode,
        country=country,
        domestic_source_ids=domestic_source_ids,
    )
    return await pipeline.run(articles, use_cache=use_cache, cache_path=cache_path)
