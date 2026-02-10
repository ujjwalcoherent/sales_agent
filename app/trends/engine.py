"""
RecursiveTrendEngine — core trend detection pipeline.

Articles → UMAP+HDBSCAN clustering → signal scoring → LLM synthesis → TrendTree.

REF: BERTopic (Grootendorst 2022), BERTrend (Boutaleb et al. 2024),
     HDBSCAN (Campello et al. 2013), UMAP (McInnes et al. 2018).
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

from app.schemas.base import ConfidenceScore, Sector
from app.schemas.news import NewsArticle
from app.schemas.trends import SignalStrength, TrendDepth, TrendNode, TrendTree
from app.news.dedup import ArticleDeduplicator
from app.news.entity_extractor import EntityExtractor
from app.news.scraper import scrape_articles
from app.news.event_classifier import EmbeddingEventClassifier
from app.news.entity_cooccurrence import EntityCooccurrenceTracker
from app.tools.embeddings import EmbeddingTool
from app.trends.reduction import DimensionalityReducer
from app.trends.keywords import KeywordExtractor
from app.trends.signals import compute_all_signals
from app.trends.tree_builder import build_trend_tree
from app.trends.synthesis import synthesize_clusters, synthesize_cluster
from app.trends.subclustering import recursive_subcluster
from app.trends.coherence import validate_and_refine_clusters
from app.trends.trend_memory import TrendMemory

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
logger.propagate = False



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


class RecursiveTrendEngine:
    """Full trend detection pipeline: articles → TrendTree."""

    def __init__(
        self,
        dedup_threshold: float = 0.25,
        dedup_shingle_size: int = 2,
        semantic_dedup_threshold: float = 0.78,
        spacy_model: str = "en_core_web_sm",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
        umap_metric: str = "cosine",
        min_cluster_size: int = 5,
        min_samples: int = 3,
        cluster_selection_method: str = "eom",
        max_depth: int = 3,
        max_concurrent_llm: int = 14,
        subcluster_min_coherence: float = 0.20,
        subcluster_min_differentiation: float = 0.08,
        subcluster_min_articles: int = 4,
        llm_tool=None,
        mock_mode: bool = False,
        country: str = "",
        domestic_source_ids: Optional[set] = None,
    ):
        self.deduplicator = ArticleDeduplicator(
            threshold=dedup_threshold, shingle_size=dedup_shingle_size,
        )
        self.entity_extractor = EntityExtractor(model_name=spacy_model)
        self.embedding_tool = EmbeddingTool()
        self.reducer = DimensionalityReducer(
            n_components=umap_n_components, n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist, metric=umap_metric,
        )
        self.keyword_extractor = KeywordExtractor()

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.max_depth = max_depth
        self.max_concurrent_llm = max_concurrent_llm
        self.min_subcluster_size = max(min_cluster_size * 2, 10)
        self.subcluster_min_coherence = subcluster_min_coherence
        self.subcluster_min_differentiation = subcluster_min_differentiation
        self.subcluster_min_articles = subcluster_min_articles

        self._llm_tool = llm_tool
        self.mock_mode = mock_mode
        self.metrics: Dict[str, Any] = {"phase_times": {}, "article_counts": {}}

        # Cluster coherence validation thresholds
        self.coherence_min = 0.35           # Clusters below this get split
        self.coherence_reject = 0.20        # Clusters below this → noise
        self.merge_threshold = 0.70         # Clusters more similar → merge

        # Historical trend memory
        self.trend_memory = TrendMemory()

        # Geographic relevance filter (dynamic, entity-based)
        self._target_country = country
        self._domestic_source_ids = domestic_source_ids or set()

    @property
    def llm_tool(self):
        if self._llm_tool is None:
            from app.tools.llm_tool import LLMTool
            self._llm_tool = LLMTool(mock_mode=self.mock_mode)
        return self._llm_tool

    # ════════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ════════════════════════════════════════════════════════════════════

    async def run(self, articles: List[NewsArticle]) -> TrendTree:
        """Execute the full pipeline: articles → TrendTree."""
        total_start = time.time()
        self.metrics = {"phase_times": {}, "article_counts": {"input": len(articles)}}

        if not articles:
            logger.warning("No articles provided")
            return TrendTree(root_ids=[], nodes={})

        # Pre-processing: scrape + event classify run IN PARALLEL
        # Scrape enriches article.content (I/O-bound, async).
        # Classify uses title+summary only (CPU-bound, from RSS data).
        # Neither depends on the other → safe to overlap.
        scrape_task = asyncio.create_task(self._phase_scrape(articles))
        classify_task = asyncio.to_thread(self._phase_classify_events, articles)
        await asyncio.gather(scrape_task, classify_task)
        articles = scrape_task.result()

        articles = self._phase_relevance_filter(articles)
        articles = self._phase_dedup(articles)
        articles = self._phase_ner(articles)
        articles = self._phase_geo_filter(articles)
        self._phase_entity_cooccurrence(articles)
        self._phase_article_sentiment(articles)

        # Embedding + dedup
        embeddings = self._phase_embed(articles)
        articles, embeddings = self._phase_semantic_dedup(
            articles, embeddings, threshold=self.semantic_dedup_threshold
        )
        self.metrics["article_counts"]["after_semantic_dedup"] = len(articles)
        self._all_articles = articles

        # Clustering pipeline
        reduced = self._phase_reduce(embeddings)
        labels, noise_count = self._phase_cluster(reduced)

        # Phase 5.5: Coherence validation — split incoherent, merge redundant
        # Uses ORIGINAL embeddings (not UMAP-reduced) for accurate similarity
        cluster_articles = self._group_by_cluster(articles, labels)
        cluster_articles, labels, noise_delta = self._phase_coherence_validation(
            cluster_articles, embeddings, articles, labels
        )
        noise_count += noise_delta

        cluster_keywords = self._phase_keywords(cluster_articles)

        # Signal computation + cluster quality + search interest validation
        cluster_signals = self._phase_signals(cluster_articles)
        self._inject_entity_graph_signals(cluster_articles, cluster_signals)
        self._compute_cluster_quality(cluster_articles, cluster_signals, embeddings, articles, labels)
        self._phase_search_interest(cluster_signals, cluster_keywords)

        # Phase 7.7: Historical trend memory — match against past trends
        self._phase_trend_memory(cluster_articles, cluster_signals, cluster_keywords, embeddings, articles)

        # LLM synthesis + V9 quality gate + tree assembly
        cluster_summaries = await self._phase_synthesize(cluster_articles, cluster_keywords)
        cluster_summaries = self._phase_quality_gate(cluster_summaries, cluster_signals)
        tree = self._phase_tree(
            cluster_articles, cluster_keywords, cluster_signals,
            cluster_summaries, noise_count, len(articles),
        )

        # Recursive sub-clustering
        if self.max_depth > 1:
            tree = await recursive_subcluster(self, tree, cluster_summaries)

        # Cross-trend linking (Feedly Leo-inspired entity linking)
        self._link_related_trends(tree)

        # Store trends in memory for future runs
        self._store_trends_to_memory(
            cluster_articles, cluster_summaries, cluster_keywords,
            cluster_signals, embeddings, articles,
        )

        total_time = time.time() - total_start
        self.metrics["total_seconds"] = round(total_time, 2)
        logger.info(
            f"Pipeline complete: {len(articles)} articles → "
            f"{len(tree.root_ids)} trends, {len(tree.nodes)} nodes "
            f"(depth {tree.max_depth_reached}) in {total_time:.1f}s"
        )
        return tree

    # ════════════════════════════════════════════════════════════════════
    # PHASE IMPLEMENTATIONS
    # ════════════════════════════════════════════════════════════════════

    async def _phase_scrape(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 0: Scrape full article content for richer embeddings."""
        t = time.time()
        try:
            enriched = await scrape_articles(articles, max_concurrent=10)
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

    def _phase_relevance_filter(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 0.7: TAG articles as business/non-business. Keeps ALL articles.

        No hardcoded regex. Uses embedding classification confidence to tag:
        - _is_business: True if matched event type or high confidence
        - _business_confidence: raw confidence score for downstream weighting
        Non-business articles stay in the pipeline — they provide context and
        clustering naturally separates them.
        """
        t = time.time()
        business_count = 0
        for a in articles:
            event = getattr(a, '_trigger_event', 'general')
            confidence = getattr(a, '_trigger_confidence', 0.0)
            a._is_business = (event != 'general') or (confidence >= 0.15)
            a._business_confidence = confidence
            if a._is_business:
                business_count += 1
        elapsed = time.time() - t
        self.metrics["phase_times"]["relevance_filter"] = round(elapsed, 2)
        self.metrics["article_counts"]["after_relevance"] = len(articles)
        self.metrics["article_counts"]["business_tagged"] = business_count
        logger.info(
            f"Phase 0.7 (relevance): {business_count}/{len(articles)} tagged as business "
            f"(all kept) in {elapsed:.2f}s"
        )
        return articles  # Return ALL — don't filter

    def _phase_dedup(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 1: Remove near-duplicate articles (MinHash LSH)."""
        t = time.time()
        result = self.deduplicator.deduplicate(articles)
        elapsed = time.time() - t
        self.metrics["phase_times"]["dedup"] = round(elapsed, 2)
        self.metrics["article_counts"]["after_dedup"] = len(result)
        logger.info(f"Phase 1 (dedup): {len(articles)} → {len(result)} in {elapsed:.2f}s")
        return result

    def _phase_ner(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Phase 2: Extract named entities (spaCy)."""
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

    def _phase_entity_cooccurrence(self, articles: List[NewsArticle]) -> None:
        """Phase 2.5: Build entity co-occurrence graph."""
        t = time.time()
        try:
            tracker = EntityCooccurrenceTracker()
            summary = tracker.process_articles(articles)
            self.metrics["phase_times"]["entity_cooccurrence"] = round(time.time() - t, 2)
            self.metrics["entity_graph"] = summary
            self._entity_tracker = tracker
            logger.info(
                f"Phase 2.5 (co-occurrence): {summary['total_entities']} entities, "
                f"{summary['total_edges']} edges in {time.time() - t:.3f}s"
            )
        except Exception as e:
            self.metrics["phase_times"]["entity_cooccurrence"] = round(time.time() - t, 2)
            logger.warning(f"Phase 2.5 failed: {e}")

    def _phase_article_sentiment(self, articles: List[NewsArticle]) -> None:
        """
        Phase 2.7: Pre-compute per-article sentiment using VADER.

        Populates article.sentiment_score BEFORE clustering so downstream
        signals (temporal histogram sentiment_avg, content signals) have
        real data instead of all-zeros.

        Performance: VADER processes ~10K articles/second. Negligible overhead.

        Edge cases:
        - Articles with existing non-zero sentiment_score → kept (already computed)
        - Articles with empty title+summary → sentiment = 0.0 (neutral)
        - VADER not installed → keyword fallback (from content.py)
        """
        t = time.time()
        try:
            from app.trends.signals.content import analyze_article_sentiment

            computed = 0
            skipped = 0
            for article in articles:
                # Only compute if not already set (avoid double-work)
                existing = getattr(article, 'sentiment_score', 0.0)
                if existing != 0.0:
                    skipped += 1
                    continue
                article.sentiment_score = analyze_article_sentiment(article)
                computed += 1

            elapsed = time.time() - t
            self.metrics["phase_times"]["article_sentiment"] = round(elapsed, 3)

            # Log distribution stats for observability
            scores = [a.sentiment_score for a in articles]
            if scores:
                mean_s = sum(scores) / len(scores)
                positive = sum(1 for s in scores if s > 0.05)
                negative = sum(1 for s in scores if s < -0.05)
                neutral = len(scores) - positive - negative
                logger.info(
                    f"Phase 2.7 (sentiment): {computed} computed, {skipped} pre-existing | "
                    f"mean={mean_s:.3f}, +{positive}/-{negative}/~{neutral} in {elapsed:.3f}s"
                )
            else:
                logger.info(f"Phase 2.7 (sentiment): no articles to score")

        except Exception as e:
            elapsed = time.time() - t
            self.metrics["phase_times"]["article_sentiment"] = round(elapsed, 3)
            logger.warning(f"Phase 2.7 (sentiment) failed: {e}")

    def _phase_embed(self, articles: List[NewsArticle]) -> List[List[float]]:
        """Phase 3: Content-aware embeddings (title + body + entities + event)."""
        t = time.time()

        # Get event descriptions for richer event-aware embeddings
        # V6: EVENT_DESCRIPTION_VARIANTS maps event → list of descriptions; use first variant
        from app.news.event_classifier import EVENT_DESCRIPTION_VARIANTS

        def _embed_text(a):
            title = a.title or ""
            body = a.content or a.summary or ""
            entities_str = ""
            if hasattr(a, 'entities') and a.entities:
                ent_names = [getattr(e, 'name', str(e)) for e in a.entities[:10]]
                entities_str = " ".join(ent_names)
            # Use the full event description (not just the tag) for event-aware embedding.
            # "regulation" → "Government regulation, compliance mandate, regulatory change..."
            # This gives the embedding model richer context about the event type.
            event = getattr(a, '_trigger_event', '')
            if event and event != 'general' and event in EVENT_DESCRIPTION_VARIANTS:
                event_str = f" {EVENT_DESCRIPTION_VARIANTS[event][0][:100]}"
            else:
                event_str = ""
            return f"{title}. {title}.{event_str} {entities_str} {body[:2000]}"

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
        MAX_DEDUP_ARTICLES = 2000
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

    def _phase_reduce(self, embeddings: List[List[float]]) -> np.ndarray:
        """Phase 4: UMAP dimensionality reduction."""
        t = time.time()
        if not embeddings:
            self.metrics["phase_times"]["reduce"] = 0
            return np.array([])
        reduced = self.reducer.reduce(embeddings)
        elapsed = time.time() - t
        self.metrics["phase_times"]["reduce"] = round(elapsed, 2)
        dim_in = len(embeddings[0]) if embeddings else 0
        dim_out = reduced.shape[1] if len(reduced.shape) > 1 else 0
        logger.info(f"Phase 4 (UMAP): {dim_in}-dim → {dim_out}-dim in {elapsed:.2f}s")
        return reduced

    def _phase_cluster(self, reduced: np.ndarray):
        """Phase 5: Auto-dynamic HDBSCAN clustering."""
        t = time.time()
        n_articles = len(reduced)

        if n_articles < 3:
            return np.full(n_articles, -1, dtype=int), n_articles

        # Auto-dynamic: sqrt(N)/2 for min_cluster_size, leaf method for news
        base_mcs = max(3, min(8, int(math.sqrt(n_articles) / 2)))
        auto_method = 'eom' if n_articles > 500 else 'leaf'
        # Store for coherence validation to use the same threshold
        self._active_min_cluster_size = base_mcs

        logger.info(
            f"Phase 5: auto-dynamic clustering: {n_articles} articles, "
            f"min_cluster_size={base_mcs}, method={auto_method}"
        )

        labels, noise_count = self._cluster_hdbscan(
            reduced, min_cluster_size_override=base_mcs,
            cluster_method_override=auto_method, cluster_selection_epsilon=0.0,
        )

        # Auto-recovery from degenerate results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = noise_count / max(n_articles, 1)

        if n_clusters == 0 or noise_pct > 0.8:
            relaxed = max(2, base_mcs - 1)
            logger.info(f"  Auto-recovery: {n_clusters} clusters, {noise_pct:.0%} noise → retry mcs={relaxed}")
            labels, noise_count = self._cluster_hdbscan(
                reduced, min_cluster_size_override=relaxed,
                cluster_method_override='leaf', cluster_selection_epsilon=0.0,
            )
        elif n_clusters == 1 and n_articles > 15:
            relaxed = max(2, base_mcs - 1)
            logger.info(f"  Auto-recovery: single mega-cluster → retry mcs={relaxed}, leaf")
            labels, noise_count = self._cluster_hdbscan(
                reduced, min_cluster_size_override=relaxed,
                cluster_method_override='leaf', cluster_selection_epsilon=0.0,
            )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        elapsed = time.time() - t
        self.metrics["phase_times"]["cluster"] = round(elapsed, 2)
        self.metrics["n_clusters"] = n_clusters
        self.metrics["noise_count"] = noise_count

        if n_clusters > 0:
            sizes = sorted(Counter(l for l in labels if l >= 0).values(), reverse=True)
            logger.info(
                f"Phase 5 (HDBSCAN): {n_clusters} clusters, "
                f"sizes={sizes[:5]}{'...' if len(sizes) > 5 else ''}, "
                f"{noise_count} noise ({noise_count*100//max(n_articles,1)}%) in {elapsed:.2f}s"
            )
        else:
            logger.info(f"Phase 5 (HDBSCAN): 0 clusters, {noise_count} noise in {elapsed:.2f}s")

        return labels, noise_count

    def _cluster_hdbscan(
        self, reduced: np.ndarray,
        min_cluster_size_override: int = None,
        cluster_method_override: str = None,
        cluster_selection_epsilon: float = 0.0,
        log_prefix: str = "",
    ):
        """HDBSCAN clustering with auto-dynamic parameters."""
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        mcs = min_cluster_size_override or self.min_cluster_size
        method = cluster_method_override or self.cluster_selection_method
        ms = 1 if method == 'leaf' else max(1, int(math.sqrt(mcs)))

        logger.debug(f"{log_prefix}HDBSCAN: n={len(reduced)}, mcs={mcs}, ms={ms}, method={method}, eps={cluster_selection_epsilon:.3f}")

        clusterer = SklearnHDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method=method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean', n_jobs=-1,
            store_centers='centroid',
        )
        labels = clusterer.fit_predict(reduced)
        noise_count = int(np.sum(labels == -1))

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Extract DBCV (Density-Based Cluster Validation) — the correct metric for HDBSCAN
        try:
            dbcv = float(clusterer.relative_validity_)
            self.metrics["dbcv_score"] = round(dbcv, 4)
            logger.debug(f"{log_prefix}DBCV score: {dbcv:.4f}")
        except Exception:
            pass

        logger.debug(f"{log_prefix}HDBSCAN result: {n_clusters} clusters, {noise_count} noise")
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
        cluster_signals = {cid: compute_all_signals(arts) for cid, arts in cluster_articles.items()}

        # Pass 2: True percentile classification (T2 — BERTrend P10/P50)
        from app.trends.signals.composite import compute_percentiles, classify_signal_strength
        trend_scores = [s.get("trend_score", 0.0) for s in cluster_signals.values()]

        percentiles = compute_percentiles(trend_scores)
        p10 = percentiles["p10"]
        p50 = percentiles["p50"]
        self.metrics["percentile_p10"] = p10
        self.metrics["percentile_p50"] = p50
        self.metrics["percentile_distribution"] = percentiles

        # Re-classify each cluster with actual percentiles
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

    def _inject_entity_graph_signals(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_signals: Dict[int, Dict[str, Any]],
    ) -> None:
        """
        Phase 7.1: Inject entity co-occurrence graph data into cluster signals.

        Uses the EntityCooccurrenceTracker built in Phase 2.5 to find:
        - Bridge entities: entities in THIS cluster that also appear in OTHER clusters
        - Intra-cluster entity edges: co-occurrence density within the cluster
        - Graph density: how interconnected the cluster's entities are

        Edge cases:
        - No entity tracker available → skip silently (Phase 2.5 may have failed)
        - Cluster with no entities → empty graph signals
        - Single-article cluster → no edges possible, density = 0
        """
        tracker = getattr(self, '_entity_tracker', None)
        if tracker is None:
            logger.debug("Phase 7.1 (entity graph): no tracker available, skipping")
            return

        t = time.time()

        # Build article_id → cluster_id mapping for cross-cluster detection
        article_to_cluster: Dict[str, int] = {}
        for cid, arts in cluster_articles.items():
            for a in arts:
                article_to_cluster[str(a.id)] = cid

        # Build cluster → entity set mapping
        cluster_entities: Dict[int, set] = {}
        for cid, arts in cluster_articles.items():
            entities = set()
            for a in arts:
                for name in getattr(a, 'entity_names', []):
                    normalized = name.strip().title()
                    if len(normalized) > 1:
                        entities.add(normalized)
            cluster_entities[cid] = entities

        # For each cluster, find bridge entities and compute graph density
        for cid in cluster_articles:
            this_entities = cluster_entities.get(cid, set())
            if not this_entities:
                cluster_signals[cid]["entity_graph"] = {
                    "bridge_entities": [],
                    "bridge_count": 0,
                    "intra_edges": 0,
                    "graph_density": 0.0,
                }
                continue

            # Find bridge entities: entities in this cluster that also appear in other clusters
            bridge_entities = []
            for entity in this_entities:
                entity_article_ids = tracker.entity_articles.get(entity, set())
                # Check if this entity appears in articles belonging to OTHER clusters
                other_cluster_ids = set()
                for aid in entity_article_ids:
                    other_cid = article_to_cluster.get(aid)
                    if other_cid is not None and other_cid != cid:
                        other_cluster_ids.add(other_cid)
                if other_cluster_ids:
                    bridge_entities.append({
                        "entity": entity,
                        "shared_clusters": len(other_cluster_ids),
                        "total_articles": len(entity_article_ids),
                    })

            # Sort by number of clusters shared (most connected first)
            bridge_entities.sort(key=lambda x: x["shared_clusters"], reverse=True)

            # Count intra-cluster edges (co-occurrence pairs within this cluster)
            intra_edges = 0
            for (a, b), weight in tracker.edges.items():
                if a in this_entities and b in this_entities:
                    intra_edges += weight

            # Graph density: actual edges / possible edges
            n_entities = len(this_entities)
            max_possible = n_entities * (n_entities - 1) / 2 if n_entities > 1 else 1
            graph_density = min(1.0, intra_edges / max_possible) if max_possible > 0 else 0.0

            cluster_signals[cid]["entity_graph"] = {
                "bridge_entities": bridge_entities[:10],  # Top 10 bridges
                "bridge_count": len(bridge_entities),
                "intra_edges": intra_edges,
                "graph_density": round(graph_density, 3),
            }

        elapsed = time.time() - t
        self.metrics["phase_times"]["entity_graph_injection"] = round(elapsed, 3)
        total_bridges = sum(
            s.get("entity_graph", {}).get("bridge_count", 0)
            for s in cluster_signals.values()
        )
        logger.info(
            f"Phase 7.1 (entity graph): {total_bridges} bridge entities across "
            f"{len(cluster_signals)} clusters in {elapsed:.3f}s"
        )

    def _link_related_trends(self, tree) -> None:
        """
        Phase 10: Link related trends using shared entities (Feedly Leo approach).

        For each TrendNode, find other nodes that share 2+ entities. Also link
        nodes sharing primary sectors. Caps at 5 related trends per node.

        Edge cases:
        - Empty tree → skip
        - Single node → no relations possible
        - No entity overlap → only sector-based links
        - Circular references → prevented (don't link A→A)
        """
        t = time.time()
        nodes = tree.nodes
        if len(nodes) < 2:
            logger.debug("Phase 10 (trend linking): <2 nodes, skipping")
            return

        # Build node → entity set mapping from key_entities
        node_entities: Dict[str, set] = {}
        node_sectors: Dict[str, set] = {}
        for nid, node in nodes.items():
            node_entities[nid] = set(e.lower().strip() for e in node.key_entities if e)
            node_sectors[nid] = set(
                s.lower() if isinstance(s, str) else str(s).lower()
                for s in node.primary_sectors
            )

        total_links = 0
        node_ids = list(nodes.keys())

        for i, nid in enumerate(node_ids):
            related_ids = []
            related_titles = []
            relationship_types = []

            for j, other_nid in enumerate(node_ids):
                if i == j:
                    continue

                # Check entity overlap (need 2+ shared entities for meaningful link)
                shared_entities = node_entities[nid] & node_entities[other_nid]
                if len(shared_entities) >= 2:
                    related_ids.append(other_nid)
                    related_titles.append(nodes[other_nid].trend_title)
                    relationship_types.append("shares_entities")
                    continue  # Don't double-link

                # Check sector overlap (weaker signal, secondary)
                shared_sectors = node_sectors[nid] & node_sectors[other_nid]
                if shared_sectors:
                    related_ids.append(other_nid)
                    related_titles.append(nodes[other_nid].trend_title)
                    relationship_types.append("same_sector")

            # Cap at 5 related trends per node
            max_related = 5
            nodes[nid].related_trend_ids = related_ids[:max_related]
            nodes[nid].related_trend_titles = related_titles[:max_related]
            nodes[nid].relationship_types = relationship_types[:max_related]
            total_links += len(nodes[nid].related_trend_ids)

        elapsed = time.time() - t
        self.metrics["phase_times"]["trend_linking"] = round(elapsed, 3)
        logger.info(
            f"Phase 10 (trend linking): {total_links} links across "
            f"{len(nodes)} nodes in {elapsed:.3f}s"
        )

    def _compute_cluster_quality(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_signals: Dict[int, Dict[str, Any]],
        embeddings,
        articles: List[NewsArticle],
        labels: np.ndarray,
    ) -> None:
        """Compute cluster quality metrics and inject into per-cluster signals.

        Adds intra-cluster cosine similarity: how tight are the clusters?
        This feeds directly into each cluster's confidence_score (not just
        global metrics), so the confidence score reflects actual cluster coherence.
        """
        try:
            # Convert to ndarray for fancy indexing (embeddings may be a Python list)
            emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings

            # Build article-id to embedding index map
            aid_to_idx = {id(a): i for i, a in enumerate(articles)}

            cosine_sims = []
            for cid, arts in cluster_articles.items():
                if len(arts) < 2:
                    # Single-article cluster: perfect coherence by definition
                    if cid in cluster_signals:
                        cluster_signals[cid]["intra_cluster_cosine"] = 1.0
                    continue
                # Get embeddings for this cluster's articles
                idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
                if len(idxs) < 2:
                    continue
                cluster_embs = emb_array[idxs]
                # Normalize
                norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normed = cluster_embs / norms
                # Mean pairwise cosine similarity
                sim_matrix = np.dot(normed, normed.T)
                n = len(sim_matrix)
                # Exclude diagonal (self-similarity = 1.0)
                total = (sim_matrix.sum() - n) / max(n * (n - 1), 1)
                cosine_sims.append(total)

                # Inject into per-cluster signals so confidence_score can use it
                if cid in cluster_signals:
                    cluster_signals[cid]["intra_cluster_cosine"] = round(total, 4)

            if cosine_sims:
                avg_sim = sum(cosine_sims) / len(cosine_sims)
                self.metrics["avg_intra_cluster_cosine"] = round(avg_sim, 4)
                self.metrics["min_intra_cluster_cosine"] = round(min(cosine_sims), 4)
                logger.info(
                    f"Cluster quality: avg intra-cosine={avg_sim:.3f}, "
                    f"min={min(cosine_sims):.3f}, "
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
        rejects noise masquerading as clusters. All in original 384-dim space.
        """
        t = time.time()
        try:
            refined_articles, refined_labels, noise_delta = validate_and_refine_clusters(
                cluster_articles=cluster_articles,
                embeddings=embeddings,
                articles=articles,
                labels=labels,
                min_coherence=self.coherence_min,
                reject_threshold=self.coherence_reject,
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

    def _phase_trend_memory(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_signals: Dict[int, Dict[str, Any]],
        cluster_keywords: Dict[int, List[str]],
        embeddings: np.ndarray,
        articles: List[NewsArticle],
    ) -> None:
        """Phase 7.7: Match clusters against historical trend memory.

        Injects continuity_score and novelty_score into cluster_signals.
        Continuation of known trends → higher confidence.
        Novel trends → flagged as genuinely new.
        """
        t = time.time()
        try:
            aid_to_idx = {id(a): i for i, a in enumerate(articles)}
            emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings

            # Compute cluster centroids in original embedding space
            cluster_centroids: Dict[int, np.ndarray] = {}
            for cid, arts in cluster_articles.items():
                idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
                if idxs:
                    cluster_centroids[cid] = emb_array[idxs].mean(axis=0)

            # Batch match against historical memory
            cluster_titles = {}
            for cid in cluster_articles:
                kw = cluster_keywords.get(cid, [])
                cluster_titles[cid] = ", ".join(kw[:3]) if kw else f"Cluster {cid}"

            results = self.trend_memory.match_clusters_batch(cluster_centroids, cluster_titles)

            # Inject into signals
            continuations = 0
            for cid, match in results.items():
                if cid in cluster_signals:
                    cluster_signals[cid]["continuity_score"] = match["continuity_score"]
                    cluster_signals[cid]["novelty_score"] = match["novelty_score"]
                    cluster_signals[cid]["is_trend_continuation"] = match["is_continuation"]
                    if match["is_continuation"]:
                        cluster_signals[cid]["matched_past_trend"] = match["matched_trend_title"]
                        cluster_signals[cid]["trend_age_days"] = match["matched_trend_age_days"]
                        cluster_signals[cid]["trend_seen_count"] = match["matched_trend_seen_count"]
                        continuations += 1

            elapsed = time.time() - t
            self.metrics["phase_times"]["trend_memory"] = round(elapsed, 2)
            self.metrics["trend_continuations"] = continuations
            self.metrics["trend_novel"] = len(results) - continuations
            logger.info(
                f"Phase 7.7 (trend memory): {continuations}/{len(results)} continue past trends, "
                f"{len(results) - continuations} novel in {elapsed:.2f}s"
            )
        except Exception as e:
            elapsed = time.time() - t
            self.metrics["phase_times"]["trend_memory"] = round(elapsed, 2)
            logger.warning(f"Phase 7.7 (trend memory) failed: {e}")

    def _store_trends_to_memory(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_summaries: Dict[int, Dict[str, Any]],
        cluster_keywords: Dict[int, List[str]],
        cluster_signals: Dict[int, Dict[str, Any]],
        embeddings: np.ndarray,
        articles: List[NewsArticle],
    ) -> None:
        """Post-pipeline: persist current trends for future run matching."""
        try:
            aid_to_idx = {id(a): i for i, a in enumerate(articles)}
            emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings

            centroids = {}
            titles = {}
            scores = {}
            counts = {}

            for cid, arts in cluster_articles.items():
                idxs = [aid_to_idx[id(a)] for a in arts if id(a) in aid_to_idx]
                if idxs:
                    centroids[cid] = emb_array[idxs].mean(axis=0)
                summary = cluster_summaries.get(cid, {})
                kw = cluster_keywords.get(cid, [])
                fallback = " / ".join(w.capitalize() for w in kw[:3]) if kw else "Unknown Trend"
                titles[cid] = summary.get("trend_title", fallback)
                scores[cid] = cluster_signals.get(cid, {}).get("trend_score", 0.0)
                counts[cid] = len(arts)

            self.trend_memory.store_trends(
                cluster_centroids=centroids,
                cluster_titles=titles,
                cluster_keywords=cluster_keywords,
                cluster_scores=scores,
                cluster_article_counts=counts,
            )
        except Exception as e:
            logger.debug(f"Trend memory store failed: {e}")

    def _phase_search_interest(
        self, cluster_signals: Dict[int, Dict[str, Any]],
        cluster_keywords: Dict[int, List[str]],
    ) -> None:
        """Phase 7.5: Validate trends against Google Trends search interest."""
        t = time.time()
        try:
            from app.trends.signals.search_interest import compute_search_signals_batch
            from app.trends.signals.composite import compute_actionability_score
            search_results = compute_search_signals_batch(cluster_keywords)
            matched = 0
            for cid, result in search_results.items():
                if cid in cluster_signals:
                    cluster_signals[cid]["search_interest_score"] = result["search_interest_score"]
                    cluster_signals[cid]["matching_trends"] = result.get("matching_trends", [])
                    if result["search_interest_score"] > 0:
                        matched += 1
                        cluster_signals[cid]["actionability_score"] = compute_actionability_score(cluster_signals[cid])
            elapsed = time.time() - t
            self.metrics["phase_times"]["search_interest"] = round(elapsed, 2)
            logger.info(f"Phase 7.5 (search interest): {matched}/{len(cluster_signals)} match Google Trends in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"Phase 7.5 failed: {e}")

    async def _phase_synthesize(
        self, cluster_articles: Dict[int, List[NewsArticle]],
        cluster_keywords: Dict[int, List[str]],
    ) -> Dict[int, Dict[str, Any]]:
        """Phase 8: LLM synthesis (delegated to synthesis module)."""
        t = time.time()
        result = await synthesize_clusters(
            cluster_articles, cluster_keywords, self.llm_tool, self.max_concurrent_llm,
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
            min_confidence = get_settings().min_synthesis_confidence
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

            # Gate 3: Low confidence from signals
            signals = cluster_signals.get(cid, {})
            confidence = signals.get("confidence", 1.0)

            # Also check V10 validation score if present
            val_meta = summary.get("_validation", {})
            val_score = val_meta.get("score", 1.0)
            effective_confidence = min(confidence, val_score) if val_meta else confidence

            if effective_confidence < min_confidence:
                dropped_low_confidence += 1
                logger.debug(
                    f"V9: Dropped cluster {cid} — confidence {effective_confidence:.2f} "
                    f"< threshold {min_confidence}"
                )
                continue

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

    def _phase_tree(
        self,
        cluster_articles: Dict[int, List[NewsArticle]],
        cluster_keywords: Dict[int, List[str]],
        cluster_signals: Dict[int, Dict[str, Any]],
        cluster_summaries: Dict[int, Dict[str, Any]],
        noise_count: int,
        total_articles: int,
    ) -> TrendTree:
        """Phase 9: Assemble TrendTree (delegated to tree_builder module)."""
        t = time.time()
        tree = build_trend_tree(
            cluster_articles, cluster_keywords, cluster_signals,
            cluster_summaries, noise_count, total_articles,
        )
        elapsed = time.time() - t
        self.metrics["phase_times"]["tree"] = round(elapsed, 2)
        logger.info(f"Phase 9 (tree): {len(tree.root_ids)} trends, {noise_count} noise in {elapsed:.2f}s")
        return tree


# ── Convenience function ─────────────────────────────────────────────────


async def detect_trend_tree(
    articles: List[NewsArticle],
    min_cluster_size: int = 5,
    max_depth: int = 3,
    mock_mode: bool = False,
    country: str = "",
    domestic_source_ids: Optional[set] = None,
) -> TrendTree:
    """Quick function to run the full trend detection pipeline."""
    engine = RecursiveTrendEngine(
        min_cluster_size=min_cluster_size,
        max_depth=max_depth,
        mock_mode=mock_mode,
        country=country,
        domestic_source_ids=domestic_source_ids,
    )
    return await engine.run(articles)
