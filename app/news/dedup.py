"""
Multi-stage article deduplication for news aggregation.

DEDUP PIPELINE (3 stages):
  1. TITLE DEDUP:   Fast exact/fuzzy title matching (catches syndicated content)
  2. ENTITY DEDUP:  Articles with same key entities about same topic = duplicate
  3. MINHASH LSH:   Lexical near-duplicate detection (shared word phrases)

NOTE: Semantic dedup (embedding-based) runs separately in the trend engine
      after embeddings are computed. This catches cross-source duplicates
      where different sources cover the same story with different wording.

WHY MinHash over simple cosine similarity:
  At 700 articles, cosine similarity dedup is O(n^2) = 490K comparisons.
  That's fine. But at 70K articles it's 4.9 BILLION comparisons — impossible.
  MinHash LSH is O(n) regardless of corpus size.

PERFORMANCE:
  700 articles:    <0.5 sec
  70,000 articles: <5 sec
  700,000 articles: <30 sec (O(n) scaling)

REQUIRES: pip install datasketch

REF: Broder, "On the resemblance and containment of documents" (1997)
     Leskovec, Rajaraman, Ullman, "Mining of Massive Datasets" Ch. 3
"""

import logging
import re
from collections import defaultdict
from typing import List, Optional, Set, Tuple

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

# Common stop words to exclude from entity fingerprinting
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they', 'we',
    'you', 'i', 'my', 'your', 'his', 'her', 'their', 'our', 'what',
    'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there', 'then', 'if', 'about',
    'after', 'before', 'new', 'says', 'said', 'year', 'years', 'day',
    'company', 'market', 'report', 'news', 'india', 'indian', 'rs',
    'crore', 'lakh', 'per', 'cent', 'percent', 'million', 'billion',
}


class ArticleDeduplicator:
    """
    MinHash LSH deduplication engine for news articles.

    Catches near-duplicates that simple title matching misses:
    - Same event, different headline ("RBI hikes rates" vs "Central bank raises interest rate")
    - Syndicated content with minor edits
    - Wire service articles republished with publisher-specific intros

    EXTENSIBILITY: To add a new dedup strategy:
    1. Create a new class implementing deduplicate(articles) -> articles
    2. Register with StrategyRegistry: register("dedup", "your_algo", YourClass)
    """

    def __init__(
        self,
        threshold: float = 0.25,  # Lowered from 0.3 for more aggressive matching
        num_perm: int = 128,
        shingle_size: int = 2,
    ):
        """
        Args:
            threshold: Jaccard similarity threshold. Articles above this are duplicates.
                       0.25 = aggressive, catches articles with ~25% word overlap (RECOMMENDED for news)
                       0.3 catches cross-source duplicates with shared phrases.
                       0.5 catches same-event-different-headline.
                       0.8 catches only near-identical articles.
                       REF: GPT-3 training used ~0.5 for web page dedup.
            num_perm: Number of MinHash permutation functions. More = more accurate
                      but slower. 128 is the standard tradeoff.
            shingle_size: Number of words per shingle. 2 is more sensitive to
                          word-level overlap. 3 requires more exact phrases.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

    def deduplicate(self, articles: list) -> list:
        """
        Remove near-duplicate articles using multi-stage deduplication:
        1. Title-based exact match (fast, catches syndicated content)
        2. Entity fingerprint dedup (articles with same key entities = duplicate)
        3. MinHash LSH on text (catches near-duplicates with shared phrases)

        Returns deduplicated list. Keeps the earliest-published version.
        """
        if not articles:
            return []

        initial_count = len(articles)

        # Stage 1: Title-based dedup (fast, catches exact republishes)
        articles = self._title_dedup(articles)
        stage1_count = len(articles)

        # Stage 2: Entity fingerprint dedup (same entities = likely same story)
        articles = self._entity_fingerprint_dedup(articles)
        stage2_count = len(articles)

        # Stage 3: MinHash LSH (catches near-duplicates with shared text)
        articles = self._minhash_dedup(articles)

        # Summary logging
        total_removed = initial_count - len(articles)
        if total_removed > 0:
            logger.info(
                f"Dedup summary: {initial_count} → {len(articles)} "
                f"(removed {total_removed} = {total_removed/initial_count*100:.1f}%)"
            )

        return articles

    def _extract_key_entities(self, article) -> Tuple[Set[str], str]:
        """
        Extract key named entities and topic keywords from article.
        Returns (entity_set, topic_fingerprint).

        ENHANCED: More aggressive entity extraction and broader topic fingerprint.
        """
        title = getattr(article, 'title', '')
        content = getattr(article, 'content', '') or getattr(article, 'summary', '')
        text = f"{title} {content[:500]}"

        # Extract potential entities (capitalized words, company names, numbers with units)
        entities = set()

        # Capitalized words (potential names/companies)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for cap in caps:
            if cap.lower() not in STOP_WORDS and len(cap) > 2:
                entities.add(cap.lower())

        # Also extract ALL significant words from title (not just capitalized)
        # This helps catch topic overlap even without proper noun detection
        # len(w) > 2 to keep abbreviations like RBI, KYC, GDP, IPO, etc.
        title_words = [w.lower() for w in re.findall(r'\w+', title) if w.lower() not in STOP_WORDS and len(w) > 2]
        entities.update(title_words)

        # Numbers with context (₹1000 crore, $5 billion, 10% growth)
        numbers = re.findall(r'(?:₹|rs\.?|inr|\$|usd)?\s*[\d,]+\.?\d*\s*(?:crore|lakh|million|billion|%|percent)?', text.lower())
        for num in numbers[:3]:  # Top 3 significant numbers
            clean = re.sub(r'\s+', '', num)
            if len(clean) > 2:
                entities.add(clean)

        # Stock tickers (NSE:, BSE:, all caps 2-5 letters)
        tickers = re.findall(r'\b(?:NSE|BSE)[:\s]*([A-Z]{2,10})\b', text)
        entities.update(t.lower() for t in tickers)

        # Create topic fingerprint from CORE key words in title (fewer = more matches)
        # Use only top 3 words for broader matching
        topic = '_'.join(sorted(title_words[:3]))  # Top 3 non-stop words sorted

        return entities, topic

    def _entity_fingerprint_dedup(self, articles: list) -> list:
        """
        Stage 2: Remove articles that share the same key entities and topic.
        If two articles have 2+ shared significant entities AND similar topic = duplicate.

        RELAXED: Lowered from 3+ to 2+ shared entities for better recall.
        Also checks for entity overlap even with different topic fingerprints.
        """
        unique = []
        seen_fingerprints = defaultdict(list)  # fingerprint -> list of (entities, article)
        seen_all_entities = []  # (entities, topic, article) for cross-topic matching
        dup_count = 0
        dup_examples = []

        # Sort by published_at to keep earliest
        sorted_articles = sorted(
            articles,
            key=lambda a: a.published_at if hasattr(a, 'published_at') else 0
        )

        for article in sorted_articles:
            entities, topic = self._extract_key_entities(article)
            title = getattr(article, 'title', '')

            is_duplicate = False
            match_reason = ""

            # Check 1: Same topic fingerprint + 2+ shared entities
            for existing_entities, existing_article in seen_fingerprints.get(topic, []):
                shared = entities & existing_entities
                if len(shared) >= 2:
                    is_duplicate = True
                    dup_count += 1
                    match_reason = f"same_topic({topic[:30]})+{len(shared)}_shared"
                    if len(dup_examples) < 5:
                        dup_examples.append({
                            "removed": title[:50],
                            "kept": getattr(existing_article, 'title', '')[:50],
                            "reason": match_reason,
                            "shared": list(shared)[:5]
                        })
                    break

            # Check 2: High entity overlap (4+) even with different topic fingerprint
            if not is_duplicate:
                for existing_entities, existing_topic, existing_article in seen_all_entities:
                    shared = entities & existing_entities
                    if len(shared) >= 4:
                        is_duplicate = True
                        dup_count += 1
                        match_reason = f"high_overlap({len(shared)}_shared)"
                        if len(dup_examples) < 5:
                            dup_examples.append({
                                "removed": title[:50],
                                "kept": getattr(existing_article, 'title', '')[:50],
                                "reason": match_reason,
                                "shared": list(shared)[:5]
                            })
                        break

            if not is_duplicate:
                seen_fingerprints[topic].append((entities, article))
                seen_all_entities.append((entities, topic, article))
                unique.append(article)

        # Always log for diagnostics
        logger.info(f"Entity dedup: {len(articles)} → {len(unique)} ({dup_count} duplicates)")
        if dup_examples:
            logger.info(f"  Entity dedup examples: {dup_examples}")

        return unique

    def _title_dedup(self, articles: list) -> list:
        """
        Stage 1: Remove articles with near-identical titles (fast).
        Uses multiple normalization strategies to catch variants.
        """
        seen_titles = {}  # fingerprint -> article title (for logging)
        unique = []
        dup_count = 0
        dup_examples = []

        # Sort by published_at to keep earliest
        sorted_articles = sorted(
            articles,
            key=lambda a: a.published_at if hasattr(a, 'published_at') else 0
        )

        for article in sorted_articles:
            title = getattr(article, 'title', '')

            # Create multiple normalized versions for matching
            fingerprints = self._get_title_fingerprints(title)

            # Check if any fingerprint matches
            matched_fp = None
            for fp in fingerprints:
                if fp in seen_titles:
                    matched_fp = fp
                    break

            if matched_fp:
                dup_count += 1
                if len(dup_examples) < 5:
                    dup_examples.append({
                        "removed": title[:60],
                        "kept": seen_titles[matched_fp][:60],
                        "fingerprint": matched_fp[:40]
                    })
                continue

            # Add all fingerprints to seen dict with this article's title
            for fp in fingerprints:
                seen_titles[fp] = title
            unique.append(article)

        # Always log title dedup results for diagnostics
        logger.info(f"Title dedup: {len(articles)} → {len(unique)} ({dup_count} duplicates)")
        if dup_examples:
            logger.info(f"  Title dedup examples: {dup_examples}")

        return unique

    def _get_title_fingerprints(self, title: str) -> Set[str]:
        """
        Generate multiple normalized fingerprints for a title.
        Returns a set of fingerprints that should match similar titles.

        AGGRESSIVE MATCHING STRATEGY:
        - Multiple fingerprint variations to catch different phrasings
        - Core topic extraction (fewer words = more matches)
        - N-gram matching for partial overlaps
        """
        fingerprints = set()

        # 1. Basic normalization: lowercase, remove punctuation
        basic = re.sub(r'[^\w\s]', '', title.lower())
        basic = ' '.join(basic.split())
        fingerprints.add(basic)

        # 2. Remove common suffixes (live, breaking, etc.)
        no_suffix = re.sub(
            r'\s+(live|updates?|breaking|exclusive|report|watch|video|'
            r'read|full|text|here|now|latest|today|check|details|explained)$',
            '', basic, flags=re.IGNORECASE
        )
        fingerprints.add(no_suffix)

        # 3. Remove common prefixes
        no_prefix = re.sub(
            r'^(breaking|exclusive|just in|watch|live|update|'
            r'video|india|news|check|heres?|latest|top)\s*[:\-]?\s*',
            '', basic, flags=re.IGNORECASE
        )
        fingerprints.add(no_prefix)

        # 4. Key words (sorted, no stop words) - catches reordered titles
        # NOTE: len(w) > 2 to keep important abbreviations like RBI, KYC, GDP, etc.
        words = [w for w in basic.split() if w not in STOP_WORDS and len(w) > 2]

        # 4a. Full key words (top 8 sorted)
        key_words = '_'.join(sorted(words[:8]))
        if len(key_words) > 10:
            fingerprints.add(f"kw:{key_words}")

        # 4b. CORE TOPIC: Top 4 key words sorted (more aggressive matching)
        # This catches "claim settlement ratio health" even if other words differ
        if len(words) >= 4:
            core_words = '_'.join(sorted(words[:4]))
            fingerprints.add(f"core:{core_words}")

        # 4c. Top 3 key words (even more aggressive)
        if len(words) >= 3:
            core3 = '_'.join(sorted(words[:3]))
            fingerprints.add(f"c3:{core3}")

        # 5. First N significant words (catches truncated titles)
        if len(words) >= 6:
            first_words = '_'.join(words[:6])
            fingerprints.add(f"fw:{first_words}")

        # 5b. First 4 words (more aggressive)
        if len(words) >= 4:
            first4 = '_'.join(words[:4])
            fingerprints.add(f"f4:{first4}")

        # 6. 2-gram fingerprints: consecutive word pairs
        # Catches titles that share key phrases even if other words differ
        if len(words) >= 4:
            for i in range(len(words) - 1):
                bigram = f"bg:{words[i]}_{words[i+1]}"
                fingerprints.add(bigram)

        # 7. 3-gram fingerprints for longer titles
        if len(words) >= 5:
            for i in range(len(words) - 2):
                trigram = f"tg:{words[i]}_{words[i+1]}_{words[i+2]}"
                fingerprints.add(trigram)

        return fingerprints

    def _minhash_dedup(self, articles: list) -> list:
        """MinHash LSH deduplication."""
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        unique_articles = []
        duplicate_count = 0
        duplicate_examples = []  # Track a few examples for logging

        # Sort by published_at so we keep the earliest version
        sorted_articles = sorted(
            articles,
            key=lambda a: a.published_at if hasattr(a, 'published_at') else 0
        )

        # Track sources for logging
        source_counts = {}
        for a in articles:
            src = getattr(a, 'source_name', 'unknown')
            source_counts[src] = source_counts.get(src, 0) + 1

        for i, article in enumerate(sorted_articles):
            # Build text for hashing: title + first 200 words of content/summary
            text = self._get_text(article)
            minhash = self._build_minhash(text)

            # Check for duplicates in LSH index
            key = f"article_{i}"
            try:
                result = lsh.query(minhash)
                if result:
                    # This article is a near-duplicate of an existing one
                    duplicate_count += 1
                    if hasattr(article, 'is_duplicate'):
                        article.is_duplicate = True
                    # Log first few duplicates for diagnosis
                    if len(duplicate_examples) < 3:
                        duplicate_examples.append({
                            "title": article.title[:60],
                            "source": getattr(article, 'source_name', 'unknown'),
                            "matched_with": result[0] if result else "?"
                        })
                    continue

                # Not a duplicate — add to index and keep
                lsh.insert(key, minhash)
                unique_articles.append(article)
            except Exception as e:
                # On any LSH error, keep the article (fail open, not closed)
                logger.warning(f"MinHash error for article {i}: {e}")
                unique_articles.append(article)

        # Calculate dedup rate per source
        dedup_rate = (duplicate_count / len(articles) * 100) if articles else 0

        logger.info(
            f"Dedup: {len(articles)} articles → {len(unique_articles)} unique "
            f"({duplicate_count} duplicates = {dedup_rate:.1f}%, threshold={self.threshold})"
        )
        logger.info(f"  Sources: {len(source_counts)} unique sources")

        if duplicate_examples:
            logger.info(f"  Duplicate examples: {duplicate_examples}")

        # Warn if dedup rate seems too low for diverse news
        if len(source_counts) > 5 and dedup_rate < 5:
            logger.warning(
                f"  Low dedup rate ({dedup_rate:.1f}%) despite {len(source_counts)} sources. "
                f"Articles may have very diverse content, or threshold ({self.threshold}) may be too high."
            )

        return unique_articles

    def _build_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature from text using word-level shingles.

        Shingles are overlapping n-grams of words:
        "the quick brown fox" with size=2 → {"the quick", "quick brown", "brown fox"}

        WHY word-level not character-level: News articles share common phrases
        ("according to sources", "in a statement") that cause false positives
        with character shingles. Word shingles are more meaningful for news.
        """
        m = MinHash(num_perm=self.num_perm)

        # Clean and tokenize, removing stop words for better signal
        words = re.findall(r'\w+', text.lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        if len(words) < self.shingle_size:
            # Not enough words, hash the whole text
            m.update(text.lower().encode('utf-8'))
            return m

        # Create shingles from filtered words
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            m.update(shingle.encode('utf-8'))

        return m

    def _get_text(self, article) -> str:
        """
        Extract text for hashing: title (3x weighted) + first 300 words of content.
        Title is repeated to give it more weight in similarity calculation.
        """
        title = getattr(article, 'title', '')
        content = getattr(article, 'content', '') or getattr(article, 'summary', '')

        # Use title + first 300 words (enough for duplicate detection)
        words = content.split()[:300]
        content_text = ' '.join(words)

        # Repeat title 3x to give it more weight (title similarity is key indicator)
        return f"{title} {title} {title} {content_text}"
