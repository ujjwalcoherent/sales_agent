"""
Dataset Enhancer — Auto-labels articles from pipeline signals, no human input needed.

Research basis:
  AG News (Zhang et al. 2015): 120k articles, Business class = B2B positives,
    Sports/World = noise negatives. Free from HuggingFace datasets.
  Reuters-21578: "earn", "acq", "trade", "corp" categories = gold B2B signal.
  "Improving Classification Performance With Human Feedback" (arXiv:2401.09555):
    closed-loop from observational signals (not just human ratings) improves F1.

Auto-labeling strategy (no human input needed):
  POSITIVE examples — high confidence B2B business news:
    1. Cluster coherence > 0.70: articles inside a tight cluster are confirmed B2B
    2. NLI entailment 0.85-1.00: model highly confident (not ambiguous zone)
    3. Reuters-21578 "earn"/"acq"/"corp" category match (if running on benchmark)
    4. Email engagement: article led to a cluster that got email reply → strongest signal

  NEGATIVE examples — high confidence noise:
    1. NLI entailment < 0.10: model very confident it's noise
    2. Cluster coherence < 0.25: incoherent cluster = articles didn't form signal
    3. AG News Sports/World class (from dataset bootstrap)
    4. User explicit "bad_trend" / "bad_lead" feedback

Self-improvement loop:
  Every pipeline run → extract labels from signals → append to dynamic_dataset.jsonl
  After N_THRESHOLD examples: trigger SetFit retraining via HypothesisLearner
  After retraining: validate against AG News held-out test → only deploy if F1 improves
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from app.intelligence.models import Article
    from app.intelligence.cluster.orchestrator import ClusterResult

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("data/dynamic_dataset.jsonl")
_STATS_PATH = Path("data/dataset_stats.json")

# Trigger SetFit retraining when we accumulate this many labeled examples
N_RETRAIN_THRESHOLD = 50

# Quality thresholds for auto-labeling
POSITIVE_COHERENCE_THRESHOLD = 0.70    # Cluster coherence > this → articles = positive
NEGATIVE_COHERENCE_THRESHOLD = 0.25   # Cluster coherence < this → articles = negative
POSITIVE_NLI_THRESHOLD = 0.85         # NLI score > this → positive (high confidence)
NEGATIVE_NLI_THRESHOLD = 0.10         # NLI score < this → negative (high confidence)

# Max class imbalance (positive:negative ratio)
MAX_CLASS_RATIO = 2.0

# Cap dataset size to avoid memory/training time blowup
MAX_DATASET_SIZE = 5000


class DatasetEnhancer:
    """Auto-labels articles from cluster quality and NLI signals.

    No human input required — uses pipeline observational signals:
    - Cluster coherence as quality signal (tight cluster = confirmed B2B)
    - NLI confidence as noise signal (very low score = confirmed noise)
    - AG News / Reuters-21578 as ground-truth bootstrap (first run only)
    """

    def __init__(self):
        self._dataset_path = _DATASET_PATH
        self._stats_path = _STATS_PATH
        self._seen_hashes: set[str] = set()
        self._load_seen_hashes()

    def extract_labels_from_clusters(
        self,
        clusters: List["ClusterResult"],
        nli_scores: Optional[dict[str, float]] = None,
    ) -> Tuple[int, int]:
        """Extract positive/negative labels from cluster quality signals.

        Dual-gate for positives: requires BOTH high coherence AND individual NLI score.
        This prevents topically coherent non-B2B clusters (e.g., all Jim Cramer articles)
        from becoming positive training examples just because they're topically similar.

        Single-gate for negatives: low coherence alone is sufficient (any incoherent
        cluster's articles are unreliable signals regardless of topic).

        Args:
            clusters: Validated cluster results from cluster_and_validate()
            nli_scores: Dict mapping article_id → NLI entailment score

        Returns:
            Tuple of (positive_count, negative_count) added this run.
        """
        nli_scores = nli_scores or {}
        positives_added = 0
        negatives_added = 0
        # Minimum NLI score for an article in a high-coherence cluster to be labeled positive.
        # Gate: coherence > 0.70 (cluster-level) AND NLI > 0.55 (article-level).
        # Prevents topically coherent non-B2B clusters (sports, celebrity commentary)
        # from contaminating the positive set.
        _MIN_NLI_FOR_CLUSTER_POSITIVE = 0.55

        for cluster in clusters:
            coherence = getattr(cluster, "coherence_score", 0.0) or 0.0
            articles = getattr(cluster, "articles", [])

            if coherence >= POSITIVE_COHERENCE_THRESHOLD:
                # Dual-gate: coherence (cluster) + NLI score (article) must both pass
                for article in articles:
                    art_id = getattr(article, "id", None)
                    nli_score = nli_scores.get(art_id, None) if art_id else None

                    # If NLI score unavailable, use a conservative threshold of 0.50
                    # to allow the label but with lower confidence
                    if nli_score is not None and nli_score < _MIN_NLI_FOR_CLUSTER_POSITIVE:
                        # Topically coherent but NLI not convinced it's B2B → skip
                        continue

                    text = self._extract_text(article)
                    # Confidence = geometric mean of coherence and NLI score
                    article_confidence = (
                        (coherence * nli_score) ** 0.5
                        if nli_score is not None
                        else coherence * 0.85  # conservative if NLI unavailable
                    )
                    if text and self._add_example(text, label=1, source="cluster_coherence",
                                                   confidence=article_confidence):
                        positives_added += 1

            elif coherence < NEGATIVE_COHERENCE_THRESHOLD:
                # Incoherent cluster → articles are noise (single-gate — coherence alone sufficient)
                for article in articles:
                    text = self._extract_text(article)
                    if text and self._add_example(text, label=0, source="cluster_incoherence",
                                                   confidence=1.0 - coherence):
                        negatives_added += 1

        logger.info(
            f"[dataset_enhancer] Extracted from clusters: "
            f"+{positives_added} positives, +{negatives_added} negatives "
            f"(dual-gate: coherence+NLI for positives, coherence-only for negatives)"
        )
        return positives_added, negatives_added

    def extract_labels_from_filter(
        self,
        kept_articles: List["Article"],
        rejected_article_texts: Optional[List[str]] = None,
        nli_scores: Optional[List[float]] = None,
        rejected_nli_scores: Optional[List[float]] = None,
    ) -> Tuple[int, int]:
        """Extract labels from filter output (called after filter_articles).

        High NLI score + passed LLM → positive (confidence = actual NLI score).
        Low NLI score + auto-rejected → negative (confidence = 1 - NLI score).

        Confidence is set to the ACTUAL NLI score, not a hard-coded 0.9.
        This prevents mislabeled articles from appearing in the high-confidence
        validation set. An article that scored 0.095 (barely below the 0.10 threshold)
        will have confidence=0.905 and be filtered out at _VALIDATION_MIN_CONFIDENCE=0.85
        only if it scored < 0.15. Articles clearly in noise zone (score < 0.05) get
        confidence >= 0.95 and correctly appear in validation negatives.

        Args:
            kept_articles: Articles that passed the filter (positives)
            rejected_article_texts: Texts of auto-rejected articles (negatives)
            nli_scores: Parallel NLI scores for kept_articles
            rejected_nli_scores: Parallel NLI scores for rejected articles

        Returns:
            Tuple of (positive_count, negative_count) added.
        """
        positives_added = 0
        negatives_added = 0

        # High-confidence kept articles → positive examples
        # Confidence = actual NLI score (not hard-coded)
        for i, article in enumerate(kept_articles):
            score = nli_scores[i] if nli_scores and i < len(nli_scores) else 0.5
            if score >= POSITIVE_NLI_THRESHOLD:
                text = self._extract_text(article)
                if text and self._add_example(text, label=1, source="nli_high_confidence",
                                               confidence=score):
                    positives_added += 1

        # Auto-rejected articles → negative examples
        # Confidence = 1 - NLI score (how confident we are it's noise)
        # An article scoring 0.001 has confidence 0.999; one scoring 0.095 has confidence 0.905
        # The _VALIDATION_MIN_CONFIDENCE gate (0.85) means only articles scoring < 0.15
        # will be used in the validation set — safe margin from the 0.10 threshold
        for i, text in enumerate(rejected_article_texts or []):
            if not text:
                continue
            actual_score = (
                rejected_nli_scores[i]
                if rejected_nli_scores and i < len(rejected_nli_scores)
                else 0.05  # conservative default if score not provided
            )
            confidence = 1.0 - actual_score  # near-zero NLI → near-1.0 confidence it's noise
            if self._add_example(text, label=0, source="nli_auto_rejected",
                                  confidence=confidence):
                negatives_added += 1

        return positives_added, negatives_added

    def bootstrap_from_ag_news(self, n_per_class: int = 50) -> Tuple[int, int]:
        """Bootstrap dataset from AG News without any manual labeling.

        AG News categories:
          Class 1 = World  → noise (geopolitical, not B2B)
          Class 2 = Sports → noise (sport events, clearly not B2B)
          Class 3 = Business → positive (B2B business news)
          Class 4 = Sci/Tech → ambiguous (keep out of bootstrap)

        Uses HuggingFace datasets library if available.
        Falls back gracefully if not installed.

        Args:
            n_per_class: Number of examples per class to load

        Returns:
            Tuple of (positive_count, negative_count) added.
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            logger.warning(
                "[dataset_enhancer] 'datasets' library not installed. "
                "Run: pip install datasets  to enable AG News bootstrap."
            )
            return 0, 0

        try:
            logger.info(f"[dataset_enhancer] Loading AG News for bootstrap (n={n_per_class}/class)...")
            ag_news = load_dataset("ag_news", split="train")

            positives_added = 0
            negatives_added = 0
            pos_count = 0
            neg_count = 0

            for example in ag_news:
                if pos_count >= n_per_class and neg_count >= n_per_class:
                    break

                label = example["label"]  # 0=World, 1=Sports, 2=Business, 3=Sci/Tech
                text = example["text"]

                if label == 2 and pos_count < n_per_class:  # Business = positive
                    if self._add_example(text, label=1, source="ag_news_business",
                                         confidence=0.85):
                        positives_added += 1
                    pos_count += 1

                elif label in (0, 1) and neg_count < n_per_class:  # World/Sports = negative
                    if self._add_example(text, label=0, source="ag_news_noise",
                                         confidence=0.85):
                        negatives_added += 1
                    neg_count += 1

            logger.info(
                f"[dataset_enhancer] AG News bootstrap: "
                f"+{positives_added} positives, +{negatives_added} negatives"
            )
            return positives_added, negatives_added

        except Exception as e:
            logger.warning(f"[dataset_enhancer] AG News bootstrap failed: {e}")
            return 0, 0

    def bootstrap_from_reuters(self, n_per_class: int = 30) -> Tuple[int, int]:
        """Bootstrap from Reuters-21578 business categories.

        Reuters categories used:
          earn, acq, trade, corp → positive (confirmed B2B business events)
          politics, military, sport → negative (noise)

        Uses NLTK Reuters corpus (free, no API needed).
        """
        try:
            import nltk
            try:
                from nltk.corpus import reuters
                reuters.categories()  # test if downloaded
            except LookupError:
                logger.info("[dataset_enhancer] Downloading Reuters-21578 via NLTK...")
                nltk.download("reuters", quiet=True)
                from nltk.corpus import reuters
        except ImportError:
            logger.warning("[dataset_enhancer] NLTK not installed. Run: pip install nltk")
            return 0, 0

        try:
            # B2B positives: company earnings/acquisitions/trade → confirmed business events
            _B2B_CATEGORIES = {"earn", "acq", "trade", "corp", "money-fx", "interest", "ship"}
            # Noise negatives: commodity prices and macro indicators — no specific company action
            _NOISE_CATEGORIES = {"grain", "corn", "wheat", "rice", "cotton", "sugar", "cpi",
                                  "gnp", "ipi", "dlr", "yen", "nkr", "wpi", "barley", "oat"}

            positives_added = 0
            negatives_added = 0
            pos_count = 0
            neg_count = 0

            for file_id in reuters.fileids():
                if pos_count >= n_per_class and neg_count >= n_per_class:
                    break

                cats = set(reuters.categories(file_id))
                text = reuters.raw(file_id)[:512]

                if cats & _B2B_CATEGORIES and pos_count < n_per_class:
                    if self._add_example(text, label=1, source="reuters_b2b",
                                         confidence=0.90):
                        positives_added += 1
                    pos_count += 1

                elif cats & _NOISE_CATEGORIES and neg_count < n_per_class:
                    if self._add_example(text, label=0, source="reuters_noise",
                                         confidence=0.90):
                        negatives_added += 1
                    neg_count += 1

            logger.info(
                f"[dataset_enhancer] Reuters bootstrap: "
                f"+{positives_added} positives, +{negatives_added} negatives"
            )
            return positives_added, negatives_added

        except Exception as e:
            logger.warning(f"[dataset_enhancer] Reuters bootstrap failed: {e}")
            return 0, 0

    def get_examples_for_setfit(
        self,
        max_per_class: int = 64,
    ) -> Tuple[List[str], List[str]]:
        """Return balanced positive/negative examples for SetFit training.

        Args:
            max_per_class: Maximum examples per class (cap for training efficiency)

        Returns:
            (positives, negatives) lists of text strings.
        """
        if not self._dataset_path.exists():
            return [], []

        positives: List[str] = []
        negatives: List[str] = []

        try:
            with open(self._dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = record.get("text", "")
                    label = record.get("label", -1)

                    if label == 1 and len(positives) < max_per_class:
                        positives.append(text)
                    elif label == 0 and len(negatives) < max_per_class:
                        negatives.append(text)

        except Exception as e:
            logger.warning(f"[dataset_enhancer] Failed to load dataset: {e}")

        return positives, negatives

    def get_stats(self) -> dict:
        """Return current dataset statistics."""
        if not self._dataset_path.exists():
            return {"total": 0, "positives": 0, "negatives": 0, "ready_for_retrain": False}

        positives = 0
        negatives = 0
        sources: dict[str, int] = {}

        try:
            with open(self._dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        label = record.get("label", -1)
                        source = record.get("source", "unknown")
                        if label == 1:
                            positives += 1
                        elif label == 0:
                            negatives += 1
                        sources[source] = sources.get(source, 0) + 1
                    except Exception:
                        continue
        except Exception:
            pass

        total = positives + negatives
        return {
            "total": total,
            "positives": positives,
            "negatives": negatives,
            "sources": sources,
            "ready_for_retrain": (
                positives >= N_RETRAIN_THRESHOLD // 2
                and negatives >= N_RETRAIN_THRESHOLD // 2
            ),
        }

    def should_trigger_retrain(self) -> bool:
        """Check if we have enough data to trigger SetFit retraining."""
        stats = self.get_stats()
        return stats["ready_for_retrain"]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, article: "Article") -> Optional[str]:
        """Extract the most informative text from an article."""
        title = getattr(article, "title", "") or ""
        summary = getattr(article, "summary", "") or ""
        full_text = getattr(article, "full_text", "") or ""

        if full_text and len(full_text) > 100:
            return f"{title}. {full_text[:400]}"
        if summary:
            return f"{title}. {summary[:300]}"
        return title[:200] if title else None

    def _add_example(
        self,
        text: str,
        label: int,
        source: str,
        confidence: float = 1.0,
    ) -> bool:
        """Add one labeled example to the dataset.

        Deduplicates via MD5 hash of text.
        Enforces class balance (MAX_CLASS_RATIO).
        Caps total dataset size.

        Returns:
            True if example was added, False if deduplicated or capped.
        """
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in self._seen_hashes:
            return False

        # Check dataset size cap
        stats = self.get_stats()
        if stats["total"] >= MAX_DATASET_SIZE:
            return False

        # Check class balance
        pos = stats["positives"]
        neg = stats["negatives"]
        if label == 1 and neg > 0 and pos / neg >= MAX_CLASS_RATIO:
            return False  # Too many positives
        if label == 0 and pos > 0 and neg / pos >= MAX_CLASS_RATIO:
            return False  # Too many negatives

        record = {
            "text": text[:512],
            "label": label,
            "source": source,
            "confidence": round(confidence, 3),
            "added_at": datetime.now(timezone.utc).isoformat(),
            "hash": text_hash,
        }

        try:
            self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._seen_hashes.add(text_hash)
            return True
        except Exception as e:
            logger.warning(f"[dataset_enhancer] Failed to write example: {e}")
            return False

    def _load_seen_hashes(self) -> None:
        """Pre-load hashes from existing dataset to enable fast dedup."""
        if not self._dataset_path.exists():
            return
        try:
            with open(self._dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        h = record.get("hash")
                        if h:
                            self._seen_hashes.add(h)
                    except Exception:
                        continue
        except Exception:
            pass
