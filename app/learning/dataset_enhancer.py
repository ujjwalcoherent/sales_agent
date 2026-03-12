"""
Dataset Enhancer — Label accumulator for pipeline observational signals.

Accumulates positive/negative article labels from cluster quality and NLI scores
during each pipeline run. Labels are written to data/dynamic_dataset.jsonl for
future retraining cycles (not triggered automatically).

Auto-labeling strategy (no human input needed):
  POSITIVE examples — high confidence B2B business news:
    1. Cluster coherence > 0.70 AND NLI score > 0.55 (dual-gate for cluster articles)
    2. NLI entailment >= 0.85: filter kept high-confidence articles

  NEGATIVE examples — high confidence noise:
    1. NLI entailment < 0.10: auto-rejected by filter
    2. Cluster coherence < 0.25: incoherent cluster (single-gate — coherence alone sufficient)

Called from:
  app/intelligence/filter.py → extract_labels_from_filter()
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

logger = logging.getLogger(__name__)

_DATASET_PATH = Path("data/dynamic_dataset.jsonl")
_STATS_PATH = Path("data/dataset_stats.json")

# Minimum labeled examples per class before dataset is considered ready for retraining
N_RETRAIN_THRESHOLD = 50

# Quality thresholds for auto-labeling
POSITIVE_NLI_THRESHOLD = 0.85         # NLI score > this → positive (high confidence)

# Max class imbalance (positive:negative ratio)
MAX_CLASS_RATIO = 2.0

# Cap dataset size to avoid memory/training time blowup
MAX_DATASET_SIZE = 5000


class DatasetEnhancer:
    """Label accumulator — auto-labels articles from cluster quality and NLI signals.

    No human input required — uses pipeline observational signals:
    - Cluster coherence as quality signal (tight cluster = confirmed B2B)
    - NLI confidence as noise signal (very low score = confirmed noise)
    """

    def __init__(self):
        self._dataset_path = _DATASET_PATH
        self._stats_path = _STATS_PATH
        self._seen_hashes: set[str] = set()
        self._cached_positives: int = 0
        self._cached_negatives: int = 0
        self._load_seen_hashes()

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
        except Exception as e:
            logger.warning(f"Failed to read training dataset: {e}")

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

        Uses in-memory cached stats to avoid re-reading the JSONL file on every call.
        Stats were loaded once in _load_seen_hashes() and are updated atomically here.

        Returns:
            True if example was added, False if deduplicated or capped.
        """
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in self._seen_hashes:
            return False

        # Use cached counters — avoids O(dataset_size) file scan per call
        pos = self._cached_positives
        neg = self._cached_negatives
        total = pos + neg

        # Check dataset size cap
        if total >= MAX_DATASET_SIZE:
            return False

        # Check class balance
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
            if label == 1:
                self._cached_positives += 1
            else:
                self._cached_negatives += 1
            return True
        except Exception as e:
            logger.warning(f"[dataset_enhancer] Failed to write example: {e}")
            return False

    def _load_seen_hashes(self) -> None:
        """Pre-load hashes and label counts from existing dataset.

        Single pass: populates both the dedup hash set and cached stat counters
        so _add_example() never needs to re-read the file for stats.
        """
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
                        label = record.get("label", -1)
                        if label == 1:
                            self._cached_positives += 1
                        elif label == 0:
                            self._cached_negatives += 1
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to load training dataset hashes: {e}")
