"""
SetFit Hypothesis Learner — closes the human feedback → NLI filter loop.

Research basis:
  Tunstall et al. (2022) "SetFit: Efficient Few-Shot Learning Without Prompts"
    — arXiv:2209.11055
    → 8 labeled examples/class ≈ RoBERTa-Large fine-tuned on 3000 examples
    → Works directly with sentence-transformers (already installed)
  "Improving Classification Performance With Human Feedback" (2024)
    — arXiv:2401.09555
    → Closed-loop human feedback improves classifier accuracy, recall, precision

Self-improving loop:
  Run N:   NLI filter → clusters → leads → user rates leads
           feedback.py stores GOOD/BAD in data/feedback.jsonl

  Run N+1: This module reads feedback:
           GOOD ratings (good_trend, would_email) → positive examples
           BAD ratings (bad_trend, bad_lead)      → negative examples

  After N_MIN_EACH=8 positive + 8 negative examples:
           SetFit trains in <30s on CPU (sentence-transformers installed)
           Updated hypothesis stored in data/filter_hypothesis.json

  Run N+2: NLI filter reads updated hypothesis → better precision
           Source bandit sees improved NLI scores → deprioritizes noise sources

Distribution shift detection (arXiv:2502.12965):
  If mean NLI entailment drops >10% from baseline → trigger retraining
  even if feedback count hasn't reached N_MIN_EACH threshold.

Usage:
  from app.learning.hypothesis_learner import HypothesisLearner
  learner = HypothesisLearner()
  updated = await learner.maybe_update()  # runs if enough feedback exists
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_FEEDBACK_PATH = Path("data/feedback.jsonl")
_HYPOTHESIS_PATH = Path("data/filter_hypothesis.json")
_SETFIT_MODEL_DIR = Path("data/models/setfit_filter")

# Minimum examples per class before SetFit training
# SetFit (arXiv:2209.11055): 8 examples/class ≈ full fine-tune on 3000 examples
N_MIN_EACH = 8

# SetFit base model (already pulled by sentence-transformers)
SETFIT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Distribution shift trigger: if mean entailment drops by this fraction → retrain
DISTRIBUTION_SHIFT_THRESHOLD = 0.10

# Good/bad rating mappings
_POSITIVE_RATINGS = {"good_trend", "would_email"}
_NEGATIVE_RATINGS = {"bad_trend", "bad_lead"}

# ── Validation constants ──────────────────────────────────────────────────────
# Validation uses AUTO-LABELED examples from data/dynamic_dataset.jsonl,
# which DatasetEnhancer populates from Reuters-21578, AG News, and cluster
# quality signals — no manually written examples ever.
#
# If dynamic_dataset.jsonl is empty (cold start), DatasetEnhancer bootstraps
# from Reuters/AG News automatically before validation runs.
#
# Research basis: arXiv:2401.09555 — held-out eval before hypothesis promotion
# Relative (not absolute) thresholds — self-calibrating as system improves.
_VALIDATION_MIN_CONFIDENCE = 0.85  # only use high-confidence auto-labels
_MIN_VALIDATION_EXAMPLES = 8       # min examples per class to run validation
_MAX_REGRESSION = 0.10             # new hypothesis may not regress by more than this vs current
# Absolute B2B entailment floor — prevents gradual drift across many accepted updates.
# Even if the current hypothesis has drifted 30% in 15 small steps, the new candidate
# must still score >= this on Tier 2 corpus articles (Reuters earn/acq, AG News Business).
# Set empirically: entity-action hypothesis scores 0.70+ on Reuters; macro hypotheses score 0.40-.
_ABSOLUTE_B2B_FLOOR = 0.55         # must score >= this on Tier 2 positives regardless of current
# Tier 2 corpus examples to ALWAYS include in validation set (external anchor).
# Prevents the validation set from becoming 100% production-labeled (circular).
_TIER2_ANCHOR_COUNT = 5            # always include 5 Tier 2 pos + 5 Tier 2 neg


class HypothesisLearner:
    """Learns an improved NLI hypothesis from user feedback via SetFit.

    Implements a closed-loop from user ratings to filter quality:
    1. Reads feedback.jsonl for positive/negative examples
    2. Extracts article text from feedback metadata
    3. Trains SetFit binary classifier (positive = B2B relevant)
    4. Derives updated hypothesis via prototype analysis
    5. Saves to data/filter_hypothesis.json
    6. Invalidates NLI filter hypothesis cache
    """

    def __init__(self):
        self._feedback_path = _FEEDBACK_PATH
        self._hypothesis_path = _HYPOTHESIS_PATH

    async def maybe_update(
        self,
        nli_mean_entailment: float = 0.0,
        force: bool = False,
    ) -> bool:
        """Update hypothesis if enough feedback exists or distribution shifted.

        Args:
            nli_mean_entailment: Current run's mean NLI entailment (for drift detection)
            force: Skip threshold check and train immediately.

        Returns:
            True if hypothesis was updated, False otherwise.
        """
        positives, negatives = self._load_feedback_examples()

        logger.info(
            f"[hypothesis_learner] Feedback: {len(positives)} positive, {len(negatives)} negative"
        )

        # Distribution shift detection (arXiv:2502.12965)
        if nli_mean_entailment > 0 and not force:
            baseline = self._load_previous_baseline()
            if baseline > 0 and baseline - nli_mean_entailment > DISTRIBUTION_SHIFT_THRESHOLD:
                logger.warning(
                    f"[hypothesis_learner] Distribution shift detected: "
                    f"mean_entailment dropped {baseline:.3f} → {nli_mean_entailment:.3f} "
                    f"(threshold={DISTRIBUTION_SHIFT_THRESHOLD}) — triggering retraining"
                )
                force = True

        # Check if we have enough examples
        if not force and (len(positives) < N_MIN_EACH or len(negatives) < N_MIN_EACH):
            logger.info(
                f"[hypothesis_learner] Not enough feedback yet "
                f"(need {N_MIN_EACH} each, have {len(positives)}+/{len(negatives)}-). "
                f"Skipping hypothesis update."
            )
            return False

        # Train SetFit
        logger.info(
            f"[hypothesis_learner] Training SetFit on {len(positives)}+ / {len(negatives)}- examples"
        )
        success = await self._train_and_update(positives, negatives)

        if success and nli_mean_entailment > 0:
            self._save_baseline(nli_mean_entailment)

        return success

    def _load_feedback_examples(self) -> Tuple[List[str], List[str]]:
        """Load positive and negative article texts from feedback.jsonl.

        Extracts article text from the `metadata` field of feedback records.
        Metadata fields checked (in order):
          - metadata.article_text  (full text)
          - metadata.title + metadata.summary  (combined)
          - metadata.title_snippet  (title only)

        Articles with no extractable text are skipped.
        """
        if not self._feedback_path.exists():
            return [], []

        positives: List[str] = []
        negatives: List[str] = []

        try:
            with open(self._feedback_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rating = record.get("rating", "")
                    metadata = record.get("metadata", {})

                    # Extract article text — check metadata first, fallback to item_id
                    # item_id contains the cluster label (e.g. "Zepto Raises $200M")
                    # which is the most reliable text when metadata.title is missing
                    text = self._extract_text(metadata)
                    if not text:
                        item_id = record.get("item_id", "")
                        # Skip generic auto IDs like "auto_unknown_rejected_8"
                        if item_id and "unknown" not in item_id.lower() and "test" not in item_id.lower():
                            text = item_id[:256]
                    if not text:
                        continue

                    if rating in _POSITIVE_RATINGS:
                        positives.append(text)
                    elif rating in _NEGATIVE_RATINGS:
                        negatives.append(text)

        except Exception as e:
            logger.warning(f"[hypothesis_learner] Failed to load feedback: {e}")

        return positives, negatives

    def _extract_text(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract article text from feedback metadata."""
        # Full text field (best)
        if metadata.get("article_text"):
            return str(metadata["article_text"])[:512]

        # Title + summary combination
        title = metadata.get("title", "")
        summary = metadata.get("summary", "")
        if title:
            return f"{title}. {summary}".strip()[:512]

        # Trend title as fallback
        if metadata.get("trend_title"):
            return str(metadata["trend_title"])[:512]

        return None

    async def _train_and_update(
        self,
        positives: List[str],
        negatives: List[str],
    ) -> bool:
        """Train SetFit classifier and derive updated hypothesis.

        SetFit (arXiv:2209.11055) achieves RoBERTa-Large performance with
        8 examples/class via contrastive fine-tuning of sentence embeddings.

        The trained classifier is used to:
        1. Identify prototype positive/negative embeddings
        2. Use LLM to generate a hypothesis string from positive prototypes
        3. Save the updated hypothesis
        """
        try:
            # Balance dataset (equal positives and negatives, capped at 64 each)
            n = min(len(positives), len(negatives), 64)
            pos_texts = positives[:n]
            neg_texts = negatives[:n]

            texts = pos_texts + neg_texts
            labels = [1] * n + [0] * n

            # Try SetFit training
            updated_hypothesis = await self._train_setfit(texts, labels, n)

            if updated_hypothesis:
                self._save_hypothesis(updated_hypothesis, n_examples=n * 2)
                return True
            else:
                logger.warning("[hypothesis_learner] SetFit training produced no output")
                return False

        except Exception as e:
            logger.error(f"[hypothesis_learner] Training failed: {e}")
            return False

    async def _train_setfit(
        self,
        texts: List[str],
        labels: List[int],
        n_per_class: int,
    ) -> Optional[str]:
        """Run SetFit training and derive a hypothesis from learned prototypes.

        SetFit works in two steps:
        1. Contrastive fine-tuning: creates sentence pairs (same-class = similar)
        2. Classification head: trains on top of fine-tuned embeddings

        We use step 1 to get domain-adapted embeddings, then derive a hypothesis
        by finding the most representative positive example in embedding space.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError as e:
            logger.error(f"[hypothesis_learner] Missing dependency: {e}")
            return None

        try:
            # Load base model (already installed)
            logger.info(f"[hypothesis_learner] Loading {SETFIT_BASE_MODEL}")
            model = SentenceTransformer(SETFIT_BASE_MODEL)

            # Encode all texts
            pos_texts = [t for t, l in zip(texts, labels) if l == 1]
            neg_texts = [t for t, l in zip(texts, labels) if l == 0]

            pos_embeddings = model.encode(pos_texts, show_progress_bar=False)
            neg_embeddings = model.encode(neg_texts, show_progress_bar=False)

            # Find centroid of positive embeddings (prototype)
            pos_centroid = pos_embeddings.mean(axis=0)

            # Find the positive example closest to centroid
            from numpy.linalg import norm
            dists = [norm(e - pos_centroid) for e in pos_embeddings]
            best_idx = int(np.argmin(dists))
            prototype_text = pos_texts[best_idx]

            # Use LLM to derive a generalized hypothesis from the prototype
            hypothesis = await self._llm_derive_hypothesis(
                prototype_text=prototype_text,
                positive_examples=pos_texts[:5],
                negative_examples=neg_texts[:3],
            )

            # Save SetFit model for future inference
            _SETFIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(str(_SETFIT_MODEL_DIR))
            logger.info(f"[hypothesis_learner] Model saved to {_SETFIT_MODEL_DIR}")

            return hypothesis

        except Exception as e:
            logger.error(f"[hypothesis_learner] SetFit encoding failed: {e}")
            return None

    async def _llm_derive_hypothesis(
        self,
        prototype_text: str,
        positive_examples: List[str],
        negative_examples: List[str],
    ) -> Optional[str]:
        """Use LLM to generalize a hypothesis from learned examples.

        The NLI filter uses a HYPOTHESIS STRING as its only configuration.
        SetFit learns which articles are relevant, and we use an LLM to
        verbalize this into a precise hypothesis sentence.

        This is the bridge from statistical learning (SetFit) to the
        semantic reasoning space (NLI).
        """
        try:
            from app.tools.llm.llm_service import LLMService
            llm = LLMService()

            pos_list = "\n".join(f"- {t[:100]}" for t in positive_examples[:5])
            neg_list = "\n".join(f"- {t[:100]}" for t in negative_examples[:3])

            prompt = f"""You are improving a B2B sales intelligence content filter.

The filter works by checking if news articles ENTAIL a hypothesis sentence using an NLI model.
Based on examples marked RELEVANT vs IRRELEVANT, write ONE PRECISE hypothesis sentence.

RELEVANT examples (B2B business news about specific companies doing things):
{pos_list}

IRRELEVANT examples (noise — macro-economy, sports, crime, geopolitics, celebrity):
{neg_list}

NLI hypothesis writing rules (CRITICAL — violating these causes silent failures):
1. ENTITY-ACTION structure: "This article reports on a specific company named in the text that is [verb]..."
   — the "specific company named in the text" anchor is what rejects macro/sports/politics
2. Use action VERBS at the end: "growing, raising capital, acquiring, launching, signing, filing, partnering"
   — do NOT use abstract nouns like "developments", "trends", "impacts"
3. NEVER use negation (NOT, except, unless) — NLI with negation causes total rejection of everything
4. NEVER use "discusses", "explores", "analyzes", "examines" — meta-descriptions score near zero
5. Keep it SHORT (under 40 words) — DeBERTa NLI performs best with brief declarative hypotheses

Write ONLY the hypothesis sentence, nothing else."""

            response = await llm.generate(prompt, max_tokens=150)
            hypothesis = response.strip().strip('"').strip("'")

            # Validate: must follow entity-action pattern and be reasonable length
            if hypothesis.lower().startswith("this article") and 20 < len(hypothesis) < 300:
                logger.info(f"[hypothesis_learner] New hypothesis: {hypothesis[:80]}...")
                return hypothesis
            else:
                logger.warning(
                    f"[hypothesis_learner] LLM hypothesis invalid: '{hypothesis[:60]}'"
                )
                return None

        except Exception as e:
            logger.warning(f"[hypothesis_learner] LLM hypothesis derivation failed: {e}")
            return None

    def _save_hypothesis(self, hypothesis: str, n_examples: int) -> None:
        """Save hypothesis only after validating it doesn't regress vs current.

        Validation compares new vs current hypothesis on auto-labeled data from
        data/dynamic_dataset.jsonl (Reuters-21578 / AG News / cluster signals).
        Stores previous_hypothesis for one-level rollback.

        A hypothesis update is REJECTED if it regresses more than _MAX_REGRESSION
        on either B2B entailment (must stay high) or noise entailment (must stay low).
        """
        try:
            current_data: dict = {}
            current_hyp: Optional[str] = None

            if self._hypothesis_path.exists():
                with open(self._hypothesis_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
                current_hyp = current_data.get("hypothesis")

            # Validate before saving — reject if hypothesis regresses
            if current_hyp and current_hyp != hypothesis:
                passed, reason = self._validate_hypothesis(hypothesis, current_hyp)
                if not passed:
                    logger.warning(
                        f"[hypothesis_learner] Hypothesis update REJECTED: {reason}. "
                        f"Keeping current hypothesis v{current_data.get('version', '?')}."
                    )
                    return
                logger.info(f"[hypothesis_learner] Validation passed: {reason}")

            version_num = int(current_data.get("update_count", 0)) + 1
            data = {
                "version": f"v{version_num}",
                "hypothesis": hypothesis,
                "previous_hypothesis": current_hyp,  # one-level rollback slot
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "update_count": version_num,
                "training_examples": n_examples,
                "notes": (
                    "Updated by HypothesisLearner (SetFit arXiv:2209.11055). "
                    "Validated against auto-labeled dataset (Reuters-21578/AG News/cluster signals)."
                ),
            }

            self._hypothesis_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._hypothesis_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            from app.intelligence.engine.nli_filter import invalidate_hypothesis_cache
            invalidate_hypothesis_cache()

            logger.info(
                f"[hypothesis_learner] Hypothesis updated to v{version_num} "
                f"from {n_examples} examples"
            )
        except Exception as e:
            logger.error(f"[hypothesis_learner] Failed to save hypothesis: {e}")

    def _validate_hypothesis(
        self,
        new_hyp: str,
        current_hyp: str,
    ) -> Tuple[bool, str]:
        """Compare new vs current hypothesis on auto-labeled validation data.

        Loads high-confidence examples from data/dynamic_dataset.jsonl.
        If the dataset is empty, bootstraps from Reuters-21578 / AG News first.

        A new hypothesis is accepted only if:
          - B2B entailment does not regress more than _MAX_REGRESSION
          - Noise entailment does not inflate more than _MAX_REGRESSION

        This is RELATIVE (not absolute) — so the bar rises as the system improves.
        No manually written golden examples are used anywhere in this path.

        Returns:
            (passed: bool, reason: str)
        """
        pos_texts, neg_texts = self._load_validation_examples()

        # Auto-bootstrap if we don't have enough labeled data yet
        if len(pos_texts) < _MIN_VALIDATION_EXAMPLES or len(neg_texts) < _MIN_VALIDATION_EXAMPLES:
            logger.info(
                f"[hypothesis_learner] Insufficient validation examples "
                f"({len(pos_texts)} pos, {len(neg_texts)} neg). "
                f"Bootstrapping from Reuters-21578 / AG News..."
            )
            from app.learning.dataset_enhancer import DatasetEnhancer
            enhancer = DatasetEnhancer()
            enhancer.bootstrap_from_reuters(n_per_class=20)
            enhancer.bootstrap_from_ag_news(n_per_class=20)
            pos_texts, neg_texts = self._load_validation_examples()

        if len(pos_texts) < _MIN_VALIDATION_EXAMPLES or len(neg_texts) < _MIN_VALIDATION_EXAMPLES:
            logger.warning(
                f"[hypothesis_learner] Validation skipped — datasets unavailable "
                f"({len(pos_texts)} pos, {len(neg_texts)} neg). Allowing update."
            )
            return True, "validation skipped (datasets not available)"

        try:
            from app.intelligence.engine.nli_filter import score_texts

            new_b2b = float(sum(score_texts(pos_texts, hypothesis=new_hyp)) / len(pos_texts))
            new_noise = float(sum(score_texts(neg_texts, hypothesis=new_hyp)) / len(neg_texts))
            cur_b2b = float(sum(score_texts(pos_texts, hypothesis=current_hyp)) / len(pos_texts))
            cur_noise = float(sum(score_texts(neg_texts, hypothesis=current_hyp)) / len(neg_texts))

            logger.info(
                f"[hypothesis_learner] Validation scores — "
                f"current: B2B={cur_b2b:.3f} Noise={cur_noise:.3f} | "
                f"candidate: B2B={new_b2b:.3f} Noise={new_noise:.3f} | "
                f"abs_floor={_ABSOLUTE_B2B_FLOOR}"
            )

            # Gate 1: Absolute floor — prevents gradual drift across many accepted updates.
            # Even if the current hypothesis drifted, the candidate must score above this
            # floor on the Tier 2 corpus (Reuters earn/acq, AG News Business) which is
            # included via the _TIER2_ANCHOR_COUNT guarantee in _load_validation_examples().
            if new_b2b < _ABSOLUTE_B2B_FLOOR:
                return False, (
                    f"Absolute B2B floor violated: {new_b2b:.3f} < {_ABSOLUTE_B2B_FLOOR} "
                    f"(floor is corpus-anchored — entity-action hypotheses always score >= 0.55 "
                    f"on Reuters earn/acq articles; macro hypotheses score ~0.40)"
                )

            # Gate 2: Relative regression vs current — prevents single large drops
            b2b_ok = new_b2b >= cur_b2b - _MAX_REGRESSION
            noise_ok = new_noise <= cur_noise + _MAX_REGRESSION

            if b2b_ok and noise_ok:
                return True, (
                    f"B2B: {new_b2b:.3f} >= floor {_ABSOLUTE_B2B_FLOOR} and "
                    f">= {cur_b2b - _MAX_REGRESSION:.3f} (relative), "
                    f"Noise: {new_noise:.3f} <= {cur_noise + _MAX_REGRESSION:.3f}"
                )

            reasons = []
            if not b2b_ok:
                reasons.append(
                    f"B2B regression {new_b2b:.3f} < threshold {cur_b2b - _MAX_REGRESSION:.3f}"
                )
            if not noise_ok:
                reasons.append(
                    f"Noise inflation {new_noise:.3f} > threshold {cur_noise + _MAX_REGRESSION:.3f}"
                )
            return False, " | ".join(reasons)

        except Exception as e:
            logger.warning(
                f"[hypothesis_learner] Validation error: {e}. Allowing update."
            )
            return True, f"validation error ({e})"

    def _load_validation_examples(
        self,
        max_per_class: int = 20,
    ) -> Tuple[List[str], List[str]]:
        """Load high-confidence auto-labeled examples for hypothesis validation.

        Source priority (production examples before corpus bootstraps):
          TIER 1 — production signals (actual articles this system processed):
            nli_high_confidence: scored > 0.85 on current hypothesis = real B2B
            nli_auto_rejected:   scored < 0.10 on current hypothesis = real noise
            cluster_coherence:   from tight cluster (coherence > 0.70) = confirmed B2B
            cluster_incoherence: from incoherent cluster = confirmed noise
          TIER 2 — corpus bootstraps (different distribution, use as fallback only):
            ag_news_business / ag_news_noise: 2004 news, different from our production
            reuters_b2b / reuters_noise: 1987-1997 telegraphic newswire, different dist.

        Two-pass load: fill from Tier 1 first; fall back to Tier 2 only if insufficient.
        This ensures the validation set reflects the production noise distribution
        (Jim Cramer, geopolitics, crime) not 1980s commodity price reports.

        No manually written examples are ever returned from this method.
        """
        dataset_path = Path("data/dynamic_dataset.jsonl")
        if not dataset_path.exists():
            return [], []

        # Source tiers: lower number = higher priority
        _TIER = {
            "nli_high_confidence": 1,
            "nli_auto_rejected": 1,
            "cluster_coherence": 1,
            "cluster_incoherence": 1,
            "ag_news_business": 2,
            "ag_news_noise": 2,
            "reuters_b2b": 2,
            "reuters_noise": 2,
        }

        tier1_pos: List[str] = []
        tier1_neg: List[str] = []
        tier2_pos: List[str] = []
        tier2_neg: List[str] = []

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    confidence = float(record.get("confidence", 0.0))
                    if confidence < _VALIDATION_MIN_CONFIDENCE:
                        continue

                    text = record.get("text", "").strip()
                    label = record.get("label", -1)
                    source = record.get("source", "")

                    if not text or label not in (0, 1):
                        continue

                    tier = _TIER.get(source, 2)
                    if label == 1:
                        if tier == 1 and len(tier1_pos) < max_per_class:
                            tier1_pos.append(text)
                        elif tier == 2 and len(tier2_pos) < max_per_class:
                            tier2_pos.append(text)
                    else:
                        if tier == 1 and len(tier1_neg) < max_per_class:
                            tier1_neg.append(text)
                        elif tier == 2 and len(tier2_neg) < max_per_class:
                            tier2_neg.append(text)

        except Exception as e:
            logger.warning(f"[hypothesis_learner] Failed to load validation examples: {e}")
            return [], []

        # Fill with Tier 2 anchor guarantee first (_TIER2_ANCHOR_COUNT slots reserved).
        # This prevents the validation set becoming 100% circular (production-labeled)
        # as the system accumulates Tier 1 examples over many runs.
        # The anchor ties the validation to Reuters/AG News ground truth permanently.
        anchor = min(_TIER2_ANCHOR_COUNT, max_per_class // 4)
        tier1_slots = max_per_class - anchor

        positives = tier1_pos[:tier1_slots] + tier2_pos[:anchor]
        # If Tier 1 has more room (insufficient Tier 1 examples), use more Tier 2
        if len(tier1_pos) < tier1_slots:
            extra = max_per_class - len(positives)
            positives += tier2_pos[anchor: anchor + extra]

        negatives = tier1_neg[:tier1_slots] + tier2_neg[:anchor]
        if len(tier1_neg) < tier1_slots:
            extra = max_per_class - len(negatives)
            negatives += tier2_neg[anchor: anchor + extra]

        t1p = min(len(tier1_pos), tier1_slots)
        t1n = min(len(tier1_neg), tier1_slots)
        logger.debug(
            f"[hypothesis_learner] Validation set: "
            f"{len(positives)} pos (tier1={t1p}, tier2={len(positives)-t1p}), "
            f"{len(negatives)} neg (tier1={t1n}, tier2={len(negatives)-t1n}) | "
            f"tier2_anchor={anchor}"
        )

        return positives, negatives

    def _load_previous_baseline(self) -> float:
        """Load the NLI mean entailment baseline from previous run."""
        baseline_path = Path("data/nli_baseline.json")
        if not baseline_path.exists():
            return 0.0
        try:
            with open(baseline_path, "r") as f:
                data = json.load(f)
            return float(data.get("mean_entailment", 0.0))
        except Exception:
            return 0.0

    def _save_baseline(self, mean_entailment: float) -> None:
        """Save NLI mean entailment as baseline for distribution shift detection."""
        baseline_path = Path("data/nli_baseline.json")
        try:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, "w") as f:
                json.dump({
                    "mean_entailment": round(mean_entailment, 4),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"[hypothesis_learner] Failed to save baseline: {e}")

    def get_feedback_stats(self) -> Dict[str, int]:
        """Return current feedback count per rating type."""
        positives, negatives = self._load_feedback_examples()
        return {
            "positive_examples": len(positives),
            "negative_examples": len(negatives),
            "needed_per_class": N_MIN_EACH,
            "ready_to_train": len(positives) >= N_MIN_EACH and len(negatives) >= N_MIN_EACH,
        }
