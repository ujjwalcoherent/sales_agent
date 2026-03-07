"""
NLI-based article relevance filter.

Replaces manual keyword salience (Dunietz & Gillick 2014) with zero-shot
Natural Language Inference (NLI) classification.

Research basis:
  Yin et al. (2019) "Benchmarking Zero-shot Text Classification" — arXiv:1909.00161
  "Building Efficient Universal Classifiers with NLI" (2024) — arXiv:2312.17543
    → +9.4% F1 over keyword classifiers on zero-shot tasks

Model: cross-encoder/nli-deberta-v3-small (~60MB, local CPU, ~50ms/article)
  - Input: (premise=article_text, hypothesis=business_relevance_string)
  - Output: entailment / neutral / contradiction scores via softmax
  - No keyword lists. No regex. No manual maintenance.
  - Hypothesis is a CONFIG STRING loaded from data/filter_hypothesis.json

Label order (verified from model config): {0: contradiction, 1: entailment, 2: neutral}

Key property: The hypothesis string is the ONLY configuration knob.
When user feedback improves the hypothesis (via hypothesis_learner.py),
the filter automatically gets better without any code changes.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_HYPOTHESIS = (
    "This article reports on a specific company named in the text that is growing, "
    "raising capital, or making a strategic business move."
)
# HYPOTHESIS SELECTION — benchmark results (2026-03-08):
#
# H_entity_action (current): B2B_mean=0.859, Noise_mean=0.333, B2B_pass=7/8, Noise_rej=6/10
# H1 "enterprise purchasing decisions":  B2B_mean=0.611, Noise_mean=0.951 (leaks ALL geopolitics)
#
# H_entity_action wins because:
# 1. Requires a SPECIFIC NAMED COMPANY in the text — filters macro/geopolitical news naturally
# 2. "growing, raising capital, or making a strategic business move" = tight B2B event scope
# 3. Short + natural — NLI (SNLI/MultiNLI trained) best with brief, declarative hypotheses
# 4. Key noise articles: PM-US deal=0.116, India-US oil=0.008, ship insurance=0.001 → rejected
# 5. Key B2B articles: Zepto=0.989, Infosys Azure=0.988, Swiggy IPO=0.986, TCS Q3=0.981
#
# Known edge case: RBI penalty on Paytm scores 0.000 (regulatory news hits different frame).
# These go to LLM fallback via nli_auto_reject=0.05 threshold.

_HYPOTHESIS_PATH = Path("data/filter_hypothesis.json")
_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Index of "entailment" label in model output
# Verified from cross-encoder/nli-deberta-v3-small config.json:
#   id2label: {0: "contradiction", 1: "entailment", 2: "neutral"}
_ENTAILMENT_IDX = 1

# Thread-safe lazy-loaded singleton
_model_lock = threading.Lock()
_model_instance = None
_cached_hypothesis: Optional[str] = None


# ── Public API ────────────────────────────────────────────────────────────────

def score_articles(
    articles: List[Any],
    batch_size: int = 32,
    hypothesis: Optional[str] = None,
) -> List[float]:
    """Score articles with NLI entailment against the business relevance hypothesis.

    Args:
        articles: List of Article objects with .title and .summary attributes.
        batch_size: Inference batch size (32 = optimal for CPU DeBERTa).
        hypothesis: Override hypothesis string (None = load from config file).

    Returns:
        List of entailment scores in [0.0, 1.0] — one per article.
        Higher score = more likely to be a business-relevant article.
        On model failure: returns 0.40 for all (ambiguous → LLM decides).
    """
    if not articles:
        return []

    if hypothesis is None:
        hypothesis = load_hypothesis()

    # Build text pairs: (article_text, hypothesis)
    pairs = []
    for a in articles:
        title = getattr(a, "title", "") or ""
        summary = getattr(a, "summary", "") or ""
        # DeBERTa context limit: use title + first 200 chars of summary
        text = f"{title}. {summary[:200]}".strip()
        pairs.append((text, hypothesis))

    try:
        model = _load_model()

        # predict() returns raw logits, shape: (n_samples, n_labels)
        raw_logits = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # Apply softmax to convert logits → probabilities
        raw_logits = np.array(raw_logits)
        if raw_logits.ndim == 1:
            # Some model versions return (n,) for single class — shouldn't happen
            # but handle gracefully
            logger.warning("NLI model returned 1D output — unexpected for NLI task")
            return [float(np.clip(s, 0.0, 1.0)) for s in raw_logits]

        # Softmax along classes axis
        shifted = raw_logits - raw_logits.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)

        entailment_scores = probs[:, _ENTAILMENT_IDX].tolist()

        mean_e = float(np.mean(entailment_scores))
        high_count = sum(1 for s in entailment_scores if s > 0.55)
        low_count = sum(1 for s in entailment_scores if s < 0.20)

        logger.info(
            f"[nli] Scored {len(articles)} articles: "
            f"mean_entailment={mean_e:.3f}, "
            f"high(>0.55)={high_count}, "
            f"low(<0.20)={low_count}, "
            f"hypothesis_len={len(hypothesis)}"
        )

        return entailment_scores

    except Exception as e:
        logger.error(
            f"[nli] Scoring failed: {e} — returning 0.40 (ambiguous → LLM decides)"
        )
        # 0.40 falls in the ambiguous zone → LLM fallback handles these articles
        return [0.40] * len(articles)


def load_hypothesis() -> str:
    """Load hypothesis from data/filter_hypothesis.json, fallback to default.

    The hypothesis string is the sole configuration knob for the NLI filter.
    Updated by hypothesis_learner.py after accumulating user feedback.
    """
    global _cached_hypothesis

    # In-memory cache to avoid repeated file I/O on every article batch
    if _cached_hypothesis is not None:
        return _cached_hypothesis

    if _HYPOTHESIS_PATH.exists():
        try:
            with open(_HYPOTHESIS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            hypothesis = data.get("hypothesis", "").strip()
            if hypothesis:
                _cached_hypothesis = hypothesis
                logger.debug(
                    f"[nli] Loaded hypothesis v{data.get('version', '?')}: "
                    f"{hypothesis[:80]}..."
                )
                return _cached_hypothesis
        except Exception as e:
            logger.warning(f"[nli] Failed to load hypothesis: {e}")

    _cached_hypothesis = _DEFAULT_HYPOTHESIS
    return _DEFAULT_HYPOTHESIS


def invalidate_hypothesis_cache() -> None:
    """Force reload of hypothesis on next call (called by hypothesis_learner after update)."""
    global _cached_hypothesis
    _cached_hypothesis = None
    logger.info("[nli] Hypothesis cache invalidated — will reload from file on next call")


def get_hypothesis_version() -> str:
    """Return current hypothesis version string."""
    if _HYPOTHESIS_PATH.exists():
        try:
            with open(_HYPOTHESIS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return str(data.get("version", "v0"))
        except Exception:
            pass
    return "v0"


def warmup() -> bool:
    """Pre-load NLI model during app startup to avoid first-request latency."""
    try:
        _load_model()
        # Test with minimal pair to verify model works end-to-end
        test = type("A", (), {"title": "Test startup raises funding", "summary": ""})()
        scores = score_articles([test], batch_size=1)
        ok = len(scores) == 1 and 0.0 <= scores[0] <= 1.0
        if ok:
            logger.info(f"[nli] Model warmed up successfully. Test entailment={scores[0]:.3f}")
        return ok
    except Exception as e:
        logger.warning(f"[nli] Warmup failed: {e}")
        return False


# ── Internals ─────────────────────────────────────────────────────────────────

def _load_model():
    """Lazy-load the NLI CrossEncoder model (singleton, thread-safe)."""
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:
            return _model_instance

        logger.info(f"[nli] Loading model: {_MODEL_NAME} (first call — may take 10-30s)")

        try:
            # sentence-transformers >= 3.x: CrossEncoder at top level
            from sentence_transformers import CrossEncoder
        except ImportError:
            from sentence_transformers.cross_encoder import CrossEncoder

        _model_instance = CrossEncoder(_MODEL_NAME)
        logger.info(f"[nli] Model ready: {_MODEL_NAME}")

    return _model_instance


def nli_scores_by_source(
    articles: List[Any],
    scores: List[float],
) -> Dict[str, float]:
    """Compute mean NLI entailment score per source.

    Used by SourceBandit to reward sources that consistently produce
    business-relevant articles (Thompson Sampling reward signal).

    Research: Chapelle & Li (2011) Thompson Sampling — correct mechanism,
    but reward must reflect NLI entailment not just cluster membership.

    Args:
        articles: List of Article objects with .source_name attribute.
        scores: Parallel list of NLI entailment scores.

    Returns:
        {source_name: mean_entailment_score}
    """
    from collections import defaultdict
    source_scores: Dict[str, List[float]] = defaultdict(list)

    for article, score in zip(articles, scores):
        source = getattr(article, "source_name", "unknown") or "unknown"
        source_scores[source].append(score)

    return {
        src: float(np.mean(src_scores))
        for src, src_scores in source_scores.items()
    }
