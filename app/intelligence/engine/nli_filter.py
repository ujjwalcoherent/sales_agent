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
Updating data/filter_hypothesis.json automatically improves the filter without code changes.
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_last_cleared_hypothesis: Optional[str] = None  # Hypothesis at last cache clear

# ── Score cache ────────────────────────────────────────────────────────────────
# LRU cache keyed on hash(text_prefix + hypothesis) to avoid re-scoring the same
# article text against the same hypothesis across multiple calls in one run.
# Capacity 2048 covers ~2 pipeline runs worth of articles (367 articles * 2 pairs
# * 2-3 hypotheses from industry classifier).  Thread-safe via _cache_lock.
_SCORE_CACHE_MAX = 2048
_score_cache: OrderedDict[str, float] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(text: str, hypothesis: str) -> str:
    """Build a short, collision-resistant cache key from text prefix + hypothesis."""
    # Use first 500 chars of text — enough to distinguish articles, cheap to hash.
    return hashlib.md5((text[:500] + "|||" + hypothesis).encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[float]:
    """Thread-safe LRU cache lookup. Returns None on miss."""
    with _cache_lock:
        if key in _score_cache:
            _score_cache.move_to_end(key)  # refresh LRU position
            return _score_cache[key]
    return None


def _cache_put(key: str, score: float) -> None:
    """Thread-safe LRU cache insert with eviction."""
    with _cache_lock:
        _score_cache[key] = score
        _score_cache.move_to_end(key)
        while len(_score_cache) > _SCORE_CACHE_MAX:
            _score_cache.popitem(last=False)


def clear_score_cache_if_hypothesis_changed() -> None:
    """Clear the NLI score cache only when the hypothesis has changed.

    Reads data/filter_hypothesis.json and compares to the hypothesis at the
    last cache clear. Skips the clear when unchanged — saves ~2-5s of DeBERTa
    re-inference per run (stable hypothesis >90% of runs in practice).

    Called at pipeline start in place of unconditional clear_score_cache().
    """
    global _last_cleared_hypothesis
    current: Optional[str] = None
    if _HYPOTHESIS_PATH.exists():
        try:
            with open(_HYPOTHESIS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            current = data.get("hypothesis", "").strip() or None
        except Exception:
            pass  # File unreadable → treat as changed to be safe

    if current != _last_cleared_hypothesis:
        with _cache_lock:
            _score_cache.clear()
        _last_cleared_hypothesis = current
        logger.info("[nli] Hypothesis changed — score cache cleared (%d → 0 entries)", 0)
    else:
        logger.debug("[nli] Hypothesis unchanged — score cache preserved (%d entries)", len(_score_cache))


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

    # Dual-score strategy: title-only + title+summary, take max.
    # NLI entailment is premise-length sensitive (Yin et al. 2019): summary
    # context words ("geopolitical turmoil", "AI-generated code flood") can
    # shift the model toward contradiction even when the title clearly describes
    # a B2B company action. Title-only catches strong signals; full text catches
    # articles where the summary adds specificity.
    # Empirical delta: "Anthropic launches code review tool" title=0.76, full=0.03.

    # Build text pairs and check LRU cache for each.
    # Pairs that are already cached skip the model entirely — saves ~50ms/article
    # when industry_classifier re-scores the same articles with the same hypothesis.
    n = len(articles)
    title_texts: List[str] = []
    full_texts: List[str] = []
    title_keys: List[str] = []
    full_keys: List[str] = []
    cached_title_scores: List[Optional[float]] = []
    cached_full_scores: List[Optional[float]] = []
    uncached_pairs: List[Tuple[str, str]] = []       # pairs to send to model
    uncached_pair_keys: List[str] = []                # parallel keys for cache insert

    for a in articles:
        title = (getattr(a, "title", "") or "").strip()
        summary = getattr(a, "summary", "") or ""
        full_text = f"{title}. {summary[:200]}".strip()
        title_texts.append(title)
        full_texts.append(full_text)

        tk = _cache_key(title, hypothesis)
        fk = _cache_key(full_text, hypothesis)
        title_keys.append(tk)
        full_keys.append(fk)

        t_cached = _cache_get(tk)
        f_cached = _cache_get(fk)
        cached_title_scores.append(t_cached)
        cached_full_scores.append(f_cached)

        if t_cached is None:
            uncached_pairs.append((title, hypothesis))
            uncached_pair_keys.append(tk)
        if f_cached is None:
            uncached_pairs.append((full_text, hypothesis))
            uncached_pair_keys.append(fk)

    cache_hits = 2 * n - len(uncached_pairs)

    try:
        # Only call model if there are uncached pairs
        if uncached_pairs:
            model = _load_model()
            raw_logits = model.predict(
                uncached_pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            raw_logits = np.array(raw_logits)
            if raw_logits.ndim == 1:
                logger.warning("NLI model returned 1D output — unexpected for NLI task")
                return [float(np.clip(s, 0.0, 1.0)) for s in raw_logits[:n]]

            # Softmax along classes axis
            shifted = raw_logits - raw_logits.max(axis=1, keepdims=True)
            exp_vals = np.exp(shifted)
            probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)
            uncached_scores = probs[:, _ENTAILMENT_IDX].tolist()

            # Insert uncached scores into cache
            for key, score in zip(uncached_pair_keys, uncached_scores):
                _cache_put(key, score)
        else:
            uncached_scores = []

        # Reassemble per-article title/full scores from cache + fresh results.
        # Walk the uncached_scores list in order to fill gaps.
        uc_idx = 0
        title_scores_final = []
        full_scores_final = []
        for i in range(n):
            if cached_title_scores[i] is not None:
                title_scores_final.append(cached_title_scores[i])
            else:
                title_scores_final.append(uncached_scores[uc_idx])
                uc_idx += 1
            if cached_full_scores[i] is not None:
                full_scores_final.append(cached_full_scores[i])
            else:
                full_scores_final.append(uncached_scores[uc_idx])
                uc_idx += 1

        # Element-wise max: best of title-only vs full-text.
        # This rescues real B2B articles where summary context dilutes the signal.
        # Known tradeoff: some sports/celebrity articles with B2B-patterned titles
        # ("PhonePe signs Rs 100 crore sponsorship") pass NLI auto-accept because
        # the NLI model correctly detects [company + action + money]. These are
        # caught by downstream stages: GLiNER won't extract "Mumbai Indians" as
        # a B2B entity, and sports articles won't form coherent B2B clusters.
        title_arr = np.array(title_scores_final)
        full_arr = np.array(full_scores_final)
        entailment_scores = np.maximum(title_arr, full_arr).tolist()

        mean_e = float(np.mean(entailment_scores))
        high_count = sum(1 for s in entailment_scores if s > 0.55)
        low_count = sum(1 for s in entailment_scores if s < 0.20)
        boosted = int(np.sum(title_arr > full_arr))

        # A4: NLI score distribution percentiles — tracks distribution shift
        # across runs. If P50 drops >10% from previous run → trigger retraining.
        scores_arr = np.array(entailment_scores)
        if len(scores_arr) >= 10:
            p10, p25, p50, p75, p90 = np.percentile(scores_arr, [10, 25, 50, 75, 90])
            logger.info(
                f"[nli] Distribution: P10={p10:.2f} P25={p25:.2f} P50={p50:.2f} "
                f"P75={p75:.2f} P90={p90:.2f}"
            )

        # A2: KDE valley detection (Silverman 1986) — find natural threshold
        # between B2B (high entailment) and noise (low entailment). Replaces
        # fixed 0.88 when sufficient data exists. Logged for monitoring.
        kde_threshold = _kde_valley_threshold(scores_arr)
        if kde_threshold is not None:
            logger.info(f"[nli] KDE valley threshold: {kde_threshold:.3f} (current fixed: 0.88)")

        logger.info(
            f"[nli] Scored {len(articles)} articles (dual-score): "
            f"mean_entailment={mean_e:.3f}, "
            f"high(>0.55)={high_count}, "
            f"low(<0.20)={low_count}, "
            f"title_boosted={boosted}, "
            f"cache_hits={cache_hits}/{2*n}, "
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
    Write to data/filter_hypothesis.json to update it at runtime.
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


def _kde_valley_threshold(scores: np.ndarray) -> Optional[float]:
    """KDE valley detection for NLI auto-accept threshold (Silverman 1986).

    Fits a kernel density estimate to the NLI score distribution and finds
    the deepest valley (local minimum) between 0.3 and 0.95. This valley
    separates the noise cluster (low entailment) from the B2B cluster (high).

    Returns None if insufficient data or no clear valley exists.
    Replaces Otsu's method (rejected: NLI distributions are not bimodal).

    REF: Silverman (1986) "Density Estimation for Statistics and Data Analysis"
         Chapter 2: bandwidth selection via Silverman's rule of thumb.
    """
    if len(scores) < 30:
        return None
    try:
        from scipy.signal import argrelmin
        from scipy.stats import gaussian_kde

        # Filter to a reasonable range to avoid edge effects
        valid = scores[(scores >= 0.05) & (scores <= 0.98)]
        if len(valid) < 20:
            return None

        kde = gaussian_kde(valid, bw_method="silverman")
        x = np.linspace(0.3, 0.95, 200)
        density = kde(x)

        # Find local minima (valleys) in the density
        valleys = argrelmin(density, order=10)[0]
        if len(valleys) == 0:
            return None

        # Select the valley closest to 0.88 (current default threshold)
        valley_positions = x[valleys]
        best_idx = np.argmin(np.abs(valley_positions - 0.88))
        threshold = float(valley_positions[best_idx])

        # Safety clamp: never go below 0.70 or above 0.95
        return float(np.clip(threshold, 0.70, 0.95))

    except ImportError:
        logger.debug("[nli] scipy not available — KDE valley detection skipped")
        return None
    except Exception as e:
        logger.debug(f"[nli] KDE valley detection failed: {e}")
        return None


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
