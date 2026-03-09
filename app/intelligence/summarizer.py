"""
SynthesisAgent — FIRST LLM CALL (Math Gate 7).

This is step 8 of the 9-step pipeline. Steps 1-7 were all math.
The LLM only sees pre-validated, coherent clusters — never raw noisy articles.

Algorithm:
  1. Select top-k representative articles per cluster (centroid similarity)
  2. Build structured prompt with article titles + first paragraphs
  3. LLM generates: 3-8 word label + 2-3 sentence summary
  4. Math-validate output: word count, proper noun presence, no HTML/URLs
  5. If fails: Reflexion retry with specific critique (max 3 retries)
  6. After 3 failures: flag as requires_review, use fallback label

Reflexion pattern (Shinn et al. 2023):
  attempt 1 → validate → FAIL → critique → attempt 2 (with critique in context) → ...
  Stores last_critique in ClusterLabel for audit trail.

CRITIC pattern (Gou et al. 2023):
  math tool validates LLM output → critique fed back to LLM → retry with critique

Model: GPT-4.1-mini (via ProviderManager.get_model("standard"))
       Falls back to any available LLM via provider chain.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional, Tuple

from app.intelligence.config import ClusteringParams, DEFAULT_PARAMS
from app.intelligence.models import (
    Article, ClusterLabel, ClusterResult, CriticResult, EvidenceChain,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_REPRESENTATIVE_K = 5   # Articles to include in LLM prompt per cluster

# Synthesis prompt template
_SYSTEM_PROMPT = (
    "You are a senior B2B sales intelligence analyst. "
    "Your job: given a cluster of related news articles, generate:\n"
    "  1. A concise label (3-8 words) that names the key company/event\n"
    "  2. A summary (exactly 2-3 sentences) with specific facts from the articles\n\n"
    "Rules:\n"
    "  - Label must contain at least one proper noun (company name, product name)\n"
    "  - Label must be 3-8 words (count carefully)\n"
    "  - No HTML, URLs, markdown, or formatting\n"
    "  - Summary must contain specific facts: numbers, dates, company names\n"
    "  - Be specific, not generic. 'Q4 earnings miss' > 'financial news'"
)

_USER_TEMPLATE = """Here are {k} representative articles from a news cluster:

{articles_text}

Generate:
LABEL: [3-8 word label naming the key company/event]
SUMMARY: [2-3 sentence summary with specific facts]"""

_RETRY_TEMPLATE = """Here are {k} representative articles from a news cluster:

{articles_text}

PREVIOUS ATTEMPT FAILED: {critique}
Fix the specific issue above and retry.

Generate:
LABEL: [3-8 word label naming the key company/event]
SUMMARY: [2-3 sentence summary with specific facts]"""


async def synthesize_clusters(
    clusters: List[ClusterResult],
    articles: List[Article],
    params: Optional[ClusteringParams] = None,
) -> List[ClusterResult]:
    """Generate labels and summaries for all passed clusters.

    Processes clusters concurrently (max 5 at a time to respect rate limits).
    Updates cluster.label and cluster.summary in-place.

    Args:
        clusters: passed clusters from ValidationAgent
        articles: full filtered article list (for looking up article text)
        params: ClusteringParams with synthesis settings

    Returns:
        Updated clusters with label and summary populated.
    """
    if not clusters:
        return []

    if params is None:
        params = DEFAULT_PARAMS

    # Build article lookup map
    article_map = {a.run_index: a for a in articles if a.run_index >= 0}

    # Process with bounded concurrency
    sem = asyncio.Semaphore(5)
    tasks = [_synthesize_one(cluster, article_map, params, sem) for cluster in clusters]
    labels = await asyncio.gather(*tasks, return_exceptions=True)

    # Apply labels to clusters
    labeled = []
    for cluster, label in zip(clusters, labels):
        if isinstance(label, Exception):
            logger.warning(f"[synthesis] Failed for cluster {cluster.cluster_id}: {label}")
            cluster.label = _fallback_label(cluster)
            cluster.requires_review = True
        elif isinstance(label, ClusterLabel):
            cluster.label = label.label
            cluster.summary = label.summary
            cluster.requires_review = label.requires_review
            cluster.representative_article_indices = _select_representatives(
                cluster, article_map, params.synthesis_representative_k
            )
        labeled.append(cluster)

    labeled_count = sum(1 for c in labeled if not c.requires_review)
    logger.info(f"[synthesis] {labeled_count}/{len(labeled)} clusters labeled successfully")
    return labeled


async def _synthesize_one(
    cluster: ClusterResult,
    article_map: dict,
    params: ClusteringParams,
    sem: asyncio.Semaphore,
) -> ClusterLabel:
    """Synthesize label + summary for one cluster with Reflexion retry."""
    async with sem:
        # Select representative articles
        rep_indices = _select_representatives(cluster, article_map, params.synthesis_representative_k)
        rep_articles = [article_map[i] for i in rep_indices if i in article_map]

        if not rep_articles:
            return ClusterLabel(
                cluster_id=cluster.cluster_id,
                label=_fallback_label(cluster),
                summary="Insufficient article data for synthesis.",
                requires_review=True,
            )

        articles_text = _format_articles_for_prompt(rep_articles)
        last_critique = ""

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Build prompt (with critique from previous attempt if applicable)
                if attempt == 1:
                    user_msg = _USER_TEMPLATE.format(
                        k=len(rep_articles), articles_text=articles_text
                    )
                else:
                    user_msg = _RETRY_TEMPLATE.format(
                        k=len(rep_articles), articles_text=articles_text,
                        critique=last_critique,
                    )

                # LLM call
                raw = await _call_llm(user_msg)

                # Parse response
                label_text, summary_text = _parse_llm_response(raw)

                # Math validation (CRITIC pattern)
                valid, critique = _validate_label(label_text, summary_text, cluster)

                if valid:
                    return ClusterLabel(
                        cluster_id=cluster.cluster_id,
                        label=label_text,
                        summary=summary_text,
                        requires_review=False,
                        attempt_count=attempt,
                        last_critique="",
                    )
                else:
                    last_critique = critique
                    logger.debug(f"[synthesis] Attempt {attempt} failed for {cluster.cluster_id}: {critique}")

            except Exception as exc:
                last_critique = f"LLM error: {exc}"
                logger.warning(f"[synthesis] Attempt {attempt} error: {exc}")

        # All retries exhausted
        logger.warning(f"[synthesis] {cluster.cluster_id} failed after {_MAX_RETRIES} retries: {last_critique}")
        return ClusterLabel(
            cluster_id=cluster.cluster_id,
            label=_fallback_label(cluster),
            summary=f"[Auto-label: {_MAX_RETRIES} synthesis attempts failed. Last: {last_critique[:100]}]",
            requires_review=True,
            attempt_count=_MAX_RETRIES,
            last_critique=last_critique,
        )


async def _call_llm(user_message: str) -> str:
    """Call LLM via ProviderManager (GPT-4.1-mini, fallback chain)."""
    try:
        from app.tools.llm.llm_service import LLMService
        llm = LLMService()
        return await llm.generate(
            prompt=user_message,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=200,
            temperature=0.3,
        )
    except Exception:
        # Try direct OpenAI if LLMService fails
        from openai import AsyncOpenAI
        from app.config import get_settings
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""


def _parse_llm_response(raw: str) -> Tuple[str, str]:
    """Parse LABEL: ... SUMMARY: ... from LLM response."""
    label = ""
    summary = ""

    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            label = line[6:].strip()
        elif line.upper().startswith("SUMMARY:"):
            summary = line[8:].strip()

    # If structured parsing failed, try to extract from free text
    if not label:
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if lines:
            label = lines[0][:60]
    if not summary:
        rest = raw.strip()
        if label and label in rest:
            rest = rest[rest.find(label) + len(label):].strip()
        summary = rest[:300]

    return label.strip(), summary.strip()


def _validate_label(label: str, summary: str, cluster: ClusterResult) -> Tuple[bool, str]:
    """Math-validate LLM output (CRITIC pattern).

    Returns (is_valid, critique_string)
    Critique is SPECIFIC — tells the LLM exactly what to fix.
    """
    # Check 1: word count
    words = label.strip().split()
    if not (3 <= len(words) <= 8):
        return False, (
            f"Label '{label}' has {len(words)} words. Must be 3-8 words. "
            f"Example: 'NVIDIA Q4 Earnings Beat Estimates' (5 words)"
        )

    # Check 2: no HTML/URLs/markdown
    if re.search(r"<[^>]+>|https?://|\*\*|__|\[.*?\]", label):
        return False, f"Label contains HTML, URL, or markdown: '{label}'. Plain text only."

    # Check 3: at least one proper noun (capitalized word not at start)
    proper_nouns = [w for w in words if w[0].isupper() and not w.isupper()]
    if not proper_nouns and not any(w.isupper() and len(w) > 1 for w in words):
        return False, (
            f"Label '{label}' contains no proper noun. "
            f"Include the company name: '{cluster.primary_entity or 'company'}'"
        )

    # Check 4: summary is not empty
    if not summary or len(summary) < 20:
        return False, f"Summary too short ({len(summary)} chars). Must be 2-3 specific sentences."

    # Check 5: summary sentence count
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    if not (2 <= len(sentences) <= 4):
        return False, (
            f"Summary has {len(sentences)} sentences. Must be 2-3 sentences. "
            f"Current: '{summary[:100]}...'"
        )

    return True, ""


def _select_representatives(
    cluster: ClusterResult,
    article_map: dict,
    k: int = 5,
) -> List[int]:
    """Select top-k representative articles closest to cluster centroid.

    Uses centroid_embedding if available, else picks first k.
    """
    available = [i for i in cluster.article_indices if i in article_map]
    if len(available) <= k:
        return available

    if cluster.centroid_embedding and any(x != 0 for x in cluster.centroid_embedding):
        try:
            import numpy as np
            centroid = np.array(cluster.centroid_embedding)

            articles_with_emb = [
                (i, article_map[i]) for i in available
                if article_map[i].embedding
            ]
            if len(articles_with_emb) >= 2:
                embs = np.array([a.embedding for _, a in articles_with_emb])
                norms_e = np.linalg.norm(embs, axis=1, keepdims=True)
                norms_e = np.where(norms_e == 0, 1e-9, norms_e)
                norm_c = np.linalg.norm(centroid)
                norm_c = norm_c if norm_c > 1e-9 else 1e-9

                sims = (embs / norms_e) @ (centroid / norm_c)
                top_k = np.argsort(sims)[-k:][::-1]
                return [articles_with_emb[i][0] for i in top_k]
        except Exception:
            pass

    return available[:k]


def _format_articles_for_prompt(articles: List[Article]) -> str:
    """Format articles for the synthesis prompt."""
    parts = []
    for i, art in enumerate(articles, 1):
        title = art.title or "No title"
        # First 2 sentences of summary/text
        text = art.summary or art.full_text or ""
        sentences = re.split(r"(?<=[.!?])\s+", text)[:2]
        excerpt = " ".join(sentences)[:300]
        parts.append(f"{i}. [{art.source_name}] {title}\n   {excerpt}")
    return "\n\n".join(parts)


def _fallback_label(cluster: ClusterResult) -> str:
    """Generate a fallback label when synthesis fails."""
    entity = cluster.primary_entity or (cluster.entity_names[0] if cluster.entity_names else "")
    if entity:
        event = cluster.event_type or "news"
        return f"{entity} {event.replace('_', ' ')}"
    return f"Cluster {cluster.cluster_id[:6]}"


# ══════════════════════════════════════════════════════════════════════════════
# CRITIC VALIDATION (AutoResearch quality gate — D6)
# ══════════════════════════════════════════════════════════════════════════════

_CRITIC_PROMPT = (
    "You are a B2B sales intelligence quality reviewer. "
    "Evaluate whether this trend cluster represents a REAL, ACTIONABLE business event.\n\n"
    "Score 0.0-1.0 on these criteria:\n"
    "  - Is this a specific event (not vague topic)?\n"
    "  - Is it B2B relevant (affects purchasing decisions)?\n"
    "  - Is the label accurate for the articles?\n"
    "  - Is the summary factually grounded in the articles?\n\n"
    "Output EXACTLY:\n"
    "SCORE: [0.0-1.0]\n"
    "REASONING: [1-2 sentences]\n"
    "REFINED_LABEL: [improved label if score < 0.8, else leave empty]"
)

_CRITIC_THRESHOLD = 0.6


async def critic_validate_clusters(
    clusters: List[ClusterResult],
    articles: List[Article],
    params: Optional[ClusteringParams] = None,
    region: str = "GLOBAL",
) -> List[ClusterResult]:
    """Run critic validation on synthesized clusters.

    Clusters scoring < 0.6 get one re-synthesis attempt with the critic's
    refined label as guidance. Updates critic_score and critic_reasoning in-place.
    """
    if not clusters:
        return []

    if params is None:
        params = DEFAULT_PARAMS

    if not getattr(params, "enable_critic", True):
        return clusters

    article_map = {a.run_index: a for a in articles if a.run_index >= 0}
    sem = asyncio.Semaphore(5)

    tasks = [
        _critic_one(cluster, article_map, region, sem)
        for cluster in clusters
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    validated = []
    for cluster, result in zip(clusters, results):
        if isinstance(result, CriticResult):
            cluster.critic_score = result.score
            cluster.critic_reasoning = result.reasoning
            if not result.passed and result.refined_label:
                cluster.label = result.refined_label
                cluster.requires_review = True
        elif isinstance(result, Exception):
            logger.debug(f"[critic] Failed for {cluster.cluster_id}: {result}")
            cluster.critic_score = 0.5  # Assume neutral on failure
        validated.append(cluster)

    passed = sum(1 for c in validated if c.critic_score >= _CRITIC_THRESHOLD)
    logger.info(f"[critic] {passed}/{len(validated)} clusters passed critic validation")
    return validated


async def _critic_one(
    cluster: ClusterResult,
    article_map: dict,
    region: str,
    sem: asyncio.Semaphore,
) -> CriticResult:
    """Critic-validate one cluster."""
    async with sem:
        rep_indices = cluster.representative_article_indices or cluster.article_indices[:5]
        rep_articles = [article_map[i] for i in rep_indices if i in article_map]

        articles_text = _format_articles_for_prompt(rep_articles[:3]) if rep_articles else "No articles"

        user_msg = (
            f"LABEL: {cluster.label}\n"
            f"SUMMARY: {cluster.summary}\n"
            f"REGION: {region}\n"
            f"ENTITIES: {', '.join(cluster.entity_names[:5])}\n\n"
            f"SUPPORTING ARTICLES:\n{articles_text}"
        )

        try:
            from app.tools.llm.llm_service import LLMService
            llm = LLMService()
            raw = await llm.generate(
                f"{_CRITIC_PROMPT}\n\n{user_msg}",
                temperature=0.2,
                max_tokens=150,
            )
            return _parse_critic_response(raw)
        except Exception as exc:
            logger.debug(f"[critic] LLM call failed: {exc}")
            return CriticResult(score=0.5, passed=True, reasoning=f"Critic unavailable: {exc}")


def _parse_critic_response(raw: str) -> CriticResult:
    """Parse SCORE/REASONING/REFINED_LABEL from critic response."""
    score = 0.5
    reasoning = ""
    refined_label = ""

    for line in raw.strip().split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SCORE:"):
            try:
                score = float(line[6:].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif upper.startswith("REASONING:"):
            reasoning = line[10:].strip()
        elif upper.startswith("REFINED_LABEL:"):
            refined_label = line[14:].strip()

    return CriticResult(
        score=score,
        passed=score >= _CRITIC_THRESHOLD,
        reasoning=reasoning or "No reasoning provided",
        refined_label=refined_label if score < 0.8 else "",
    )


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE CHAIN (D7 — article → trend → lead citations)
# ══════════════════════════════════════════════════════════════════════════════

def build_evidence_chain(
    cluster: ClusterResult,
    articles: List[Article],
) -> EvidenceChain:
    """Build an explicit evidence chain from cluster articles.

    Extracts key snippets and company citations for use in lead emails.
    """
    article_map = {a.run_index: a for a in articles if a.run_index >= 0}
    rep_indices = cluster.representative_article_indices or cluster.article_indices[:5]
    rep_articles = [article_map[i] for i in rep_indices if i in article_map]

    article_ids = [a.id for a in rep_articles]
    companies_cited = list(set(cluster.entity_names[:10]))

    # Extract key snippets: first sentence of each representative article's summary
    snippets = []
    for art in rep_articles[:3]:
        text = art.summary or art.full_text or ""
        if text:
            first_sentence = re.split(r"(?<=[.!?])\s+", text)[0]
            if len(first_sentence) > 20:
                snippets.append(f"[{art.source_name}] {first_sentence[:200]}")

    return EvidenceChain(
        trend_id=cluster.cluster_id,
        article_ids=article_ids,
        key_snippets=snippets,
        companies_cited=companies_cited,
        confidence=cluster.confidence,
    )
