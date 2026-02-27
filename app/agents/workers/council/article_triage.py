"""
Article Triage Agent — LLM-based batch classification for noise filtering.

Pre-clustering gate that uses LLM intelligence to separate business-relevant
articles from noise (entertainment, sports, lifestyle, celebrity, etc.).

ARCHITECTURE:
  - Receives articles that are ambiguous after Tier 1 embedding classification
    (event_type="general" OR low confidence, OR survived noise attractor but
     have low CMI relevance)
  - Batches them into groups of 10-12 for efficient LLM processing
  - Single LLM call per batch → structured output with per-article judgments
  - Track A (run_structured) with Track B (generate_json) fallback

PLACEMENT IN PIPELINE:
  After event classification + noise filter, before CMI relevance and clustering.
  Only articles that are ambiguous reach this agent — clear business articles
  and clear noise are handled by cheaper embedding-based methods.

COST MODEL:
  - 150 articles → ~20 ambiguous → 2 batch calls (10 articles each)
  - Each call: ~800 input tokens (system) + ~400 (articles) + ~300 output = ~1500 tokens
  - Total: ~3000 tokens per pipeline run (≈ $0.01 at Gemini Flash rates)

REF: LLM-as-Judge pattern (Zheng et al. 2023, "Judging LLM-as-a-Judge")
     Batch prompting (Cheng et al. 2023, "Batch Prompting")
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from ....config import get_settings
from ....tools.llm_service import LLMService

logger = logging.getLogger(__name__)

TRIAGE_SYSTEM_PROMPT = """You are a content triage specialist for a B2B market intelligence platform
(Coherent Market Insights). Your job is to classify news articles as BUSINESS-RELEVANT or NOISE.

BUSINESS-RELEVANT (is_business=true):
- Company earnings, revenue, profit reports
- Funding rounds, M&A, IPOs, partnerships
- Regulatory changes, government policy, compliance
- Supply chain disruptions, price changes, market movements
- Technology adoption, digital transformation, AI/cloud
- Infrastructure projects, expansion, market entry
- Layoffs, leadership changes, corporate restructuring
- Industry trends with business impact
- Geopolitical events affecting trade or markets

NOISE (is_business=false):
- Celebrity gossip, airport looks, fashion moments, wedding photos
- Movie/TV reviews, box office collections, OTT releases, trailers
- Sports scores, match results, player stats, tournament brackets
- Horoscopes, zodiac predictions, astrology
- Recipes, cooking tips, restaurant reviews, food trends
- Self-help, wellness, yoga, meditation, skincare advice
- Poetry, book reviews (unless about business books), literary criticism
- Travel guides, destination reviews (unless about business travel industry)
- Obituaries and death notices (unless of major business leaders)
- Opinion pieces with no business substance

EDGE CASES — mark as BUSINESS-RELEVANT:
- Bollywood/sports BUSINESS deals (IPL auction revenue, studio M&A, streaming rights deals)
- Celebrity BRAND endorsements or company launches
- Sports INFRASTRUCTURE projects (stadium construction, franchise valuations)
- Entertainment INDUSTRY business (studio mergers, content licensing deals)

Process each article INDEPENDENTLY. Do not let one article's classification influence another."""


async def triage_articles(
    articles: list,
    llm_service: LLMService = None,
    batch_size: int = None,
) -> Tuple[List, List]:
    """
    Batch-classify articles as business-relevant or noise using LLM.

    Args:
        articles: Articles to triage (typically the ambiguous subset).
        llm_service: LLM service instance (created if not provided).
        batch_size: Articles per LLM call (default from config, 10-12 optimal).

    Returns:
        Tuple of (kept_articles, filtered_articles)
    """
    if not articles:
        return [], []

    settings = get_settings()
    if batch_size is None:
        batch_size = settings.triage_batch_size
    max_articles = settings.triage_max_articles

    # Cost cap: limit number of articles sent to LLM
    overflow = []
    if len(articles) > max_articles:
        logger.info(
            f"Triage cost cap: {len(articles)} candidates > {max_articles} max, "
            f"keeping {len(articles) - max_articles} without triage"
        )
        overflow = articles[max_articles:]
        articles = articles[:max_articles]

    if llm_service is None:
        llm_service = LLMService(mock_mode=settings.mock_mode, lite=True)

    # Split into batches
    batches = [
        articles[i:i + batch_size]
        for i in range(0, len(articles), batch_size)
    ]

    # Process batches with limited concurrency
    semaphore = asyncio.Semaphore(5)
    all_judgments: Dict[int, dict] = {}  # global_idx → judgment

    async def _triage_batch(batch: list, batch_offset: int):
        async with semaphore:
            judgments = await _classify_batch(batch, llm_service)
            for local_idx, judgment in judgments.items():
                global_idx = batch_offset + local_idx
                all_judgments[global_idx] = judgment

    tasks = [
        _triage_batch(batch, i * batch_size)
        for i, batch in enumerate(batches)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Split articles based on judgments
    kept = []
    filtered = []
    for i, article in enumerate(articles):
        judgment = all_judgments.get(i)
        if judgment and not judgment.get("is_business", True):
            # Store triage metadata for debugging
            article._triage_verdict = "noise"
            article._triage_category = judgment.get("noise_category", "unknown")
            article._triage_confidence = judgment.get("confidence", 0.5)
            article._triage_reasoning = judgment.get("reasoning", "")
            filtered.append(article)
        else:
            article._triage_verdict = "business"
            kept.append(article)

    # Add overflow articles back (those that exceeded cost cap, kept without triage)
    kept.extend(overflow)

    logger.info(
        f"Article triage: {len(articles)} evaluated → "
        f"{len(kept)} kept, {len(filtered)} filtered as noise "
        f"({len(batches)} LLM batch calls)"
    )

    return kept, filtered


async def _classify_batch(
    batch: list,
    llm_service: LLMService,
) -> Dict[int, dict]:
    """Classify a single batch of articles via LLM.

    Uses Track A (run_structured) with Track B (generate_json) fallback.

    Returns:
        Dict mapping local batch index (0-based) to judgment dict.
    """
    # Build the article list for the prompt
    article_lines = []
    for i, article in enumerate(batch):
        title = (article.title or "").strip()[:120]
        summary = (article.summary or "")[:200].strip()
        event = getattr(article, '_trigger_event', 'general')
        conf = getattr(article, '_trigger_confidence', 0.0)
        article_lines.append(
            f"[{i+1}] Title: {title}\n"
            f"     Summary: {summary}\n"
            f"     Event classification: {event} (confidence: {conf:.2f})"
        )

    articles_text = "\n\n".join(article_lines)

    prompt = f"""Classify each article below as business-relevant or noise.
For each article [N], determine if it belongs on a B2B market intelligence platform.

Articles to classify:

{articles_text}

Classify ALL {len(batch)} articles. Output a JSON object with:
- "reasoning": brief overall batch assessment (1-2 sentences)
- "articles": array of objects, one per article in order:
  [{{"id": 1, "is_business": true/false, "confidence": 0.0-1.0, "noise_category": "none"/"entertainment"/"sports"/"lifestyle"/"opinion"/"astrology"/"other", "reasoning": "1 sentence"}}]"""

    # Track A: Structured output via pydantic-ai
    try:
        from ...schemas.llm_outputs import ArticleTriageBatchLLM
        result = await llm_service.run_structured(
            prompt=prompt,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            output_type=ArticleTriageBatchLLM,
        )
        return _parse_structured_result(result, len(batch))
    except Exception as e:
        logger.debug(f"Triage Track A failed: {e}")

    # Track B: Raw JSON fallback
    try:
        raw = await llm_service.generate_json(
            prompt=prompt,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
        )
        return _parse_json_result(raw, len(batch))
    except Exception as e:
        logger.warning(f"Triage both tracks failed: {e}")
        # Fail open: treat all as business-relevant
        return {}


def _parse_structured_result(result, batch_size: int) -> Dict[int, dict]:
    """Parse Track A structured output into judgment dict."""
    judgments = {}
    for item in result.articles:
        idx = item.id - 1  # Convert 1-indexed to 0-indexed
        if 0 <= idx < batch_size:
            judgments[idx] = {
                "is_business": item.is_business,
                "confidence": item.confidence,
                "noise_category": item.noise_category,
                "reasoning": item.reasoning,
            }
    return judgments


def _parse_json_result(raw: dict, batch_size: int) -> Dict[int, dict]:
    """Parse Track B raw JSON into judgment dict."""
    judgments = {}
    articles_list = raw.get("articles", [])
    if not isinstance(articles_list, list):
        return judgments

    for item in articles_list:
        if not isinstance(item, dict):
            continue
        idx = item.get("id", 0) - 1  # Convert 1-indexed to 0-indexed
        if 0 <= idx < batch_size:
            judgments[idx] = {
                "is_business": item.get("is_business", True),
                "confidence": float(item.get("confidence", 0.5)),
                "noise_category": item.get("noise_category", "none"),
                "reasoning": item.get("reasoning", ""),
            }
    return judgments


def select_triage_candidates(articles: list) -> Tuple[list, list]:
    """Split articles into those needing triage vs clear business articles.

    Triage candidates:
    - event_type == "general" (embedding classifier couldn't classify)
    - Low confidence on their event classification (< confidence floor)

    Clear articles skip the LLM call entirely (cheaper).

    Returns:
        (candidates_for_triage, clear_business_articles)
    """
    settings = get_settings()
    confidence_floor = settings.triage_confidence_floor

    candidates = []
    clear = []

    for article in articles:
        event = getattr(article, '_trigger_event', 'general')
        confidence = getattr(article, '_trigger_confidence', 0.0)

        # "general" articles always need triage — embeddings couldn't classify them
        if event == "general":
            candidates.append(article)
        # Low confidence on ANY event type → ambiguous, needs triage
        elif confidence < confidence_floor:
            candidates.append(article)
        # High-confidence valid business type → skip triage
        else:
            clear.append(article)

    return candidates, clear
