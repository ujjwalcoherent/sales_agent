"""
LLM synthesis — generates structured trend analysis from article clusters.

The prompt tells the LLM about our SPECIFIC service portfolio so it can
recommend which services to pitch, to whom, and why.

T5 IMPROVEMENTS:
  - Intelligent article sampling (credibility + entity density + content length)
  - Source diversity enforcement (no more than half from same source)
  - Pydantic validation of LLM output
  - Retry with exponential backoff on failure
  - All limits env-configurable

V3 IMPROVEMENTS:
  - Two-tier validation: critical (blocks return) vs warnings (coerce & continue)
  - _sanitize_synthesis_response() coerces types, fills missing keys, logs everything
  - Retry on critical failures instead of returning bad data
  - Strict mode (env SYNTHESIS_STRICT_MODE) promotes warnings to critical

REF: AlphaSense structured extraction, 6sense trigger event methodology
     BERTrend — uses all documents per topic for representation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.schemas.base import LifecycleStage, BuyingIntentSignalType, BuyingUrgency

logger = logging.getLogger(__name__)

# Valid values for LLM synthesis fields (used by validation)
_VALID_LIFECYCLE_STAGES = {e.value for e in LifecycleStage}
_VALID_SEVERITIES = {"high", "medium", "low"}
_VALID_SIGNAL_TYPES = {e.value for e in BuyingIntentSignalType}
_VALID_URGENCIES = {e.value for e in BuyingUrgency}
_5W1H_KEYS = {"who", "what", "whom", "when", "where", "why", "how"}


def _get_synthesis_config() -> Dict[str, Any]:
    """Load synthesis limits from env-configurable settings."""
    try:
        from app.config import get_settings
        s = get_settings()
        return {
            "max_articles": s.synthesis_max_articles,
            "char_limit": s.synthesis_article_char_limit,
            "max_retries": s.synthesis_max_retries,
        }
    except Exception:
        return {"max_articles": 16, "char_limit": 1200, "max_retries": 2}


SYSTEM_PROMPT = (
    "You are a senior business intelligence analyst at a research and consulting firm. "
    "Your firm provides 9 service verticals: Procurement Intelligence, Market Intelligence, "
    "Competitive Intelligence, Market Monitoring, Industry Analysis, Technology Research, "
    "Cross-Border Expansion, Consumer Insights, and Consulting & Advisory. "
    "Analyze news trends and determine which of these services would be most valuable "
    "to the affected companies. Always respond with valid JSON."
)

SYNTHESIS_TEMPLATE = """Analyze this cluster of related news articles and extract structured intelligence.
Your goal: identify WHO is affected by this trend and WHICH of our services they would pay for.

{context}

OUR SERVICE PORTFOLIO (recommend the most relevant ones):
1. Procurement Intelligence: Supplier identification, cost analysis, supply chain risk, benchmarking
2. Market Intelligence: Market sizing, trends, regulatory landscape, pricing, trade analysis
3. Competitive Intelligence: Competitor profiling, product comparisons, M&A tracking, go-to-market analysis
4. Market Monitoring: Real-time tracking, regulatory alerts, early warning, sentiment tracking
5. Industry Analysis: Value chain mapping, industry drivers, compliance review, key player identification
6. Technology Research: Tech trends, emerging tech assessment, patent analysis, vendor evaluation
7. Cross-Border Expansion: Market entry strategy, local partner identification, localization, risk assessment
8. Consumer Insights: Consumer behavior, segmentation, brand perception, customer journey mapping
9. Consulting & Advisory: Strategic planning, financial advisory, operational efficiency, digital transformation

Respond with JSON:
{{
    "trend_title": "Concise title (max 15 words)",
    "trend_summary": "2-3 paragraph summary: what happened, why it matters, who is affected, what comes next",
    "trend_type": "One of: regulation, policy, funding, acquisition, partnership, expansion, layoffs, hiring, product_launch, ipo, bankruptcy, technology, supply_chain, price_change, procurement, consumer_shift, general, emerging",
    "severity": "One of: high, medium, low",

    "event_5w1h": {{
        "who": "Primary actors/entities driving this trend",
        "what": "Core event or development",
        "whom": "Who is affected/impacted",
        "when": "Timeline (key dates, deadlines)",
        "where": "Geographic scope",
        "why": "Root cause or motivation",
        "how": "Mechanism of impact on businesses"
    }},

    "causal_chain": ["Step 1: Trigger event", "Step 2: First-order impact", "Step 3: Second-order business impact", "Step 4: Service opportunity for us"],

    "buying_intent": {{
        "signal_type": "One of: compliance_need, growth_opportunity, crisis_response, technology_adoption, market_entry, restructuring, procurement_optimization, competitive_pressure",
        "urgency": "One of: immediate, short_term, medium_term",
        "who_needs_help": "Specific company types (e.g., 'mid-size auto manufacturers', 'fintech startups expanding to SEA')",
        "what_they_need": "Specific services from OUR PORTFOLIO above (e.g., 'Procurement Intelligence: supply chain risk assessment', 'Market Intelligence: regulatory landscape assessment')",
        "pitch_hook": "One-sentence conversation starter referencing the specific trend and our service"
    }},

    "recommended_services": [
        {{
            "service": "Exact service name from our portfolio (e.g., 'Procurement Intelligence')",
            "relevance": "Why this service is needed given this trend (1 sentence)",
            "specific_offering": "Which specific sub-service applies (e.g., 'Supply base risk assessment')",
            "target_companies": "What kind of companies would buy this (e.g., 'Indian auto OEMs with global supply chains')"
        }}
    ],

    "affected_companies": ["ONLY companies explicitly mentioned BY NAME in the articles above"],
    "key_entities": ["ONLY companies, people, regulations, policies explicitly mentioned in the articles"],
    "primary_sectors": ["1-3 most affected sectors"],
    "affected_regions": ["Affected states/cities/countries mentioned in the articles"],
    "lifecycle_stage": "One of: emerging, growing, peak, declining",
    "actionable_insight": "One specific insight: WHO needs WHICH of our services, WHY now, and WHAT we should say to them."
}}

CRITICAL RULES:
1. ONLY include company names, entity names, and facts that are EXPLICITLY mentioned in the articles above.
2. Do NOT invent or hallucinate company names, statistics, or events.
3. If a field cannot be filled from the articles, use an empty list [] or "Not specified".
4. The affected_companies and key_entities lists must contain ONLY names found in the article text.
5. The causal_chain must be grounded in facts from the articles, not speculation."""


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT ARTICLE SAMPLING (T5 — replaces naive articles[:12])
# ══════════════════════════════════════════════════════════════════════════════

def _select_representative_articles(articles: List, max_count: int = 16) -> List:
    """
    Select the most representative articles for LLM synthesis.

    Scores each article by: credibility (0.4) + entity_count (0.3) + content_length (0.3).
    Enforces source diversity: no more than half of selected articles from same source.

    Edge cases:
    - Fewer articles than max_count → return all (no sampling needed)
    - All articles from same source → still select top-scored up to max
    - Articles without entities → entity score = 0, still eligible
    - Articles without content → content score = 0, still eligible
    """
    if len(articles) <= max_count:
        return articles

    # Score each article
    scored = []
    for article in articles:
        credibility = getattr(article, 'source_credibility', 0.5)
        entity_names = getattr(article, 'entity_names', [])
        entity_count = len(entity_names)
        content = article.content or article.summary or ""
        content_len = len(content)

        # Normalize: credibility already 0-1, entity_count cap at 15, content at 5000 chars
        cred_score = min(1.0, credibility)
        entity_score = min(1.0, entity_count / 15.0)
        content_score = min(1.0, content_len / 5000.0)

        total_score = (0.4 * cred_score) + (0.3 * entity_score) + (0.3 * content_score)
        scored.append((total_score, article))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Select with source diversity enforcement
    selected = []
    source_counts: Dict[str, int] = {}
    max_per_source = max(1, max_count // 2)  # No more than half from same source

    for score, article in scored:
        source = getattr(article, 'source_name', '') or getattr(article, 'source_id', '')
        current_count = source_counts.get(source, 0)
        if current_count >= max_per_source:
            continue
        selected.append(article)
        source_counts[source] = current_count + 1
        if len(selected) >= max_count:
            break

    # If diversity enforcement was too strict, fill remaining from top-scored
    if len(selected) < max_count:
        selected_ids = {id(a) for a in selected}
        for score, article in scored:
            if id(article) not in selected_ids:
                selected.append(article)
                if len(selected) >= max_count:
                    break

    logger.debug(
        f"Article sampling: {len(articles)} → {len(selected)} "
        f"(sources: {len(source_counts)}, top_score={scored[0][0]:.2f})"
    )
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS VALIDATION (V3 — two-tier: critical blocks, warnings coerce)
# ══════════════════════════════════════════════════════════════════════════════

def _validate_synthesis(response: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate LLM synthesis output with two severity tiers.

    Returns:
        {"critical": [...], "warnings": [...]}

    Critical issues (missing title/summary) → caller should retry.
    Warnings (bad enum, missing optional fields) → coerce via _sanitize.
    """
    critical: List[str] = []
    warnings: List[str] = []

    # ── CRITICAL: fields that MUST exist for downstream pipeline ──
    title = response.get("trend_title", "")
    if not title or len(str(title).strip()) < 5:
        critical.append(f"trend_title too short or missing: '{title}'")

    summary = response.get("trend_summary", "")
    if not summary or len(str(summary).strip()) < 20:
        critical.append(f"trend_summary too short ({len(str(summary))} chars, need 20+)")

    # ── WARNINGS: fields with bad values that can be coerced ──
    trend_type = response.get("trend_type", "")
    if not trend_type:
        warnings.append("trend_type missing, will default to 'general'")

    severity = str(response.get("severity", "")).lower()
    if severity and severity not in _VALID_SEVERITIES:
        warnings.append(f"invalid severity '{severity}', will default to 'medium'")

    # lifecycle_stage validation
    lifecycle = str(response.get("lifecycle_stage", "")).lower()
    if lifecycle and lifecycle not in _VALID_LIFECYCLE_STAGES:
        warnings.append(f"invalid lifecycle_stage '{lifecycle}', will default to 'emerging'")

    # affected_companies must be a list
    companies = response.get("affected_companies")
    if companies is not None and not isinstance(companies, list):
        warnings.append(f"affected_companies is {type(companies).__name__}, will coerce to list")

    # event_5w1h structure check
    event_5w1h = response.get("event_5w1h")
    if event_5w1h is not None:
        if not isinstance(event_5w1h, dict):
            warnings.append(f"event_5w1h is {type(event_5w1h).__name__}, will reset to {{}}")
        else:
            missing_keys = _5W1H_KEYS - set(event_5w1h.keys())
            if missing_keys:
                warnings.append(f"event_5w1h missing keys: {missing_keys}")

    # buying_intent structure check
    buying_intent = response.get("buying_intent")
    if buying_intent is not None:
        if not isinstance(buying_intent, dict):
            warnings.append(f"buying_intent is {type(buying_intent).__name__}, will reset to {{}}")
        else:
            signal = str(buying_intent.get("signal_type", "")).lower()
            if signal and signal not in _VALID_SIGNAL_TYPES:
                warnings.append(f"buying_intent.signal_type '{signal}' invalid")
            urgency = str(buying_intent.get("urgency", "")).lower()
            if urgency and urgency not in _VALID_URGENCIES:
                warnings.append(f"buying_intent.urgency '{urgency}' invalid")

    # causal_chain must be a list
    causal = response.get("causal_chain")
    if causal is not None and not isinstance(causal, list):
        warnings.append(f"causal_chain is {type(causal).__name__}, will coerce to list")

    # key_entities / primary_sectors must be lists
    for field in ("key_entities", "primary_sectors", "affected_regions"):
        val = response.get(field)
        if val is not None and not isinstance(val, list):
            warnings.append(f"{field} is {type(val).__name__}, will coerce to list")

    return {"critical": critical, "warnings": warnings}


def _sanitize_synthesis_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce LLM synthesis output to expected types. Logs every correction.

    Called AFTER validation — fixes warnings so downstream doesn't break.
    The TrendNode Pydantic validators (V1) add a second safety net, but
    sanitizing here means we log at the synthesis layer where fixes happen.
    """
    coercions = 0

    # trend_type: default to "general" if missing/invalid
    if not response.get("trend_type"):
        response["trend_type"] = "general"
        coercions += 1

    # severity: default to "medium" if invalid
    sev = str(response.get("severity", "")).lower()
    if sev not in _VALID_SEVERITIES:
        response["severity"] = "medium"
        coercions += 1

    # lifecycle_stage: default to "emerging" if invalid
    lifecycle = str(response.get("lifecycle_stage", "")).lower()
    if lifecycle not in _VALID_LIFECYCLE_STAGES:
        response["lifecycle_stage"] = "emerging"
        coercions += 1

    # List fields: coerce non-list to list
    _list_fields = [
        "affected_companies", "key_entities", "primary_sectors",
        "affected_regions", "causal_chain",
    ]
    for field in _list_fields:
        val = response.get(field)
        if val is None:
            response[field] = []
            coercions += 1
        elif isinstance(val, str):
            response[field] = [val.strip()] if val.strip() else []
            coercions += 1
        elif not isinstance(val, list):
            response[field] = [str(val)] if val else []
            coercions += 1
        else:
            # Filter None/empty from existing lists
            response[field] = [str(item).strip() for item in val if item and str(item).strip()]

    # event_5w1h: reset to {} if not dict, fill missing keys
    event = response.get("event_5w1h")
    if event is None or not isinstance(event, dict):
        response["event_5w1h"] = {}
        coercions += 1
    else:
        for key in _5W1H_KEYS:
            if key not in event or not event[key]:
                event[key] = "Not specified"
                coercions += 1

    # buying_intent: reset to {} if not dict, coerce signal_type/urgency
    intent = response.get("buying_intent")
    if intent is None or not isinstance(intent, dict):
        response["buying_intent"] = {}
        coercions += 1
    else:
        signal = str(intent.get("signal_type", "")).lower()
        if signal not in _VALID_SIGNAL_TYPES:
            intent["signal_type"] = "unknown"
            coercions += 1
        urgency = str(intent.get("urgency", "")).lower()
        if urgency not in _VALID_URGENCIES:
            intent["urgency"] = "unknown"
            coercions += 1

    # recommended_services: ensure it's a list of dicts
    services = response.get("recommended_services")
    if services is not None and not isinstance(services, list):
        response["recommended_services"] = []
        coercions += 1

    if coercions > 0:
        logger.info(f"Sanitized synthesis response: {coercions} field(s) coerced")

    return response


# ══════════════════════════════════════════════════════════════════════════════
# CORE SYNTHESIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_synthesis_context(articles: List, keywords: List[str]) -> str:
    """
    Build the article context string for LLM synthesis.

    Uses intelligent sampling (T5) and env-configurable limits.
    """
    config = _get_synthesis_config()
    max_articles = config["max_articles"]
    char_limit = config["char_limit"]

    # Use intelligent sampling instead of naive slicing
    sampled = _select_representative_articles(articles, max_count=max_articles)

    parts = [
        f"CLUSTER KEYWORDS: {', '.join(keywords[:10])}",
        f"ARTICLE COUNT: {len(articles)} (showing {len(sampled)} most representative)",
    ]

    event_types = {}
    for a in articles:
        evt = getattr(a, '_trigger_event', 'general')
        event_types[evt] = event_types.get(evt, 0) + 1
    if event_types:
        parts.append(f"EVENT TYPES: {event_types}")

    parts.append("")
    parts.append("ARTICLES:")

    for i, article in enumerate(sampled, 1):
        parts.append(f"\n[{i}] {article.title}")
        parts.append(f"    Source: {article.source_name} (credibility: {article.source_credibility:.1f})")
        body = article.content or article.summary or ""
        parts.append(f"    Content: {body[:char_limit]}")
        entity_names = getattr(article, 'entity_names', [])
        if entity_names:
            parts.append(f"    Entities: {', '.join(entity_names[:8])}")
        # Include sentiment for richer context
        sentiment = getattr(article, 'sentiment_score', 0.0)
        if sentiment != 0.0:
            sentiment_label = "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral")
            parts.append(f"    Sentiment: {sentiment_label} ({sentiment:.2f})")

    return "\n".join(parts)


async def synthesize_cluster(articles: List, keywords: List[str], llm_tool) -> Dict[str, Any]:
    """
    Generate structured trend analysis for a single cluster via LLM.

    T5 improvements:
    - Uses intelligent article sampling (credibility + entities + content)
    - Validates output structure
    - Retries on failure with logging per attempt

    V3 improvements:
    - Two-tier validation: critical (retry) vs warnings (sanitize & continue)
    - _sanitize_synthesis_response() coerces all bad types before return
    - Strict mode (SYNTHESIS_STRICT_MODE) treats all warnings as critical
    """
    config = _get_synthesis_config()
    max_retries = config["max_retries"]

    # V3: Load strict mode setting
    strict_mode = False
    try:
        from app.config import get_settings
        strict_mode = get_settings().synthesis_strict_mode
    except Exception:
        pass

    context = build_synthesis_context(articles, keywords)
    prompt = SYNTHESIS_TEMPLATE.format(context=context)

    last_error: Optional[str] = None
    last_validation: Optional[Dict[str, List[str]]] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await llm_tool.generate_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            if not isinstance(response, dict):
                last_error = f"Non-dict response: {type(response).__name__}"
                logger.warning(
                    f"Synthesis attempt {attempt}/{max_retries} for cluster "
                    f"({len(articles)} articles): {last_error}"
                )
                continue

            # Check for error dict (LLM tool wraps failures in {"error": ...})
            if "error" in response and "trend_title" not in response:
                last_error = response.get("error", "unknown error")
                logger.warning(
                    f"Synthesis attempt {attempt}/{max_retries}: LLM error: {last_error}"
                )
                continue

            # V3: Two-tier validation
            validation = _validate_synthesis(response)
            last_validation = validation

            critical = validation["critical"]
            warnings = validation["warnings"]

            # In strict mode, promote all warnings to critical
            if strict_mode and warnings:
                critical = critical + warnings
                warnings = []

            if critical:
                last_error = f"Critical validation: {critical}"
                logger.warning(
                    f"Synthesis attempt {attempt}/{max_retries}: "
                    f"critical issues (will retry): {critical}"
                )
                # Feed validation errors back to LLM on retry
                if attempt < max_retries:
                    prompt = (
                        f"{prompt}\n\n"
                        f"IMPORTANT: Your previous response had these issues: {critical}\n"
                        f"Please fix them in your next response."
                    )
                continue

            if warnings:
                logger.info(
                    f"Synthesis attempt {attempt}: {len(warnings)} warning(s) "
                    f"(will sanitize): {warnings}"
                )

            # V3: Sanitize all fields before returning
            response = _sanitize_synthesis_response(response)

            logger.debug(
                f"Synthesis successful (attempt {attempt}): "
                f"'{response.get('trend_title', 'untitled')}'"
            )
            return response

        except Exception as e:
            last_error = str(e)
            logger.warning(
                f"Synthesis attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                # Brief backoff before retry
                await asyncio.sleep(0.5 * attempt)

    logger.error(
        f"Synthesis exhausted all {max_retries} retries for cluster "
        f"({len(articles)} articles). Last error: {last_error}"
    )
    return {}


async def synthesize_cluster_validated(
    articles: List,
    keywords: List[str],
    llm_tool,
    cluster_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Synthesize a cluster with cross-validation (V10).

    Flow:
      1. synthesize_cluster() produces initial synthesis
      2. ValidatorAgent scores it against source articles (0 LLM calls)
      3. If PASS -> return as-is
      4. If REJECT -> return empty (cluster too fabricated)
      5. If REVISE -> inject feedback into prompt, call synthesize_cluster again
      6. ValidatorAgent scores revision -> final PASS or REJECT

    Budget: 1 synthesis call + 0-1 revision call = max 2 LLM calls per cluster.
    Validation itself uses NER + keywords + embeddings, no LLM.

    Falls back to unvalidated synthesis if validator import fails or is disabled.
    """
    try:
        from app.config import get_settings
        settings = get_settings()
        if not settings.validator_enabled:
            return await synthesize_cluster(articles, keywords, llm_tool)
    except Exception:
        return await synthesize_cluster(articles, keywords, llm_tool)

    try:
        from app.agents.validator_agent import ValidatorAgent
        from app.schemas.validation import ValidationVerdict
    except ImportError as e:
        logger.warning(f"Validator not available ({e}), using unvalidated synthesis")
        return await synthesize_cluster(articles, keywords, llm_tool)

    max_rounds = settings.validator_max_rounds

    # Round 1: Initial synthesis
    response = await synthesize_cluster(articles, keywords, llm_tool)
    if not response:
        return {}

    validator = ValidatorAgent()
    result = validator.validate(response, articles, cluster_id=cluster_id)

    if result.final_verdict == ValidationVerdict.PASS:
        logger.info(
            f"V10: Cluster {cluster_id} passed validation on first attempt "
            f"(score={result.final_score:.2f})"
        )
        # Tag the response with validation metadata
        response["_validation"] = {
            "verdict": "pass",
            "score": result.final_score,
            "rounds": 1,
        }
        return response

    if result.final_verdict == ValidationVerdict.REJECT:
        logger.warning(
            f"V10: Cluster {cluster_id} REJECTED (score={result.final_score:.2f}). "
            f"Synthesis too fabricated to salvage."
        )
        return {}

    # Round 2+: REVISE — regenerate with feedback
    for round_num in range(2, max_rounds + 1):
        feedback_str = validator.build_revision_feedback(result.rounds[-1])

        logger.info(
            f"V10: Cluster {cluster_id} needs revision (round {round_num}). "
            f"Injecting {len(result.rounds[-1].feedback)} feedback items."
        )

        # Build a new context with feedback appended
        config = _get_synthesis_config()
        context = build_synthesis_context(articles, keywords)
        revised_prompt = SYNTHESIS_TEMPLATE.format(context=context) + "\n\n" + feedback_str

        # Use generate_json directly for the revision (bypasses synthesize_cluster's
        # own retry loop — we already did that in round 1)
        try:
            revised_response = await llm_tool.generate_json(
                prompt=revised_prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            if not isinstance(revised_response, dict) or "error" in revised_response:
                logger.warning(
                    f"V10: Revision round {round_num} failed for cluster {cluster_id}"
                )
                break

            # Validate + sanitize the revision
            validation = _validate_synthesis(revised_response)
            if validation["critical"]:
                logger.warning(
                    f"V10: Revision has critical issues: {validation['critical']}"
                )
                break

            revised_response = _sanitize_synthesis_response(revised_response)

            # Score the revision
            result = validator.validate_with_revision(
                revised_response, articles,
                previous_result=result,
                cluster_id=cluster_id,
            )

            if result.final_verdict == ValidationVerdict.PASS:
                logger.info(
                    f"V10: Cluster {cluster_id} passed after revision "
                    f"(round {round_num}, score={result.final_score:.2f})"
                )
                revised_response["_validation"] = {
                    "verdict": "pass",
                    "score": result.final_score,
                    "rounds": round_num,
                }
                return revised_response

            if result.final_verdict == ValidationVerdict.REJECT:
                logger.warning(
                    f"V10: Cluster {cluster_id} REJECTED after revision "
                    f"(round {round_num}, score={result.final_score:.2f})"
                )
                return {}

            # Still REVISE — continue loop if more rounds available

        except Exception as e:
            logger.warning(f"V10: Revision round {round_num} exception: {e}")
            break

    # Exhausted all rounds without PASS — return last synthesis with warning
    logger.warning(
        f"V10: Cluster {cluster_id} did not pass after {max_rounds} rounds "
        f"(final_score={result.final_score:.2f}). Returning last synthesis with caveat."
    )
    response["_validation"] = {
        "verdict": "revise_exhausted",
        "score": result.final_score,
        "rounds": max_rounds,
    }
    return response


async def synthesize_clusters(
    cluster_articles: Dict[int, List],
    cluster_keywords: Dict[int, List[str]],
    llm_tool,
    max_concurrent: int = 6,
) -> Dict[int, Dict[str, Any]]:
    """
    Phase 8: LLM synthesis for all clusters with concurrency control.

    V10: Uses synthesize_cluster_validated when validator is enabled.
    Falls back to unvalidated synthesis if validator is disabled or unavailable.
    """
    # Check if validator is enabled
    use_validator = False
    try:
        from app.config import get_settings
        use_validator = get_settings().validator_enabled
    except Exception:
        pass

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(cid: int) -> tuple:
        async with semaphore:
            if use_validator:
                result = await synthesize_cluster_validated(
                    cluster_articles[cid],
                    cluster_keywords.get(cid, []),
                    llm_tool,
                    cluster_id=cid,
                )
            else:
                result = await synthesize_cluster(
                    cluster_articles[cid],
                    cluster_keywords.get(cid, []),
                    llm_tool,
                )
            return cid, result

    tasks = [_one(cid) for cid in cluster_articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = {}
    success_count = 0
    fail_count = 0
    validated_count = 0
    revised_count = 0
    rejected_count = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Synthesis task failed: {result}")
            fail_count += 1
            continue
        cid, summary = result
        summaries[cid] = summary
        if summary:
            success_count += 1
            # Track validation stats
            val_meta = summary.get("_validation", {})
            if val_meta:
                validated_count += 1
                if val_meta.get("rounds", 1) > 1:
                    revised_count += 1
        else:
            fail_count += 1
            if use_validator:
                rejected_count += 1

    validation_stats = ""
    if use_validator:
        validation_stats = (
            f", validated={validated_count}, revised={revised_count}, "
            f"rejected={rejected_count}"
        )

    logger.info(
        f"Phase 8 (synthesis): {success_count}/{len(cluster_articles)} clusters synthesized, "
        f"{fail_count} failed{validation_stats}"
    )
    return summaries
