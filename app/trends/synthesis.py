"""
LLM synthesis -- generates structured trend analysis from article clusters.

Prompt tells the LLM about our service portfolio so it can recommend which
services to pitch, to whom, and why.

Features:
  - Intelligent article sampling (credibility + entity density + content length)
  - Source diversity enforcement (no more than half from same source)
  - Two-tier validation: critical (blocks return) vs warnings (coerce & continue)
  - Type coercion via _sanitize_synthesis_response()
  - OSS specificity validation with re-prompting on low scores
  - Retry with exponential backoff on failure
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
        return {"max_articles": 10, "char_limit": 2000, "max_retries": 3}


SYSTEM_PROMPT = (
    "You are a senior business intelligence analyst at a research and consulting firm. "
    "Your firm provides 9 service verticals: Procurement Intelligence, Market Intelligence, "
    "Competitive Intelligence, Market Monitoring, Industry Analysis, Technology Research, "
    "Cross-Border Expansion, Consumer Insights, and Consulting & Advisory. "
    "Analyze news trends and determine which of these services would be most valuable "
    "to the affected companies. Always respond with valid JSON."
)

SYNTHESIS_TEMPLATE = """Analyze this cluster of related news articles and extract structured intelligence.
Your goal: identify WHO is affected by this trend, extract SPECIFIC FACTS from the articles, and determine which services they need.

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
    "trend_summary": "2-3 paragraph summary: what happened, why it matters, who is affected, what comes next. INCLUDE specific numbers, dates, percentages, and names from the articles.",
    "trend_type": "One of: regulation, policy, funding, market, acquisition, merger, partnership, expansion, layoffs, hiring, product_launch, ipo, bankruptcy, technology, supply_chain, price_change, earnings, market_movement, infrastructure, geopolitical, consumer_shift, sustainability, crisis, general, emerging",
    "severity": "One of: high, medium, low",

    "event_5w1h": {{
        "who": "Primary actors/entities driving this trend (use NAMES from articles)",
        "what": "Core event or development (include specific numbers/dates if mentioned)",
        "whom": "Who is affected/impacted (specific company types, not just 'companies')",
        "when": "Timeline — specific dates, deadlines, quarters mentioned in articles",
        "where": "Geographic scope — specific cities, states, countries from articles",
        "why": "Root cause or motivation — cite the reason given in articles",
        "how": "Mechanism: the specific chain from event to business impact"
    }},

    "causal_chain": [
        "FACT: [Specific factual trigger from the articles, with numbers/names/dates — e.g., 'Silver prices dropped 12% in a single week after COMEX margin hikes on Feb 10']",
        "FIRST-ORDER: [Direct business consequence with specifics — e.g., 'Silver jewellery manufacturers in Rajkot and Mumbai face 8-15% raw material cost uncertainty, per industry body estimates']",
        "SECOND-ORDER: [Downstream/indirect effects on specific company types — e.g., 'Mid-size electronics manufacturers using silver contacts (50-200 employees) must decide whether to lock in forward contracts or absorb margin compression']",
        "DECISION POINT: [What specific decision affected companies face RIGHT NOW — e.g., 'Procurement teams need commodity price forecasting and should-cost analysis to renegotiate supplier contracts before Q2']"
    ],

    "buying_intent": {{
        "signal_type": "One of: compliance_need, growth_opportunity, crisis_response, technology_adoption, market_entry, restructuring, procurement_optimization, competitive_pressure",
        "urgency": "One of: immediate, short_term, medium_term",
        "who_needs_help": "Specific company types with employee range (e.g., 'mid-size silver jewellery exporters in Gujarat, 50-300 employees')",
        "what_they_need": "Exact service + offering from our portfolio (e.g., 'Procurement Intelligence: commodity price forecasting and should-cost analysis for silver-dependent components')",
        "pitch_hook": "Opening line for a cold email — reference the SPECIFIC news event and name the exact deliverable we'd provide. NOT a generic value proposition."
    }},

    "recommended_services": [
        {{
            "service": "Exact service name from our portfolio (e.g., 'Procurement Intelligence')",
            "relevance": "Why this service is needed given this trend — cite a SPECIFIC fact from the articles",
            "specific_offering": "Which specific sub-service applies (e.g., 'Commodity price forecasting')",
            "target_companies": "What kind of companies would buy this — be specific about size, sector, and geography"
        }}
    ],

    "affected_companies": ["ONLY companies explicitly mentioned BY NAME in the articles above"],
    "key_entities": ["ONLY companies, people, regulations, policies explicitly mentioned in the articles"],
    "primary_sectors": ["1-3 most affected sectors"],
    "affected_regions": ["Affected states/cities/countries mentioned in the articles"],
    "lifecycle_stage": "One of: emerging, growing, peak, declining",
    "actionable_insight": "A FACTUAL insight about the market situation (NOT a sales pitch). Example: 'Silver prices have dropped 12% this week, creating cost uncertainty for 500+ mid-size manufacturers in Rajkot and Mumbai who lack commodity hedging capability.' Do NOT mention our services here — this is a market observation."
}}

CRITICAL RULES:
1. ONLY include company names, entity names, and facts that are EXPLICITLY mentioned in the articles above.
2. Do NOT invent or hallucinate company names, statistics, or events.
3. If a field cannot be filled from the articles, use an empty list [] or "Not specified".
4. The affected_companies and key_entities lists must contain ONLY names found in the article text.
5. The causal_chain MUST contain specific facts, numbers, dates, or names from the articles. NEVER write generic chains like "Event happened → Companies affected → They need help → We can help." Each step must cite specific evidence.
6. The actionable_insight must be a FACTUAL market observation. Do NOT pitch our services in it. The buying_intent.pitch_hook is where the sales angle goes.
7. Do NOT repeat the same information across causal_chain, actionable_insight, and buying_intent. Each serves a different purpose:
   - causal_chain = factual cause-and-effect chain with evidence
   - actionable_insight = market observation (what's happening and why it matters)
   - buying_intent = specific sales angle (who to call, what to offer, why now)

MANDATORY SPECIFICITY CHECK — your response will be REJECTED if:
- who_needs_help says "companies in the [sector] sector" without employee count, geography, or sub-segment
- causal_chain steps say "companies may be affected" without naming WHO specifically
- pitch_hook is a generic value proposition instead of referencing the SPECIFIC news event

EXAMPLES OF BAD vs GOOD OUTPUT:
  BAD who_needs_help: "Technology companies, financial services firms, and traditional businesses with 500-5000 employees"
  GOOD who_needs_help: "Mid-size jewellery manufacturers and retailers in Rajkot and Mumbai (50-300 employees) who import raw silver for manufacturing"

  BAD causal_chain SECOND-ORDER: "Companies in the automotive sector may need to evaluate their strategies"
  GOOD causal_chain SECOND-ORDER: "Tier-2 auto parts suppliers in Pune (50-200 employees) who supply brake components to Tata Motors face 12% cost increase on imported steel. They must decide by Q2 whether to absorb margins or renegotiate OEM contracts."

  BAD pitch_hook: "We can help companies navigate this challenging environment with our market intelligence services"
  GOOD pitch_hook: "The 12% steel tariff hits your Q2 margins — we can benchmark your procurement costs against 5 competitors and identify 3 alternative suppliers within 2 weeks."

  BAD actionable_insight: "Companies in the sector need to adapt to changing market conditions"
  GOOD actionable_insight: "Silver prices dropped 12% this week after COMEX margin hikes, creating cost uncertainty for 500+ mid-size manufacturers in Rajkot and Mumbai who lack commodity hedging capability."
"""


def _select_representative_articles(articles: List, max_count: int = 16) -> List:
    """Select the most representative articles for LLM synthesis.

    Scores each article by: credibility (0.4) + entity_count (0.3) + content_length (0.3).
    Enforces source diversity: no more than half from same source.
    """
    if len(articles) <= max_count:
        return articles

    scored = []
    for article in articles:
        credibility = getattr(article, 'source_credibility', 0.5)
        entity_count = len(getattr(article, 'entity_names', []))
        content_len = len(article.content or article.summary or "")

        total_score = (
            0.4 * min(1.0, credibility)
            + 0.3 * min(1.0, entity_count / 15.0)
            + 0.3 * min(1.0, content_len / 5000.0)
        )
        scored.append((total_score, article))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    source_counts: Dict[str, int] = {}
    max_per_source = max(1, len(articles) // 2)

    for score, article in scored:
        source = getattr(article, 'source_name', '') or getattr(article, 'source_id', '')
        current_count = source_counts.get(source, 0)
        if current_count >= max_per_source:
            continue
        selected.append(article)
        source_counts[source] = current_count + 1
        if len(selected) >= max_count:
            break

    # Fill remaining if diversity enforcement was too strict
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


def _validate_synthesis(response: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate LLM synthesis output with two severity tiers.

    Returns {"critical": [...], "warnings": [...]}.
    Critical issues block return (caller retries); warnings are coerced by _sanitize.
    """
    critical: List[str] = []
    warnings: List[str] = []

    # Critical: fields that must exist for downstream pipeline
    title = response.get("trend_title", "")
    if not title or len(str(title).strip()) < 5:
        critical.append(f"trend_title too short or missing: '{title}'")

    summary = response.get("trend_summary", "")
    if not summary or len(str(summary).strip()) < 20:
        critical.append(f"trend_summary too short ({len(str(summary))} chars, need 20+)")

    # Warnings: fields with bad values that can be coerced
    trend_type = response.get("trend_type", "")
    if not trend_type:
        warnings.append("trend_type missing, will default to 'general'")

    severity = str(response.get("severity", "")).lower()
    if severity and severity not in _VALID_SEVERITIES:
        warnings.append(f"invalid severity '{severity}', will default to 'medium'")

    lifecycle = str(response.get("lifecycle_stage", "")).lower()
    if lifecycle and lifecycle not in _VALID_LIFECYCLE_STAGES:
        warnings.append(f"invalid lifecycle_stage '{lifecycle}', will default to 'emerging'")

    companies = response.get("affected_companies")
    if companies is not None and not isinstance(companies, list):
        warnings.append(f"affected_companies is {type(companies).__name__}, will coerce to list")

    event_5w1h = response.get("event_5w1h")
    if event_5w1h is not None:
        if not isinstance(event_5w1h, dict):
            warnings.append(f"event_5w1h is {type(event_5w1h).__name__}, will reset to {{}}")
        else:
            missing_keys = _5W1H_KEYS - set(event_5w1h.keys())
            if missing_keys:
                warnings.append(f"event_5w1h missing keys: {missing_keys}")

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

    causal = response.get("causal_chain")
    if causal is not None and not isinstance(causal, list):
        warnings.append(f"causal_chain is {type(causal).__name__}, will coerce to list")

    for field in ("key_entities", "primary_sectors", "affected_regions"):
        val = response.get(field)
        if val is not None and not isinstance(val, list):
            warnings.append(f"{field} is {type(val).__name__}, will coerce to list")

    return {"critical": critical, "warnings": warnings}


def _sanitize_synthesis_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce LLM synthesis output to expected types. Logs every correction."""
    coercions = 0

    if not response.get("trend_type"):
        response["trend_type"] = "general"
        coercions += 1

    sev = str(response.get("severity", "")).lower()
    if sev not in _VALID_SEVERITIES:
        response["severity"] = "medium"
        coercions += 1

    lifecycle = str(response.get("lifecycle_stage", "")).lower()
    if lifecycle not in _VALID_LIFECYCLE_STAGES:
        response["lifecycle_stage"] = "emerging"
        coercions += 1

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
            response[field] = [str(item).strip() for item in val if item and str(item).strip()]

    event = response.get("event_5w1h")
    if event is None or not isinstance(event, dict):
        response["event_5w1h"] = {}
        coercions += 1
    else:
        for key in _5W1H_KEYS:
            if key not in event or not event[key]:
                event[key] = "Not specified"
                coercions += 1

    intent = response.get("buying_intent")
    if intent is None or not isinstance(intent, dict):
        response["buying_intent"] = {}
        coercions += 1
    else:
        # Map common LLM aliases to valid enum values
        _SIGNAL_ALIASES = {
            "market": "market_entry", "market_movement": "market_entry",
            "market_shift": "market_entry", "expansion": "growth_opportunity",
            "growth": "growth_opportunity", "risk": "crisis_response",
            "disruption": "technology_adoption", "regulation": "compliance_need",
            "compliance": "compliance_need", "competition": "competitive_pressure",
            "procurement": "procurement_optimization", "restructure": "restructuring",
        }
        signal = str(intent.get("signal_type", "")).lower().strip()
        signal = _SIGNAL_ALIASES.get(signal, signal)
        if signal not in _VALID_SIGNAL_TYPES:
            intent["signal_type"] = "unknown"
            coercions += 1
        else:
            intent["signal_type"] = signal
        urgency = str(intent.get("urgency", "")).lower().strip()
        if urgency not in _VALID_URGENCIES:
            intent["urgency"] = "unknown"
            coercions += 1
        # Coerce all values to strings (prevent Pydantic validation errors)
        for key in list(intent.keys()):
            if intent[key] is not None and not isinstance(intent[key], str):
                intent[key] = str(intent[key])

    services = response.get("recommended_services")
    if services is not None and not isinstance(services, list):
        response["recommended_services"] = []
        coercions += 1

    # Normalize LLM-generated entity names to match NER output (avoid duplicates)
    try:
        from app.news.entity_normalizer import normalize_entities_batch
        for field in ("key_entities", "affected_companies"):
            if response.get(field):
                response[field] = normalize_entities_batch(response[field], deduplicate=True)
    except Exception:
        pass  # normalization is best-effort, don't block synthesis

    if coercions > 0:
        logger.info(f"Sanitized synthesis response: {coercions} field(s) coerced")

    return response


def build_synthesis_context(articles: List, keywords: List[str]) -> str:
    """Build the article context string for LLM synthesis."""
    config = _get_synthesis_config()
    max_articles = config["max_articles"]
    char_limit = config["char_limit"]

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
        sentiment = getattr(article, 'sentiment_score', 0.0)
        if sentiment != 0.0:
            sentiment_label = "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral")
            parts.append(f"    Sentiment: {sentiment_label} ({sentiment:.2f})")

    return "\n".join(parts)


async def synthesize_cluster(articles: List, keywords: List[str], llm_tool) -> Dict[str, Any]:
    """Generate structured trend analysis for a single cluster via LLM.

    Uses intelligent article sampling, two-tier validation (critical/warning),
    type coercion, and OSS specificity re-prompting.
    """
    config = _get_synthesis_config()
    max_retries = config["max_retries"]

    strict_mode = False
    try:
        from app.config import get_settings
        strict_mode = get_settings().synthesis_strict_mode
    except Exception:
        pass

    context = build_synthesis_context(articles, keywords)
    base_prompt = SYNTHESIS_TEMPLATE.format(context=context)

    # Inject commodity price hooks if relevant keywords present
    try:
        from app.tools.commodity_signals import get_commodity_signals
        commodity_signals = get_commodity_signals()
        hooks = commodity_signals.get_hooks_for_keywords(keywords or [])
        if hooks:
            base_prompt += (
                f"\n\nCURRENT COMMODITY PRICES (use exact figures in your response):\n{hooks}"
            )
            logger.debug(f"Synthesis: injected commodity hooks: {hooks}")
    except Exception as _ce:
        pass  # Non-fatal — synthesis continues without price data

    prompt = base_prompt

    last_error: Optional[str] = None
    last_validation: Optional[Dict[str, List[str]]] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await llm_tool.generate_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            # Unwrap single-element list wrapper from LLM
            if isinstance(response, list) and len(response) >= 1 and isinstance(response[0], dict):
                response = response[0]

            if not isinstance(response, dict):
                last_error = f"Non-dict response: {type(response).__name__}"
                logger.warning(
                    f"Synthesis attempt {attempt}/{max_retries} for cluster "
                    f"({len(articles)} articles): {last_error}"
                )
                continue

            if "error" in response and "trend_title" not in response:
                last_error = response.get("error", "unknown error")
                logger.warning(
                    f"Synthesis attempt {attempt}/{max_retries}: LLM error: {last_error}"
                )
                continue

            validation = _validate_synthesis(response)
            last_validation = validation

            critical = validation["critical"]
            warnings = validation["warnings"]

            if strict_mode and warnings:
                critical = critical + warnings
                warnings = []

            if critical:
                last_error = f"Critical validation: {critical}"
                logger.warning(
                    f"Synthesis attempt {attempt}/{max_retries}: "
                    f"critical issues (will retry): {critical}"
                )
                if attempt < max_retries:
                    sanitized = [
                        s[:200].replace("\n", " ") if isinstance(s, str) else str(s)[:200]
                        for s in critical
                    ]
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"IMPORTANT: Your previous response had these issues: {sanitized}\n"
                        f"Please fix them in your next response."
                    )
                continue

            if warnings:
                logger.info(
                    f"Synthesis attempt {attempt}: {len(warnings)} warning(s) "
                    f"(will sanitize): {warnings}"
                )

            response = _sanitize_synthesis_response(response)

            # Specificity validation: reject generic output
            try:
                from app.learning.specificity import (
                    compute_specificity_score, build_specificity_feedback,
                )
                oss, oss_issues = compute_specificity_score(response)
                response["_oss"] = oss
                response["_oss_issues"] = oss_issues

                if oss < 0.4 and attempt < max_retries:
                    feedback = build_specificity_feedback(oss_issues)
                    logger.info(
                        f"Synthesis attempt {attempt}: OSS={oss:.3f} < 0.4, "
                        f"re-prompting for specificity ({len(oss_issues)} issues)"
                    )
                    prompt = f"{base_prompt}\n\n{feedback}"
                    continue
                elif oss < 0.4:
                    logger.warning(
                        f"Synthesis: OSS={oss:.3f} (low) on final attempt, "
                        f"accepting with score recorded"
                    )
            except ImportError:
                pass

            logger.debug(
                f"Synthesis successful (attempt {attempt}): "
                f"'{response.get('trend_title', 'untitled')}' "
                f"(OSS={response.get('_oss', 'n/a')})"
            )
            return response

        except Exception as e:
            last_error = str(e)
            logger.warning(
                f"Synthesis attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
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
    """Synthesize a cluster with cross-validation.

    Flow: synthesize -> validate (no LLM) -> PASS/REJECT/REVISE.
    On REVISE, injects feedback and re-synthesizes (max 2 LLM calls per cluster).
    Falls back to unvalidated synthesis if validator is disabled or unavailable.
    """
    try:
        from app.config import get_settings
        settings = get_settings()
        if not settings.validator_enabled or settings.mock_mode:
            return await synthesize_cluster(articles, keywords, llm_tool)
    except Exception:
        return await synthesize_cluster(articles, keywords, llm_tool)

    try:
        from app.agents.workers.validator_agent import ValidatorAgent
        from app.schemas.validation import ValidationVerdict
    except ImportError as e:
        logger.warning(f"Validator not available ({e}), using unvalidated synthesis")
        return await synthesize_cluster(articles, keywords, llm_tool)

    max_rounds = settings.validator_max_rounds

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

    best_response = response
    for round_num in range(2, max_rounds + 1):
        feedback_str = validator.build_revision_feedback(result.rounds[-1])

        logger.info(
            f"V10: Cluster {cluster_id} needs revision (round {round_num}). "
            f"Injecting {len(result.rounds[-1].feedback)} feedback items."
        )

        context = build_synthesis_context(articles, keywords)
        revised_prompt = SYNTHESIS_TEMPLATE.format(context=context) + "\n\n" + feedback_str

        try:
            revised_response = await llm_tool.generate_json(
                prompt=revised_prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            if isinstance(revised_response, list) and len(revised_response) == 1 and isinstance(revised_response[0], dict):
                revised_response = revised_response[0]

            if not isinstance(revised_response, dict) or "error" in revised_response:
                logger.warning(
                    f"V10: Revision round {round_num} failed for cluster {cluster_id}"
                )
                break

            validation = _validate_synthesis(revised_response)
            if validation["critical"]:
                logger.warning(
                    f"V10: Revision has critical issues: {validation['critical']}"
                )
                break

            revised_response = _sanitize_synthesis_response(revised_response)
            best_response = revised_response

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

        except Exception as e:
            logger.warning(f"V10: Revision round {round_num} exception: {e}")
            break

    logger.warning(
        f"V10: Cluster {cluster_id} did not pass after {max_rounds} rounds "
        f"(final_score={result.final_score:.2f}). Returning best revision with caveat."
    )
    best_response["_validation"] = {
        "verdict": "revise_exhausted",
        "score": result.final_score,
        "rounds": max_rounds,
    }
    return best_response


async def synthesize_clusters(
    cluster_articles: Dict[int, List],
    cluster_keywords: Dict[int, List[str]],
    llm_tool,
    max_concurrent: int = 6,
    mock_mode: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Phase 8: LLM synthesis for all clusters with concurrency control."""
    use_validator = False
    try:
        from app.config import get_settings
        use_validator = get_settings().validator_enabled
    except Exception:
        pass
    # Never run validation in mock mode — mock LLM content won't pass specificity checks
    if mock_mode:
        use_validator = False

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
