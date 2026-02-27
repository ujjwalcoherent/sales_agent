"""
Objective Specificity Score (OSS) -- measures how actionable a synthesis is.

Pure text analysis (no LLM self-rating). Counts objective properties:
  1. entity_density  (0.25): Named entities / total words
  2. numeric_density (0.20): Numbers, percentages, dates, currency amounts
  3. geo_specificity (0.20): Specific locations (not just "India")
  4. size_mention    (0.15): Employee count ranges present
  5. industry_depth  (0.20): Specific sub-segment vs generic "sector"

Primary autonomous learning signal for weight_learner.py. Computed AFTER
synthesis, BEFORE grading, logged per-cluster in pipeline metrics.

Not circular: OSS measures text properties while scoring weights influence
trend filtering -- different systems measuring different things.
"""

import functools
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Vague phrases that indicate low quality
VAGUE_PHRASES = frozenset({
    "companies need to adapt",
    "strategic optimization",
    "leverage opportunities",
    "navigate challenges",
    "various companies",
    "many organizations",
    "optimize their",
    "enhance their",
    "stakeholder engagement",
    "value chain",
    "holistic approach",
    "paradigm shift",
    "companies in the sector",
    "industry players",
    "companies must evaluate",
    "businesses should consider",
    "organizations need to",
    "companies should explore",
    "firms need to assess",
    "companies may be affected",
    "businesses across the sector",
    "enterprises in the industry",
    "technology companies",
    "financial services firms",
    "traditional businesses",
    "publicly traded companies",
})


from app.shared.geo import GENERIC_GEO as _GENERIC_GEO, get_geo_pattern as _build_geo_pattern, SPECIFIC_GEO_PATTERN as _SPECIFIC_GEO_PATTERNS

# Generic industry terms (don't count as specific sub-segments)
_GENERIC_INDUSTRY = frozenset({
    "technology", "tech", "it", "software", "digital",
    "financial", "finance", "banking", "fintech",
    "manufacturing", "industrial", "production",
    "automotive", "auto", "automobile",
    "healthcare", "pharma", "medical",
    "retail", "consumer", "e-commerce", "ecommerce",
    "energy", "power", "oil", "gas",
    "telecom", "telecommunications",
    "infrastructure", "construction", "real estate",
    "agriculture", "agri", "food",
    "education", "edtech",
    "logistics", "supply chain", "transportation",
    "media", "entertainment",
    "chemicals", "textiles",
    "services", "consulting",
})

# Specific sub-segment patterns (these count toward industry depth)
_SPECIFIC_SUBSEGMENT = re.compile(
    r'\b(?:'
    # Specific company types (plural-tolerant with s?)
    r'auto\s*parts?\s*suppliers?|component\s*manufacturers?|'
    r'jeweller[y]?\s*manufacturers?|silver\s*exporters?|gold\s*refiners?|'
    r'pcb\s*manufacturers?|chip\s*fabricat\w*|semiconductor\s*foundr\w*|'
    r'garment\s*exporters?|textile\s*mills?|spinning\s*mills?|'
    r'drug\s*formulation|api\s*manufacturers?|pharma\s*CMO|'
    r'ev\s*batter\w*|solar\s*panels?|wind\s*turbines?|'
    r'freight\s*forward\w*|cold\s*chain|last[- ]mile|'
    r'payment\s*gateways?|nbfc|microfinance|'
    r'organic\s*food|packaged\s*food|dairy\s*processors?|'
    r'defense\s*contractors?|mining\s*compan\w*|'
    r'discount\s*broker\w*|stock\s*broker\w*|'
    # Specific modifiers (tier-X, size-specific)
    r'tier[- ]?[123]\s*(?:supplier|vendor|manufacturer|company|city|broker)\w*|'
    r'mid[- ]?size|small[- ]?scale|micro[- ]?enterprise|'
    r'msme|sme|startup|'
    # Specific product/material focus
    r'silver[- ]based|gold[- ]dependent|steel[- ]intensive|'
    r'copper\s*wir\w*|aluminium\s*cast\w*|'
    r'oem\s*contracts?|contract\s*manufactur\w*'
    r')\b',
    re.IGNORECASE,
)

# Numeric patterns (percentages, currency, dates, quarters)
_NUMERIC_PATTERNS = re.compile(
    r'(?:'
    r'\d+\.?\d*\s*%'              # percentages: 12%, 3.5%
    r'|\$\s*\d+'                   # dollar amounts: $500
    r'|₹\s*\d+'                    # rupee amounts: ₹500
    r'|USD\s*\d+'                  # USD amounts
    r'|INR\s*\d+'                  # INR amounts
    r'|\d+\s*(?:crore|lakh|billion|million|thousand)'  # Indian/global units
    r'|Q[1-4]\s*(?:20\d{2}|FY)'   # quarter references: Q2 2025, Q1 FY
    r'|FY\s*\d{2,4}'              # fiscal year: FY25, FY2025
    r'|20[12]\d'                   # year: 2024, 2025, 2026
    r'|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'  # dates: 12/02/2025
    r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}'  # Feb 10
    r'|\d+[-–]\d+\s*(?:employee|staff|worker|people|person)'  # 50-200 employees
    r')',
    re.IGNORECASE,
)

# Employee count range patterns
_EMPLOYEE_RANGE = re.compile(
    r'(?:'
    r'\d+[-–]\d+\s*(?:employee|staff|worker|people|person)'
    r'|~?\d+\s*(?:employee|staff|worker)'
    r'|\d+[-–]\d+\s*(?:head\s*count|team\s*size)'
    r'|(?:50|100|200|500|1000|2000|5000)\+?\s*employee'
    r')',
    re.IGNORECASE,
)

# Named entity patterns (proper nouns, company names)
_PROPER_NOUN = re.compile(
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # CamelCase proper nouns
)
_ORG_SUFFIXES = re.compile(
    r'\b\w+\s+(?:Ltd|Inc|Corp|Group|Bank|Motors|Industries|Pharma|'
    r'Technologies|Solutions|Systems|Energy|Power|Steel|Cement|'
    r'Chemicals|Textiles|Foods|Beverages|Insurance|Finance|'
    r'Capital|Ventures|Holdings|Enterprises)\b',
    re.IGNORECASE,
)


def compute_specificity_score(synthesis: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Compute Objective Specificity Score for a synthesis output.

    Returns:
        (score, issues) where score is 0.0-1.0 (higher = more specific)
        and issues lists what's missing or generic.
    """
    issues: List[str] = []

    # Gather text from synthesis fields
    causal_chain = synthesis.get("causal_chain", [])
    if isinstance(causal_chain, str):
        causal_chain = [causal_chain]
    causal_text = " ".join(str(c) for c in causal_chain) if causal_chain else ""

    buying_intent = synthesis.get("buying_intent", {})
    if isinstance(buying_intent, dict):
        who_needs_help = buying_intent.get("who_needs_help", "")
        what_they_need = buying_intent.get("what_they_need", "")
        pitch_hook = buying_intent.get("pitch_hook", "")
    else:
        who_needs_help = ""
        what_they_need = ""
        pitch_hook = ""
    if not who_needs_help:
        who_needs_help = synthesis.get("who_needs_help", "")

    summary = synthesis.get("trend_summary", "")
    insight = synthesis.get("actionable_insight", "")
    event_5w1h = synthesis.get("event_5w1h", {})
    whom = event_5w1h.get("whom", "") if isinstance(event_5w1h, dict) else ""
    where = event_5w1h.get("where", "") if isinstance(event_5w1h, dict) else ""

    all_text = " ".join(filter(None, [
        causal_text, who_needs_help, what_they_need, pitch_hook,
        summary, insight, whom, where,
    ]))
    key_text = " ".join(filter(None, [
        causal_text, who_needs_help, whom, pitch_hook,
    ]))

    if not all_text.strip():
        return 0.0, ["No synthesis text to analyze"]

    total_words = len(all_text.split())

    # -- Component 1: Entity density (0.25) --
    proper_nouns = _PROPER_NOUN.findall(key_text)
    org_names = _ORG_SUFFIXES.findall(all_text)
    explicit_entities = (
        len(synthesis.get("affected_companies", []))
        + len(synthesis.get("key_entities", []))
    )
    meaningful_proper = [
        p for p in proper_nouns
        if p.lower() not in _GENERIC_GEO
        and p.lower() not in _GENERIC_INDUSTRY
        and len(p) > 2
    ]
    entity_count = len(meaningful_proper) + len(org_names) + explicit_entities
    entity_density = min(1.0, entity_count / 10.0)
    if entity_count < 3:
        issues.append(f"Low entity count ({entity_count}): name specific companies, people, regulations")

    # -- Component 2: Numeric density (0.20) --
    numeric_count = len(_NUMERIC_PATTERNS.findall(all_text))
    numeric_density = min(1.0, numeric_count / 5.0)
    if numeric_count < 2:
        issues.append(f"Low numeric density ({numeric_count}): include specific numbers, percentages, dates")

    # -- Component 3: Geo specificity (0.20) --
    geo_pattern = _build_geo_pattern(("IN",))
    unique_geos = set(g.lower() for g in geo_pattern.findall(all_text))
    geo_count = len(unique_geos)

    has_generic_geo = any(
        word.strip(".,;:!?'\"()-") in _GENERIC_GEO
        for word in all_text.lower().split()
    )

    if geo_count >= 3:
        geo_specificity = 1.0
    elif geo_count >= 2:
        geo_specificity = 0.8
    elif geo_count >= 1:
        geo_specificity = 0.5
    elif has_generic_geo:
        geo_specificity = 0.1
    else:
        geo_specificity = 0.0

    if geo_count == 0:
        issues.append("No specific geography: name cities, states, or industrial clusters")

    # -- Component 4: Size mention (0.15) --
    size_text = " ".join(filter(None, [who_needs_help, whom, causal_text]))
    has_employee_range = bool(_EMPLOYEE_RANGE.search(size_text))
    size_qualifiers = re.findall(
        r'\b(?:mid[- ]?size|small[- ]?scale|micro|msme|sme|'
        r'large[- ]?scale|startup|unicorn|'
        r'\d+[-–]\d+\s*(?:employee|staff|worker|people))\b',
        size_text, re.IGNORECASE
    )

    if has_employee_range:
        size_score = 1.0
    elif size_qualifiers:
        size_score = 0.5
    else:
        size_score = 0.0
        issues.append("No company size: specify employee count range (e.g., '50-200 employees')")

    # -- Component 5: Industry depth (0.20) --
    subsegment_count = len(_SPECIFIC_SUBSEGMENT.findall(all_text))
    key_words = set(key_text.lower().split())
    generic_industry_hits = key_words & _GENERIC_INDUSTRY

    if subsegment_count >= 2:
        industry_depth = 1.0
    elif subsegment_count >= 1:
        industry_depth = 0.7
    elif generic_industry_hits:
        industry_depth = 0.2
        issues.append(f"Generic industry terms only ({', '.join(generic_industry_hits)}): use specific sub-segments (e.g., 'Tier-2 auto parts suppliers' not 'automotive sector')")
    else:
        industry_depth = 0.0
        issues.append("No industry specificity: name specific company types or sub-segments")

    # -- Vagueness penalty --
    all_text_lower = all_text.lower()
    vague_hits = sum(1 for phrase in VAGUE_PHRASES if phrase in all_text_lower)
    vagueness_penalty = min(0.20, vague_hits * 0.04)
    if vague_hits > 0:
        issues.append(f"Contains {vague_hits} vague phrase(s): replace with specific details")

    # -- Composite score --
    score = (
        0.25 * entity_density
        + 0.20 * numeric_density
        + 0.20 * geo_specificity
        + 0.15 * size_score
        + 0.20 * industry_depth
        - vagueness_penalty
    )
    score = round(max(0.0, min(1.0, score)), 4)

    logger.debug(
        f"OSS={score:.3f} "
        f"(entity={entity_density:.2f}, numeric={numeric_density:.2f}, "
        f"geo={geo_specificity:.2f}, size={size_score:.2f}, "
        f"industry={industry_depth:.2f}, penalty=-{vagueness_penalty:.2f})"
    )

    return score, issues


def compute_oss(text: str) -> float:
    """Convenience wrapper: compute OSS directly from a plain text string.

    Routes the text through causal_chain, trend_summary, and buying_intent
    so all OSS components (entity, numeric, geo, size, industry) are
    evaluated. Useful for quick one-liner tests and external callers.
    """
    synth = {
        "causal_chain": [text],
        "trend_summary": text,
        "buying_intent": {"who_needs_help": text, "pitch_hook": text},
    }
    return compute_specificity_score(synth)[0]


def compute_specificity_batch(
    syntheses: Dict[int, Dict[str, Any]],
) -> Dict[int, Tuple[float, List[str]]]:
    """Compute OSS for a batch of cluster syntheses.

    Returns:
        {cluster_id: (score, issues)}
    """
    results = {}
    scores = []

    for cid, synthesis in syntheses.items():
        if not synthesis:
            results[cid] = (0.0, ["Empty synthesis"])
            continue
        score, issues = compute_specificity_score(synthesis)
        results[cid] = (score, issues)
        scores.append(score)

    if scores:
        import numpy as np
        arr = np.array(scores)
        logger.info(
            f"OSS batch ({len(scores)} clusters): "
            f"mean={arr.mean():.3f}, median={float(np.median(arr)):.3f}, "
            f"min={arr.min():.3f}, max={arr.max():.3f}, "
            f"above_0.4={int((arr >= 0.4).sum())}/{len(arr)}, "
            f"below_0.2={int((arr < 0.2).sum())}/{len(arr)}"
        )

    return results


def build_specificity_feedback(issues: List[str]) -> str:
    """Build a re-prompt instruction from OSS issues.

    Used when OSS < 0.4 to give the LLM specific feedback on what to fix.
    """
    if not issues:
        return ""

    feedback_lines = [
        "Your previous response was TOO GENERIC. Be MORE SPECIFIC about:",
    ]
    for issue in issues[:5]:  # Cap at 5 issues
        feedback_lines.append(f"  - {issue}")

    feedback_lines.extend([
        "",
        "REQUIREMENTS for a passing response:",
        "  - Include employee count ranges (e.g., '50-200 employees')",
        "  - Name specific cities or industrial clusters (not just 'India')",
        "  - Reference concrete numbers, percentages, dates from articles",
        "  - Use specific company types (e.g., 'Tier-2 auto parts suppliers in Pune')",
        "  - Name specific companies, regulators, policies from the articles",
    ])

    return "\n".join(feedback_lines)
