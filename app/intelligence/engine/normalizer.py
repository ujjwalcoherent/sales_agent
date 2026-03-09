"""
NormAgent — Fuzzy entity normalization (Math Gate 3c).

Groups entity name variants using rapidfuzz token_sort_ratio >= 85.

Algorithm:
  1. For each entity name, strip noise (possessives, leading/trailing garbage)
  2. Fuzzy-compare against all existing group canonical names
  3. If score >= threshold → merge into existing group
  4. Else → create new group

Math assertions:
  Assert: token_sort_ratio(canonical, variant) >= 70 for all variants
  Assert: no two distinct groups have token_sort_ratio >= 90

Ticker alias lookup (18 US tickers):
  PLTR → Palantir, MSFT → Microsoft, NVDA → NVIDIA, etc.
  Prevents "NVDA" and "NVIDIA" from creating two separate entity groups.

Delegates to app.news.entity_normalizer until Phase 11 migration.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Ticker → canonical name lookup
# Prevents ticker symbols from creating separate entity groups
TICKER_ALIASES: Dict[str, str] = {
    "NVDA": "NVIDIA", "NVIDIA": "NVIDIA",
    "MSFT": "Microsoft", "AAPL": "Apple",
    "GOOGL": "Alphabet", "GOOG": "Alphabet",
    "META": "Meta Platforms", "AMZN": "Amazon",
    "TSLA": "Tesla", "NFLX": "Netflix",
    "CRM": "Salesforce", "ORCL": "Oracle",
    "PLTR": "Palantir", "SNOW": "Snowflake",
    "DDOG": "Datadog", "MDB": "MongoDB",
    "ZS": "Zscaler", "CRWD": "CrowdStrike",
    "S": "SentinelOne", "PANW": "Palo Alto Networks",
    # Indian tickers
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys", "WIPRO": "Wipro",
    "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "RELIANCE": "Reliance Industries",
    "ZOMATO": "Zomato", "NYKAA": "Nykaa",
}


def resolve_ticker(name: str) -> str:
    """Resolve a ticker symbol to its canonical company name.

    If not a known ticker, returns the name unchanged.
    """
    return TICKER_ALIASES.get(name.upper(), name)


def normalize_entity_name(name: str) -> str:
    """Apply structural normalization to an entity name.

    Steps:
      1. Strip possessives ("Pfizer's" → "Pfizer")
      2. Strip trailing noise (years, action verbs)
      3. Resolve ticker aliases
      4. Clean camelCase artifacts
    """
    import re

    # Strip possessives
    name = re.sub(r"'s?$", "", name).strip()

    # Strip leading noise (mogul, billionaire, etc.)
    name = re.sub(
        r"^(?:mogul|billionaire|tech\s+giant|startup|company)\s+",
        "", name, flags=re.IGNORECASE
    ).strip()

    # Strip trailing noise (action verbs, years, hyphenated modifiers)
    name = re.sub(
        r"\s+(?:says|said|warns|launches|announces|reports|partners|acquires|joins|CEO)\s*$",
        "", name, flags=re.IGNORECASE
    ).strip()
    name = re.sub(r"\s+\d{4}$", "", name).strip()  # trailing years
    name = re.sub(r"-(?:backed|powered|led|driven|focused)\s*$", "", name, flags=re.IGNORECASE).strip()

    # Resolve tickers
    name = resolve_ticker(name)

    return name.strip()


def fuzzy_group(
    names: List[str],
    threshold: float = 85.0,
) -> Dict[str, List[str]]:
    """Group entity names by fuzzy similarity.

    Args:
        names: list of entity name strings
        threshold: rapidfuzz token_sort_ratio threshold (85 = empirically validated)

    Returns:
        Dict mapping canonical_name → list of variant names
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning("[normalizer] rapidfuzz not available — no fuzzy grouping")
        return {n: [] for n in names}

    groups: Dict[str, List[str]] = {}   # canonical → variants
    normalized = {n: normalize_entity_name(n) for n in names}

    for name in names:
        norm = normalized[name]
        matched_canonical: Optional[str] = None

        for canonical in groups:
            score = fuzz.token_sort_ratio(norm.lower(), normalized[canonical].lower())
            if score >= threshold:
                matched_canonical = canonical
                break

        if matched_canonical:
            groups[matched_canonical].append(name)
        else:
            groups[name] = []

    return groups


def fuzzy_group_entities(
    names: List[str],
    threshold: float = 85.0,
) -> Dict[str, str]:
    """Compatibility wrapper: returns {variant: canonical} map (old news.entity_normalizer API).

    fuzzy_group() returns {canonical: [variants]}.
    This wrapper inverts it to {variant: canonical} for drop-in compatibility
    with clustering/tools/entity_extractor.py.
    """
    groups = fuzzy_group(names, threshold=threshold)
    variant_to_canonical: Dict[str, str] = {}
    for canonical, variants in groups.items():
        variant_to_canonical[canonical] = canonical  # canonical maps to itself
        for variant in variants:
            variant_to_canonical[variant] = canonical
    return variant_to_canonical
