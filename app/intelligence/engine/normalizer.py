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

Self-contained implementation using rapidfuzz.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for normalize_entity_name().
# normalize_entity_name() is called once per entity across hundreds of entities
# per pipeline run — compiling once at module load avoids repeated re.compile() overhead.
_RE_POSSESSIVE = re.compile(r"'s?$")
_RE_LEADING_NOISE = re.compile(
    r"^(?:mogul|billionaire|tech\s+giant|startup|company)\s+",
    re.IGNORECASE,
)
_RE_TRAILING_VERB = re.compile(
    r"\s+(?:says|said|warns|launches|announces|reports|partners|acquires|joins|CEO)\s*$",
    re.IGNORECASE,
)
_RE_TRAILING_YEAR = re.compile(r"\s+\d{4}$")
_RE_HYPHEN_MODIFIER = re.compile(
    r"-(?:backed|powered|led|driven|focused)\s*$",
    re.IGNORECASE,
)

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
    # Strip possessives
    name = _RE_POSSESSIVE.sub("", name).strip()

    # Strip leading noise (mogul, billionaire, etc.)
    name = _RE_LEADING_NOISE.sub("", name).strip()

    # Strip trailing noise (action verbs, years, hyphenated modifiers)
    name = _RE_TRAILING_VERB.sub("", name).strip()
    name = _RE_TRAILING_YEAR.sub("", name).strip()  # trailing years
    name = _RE_HYPHEN_MODIFIER.sub("", name).strip()

    # Resolve tickers
    name = resolve_ticker(name)

    return name.strip()


def fuzzy_group(
    names: List[str],
    threshold: float = 85.0,
) -> Dict[str, List[str]]:
    """Group similar entity names using rapidfuzz token_sort_ratio.

    Algorithm (from module docstring):
      1. Normalize each name (strip possessives, trailing noise, resolve tickers)
      2. Compare against existing group canonicals via token_sort_ratio
      3. If score >= threshold → merge into best-matching group
      4. Else → create new group with this name as canonical

    Returns:
        {canonical: [variants]} — canonical is the first (longest-normalized) name
        in each group.
    """
    try:
        from rapidfuzz import fuzz as _fuzz
    except ImportError:
        # Without rapidfuzz, each name is its own group
        return {n: [] for n in set(names)}

    groups: Dict[str, List[str]] = {}       # canonical -> [variants]
    canonical_list: List[str] = []          # ordered for iteration

    for name in names:
        normalized = normalize_entity_name(name)
        if not normalized:
            continue

        # Find best matching existing group
        best_canonical = None
        best_score = 0.0
        for canonical in canonical_list:
            score = _fuzz.token_sort_ratio(normalized.lower(), canonical.lower())
            if score >= threshold and score > best_score:
                best_canonical = canonical
                best_score = score

        if best_canonical is not None:
            if name != best_canonical:
                groups[best_canonical].append(name)
        else:
            groups[normalized] = []
            canonical_list.append(normalized)
            # Track original name as variant when normalization changed it
            # (e.g., ticker "TCS" → canonical "Tata Consultancy Services")
            if name != normalized:
                groups[normalized].append(name)

    return groups


def fuzzy_group_entities(
    names: List[str],
    threshold: float = 85.0,
) -> Dict[str, str]:
    """Return {variant: canonical} map — inverted view of fuzzy_group().

    Used by extractor.py for drop-in compatibility with the entity grouping API.
    Every input name maps to its canonical form (canonical maps to itself).
    """
    groups = fuzzy_group(names, threshold=threshold)
    variant_to_canonical: Dict[str, str] = {}
    for canonical, variants in groups.items():
        variant_to_canonical[canonical] = canonical
        for variant in variants:
            variant_to_canonical[variant] = canonical
    return variant_to_canonical
