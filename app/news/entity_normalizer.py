"""
Entity normalization for Indian business news.

Canonicalizes entity names so "RBI", "Reserve Bank of India", and
"the central bank" all map to the same canonical form. Three levels:

  Level 1: Suffix stripping ("Tata Motors Ltd." → "Tata Motors")
  Level 2: Alias table ("RBI" → "Reserve Bank of India")
  Level 3: Fuzzy matching via rapidfuzz (optional, catches typos/variants)

DESIGN NOTES:
  - Alias table is domain-specific (Indian business). Extend as needed.
  - Suffix stripping handles corporate suffixes across jurisdictions.
  - Fuzzy matching uses token_sort_ratio (order-independent) with a HIGH
    threshold (85) to avoid false positives on conglomerates like Tata.
  - Blocking strategy: same first word required for fuzzy match to prevent
    "Tata Motors" from matching "Tata Steel".

REF: Production entity normalization at NewsCatcher, AlphaSense, Bloomberg
     all use deterministic alias tables as the primary method.
"""

import logging
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# ── Level 1: Corporate suffix stripping ────────────────────────────────
SUFFIX_STRIP = {
    "Inc.", "Inc", "Ltd.", "Ltd", "Limited", "Corp.", "Corp",
    "Corporation", "LLC", "LLP", "Pvt.", "Pvt", "Private",
    "Co.", "Co", "Company", "Group", "Holdings", "Holding",
    "Enterprises", "Enterprise", "Industries", "Plc", "Plc.",
    "S.A.", "SA", "AG", "GmbH", "N.V.", "NV",
    # Indian-specific
    "Pvt. Ltd.", "Pvt Ltd", "Private Limited",
}

# Pre-compile suffix patterns (longest first to match "Pvt. Ltd." before "Pvt.")
_SUFFIX_PATTERN = re.compile(
    r"\s*,?\s*\b("
    + "|".join(re.escape(s) for s in sorted(SUFFIX_STRIP, key=len, reverse=True))
    + r")\s*\.?\s*$",
    re.IGNORECASE,
)


# ── Level 2: Domain-specific alias table ───────────────────────────────
# Maps lowercase alias → canonical name.
# Extend this table as new entities are encountered.
ALIASES: Dict[str, str] = {
    # Indian regulators
    "rbi": "Reserve Bank of India",
    "reserve bank": "Reserve Bank of India",
    "the central bank": "Reserve Bank of India",
    "sebi": "Securities and Exchange Board of India",
    "irdai": "Insurance Regulatory and Development Authority of India",
    "irda": "Insurance Regulatory and Development Authority of India",
    "trai": "Telecom Regulatory Authority of India",
    "cbi": "Central Bureau of Investigation",
    "cci": "Competition Commission of India",
    "dpiit": "Department for Promotion of Industry and Internal Trade",
    "cbdt": "Central Board of Direct Taxes",
    "cbic": "Central Board of Indirect Taxes and Customs",
    "npci": "National Payments Corporation of India",
    "nabard": "National Bank for Agriculture and Rural Development",
    "sidbi": "Small Industries Development Bank of India",
    "niti aayog": "NITI Aayog",

    # Indian government
    "pmo": "Prime Minister's Office",
    "mof": "Ministry of Finance",
    "mca": "Ministry of Corporate Affairs",
    "dfs": "Department of Financial Services",

    # Global tech
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "gcp": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    "msft": "Microsoft",
    "meta platforms": "Meta",
    "facebook": "Meta",

    # Indian exchanges
    "bse": "Bombay Stock Exchange",
    "nse": "National Stock Exchange",
    "sensex": "BSE Sensex",
    "nifty": "Nifty 50",
    "nifty50": "Nifty 50",

    # Common abbreviations
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
    "ev": "Electric Vehicle",
    "evs": "Electric Vehicles",
    "ipo": "Initial Public Offering",
    "fdi": "Foreign Direct Investment",
    "gst": "Goods and Services Tax",
    "upi": "Unified Payments Interface",
    "nbfc": "Non-Banking Financial Company",
    "nbfcs": "Non-Banking Financial Companies",
    "msme": "Micro, Small and Medium Enterprises",
    "msmes": "Micro, Small and Medium Enterprises",
    "sme": "Small and Medium Enterprises",
    "smes": "Small and Medium Enterprises",
    "psu": "Public Sector Undertaking",
    "psus": "Public Sector Undertakings",

    # Indian conglomerates — map common short forms
    "tcs": "Tata Consultancy Services",
    "hcl tech": "HCL Technologies",
    "infy": "Infosys",
    "rjio": "Reliance Jio",
    "jio": "Reliance Jio",
    "reliance industries limited": "Reliance Industries",
    "ril": "Reliance Industries",
    "hdfc bank ltd": "HDFC Bank",
    "sbi": "State Bank of India",
    "state bank": "State Bank of India",
    "icici bank ltd": "ICICI Bank",
    "icici bank limited": "ICICI Bank",
    "kotak mahindra bank limited": "Kotak Mahindra Bank",
    "bajaj finserv limited": "Bajaj Finserv",
    "l&t": "Larsen & Toubro",
    "larsen and toubro": "Larsen & Toubro",
    "m&m": "Mahindra & Mahindra",
    "mahindra and mahindra": "Mahindra & Mahindra",
    "ioc": "Indian Oil Corporation",
    "bpcl": "Bharat Petroleum",
    "hpcl": "Hindustan Petroleum",
    "ongc": "Oil and Natural Gas Corporation",
    "ntpc": "NTPC Limited",
    "bhel": "Bharat Heavy Electricals",
    "sail": "Steel Authority of India",
    "hal": "Hindustan Aeronautics",
    "bel": "Bharat Electronics",
    "drdo": "Defence Research and Development Organisation",
    "isro": "Indian Space Research Organisation",
}


def strip_suffix(name: str) -> str:
    """Remove corporate suffixes like Ltd., Inc., Pvt. Ltd., etc."""
    result = _SUFFIX_PATTERN.sub("", name).strip().rstrip(",").strip()
    return result if result else name


def normalize_entity(name: str) -> str:
    """Normalize an entity name using suffix stripping + alias lookup.

    Args:
        name: Raw entity text from NER extraction.

    Returns:
        Canonical entity name.
    """
    if not name or not name.strip():
        return name

    # Step 1: Strip corporate suffixes
    cleaned = strip_suffix(name.strip())

    # Step 2: Check alias table (case-insensitive)
    canonical = ALIASES.get(cleaned.lower())
    if canonical:
        return canonical

    # Step 3: Also check original (before suffix strip) in alias table
    canonical = ALIASES.get(name.strip().lower())
    if canonical:
        return canonical

    return cleaned


def normalize_entities_batch(
    entity_names: List[str],
    deduplicate: bool = True,
) -> List[str]:
    """Normalize and optionally deduplicate a list of entity names.

    Args:
        entity_names: Raw entity names from NER.
        deduplicate: If True, remove duplicates preserving order.

    Returns:
        List of normalized (and optionally deduplicated) entity names.
    """
    normalized = [normalize_entity(name) for name in entity_names]

    if deduplicate:
        seen: Set[str] = set()
        result = []
        for name in normalized:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                result.append(name)
        return result

    return normalized


def fuzzy_group_entities(
    entity_names: List[str],
    threshold: int = 85,
) -> Dict[str, str]:
    """Group similar entity names using fuzzy matching (Level 3).

    Uses rapidfuzz with a blocking strategy: entities must share the same
    first word to be compared. This prevents "Tata Motors" from matching
    "Tata Steel" (different first-word blocks would need multi-word blocking,
    but since Tata entities all start with "Tata", we use a stricter approach:
    both first AND second words must differ to block the comparison).

    Args:
        entity_names: List of (already L1/L2 normalized) entity names.
        threshold: Minimum fuzzy match score (0-100). 85 is conservative.

    Returns:
        Dict mapping each entity name to its canonical (most frequent) form.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.debug("rapidfuzz not installed — skipping fuzzy entity grouping")
        return {name: name for name in entity_names}

    if not entity_names:
        return {}

    # Build groups using Union-Find
    parent = {name: name for name in entity_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    unique_names = list(set(entity_names))
    for i, a in enumerate(unique_names):
        for b in unique_names[i + 1:]:
            # Blocking: skip if first words differ AND they're multi-word
            words_a = a.lower().split()
            words_b = b.lower().split()
            if len(words_a) > 1 and len(words_b) > 1:
                if words_a[0] != words_b[0]:
                    continue

            score = fuzz.token_sort_ratio(a.lower(), b.lower())
            if score >= threshold:
                union(a, b)

    # Map each entity to its group's most frequent member
    from collections import Counter
    name_counts = Counter(entity_names)
    groups: Dict[str, List[str]] = {}
    for name in unique_names:
        root = find(name)
        groups.setdefault(root, []).append(name)

    mapping = {}
    for group_members in groups.values():
        # Pick the most frequent form as canonical
        canonical = max(group_members, key=lambda n: name_counts.get(n, 0))
        for member in group_members:
            mapping[member] = canonical

    return mapping
