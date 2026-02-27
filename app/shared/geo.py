"""
Consolidated geo patterns — single source for geographic specificity detection.

Used by:
  - app.learning.specificity (OSS geo component)
  - app.trends.coherence (cluster quality geo signal)
"""
from __future__ import annotations

import functools
import logging
import re

logger = logging.getLogger(__name__)


# Generic geography terms (don't count as specific)
def get_generic_geo() -> frozenset:
    """Build the generic-geo set dynamically from app settings if available."""
    try:
        from app.config import get_settings
        country = get_settings().country.lower()
        country_adj = country + "n" if not country.endswith("n") else country
    except Exception:
        country, country_adj = "india", "indian"
    return frozenset({
        country, country_adj, "global", "worldwide", "international",
        "asia", "asian", "domestic", "foreign", "overseas",
        "country", "region", "world",
    })


GENERIC_GEO = get_generic_geo()


# Hardcoded geo pattern — Indian cities, states, industrial clusters, key international
SPECIFIC_GEO_PATTERN = re.compile(
    r'\b(?:'
    # Major cities
    r'mumbai|delhi|bangalore|bengaluru|chennai|hyderabad|pune|kolkata|'
    r'ahmedabad|jaipur|lucknow|kanpur|nagpur|indore|thane|bhopal|'
    r'visakhapatnam|patna|vadodara|ghaziabad|ludhiana|agra|nashik|'
    r'faridabad|meerut|rajkot|surat|coimbatore|kochi|noida|gurgaon|'
    r'gurugram|chandigarh|trivandrum|thiruvananthapuram|mysore|mysuru|'
    # States
    r'maharashtra|karnataka|tamil\s*nadu|telangana|andhra\s*pradesh|'
    r'gujarat|rajasthan|uttar\s*pradesh|west\s*bengal|kerala|'
    r'madhya\s*pradesh|punjab|haryana|bihar|odisha|jharkhand|'
    r'chhattisgarh|assam|uttarakhand|himachal|goa|'
    # Industrial clusters
    r'peenya|bhiwandi|tiruppur|jamnagar|'
    # Key international
    r'paris|london|tokyo|singapore|dubai|beijing|shanghai|'
    r'new\s*york|san\s*francisco|silicon\s*valley|'
    r'france|germany|japan|uk|usa|china|'
    # Tier classification
    r'tier[- ]?[123]'
    r')\b',
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=8)
def get_geo_pattern(country_codes: tuple = ("IN",)) -> re.Pattern:
    """Build geo specificity regex.

    Returns the hardcoded pattern (covers all practical needs).
    LRU-cached per country_codes tuple — compiled once per config.
    """
    return SPECIFIC_GEO_PATTERN
