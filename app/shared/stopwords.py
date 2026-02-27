"""
Consolidated stopword sets — single source for entity/title filtering.

Used by:
  - app.trends.engine (entity fingerprint stopwords)
  - app.trends.enrichment (generic entities, title stopwords)
"""
from __future__ import annotations

# Entity fingerprint stopwords — tokens to ignore when building entity fingerprints
# for clustering.  Common English function words + generic corporate/geo suffixes.
ENTITY_STOP = frozenset({
    "the", "and", "for", "its", "new", "all", "has", "was", "are", "not",
    "but", "from", "with", "will", "been", "have", "this", "that", "said",
    "over", "more", "than", "also", "into", "amid", "who", "how", "why",
    "may", "can", "now", "per", "via", "set", "get", "key", "top",
    "limited", "ltd", "inc", "corp", "group", "company", "pvt",
    "india", "indian", "global", "world", "national", "market",
})

# Generic entities that appear across most articles in a domain —
# should NOT count toward entity coherence (they don't discriminate).
GENERIC_ENTITIES = frozenset({
    "india", "indian", "us", "china", "uk", "eu", "new delhi", "delhi",
    "mumbai", "bangalore", "bengaluru", "government", "ministry",
    "artificial intelligence", "ai", "market", "markets", "digital",
    "technology", "business", "economy", "global", "world",
    "million", "billion", "crore", "lakh", "percent", "%",
    "fy25", "fy26", "fy2025", "fy2026", "q1", "q2", "q3", "q4",
})

# Title vocabulary stopwords — common words to ignore when comparing article titles.
TITLE_STOP = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "with", "from", "by", "its", "it", "this", "that", "how", "what",
    "why", "who", "will", "can", "may", "could", "would", "should",
    "not", "no", "but", "if", "as", "up", "out", "about", "after",
    "new", "more", "most", "top", "big", "all", "over", "into",
    "says", "said", "set", "get", "gets", "here", "now", "also",
    "amid", "check", "know", "things", "need", "five", "three",
    "india", "indian", "market", "business", "economy",
})
