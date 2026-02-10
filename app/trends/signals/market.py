"""
Market context signal computation for news trend analysis.

Measures how a trend relates to market context, financial indicators,
and regulatory environment. The LLM handles sector classification —
these signals focus on quantitative measurements the LLM can't do.

SIGNALS:
  regulatory_flag:     Uses embedding-based event classifier (no hardcoded keywords).
  company_density:     Company mentions per article (from NER, no keywords).
  financial_indicator: Concrete financial numbers (NER MONEY entities + regex for currencies).
  financial_amounts:   Extracted dollar/rupee amounts.

WHY THESE MATTER FOR SALES:
  Regulatory changes create compliance urgency → "You need to adapt NOW."
  Financial indicators add concreteness → "$200M invested" is better than "growing."
  Company density enables targeted outreach → name specific companies.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def compute_market_signals(articles: list) -> Dict[str, Any]:
    """
    Compute market context signals from a list of articles.

    No hardcoded keyword lists. Uses:
    - Event classifier output for regulatory detection (embedding-based)
    - NER entities for company density
    - NER MONEY entities + currency regex for financial indicators
    """
    if not articles:
        return _empty_signals()

    return {
        "regulatory_flag": _check_regulatory(articles),
        "company_density": _compute_company_density(articles),
        "financial_indicator": _check_financial_indicator(articles),
        "financial_amounts": _extract_financial_amounts(articles),
    }


def _check_regulatory(articles: list) -> bool:
    """
    Is this a regulatory/policy trend?

    Uses the embedding-based event classifier output — no hardcoded keywords.
    The event classifier already understands "regulation" semantically.
    """
    for article in articles:
        # Check event classifier output (embedding-based, set in Phase 0.5)
        event = getattr(article, '_trigger_event', '')
        if event in ('regulation', 'crisis'):
            return True

        # Check trend type classification
        trend_types = getattr(article, 'detected_trend_types', [])
        for tt in trend_types:
            if str(tt).lower() in ('regulation', 'policy'):
                return True

    return False


def _compute_company_density(articles: list) -> float:
    """
    Company mentions per article (from NER, no hardcoded lists).

    Interpretation:
      <0.5: No specific companies (generic news)
      0.5-2: Some companies mentioned
      >2: Company-rich (M&A, earnings, partnerships)
    """
    if not articles:
        return 0.0

    total_companies = sum(
        len(getattr(a, 'mentioned_companies', []))
        for a in articles
    )
    return total_companies / len(articles)


def _check_financial_indicator(articles: list) -> bool:
    """
    Does the trend contain concrete financial numbers?

    Checks:
    1. NER MONEY entities (model-detected, no keywords)
    2. Currency regex patterns (₹, $, Rs — appropriate for data extraction)
    """
    money_pattern = re.compile(
        r'(?:rs\.?|inr|₹|\$|usd)\s*[\d,.]+\s*(?:crore|lakh|million|billion|mn|bn|cr|lk)?',
        re.IGNORECASE
    )

    for article in articles:
        # Check NER entities for MONEY type
        entities = getattr(article, 'entities', [])
        for ent in entities:
            if getattr(ent, 'type', '') == 'MONEY':
                return True

        # Check mentioned_amounts
        amounts = getattr(article, 'mentioned_amounts', [])
        if amounts:
            return True

        # Regex fallback on text (currency pattern extraction, not keyword matching)
        title = getattr(article, 'title', '') or ''
        summary = getattr(article, 'summary', '') or ''
        if money_pattern.search(f"{title} {summary}"):
            return True

    return False


def _extract_financial_amounts(articles: list) -> List[str]:
    """Extract unique financial amounts mentioned across articles."""
    amounts = set()

    money_pattern = re.compile(
        r'(?:Rs\.?\s*|INR\s*|₹\s*|\$\s*|USD\s*)[\d,.]+\s*(?:crore|lakh|million|billion|mn|bn|cr|lk)?',
        re.IGNORECASE
    )

    for article in articles:
        # From NER entities
        entities = getattr(article, 'entities', [])
        for ent in entities:
            if getattr(ent, 'type', '') == 'MONEY':
                amounts.add(getattr(ent, 'text', '').strip())

        # From regex (currency pattern extraction)
        title = getattr(article, 'title', '') or ''
        summary = getattr(article, 'summary', '') or ''
        for match in money_pattern.findall(f"{title} {summary}"):
            amounts.add(match.strip())

    return sorted(amounts)[:10]


def _empty_signals() -> Dict[str, Any]:
    """Return zero signals when no articles are available."""
    return {
        "regulatory_flag": False,
        "company_density": 0.0,
        "financial_indicator": False,
        "financial_amounts": [],
    }
