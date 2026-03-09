"""Mock data loader for demo mode.

Live API calls auto-save results to data/mock/*.json.
When mock_mode=True, endpoints return saved data instead of calling external APIs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MOCK_DIR = Path("./data/mock")


def load_mock(category: str) -> List[Dict[str, Any]]:
    """Load mock data for a category (company_search, lead_gen, news_articles)."""
    mock_file = MOCK_DIR / f"{category}.json"
    if not mock_file.exists():
        return []
    try:
        data = json.loads(mock_file.read_text())
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to load mock {category}: {e}")
        return []


def find_mock_search(query: str) -> Optional[Dict[str, Any]]:
    """Find a saved company search result matching the query."""
    results = load_mock("company_search")
    # Exact match
    for r in results:
        if r.get("query", "").lower() == query.lower():
            return r.get("response")
    # Partial match
    for r in results:
        if query.lower() in r.get("query", "").lower():
            return r.get("response")
    # Most recent
    return results[-1].get("response") if results else None


def find_mock_leads(company_name: str) -> Optional[Dict[str, Any]]:
    """Find saved lead gen result for a company."""
    results = load_mock("lead_gen")
    for r in results:
        if r.get("company_name", "").lower() == company_name.lower():
            return r.get("response")
    return results[-1].get("response") if results else None


def save_news_mock(articles: list, total: int) -> None:
    """Save a snapshot of news articles for mock mode."""
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    mock_file = MOCK_DIR / "news_articles.json"
    data = {"articles": articles[:100], "total": min(total, 100)}
    try:
        mock_file.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Saved {len(articles[:100])} news articles to mock")
    except Exception as e:
        logger.debug(f"News mock save failed: {e}")
