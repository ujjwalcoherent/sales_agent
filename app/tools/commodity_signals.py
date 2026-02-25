"""
Commodity price signals from stooq.com (free, no API key required).

Fetches daily commodity futures prices and injects them as numeric hooks
into the synthesis prompt so LLMs use real data not guesses.

Source: https://stooq.com — free financial data, no registration needed.
Tickers: standard futures symbols (gc.f = gold, si.f = silver, etc.)
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

CACHE_FILE = Path("data/commodity_cache.json")
CACHE_TTL_HOURS = 24

# Commodity keyword mapping: which keywords trigger which commodity lookup
KEYWORD_COMMODITY_MAP = {
    "gold": "gold",
    "silver": "silver",
    "steel": "steel_hr",
    "iron": "steel_hr",
    "oil": "crude_oil",
    "crude": "crude_oil",
    "petrol": "crude_oil",
    "petroleum": "crude_oil",
    "copper": "copper",
    "aluminium": "aluminium",
    "aluminum": "aluminium",
    "wheat": "wheat",
    "gas": "natural_gas",
    "natural gas": "natural_gas",
    "coal": "coal",
    "palladium": "palladium",
    "platinum": "platinum",
}

COMMODITIES = {
    "gold":        {"ticker": "gc.f",   "unit": "USD/troy oz",  "label": "Gold"},
    "silver":      {"ticker": "si.f",   "unit": "USD/troy oz",  "label": "Silver"},
    "crude_oil":   {"ticker": "cl.f",   "unit": "USD/bbl",      "label": "Crude Oil (WTI)"},
    "copper":      {"ticker": "hg.f",   "unit": "USD/lb",       "label": "Copper"},
    "wheat":       {"ticker": "zw.f",   "unit": "USD/bushel",   "label": "Wheat"},
    "natural_gas": {"ticker": "ng.f",   "unit": "USD/mmbtu",    "label": "Natural Gas"},
    "palladium":   {"ticker": "pa.f",   "unit": "USD/troy oz",  "label": "Palladium"},
    "platinum":    {"ticker": "pl.f",   "unit": "USD/troy oz",  "label": "Platinum"},
    # Steel/aluminium/coal don't have clean stooq futures — skip, return None
    "steel_hr":    {"ticker": None,     "unit": "USD/MT",       "label": "Steel HR"},
    "aluminium":   {"ticker": None,     "unit": "USD/MT",       "label": "Aluminium"},
    "coal":        {"ticker": None,     "unit": "USD/MT",       "label": "Coal"},
}

_STOOQ_BASE = "https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcv&h&e=json"


class CommoditySignals:
    """Fetch and cache commodity prices from stooq.com."""

    def __init__(self):
        self._cache: Dict = {}
        self._cache_time: Optional[datetime] = None
        self._load_cache()

    def _load_cache(self):
        """Load cached prices from disk."""
        if CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text())
                self._cache = data.get("prices", {})
                ts = data.get("timestamp")
                if ts:
                    self._cache_time = datetime.fromisoformat(ts)
            except Exception:
                pass

    def _save_cache(self):
        """Persist prices to disk."""
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prices": self._cache,
            }, indent=2))
        except Exception as e:
            logger.warning(f"CommoditySignals: cache save failed: {e}")

    def _is_cache_fresh(self) -> bool:
        if not self._cache_time or not self._cache:
            return False
        if self._cache_time.tzinfo is None:
            age = datetime.now(timezone.utc) - self._cache_time.replace(tzinfo=timezone.utc)
        else:
            age = datetime.now(timezone.utc) - self._cache_time
        return age < timedelta(hours=CACHE_TTL_HOURS)

    @staticmethod
    def _parse_stooq_response(text: str) -> Optional[float]:
        """Parse stooq JSON response — handles trailing commas in volume field."""
        # Extract close price directly via regex to avoid JSON parse errors
        # stooq format: {"symbols":[{..."close":5199.71,"volume":}]}
        match = re.search(r'"close"\s*:\s*([\d.]+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    async def _fetch_stooq_price(
        self,
        client: httpx.AsyncClient,
        commodity: str,
        ticker: str,
    ) -> Optional[float]:
        """Fetch latest close price from stooq.com."""
        url = _STOOQ_BASE.format(ticker=ticker)
        try:
            resp = await client.get(
                url,
                timeout=8.0,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code == 200:
                price = self._parse_stooq_response(resp.text)
                if price and price > 0:
                    return price
        except Exception as e:
            logger.debug(f"stooq fetch failed for {commodity} ({ticker}): {e}")
        return None

    async def fetch_latest(self) -> Dict:
        """Fetch all commodity prices. Returns cached data if fresh."""
        if self._is_cache_fresh():
            return self._cache

        logger.info("CommoditySignals: fetching latest prices from stooq.com...")
        prices = {}

        # Only fetch commodities that have a valid ticker
        fetchable = {
            commodity: info
            for commodity, info in COMMODITIES.items()
            if info["ticker"] is not None
        }

        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = {
                commodity: self._fetch_stooq_price(client, commodity, info["ticker"])
                for commodity, info in fetchable.items()
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for commodity, result in zip(tasks.keys(), results):
                if isinstance(result, float) and result > 0:
                    prices[commodity] = {
                        "price": round(result, 2),
                        "unit": COMMODITIES[commodity]["unit"],
                        "label": COMMODITIES[commodity]["label"],
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }

        if prices:
            self._cache = prices
            self._cache_time = datetime.now(timezone.utc)
            self._save_cache()
            logger.info(f"CommoditySignals: fetched {len(prices)} commodities")
        else:
            logger.warning("CommoditySignals: no prices fetched, using stale cache")

        return self._cache

    def get_hooks_for_keywords(self, keywords: list) -> str:
        """Return commodity price hooks relevant to given keywords.

        Returns formatted string like:
        "Gold: $2,950.00/troy oz | Silver: $33.50/troy oz"
        or empty string if no relevant commodities found in cache.
        """
        if not self._cache:
            return ""

        relevant = set()
        kw_lower = " ".join(str(k).lower() for k in keywords)
        for trigger, commodity in KEYWORD_COMMODITY_MAP.items():
            if trigger in kw_lower and commodity in self._cache:
                relevant.add(commodity)

        if not relevant:
            return ""

        parts = []
        for commodity in sorted(relevant):
            data = self._cache.get(commodity, {})
            if data and data.get("price"):
                label = data.get("label", commodity.replace("_", " ").title())
                price = data["price"]
                unit = data["unit"]
                unit_short = unit.split("/")[-1]
                parts.append(f"{label}: ${price:,.2f}/{unit_short}")

        return " | ".join(parts)


# Module-level singleton — lazy initialized
_signals_instance: Optional[CommoditySignals] = None


def get_commodity_signals() -> CommoditySignals:
    global _signals_instance
    if _signals_instance is None:
        _signals_instance = CommoditySignals()
    return _signals_instance
