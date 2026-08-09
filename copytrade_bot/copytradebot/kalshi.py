"""Kalshi market-data provider.

Kalshi (https://kalshi.com) is a CFTC-regulated prediction market platform
similar to Polymarket but with its own market universe and pricing dynamics.
This provider fetches active markets and order-book data via Kalshi's public API.

Key differences from Polymarket:
  * Binary outcomes: YES/NO (like Polymarket)
  * Decimal prices: 0.00–1.00 (like Polymarket)
  * Settlement: on-chain (like Polymarket)
  * API: RESTful (https://docs.kalshi.com), no authentication for public data
  * Venue: US-regulated, smaller market cap, tighter spreads in some categories
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("copytrade.kalshi")


@dataclass
class KalshiMarketRef:
    """Kalshi market reference for order/settlement routing."""

    market_id: str
    title: str  # market question/title
    category: str  # e.g. "Politics", "Sports", "Economics"
    yes_price: float  # best ask to buy YES (current ask level)
    no_price: float  # best ask to buy NO
    volume_24h: float = 0.0
    closed: bool = False
    outcome_prices: list[float] = field(default_factory=list)  # [yes_price, no_price] post-resolution

    def token_for_side(self, side: str) -> tuple[str, int]:
        """Map a parsed side onto (token_id, outcome_index).

        On Kalshi, outcomes are YES/NO. We use the market_id + side as the token_id.
        """
        want = "yes" if side.upper() in {"YES", "BUY", "LONG"} else "no"
        idx = 0 if want == "yes" else 1
        token_id = f"{self.market_id}-{want}"
        return token_id, idx

    def winning_index(self) -> Optional[int]:
        if not self.closed or not self.outcome_prices:
            return None
        best = max(
            range(len(self.outcome_prices)),
            key=lambda i: self.outcome_prices[i],
        )
        return best if self.outcome_prices[best] >= 0.99 else None


class KalshiData:
    """Kalshi public market-data provider (no auth required)."""

    API_BASE = "https://api.kalshi.com/trade-api/v2"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session = urllib.request.Request  # reuse requests pattern

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        """GET from Kalshi API."""
        url = f"{self.API_BASE}{path}"
        if params:
            url += f"?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "copytrade-bot/0.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            log.error(f"Kalshi API error on {path}: {e}")
            return {}

    def list_markets(
        self,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KalshiMarketRef]:
        """List active Kalshi markets, optionally filtered by category.

        Args:
            category: e.g. 'politics', 'sports', 'economics'. None = all.
            limit: max results (Kalshi caps at 100).
            offset: pagination.

        Returns:
            List of KalshiMarketRef.
        """
        params = {"limit": limit, "offset": offset, "status": "open"}
        if category:
            params["category"] = category
        data = self._get("/markets", params)
        markets = data.get("markets", []) if isinstance(data, dict) else []
        refs = []
        for m in markets:
            try:
                ref = KalshiMarketRef(
                    market_id=m.get("id", ""),
                    title=m.get("title", ""),
                    category=m.get("category", ""),
                    yes_price=float(m.get("yes_price", 0.5)),
                    no_price=float(m.get("no_price", 0.5)),
                    volume_24h=float(m.get("volume_24h", 0.0)),
                    closed=m.get("status") == "closed",
                )
                refs.append(ref)
            except (KeyError, TypeError, ValueError) as e:
                log.warning(f"Skipped malformed Kalshi market: {e}")
        return refs

    def find_market(self, query: str) -> Optional[KalshiMarketRef]:
        """Find a single Kalshi market by title/keyword search.

        Ranks by word overlap with the query (similar to Polymarket provider).
        """
        if not query:
            return None
        markets = self.list_markets(limit=20)
        if not markets:
            return None
        qwords = set(re.findall(r"\w+", query.lower()))
        scored = [
            (len(qwords & set(re.findall(r"\w+", m.title.lower()))), m)
            for m in markets
        ]
        best_score, best = max(scored, key=lambda s: s[0], default=(0, None))
        return best if best_score > 0 else None

    def get_market(self, market_id: str) -> Optional[KalshiMarketRef]:
        """Fetch a specific Kalshi market by ID."""
        data = self._get(f"/markets/{market_id}")
        if not isinstance(data, dict) or "id" not in data:
            return None
        try:
            return KalshiMarketRef(
                market_id=data.get("id", ""),
                title=data.get("title", ""),
                category=data.get("category", ""),
                yes_price=float(data.get("yes_price", 0.5)),
                no_price=float(data.get("no_price", 0.5)),
                volume_24h=float(data.get("volume_24h", 0.0)),
                closed=data.get("status") == "closed",
            )
        except (KeyError, TypeError, ValueError) as e:
            log.error(f"Failed to parse Kalshi market {market_id}: {e}")
            return None

    def get_orderbook(
        self, market_id: str
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Fetch the order book (asks, bids) for a market.

        Returns:
            (yes_asks, no_asks) where each is [(price, size), ...] sorted ascending by price.
        """
        data = self._get(f"/markets/{market_id}/orderbook")
        if not isinstance(data, dict):
            return [], []
        yes_asks = [
            (float(o["price"]), float(o["size"]))
            for o in data.get("yes_asks", [])
            if o.get("price") and o.get("size")
        ]
        no_asks = [
            (float(o["price"]), float(o["size"]))
            for o in data.get("no_asks", [])
            if o.get("price") and o.get("size")
        ]
        return sorted(yes_asks, key=lambda x: x[0]), sorted(no_asks, key=lambda x: x[0])
