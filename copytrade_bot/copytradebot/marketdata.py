"""Market-data providers — real Polymarket data plus a static test double.

The provider is the only component that talks to the network, so the rest of
the engine (sizing, slippage, settlement) is fully unit-testable against
``StaticData``. ``PolymarketData`` uses Polymarket's *public* endpoints
(Gamma + CLOB) and needs no API keys — only outbound network access.

Endpoint shapes are coded to Polymarket's documented responses but cannot be
verified offline; treat ``PolymarketData`` as best-effort and confirm against
live responses before trusting size. Override the field mapping in one place
(``_market_ref_from_gamma``) if the API shifts.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class MarketRef:
    market_id: str
    question: str
    outcomes: list[str]                 # e.g. ["Yes", "No"]
    token_ids: list[str]                # aligned with ``outcomes``
    closed: bool = False
    # Resolution prices aligned with ``outcomes`` (1.0 = winner) once resolved.
    outcome_prices: list[float] = field(default_factory=list)

    def token_for_side(self, side: str) -> tuple[str, int]:
        """Map a parsed side onto (token_id, outcome_index)."""
        want = "yes" if side.upper() in {"YES", "BUY", "LONG"} else "no"
        for i, o in enumerate(self.outcomes):
            if str(o).strip().lower() == want:
                return self.token_ids[i], i
        # Fall back to first outcome if names don't match Yes/No.
        return self.token_ids[0], 0

    def winning_index(self) -> Optional[int]:
        if not self.closed or not self.outcome_prices:
            return None
        # The resolved winner has price ~1.0.
        best = max(range(len(self.outcome_prices)),
                   key=lambda i: self.outcome_prices[i])
        return best if self.outcome_prices[best] >= 0.99 else None


class MarketDataProvider(Protocol):
    def find_market(self, query: str) -> Optional[MarketRef]: ...
    def get_market(self, market_id: str) -> Optional[MarketRef]: ...
    def get_book(self, token_id: str) -> list[tuple[float, float]]: ...
    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]: ...


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #
class StaticData:
    """In-memory provider for tests and offline runs.

    ``markets``: list[MarketRef]; ``books``: {token_id: ascending asks};
    ``prices``: {token_id: price}.
    """

    def __init__(self, markets=None, books=None, prices=None):
        self.markets = {m.market_id: m for m in (markets or [])}
        self.books = books or {}
        self.prices = prices or {}

    def find_market(self, query: str) -> Optional[MarketRef]:
        ql = (query or "").lower()
        best, best_score = None, 0
        for m in self.markets.values():
            score = sum(1 for w in re.findall(r"\w+", ql)
                        if w in m.question.lower())
            if score > best_score:
                best, best_score = m, score
        return best or next(iter(self.markets.values()), None)

    def get_market(self, market_id: str) -> Optional[MarketRef]:
        return self.markets.get(market_id)

    def get_book(self, token_id: str) -> list[tuple[float, float]]:
        return list(self.books.get(token_id, []))

    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        return self.prices.get(token_id)


# --------------------------------------------------------------------------- #
# Live Polymarket data (public endpoints, no auth)
# --------------------------------------------------------------------------- #
class PolymarketData:
    GAMMA = "https://gamma-api.polymarket.com"
    CLOB = "https://clob.polymarket.com"

    def __init__(self, user_agent: str = "copytrade-bot/0.1",
                 gamma: str | None = None, clob: str | None = None,
                 timeout: int = 15):
        self.ua = user_agent
        self.gamma = gamma or self.GAMMA
        self.clob = clob or self.CLOB
        self.timeout = timeout

    def _get(self, url: str):
        req = urllib.request.Request(url, headers={
            "User-Agent": self.ua, "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    @staticmethod
    def _as_list(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v or []

    def _market_ref_from_gamma(self, m: dict) -> Optional[MarketRef]:
        token_ids = [str(t) for t in self._as_list(m.get("clobTokenIds"))]
        outcomes = [str(o) for o in self._as_list(m.get("outcomes"))]
        if not token_ids or not outcomes:
            return None
        prices = []
        for p in self._as_list(m.get("outcomePrices")):
            try:
                prices.append(float(p))
            except (TypeError, ValueError):
                prices.append(0.0)
        return MarketRef(
            market_id=str(m.get("id") or m.get("conditionId") or ""),
            question=str(m.get("question") or m.get("title") or ""),
            outcomes=outcomes,
            token_ids=token_ids,
            closed=bool(m.get("closed")),
            outcome_prices=prices,
        )

    def find_market(self, query: str) -> Optional[MarketRef]:
        if not query:
            return None
        params = urllib.parse.urlencode({
            "search": query[:120], "active": "true", "closed": "false",
            "limit": 20,
        })
        data = self._get(f"{self.gamma}/markets?{params}")
        markets = data if isinstance(data, list) else data.get("data", [])
        refs = [r for r in (self._market_ref_from_gamma(m) for m in markets) if r]
        if not refs:
            return None
        # Rank by word overlap with the query.
        qwords = set(re.findall(r"\w+", query.lower()))
        return max(refs, key=lambda r: len(
            qwords & set(re.findall(r"\w+", r.question.lower()))))

    def get_market(self, market_id: str) -> Optional[MarketRef]:
        data = self._get(f"{self.gamma}/markets/{market_id}")
        m = data[0] if isinstance(data, list) else data
        return self._market_ref_from_gamma(m) if m else None

    def get_book(self, token_id: str) -> list[tuple[float, float]]:
        data = self._get(f"{self.clob}/book?token_id={token_id}")
        asks = data.get("asks", []) if isinstance(data, dict) else []
        out = []
        for lvl in asks:
            try:
                out.append((float(lvl["price"]), float(lvl["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(out, key=lambda x: x[0])

    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        data = self._get(f"{self.clob}/price?token_id={token_id}&side={side}")
        try:
            return float(data["price"])
        except (KeyError, TypeError, ValueError):
            return None


def build_provider(execution_cfg, settings) -> Optional[MarketDataProvider]:
    """Construct the configured provider (or ``None`` to disable live data)."""
    source = (getattr(execution_cfg, "data_source", "none") or "none").lower()
    if source == "polymarket":
        return PolymarketData(
            user_agent=getattr(execution_cfg, "user_agent", "copytrade-bot/0.1"),
            clob=getattr(settings, "polymarket_host", None) or PolymarketData.CLOB,
        )
    return None
