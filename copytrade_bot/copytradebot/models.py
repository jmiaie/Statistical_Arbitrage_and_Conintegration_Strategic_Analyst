"""Core data structures shared across the pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    """Whether an alert opens a new position or closes/updates an existing one.

    Defaults to ENTRY: only explicit close/exit phrasing flips it, because
    wrongly opening on an exit message (the old behaviour) and wrongly closing
    on an entry message are both harmful — ENTRY preserves intent for the
    common case and exits are detected conservatively."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"


class Side(str, Enum):
    """Direction of a trade. Kept broad to cover equities, crypto and
    prediction markets (Polymarket YES/NO)."""

    BUY = "BUY"
    SELL = "SELL"
    YES = "YES"
    NO = "NO"
    LONG = "LONG"
    SHORT = "SHORT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_text(cls, text: Optional[str]) -> "Side":
        if not text:
            return cls.UNKNOWN
        t = text.strip().upper()
        mapping = {
            "BUY": cls.BUY, "LONG": cls.LONG, "BULL": cls.LONG, "CALL": cls.BUY,
            "OVER": cls.YES, "YES": cls.YES,
            "SELL": cls.SELL, "SHORT": cls.SHORT, "BEAR": cls.SHORT, "PUT": cls.SELL,
            "UNDER": cls.NO, "NO": cls.NO,
        }
        return mapping.get(t, cls.UNKNOWN)


@dataclass
class Signal:
    """A parsed trade alert.

    Numeric conventions (so filters and the alert speak the same language):
      * ``win_rate``         -> fraction in [0, 1]   (65% -> 0.65)
      * ``ev``               -> percent of stake     (e.g. 12.0 means +12%)
      * ``roi``              -> percent              (e.g. 30.0 means +30%)
      * ``expected_return``  -> percent
      * ``entry_price``      -> raw price/probability as written
      * ``size``             -> the alert's *suggested* stake (currency units)
    """

    raw_text: str
    source: str = "unknown"
    received_at: float = field(default_factory=time.time)

    intent: Intent = Intent.ENTRY
    market: Optional[str] = None
    side: Side = Side.UNKNOWN
    win_rate: Optional[float] = None
    ev: Optional[float] = None
    roi: Optional[float] = None
    expected_return: Optional[float] = None
    entry_price: Optional[float] = None
    size: Optional[float] = None

    def present_fields(self) -> set[str]:
        """Which numeric/market fields were successfully extracted."""
        fields = {
            "market", "win_rate", "ev", "roi", "expected_return",
            "entry_price", "size",
        }
        out = {f for f in fields if getattr(self, f) is not None}
        if self.side is not Side.UNKNOWN:
            out.add("side")
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["intent"] = self.intent.value
        return d


@dataclass
class FilterResult:
    """Outcome of evaluating a Signal against the active FilterConfig."""

    passed: bool
    reasons: list[str] = field(default_factory=list)

    def add(self, ok: bool, reason: str) -> None:
        if not ok:
            self.passed = False
            self.reasons.append(reason)


@dataclass
class Position:
    """An open or settled position recorded by an executor."""

    signal_id: Optional[int]
    market: Optional[str]
    side: str
    entry_price: Optional[float]
    stake: float
    venue: str = "paper"
    status: str = "open"        # open | won | lost | closed
    pnl: float = 0.0
    external_id: Optional[str] = None
    opened_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    # Realistic-fill bookkeeping. ``entry_price``/``stake`` above hold the
    # *effective* fill price and total cash deployed (incl. fees); ``shares``
    # is the contract/share count acquired, and ``meta`` carries provider
    # references (token/market ids) plus slippage attribution.
    shares: float = 0.0
    meta: dict = field(default_factory=dict)
