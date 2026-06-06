"""Scanner interface and the opportunity data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..models import Signal, Side


@dataclass
class Leg:
    """One side of a (possibly multi-leg) opportunity."""

    market_id: str
    question: str
    side: str            # YES | NO
    price: float         # implied probability you pay to enter
    token_id: Optional[str] = None
    weight: float = 1.0  # relative capital weight across legs


@dataclass
class Opportunity:
    kind: str                       # arbitrage | cointegration | longshot | meanreversion
    legs: list[Leg]
    edge: float                     # expected profit per $1 deployed (fraction)
    confidence: float = 0.5         # 0-1 heuristic
    rationale: str = ""
    meta: dict = field(default_factory=dict)

    def is_single_leg(self) -> bool:
        return len(self.legs) == 1

    def to_signal(self, source: str | None = None) -> Signal:
        """Convert a single-leg opportunity into a pipeline Signal.

        ``edge`` (fraction) maps to EV/ROI percent so the existing FilterEngine
        thresholds apply uniformly to alerts and scanner output.
        """
        if not self.is_single_leg():
            raise ValueError("Only single-leg opportunities convert to a Signal.")
        leg = self.legs[0]
        ev_pct = round(self.edge * 100, 2)
        return Signal(
            raw_text=self.rationale,
            source=source or f"scanner:{self.kind}",
            market=leg.question,
            side=Side.from_text(leg.side),
            win_rate=self.confidence,
            ev=ev_pct,
            roi=ev_pct,
            entry_price=leg.price,
        )


class Scanner(ABC):
    """Produces opportunities from market data."""

    kind: str = "base"

    @abstractmethod
    def scan(self, *args, **kwargs) -> list[Opportunity]:
        ...
