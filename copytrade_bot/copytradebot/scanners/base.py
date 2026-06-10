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

    def _leg_signal(self, leg: "Leg", source: str | None) -> Signal:
        # The basket-level edge maps to EV/ROI percent so the existing
        # FilterEngine thresholds apply uniformly to alerts and scanner output.
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
            size=None,
        )

    def to_signal(self, source: str | None = None) -> Signal:
        """Convert a single-leg opportunity into a pipeline Signal."""
        if not self.is_single_leg():
            raise ValueError(
                "Multi-leg opportunity: use to_signals() (plural).")
        return self._leg_signal(self.legs[0], source)

    def to_signals(self, source: str | None = None) -> list[Signal]:
        """Convert every leg into a Signal (works for single- or multi-leg).

        Each leg becomes its own Signal carrying the basket-level edge; the
        relative ``Leg.weight`` is preserved separately for sizing by the
        caller (see ``Pipeline.place_opportunity``)."""
        return [self._leg_signal(leg, source) for leg in self.legs]


class Scanner(ABC):
    """Produces opportunities from market data."""

    kind: str = "base"

    @abstractmethod
    def scan(self, *args, **kwargs) -> list[Opportunity]:
        ...
