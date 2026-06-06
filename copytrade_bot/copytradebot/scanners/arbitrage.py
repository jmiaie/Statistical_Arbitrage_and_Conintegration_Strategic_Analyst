"""Arbitrage scanner — the most defensible, near-riskless edge.

If the prices to buy *every* outcome of a market sum to less than $1 (after
costs), buying the complete set locks in a profit: exactly one outcome pays
$1 at resolution. Works for binary (YES+NO) and multi-outcome markets.

The edge is mechanical, not predictive — but it is capacity-limited and
competitive, so realistic slippage/fee buffers matter a lot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Scanner, Opportunity, Leg


@dataclass
class MarketSnapshot:
    market_id: str
    question: str
    outcomes: list[str]          # e.g. ["Yes", "No"] or candidate names
    ask_prices: list[float]      # best ask to BUY each outcome
    token_ids: list[str] | None = None


class ArbitrageScanner(Scanner):
    kind = "arbitrage"

    def __init__(self, cost_buffer: float = 0.01, min_edge: float = 0.005):
        # cost_buffer ~ slippage+fees reserved; min_edge to act on.
        self.cost_buffer = cost_buffer
        self.min_edge = min_edge

    def scan(self, snapshots: list[MarketSnapshot]) -> list[Opportunity]:
        opps = []
        for s in snapshots:
            if not s.ask_prices or any(p <= 0 for p in s.ask_prices):
                continue
            total = sum(s.ask_prices)
            edge = 1.0 - total - self.cost_buffer
            if edge < self.min_edge:
                continue
            legs = []
            for i, (oc, price) in enumerate(zip(s.outcomes, s.ask_prices)):
                token = s.token_ids[i] if s.token_ids else None
                # Buy each outcome with weight proportional to its price so the
                # payout ($1 * shares) is equalised across outcomes.
                legs.append(Leg(market_id=s.market_id, question=s.question,
                                side=oc, price=price, token_id=token,
                                weight=price))
            opps.append(Opportunity(
                kind=self.kind, legs=legs, edge=round(edge, 4),
                confidence=0.95,
                rationale=(f"Outcome prices sum to {total:.3f} (<1). "
                           f"Buy all for ~{edge*100:.1f}% locked edge."),
                meta={"sum_prices": round(total, 4)},
            ))
        return opps
