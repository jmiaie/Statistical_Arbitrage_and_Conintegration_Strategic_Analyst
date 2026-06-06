"""Longshot-bias scanner.

The favorite-longshot bias is one of the most robust empirical regularities in
betting/prediction markets: longshots (low-priced outcomes) are systematically
*overbet* (win less often than their price implies) and favorites
(high-priced) are *underbet*. So:

  * price >= favorite_threshold  -> back it (buy that outcome)
  * price <= longshot_threshold  -> fade it (buy the opposite outcome)

Edge is thin and slow; this is a volume game. The default bias curve is a
placeholder — calibrate ``bias_at`` to your own settled-market history.
"""

from __future__ import annotations

from .base import Scanner, Opportunity, Leg
from .arbitrage import MarketSnapshot


def default_bias(price: float) -> float:
    """Estimated (true_prob - price). Positive for favorites, negative for
    longshots. A simple monotonic curve; replace with a fitted one."""
    return 0.08 * (price - 0.5)


class LongshotScanner(Scanner):
    kind = "longshot"

    def __init__(self, favorite_threshold: float = 0.65,
                 longshot_threshold: float = 0.20, min_edge: float = 0.01,
                 bias_at=default_bias):
        self.fav = favorite_threshold
        self.long = longshot_threshold
        self.min_edge = min_edge
        self.bias_at = bias_at

    def scan(self, snapshots: list[MarketSnapshot]) -> list[Opportunity]:
        opps = []
        for s in snapshots:
            for i, (oc, price) in enumerate(zip(s.outcomes, s.ask_prices)):
                if not (0 < price < 1):
                    continue
                token = s.token_ids[i] if s.token_ids else None
                if price >= self.fav:
                    bias = self.bias_at(price)
                    # back the favorite: EV per $1 = true*(1/price-1)-(1-true)
                    true = min(0.99, price + bias)
                    edge = true * (1 / price - 1) - (1 - true)
                    side, entry = oc, price
                elif price <= self.long:
                    bias = self.bias_at(price)            # negative
                    # fade the longshot -> buy opposite at (1-price)
                    opp_price = 1 - price
                    true_opp = min(0.99, opp_price - bias)  # opposite gains
                    edge = true_opp * (1 / opp_price - 1) - (1 - true_opp)
                    side, entry = ("No" if oc.lower() == "yes" else "Yes"), opp_price
                else:
                    continue
                if edge < self.min_edge:
                    continue
                opps.append(Opportunity(
                    kind=self.kind,
                    legs=[Leg(s.market_id, s.question, side, round(entry, 4),
                              token)],
                    edge=round(edge, 4),
                    confidence=round(min(0.99, max(side == oc and price or
                                                   (1 - price), 0.5)), 3),
                    rationale=(f"{'Back favorite' if price>=self.fav else 'Fade longshot'} "
                               f"@ {price:.2f} (bias-adjusted edge {edge*100:.1f}%)"),
                    meta={"raw_price": price},
                ))
        return opps
