"""Mean-reversion scanner.

Fade short-term price overshoots: when a market's price spikes far above its
recent rolling mean (high positive z-score) we expect partial reversion, so we
take the cheaper opposite side; symmetrically for sharp drops.

The danger is trading *genuine news* (a permanent re-rating, not noise). Two
guards: require enough history, and skip moves so large/monotonic they look
like a regime change rather than an overshoot.
"""

from __future__ import annotations

from .base import Scanner, Opportunity, Leg
from . import stats


class MeanReversionScanner(Scanner):
    kind = "meanreversion"

    def __init__(self, lookback: int = 20, entry_z: float = 2.0,
                 max_move: float = 0.25, min_edge: float = 0.01):
        self.lookback = lookback
        self.entry_z = entry_z
        self.max_move = max_move      # skip if recent move exceeds this (news)
        self.min_edge = min_edge

    def scan(self, histories: dict) -> list[Opportunity]:
        """``histories``: {market_id: (question, [price, ...], token_yes,
        token_no)}. Prices are the YES implied probability over time."""
        opps = []
        for market_id, payload in histories.items():
            question, prices, tok_yes, tok_no = self._unpack(payload)
            if len(prices) < self.lookback + 1:
                continue
            window = prices[-self.lookback:]
            z = stats.zscore_last(window)
            move = abs(prices[-1] - stats.mean(window[:-1]))
            if move > self.max_move:
                continue  # looks like news, not an overshoot
            mu = stats.mean(window[:-1])
            if z >= self.entry_z:
                # YES overshot up -> fade by buying NO (cheap now)
                side, token = "No", tok_no
                entry = 1 - prices[-1]
                target = 1 - mu
            elif z <= -self.entry_z:
                # YES overshot down -> buy YES cheap, expect reversion up
                side, token = "Yes", tok_yes
                entry = prices[-1]
                target = mu
            else:
                continue
            if entry <= 0 or entry >= 1:
                continue
            edge = (target - entry) / entry  # expected fractional gain to mean
            if edge < self.min_edge:
                continue
            opps.append(Opportunity(
                kind=self.kind,
                legs=[Leg(market_id, question, side, round(entry, 4), token)],
                edge=round(edge, 4),
                confidence=round(min(0.9, 0.5 + abs(z) / 20), 3),
                rationale=(f"z={z:+.1f} vs {self.lookback}-pt mean; fade "
                           f"overshoot ({side}) targeting {target:.2f}"),
                meta={"zscore": round(z, 2), "half_life":
                      round(stats.half_life(window), 1)},
            ))
        return opps

    @staticmethod
    def _unpack(payload):
        if isinstance(payload, dict):
            return (payload.get("question", ""), payload["prices"],
                    payload.get("token_yes"), payload.get("token_no"))
        return payload  # (question, prices, tok_yes, tok_no)
