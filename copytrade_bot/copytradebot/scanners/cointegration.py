"""Cointegration / statistical-arbitrage scanner — the repo's theme on
prediction markets.

For each pair of related markets it runs the Engle-Granger two step:

  1. Regress one price series on the other (OLS) to get the hedge ratio and
     the residual spread.
  2. Test the spread for stationarity (Dickey-Fuller). If stationary, the pair
     is cointegrated and the spread mean-reverts.

When the current spread's z-score is extreme we trade convergence: go long the
cheap leg and short the rich leg (on Polymarket, "short" = buy the opposite
YES/NO token). Exit when the spread reverts toward zero.
"""

from __future__ import annotations

from itertools import combinations

from .base import Scanner, Opportunity, Leg
from . import stats


class CointegrationScanner(Scanner):
    kind = "cointegration"

    def __init__(self, lookback: int = 60, entry_z: float = 2.0,
                 adf_threshold: float = -3.0, min_half_life: float = 1.0,
                 max_half_life: float = 40.0):
        self.lookback = lookback
        self.entry_z = entry_z
        self.adf_threshold = adf_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def scan(self, series: dict) -> list[Opportunity]:
        """``series``: {market_id: {"question","prices","token_yes","token_no"}}.
        All price lists must be aligned/equal length."""
        opps = []
        ids = list(series)
        for a_id, b_id in combinations(ids, 2):
            A, B = series[a_id], series[b_id]
            ya = A["prices"][-self.lookback:]
            yb = B["prices"][-self.lookback:]
            if len(ya) < self.lookback or len(yb) < self.lookback:
                continue
            # Step 1: hedge ratio + spread.
            alpha, beta, resid = stats.ols(yb, ya)  # ya = alpha + beta*yb
            if beta <= 0:
                continue
            # Step 2: stationarity of the spread.
            t = stats.adf_tstat(resid)
            if t > self.adf_threshold:
                continue
            hl = stats.half_life(resid)
            if not (self.min_half_life <= hl <= self.max_half_life):
                continue
            z = stats.zscore_last(resid)
            if abs(z) < self.entry_z:
                continue

            # z>0 => A rich vs B => short A (buy NO_A), long B (buy YES_B).
            if z > 0:
                legs = [
                    Leg(a_id, A["question"], "No", round(1 - ya[-1], 4),
                        A.get("token_no"), weight=1.0),
                    Leg(b_id, B["question"], "Yes", round(yb[-1], 4),
                        B.get("token_yes"), weight=beta),
                ]
            else:
                legs = [
                    Leg(a_id, A["question"], "Yes", round(ya[-1], 4),
                        A.get("token_yes"), weight=1.0),
                    Leg(b_id, B["question"], "No", round(1 - yb[-1], 4),
                        B.get("token_no"), weight=beta),
                ]
            # Edge proxy: expected reversion of the spread to its mean, scaled.
            edge = min(0.5, abs(z) * stats.std(resid))
            opps.append(Opportunity(
                kind=self.kind, legs=legs, edge=round(edge, 4),
                confidence=round(min(0.9, 0.5 + abs(z) / 20), 3),
                rationale=(f"Cointegrated (ADF t={t:.2f}, half-life={hl:.1f}); "
                           f"spread z={z:+.1f}, trade convergence."),
                meta={"beta": round(beta, 3), "adf_t": round(t, 2),
                      "half_life": round(hl, 1), "zscore": round(z, 2)},
            ))
        return opps
