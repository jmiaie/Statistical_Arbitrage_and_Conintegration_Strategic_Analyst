"""Transaction-cost models. Square-root impact + linear half-spread."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SquareRootImpact:
    """Square-root market-impact model (Almgren-style).

    cost_bps = half_spread_bps + kappa * sigma_daily_bps * sqrt(Q / ADV)

    `kappa` of ~0.5 is consistent with published equity TCA studies.
    """

    half_spread_bps: float = 1.0
    kappa: float = 0.5

    def cost_bps(
        self,
        traded_notional: pd.Series,
        adv_notional: pd.Series,
        sigma_daily: pd.Series,
    ) -> pd.Series:
        participation = (traded_notional.abs() / adv_notional.replace(0, np.nan)).fillna(0)
        impact = self.kappa * sigma_daily * 1e4 * np.sqrt(participation)
        return self.half_spread_bps + impact


def apply_costs(
    positions: pd.Series,
    spread_returns: pd.Series,
    cost_bps: pd.Series | float,
) -> pd.DataFrame:
    """Convert positions + spread returns into gross/net P&L series.

    `positions` is the desired position at the *close* of each bar; we trade the
    difference at the next bar's open and pay costs on the absolute change.
    """
    pos = positions.astype(float)
    turnover = pos.diff().abs().fillna(pos.abs().iloc[0])
    gross = pos.shift(1).fillna(0) * spread_returns
    if isinstance(cost_bps, (int, float)):
        cost = turnover * (float(cost_bps) / 1e4)
    else:
        cost = turnover * (cost_bps.reindex(turnover.index).fillna(0) / 1e4)
    net = gross - cost
    return pd.DataFrame(
        {"gross_return": gross, "cost": cost, "net_return": net, "turnover": turnover}
    )
