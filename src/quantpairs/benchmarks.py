"""Naïve benchmarks to demonstrate Kalman's lift over static OLS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from quantpairs.backtest import BacktestResult, _hit_rate, _max_drawdown, _sharpe
from quantpairs.signals import ZScoreSignal, generate_signals
from quantpairs.tca import apply_costs

ANNUALISATION = 252


def static_ols_backtest(
    log_y: pd.Series,
    log_x: pd.Series,
    signal_cfg: ZScoreSignal | None = None,
    cost_bps: float = 2.0,
    warmup: int = 60,
    zscore_window: int = 60,
) -> dict[str, float]:
    """OLS hedge ratio fit once at t=0; same z-score + threshold logic.

    This is the obvious naive baseline. Comparing its KPIs to the Kalman
    backtest is the cleanest way to show that adaptive β earns its keep.
    """
    df = pd.concat([log_y.rename("y"), log_x.rename("x")], axis=1).dropna()
    beta = float(sm.OLS(df["y"], sm.add_constant(df["x"])).fit().params.iloc[1])
    alpha = float(df["y"].iloc[0] - beta * df["x"].iloc[0])
    spread = df["y"] - beta * df["x"] - alpha

    rolling_mean = spread.rolling(zscore_window).mean()
    rolling_std = spread.rolling(zscore_window).std(ddof=1).replace(0, np.nan)
    z = (spread - rolling_mean) / rolling_std
    z.iloc[:warmup] = np.nan
    positions = generate_signals(z, signal_cfg)

    pnl = apply_costs(positions, spread.diff().fillna(0), cost_bps=cost_bps)
    equity = (1 + pnl["net_return"]).cumprod()
    net = pnl["net_return"]
    return {
        "sharpe_net": _sharpe(net),
        "annual_return": float(net.mean() * ANNUALISATION),
        "annual_vol": float(net.std(ddof=1) * np.sqrt(ANNUALISATION)),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": _hit_rate(net),
        "n_trades": float((pnl["turnover"] > 0).sum()),
        "static_beta": beta,
    }


def compare_to_kalman(kalman: BacktestResult, ols: dict[str, float]) -> pd.DataFrame:
    """Side-by-side KPI comparison."""
    keys = ["sharpe_net", "annual_return", "annual_vol", "max_drawdown", "hit_rate", "n_trades"]
    rows = []
    for k in keys:
        rows.append(
            {
                "Metric": k,
                "Kalman": kalman.kpis[k],
                "Static OLS": ols[k],
                "Lift": kalman.kpis[k] - ols[k],
            }
        )
    return pd.DataFrame(rows)
