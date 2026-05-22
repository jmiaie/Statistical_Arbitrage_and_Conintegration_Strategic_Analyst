"""Multi-pair portfolio overlay.

Runs the single-pair backtester on each pair independently, then combines
the net-return streams with equal-risk weighting (inverse trailing vol).
This is the simplest credible portfolio construction — it is sector-
agnostic, deterministic, and rebalances daily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpairs.backtest import BacktestResult, run_backtest
from quantpairs.kalman import KalmanHedge
from quantpairs.signals import ZScoreSignal
from quantpairs.sizing import vol_target

ANNUALISATION = 252


@dataclass(frozen=True)
class PortfolioResult:
    """Aggregated multi-pair portfolio output."""

    pair_results: dict[tuple[str, str], BacktestResult]
    pair_returns: pd.DataFrame
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    kpis: dict[str, float]


def run_portfolio(
    pairs: dict[tuple[str, str], tuple[pd.Series, pd.Series]],
    kalman: KalmanHedge | None = None,
    signal_cfg: ZScoreSignal | None = None,
    cost_bps: float = 2.0,
    target_annual_vol: float = 0.10,
    vol_lookback: int = 60,
) -> PortfolioResult:
    """Backtest each pair, then combine with inverse-vol weights.

    Parameters
    ----------
    pairs
        Mapping `(y_ticker, x_ticker) -> (log_y, log_x)`.
    target_annual_vol
        Portfolio-level annualised vol target.
    """
    if not pairs:
        raise ValueError("pairs must contain at least one entry.")

    pair_results: dict[tuple[str, str], BacktestResult] = {}
    per_pair_returns: dict[str, pd.Series] = {}
    for (y_t, x_t), (log_y, log_x) in pairs.items():
        res = run_backtest(log_y, log_x, kalman=kalman, signal_cfg=signal_cfg, cost_bps=cost_bps)
        pair_results[(y_t, x_t)] = res
        per_pair_returns[f"{y_t}~{x_t}"] = res.pnl["net_return"]

    returns = pd.DataFrame(per_pair_returns).fillna(0)
    # Inverse-vol weights on a rolling window, normalised to sum to 1 per row
    rolling_vol = returns.rolling(vol_lookback).std(ddof=1) * np.sqrt(ANNUALISATION)
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0)
    weights = inv_vol.div(inv_vol.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    raw_portfolio = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    scale = vol_target(raw_portfolio, target_annual_vol=target_annual_vol, lookback=vol_lookback)
    portfolio_returns = (scale * raw_portfolio).rename("portfolio")

    equity = (1 + portfolio_returns).cumprod()
    sd = portfolio_returns.std(ddof=1)
    sharpe = (
        float(portfolio_returns.mean() / sd * np.sqrt(ANNUALISATION))
        if sd and not np.isnan(sd)
        else 0.0
    )
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    avg_corr = returns.corr().where(~np.eye(len(returns.columns), dtype=bool)).stack().mean()
    kpis = {
        "portfolio_sharpe": sharpe,
        "annual_return": float(portfolio_returns.mean() * ANNUALISATION),
        "annual_vol": float(portfolio_returns.std(ddof=1) * np.sqrt(ANNUALISATION)),
        "max_drawdown": max_dd,
        "n_pairs": float(len(pairs)),
        "avg_pair_correlation": float(avg_corr) if not np.isnan(avg_corr) else 0.0,
    }
    return PortfolioResult(
        pair_results=pair_results,
        pair_returns=returns,
        weights=weights,
        portfolio_returns=portfolio_returns,
        equity_curve=equity,
        kpis=kpis,
    )
