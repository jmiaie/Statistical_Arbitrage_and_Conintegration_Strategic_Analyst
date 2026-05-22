"""End-to-end pair backtester wiring Kalman -> signals -> TCA -> KPIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpairs.kalman import KalmanHedge, KalmanState
from quantpairs.signals import ZScoreSignal, generate_signals
from quantpairs.tca import apply_costs

ANNUALISATION = 252


@dataclass(frozen=True)
class BacktestResult:
    """Container for a single backtest run."""

    state: KalmanState
    positions: pd.Series
    pnl: pd.DataFrame
    equity_curve: pd.Series
    kpis: dict[str, float]

    def summary(self) -> str:
        lines = [f"{k:>22}: {v:>10.4f}" for k, v in self.kpis.items()]
        return "\n".join(lines)


def _sharpe(returns: pd.Series) -> float:
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * np.sqrt(ANNUALISATION))


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def _hit_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def run_backtest(
    log_y: pd.Series,
    log_x: pd.Series,
    kalman: KalmanHedge | None = None,
    signal_cfg: ZScoreSignal | None = None,
    cost_bps: float = 2.0,
    warmup: int = 60,
    zscore_window: int = 60,
) -> BacktestResult:
    """Run the full Kalman + threshold pipeline on a single pair.

    Trading signals use a rolling z-score of the Kalman spread rather than the
    filter's instantaneous innovation variance, which is dominated by R and
    suppresses the signal range. The rolling window is set by `zscore_window`.
    """
    kalman = kalman or KalmanHedge()
    state = kalman.filter(log_y, log_x)

    spread = state.spread
    rolling_mean = spread.rolling(zscore_window).mean()
    rolling_std = spread.rolling(zscore_window).std(ddof=1)
    z = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    z.iloc[:warmup] = np.nan  # ignore filter burn-in
    positions = generate_signals(z, signal_cfg)

    spread_returns = state.spread.diff().fillna(0)
    pnl = apply_costs(positions, spread_returns, cost_bps=cost_bps)
    equity = (1 + pnl["net_return"]).cumprod()

    net = pnl["net_return"]
    kpis = {
        "sharpe_net": _sharpe(net),
        "sharpe_gross": _sharpe(pnl["gross_return"]),
        "annual_return": float(net.mean() * ANNUALISATION),
        "annual_vol": float(net.std(ddof=1) * np.sqrt(ANNUALISATION)),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": _hit_rate(net),
        "avg_turnover": float(pnl["turnover"].mean()),
        "n_trades": float((pnl["turnover"] > 0).sum()),
    }
    return BacktestResult(
        state=state, positions=positions, pnl=pnl, equity_curve=equity, kpis=kpis
    )
