"""Walk-forward backtesting with expanding train window and held-out OOS slices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpairs.backtest import BacktestResult, run_backtest
from quantpairs.kalman import KalmanHedge
from quantpairs.signals import ZScoreSignal


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated out-of-sample performance from a walk-forward run."""

    oos_returns: pd.Series
    folds: list[BacktestResult]
    fold_kpis: pd.DataFrame
    oos_kpis: dict[str, float]


def _fold_indices(
    n: int, train_size: int, test_size: int, step: int | None = None
) -> Iterator[tuple[slice, slice]]:
    step = step or test_size
    start = 0
    while start + train_size + test_size <= n:
        train = slice(start, start + train_size)
        test = slice(start + train_size, start + train_size + test_size)
        yield train, test
        start += step


def walk_forward(
    log_y: pd.Series,
    log_x: pd.Series,
    train_size: int = 504,
    test_size: int = 126,
    step: int | None = None,
    kalman: KalmanHedge | None = None,
    signal_cfg: ZScoreSignal | None = None,
    cost_bps: float = 2.0,
) -> WalkForwardResult:
    """Expanding-anchor walk-forward: each fold filters on `train+test`, but
    only the test slice contributes to OOS P&L."""
    df = pd.concat([log_y.rename("y"), log_x.rename("x")], axis=1).dropna()
    n = len(df)

    fold_results: list[BacktestResult] = []
    oos_chunks: list[pd.Series] = []
    rows: list[dict[str, object]] = []

    for fi, (train_idx, test_idx) in enumerate(_fold_indices(n, train_size, test_size, step)):
        window = df.iloc[train_idx.start : test_idx.stop]
        res = run_backtest(
            window["y"], window["x"], kalman=kalman, signal_cfg=signal_cfg, cost_bps=cost_bps
        )
        test_returns = res.pnl["net_return"].iloc[train_size:]
        oos_chunks.append(test_returns)
        fold_results.append(res)
        rows.append(
            {
                "fold": fi,
                "test_start": str(test_returns.index[0].date()) if len(test_returns) else "",
                "test_end": str(test_returns.index[-1].date()) if len(test_returns) else "",
                "sharpe_net": res.kpis["sharpe_net"],
                "n_trades": res.kpis["n_trades"],
            }
        )

    oos = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    equity = (1 + oos).cumprod()
    sd = oos.std(ddof=1)
    sharpe = float(oos.mean() / sd * np.sqrt(252)) if sd and not np.isnan(sd) else 0.0
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    oos_kpis = {
        "oos_sharpe": sharpe,
        "oos_annual_return": float(oos.mean() * 252),
        "oos_max_drawdown": max_dd,
        "oos_n_obs": float(len(oos)),
    }
    return WalkForwardResult(
        oos_returns=oos,
        folds=fold_results,
        fold_kpis=pd.DataFrame(rows),
        oos_kpis=oos_kpis,
    )
