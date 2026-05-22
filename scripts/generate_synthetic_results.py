"""Generate `results/` artefacts from synthetic data — offline-runnable.

Exercises the same pipeline as `research/main_backtest.py`, but with a
deterministic synthetic pair so CI and reviewers don't need network access
to see what the output looks like.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantpairs.backtest import run_backtest
from quantpairs.benchmarks import compare_to_kalman, static_ols_backtest
from quantpairs.cointegration import engle_granger_test
from quantpairs.stats import sharpe_ci
from quantpairs.tearsheet import write_tearsheet
from quantpairs.tracking import log_run
from quantpairs.wfo import walk_forward

RESULTS = Path("results")


def _synthetic_pair(n: int = 1800, seed: int = 11) -> tuple[pd.Series, pd.Series]:
    """Mean-reverting pair with a slow drift in the true hedge ratio."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    # Random walk for x (log price)
    x = pd.Series(np.cumsum(rng.normal(0.0003, 0.012, n)) + 5.0, index=idx)
    # Mean-reverting AR(1) residual with smaller noise so spread is tradeable
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.92 * eps[t - 1] + rng.normal(0, 0.006)
    # Slowly drifting hedge ratio (regime shift the Kalman filter should catch)
    drift = np.linspace(1.48, 1.52, n)
    y = pd.Series(drift * x.values + eps, index=idx)
    return y, x


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    log_y, log_x = _synthetic_pair()

    coint = engle_granger_test(log_y, log_x)
    bt = run_backtest(log_y, log_x, cost_bps=2.0)
    wfo = walk_forward(log_y, log_x, train_size=504, test_size=126, cost_bps=2.0)

    fig, ax = plt.subplots(figsize=(11, 5))
    equity = (1 + wfo.oos_returns).cumprod()
    ax.plot(equity.index, equity.values, color="#0b6efb", linewidth=1.5)
    ax.set_title("Out-of-sample Equity Curve — synthetic pair (demo)")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.grid(True, color="#e6e6e6")
    fig.tight_layout()
    fig.savefig(RESULTS / "equity_curve.png", dpi=160)
    plt.close(fig)

    wfo.fold_kpis.to_csv(RESULTS / "wfo_folds.csv", index=False)

    # Sprint 2: bootstrap CI and OLS benchmark
    ci = sharpe_ci(wfo.oos_returns, n_boot=1000, seed=42)
    ols = static_ols_backtest(log_y, log_x, cost_bps=2.0)
    comparison = compare_to_kalman(bt, ols)
    comparison.to_csv(RESULTS / "kalman_vs_ols.csv", index=False)

    # Sprint 2: tear-sheet
    write_tearsheet(bt, RESULTS / "tearsheet.html", title="Synthetic Pair — Kalman Backtest")

    payload = {
        "note": "Synthetic-data demo. Use research/main_backtest.py with real tickers.",
        "cointegration": {
            "eg_pvalue": coint.pvalue,
            "adf_pvalue": coint.adf_pvalue,
            "hedge_ratio": coint.hedge_ratio,
            "half_life_days": coint.half_life,
        },
        "in_sample_kpis": bt.kpis,
        "oos_kpis": wfo.oos_kpis,
        "oos_sharpe_ci_95": {"point": ci.point, "lower": ci.lower, "upper": ci.upper},
        "kalman_vs_ols_lift_sharpe": float(
            comparison.loc[comparison["Metric"] == "sharpe_net", "Lift"].iloc[0]
        ),
    }
    (RESULTS / "kpis.json").write_text(json.dumps(payload, indent=2))

    # Sprint 3: experiment manifest
    log_run(
        tag="synthetic-demo",
        params={"cost_bps": 2.0, "train_size": 504, "test_size": 126},
        kpis={**bt.kpis, **wfo.oos_kpis},
        notes="Synthetic mean-reverting pair, deterministic seed.",
    )

    print("Synthetic results written to", RESULTS)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
