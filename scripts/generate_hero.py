"""Generate the README hero image (equity curve + factor-beta panel).

Uses synthetic-but-realistic data so the script runs in CI without external
network access. The artefact is committed to docs/assets/hero.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantpairs.backtest import run_backtest


def _synthetic_pair(n: int = 2000, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n, freq="B")
    x = pd.Series(np.cumsum(rng.normal(0.0002, 0.012, n)) + 5.0, index=idx)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.88 * eps[t - 1] + rng.normal(0, 0.012)
    drift = np.linspace(1.4, 1.6, n)  # slow regime drift in beta
    y = pd.Series(drift * x.values + eps, index=idx)
    return y, x


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333",
            "axes.labelcolor": "#222",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlecolor": "#0b1d3a",
            "xtick.color": "#444",
            "ytick.color": "#444",
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
        }
    )


def main(out: Path = Path("docs/assets/hero.png")) -> Path:
    _style()
    y, x = _synthetic_pair()
    bt = run_backtest(y, x, cost_bps=2.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    ax.plot(bt.equity_curve.index, bt.equity_curve.values, color="#0b6efb", linewidth=1.8)
    ax.set_title("Net Equity Curve — Kalman Pairs Strategy (synthetic demo)")
    ax.set_ylabel("Equity (start = 1.0)")
    sharpe = bt.kpis["sharpe_net"]
    mdd = bt.kpis["max_drawdown"]
    ax.text(
        0.02,
        0.95,
        f"Net Sharpe: {sharpe:.2f}\nMax DD: {mdd:.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f7ff", edgecolor="#0b6efb"),
    )

    ax = axes[1]
    rolling_beta = bt.state.beta.rolling(60).mean()
    ax.plot(rolling_beta.index, rolling_beta.values, color="#d62728", linewidth=1.6)
    ax.axhline(1.5, color="#666", linestyle="--", linewidth=0.8, label="true β (drifting 1.4→1.6)")
    ax.set_title("Kalman-Filtered Hedge Ratio (β_t)")
    ax.set_ylabel("β_t")
    ax.legend(loc="lower right")

    fig.suptitle(
        "quant-pairs-lab — Dynamic Statistical Arbitrage",
        fontsize=15,
        color="#0b1d3a",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":  # pragma: no cover
    path = main()
    print(f"Wrote {path}")
