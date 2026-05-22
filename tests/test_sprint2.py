"""Sprint 2: portfolio, sizing, bootstrap, benchmark, tearsheet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantpairs.backtest import run_backtest
from quantpairs.benchmarks import compare_to_kalman, static_ols_backtest
from quantpairs.portfolio import run_portfolio
from quantpairs.sizing import kelly_fraction, vol_target
from quantpairs.stats import sharpe_ci
from quantpairs.tearsheet import write_tearsheet


def test_vol_target_scales_inversely_with_vol(rng):
    high_vol = pd.Series(rng.normal(0, 0.02, 500))
    low_vol = pd.Series(rng.normal(0, 0.005, 500))
    s_high = vol_target(high_vol, target_annual_vol=0.10).iloc[-1]
    s_low = vol_target(low_vol, target_annual_vol=0.10).iloc[-1]
    assert s_low > s_high


def test_vol_target_rejects_silly_targets():
    with pytest.raises(ValueError):
        vol_target(pd.Series([0.01] * 100), target_annual_vol=-0.1)


def test_kelly_fraction_finite(rng):
    r = pd.Series(rng.normal(0.0005, 0.01, 600))
    f = kelly_fraction(r, lookback=100, fraction=0.25)
    assert np.isfinite(f).all()
    assert (f.abs() <= 4).all()


def test_sharpe_ci_brackets_point_estimate(rng):
    r = pd.Series(rng.normal(0.001, 0.01, 600))
    ci = sharpe_ci(r, n_boot=300, seed=1)
    assert ci.lower <= ci.point <= ci.upper
    assert 0 < ci.upper - ci.lower < 10


def test_sharpe_ci_requires_enough_data():
    with pytest.raises(ValueError):
        sharpe_ci(pd.Series([0.01] * 10))


def test_static_ols_benchmark_runs(cointegrated_pair):
    y, x, _ = cointegrated_pair
    out = static_ols_backtest(y, x, cost_bps=1.0)
    for k in ["sharpe_net", "annual_return", "max_drawdown", "static_beta"]:
        assert k in out
        assert np.isfinite(out[k])


def test_compare_to_kalman_returns_dataframe(cointegrated_pair):
    y, x, _ = cointegrated_pair
    bt = run_backtest(y, x, cost_bps=1.0)
    ols = static_ols_backtest(y, x, cost_bps=1.0)
    df = compare_to_kalman(bt, ols)
    assert {"Kalman", "Static OLS", "Lift"}.issubset(df.columns)
    assert len(df) == 6


def test_portfolio_combines_pairs(cointegrated_pair, rng):
    y, x, _ = cointegrated_pair
    # Build a second synthetic pair
    n = len(y)
    idx = y.index
    x2 = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 5, index=idx)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.9 * eps[t - 1] + rng.normal(0, 0.008)
    y2 = pd.Series(1.2 * x2.values + eps, index=idx)

    res = run_portfolio(
        {("A", "B"): (y, x), ("C", "D"): (y2, x2)},
        cost_bps=1.0,
        target_annual_vol=0.10,
        vol_lookback=60,
    )
    assert res.kpis["n_pairs"] == 2
    assert len(res.portfolio_returns) == n
    assert (res.equity_curve > 0).all()


def test_portfolio_rejects_empty():
    with pytest.raises(ValueError):
        run_portfolio({})


def test_tearsheet_writes_valid_html(cointegrated_pair, tmp_path):
    y, x, _ = cointegrated_pair
    bt = run_backtest(y, x, cost_bps=1.0)
    out = write_tearsheet(bt, output_path=tmp_path / "ts.html")
    content = out.read_text()
    assert "<title>" in content
    assert "Equity Curve" in content
    assert "data:image/png;base64," in content
    assert out.stat().st_size > 5000
