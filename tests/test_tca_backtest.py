from __future__ import annotations

import numpy as np
import pandas as pd

from quantpairs.backtest import run_backtest
from quantpairs.tca import SquareRootImpact, apply_costs
from quantpairs.wfo import walk_forward


def test_square_root_impact_scales_with_participation():
    model = SquareRootImpact(half_spread_bps=1.0, kappa=0.5)
    notional = pd.Series([1e6, 4e6])  # 4x participation -> 2x impact
    adv = pd.Series([1e8, 1e8])
    sigma = pd.Series([0.02, 0.02])
    cost = model.cost_bps(notional, adv, sigma)
    impact_small = cost.iloc[0] - 1.0
    impact_large = cost.iloc[1] - 1.0
    assert abs(impact_large / impact_small - 2.0) < 1e-9


def test_apply_costs_zero_position_zero_pnl():
    pos = pd.Series([0, 0, 0])
    rets = pd.Series([0.01, -0.02, 0.005])
    pnl = apply_costs(pos, rets, cost_bps=2.0)
    assert (pnl["gross_return"] == 0).all()
    assert (pnl["net_return"] == 0).all()


def test_backtest_produces_finite_kpis(cointegrated_pair):
    y, x, _ = cointegrated_pair
    res = run_backtest(y, x, cost_bps=1.0)
    for k, v in res.kpis.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"
    assert (res.equity_curve > 0).all()


def test_walk_forward_yields_oos_returns(cointegrated_pair):
    y, x, _ = cointegrated_pair
    wfo = walk_forward(y, x, train_size=400, test_size=100)
    assert len(wfo.folds) >= 2
    assert "oos_sharpe" in wfo.oos_kpis
    assert len(wfo.oos_returns) > 0
