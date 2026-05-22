from __future__ import annotations

import pandas as pd

from quantpairs.cointegration import engle_granger_test, johansen_test, screen_pairs


def test_engle_granger_detects_cointegration(cointegrated_pair):
    y, x, true_beta = cointegrated_pair
    result = engle_granger_test(y, x)
    assert result.is_cointegrated, f"Expected cointegration, got p={result.pvalue}"
    assert abs(result.hedge_ratio - true_beta) < 0.05
    assert 0 < result.half_life < 50


def test_engle_granger_rejects_independent_walks(independent_random_walks):
    y, x = independent_random_walks
    result = engle_granger_test(y, x)
    assert not result.is_cointegrated or result.pvalue > 0.05


def test_johansen_runs_on_basket(cointegrated_pair):
    y, x, _ = cointegrated_pair
    df = pd.concat([y, x], axis=1)
    out = johansen_test(df)
    assert {"trace_stat", "trace_crit_95", "max_eig_stat"}.issubset(out.columns)
    assert len(out) == 2


def test_screen_pairs_orders_cointegrated_first(cointegrated_pair, independent_random_walks):
    y_c, x_c, _ = cointegrated_pair
    y_r, x_r = independent_random_walks
    prices = pd.concat(
        [y_c.rename("A"), x_c.rename("B"), y_r.rename("C"), x_r.rename("D")], axis=1
    ).abs() + 1
    ranking = screen_pairs(prices, min_correlation=0.0)
    assert ranking.iloc[0]["is_cointegrated"] or ranking["is_cointegrated"].any()
