from __future__ import annotations

import numpy as np

from quantpairs.kalman import KalmanHedge


def test_kalman_converges_to_true_beta(cointegrated_pair):
    y, x, true_beta = cointegrated_pair
    kf = KalmanHedge(delta=1e-4, observation_var=1e-3)
    state = kf.filter(y, x)

    # After burn-in the filtered beta should be close to the truth
    tail = state.beta.iloc[-200:].mean()
    assert abs(tail - true_beta) < 0.1, f"Filter did not converge: {tail} vs {true_beta}"


def test_kalman_spread_is_zero_mean(cointegrated_pair):
    y, x, _ = cointegrated_pair
    kf = KalmanHedge()
    state = kf.filter(y, x)
    tail_mean = state.spread.iloc[-500:].mean()
    assert abs(tail_mean) < 0.05


def test_kalman_zscore_well_defined(cointegrated_pair):
    y, x, _ = cointegrated_pair
    state = KalmanHedge().filter(y, x)
    z = state.zscore().iloc[100:]
    assert np.isfinite(z).all()


def test_invalid_delta_rejected():
    import pytest

    with pytest.raises(ValueError):
        KalmanHedge(delta=0)
