"""Sprint 3: adaptive Kalman MLE, regime filter, experiment tracking."""

from __future__ import annotations

import json

import pandas as pd

from quantpairs.adaptive import fit_kalman_mle
from quantpairs.kalman import KalmanHedge
from quantpairs.regime import apply_regime_filter, realised_vol, regime_mask
from quantpairs.tracking import list_runs, log_run


def test_mle_returns_valid_params(cointegrated_pair):
    y, x, _ = cointegrated_pair
    fit = fit_kalman_mle(y, x, initial_delta=1e-4, initial_r=1e-3)
    assert 0 < fit.delta < 1
    assert fit.observation_var > 0
    # The MLE-tuned filter should produce a finite likelihood
    state = KalmanHedge(delta=fit.delta, observation_var=fit.observation_var).filter(y, x)
    assert state.beta.notna().all()


def test_regime_mask_separates_calm_and_stressed(cointegrated_pair):
    y, x, _ = cointegrated_pair
    spread = y - 1.5 * x
    mask = regime_mask(spread, window=20, calm_quantile=0.7)
    # Roughly the requested quantile of bars are calm
    assert 0.4 < mask.mean() < 0.85


def test_realised_vol_positive_and_finite(cointegrated_pair):
    y, x, _ = cointegrated_pair
    rv = realised_vol(y - 1.5 * x).dropna()
    assert (rv >= 0).all()
    assert rv.notna().all()


def test_apply_regime_filter_blocks_new_entries():
    pos = pd.Series([0, 1, 1, 1, 0, -1, -1, 0])
    allowed = pd.Series([True, False, True, True, True, True, True, True])
    out = apply_regime_filter(pos, allowed)
    # The 0->1 entry at index 1 should be blocked (allowed=False)
    assert out.iloc[1] == 0
    # Existing positions and other entries pass through
    assert (out.iloc[2:4] == 1).all()


def test_log_run_writes_manifest(tmp_path):
    path = log_run(
        tag="unit-test",
        params={"cost_bps": 2.0, "entry": 2.0},
        kpis={"sharpe_net": 1.23},
        notes="smoke",
        window=("2018-01-01", "2024-12-31"),
        output_dir=tmp_path,
    )
    payload = json.loads(path.read_text())
    assert payload["tag"] == "unit-test"
    assert payload["kpis"]["sharpe_net"] == 1.23
    assert payload["code_version"]
    runs = list_runs(tmp_path)
    assert len(runs) == 1
