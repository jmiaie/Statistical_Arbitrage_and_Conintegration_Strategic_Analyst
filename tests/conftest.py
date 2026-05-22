"""Shared fixtures: synthetic cointegrated and non-cointegrated series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def cointegrated_pair(rng: np.random.Generator) -> tuple[pd.Series, pd.Series, float]:
    """y = 1.5 * x + stationary noise. Known true beta = 1.5."""
    n = 1500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx, name="x") + 5.0
    # AR(1) stationary residual
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.85 * eps[t - 1] + rng.normal(0, 0.01)
    y = pd.Series(1.5 * x.values + eps, index=idx, name="y")
    return y, x, 1.5


@pytest.fixture
def independent_random_walks(rng: np.random.Generator) -> tuple[pd.Series, pd.Series]:
    """Two unrelated random walks — should NOT be cointegrated."""
    n = 1500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    y = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 10, index=idx, name="y")
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 10, index=idx, name="x")
    return y, x
