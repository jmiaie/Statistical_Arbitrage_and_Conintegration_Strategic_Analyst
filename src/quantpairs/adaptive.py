"""Maximum-likelihood tuning of Kalman noise covariances Q, R.

The hand-tuned `KalmanHedge(delta=1e-4, observation_var=1e-3)` is a
sensible default, but per-pair MLE can lift Sharpe by re-fitting the
state-evolution noise to each spread's actual mean-reversion speed.

Negative log-likelihood is computed from the filter's innovation sequence
and minimised over (log delta, log R) via SciPy's `minimize`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantpairs.kalman import KalmanHedge


@dataclass(frozen=True)
class FitResult:
    """Outcome of MLE Kalman fit."""

    delta: float
    observation_var: float
    neg_log_likelihood: float
    converged: bool


def _neg_log_likelihood(params: np.ndarray, y: pd.Series, x: pd.Series) -> float:
    log_delta, log_r = params
    delta = float(np.exp(log_delta))
    r = float(np.exp(log_r))
    # Clip into valid range for delta
    delta = min(max(delta, 1e-9), 1 - 1e-9)
    kf = KalmanHedge(delta=delta, observation_var=r)
    state = kf.filter(y, x)
    s = state.innovation_var.values
    v = state.innovation.values
    # Gaussian log-likelihood, ignoring the warmup tail
    valid = slice(60, None)
    ll = -0.5 * np.sum(np.log(2 * np.pi * s[valid]) + v[valid] ** 2 / s[valid])
    return float(-ll)


def fit_kalman_mle(
    log_y: pd.Series,
    log_x: pd.Series,
    initial_delta: float = 1e-4,
    initial_r: float = 1e-3,
) -> FitResult:
    """Fit (delta, R) by maximum likelihood on the observed innovation sequence."""
    x0 = np.array([np.log(initial_delta), np.log(initial_r)])
    res = minimize(
        _neg_log_likelihood,
        x0,
        args=(log_y, log_x),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200},
    )
    log_delta, log_r = res.x
    delta = float(np.clip(np.exp(log_delta), 1e-9, 1 - 1e-9))
    return FitResult(
        delta=delta,
        observation_var=float(np.exp(log_r)),
        neg_log_likelihood=float(res.fun),
        converged=bool(res.success),
    )
