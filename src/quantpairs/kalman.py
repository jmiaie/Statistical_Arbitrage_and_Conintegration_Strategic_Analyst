"""Kalman filter for time-varying hedge ratio estimation.

We model the relationship between two log-prices as:

    y_t = beta_t * x_t + alpha_t + eps_t,    eps_t ~ N(0, R)
    [beta_t; alpha_t] = [beta_{t-1}; alpha_{t-1}] + eta_t,  eta_t ~ N(0, Q)

This is a 2-state random-walk model. Q is a 2x2 diagonal matrix
controlling how quickly the slope and intercept may drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanState:
    """Per-bar filter output."""

    beta: pd.Series
    alpha: pd.Series
    spread: pd.Series
    innovation: pd.Series
    innovation_var: pd.Series

    def zscore(self) -> pd.Series:
        """Standardised innovation, the trading signal input."""
        return self.innovation / np.sqrt(self.innovation_var.clip(lower=1e-12))


class KalmanHedge:
    """Dynamic hedge-ratio estimator using a random-walk Kalman filter."""

    def __init__(
        self,
        delta: float = 1e-4,
        observation_var: float = 1e-3,
        initial_state_var: float = 1.0,
    ) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1).")
        self.delta = delta
        self.R = observation_var
        self.Q = (delta / (1 - delta)) * np.eye(2)
        self.P0 = initial_state_var * np.eye(2)

    def filter(self, y: pd.Series, x: pd.Series) -> KalmanState:
        """Run the filter forward over aligned series `y` and `x`."""
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        n = len(df)
        if n < 2:
            raise ValueError("Need at least two observations.")

        beta = np.zeros(n)
        alpha = np.zeros(n)
        spread = np.zeros(n)
        innov = np.zeros(n)
        innov_var = np.zeros(n)

        state = np.zeros(2)
        P = self.P0.copy()

        for t in range(n):
            xt = float(df["x"].iloc[t])
            yt = float(df["y"].iloc[t])
            H = np.array([xt, 1.0])

            # Predict (random walk: F = I, so state unchanged; covariance grows by Q)
            P_pred = P + self.Q

            # Innovation
            y_hat = float(H @ state)
            v = yt - y_hat
            S = float(H @ P_pred @ H.T + self.R)

            # Update
            K = (P_pred @ H) / S
            state = state + K * v
            P = P_pred - np.outer(K, H) @ P_pred

            beta[t] = state[0]
            alpha[t] = state[1]
            spread[t] = yt - state[0] * xt - state[1]
            innov[t] = v
            innov_var[t] = S

        idx = df.index
        return KalmanState(
            beta=pd.Series(beta, index=idx, name="beta"),
            alpha=pd.Series(alpha, index=idx, name="alpha"),
            spread=pd.Series(spread, index=idx, name="spread"),
            innovation=pd.Series(innov, index=idx, name="innovation"),
            innovation_var=pd.Series(innov_var, index=idx, name="innovation_var"),
        )
