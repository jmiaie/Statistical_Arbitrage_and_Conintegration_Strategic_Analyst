"""Bootstrap confidence intervals for Sharpe and other path-dependent KPIs.

Uses the stationary bootstrap of Politis & Romano (1994) with geometric
block lengths, which preserves serial dependence — important for daily
strategy returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALISATION = 252


@dataclass(frozen=True)
class BootstrapCI:
    """Two-sided confidence interval from a bootstrap resample."""

    point: float
    lower: float
    upper: float
    confidence: float

    def __str__(self) -> str:
        return (
            f"{self.point:.3f}  "
            f"[{self.lower:.3f}, {self.upper:.3f}]  (CI={self.confidence:.0%})"
        )


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(x.mean() / sd * np.sqrt(ANNUALISATION))


def _stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: indices for one resample of length n."""
    p = 1.0 / mean_block
    out = np.empty(n, dtype=np.int64)
    out[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            out[t] = rng.integers(0, n)
        else:
            out[t] = (out[t - 1] + 1) % n
    return out


def sharpe_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Stationary-bootstrap CI for annualised Sharpe ratio."""
    arr = returns.dropna().values
    n = len(arr)
    if n < 30:
        raise ValueError("Need at least 30 observations for a meaningful CI.")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block, rng)
        samples[b] = _sharpe(arr[idx])
    alpha = (1 - confidence) / 2
    return BootstrapCI(
        point=_sharpe(arr),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1 - alpha)),
        confidence=confidence,
    )
