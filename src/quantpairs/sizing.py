"""Position-sizing helpers: vol targeting + Kelly fraction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target(
    returns: pd.Series,
    target_annual_vol: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 4.0,
) -> pd.Series:
    """Scale `returns` by inverse trailing realised volatility to hit a target."""
    if not 0 < target_annual_vol < 5:
        raise ValueError("target_annual_vol must be a sensible decimal (e.g. 0.10).")
    realised = returns.rolling(lookback).std(ddof=1) * np.sqrt(252)
    scale = (target_annual_vol / realised).clip(upper=max_leverage).fillna(0)
    return scale.shift(1).fillna(0)


def kelly_fraction(returns: pd.Series, lookback: int = 252, fraction: float = 0.25) -> pd.Series:
    """Conservative fractional-Kelly sizing: f = (μ / σ²) * fraction.

    `fraction` defaults to 0.25 (quarter-Kelly) — full Kelly is dangerous in
    practice because moments are estimated with noise.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    mu = returns.rolling(lookback).mean()
    var = returns.rolling(lookback).var(ddof=1).replace(0, np.nan)
    raw = (mu / var) * fraction
    return raw.shift(1).fillna(0).clip(lower=-4, upper=4)
