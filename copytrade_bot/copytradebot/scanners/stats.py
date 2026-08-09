"""Lightweight statistics for the stat-arb scanners (pure Python, no deps).

Just enough Engle-Granger machinery to test a pair of prediction-market price
series for cointegration and to time entries on the spread's z-score.
"""

from __future__ import annotations

import math


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ols(x: list[float], y: list[float]) -> tuple[float, float, list[float]]:
    """Fit ``y = a + b*x``; return (a, b, residuals)."""
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0, 0.0, []
    sx, sy = sum(x), sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        b = 0.0
    else:
        b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    return a, b, resid


def zscore_last(series: list[float], lookback: int | None = None) -> float:
    """Z-score of the most recent point vs. the (rolling) window."""
    s = series[-lookback:] if lookback else series
    sd = std(s)
    if sd < 1e-12:
        return 0.0
    return (s[-1] - mean(s)) / sd


def adf_tstat(series: list[float]) -> float:
    """Dickey-Fuller t-stat (no lags, no trend) on ``series``.

    Regress Δs_t on s_{t-1}: more negative => stronger mean-reversion /
    stationarity. Compare against a critical value (e.g. -3.0 for EG residuals).
    """
    if len(series) < 5:
        return 0.0
    lag = series[:-1]
    delta = [series[i] - series[i - 1] for i in range(1, len(series))]
    # Regress delta on lag with intercept.
    a, b, resid = ols(lag, delta)
    n = len(delta)
    ssr = sum(r * r for r in resid)
    sigma2 = ssr / (n - 2) if n > 2 else 0.0
    m = mean(lag)
    sxx = sum((xi - m) ** 2 for xi in lag)
    if sigma2 <= 0 or sxx <= 0:
        return 0.0
    se = math.sqrt(sigma2 / sxx)
    return b / se if se > 0 else 0.0


def half_life(series: list[float]) -> float:
    """Mean-reversion half-life (periods) from an AR(1) fit; inf if none."""
    if len(series) < 5:
        return float("inf")
    lag = series[:-1]
    delta = [series[i] - series[i - 1] for i in range(1, len(series))]
    _, k, _ = ols(lag, delta)
    if k >= 0 or (1 + k) <= 0:
        return float("inf")
    return -math.log(2) / math.log(1 + k)
