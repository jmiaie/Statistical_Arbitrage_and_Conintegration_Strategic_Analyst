"""Cointegration testing: Engle-Granger (pairs) and Johansen (baskets)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class CointegrationResult:
    """Outcome of a cointegration test."""

    is_cointegrated: bool
    pvalue: float
    hedge_ratio: float
    adf_pvalue: float
    half_life: float

    def __str__(self) -> str:
        flag = "YES" if self.is_cointegrated else "NO"
        return (
            f"Cointegrated: {flag} | EG p={self.pvalue:.4f} | "
            f"beta={self.hedge_ratio:.4f} | half-life={self.half_life:.1f}d"
        )


def _half_life(spread: pd.Series) -> float:
    """Estimate Ornstein-Uhlenbeck half-life from spread autocorrelation."""
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")
    model = sm.OLS(delta, sm.add_constant(lagged)).fit()
    theta = float(model.params.iloc[1])
    if theta >= 0:
        return float("inf")
    return float(-np.log(2) / theta)


def engle_granger_test(
    y: pd.Series, x: pd.Series, significance: float = 0.05
) -> CointegrationResult:
    """Two-step Engle-Granger cointegration test on prices `y` and `x`."""
    aligned = pd.concat([y, x], axis=1).dropna()
    y_a, x_a = aligned.iloc[:, 0], aligned.iloc[:, 1]
    _, p_coint, _ = coint(y_a, x_a)
    fit = sm.OLS(y_a, sm.add_constant(x_a)).fit()
    beta = float(fit.params.iloc[1])
    spread = y_a - beta * x_a
    adf_p = float(adfuller(spread, autolag="AIC")[1])
    return CointegrationResult(
        is_cointegrated=p_coint < significance and adf_p < significance,
        pvalue=float(p_coint),
        hedge_ratio=beta,
        adf_pvalue=adf_p,
        half_life=_half_life(spread),
    )


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    """Johansen cointegration test for `n >= 2` series.

    Returns a tidy DataFrame of trace and max-eigenvalue statistics with critical values.
    """
    if prices.shape[1] < 2:
        raise ValueError("Johansen test requires at least two columns.")
    result = coint_johansen(prices.dropna().values, det_order, k_ar_diff)
    rows = []
    for i in range(prices.shape[1]):
        rows.append(
            {
                "r<=": i,
                "trace_stat": float(result.lr1[i]),
                "trace_crit_95": float(result.cvt[i, 1]),
                "max_eig_stat": float(result.lr2[i]),
                "max_eig_crit_95": float(result.cvm[i, 1]),
            }
        )
    return pd.DataFrame(rows)


def screen_pairs(
    prices: pd.DataFrame,
    min_correlation: float = 0.7,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise Engle-Granger over all column combinations and return ranked candidates."""
    cols = list(prices.columns)
    log_prices = np.log(prices)
    corr = log_prices.corr()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if abs(corr.loc[a, b]) < min_correlation:
                continue
            res = engle_granger_test(log_prices[a], log_prices[b], significance)
            rows.append(
                {
                    "asset_y": a,
                    "asset_x": b,
                    "corr": float(corr.loc[a, b]),
                    "eg_pvalue": res.pvalue,
                    "adf_pvalue": res.adf_pvalue,
                    "hedge_ratio": res.hedge_ratio,
                    "half_life_days": res.half_life,
                    "is_cointegrated": res.is_cointegrated,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["is_cointegrated", "eg_pvalue"], ascending=[False, True])
        .reset_index(drop=True)
    )
