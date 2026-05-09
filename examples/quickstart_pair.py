"""Minimal end-to-end example: cointegration test on a single pair.

This script is intentionally short — it exists to give a reader something
runnable in under 30 seconds. The full research pipeline lives in the
notebook under research/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint


def fetch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return data["Close"].dropna()


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    _, p_coint, _ = coint(y, x)
    beta_hat = sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1]
    spread = y - beta_hat * x
    adf_p = adfuller(spread)[1]
    return {"coint_p": p_coint, "adf_p": adf_p, "beta_hat": float(beta_hat)}


if __name__ == "__main__":
    prices = fetch(["KO", "PEP"], start="2018-01-01", end="2024-12-31")
    result = engle_granger(np.log(prices["KO"]), np.log(prices["PEP"]))
    print(f"Engle-Granger p-value : {result['coint_p']:.4f}")
    print(f"ADF on residuals      : {result['adf_p']:.4f}")
    print(f"Static hedge ratio    : {result['beta_hat']:.4f}")
