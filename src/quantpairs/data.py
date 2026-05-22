"""Data loaders. yfinance for prices, Ken French library for factors."""

from __future__ import annotations

from datetime import date

import pandas as pd


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Adjusted-close prices for a list of tickers via yfinance."""
    import yfinance as yf

    end = end or date.today().isoformat()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.dropna(how="all")


def fetch_fama_french(start: str, end: str | None = None) -> pd.DataFrame:
    """Daily Fama-French 3 factors + RF, in decimal form."""
    from pandas_datareader import data as pdr

    end = end or date.today().isoformat()
    raw = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0]
    raw.index = pd.to_datetime(raw.index)
    return raw / 100.0
