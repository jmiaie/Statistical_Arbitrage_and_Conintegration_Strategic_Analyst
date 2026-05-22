"""Two-state regime detector for the spread.

A pragmatic alternative to a full HMM: classify each day as "calm" (low
realised vol of the spread) or "stressed" (high realised vol), then mask
trading entries while in the stressed regime. The full strategy can still
exit existing positions.

Avoids the moving-target problem of fitting a Gaussian HMM in real time —
the threshold is a percentile of trailing vol, recomputed only on the
training window of each WFO fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realised_vol(spread: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling std of spread differences."""
    return spread.diff().rolling(window).std(ddof=1) * np.sqrt(252)


def regime_mask(
    spread: pd.Series, window: int = 20, calm_quantile: float = 0.7, train_size: int | None = None
) -> pd.Series:
    """Boolean mask: True where the regime is calm enough to trade.

    The threshold is estimated on the first `train_size` observations to
    avoid look-ahead. Pass `None` to use the full series (in-sample only).
    """
    vol = realised_vol(spread, window)
    train = vol.iloc[:train_size] if train_size else vol
    threshold = float(np.nanquantile(train.values, calm_quantile))
    return (vol <= threshold).fillna(False)


def apply_regime_filter(positions: pd.Series, allowed: pd.Series) -> pd.Series:
    """Zero out new entries when the regime is stressed; keep exits intact."""
    pos = positions.copy()
    prev = pos.shift(1).fillna(0)
    new_entry = (pos != 0) & (prev == 0)
    blocked = new_entry & (~allowed.reindex(pos.index, fill_value=False))
    pos[blocked] = 0
    return pos
