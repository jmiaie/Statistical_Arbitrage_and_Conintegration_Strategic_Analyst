"""Z-score entry/exit logic for mean-reverting spreads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZScoreSignal:
    """Stateless threshold configuration for spread trading."""

    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0

    def __post_init__(self) -> None:
        if not (0 <= self.exit < self.entry < self.stop):
            raise ValueError("Require 0 <= exit < entry < stop.")


def generate_signals(zscore: pd.Series, cfg: ZScoreSignal | None = None) -> pd.Series:
    """Translate a z-score series into a stateful position in {-1, 0, +1}.

    Convention: position is on the spread (long spread = +1, short = -1).
    Long spread when z is deeply negative (expect mean-reversion up);
    short spread when z is deeply positive.
    """
    cfg = cfg or ZScoreSignal()
    z = zscore.values
    n = len(z)
    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = state
            continue
        if state == 0:
            if zt <= -cfg.entry:
                state = 1
            elif zt >= cfg.entry:
                state = -1
        elif (state == 1 and (zt >= -cfg.exit or zt <= -cfg.stop)) or (
            state == -1 and (zt <= cfg.exit or zt >= cfg.stop)
        ):
            state = 0
        pos[t] = state
    return pd.Series(pos, index=zscore.index, name="position")
