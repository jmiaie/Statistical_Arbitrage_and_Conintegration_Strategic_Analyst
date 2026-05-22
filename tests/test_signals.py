from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantpairs.signals import ZScoreSignal, generate_signals


def test_signal_enters_short_at_high_z():
    z = pd.Series([0, 1.5, 2.5, 2.0, 0.4, 0.1])
    pos = generate_signals(z, ZScoreSignal(entry=2.0, exit=0.5, stop=4.0))
    assert pos.iloc[0] == 0
    assert pos.iloc[2] == -1  # entered short
    assert pos.iloc[4] == 0  # exited at |z|<=0.5


def test_signal_enters_long_at_low_z():
    z = pd.Series([0, -2.5, -1.0, -0.3])
    pos = generate_signals(z)
    assert pos.iloc[1] == 1
    assert pos.iloc[-1] == 0


def test_stop_out_triggers():
    z = pd.Series([0, 2.5, 4.5])
    pos = generate_signals(z, ZScoreSignal(entry=2.0, exit=0.5, stop=4.0))
    assert pos.iloc[1] == -1
    assert pos.iloc[2] == 0


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        ZScoreSignal(entry=1.0, exit=2.0, stop=4.0)


def test_nan_zscore_preserves_state():
    z = pd.Series([0, 2.5, np.nan, 1.0])
    pos = generate_signals(z)
    assert pos.iloc[1] == -1
    assert pos.iloc[2] == -1  # held through NaN
