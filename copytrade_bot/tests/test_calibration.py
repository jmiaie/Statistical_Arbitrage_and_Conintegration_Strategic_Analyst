"""Tests for the per-source win-rate calibration layer (P1).

A source that overstates its win rate should get discounted toward its realized
rate before filtering and Kelly sizing — but only as settled evidence builds,
and never for a source with no history.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.calibration import Calibrator, SourceStats, PRIOR_STRENGTH
from copytradebot.config import (StrategyConfig, Settings, FilterConfig,
                                 SizingConfig)
from copytradebot.storage import Storage
from copytradebot.pipeline import Pipeline
from copytradebot.executors.paper import PaperExecutor


# ---- calibrator math ------------------------------------------------------ #
def test_unknown_source_is_unchanged():
    cal = Calibrator()
    assert cal.calibrate("nobody", 0.7) == 0.7
    assert cal.calibrate("nobody", None) is None


def test_overstating_source_is_discounted():
    # 40 settled trades quoted at 0.80 but only 0.50 realized -> overstating.
    rows = [("chanA", 0.80, i < 20) for i in range(40)]  # 20 wins / 40
    cal = Calibrator.from_rows(rows)
    out = cal.calibrate("chanA", 0.80)
    assert out < 0.80
    # raw bias 0.30, shrink 40/(40+20)=0.667 -> ~0.20 -> ~0.60
    assert abs(out - 0.60) < 0.02


def test_shrinkage_grows_with_sample_size():
    small = Calibrator.from_rows([("s", 0.8, False)] * 4 + [("s", 0.8, True)] * 0)
    big = Calibrator.from_rows([("s", 0.8, False)] * 40)
    # Both never win; the larger sample should pull the quote down further.
    assert big.calibrate("s", 0.8) < small.calibrate("s", 0.8)


def test_honest_source_barely_moves():
    rows = [("honest", 0.65, i < 65) for i in range(100)]  # realized 0.65
    cal = Calibrator.from_rows(rows)
    assert abs(cal.calibrate("honest", 0.65) - 0.65) < 0.01


# ---- storage round-trip --------------------------------------------------- #
def test_calibration_rows_from_settled_positions():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StrategyConfig(
            calibrate=False,  # isolate: don't calibrate while seeding history
            filters=FilterConfig(min_win_rate=0.5, min_ev=0.0),
            sizing=SizingConfig(mode="fraction", bankroll=1000, fraction=0.05,
                                max_position=200, min_position=1))
        storage = Storage(os.path.join(tmp, "t.db"))
        pipe = Pipeline(cfg, Settings(db_path=os.path.join(tmp, "t.db")), storage)
        ex = PaperExecutor(storage)
        # Open three trades from chanX quoting 80%, settle 1 win / 2 losses.
        for _ in range(3):
            d = pipe.process("Market: X\nYES win rate 80% entry 0.5", source="chanX")
            assert d.placed
        ids = [r["id"] for r in storage.open_positions()]
        ex.resolve(ids[0], "win")
        ex.resolve(ids[1], "loss")
        ex.resolve(ids[2], "loss")
        rows = storage.calibration_rows()
        assert len(rows) == 3
        assert all(src == "chanX" and q == 0.8 for src, q, _ in rows)
        assert sum(1 for _, _, won in rows if won) == 1


# ---- end-to-end effect on the pipeline ------------------------------------ #
def test_calibration_can_filter_out_an_inflating_source():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StrategyConfig(
            calibrate=True,
            filters=FilterConfig(min_win_rate=0.65, min_ev=None,
                                 require_fields=["win_rate"]),
            sizing=SizingConfig(mode="fraction", bankroll=1000, fraction=0.05,
                                max_position=200, min_position=1))
        storage = Storage(os.path.join(tmp, "t.db"))
        pipe = Pipeline(cfg, Settings(db_path=os.path.join(tmp, "t.db")), storage)
        ex = PaperExecutor(storage)

        alert = "Market: X\nYES win rate 70% entry 0.5"
        # Seed lots of settled history: chanX quotes 70% but always loses.
        for _ in range(40):
            d = pipe.process(alert, source="chanX")
            if d.placed:
                ex.resolve(d.position.external_id and
                           storage.open_positions()[0]["id"], "loss")

        # A fresh 70% quote from this discredited source is now calibrated below
        # the 65% threshold and rejected.
        d = pipe.process(alert, source="chanX")
        assert d.signal.calibrated_win_rate < 0.65
        assert not d.placed
        assert any("win rate" in r for r in d.result.reasons)

        # The same quote from an untainted source still passes.
        d2 = pipe.process(alert, source="freshChan")
        assert d2.placed
