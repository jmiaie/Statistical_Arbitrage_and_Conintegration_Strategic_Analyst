"""Tests for multi-leg opportunity execution (P0 defect #5).

A multi-leg Opportunity (arbitrage, cointegration) must be convertible and
placeable as a basket — previously to_signal() raised and there was no path to
execute more than one leg.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.scanners.base import Opportunity, Leg
from copytradebot.config import StrategyConfig, Settings, FilterConfig, SizingConfig
from copytradebot.storage import Storage
from copytradebot.pipeline import Pipeline


def _arb_opp():
    return Opportunity(
        kind="arbitrage",
        legs=[
            Leg("m1", "Will BTC close above 70k?", "Yes", 0.45, "y1", weight=0.45),
            Leg("m1", "Will BTC close above 70k?", "No", 0.50, "n1", weight=0.50),
        ],
        edge=0.04, confidence=0.95, rationale="prices sum < 1",
    )


# ---- conversion ----------------------------------------------------------- #
def test_to_signals_handles_multi_leg():
    sigs = _arb_opp().to_signals()
    assert len(sigs) == 2
    assert {s.side.value for s in sigs} == {"YES", "NO"}
    assert all(abs(s.ev - 4.0) < 1e-9 for s in sigs)  # basket edge on each leg


def test_to_signal_singular_raises_for_multi_leg():
    with pytest.raises(ValueError):
        _arb_opp().to_signal()


# ---- placement ------------------------------------------------------------ #
def _pipeline(tmp, **over):
    cfg = StrategyConfig(
        mode="paper",
        filters=FilterConfig(min_win_rate=0.55, min_ev=0.0),
        sizing=SizingConfig(mode="fraction", bankroll=1000, fraction=0.05,
                            max_position=200, min_position=1),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    storage = Storage(os.path.join(tmp, "t.db"))
    settings = Settings(db_path=os.path.join(tmp, "t.db"))
    return Pipeline(cfg, settings, storage), storage


def test_place_opportunity_opens_all_legs_linked():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        d = pipe.place_opportunity(_arb_opp(), source="scanner:arbitrage")
        assert d.placed
        assert len(d.legs) == 2
        assert storage.count_open_positions() == 2
        # Both legs share one signal_id (the group linkage).
        assert all(p.signal_id == d.signal_id for p in d.legs)
        # Basket total split across legs by weight; total within the cap.
        assert abs(d.total_stake - sum(p.stake for p in d.legs)) < 0.05
        assert d.total_stake <= 200.0


def test_place_opportunity_gated_by_edge():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp,
                                  filters=FilterConfig(min_win_rate=0.55, min_ev=10.0))
        d = pipe.place_opportunity(_arb_opp())  # edge 4% < 10%
        assert not d.placed
        assert storage.count_open_positions() == 0
        assert "min_ev" in d.note


def test_place_opportunity_respects_exposure_cap():
    with tempfile.TemporaryDirectory() as tmp:
        from copytradebot.config import RiskConfig
        # cap = 0.02 x 1000 = 20; basket total 50 -> blocked.
        pipe, storage = _pipeline(tmp, risk=RiskConfig(max_exposure_fraction=0.02))
        d = pipe.place_opportunity(_arb_opp())
        assert not d.placed
        assert "exposure cap" in d.note
        assert storage.count_open_positions() == 0


def test_place_opportunity_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp, dry_run=True)
        d = pipe.place_opportunity(_arb_opp())
        assert not d.placed
        assert d.note.startswith("dry-run")
        assert storage.count_open_positions() == 0
