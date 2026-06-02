import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.config import StrategyConfig, Settings, FilterConfig, SizingConfig
from copytradebot.storage import Storage
from copytradebot.pipeline import Pipeline
from copytradebot.executors.paper import PaperExecutor
from copytradebot.sizing import compute_stake, kelly_fraction
from copytradebot.parser import parse_alert, enrich


def _pipeline(tmp, **overrides):
    cfg = StrategyConfig(
        mode="paper",
        filters=FilterConfig(min_win_rate=0.6, min_ev=0.0),
        sizing=SizingConfig(mode="fraction", bankroll=1000, fraction=0.05,
                            max_position=200, min_position=1),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    storage = Storage(os.path.join(tmp, "t.db"))
    settings = Settings(db_path=os.path.join(tmp, "t.db"))
    return Pipeline(cfg, settings, storage), storage


def test_passing_alert_places_paper_trade():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        d = pipe.process("Market: BTC up?\nYES win rate 70% EV +12% entry 0.5")
        assert d.placed
        assert d.position.venue == "paper"
        assert d.stake == 50.0  # 5% of 1000
        assert storage.count_open_positions() == 1


def test_failing_alert_is_recorded_not_placed():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        d = pipe.process("Market: X\nwin rate 40%")
        assert not d.placed
        assert storage.count_open_positions() == 0
        assert storage.stats()["signals_seen"] == 1


def test_dry_run_does_not_place():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp, dry_run=True)
        d = pipe.process("Market: X\nYES win rate 70% entry 0.5")
        assert not d.placed
        assert d.note.startswith("dry-run")


def test_paper_resolution_updates_pnl():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        d = pipe.process("Market: X\nYES win rate 70% entry 0.5")
        pos_id = storage.open_positions()[0]["id"]
        ex = PaperExecutor(storage)
        ok, pnl, status = ex.resolve(pos_id, "win")
        assert ok and status == "won"
        # stake 50 at price 0.5 -> profit = 50*(1/0.5 - 1) = 50
        assert abs(pnl - 50.0) < 1e-6
        assert storage.stats()["win_rate"] == 1.0


def test_kelly_fraction():
    # p=0.6, price=0.5, b=1 -> f = 0.6 - 0.4/1 = 0.2
    assert abs(kelly_fraction(0.6, 0.5) - 0.2) < 1e-9
    assert kelly_fraction(0.4, 0.5) == 0.0  # no edge -> no bet


def test_sizing_respects_max_position():
    sig = enrich(parse_alert("YES win rate 90% entry 0.5"))
    cfg = SizingConfig(mode="fraction", bankroll=100000, fraction=0.5,
                       max_position=200)
    assert compute_stake(sig, cfg) == 200.0
