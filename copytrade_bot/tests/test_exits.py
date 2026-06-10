"""Tests for exit/close intent: parsing and pipeline settlement.

Covers P0 defect #4 — an exit alert must never open a position, and should
settle the open position(s) it refers to.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.models import Intent
from copytradebot.parser import parse_alert
from copytradebot.config import StrategyConfig, Settings, FilterConfig, SizingConfig
from copytradebot.storage import Storage
from copytradebot.pipeline import Pipeline


# ---- intent parsing ------------------------------------------------------- #
def test_entry_alert_is_entry():
    s = parse_alert("NEW PLAY: Market BTC up? YES win rate 70% entry 0.5 TP 0.8")
    assert s.intent is Intent.ENTRY


def test_close_alert_is_exit():
    for txt in ["Closing my NVDA position", "Sold out of BTC",
                "stopped out of the ETH long", "TP hit on Fed market, out at 0.82",
                "booking profit on Lakers"]:
        assert parse_alert(txt).intent is Intent.EXIT, txt


def test_take_profit_target_in_entry_is_not_exit():
    # "take profit" as a target level (no completion cue) must stay an entry.
    s = parse_alert("LONG BTC entry 0.42, take profit 0.80, stop 0.30")
    assert s.intent is Intent.ENTRY


def test_explicit_entry_cue_overrides_exit_word():
    s = parse_alert("New position: close the gap trade, YES entry 0.40")
    assert s.intent is Intent.ENTRY


def test_exit_price_parsed():
    s = parse_alert("Closed BTC market, out at 0.82")
    assert s.intent is Intent.EXIT
    assert abs(s.entry_price - 0.82) < 1e-9


# ---- pipeline exit routing ------------------------------------------------ #
def _pipeline(tmp):
    cfg = StrategyConfig(
        mode="paper",
        filters=FilterConfig(min_win_rate=0.6, min_ev=0.0),
        sizing=SizingConfig(mode="fraction", bankroll=1000, fraction=0.05,
                            max_position=200, min_position=1),
    )
    storage = Storage(os.path.join(tmp, "t.db"))
    settings = Settings(db_path=os.path.join(tmp, "t.db"))
    return Pipeline(cfg, settings, storage), storage


def test_exit_does_not_open_a_position():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        d = pipe.process("Closing the BTC market position, out at 0.6")
        assert not d.placed
        assert storage.count_open_positions() == 0


def test_exit_closes_matching_open_position():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        # Open a position on a BTC market.
        opened = pipe.process(
            "Market: Will BTC close above 70k?\nYES win rate 70% entry 0.5")
        assert opened.placed
        assert storage.count_open_positions() == 1

        # A close alert referencing the same market settles it.
        closed = pipe.process("Closing the BTC close above 70k market, out at 0.9")
        assert closed.closed, closed.note
        pid, status, pnl = closed.closed[0]
        assert status == "won"          # exit 0.9 vs entry 0.5
        assert storage.count_open_positions() == 0


def test_exit_without_match_does_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        pipe.process("Market: Will BTC close above 70k?\nYES win rate 70% entry 0.5")
        d = pipe.process("Closed the Ethereum merge market, out at 0.7")
        assert not d.closed
        assert "no open position matched" in d.note
        assert storage.count_open_positions() == 1   # untouched


def test_exit_without_price_leaves_position_open_with_hint():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _pipeline(tmp)
        pipe.process("Market: Will BTC close above 70k?\nYES win rate 70% entry 0.5")
        d = pipe.process("Closing the BTC close above 70k market")  # no price
        assert not d.closed
        assert "/resolve" in d.note
        assert storage.count_open_positions() == 1
