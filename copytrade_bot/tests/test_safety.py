"""Tests for the safety-critical live-execution guards and update dedup.

These cover the P0 fixes: the live Polymarket executor must refuse to guess a
market/side/price rather than place a wrong real-money order, and the bot must
not act on the same Telegram update twice.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.executors.base import ExecutionError
from copytradebot.executors.polymarket import select_market
from copytradebot.storage import Storage
from copytradebot.config import (StrategyConfig, Settings, FilterConfig,
                                 SizingConfig, RiskConfig)
from copytradebot.pipeline import Pipeline


# ---- market selection: refuse rather than guess --------------------------- #
def _mkt(question, i=0):
    return {"question": question, "clobTokenIds": [f"y{i}", f"n{i}"],
            "outcomes": ["Yes", "No"]}


def test_select_market_picks_clear_winner():
    markets = [
        _mkt("Will the Fed cut rates in June 2026?", 0),
        _mkt("Will Bitcoin close above 100k in 2026?", 1),
    ]
    chosen = select_market(markets, "Fed cut rates June")
    assert "Fed" in chosen["question"]


def test_select_market_refuses_weak_match():
    markets = [_mkt("Will Bitcoin close above 100k in 2026?")]
    # Only "bitcoin" overlaps -> below the minimum shared-word threshold.
    with pytest.raises(ExecutionError):
        select_market(markets, "bitcoin")


def test_select_market_refuses_ambiguous_match():
    markets = [
        _mkt("Will the Lakers win the title this year?", 0),
        _mkt("Will the Lakers win the division this year?", 1),
    ]
    # Both share the same words with the query -> ambiguous, must refuse.
    with pytest.raises(ExecutionError):
        select_market(markets, "Lakers win this year")


def test_select_market_refuses_when_empty():
    with pytest.raises(ExecutionError):
        select_market([], "anything at all")


def test_select_market_refuses_query_without_usable_words():
    markets = [_mkt("Will the Fed cut rates in June 2026?")]
    with pytest.raises(ExecutionError):
        select_market(markets, "a b c")  # all tokens too short


# ---- idempotency / update dedup ------------------------------------------- #
def test_mark_seen_is_true_only_once():
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(os.path.join(tmp, "t.db"))
        assert storage.mark_seen("123:7") is True
        assert storage.mark_seen("123:7") is False   # edit/redelivery
        assert storage.mark_seen("123:8") is True     # different message


def test_mark_seen_persists_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.db")
        Storage(path).mark_seen("123:7")
        # A restart must still treat the update as already processed.
        assert Storage(path).mark_seen("123:7") is False


# ---- total-exposure cap --------------------------------------------------- #
def _exposure_pipeline(tmp, fraction):
    cfg = StrategyConfig(
        mode="paper",
        filters=FilterConfig(min_win_rate=0.6, min_ev=0.0),
        sizing=SizingConfig(mode="fraction", bankroll=1000, fraction=0.05,
                            max_position=200, min_position=1),
        risk=RiskConfig(max_exposure_fraction=fraction),
    )
    storage = Storage(os.path.join(tmp, "t.db"))
    settings = Settings(db_path=os.path.join(tmp, "t.db"))
    return Pipeline(cfg, settings, storage), storage


_ALERT = "Market: BTC up?\nYES win rate 70% EV +12% entry 0.5"


def test_exposure_cap_blocks_when_breached():
    # cap = 0.5 x 1000 = 500; each stake is 5% of 1000 = 50 -> 10 fit, 11th blocked.
    with tempfile.TemporaryDirectory() as tmp:
        pipe, storage = _exposure_pipeline(tmp, fraction=0.5)
        placed = sum(1 for _ in range(15) if pipe.process(_ALERT).placed)
        assert placed == 10
        assert storage.open_exposure() == 500.0
        last = pipe.process(_ALERT)
        assert not last.placed
        assert "exposure cap" in last.note


def test_exposure_cap_disabled_allows_all():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, _ = _exposure_pipeline(tmp, fraction=None)
        placed = sum(1 for _ in range(15) if pipe.process(_ALERT).placed)
        assert placed == 15
