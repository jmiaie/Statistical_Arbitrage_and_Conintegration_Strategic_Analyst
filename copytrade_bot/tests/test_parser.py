import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.parser import parse_alert, enrich
from copytradebot.models import Side


SAMPLE = """🔥 NEW PLAY 🔥
Market: Will BTC close above $70k in June?
Side: YES
Entry: 0.42
Win rate ~ 68%
EV: +14%   ROI potential: 30%
Suggested size: $75
"""


def test_parses_all_fields():
    s = parse_alert(SAMPLE, source="chan")
    assert s.market.startswith("Will BTC")
    assert s.side is Side.YES
    assert abs(s.entry_price - 0.42) < 1e-9
    assert abs(s.win_rate - 0.68) < 1e-9
    assert s.ev == 14.0
    assert s.roi == 30.0
    assert s.size == 75.0
    assert s.source == "chan"


def test_win_rate_normalisation_fraction_form():
    s = parse_alert("confidence 0.72 on this one")
    assert abs(s.win_rate - 0.72) < 1e-9


def test_win_rate_percent_form():
    s = parse_alert("Hit rate: 80%")
    assert abs(s.win_rate - 0.80) < 1e-9


def test_ticker_detection():
    s = parse_alert("Buy $AAPL entry 195.5 win rate 60%")
    assert s.market == "AAPL"
    assert s.side is Side.BUY
    assert s.entry_price == 195.5


def test_missing_fields_are_none():
    s = parse_alert("just some chatter, no signal here")
    assert s.win_rate is None
    assert s.ev is None
    assert s.entry_price is None


def test_enrich_computes_ev_for_prediction_market():
    # 60% win prob bought at 0.50 implied prob -> positive EV.
    s = parse_alert("Market: Coin flip\nYES win rate 60% entry 0.50")
    s = enrich(s)
    # EV = 0.6*(1/0.5 - 1) - 0.4 = 0.6*1 - 0.4 = 0.2 -> 20%
    assert abs(s.ev - 20.0) < 1e-6


def test_handles_empty_text():
    s = parse_alert("")
    assert s.win_rate is None
    assert s.market is None
