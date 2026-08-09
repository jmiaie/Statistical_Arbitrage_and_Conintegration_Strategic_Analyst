import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.config import FilterConfig
from copytradebot.filters import FilterEngine
from copytradebot.parser import parse_alert, enrich


def _eval(text, **cfg):
    sig = enrich(parse_alert(text))
    return FilterEngine(FilterConfig(**cfg)).evaluate(sig)


def test_passes_when_above_thresholds():
    r = _eval("Market: X\nYES win rate 70% EV +15% entry 0.4",
              min_win_rate=0.6, min_ev=10.0)
    assert r.passed, r.reasons


def test_fails_low_win_rate():
    r = _eval("Market: X\nwin rate 50% EV +20%", min_win_rate=0.6)
    assert not r.passed
    assert any("win rate" in x for x in r.reasons)


def test_required_field_missing():
    r = _eval("no metrics here", min_win_rate=None,
              require_fields=["win_rate"])
    assert not r.passed
    assert any("required field 'win_rate'" in x for x in r.reasons)


def test_entry_price_bounds():
    r = _eval("Market: X\nwin rate 70% entry 0.95",
              min_win_rate=0.5, max_entry_price=0.8)
    assert not r.passed
    assert any("entry price" in x for x in r.reasons)


def test_blocked_keyword():
    r = _eval("Market: X\nwin rate 90% (paper only test)",
              min_win_rate=0.5, blocked_keywords=["paper only"])
    assert not r.passed


def test_blocked_source():
    sig = enrich(parse_alert("Market: X\nwin rate 90%", source="spam"))
    r = FilterEngine(FilterConfig(min_win_rate=0.5,
                                  blocked_sources=["spam"])).evaluate(sig)
    assert not r.passed
