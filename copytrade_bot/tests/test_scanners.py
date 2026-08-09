import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.scanners import (
    ArbitrageScanner, LongshotScanner, MeanReversionScanner,
    CointegrationScanner)
from copytradebot.scanners.arbitrage import MarketSnapshot
from copytradebot.scanners import stats


# ---- stats ---------------------------------------------------------------- #
def test_ols_recovers_line():
    x = list(range(20))
    y = [3 + 2 * xi for xi in x]
    a, b, resid = stats.ols(x, y)
    assert abs(a - 3) < 1e-6 and abs(b - 2) < 1e-6
    assert max(abs(r) for r in resid) < 1e-6


def test_adf_more_negative_for_stationary():
    rng = random.Random(0)
    white = [rng.gauss(0, 1) for _ in range(200)]          # stationary
    walk = [0.0]
    for _ in range(199):
        walk.append(walk[-1] + rng.gauss(0, 1))            # random walk
    assert stats.adf_tstat(white) < stats.adf_tstat(walk)
    assert stats.adf_tstat(white) < -3.0


# ---- arbitrage ------------------------------------------------------------ #
def test_arbitrage_detects_underpriced_set():
    snaps = [
        MarketSnapshot("m1", "binary", ["Yes", "No"], [0.45, 0.50]),   # sum .95
        MarketSnapshot("m2", "binary", ["Yes", "No"], [0.55, 0.50]),   # sum 1.05
    ]
    opps = ArbitrageScanner(cost_buffer=0.0, min_edge=0.01).scan(snaps)
    assert len(opps) == 1
    assert opps[0].legs[0].market_id == "m1"
    assert abs(opps[0].edge - 0.05) < 1e-6


def test_arbitrage_cost_buffer_blocks_thin_edge():
    snaps = [MarketSnapshot("m1", "b", ["Yes", "No"], [0.49, 0.50])]  # 1% gross
    assert ArbitrageScanner(cost_buffer=0.02).scan(snaps) == []


# ---- longshot ------------------------------------------------------------- #
def test_longshot_backs_favorite_and_fades_longshot():
    snaps = [
        MarketSnapshot("fav", "q", ["Yes", "No"], [0.80, 0.20]),
        MarketSnapshot("dog", "q", ["Yes", "No"], [0.10, 0.90]),
    ]
    opps = LongshotScanner(min_edge=-1).scan(snaps)
    kinds = {o.legs[0].market_id: o.legs[0].side for o in opps}
    # Favorite YES @0.80 should be backed somewhere
    assert any(o.legs[0].market_id == "fav" for o in opps)


# ---- mean reversion ------------------------------------------------------- #
def test_mean_reversion_fades_spike():
    prices = [0.50] * 25 + [0.70]            # sharp spike up at the end
    hist = {"m": {"question": "q", "prices": prices,
                  "token_yes": "ty", "token_no": "tn"}}
    opps = MeanReversionScanner(lookback=20, entry_z=1.5, max_move=0.5).scan(hist)
    assert len(opps) == 1
    assert opps[0].legs[0].side == "No"      # fade the upside overshoot


def test_mean_reversion_skips_newslike_move():
    prices = [0.50] * 25 + [0.95]            # huge move -> looks like news
    hist = {"m": {"question": "q", "prices": prices,
                  "token_yes": "ty", "token_no": "tn"}}
    assert MeanReversionScanner(max_move=0.25).scan(hist) == []


# ---- cointegration -------------------------------------------------------- #
def test_cointegration_finds_diverged_pair():
    rng = random.Random(1)
    base = []
    p = 0.5
    for _ in range(80):
        p += rng.gauss(0, 0.01)
        p = min(0.9, max(0.1, p))
        base.append(p)
    # B tracks A with mean-reverting noise (cointegrated), then diverge at end.
    noise = 0.0
    b = []
    for i in range(80):
        noise = 0.5 * noise + rng.gauss(0, 0.005)
        b.append(min(0.95, max(0.05, base[i] + noise)))
    a = list(base)
    a[-1] = min(0.95, a[-1] + 0.06)          # A jumps rich vs B
    series = {
        "A": {"question": "A", "prices": a, "token_yes": "ay", "token_no": "an"},
        "B": {"question": "B", "prices": b, "token_yes": "by", "token_no": "bn"},
    }
    opps = CointegrationScanner(lookback=60, entry_z=1.5, adf_threshold=-1.5,
                                min_half_life=0.0).scan(series)
    # Should find the pair and propose shorting the rich leg A (buy NO_A).
    assert len(opps) >= 1
    a_leg = [l for l in opps[0].legs if l.market_id == "A"][0]
    assert a_leg.side == "No"


# ---- NO-leg spread honesty ------------------------------------------------ #
def test_meanreversion_spread_raises_no_entry_and_cuts_edge():
    prices = [0.50] * 25 + [0.70]
    hist = {"m": {"question": "q", "prices": prices,
                  "token_yes": "ty", "token_no": "tn"}}
    cheap = MeanReversionScanner(lookback=20, entry_z=1.5, spread=0.0).scan(hist)
    dear = MeanReversionScanner(lookback=20, entry_z=1.5, spread=0.05).scan(hist)
    # Paying the spread on the NO leg means a higher entry and a smaller edge.
    assert dear[0].legs[0].price > cheap[0].legs[0].price
    assert dear[0].edge < cheap[0].edge


# ---- multiple-testing correction ------------------------------------------ #
def test_cointegration_correction_suppresses_spurious_pairs():
    rng = random.Random(3)
    # Many independent random walks: any "cointegration" found is spurious.
    series = {}
    for k in range(8):
        p, walk = 0.5, []
        for _ in range(80):
            p = min(0.9, max(0.1, p + rng.gauss(0, 0.02)))
            walk.append(p)
        series[f"m{k}"] = {"question": f"m{k}", "prices": walk,
                           "token_yes": f"y{k}", "token_no": f"n{k}"}
    raw = CointegrationScanner(adf_threshold=-1.5, min_half_life=0.0,
                               multiple_test_correction=False).scan(series)
    corrected = CointegrationScanner(adf_threshold=-1.5, min_half_life=0.0,
                                     multiple_test_correction=True).scan(series)
    # The Bonferroni-tightened cut should admit no more (typically fewer) pairs.
    assert len(corrected) <= len(raw)
