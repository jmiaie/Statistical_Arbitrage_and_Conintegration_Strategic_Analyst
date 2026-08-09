import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.config import ExecutionConfig
from copytradebot.slippage import SlippageModel
from copytradebot.marketdata import StaticData, MarketRef
from copytradebot.storage import Storage
from copytradebot.executors.realistic_paper import RealisticPaperExecutor
from copytradebot.parser import parse_alert, enrich


# ---- slippage math -------------------------------------------------------- #
def test_flat_slippage_buy():
    m = SlippageModel(slippage_bps=100, fee_bps=0)  # +1%
    fill = m.buy_at_price(stake=100, ref_price=0.50)
    # effective price 0.505 -> shares = 100/0.505
    assert abs(fill.effective_price - 0.505) < 1e-6
    assert abs(fill.shares - (100 / 0.505)) < 1e-6
    assert abs(fill.slippage_bps - 100) < 1.0


def test_fee_reduces_shares():
    m = SlippageModel(slippage_bps=0, fee_bps=200)  # 2% fee
    fill = m.buy_at_price(stake=100, ref_price=0.50)
    # spend after fee = 98 at price 0.50 -> 196 shares
    assert abs(fill.shares - 196.0) < 1e-6
    assert fill.filled_stake == 100  # total cash out unchanged


def test_book_walk_size_aware():
    m = SlippageModel(slippage_bps=0, fee_bps=0)
    asks = [(0.50, 100), (0.55, 100), (0.60, 100)]  # shares per level
    # Spend $100: take 100 shares @0.50 = $50, then $50 more @0.55 = 90.9 shares
    fill = m.buy_book(stake=100, asks=asks)
    expected_shares = 100 + (50 / 0.55)
    assert abs(fill.shares - expected_shares) < 1e-6
    # blended effective price between 0.50 and 0.55
    assert 0.50 < fill.effective_price < 0.55


def test_book_thin_leaves_unfilled():
    m = SlippageModel()
    asks = [(0.50, 10)]  # only $5 of liquidity
    fill = m.buy_book(stake=100, asks=asks)
    assert fill.shares == 10
    assert abs(fill.unfilled_stake - 95.0) < 1e-6


# ---- realistic executor with live data ------------------------------------ #
def _market():
    return MarketRef(
        market_id="m1",
        question="Will BTC close above 70k in June?",
        outcomes=["Yes", "No"],
        token_ids=["tokYES", "tokNO"],
    )


def _provider(asks=None, prices=None, market=None):
    return StaticData(markets=[market or _market()],
                      books={"tokYES": asks or []},
                      prices=prices or {})


def _exec(tmp, provider, fill_model="book", **kw):
    storage = Storage(os.path.join(tmp, "r.db"))
    cfg = ExecutionConfig(data_source="polymarket", fill_model=fill_model, **kw)
    sm = SlippageModel(slippage_bps=cfg.slippage_bps, fee_bps=cfg.fee_bps)
    return RealisticPaperExecutor(storage, provider, sm, cfg), storage


def test_book_fill_records_effective_price_and_shares():
    with tempfile.TemporaryDirectory() as tmp:
        prov = _provider(asks=[(0.40, 100), (0.45, 200)])
        ex, storage = _exec(tmp, prov, fill_model="book", slippage_bps=0)
        sig = enrich(parse_alert("Market: BTC above 70k? YES win rate 70% entry 0.40"))
        pos = ex.place(sig, stake=60, signal_id=1)
        # $40 @0.40 = 100 shares, $20 @0.45 = 44.4 shares
        assert abs(pos.shares - (100 + 20 / 0.45)) < 1e-4
        assert pos.meta["token_id"] == "tokYES"
        assert pos.meta["fill_model"] == "book"
        assert storage.count_open_positions() == 1


def test_mid_model_uses_live_price_over_alert():
    with tempfile.TemporaryDirectory() as tmp:
        prov = _provider(prices={"tokYES": 0.60})  # live price differs from alert
        ex, storage = _exec(tmp, prov, fill_model="mid", slippage_bps=0)
        sig = enrich(parse_alert("Market: BTC above 70k? YES entry 0.40 win rate 70%"))
        pos = ex.place(sig, stake=60, signal_id=1)
        # filled at live 0.60, not the alert's 0.40
        assert abs(pos.entry_price - 0.60) < 1e-6
        assert pos.meta["alert_price"] == 0.40


def test_degrades_when_no_market():
    with tempfile.TemporaryDirectory() as tmp:
        prov = StaticData(markets=[])  # nothing matches
        ex, storage = _exec(tmp, prov, fill_model="book")
        sig = enrich(parse_alert("Market: Unknown thing YES entry 0.30 win rate 70%"))
        pos = ex.place(sig, stake=50, signal_id=1)
        assert pos.meta.get("degraded") == "no-live-market"
        assert abs(pos.entry_price - 0.30 * 1.005) < 1e-3  # alert price + 50bps


def test_auto_settle_on_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        prov = _provider(asks=[(0.50, 1000)])
        ex, storage = _exec(tmp, prov, fill_model="book", slippage_bps=0)
        sig = enrich(parse_alert("Market: BTC above 70k? YES entry 0.50 win rate 70%"))
        ex.place(sig, stake=50, signal_id=1)  # 100 shares @0.50
        # Market resolves YES (index 0 wins).
        resolved = MarketRef("m1", "Will BTC close above 70k in June?",
                             ["Yes", "No"], ["tokYES", "tokNO"],
                             closed=True, outcome_prices=[1.0, 0.0])
        prov.markets["m1"] = resolved
        results = ex.settle_resolved()
        assert len(results) == 1
        _, status, pnl = results[0]
        assert status == "won"
        assert abs(pnl - 50.0) < 1e-6   # 100 shares pay $100, cost $50
        assert storage.stats()["win_rate"] == 1.0


def test_mark_to_market_unrealized():
    with tempfile.TemporaryDirectory() as tmp:
        prov = _provider(asks=[(0.50, 1000)], prices={"tokYES": 0.70})
        ex, storage = _exec(tmp, prov, fill_model="book", slippage_bps=0)
        sig = enrich(parse_alert("Market: BTC above 70k? YES entry 0.50 win rate 70%"))
        ex.place(sig, stake=50, signal_id=1)  # 100 shares, cost 50
        rows = ex.mark_to_market()
        assert len(rows) == 1
        # 100 shares * 0.70 - 50 = 20
        assert abs(rows[0]["unrealized_pnl"] - 20.0) < 1e-6
