"""Slippage & fee modelling for realistic fills.

Prediction-market convention: a "share" of an outcome token costs its price
``p`` in (0, 1) and pays out $1 if that outcome resolves true. Buying spends
cash and acquires shares; the *effective price* is total cash out (including
fees) divided by shares acquired.

Two fill models are supported:

* ``buy_at_price`` — apply a flat slippage pad (bps) to a reference price.
  Cheap, no order book needed. Good when you only have a midpoint/last price.
* ``buy_book``     — walk the live order book so larger stakes pay up through
  the levels (true size-aware impact). The realistic option.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp_price(p: float) -> float:
    return max(0.001, min(0.999, p))


@dataclass
class Fill:
    shares: float            # contracts acquired
    filled_stake: float      # total cash out, INCLUDING fees
    effective_price: float   # filled_stake / shares
    unfilled_stake: float    # cash that couldn't be filled (book too thin)
    fee_paid: float
    reference_price: float   # the pre-slippage price we compared against

    @property
    def slippage_bps(self) -> float:
        if not self.reference_price or not self.effective_price:
            return 0.0
        return (self.effective_price / self.reference_price - 1.0) * 10_000


@dataclass
class SlippageModel:
    slippage_bps: float = 50.0   # flat pad for the price-based model
    fee_bps: float = 0.0         # taker fee (Polymarket is 0 today; configurable)

    # ---- price-based (no book) ----------------------------------------- #
    def buy_at_price(self, stake: float, ref_price: float) -> Fill:
        ref = _clamp_price(ref_price)
        eff = _clamp_price(ref * (1 + self.slippage_bps / 10_000))
        fee = stake * self.fee_bps / 10_000
        spend = stake - fee
        shares = spend / eff if eff > 0 else 0.0
        eff_all_in = stake / shares if shares else 0.0
        return Fill(shares, stake, eff_all_in, 0.0, fee, ref)

    # ---- order-book walk ----------------------------------------------- #
    def buy_book(self, stake: float, asks: list[tuple[float, float]]) -> Fill:
        """``asks`` = ascending [(price, size_in_shares), ...]."""
        ref = _clamp_price(asks[0][0]) if asks else 0.5
        remaining = stake
        shares = 0.0
        spent = 0.0
        for price, size in sorted(asks, key=lambda x: x[0]):
            price = _clamp_price(price)
            level_cost = price * size
            if remaining <= level_cost:
                shares += remaining / price
                spent += remaining
                remaining = 0.0
                break
            shares += size
            spent += level_cost
            remaining -= level_cost

        fee = spent * self.fee_bps / 10_000
        filled = spent + fee
        eff = filled / shares if shares else 0.0
        return Fill(shares, filled, eff, remaining, fee, ref)

    # ---- dispatch ------------------------------------------------------- #
    def simulate_buy(self, stake: float, ref_price: float | None,
                     asks: list[tuple[float, float]] | None) -> Fill:
        if asks:
            return self.buy_book(stake, asks)
        return self.buy_at_price(stake, ref_price if ref_price else 0.5)
