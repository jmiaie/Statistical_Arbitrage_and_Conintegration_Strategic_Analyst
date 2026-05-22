"""End-to-end research script — pulls real data, runs the full pipeline, writes artefacts.

Reproducible:
    python research/main_backtest.py --pair KO PEP --start 2018-01-01 --end 2024-12-31

Artefacts:
    results/equity_curve.png
    results/factor_attribution.csv
    results/wfo_folds.csv
    results/kpis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantpairs.attribution import FamaFrenchAttribution
from quantpairs.backtest import run_backtest
from quantpairs.cointegration import engle_granger_test
from quantpairs.wfo import walk_forward

RESULTS = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, default=["KO", "PEP"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    args = parser.parse_args()

    from quantpairs.data import fetch_fama_french, fetch_prices

    RESULTS.mkdir(exist_ok=True)
    y_t, x_t = args.pair
    prices = fetch_prices([y_t, x_t], args.start, args.end)
    log_y, log_x = np.log(prices[y_t]), np.log(prices[x_t])

    print(f"\n=== Cointegration: {y_t} ~ {x_t} ===")
    coint = engle_granger_test(log_y, log_x)
    print(coint)

    print("\n=== In-sample backtest ===")
    bt = run_backtest(log_y, log_x, cost_bps=args.cost_bps)
    print(bt.summary())

    print("\n=== Walk-forward OOS ===")
    wfo = walk_forward(log_y, log_x, train_size=504, test_size=126, cost_bps=args.cost_bps)
    print(json.dumps(wfo.oos_kpis, indent=2))

    print("\n=== Fama-French attribution (OOS) ===")
    ff = fetch_fama_french(args.start, args.end)
    attribution = FamaFrenchAttribution.fit(wfo.oos_returns, ff, risk_free=ff["RF"])
    print(attribution.to_frame().to_string(index=False))

    # Persist artefacts
    fig, ax = plt.subplots(figsize=(11, 5))
    equity = (1 + wfo.oos_returns).cumprod()
    ax.plot(equity.index, equity.values, color="#0b6efb", linewidth=1.5)
    ax.set_title(f"Out-of-sample Equity Curve — {y_t} ~ {x_t}")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.grid(True, color="#e6e6e6")
    fig.tight_layout()
    fig.savefig(RESULTS / "equity_curve.png", dpi=160)
    plt.close(fig)

    attribution.to_frame().to_csv(RESULTS / "factor_attribution.csv", index=False)
    wfo.fold_kpis.to_csv(RESULTS / "wfo_folds.csv", index=False)
    payload = {
        "pair": list(args.pair),
        "window": [args.start, args.end],
        "cost_bps": args.cost_bps,
        "cointegration": {
            "eg_pvalue": coint.pvalue,
            "adf_pvalue": coint.adf_pvalue,
            "hedge_ratio": coint.hedge_ratio,
            "half_life_days": coint.half_life,
        },
        "in_sample_kpis": bt.kpis,
        "oos_kpis": wfo.oos_kpis,
        "ff_attribution": {
            "alpha_annual": attribution.alpha_annual,
            "alpha_tstat": attribution.alpha_tstat,
            "beta_mkt": attribution.beta_mkt,
            "beta_smb": attribution.beta_smb,
            "beta_hml": attribution.beta_hml,
            "r_squared": attribution.r_squared,
        },
    }
    (RESULTS / "kpis.json").write_text(json.dumps(payload, indent=2))
    print(f"\nArtefacts written to {RESULTS}/")


if __name__ == "__main__":
    main()
