"""Research-agent loop: given a pair, run the pipeline and write a markdown report.

This is a thin orchestrator that:
    1. Fetches prices.
    2. Runs Engle-Granger + Kalman backtest.
    3. Calls Claude to write a one-page research memo summarising the result.

Demonstrates the Claude API + a real, useful tool-use pattern.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

from quantpairs.backtest import run_backtest
from quantpairs.cointegration import engle_granger_test

DEFAULT_MODEL = "claude-opus-4-7"

_MEMO_SYSTEM = """You are a quant researcher writing a one-page memo for a
portfolio manager. Be precise, cite the numbers verbatim from the data block,
and explicitly call out limitations (cost assumptions, in-sample bias). Use
markdown with these sections: Summary, Statistical Evidence, Economic
Rationale, Performance, Risks. Keep it under 400 words."""


def _format_data_block(stats: dict[str, float], pair: tuple[str, str]) -> str:
    y, x = pair
    return dedent(
        f"""
        PAIR: {y} ~ {x}

        Cointegration:
          Engle-Granger p-value: {stats['eg_p']:.4f}
          ADF on residuals     : {stats['adf_p']:.4f}
          Static hedge ratio   : {stats['beta']:.4f}
          Half-life (days)     : {stats['half_life']:.1f}

        Backtest (Kalman + 2.0 bps cost):
          Net Sharpe           : {stats['sharpe_net']:.2f}
          Gross Sharpe         : {stats['sharpe_gross']:.2f}
          Annual return        : {stats['annual_return']:.2%}
          Annual volatility    : {stats['annual_vol']:.2%}
          Max drawdown         : {stats['max_dd']:.2%}
          Hit rate             : {stats['hit_rate']:.2%}
          Trades               : {stats['n_trades']:.0f}
        """
    ).strip()


def write_research_memo(
    pair: tuple[str, str],
    start: str = "2018-01-01",
    end: str | None = None,
    output_path: str | Path = "results/memo.md",
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> Path:
    """Run the pipeline on `pair` and ask Claude to author a research memo."""
    import anthropic
    import numpy as np

    from quantpairs.data import fetch_prices

    y_ticker, x_ticker = pair
    prices = fetch_prices([y_ticker, x_ticker], start=start, end=end)
    log_y, log_x = np.log(prices[y_ticker]), np.log(prices[x_ticker])

    coint = engle_granger_test(log_y, log_x)
    bt = run_backtest(log_y, log_x)

    stats = {
        "eg_p": coint.pvalue,
        "adf_p": coint.adf_pvalue,
        "beta": coint.hedge_ratio,
        "half_life": coint.half_life,
        "sharpe_net": bt.kpis["sharpe_net"],
        "sharpe_gross": bt.kpis["sharpe_gross"],
        "annual_return": bt.kpis["annual_return"],
        "annual_vol": bt.kpis["annual_vol"],
        "max_dd": bt.kpis["max_drawdown"],
        "hit_rate": bt.kpis["hit_rate"],
        "n_trades": bt.kpis["n_trades"],
    }
    data_block = _format_data_block(stats, pair)

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=[
            {"type": "text", "text": _MEMO_SYSTEM, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[
            {
                "role": "user",
                "content": f"Write a research memo from this data:\n\n{data_block}",
            }
        ],
    )
    memo = "".join(block.text for block in response.content if block.type == "text")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(memo, encoding="utf-8")
    return out


def cli() -> None:
    """`quantpairs-research KO PEP --start 2018-01-01`."""
    import argparse

    parser = argparse.ArgumentParser(description="Pair research agent.")
    parser.add_argument("y", help="Asset Y ticker")
    parser.add_argument("x", help="Asset X ticker")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="results/memo.md")
    args = parser.parse_args()
    path = write_research_memo(
        (args.y, args.x), start=args.start, end=args.end, output_path=args.output
    )
    print(f"Memo written to {path}")


if __name__ == "__main__":  # pragma: no cover
    cli()
