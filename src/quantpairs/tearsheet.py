"""Generate a professional HTML tear-sheet for a backtest run.

HTML (not PDF) so it renders cleanly on GitHub, works in any browser, and
avoids the ReportLab dependency. Embed an equity curve and a drawdown
panel as base64 PNGs so the file is self-contained.
"""

from __future__ import annotations

import base64
import io
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quantpairs.backtest import BacktestResult


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_and_drawdown(equity: pd.Series) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(equity.index, equity.values, color="#0b6efb", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.grid(True, color="#e6e6e6")
    fig.tight_layout()
    eq = _fig_to_b64(fig)

    dd = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax.set_title("Drawdown")
    ax.grid(True, color="#e6e6e6")
    fig.tight_layout()
    return eq, _fig_to_b64(fig)


def _kpi_rows(kpis: dict[str, float]) -> str:
    fmt: dict[str, str] = {
        "sharpe_net": "{:.2f}",
        "sharpe_gross": "{:.2f}",
        "annual_return": "{:.2%}",
        "annual_vol": "{:.2%}",
        "max_drawdown": "{:.2%}",
        "hit_rate": "{:.2%}",
        "avg_turnover": "{:.3f}",
        "n_trades": "{:.0f}",
    }
    rows = "".join(
        f"<tr><td>{k}</td><td>{fmt.get(k, '{:.4f}').format(v)}</td></tr>"
        for k, v in kpis.items()
    )
    return rows


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 980px; margin: 2rem auto; color: #1a1a1a; }}
  h1 {{ color: #0b1d3a; margin-bottom: 0.2rem; }}
  .meta {{ color: #555; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 60%; margin-bottom: 1.5rem; }}
  th, td {{ padding: 6px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
  th {{ background: #f4f7ff; }}
  img {{ width: 100%; border: 1px solid #eaeaea; border-radius: 6px; }}
  .footer {{ font-size: 0.85rem; color: #888; margin-top: 2rem; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Generated {generated} · quant-pairs-lab v{version}</div>

  <h2>Key Performance Indicators</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{kpi_rows}</tbody>
  </table>

  <h2>Equity Curve</h2>
  <img src="data:image/png;base64,{eq_b64}"/>

  <h2>Drawdown</h2>
  <img src="data:image/png;base64,{dd_b64}"/>

  <div class="footer">
    Numbers are net of {cost_bps:.1f} bps per-trade transaction cost.
    See <code>RESULTS.md</code> and <code>docs/methodology.md</code> for assumptions.
  </div>
</body></html>"""


def write_tearsheet(
    result: BacktestResult,
    output_path: str | Path = "results/tearsheet.html",
    title: str = "Pair Backtest — Tear Sheet",
    cost_bps: float = 2.0,
) -> Path:
    """Render a self-contained HTML tear-sheet."""
    from quantpairs import __version__

    eq_b64, dd_b64 = _equity_and_drawdown(result.equity_curve)
    html = _HTML.format(
        title=title,
        generated=date.today().isoformat(),
        version=__version__,
        kpi_rows=_kpi_rows(result.kpis),
        eq_b64=eq_b64,
        dd_b64=dd_b64,
        cost_bps=cost_bps,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
