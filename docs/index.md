# quant-pairs-lab

Dynamic statistical arbitrage with Kalman-filtered hedge ratios, Fama–French attribution, and a square-root TCA model.

This site is the rendered version of the repository documentation. For the source, see [GitHub](https://github.com/jmiaie/quant-pairs-lab).

## Quick links

- [Methodology](methodology.md) — the full quantitative write-up.
- [Results](results.md) — committed performance artefacts.
- [API Reference](reference.md) — auto-generated docstring docs.

## Why this repo exists

Most public pairs-trading repositories show in-sample success on toy pairs without TCA. This one explicitly stresses:

- Adaptive hedge ratios (Kalman, not OLS)
- Out-of-sample walk-forward, not "fit-and-show"
- Net-of-cost Sharpe with a published-style impact model
- Factor attribution to confirm market-neutrality
