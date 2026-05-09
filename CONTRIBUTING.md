# Contributing

This is a portfolio repository maintained by the author. The codebase is licensed under a proprietary license (see [`LICENSE`](./LICENSE)) and is **not open for external pull requests**.

If you would like to discuss the methodology, suggest references, or propose a collaboration:

- **Issues:** opening a GitHub issue for typos, broken links, or methodological questions is welcomed.
- **Email:** jmilam.emba@gmail.com for collaboration, licensing, or hiring discussions.

## Style notes (for the author's own reference)

- Python 3.10+, type hints where they aid clarity.
- Notebooks are research artefacts; production-quality logic lives under a `src/` package.
- Reproducibility: every backtest seeds RNGs and pins the data vintage where possible.
- All performance numbers must be reported **net of modelled transaction costs**.
