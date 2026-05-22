"""Lightweight experiment tracking.

Writes a JSON manifest per run to `results/runs/<timestamp>__<tag>.json`
so backtests are reproducible and comparable without an external service
like MLflow or Weights & Biases. Each manifest captures: code version,
strategy hyperparameters, dataset window, full KPI dict, and a free-form
notes string.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    """Reproducibility metadata for a single backtest run."""

    tag: str
    params: dict[str, Any]
    kpis: dict[str, float]
    window: tuple[str, str] | None = None
    notes: str = ""
    code_version: str = ""
    git_commit: str = ""
    host: str = field(default_factory=socket.gethostname)
    python: str = field(default_factory=platform.python_version)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def log_run(
    tag: str,
    params: dict[str, Any],
    kpis: dict[str, float],
    notes: str = "",
    window: tuple[str, str] | None = None,
    output_dir: str | Path = "results/runs",
) -> Path:
    """Persist a run manifest and return its path."""
    from quantpairs import __version__

    manifest = RunManifest(
        tag=tag,
        params=params,
        kpis=kpis,
        window=window,
        notes=notes,
        code_version=__version__,
        git_commit=_git_commit(),
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}__{tag}.json"
    path.write_text(json.dumps(asdict(manifest), indent=2, default=str))
    return path


def list_runs(output_dir: str | Path = "results/runs") -> list[dict[str, Any]]:
    """Load all manifests from `output_dir`, newest first."""
    d = Path(output_dir)
    if not d.exists():
        return []
    runs = []
    for p in sorted(d.glob("*.json"), reverse=True):
        runs.append(json.loads(p.read_text()))
    return runs
