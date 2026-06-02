"""Runtime configuration: environment settings + adjustable filter/sizing config.

The filter, sizing and risk parameters live in a YAML file so they survive
restarts and can be edited by hand. They are also adjustable live from
Telegram via ``/set`` (see ``bot.py``), which writes back to the same file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_FILTERS_PATH = Path(__file__).resolve().parent.parent / "config" / "filters.yaml"


# --------------------------------------------------------------------------- #
# Adjustable strategy configuration (persisted to YAML)
# --------------------------------------------------------------------------- #
@dataclass
class FilterConfig:
    """Criteria an alert must satisfy to be traded. ``None`` disables a bound."""

    min_win_rate: Optional[float] = 0.55      # fraction 0-1
    min_ev: Optional[float] = 0.0             # percent
    min_roi: Optional[float] = None           # percent
    min_expected_return: Optional[float] = None  # percent
    min_entry_price: Optional[float] = None
    max_entry_price: Optional[float] = None
    min_size: Optional[float] = None
    max_size: Optional[float] = None
    require_fields: list[str] = field(default_factory=lambda: ["win_rate"])
    blocked_keywords: list[str] = field(default_factory=list)
    allowed_sources: list[str] = field(default_factory=list)   # empty = all
    blocked_sources: list[str] = field(default_factory=list)


@dataclass
class SizingConfig:
    """How much to stake on a passing signal."""

    mode: str = "fraction"        # fixed | fraction | kelly
    fixed_amount: float = 50.0
    bankroll: float = 1000.0
    fraction: float = 0.02        # fraction of bankroll for mode=fraction
    kelly_fraction: float = 0.5   # fractional Kelly multiplier
    max_position: float = 200.0
    min_position: float = 1.0


@dataclass
class RiskConfig:
    max_open_positions: int = 25
    max_daily_loss: float = 250.0


@dataclass
class StrategyConfig:
    enabled: bool = True
    mode: str = "paper"           # paper | live
    dry_run: bool = False         # if True, never actually place (even paper)
    filters: FilterConfig = field(default_factory=FilterConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # ---- persistence ---------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "filters": asdict(self.filters),
            "sizing": asdict(self.sizing),
            "risk": asdict(self.risk),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        d = d or {}
        return cls(
            enabled=d.get("enabled", True),
            mode=d.get("mode", "paper"),
            dry_run=d.get("dry_run", False),
            filters=FilterConfig(**(d.get("filters") or {})),
            sizing=SizingConfig(**(d.get("sizing") or {})),
            risk=RiskConfig(**(d.get("risk") or {})),
        )

    def save(self, path: os.PathLike | str = DEFAULT_FILTERS_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def load(cls, path: os.PathLike | str = DEFAULT_FILTERS_PATH) -> "StrategyConfig":
        path = Path(path)
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})


# Keys that are settable via ``/set <key> <value>`` and their coercion type.
SETTABLE_FIELDS: dict[str, tuple[str, type]] = {
    # filters
    "min_win_rate": ("filters", float),
    "min_ev": ("filters", float),
    "min_roi": ("filters", float),
    "min_expected_return": ("filters", float),
    "min_entry_price": ("filters", float),
    "max_entry_price": ("filters", float),
    "min_size": ("filters", float),
    "max_size": ("filters", float),
    # sizing
    "mode": ("sizing", str),                 # sizing.mode (fixed|fraction|kelly)
    "fixed_amount": ("sizing", float),
    "bankroll": ("sizing", float),
    "fraction": ("sizing", float),
    "kelly_fraction": ("sizing", float),
    "max_position": ("sizing", float),
    "min_position": ("sizing", float),
    # risk
    "max_open_positions": ("risk", int),
    "max_daily_loss": ("risk", float),
}


def coerce(value: str, typ: type):
    """Coerce a string command argument, treating 'none'/'off' as None."""
    if value.strip().lower() in {"none", "null", "off", "-"}:
        return None
    if typ is float:
        return float(value)
    if typ is int:
        return int(float(value))
    return value


# --------------------------------------------------------------------------- #
# Environment / secrets (never persisted to YAML)
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    telegram_token: str = ""
    # Chats whose posts are treated as incoming alerts.
    source_chat_ids: list[int] = field(default_factory=list)
    # Chats allowed to issue admin commands & receive notifications.
    admin_chat_ids: list[int] = field(default_factory=list)
    notify_chat_id: Optional[int] = None
    db_path: str = "copytrade.db"
    filters_path: str = str(DEFAULT_FILTERS_PATH)
    poll_timeout: int = 30

    # Live trading (Polymarket) — only used when StrategyConfig.mode == "live".
    live_trading: bool = False
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polymarket_funder: str = ""
    polymarket_host: str = "https://clob.polymarket.com"

    @classmethod
    def from_env(cls) -> "Settings":
        def ids(name: str) -> list[int]:
            raw = os.getenv(name, "").strip()
            return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]

        notify = os.getenv("TELEGRAM_NOTIFY_CHAT_ID", "").strip()
        return cls(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            source_chat_ids=ids("TELEGRAM_SOURCE_CHAT_IDS"),
            admin_chat_ids=ids("TELEGRAM_ADMIN_CHAT_IDS"),
            notify_chat_id=int(notify) if notify else None,
            db_path=os.getenv("COPYTRADE_DB", "copytrade.db"),
            filters_path=os.getenv("COPYTRADE_FILTERS", str(DEFAULT_FILTERS_PATH)),
            poll_timeout=int(os.getenv("TELEGRAM_POLL_TIMEOUT", "30")),
            live_trading=os.getenv("LIVE_TRADING", "").lower() in {"1", "true", "yes"},
            polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            polymarket_api_key=os.getenv("POLYMARKET_API_KEY", ""),
            polymarket_api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
            polymarket_api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
            polymarket_funder=os.getenv("POLYMARKET_FUNDER", ""),
            polymarket_host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
        )
