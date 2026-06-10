"""SQLite persistence for signals, positions and performance stats."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import Optional

from .models import Signal, Position, FilterResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,
    raw TEXT,
    market TEXT,
    side TEXT,
    win_rate REAL,
    ev REAL,
    roi REAL,
    expected_return REAL,
    entry_price REAL,
    size REAL,
    passed INTEGER NOT NULL,
    reasons TEXT
);
CREATE TABLE IF NOT EXISTS seen_updates (
    key TEXT PRIMARY KEY,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    ts REAL NOT NULL,
    venue TEXT,
    market TEXT,
    side TEXT,
    entry_price REAL,
    stake REAL,
    status TEXT NOT NULL DEFAULT 'open',
    pnl REAL NOT NULL DEFAULT 0,
    external_id TEXT,
    resolved_at REAL,
    shares REAL NOT NULL DEFAULT 0,
    meta TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);
"""

# Columns added after the initial release; applied idempotently on open so
# existing databases pick them up without a manual migration.
_MIGRATIONS = {
    "positions": {"shares": "REAL NOT NULL DEFAULT 0", "meta": "TEXT"},
}


class Storage:
    def __init__(self, path: str = "copytrade.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with closing(self.conn.cursor()) as cur:
            cur.executescript(_SCHEMA)
            self._migrate(cur)
        self.conn.commit()

    @staticmethod
    def _migrate(cur) -> None:
        for table, cols in _MIGRATIONS.items():
            existing = {row[1] for row in cur.execute(
                f"PRAGMA table_info({table})")}
            for name, decl in cols.items():
                if name not in existing:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def close(self) -> None:
        self.conn.close()

    # ---- idempotency ---------------------------------------------------- #
    def mark_seen(self, key: str) -> bool:
        """Record an update key; return True only the first time it's seen.

        Used to dedupe Telegram updates so an edited post (same chat+message id)
        or a redelivered update can't fire the pipeline — and a trade — twice.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO seen_updates (key, ts) VALUES (?, ?)",
            (key, time.time()),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- signals -------------------------------------------------------- #
    def record_signal(self, sig: Signal, result: FilterResult) -> int:
        cur = self.conn.execute(
            """INSERT INTO signals
               (ts, source, raw, market, side, win_rate, ev, roi,
                expected_return, entry_price, size, passed, reasons)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sig.received_at, sig.source, sig.raw_text, sig.market,
                sig.side.value, sig.win_rate, sig.ev, sig.roi,
                sig.expected_return, sig.entry_price, sig.size,
                1 if result.passed else 0, "; ".join(result.reasons),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_signals(self, limit: int = 10) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        ))

    # ---- positions ------------------------------------------------------ #
    def record_position(self, pos: Position) -> int:
        import json
        cur = self.conn.execute(
            """INSERT INTO positions
               (signal_id, ts, venue, market, side, entry_price, stake,
                status, pnl, external_id, resolved_at, shares, meta)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                pos.signal_id, pos.opened_at, pos.venue, pos.market, pos.side,
                pos.entry_price, pos.stake, pos.status, pos.pnl,
                pos.external_id, pos.resolved_at, pos.shares,
                json.dumps(pos.meta or {}),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def open_positions(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM positions WHERE status='open' ORDER BY id DESC"
        ))

    def open_exposure(self) -> float:
        """Total cash currently deployed across open positions."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(stake),0) AS s FROM positions "
            "WHERE status='open'"
        ).fetchone()
        return float(row["s"])

    def count_open_positions(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE status='open'"
        ).fetchone()
        return int(row["n"])

    def get_position(self, pos_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM positions WHERE id=?", (pos_id,)
        ).fetchone()

    def resolve_position(self, pos_id: int, status: str, pnl: float) -> bool:
        cur = self.conn.execute(
            "UPDATE positions SET status=?, pnl=?, resolved_at=? "
            "WHERE id=? AND status='open'",
            (status, pnl, time.time(), pos_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def realized_pnl_since(self, since_ts: float) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl),0) AS p FROM positions "
            "WHERE resolved_at IS NOT NULL AND resolved_at>=?",
            (since_ts,),
        ).fetchone()
        return float(row["p"])

    # ---- stats ---------------------------------------------------------- #
    def stats(self) -> dict:
        c = self.conn
        seen = c.execute("SELECT COUNT(*) n FROM signals").fetchone()["n"]
        passed = c.execute(
            "SELECT COUNT(*) n FROM signals WHERE passed=1"
        ).fetchone()["n"]
        settled = c.execute(
            "SELECT COUNT(*) n FROM positions WHERE status IN ('won','lost')"
        ).fetchone()["n"]
        won = c.execute(
            "SELECT COUNT(*) n FROM positions WHERE status='won'"
        ).fetchone()["n"]
        pnl = c.execute(
            "SELECT COALESCE(SUM(pnl),0) p FROM positions"
        ).fetchone()["p"]
        open_n = self.count_open_positions()
        return {
            "signals_seen": int(seen),
            "signals_passed": int(passed),
            "open_positions": int(open_n),
            "settled": int(settled),
            "wins": int(won),
            "win_rate": (won / settled) if settled else None,
            "realized_pnl": float(pnl),
        }
