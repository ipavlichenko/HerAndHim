"""Durable per-purpose cost reservations for fail-closed outbound calls."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .policy import PolicyError


def _default_path() -> Path:
    from ... import config

    return Path(config.HERANDHIM_HOME) / "context" / "state" / "outbound_costs.sqlite3"


class CostLedger:
    """Reserve a worst-case call amount before I/O and reconcile afterwards.

    Leaving an ambiguous/failed call reserved is intentionally conservative:
    the provider may have charged it even when the client saw a timeout.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cost_reservations ("
                "id TEXT PRIMARY KEY, purpose TEXT NOT NULL, day TEXT NOT NULL, "
                "amount_usd REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def reserve(self, purpose: str, per_call_cap: float, daily_cap: float) -> str:
        day = datetime.now(timezone.utc).date().isoformat()
        reservation_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            spent = float(conn.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM cost_reservations "
                "WHERE purpose=? AND day=?", (purpose, day),
            ).fetchone()[0])
            if spent + per_call_cap > daily_cap + 1e-12:
                raise PolicyError("daily_cost_cap_exceeded")
            conn.execute(
                "INSERT INTO cost_reservations(id, purpose, day, amount_usd) VALUES (?, ?, ?, ?)",
                (reservation_id, purpose, day, per_call_cap),
            )
        return reservation_id

    def reconcile(self, reservation_id: str, actual_cost: float) -> None:
        if actual_cost < 0:
            raise PolicyError("invalid_response_cost")
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE cost_reservations SET amount_usd=? WHERE id=?",
                (actual_cost, reservation_id),
            )
