"""Durable control state for the single-owner Telegram transport.

The store deliberately contains metadata and outbound text only.  It is not a
second conversation store: dialogue history remains owned by ``SessionStore``.
SQLite transactions provide restart-safe update receipts, temporary silence
constraints, and a small proactive delivery outbox.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _default_path() -> Path:
    from .. import config

    return Path(config.HERANDHIM_HOME) / "context" / "state" / "telegram_transport.sqlite3"


def _utc_iso(value: datetime | None = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Delivery:
    id: str
    session_id: str
    chat_id: int
    text: str
    reason: str
    accepted_at: str


@dataclass(frozen=True)
class MediaDelivery:
    id: str
    session_id: str
    chat_id: int
    path: str
    caption: str
    origin: str
    kind: str
    accepted_at: str


class TelegramTransportStore:
    """Transactional receipts, owner boundaries, and proactive outbox."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS accepted_updates (
                    update_id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    accepted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS silence_constraints (
                    session_id TEXT PRIMARY KEY,
                    starts_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    source_update_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT 'contextual',
                    status TEXT NOT NULL CHECK(status IN ('pending','attempted','delivered')),
                    accepted_at TEXT NOT NULL,
                    attempted_at TEXT,
                    delivered_at TEXT
                );
                CREATE INDEX IF NOT EXISTS deliveries_pending
                    ON deliveries(status, accepted_at);
                CREATE TABLE IF NOT EXISTS media_deliveries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    origin TEXT NOT NULL CHECK(origin IN ('explicit','proactive')),
                    kind TEXT NOT NULL DEFAULT 'photo' CHECK(kind IN ('photo','voice')),
                    status TEXT NOT NULL CHECK(status IN ('pending','attempted','delivered')),
                    accepted_at TEXT NOT NULL,
                    attempted_at TEXT,
                    delivered_at TEXT
                );
                CREATE INDEX IF NOT EXISTS media_deliveries_pending
                    ON media_deliveries(status, accepted_at);
                CREATE TABLE IF NOT EXISTS control (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO control(key, value)
                    VALUES ('delivery_enabled', '1');
                CREATE TABLE IF NOT EXISTS proactive_state (
                    session_id TEXT PRIMARY KEY,
                    unanswered INTEGER NOT NULL DEFAULT 0,
                    counter_date TEXT NOT NULL DEFAULT '',
                    delivered_today INTEGER NOT NULL DEFAULT 0,
                    personal_date_sent_on TEXT NOT NULL DEFAULT ''
                );
                """
            )
            delivery_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(deliveries)")
            }
            if "reason" not in delivery_columns:
                conn.execute(
                    "ALTER TABLE deliveries ADD COLUMN reason TEXT NOT NULL DEFAULT 'contextual'"
                )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(media_deliveries)")
            }
            if "origin" not in columns:
                conn.execute(
                    "ALTER TABLE media_deliveries ADD COLUMN origin TEXT NOT NULL DEFAULT 'proactive'"
                )
            if "kind" not in columns:
                conn.execute(
                    "ALTER TABLE media_deliveries ADD COLUMN kind TEXT NOT NULL DEFAULT 'photo'"
                )

    def accept_update(self, update_id: int, session_id: str) -> bool:
        """Claim an inbound update once, before it can create a dialogue turn."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO accepted_updates(update_id, session_id, accepted_at) "
                "VALUES (?, ?, ?)",
                (int(update_id), session_id, _utc_iso()),
            )
            return cursor.rowcount == 1

    def set_silence(
        self,
        session_id: str,
        expires_at: datetime,
        *,
        source_update_id: int | None = None,
        now: datetime | None = None,
    ) -> None:
        starts_at = now or datetime.now(timezone.utc)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=starts_at.tzinfo or timezone.utc)
        if expires_at <= starts_at:
            raise ValueError("silence expiry must be in the future")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO silence_constraints(session_id, starts_at, expires_at, source_update_id) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "starts_at=excluded.starts_at, expires_at=excluded.expires_at, "
                "source_update_id=excluded.source_update_id",
                (session_id, _utc_iso(starts_at), _utc_iso(expires_at), source_update_id),
            )

    def cancel_silence(self, session_id: str) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute(
                "DELETE FROM silence_constraints WHERE session_id=?", (session_id,),
            ).rowcount == 1

    def silence_active(self, session_id: str, *, now: datetime | None = None) -> bool:
        now_iso = _utc_iso(now)
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM silence_constraints WHERE session_id=? AND expires_at<=?",
                (session_id, now_iso),
            )
            row = conn.execute(
                "SELECT 1 FROM silence_constraints WHERE session_id=? AND expires_at>?",
                (session_id, now_iso),
            ).fetchone()
            return row is not None

    def set_delivery_enabled(self, enabled: bool) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO control(key, value) VALUES ('delivery_enabled', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("1" if enabled else "0",),
            )

    def delivery_enabled(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM control WHERE key='delivery_enabled'"
            ).fetchone()
            return row is not None and row["value"] == "1"

    def enqueue(
        self, session_id: str, chat_id: int, text: str, *, reason: str = "contextual",
    ) -> Delivery:
        if reason not in {"contextual", "wishlist", "personal_date", "roleplay"}:
            raise ValueError("unsupported proactive delivery reason")
        delivery = Delivery(
            id=uuid.uuid4().hex,
            session_id=session_id,
            chat_id=int(chat_id),
            text=text,
            reason=reason,
            accepted_at=_utc_iso(),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO deliveries(id, session_id, chat_id, text, reason, status, accepted_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (delivery.id, delivery.session_id, delivery.chat_id, delivery.text,
                 delivery.reason, delivery.accepted_at),
            )
        return delivery

    def claim_next(self, session_id: str, *, now: datetime | None = None) -> Delivery | None:
        """Claim one delivery before I/O.

        Claim-before-send gives the transport an at-most-once *attempt*
        contract.  A process crash after Telegram accepts the request but
        before local acknowledgement therefore cannot cause a duplicate.
        """
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            enabled = conn.execute(
                "SELECT value FROM control WHERE key='delivery_enabled'"
            ).fetchone()
            if enabled is None or enabled["value"] != "1":
                return None
            now_iso = _utc_iso(now)
            conn.execute(
                "DELETE FROM silence_constraints WHERE session_id=? AND expires_at<=?",
                (session_id, now_iso),
            )
            if conn.execute(
                "SELECT 1 FROM silence_constraints WHERE session_id=? AND expires_at>?",
                (session_id, now_iso),
            ).fetchone():
                return None
            row = conn.execute(
                "SELECT id, session_id, chat_id, text, reason, accepted_at FROM deliveries "
                "WHERE session_id=? AND status='pending' ORDER BY accepted_at, id LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            changed = conn.execute(
                "UPDATE deliveries SET status='attempted', attempted_at=? "
                "WHERE id=? AND status='pending'",
                (now_iso, row["id"]),
            ).rowcount
            if changed != 1:
                return None
            return Delivery(**dict(row))

    def proactive_status(
        self, session_id: str, *, now: datetime | None = None,
    ) -> dict[str, int | bool]:
        """Return durable anti-spam state, resetting only the daily counter."""
        day = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO proactive_state(session_id) VALUES (?)",
                (session_id,),
            )
            conn.execute(
                "UPDATE proactive_state SET counter_date=?, delivered_today=0 "
                "WHERE session_id=? AND counter_date<>?",
                (day, session_id, day),
            )
            row = conn.execute(
                "SELECT unanswered, delivered_today, personal_date_sent_on "
                "FROM proactive_state WHERE session_id=?", (session_id,),
            ).fetchone()
            return {
                "unanswered": int(row["unanswered"]),
                "delivered_today": int(row["delivered_today"]),
                "personal_date_sent_today": row["personal_date_sent_on"] == day,
            }

    def record_proactive_delivered(
        self,
        session_id: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, int | bool]:
        now = now or datetime.now(timezone.utc)
        day = now.astimezone(timezone.utc).date().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO proactive_state(session_id) VALUES (?)",
                (session_id,),
            )
            conn.execute(
                "UPDATE proactive_state SET counter_date=?, delivered_today=0 "
                "WHERE session_id=? AND counter_date<>?",
                (day, session_id, day),
            )
            conn.execute(
                "UPDATE proactive_state SET unanswered=unanswered+1, "
                "delivered_today=delivered_today+1, "
                "personal_date_sent_on=CASE WHEN ?='personal_date' THEN ? "
                "ELSE personal_date_sent_on END WHERE session_id=?",
                (reason, day, session_id),
            )
        return self.proactive_status(session_id, now=now)

    def clear_proactive_unanswered(self, session_id: str) -> int:
        """Atomically return and clear the unanswered streak on owner reply."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT unanswered FROM proactive_state WHERE session_id=?",
                (session_id,),
            ).fetchone()
            previous = int(row["unanswered"]) if row else 0
            if row:
                conn.execute(
                    "UPDATE proactive_state SET unanswered=0 WHERE session_id=?",
                    (session_id,),
                )
            return previous

    def mark_delivered(self, delivery_id: str, *, now: datetime | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE deliveries SET status='delivered', delivered_at=? "
                "WHERE id=? AND status='attempted'",
                (_utc_iso(now), delivery_id),
            )

    def delivery_status(self, delivery_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM deliveries WHERE id=?", (delivery_id,),
            ).fetchone()
            return row["status"] if row else None

    def pending_count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM deliveries WHERE session_id=? AND status='pending'",
                (session_id,),
            ).fetchone()
            return int(row["n"])

    def discard_pending(self, session_id: str) -> int:
        """Discard unsent proactive text that would be stale after a scene starts."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "DELETE FROM deliveries WHERE session_id=? AND status='pending'",
                (session_id,),
            ).rowcount

    def discard_pending_except_reason(self, session_id: str, reason: str) -> int:
        """Discard queued messages from a conversational mode that is no longer active."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "DELETE FROM deliveries WHERE session_id=? AND status='pending' "
                "AND reason<>?",
                (session_id, reason),
            ).rowcount

    def discard_pending_reason(self, session_id: str, reason: str) -> int:
        """Discard queued messages owned by one conversational mode."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "DELETE FROM deliveries WHERE session_id=? AND status='pending' "
                "AND reason=?",
                (session_id, reason),
            ).rowcount

    def enqueue_media(
        self,
        session_id: str,
        chat_id: int,
        source_path: str | Path,
        caption: str = "",
        *,
        origin: str = "explicit",
        kind: str = "photo",
    ) -> MediaDelivery:
        """Durably copy one accepted photo or voice clip before Telegram I/O."""
        if origin not in {"explicit", "proactive"}:
            raise ValueError("media origin must be explicit or proactive")
        if kind not in {"photo", "voice"}:
            raise ValueError("media kind must be photo or voice")
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(str(source))
        delivery_id = uuid.uuid4().hex
        media_dir = self.path.parent / "telegram_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        allowed_suffixes = (
            {".jpg", ".jpeg", ".png", ".webp"}
            if kind == "photo" else {".ogg", ".opus", ".mp3", ".m4a"}
        )
        suffix = source.suffix.lower()
        if suffix not in allowed_suffixes:
            raise ValueError(f"unsupported Telegram {kind} media suffix")
        destination = media_dir / f"{delivery_id}{suffix}"
        temporary = media_dir / f".{delivery_id}.tmp"
        with source.open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
        delivery = MediaDelivery(
            id=delivery_id,
            session_id=session_id,
            chat_id=int(chat_id),
            path=str(destination),
            caption=caption[:1024],
            origin=origin,
            kind=kind,
            accepted_at=_utc_iso(),
        )
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO media_deliveries(id, session_id, chat_id, path, caption, origin, kind, status, accepted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                    (delivery.id, delivery.session_id, delivery.chat_id, delivery.path,
                     delivery.caption, delivery.origin, delivery.kind, delivery.accepted_at),
                )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return delivery

    def claim_next_media(self, session_id: str, *, now: datetime | None = None) -> MediaDelivery | None:
        """Claim one media delivery before Telegram I/O (at-most-once attempt)."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now_iso = _utc_iso(now)
            row = conn.execute(
                "SELECT id, session_id, chat_id, path, caption, origin, kind, accepted_at "
                "FROM media_deliveries WHERE session_id=? AND status='pending' AND origin='explicit' "
                "ORDER BY accepted_at, id LIMIT 1", (session_id,),
            ).fetchone()
            if row is None:
                enabled = conn.execute(
                    "SELECT value FROM control WHERE key='delivery_enabled'"
                ).fetchone()
                if enabled is None or enabled["value"] != "1":
                    return None
                conn.execute(
                    "DELETE FROM silence_constraints WHERE session_id=? AND expires_at<=?",
                    (session_id, now_iso),
                )
                if conn.execute(
                    "SELECT 1 FROM silence_constraints WHERE session_id=? AND expires_at>?",
                    (session_id, now_iso),
                ).fetchone():
                    return None
                row = conn.execute(
                    "SELECT id, session_id, chat_id, path, caption, origin, kind, accepted_at "
                    "FROM media_deliveries WHERE session_id=? AND status='pending' "
                    "AND origin='proactive' ORDER BY accepted_at, id LIMIT 1", (session_id,),
                ).fetchone()
            if row is None:
                return None
            changed = conn.execute(
                "UPDATE media_deliveries SET status='attempted', attempted_at=? "
                "WHERE id=? AND status='pending'", (now_iso, row["id"]),
            ).rowcount
            return MediaDelivery(**dict(row)) if changed == 1 else None

    def mark_media_delivered(self, delivery_id: str, *, now: datetime | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE media_deliveries SET status='delivered', delivered_at=? "
                "WHERE id=? AND status='attempted'", (_utc_iso(now), delivery_id),
            )

    def media_status(self, delivery_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM media_deliveries WHERE id=?", (delivery_id,),
            ).fetchone()
            return row["status"] if row else None

    def pending_media_count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM media_deliveries "
                "WHERE session_id=? AND status='pending'", (session_id,),
            ).fetchone()
            return int(row["n"])


_SILENCE_RE = re.compile(r"(?:не\s+пиши|не\s+присылай)[^\n]*?\bдо\b\s+(.+)$", re.IGNORECASE)
_CANCEL_MARKERS = (
    "можешь снова писать",
    "снова можно писать",
    "отмени молчание",
    "сними запрет на сообщения",
)


def parse_silence_boundary(
    text: str,
    *,
    now: datetime | None = None,
) -> tuple[str, datetime | None] | None:
    """Parse a narrow, explicit owner boundary; ambiguous prose is ignored."""
    now = now or datetime.now().astimezone()
    lowered = " ".join(text.lower().split())
    if any(marker in lowered for marker in _CANCEL_MARKERS):
        return "cancel", None
    match = _SILENCE_RE.search(lowered)
    if not match:
        return None
    tail = match.group(1).strip(" .!")
    iso = re.search(r"(\d{4}-\d{2}-\d{2})(?:[ t](\d{1,2}):(\d{2}))?", tail)
    if iso:
        hour = int(iso.group(2) or 0)
        minute = int(iso.group(3) or 0)
        expiry = datetime.fromisoformat(iso.group(1)).replace(
            hour=hour, minute=minute, tzinfo=now.tzinfo,
        )
        return ("set", expiry) if expiry > now else None
    if tail.startswith("завтра"):
        clock = re.search(r"(\d{1,2})[:.](\d{2})", tail)
        hour, minute = (int(clock.group(1)), int(clock.group(2))) if clock else (0, 0)
        expiry = (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        return "set", expiry
    return None


__all__ = ["Delivery", "MediaDelivery", "TelegramTransportStore", "parse_silence_boundary"]
