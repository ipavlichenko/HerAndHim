"""Restart-safe selection metadata; conversation contents live in isolated homes."""
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .characters import character


def user_key(value: int) -> str:
    if type(value) is not int or value <= 0:
        raise ValueError("Private Telegram user required")
    return f"telegram:{value}"


@dataclass(frozen=True)
class Episode:
    user_id: int
    character_id: str
    id: str
    legacy: bool = False

    @property
    def session_id(self):
        return user_key(self.user_id) if self.legacy else f"episode:{self.id}"

    def data_home(self, root: Path) -> Path:
        return root if self.legacy else root / "catalog" / "players" / str(self.user_id) / self.character_id / self.id


class CatalogState:
    def __init__(self, root: Path, *, legacy_owner: int | None = None, max_users: int = 50):
        self.path = root / "context/state/catalog.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_users = max_users
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS catalog_players (
                    user_id INTEGER PRIMARY KEY, adult INTEGER NOT NULL DEFAULT 0,
                    active_character TEXT, proactive INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS catalog_episodes (
                    user_id INTEGER NOT NULL, character_id TEXT NOT NULL,
                    id TEXT NOT NULL UNIQUE, legacy INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id, character_id)
                );
            """)
            if legacy_owner is not None:
                user_key(legacy_owner)
                db.execute("INSERT OR IGNORE INTO catalog_players VALUES (?,1,'sara',1)", (legacy_owner,))
                db.execute("INSERT OR IGNORE INTO catalog_episodes VALUES (?,'sara',?,1)",
                           (legacy_owner, f"legacy-{legacy_owner}"))

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def admit(self, user_id: int):
        user_key(user_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM catalog_players WHERE user_id=?", (user_id,)).fetchone():
                return
            if db.execute("SELECT count(*) FROM catalog_players").fetchone()[0] >= self.max_users:
                raise ValueError("Pilot is full")
            db.execute("INSERT INTO catalog_players(user_id) VALUES (?)", (user_id,))

    def confirm_adult(self, user_id: int):
        self.admit(user_id)
        with self.connect() as db:
            db.execute("UPDATE catalog_players SET adult=1 WHERE user_id=?", (user_id,))

    def is_adult(self, user_id: int):
        with self.connect() as db:
            row = db.execute("SELECT adult FROM catalog_players WHERE user_id=?", (user_id,)).fetchone()
            return bool(row and row[0])

    def selected(self, user_id: int) -> Episode | None:
        user_key(user_id)
        with self.connect() as db:
            row = db.execute("SELECT e.* FROM catalog_episodes e JOIN catalog_players p "
                             "ON p.user_id=e.user_id AND p.active_character=e.character_id "
                             "WHERE p.user_id=?", (user_id,)).fetchone()
            return Episode(**dict(row)) if row else None

    def select(self, user_id: int, character_id: str, *, new_episode: bool = False) -> Episode:
        character(character_id)
        user_key(user_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT adult FROM catalog_players WHERE user_id=?", (user_id,)).fetchone()
            if not row or not row[0]:
                raise ValueError("Adult confirmation required")
            existing = db.execute("SELECT * FROM catalog_episodes WHERE user_id=? AND character_id=?",
                                  (user_id, character_id)).fetchone()
            if existing and not new_episode:
                episode = Episode(**dict(existing))
            else:
                episode = Episode(user_id, character_id, uuid.uuid4().hex)
                db.execute("INSERT OR REPLACE INTO catalog_episodes VALUES (?,?,?,0)",
                           (user_id, character_id, episode.id))
            db.execute("UPDATE catalog_players SET active_character=? WHERE user_id=?", (character_id, user_id))
            return episode

    def set_proactive(self, user_id: int, enabled: bool):
        with self.connect() as db:
            db.execute("UPDATE catalog_players SET proactive=? WHERE user_id=?", (int(enabled), user_id))

    def active_users(self):
        with self.connect() as db:
            return [row[0] for row in db.execute(
                "SELECT user_id FROM catalog_players WHERE adult=1 AND proactive=1 AND active_character IS NOT NULL")]
