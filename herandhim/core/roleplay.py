"""Durable, bounded state for Sarah's continuous role-play scenes."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from .. import config

ROLEPLAY_BUTTON_TEXT = "🎭 Незнакомцы на одну ночь"
ROLEPLAY_START_SENTINEL = "[[ROLEPLAY_START:strangers_one_night]]"
ROLEPLAY_SCENE_ID = "strangers_one_night"

ROLEPLAY_SELFIE_HINT = (
    "Sarah has just opened the front door of her Moscow apartment at night. "
    "She stands just inside the doorway in an elegant short black dress, fully "
    "clothed, one hand on the door handle, looking directly at the arriving guest "
    "with calm confidence and restrained invitation. Warm cinematic hallway light, "
    "realistic vertical full-body smartphone photo, safe-for-work, no nudity."
)
# The opening prose follows immediately after the delivered photo. Keeping the
# image caption empty avoids repeating the same arrival beat twice.
ROLEPLAY_SELFIE_CAPTION = ""

ROLEPLAY_START_REQUEST = (
    f"{ROLEPLAY_START_SENTINEL}\n"
    "Начни сцену «Незнакомцы на одну ночь». Снача пришли стартовую "
    "фотографию: ты открываешь мне дверь квартиры вечером. Затем продолжи "
    "сцену от первого лица Сары: я опоздал на семь минут, а сегодня ты не моя девушка, "
    "а женщина, которая наблюдала за мной из-за бокала в баре. Не описывай мои несказанные "
    "действия и оставь мне конкретный ход."
)

ROLEPLAY_STATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_roleplay_scene",
        "description": (
            "Update the durable state of the active role-play scene before replying. "
            "Record only facts already established by the user's newest message or the "
            "immediately preceding scene beat; never invent the forthcoming reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {"type": "string", "description": "Short current scene phase."},
                "location": {"type": "string", "description": "Exact current location."},
                "sarah_position": {"type": "string"},
                "ilya_position": {"type": "string"},
                "sarah_clothing": {"type": "string"},
                "ilya_clothing": {"type": "string"},
                "props": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "rules": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "established_facts": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 12,
                },
                "last_beat": {"type": "string", "description": "Last established action/beat."},
                "consent_state": {
                    "type": "string",
                    "enum": ["mutual", "check_in", "paused", "stopped"],
                },
            },
            "required": ["phase", "last_beat", "consent_state"],
            "additionalProperties": False,
        },
    },
}

_END_RE = re.compile(
    r"^\s*(?:закончи(?:м|ть)\s+сцену|закончи\s+сцену|"
    r"выйдем\s+из\s+роли|выйти\s+из\s+роли|стоп\s+игра)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def is_roleplay_start(value: str | list) -> bool:
    return isinstance(value, str) and ROLEPLAY_START_SENTINEL in value


def is_roleplay_end(value: str | list) -> bool:
    return isinstance(value, str) and bool(_END_RE.match(value))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _bounded_text(value: object, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _bounded_list(value: object, *, items: int, chars: int = 240) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for row in value[:items]:
        text = _bounded_text(row, chars)
        if text and text not in result:
            result.append(text)
    return result


class RoleplaySceneStore:
    """One atomic JSON document per session, with strict size and field bounds."""

    def __init__(self, session_id: str | None, root: str | Path | None = None) -> None:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id or "default")[:160]
        base = Path(root) if root is not None else Path(config.HERANDHIM_HOME) / "context/state/roleplay"
        self.path = base / f"{safe}.json"
        self._lock = RLock()

    @staticmethod
    def initial_state() -> dict:
        stamp = _now()
        return {
            "version": 1,
            "scene_id": ROLEPLAY_SCENE_ID,
            "title": "Незнакомцы на одну ночь",
            "active": True,
            "status": "active",
            "phase": "arrival",
            "location": "прихожая квартиры Сары у м. Аэропорт, вечер",
            "sarah_position": "стоит внутри у открытой двери, рука на ручке",
            "ilya_position": "стоит снаружи за порогом",
            "sarah_clothing": "элегантное короткое чёрное платье; полностью одета",
            "ilya_clothing": "не установлена",
            "props": ["открытая входная дверь", "телефон Ильи"],
            "rules": [
                "не спешить",
                "стоп или колебание немедленно останавливают интимное действие",
                "Сара сама решает, как менять темп и куда вести сцену",
            ],
            "established_facts": [
                "Илья опоздал на семь минут",
                "в этой сцене они играют незнакомцев, а не пару",
                "Сара заметила его в баре из-за бокала вина",
            ],
            "last_beat": "Сара открыла дверь и встретила Илью на пороге",
            "consent_state": "mutual",
            "opening_pending": True,
            "turn_index": 0,
            "created_at": stamp,
            "updated_at": stamp,
        }

    def load(self) -> dict | None:
        with self._lock:
            try:
                raw = self.path.read_bytes()
                if len(raw) > 32_768:
                    return None
                state = json.loads(raw)
            except (OSError, json.JSONDecodeError, TypeError):
                return None
            if not isinstance(state, dict) or state.get("scene_id") != ROLEPLAY_SCENE_ID:
                return None
            return state

    def is_active(self) -> bool:
        state = self.load()
        return bool(state and state.get("active") and state.get("status") == "active")

    def start(self) -> dict:
        state = self.initial_state()
        self._write(state)
        return state

    def update(self, patch: dict) -> dict:
        with self._lock:
            state = self.load() or self.initial_state()
            for key in (
                "phase", "location", "sarah_position", "ilya_position",
                "sarah_clothing", "ilya_clothing", "last_beat",
            ):
                if key in patch and _bounded_text(patch[key]):
                    state[key] = _bounded_text(patch[key])
            for key, maximum in (("props", 12), ("rules", 8), ("established_facts", 12)):
                if key in patch:
                    state[key] = _bounded_list(patch[key], items=maximum)
            if patch.get("consent_state") in {"mutual", "check_in", "paused", "stopped"}:
                state["consent_state"] = patch["consent_state"]
            state["turn_index"] = min(int(state.get("turn_index", 0)) + 1, 1_000_000)
            state["updated_at"] = _now()
            if state.get("consent_state") == "stopped":
                state["active"] = False
                state["status"] = "ended"
            self._write(state)
            return state

    def mark_opened(self) -> None:
        with self._lock:
            state = self.load()
            if not state:
                return
            state["opening_pending"] = False
            state["turn_index"] = max(1, int(state.get("turn_index", 0)))
            state["updated_at"] = _now()
            self._write(state)

    def end(self) -> None:
        with self._lock:
            state = self.load()
            if not state:
                return
            state.update({
                "active": False,
                "status": "ended",
                "consent_state": "stopped",
                "updated_at": _now(),
            })
            self._write(state)

    def prompt_block(self) -> str:
        state = self.load()
        if not state or not state.get("active"):
            return ""
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        opening = ""
        if state.get("opening_pending"):
            opening = (
                " This is the opening turn, immediately after the door photo. Sarah notes "
                "that Ilya is seven minutes late, takes his phone without presuming any "
                "further action by him, and says that tonight she is not his girlfriend but "
                "the woman who watched him over a wine glass in a bar. She establishes three "
                "rules in character: do not rush; stop means stop; she decides how the scene's "
                "pace and direction develop. She finishes by asking what he had wanted to do "
                "in the bar. Render this as lived action and dialogue, never as a rule list."
            )
        return (
            "## ACTIVE ROLE-PLAY SCENE (durable local state)\n"
            f"{state_json}\n"
            "Treat this state as continuity, not as permission to control Ilya. "
            "Write in scene form: alternate *observable actions, sensations, and reactions* "
            "with Sarah's spoken dialogue. Sarah has her own desire, judgment, initiative, "
            "and boundaries; she may lead, refuse, redirect, slow down, or stop. Never narrate "
            "Ilya's unspoken decisions, consent, speech, feelings, or actions. Preserve location, "
            "positions, clothing, props, rules, and established facts until an explicit scene "
            "event changes them. Consensual adult explicitness may develop naturally. Any stop, "
            "hesitation, ambiguity, or withdrawn consent immediately pauses intimate action and "
            "requires a check-in. End each turn on a concrete playable opening, not a meta question."
            + opening
        )

    def _write(self, state: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
