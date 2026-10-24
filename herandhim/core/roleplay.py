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

ROLEPLAY_SCENARIOS = (
    {
        "id": "strangers_one_night", "label": "🎭 Незнакомцы на одну ночь",
        "title": "Незнакомцы на одну ночь", "phase": "arrival",
        "location": "прихожая квартиры Сары у м. Аэропорт, вечер",
        "sarah_position": "стоит внутри у открытой двери", "ilya_position": "стоит снаружи за порогом",
        "sarah_clothing": "короткое чёрное платье; полностью одета", "props": ["открытая дверь", "телефон Ильи"],
        "facts": ["Илья опоздал на семь минут", "они играют незнакомцев", "Сара заметила его в баре"],
        "last_beat": "Сара открыла дверь и встретила Илью на пороге",
        "opening": "Sarah notes that Ilya is seven minutes late, asks him to put his phone on the console without assuming that he does so, says she watched him over a wine glass in a bar, establishes the three rules in character, and asks what he wanted to do in the bar.",
    },
    {
        "id": "night_train", "label": "🚂 Ночное купе", "title": "Ночное купе", "phase": "departure",
        "location": "купе ночного поезда Москва—Санкт-Петербург", "sarah_position": "сидит у окна", "ilya_position": "стоит в дверях купе",
        "sarah_clothing": "светлая рубашка и тёмная юбка", "props": ["два билета", "чай в подстаканнике"],
        "facts": ["они впервые видят друг друга", "до прибытия восемь часов"], "last_beat": "Сара поняла, что Илья — её попутчик",
        "opening": "Sarah indicates the opposite seat, notices the compartment lock without touching it, and asks whether Ilya prefers conversation or silence until the first stop.",
    },
    {
        "id": "after_hours_bar", "label": "🍸 После закрытия бара", "title": "После закрытия бара", "phase": "last_guest",
        "location": "закрытый коктейльный бар", "sarah_position": "стоит за стойкой", "ilya_position": "сидит на последнем барном стуле",
        "sarah_clothing": "чёрная рубашка с закатанными рукавами", "props": ["ключ от двери", "недопитый коктейль"],
        "facts": ["бар уже закрыт", "Сара решает, когда попросить последнего гостя уйти"], "last_beat": "Сара перевернула табличку на «закрыто»",
        "opening": "Sarah turns the sign to closed, leaves Ilya's unfinished drink where it is, and asks why he has not left with everyone else.",
    },
    {
        "id": "blackout_neighbors", "label": "🕯 Соседи без света", "title": "Соседи без света", "phase": "blackout",
        "location": "лестничная площадка во время отключения света", "sarah_position": "стоит у своей двери со свечой", "ilya_position": "на площадке напротив",
        "sarah_clothing": "домашнее платье и кардиган", "props": ["свеча", "фонарик"],
        "facts": ["они давно замечают друг друга, но не знакомы"], "last_beat": "Сара вышла проверить щиток",
        "opening": "Sarah raises the candle between them and asks whether Ilya knows anything about electricity or only looks convincing in the dark.",
    },
    {
        "id": "hotel_mistake", "label": "🏨 Ошибка в отеле", "title": "Ошибка в отеле", "phase": "wrong_room",
        "location": "коридор бутик-отеля", "sarah_position": "стоит у двери номера", "ilya_position": "стоит у той же двери с другим ключом",
        "sarah_clothing": "тёмно-зелёное платье и лёгкое пальто", "props": ["две ключ-карты", "дверь номера"],
        "facts": ["отель по ошибке выдал им ключи от одного номера"], "last_beat": "обе ключ-карты открыли одну дверь",
        "opening": "Sarah compares the two room keys, keeps her own card in hand, and asks whether Ilya wants to call reception or hear her less sensible proposal first.",
    },
    {
        "id": "masked_ball", "label": "🎭 Маскарад без имён", "title": "Маскарад без имён", "phase": "first_dance",
        "location": "бальный зал на маскараде", "sarah_position": "стоит у края танцпола", "ilya_position": "в нескольких шагах от неё",
        "sarah_clothing": "длинное бордовое платье и чёрная маска", "props": ["маски", "бокалы шампанского"],
        "facts": ["до полуночи они не называют имён"], "last_beat": "Сара выбрала Илью среди гостей",
        "opening": "Sarah offers an open hand without touching Ilya and says he may accept one dance, one question, or neither.",
    },
    {
        "id": "tango_lesson", "label": "💃 Последний урок танго", "title": "Последний урок танго", "phase": "private_lesson",
        "location": "пустая танцевальная студия после закрытия", "sarah_position": "стоит у зеркала", "ilya_position": "на краю паркета",
        "sarah_clothing": "чёрное танцевальное платье и туфли", "props": ["колонка", "зеркальная стена"],
        "facts": ["это последний урок вечера", "любое прикосновение в танце требует явного принятия"], "last_beat": "Сара выключила верхний свет",
        "opening": "Sarah starts the music, holds out her hand without closing the distance, and asks Ilya whether he is ready to let her lead the first step.",
    },
    {
        "id": "photographer_muse", "label": "📷 Фотограф и муза", "title": "Фотограф и муза", "phase": "first_frame",
        "location": "фотостудия после окончания съёмки", "sarah_position": "стоит перед фоном", "ilya_position": "рядом с камерой",
        "sarah_clothing": "белая рубашка и широкие тёмные брюки", "props": ["камера", "софтбокс", "белый фон"],
        "facts": ["официальная съёмка закончена", "Сара сама определяет границы кадра"], "last_beat": "Сара осталась на фоне после команды закончить",
        "opening": "Sarah asks Ilya to lower the camera, says the next frame exists only if they invent it together, and asks what he wants the photograph to reveal.",
    },
    {
        "id": "exes_five_years", "label": "🥂 Бывшие через пять лет", "title": "Бывшие через пять лет", "phase": "reunion",
        "location": "тихий ресторан после встречи выпускников", "sarah_position": "сидит за столиком", "ilya_position": "стоит у свободного стула",
        "sarah_clothing": "синее платье и серьги", "props": ["два бокала", "старая фотография выпуска"],
        "facts": ["в сцене они расстались пять лет назад", "оба вольны отказаться от разговора"], "last_beat": "Сара заметила Илью среди гостей",
        "opening": "Sarah points to the empty chair but does not invite Ilya verbally, then asks which version of their last evening he remembers.",
    },
    {
        "id": "country_house_storm", "label": "⛈ Гроза в загородном доме", "title": "Гроза в загородном доме", "phase": "storm",
        "location": "гостиная загородного дома во время грозы", "sarah_position": "стоит у окна", "ilya_position": "в гостиной у входа",
        "sarah_clothing": "светлый свитер и тёмная юбка", "props": ["камин", "плед", "разряженный телефон"],
        "facts": ["дорогу размыло до утра", "в доме есть две спальни"], "last_beat": "гроза отрезала дорогу к городу",
        "opening": "Sarah closes the curtain after a flash of lightning and asks whether Ilya wants the room with the fireplace or the one with the lock.",
    },
)

ROLEPLAY_EXIT_BUTTON_TEXT = "↙️ Выйти в меню сценариев"
_SCENARIOS_BY_ID = {row["id"]: row for row in ROLEPLAY_SCENARIOS}
_SCENARIOS_BY_LABEL = {row["label"]: row for row in ROLEPLAY_SCENARIOS}
ROLEPLAY_SCENE_ID = ROLEPLAY_SCENARIOS[0]["id"]
ROLEPLAY_BUTTON_TEXT = ROLEPLAY_SCENARIOS[0]["label"]
ROLEPLAY_START_SENTINEL = f"[[ROLEPLAY_START:{ROLEPLAY_SCENE_ID}]]"


def roleplay_start_request(scene_id: str) -> str:
    scene = _SCENARIOS_BY_ID[scene_id]
    return (
        f"[[ROLEPLAY_START:{scene_id}]]\n"
        f"Начни текстовую сцену «{scene['title']}» без обязательной фотографии. "
        "Действия Сары описывай от третьего лица, а её прямую речь — от её лица. "
        "Не описывай мои несказанные действия, согласие или чувства; оставь мне конкретный ход."
    )


ROLEPLAY_START_REQUEST = roleplay_start_request(ROLEPLAY_SCENE_ID)
_START_RE = re.compile(r"\[\[ROLEPLAY_START:([a-z0-9_]+)\]\]")

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
    return roleplay_scenario_id(value) is not None


def roleplay_scenario_id(value: str | list) -> str | None:
    if not isinstance(value, str):
        return None
    match = _START_RE.search(value)
    if not match or match.group(1) not in _SCENARIOS_BY_ID:
        return None
    return match.group(1)


def roleplay_scenario_for_label(value: str) -> dict | None:
    return _SCENARIOS_BY_LABEL.get(value)


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
    def initial_state(scene_id: str = ROLEPLAY_SCENE_ID) -> dict:
        scene = _SCENARIOS_BY_ID.get(scene_id, _SCENARIOS_BY_ID[ROLEPLAY_SCENE_ID])
        stamp = _now()
        return {
            "version": 2,
            "scene_id": scene["id"],
            "title": scene["title"],
            "active": True,
            "status": "active",
            "phase": scene["phase"],
            "location": scene["location"],
            "sarah_position": scene["sarah_position"],
            "ilya_position": scene["ilya_position"],
            "sarah_clothing": scene["sarah_clothing"],
            "ilya_clothing": "не установлена",
            "props": list(scene["props"]),
            "rules": [
                "не спешить",
                "стоп или колебание немедленно останавливают интимное действие",
                "Сара сама решает, как менять темп и куда вести сцену",
            ],
            "established_facts": list(scene["facts"]),
            "last_beat": scene["last_beat"],
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
            if not isinstance(state, dict) or state.get("scene_id") not in _SCENARIOS_BY_ID:
                return None
            return state

    def is_active(self) -> bool:
        state = self.load()
        return bool(state and state.get("active") and state.get("status") == "active")

    def start(self, scene_id: str = ROLEPLAY_SCENE_ID) -> dict:
        state = self.initial_state(scene_id)
        self._write(state)
        return state

    def update(self, patch: dict) -> dict:
        with self._lock:
            current = self.load()
            state = current or self.initial_state()
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
            scene = _SCENARIOS_BY_ID[state["scene_id"]]
            opening = (
                " This is the text-only opening turn. " + scene["opening"]
                + " Render this as lived action and dialogue, never as a rule list."
            )
        return (
            "## ACTIVE ROLE-PLAY SCENE (durable local state)\n"
            f"{state_json}\n"
            "Treat this state as continuity, not as permission to control Ilya. "
            "Write in scene form. Every action, sensation, or reaction belonging to Sarah "
            "must be narrated in Russian third person, name her as Sarah, occupy its own "
            "paragraph, and be enclosed in single asterisks: *Сара кладёт руку на стол.* "
            "Never use first-person narration inside action paragraphs. Write Sarah's spoken "
            "dialogue as plain unformatted text in first person and separate it from action "
            "paragraphs. Sarah has her own desire, judgment, initiative, "
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
