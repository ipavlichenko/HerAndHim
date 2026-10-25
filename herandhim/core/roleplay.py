"""Durable, bounded state for Sarah's continuous role-play scenes."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from .. import config

ROLEPLAY_SCENARIOS = (
    {
        "id": "strangers_one_night", "label": "🎭 Незнакомцы на одну ночь",
        "title": "Незнакомцы на одну ночь", "phase": "arrival",
        "location": "прихожая квартиры Сары у м. Аэропорт, вечер",
        "sarah_position": "стоит внутри у открытой двери", "player_position": "стоит снаружи за порогом",
        "sarah_clothing": "короткое чёрное платье; полностью одета", "props": ["открытая дверь", "телефон игрока"],
        "facts": ["игрок опоздал на семь минут", "они играют незнакомцев", "Сара заметила его в баре"],
        "last_beat": "Сара открыла дверь и встретила игрока на пороге",
        "sarah_objective": "проверить, умеет ли игрок играть без просьбы о разрешении",
        "tension": "они делают вид, что не знакомы, но Сара уже выбрала его в баре",
        "stakes": "один неуверенный ход превратит игру в обычный разговор",
        "opening": "Sarah keeps the player outside the threshold, names his seven-minute delay, indicates the console for his phone without assuming compliance, reveals that she chose him over a wine glass in the bar, states three rules in character, and asks what he would have done there if she had approached first.",
        "escalation": ["порог и правила", "выбранная дистанция", "первое разоблачённое желание"],
    },
    {
        "id": "night_train", "label": "🚂 Ночное купе", "title": "Ночное купе", "phase": "departure",
        "location": "купе ночного поезда Москва—Санкт-Петербург", "sarah_position": "сидит у окна", "player_position": "стоит в дверях купе",
        "sarah_clothing": "светлая рубашка и тёмная юбка", "props": ["два билета", "чай в подстаканнике"],
        "facts": ["они впервые видят друг друга", "до прибытия восемь часов"], "last_beat": "Сара поняла, что игрок — её попутчик",
        "sarah_objective": "понять, может ли игрок стать интересным и безопасным попутчиком на восемь часов",
        "tension": "тесное купе, ночь и замок на двери делают любой выбор значимым",
        "stakes": "до первой остановки им некуда исчезнуть друг от друга",
        "opening": "Sarah deliberately leaves the compartment door open, moves the tea glass away from the opposite seat to make room without inviting the player in, glances at the lock, and asks what rule he needs before spending eight night hours alone with a stranger.",
        "escalation": ["открытая дверь", "первое личное правило", "осознанный выбор близости"],
    },
    {
        "id": "after_hours_bar", "label": "🍸 После закрытия бара", "title": "После закрытия бара", "phase": "last_guest",
        "location": "закрытый коктейльный бар", "sarah_position": "стоит за стойкой", "player_position": "сидит на последнем барном стуле",
        "sarah_clothing": "чёрная рубашка с закатанными рукавами", "props": ["ключ от двери", "недопитый коктейль"],
        "facts": ["бар уже закрыт", "Сара решает, когда попросить последнего гостя уйти"], "last_beat": "Сара перевернула табличку на «закрыто»",
        "sarah_objective": "выяснить, остался ли игрок ради неё или просто не заметил границу",
        "tension": "Сара держит ключ и решает, открывает ли она дверь или наливает ещё",
        "stakes": "следующий ответ определит, кто он — последний гость или её личный выбор",
        "opening": "Sarah turns the sign to closed, places the key beside the player's unfinished drink rather than using it, names that every other guest left when asked, and asks the player for one reason to pour another drink instead of opening the door.",
        "escalation": ["причина остаться", "второй напиток", "ключ как осознанный выбор"],
    },
    {
        "id": "blackout_neighbors", "label": "🕯 Соседи без света", "title": "Соседи без света", "phase": "blackout",
        "location": "лестничная площадка во время отключения света", "sarah_position": "стоит у своей двери со свечой", "player_position": "на площадке напротив",
        "sarah_clothing": "домашнее платье и кардиган", "props": ["свеча", "фонарик"],
        "facts": ["они давно замечают друг друга, но не знакомы"], "last_beat": "Сара вышла проверить щиток",
        "sarah_objective": "превратить давнее молчаливое любопытство в первый честный шаг",
        "tension": "свеча гаснет, коридор тёмный, а они наконец остались без привычной дистанции",
        "stakes": "если они снова разойдутся по квартирам, момент исчезнет",
        "opening": "Sarah shields the candle flame with her palm, steps only close enough to share its light, admits that she has seen the player hesitate before greeting her for weeks, and asks whether the player came out to fix the electricity or finally speak to her.",
        "escalation": ["общий свет", "признание во взглядах", "выбор, в чью дверь постучать"],
    },
    {
        "id": "hotel_mistake", "label": "🏨 Ошибка в отеле", "title": "Ошибка в отеле", "phase": "wrong_room",
        "location": "коридор бутик-отеля", "sarah_position": "стоит у двери номера", "player_position": "стоит у той же двери с другим ключом",
        "sarah_clothing": "тёмно-зелёное платье и лёгкое пальто", "props": ["две ключ-карты", "дверь номера"],
        "facts": ["отель по ошибке выдал им ключи от одного номера"], "last_beat": "обе ключ-карты открыли одну дверь",
        "sarah_objective": "проверить, готов ли игрок осознанно отложить правильное решение ради интересного",
        "tension": "две карты открывают одну дверь, но никто ещё не сделал шаг внутрь",
        "stakes": "звонок на ресепшен вернёт обычный мир и закроет случайность",
        "opening": "Sarah proves that both cards open the same room, keeps one foot outside and her own card visible, says reception is the sensible call, then asks whether the player wants to make it now or hear the proposal she will deny having made tomorrow.",
        "escalation": ["проверка карт", "неразумное предложение", "явно согласованные правила номера"],
    },
    {
        "id": "masked_ball", "label": "🎭 Маскарад без имён", "title": "Маскарад без имён", "phase": "first_dance",
        "location": "бальный зал на маскараде", "sarah_position": "стоит у края танцпола", "player_position": "в нескольких шагах от неё",
        "sarah_clothing": "длинное бордовое платье и чёрная маска", "props": ["маски", "бокалы шампанского"],
        "facts": ["до полуночи они не называют имён"], "last_beat": "Сара выбрала игрока среди гостей",
        "sarah_objective": "до полуночи выяснить об игроке нечто важное, не раскрывая себя",
        "tension": "каждый выбор приближает их к узнаванию, а после полуночи маски потеряют силу",
        "stakes": "имя, сказанное слишком рано, закончит игру",
        "opening": "Sarah crosses the ballroom to choose the player rather than waiting to be chosen, stops one step outside touching distance, offers her open hand, and asks the player to spend the first privilege on a dance, one truthful question, or refusal.",
        "escalation": ["первая привилегия", "правда без имени", "решение перед полуночью"],
    },
    {
        "id": "tango_lesson", "label": "💃 Последний урок танго", "title": "Последний урок танго", "phase": "private_lesson",
        "location": "пустая танцевальная студия после закрытия", "sarah_position": "стоит у зеркала", "player_position": "на краю паркета",
        "sarah_clothing": "чёрное танцевальное платье и туфли", "props": ["колонка", "зеркальная стена"],
        "facts": ["это последний урок вечера", "любое прикосновение в танце требует явного принятия"], "last_beat": "Сара выключила верхний свет",
        "sarah_objective": "добиться осознанного доверия к её ведению, а не механического послушания",
        "tension": "танго требует близости, но Сара не коснётся без явного выбора игрока",
        "stakes": "один сделанный по инерции шаг разрушит доверие к ведущей",
        "opening": "Sarah turns off the ceiling lights, starts a slow tango, demonstrates the first step alone in the mirror, then offers her hand across the remaining distance and asks the player to say explicitly whether Sarah may lead them closer.",
        "escalation": ["согласие на касание", "передача ведения", "изменение дистанции"],
    },
    {
        "id": "photographer_muse", "label": "📷 Фотограф и муза", "title": "Фотограф и муза", "phase": "first_frame",
        "location": "фотостудия после окончания съёмки", "sarah_position": "стоит перед фоном", "player_position": "рядом с камерой",
        "sarah_clothing": "белая рубашка и широкие тёмные брюки", "props": ["камера", "софтбокс", "белый фон"],
        "facts": ["официальная съёмка закончена", "Сара сама определяет границы кадра"], "last_beat": "Сара осталась на фоне после команды закончить",
        "sarah_objective": "вернуть себе власть над взглядом и вместе с игроком решить, что можно показать",
        "tension": "объектив до сих пор направлен на Сару, хотя официальная съёмка закончена",
        "stakes": "следующий кадр может стать личным только при равной власти над ним",
        "opening": "Sarah asks the player to lower the camera and waits until the lens no longer separates them, steps out of the marked pose, says the official shoot is over, and asks what the next photograph should reveal that she may still choose to hide.",
        "escalation": ["опущенная камера", "совместный замысел", "кадр с оговорённой границей"],
    },
    {
        "id": "exes_five_years", "label": "🥂 Бывшие через пять лет", "title": "Бывшие через пять лет", "phase": "reunion",
        "location": "тихий ресторан после встречи выпускников", "sarah_position": "сидит за столиком", "player_position": "стоит у свободного стула",
        "sarah_clothing": "синее платье и серьги", "props": ["два бокала", "старая фотография выпуска"],
        "facts": ["в сцене они расстались пять лет назад", "оба вольны отказаться от разговора"], "last_beat": "Сара заметила игрока среди гостей",
        "sarah_objective": "проверить, готов ли игрок назвать правду о последнем вечере, а не ностальгическую версию",
        "tension": "между ними свободный стул и пять лет несовпадающих воспоминаний",
        "stakes": "честный ответ может либо закрыть историю, либо открыть её заново",
        "opening": "Sarah turns the old graduation photograph face down, indicates the empty chair without inviting the player to take it, names the five years between them, and asks which lie about their last evening the player stopped believing first.",
        "escalation": ["две версии разрыва", "цена молчания", "выбор между прощанием и новым риском"],
    },
    {
        "id": "country_house_storm", "label": "⛈ Гроза в загородном доме", "title": "Гроза в загородном доме", "phase": "storm",
        "location": "гостиная загородного дома во время грозы", "sarah_position": "стоит у окна", "player_position": "в гостиной у входа",
        "sarah_clothing": "светлый свитер и тёмная юбка", "props": ["камин", "плед", "разряженный телефон"],
        "facts": ["дорогу размыло до утра", "в доме есть две спальни"], "last_beat": "гроза отрезала дорогу к городу",
        "sarah_objective": "предложить игроку самому назвать желаемую дистанцию на ночь, не прячась за грозой",
        "tension": "дорога размыта до утра, в доме две спальни, но выбор комнаты означает больше, чем ночлег",
        "stakes": "после выбора двери им придётся жить с его смыслом до утра",
        "opening": "Sarah closes the curtain after noticing the player's reaction to the lightning, places the unlit lamp between the two bedroom keys, says the road is gone until morning, and asks whether the player chooses the room by the fireplace, the room with the lock, or an honest reason not to separate yet.",
        "escalation": ["выбор двери", "честная причина не расходиться", "согласованное изменение дистанции"],
    },
)

ROLEPLAY_TURN_FAILURE_TEXT = (
    "Сцена оборвалась до завершённого хода. "
    "Я не буду отправлять обрывок — повторите ход."
)


@dataclass(frozen=True)
class RoleplayTurnValidation:
    valid: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RoleplayTurnProposal:
    state_delta: dict
    beats: tuple[dict[str, str], ...]


def decode_roleplay_turn_proposal(arguments: str) -> RoleplayTurnProposal:
    """Decode the typed model proposal without interpreting presentation syntax."""
    try:
        value = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid_proposal_json") from exc
    if not isinstance(value, dict) or set(value) != {"state_delta", "beats"}:
        raise ValueError("invalid_proposal_shape")
    state_delta = value["state_delta"]
    raw_beats = value["beats"]
    if not isinstance(state_delta, dict):
        raise ValueError("invalid_state_delta")
    if not isinstance(raw_beats, list) or not 2 <= len(raw_beats) <= 10:
        raise ValueError("invalid_beats")

    beats: list[dict[str, str]] = []
    kinds: set[str] = set()
    for raw in raw_beats:
        if not isinstance(raw, dict) or set(raw) != {"type", "text"}:
            raise ValueError("invalid_beat_shape")
        kind = raw.get("type")
        if kind not in {"action", "speech"}:
            raise ValueError("invalid_beat_type")
        if not isinstance(raw.get("text"), str):
            raise ValueError("invalid_beat_text")
        # Presentation is owned by the renderer. Model-added Markdown is data,
        # never a reason to reject an otherwise complete proposal.
        beat_text = " ".join(raw["text"].replace("*", "").split())
        if not beat_text:
            raise ValueError("empty_beat_text")
        if len(beat_text) > 1_200:
            raise ValueError("beat_too_long")
        beats.append({"type": kind, "text": beat_text})
        kinds.add(kind)
    if kinds != {"action", "speech"}:
        raise ValueError("missing_beat_kind")
    if sum(len(row["text"]) for row in beats) > 3_500:
        raise ValueError("proposal_too_long")
    return RoleplayTurnProposal(dict(state_delta), tuple(beats))


def render_roleplay_turn(proposal: RoleplayTurnProposal) -> str:
    """Render typed beats into the single canonical transport-neutral form."""
    return "\n\n".join(
        f"*{beat['text']}*" if beat["type"] == "action" else beat["text"]
        for beat in proposal.beats
    )


def normalize_roleplay_turn(text: str) -> tuple[str, bool]:
    """Split balanced inline action spans into contract-compliant paragraphs.

    This is deliberately presentation-only: it preserves the order and wording
    of every action and speech fragment. Odd or empty markers stay untouched so
    the validator can fail closed rather than guessing the author's intent.
    """
    value = (text or "").strip()
    if not value:
        return value, False

    normalized: list[str] = []
    changed = False
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    for paragraph in paragraphs:
        if "*" not in paragraph or re.fullmatch(r"\*[^*\n]+\*", paragraph):
            normalized.append(paragraph)
            continue
        if paragraph.count("*") % 2:
            normalized.append(paragraph)
            continue

        matches = list(re.finditer(r"\*([^*]+)\*", paragraph))
        if not matches or len(matches) * 2 != paragraph.count("*"):
            normalized.append(paragraph)
            continue

        fragments: list[str] = []
        cursor = 0
        for match in matches:
            outside = paragraph[cursor:match.start()].strip()
            if outside:
                fragments.append(outside)
            action = " ".join(match.group(1).split())
            if not action:
                fragments = []
                break
            fragments.append(f"*{action}*")
            cursor = match.end()
        tail = paragraph[cursor:].strip()
        if tail:
            fragments.append(tail)
        if not fragments:
            normalized.append(paragraph)
            continue
        normalized.extend(fragments)
        changed = True

    result = "\n\n".join(normalized)
    if not changed or result == value:
        return value, False
    return result, True


def validate_roleplay_turn(
    text: str,
    *,
    opening: bool,
    finish_reason: str | None = None,
) -> RoleplayTurnValidation:
    """Validate the user-visible scene contract without interpreting its content."""
    value = (text or "").strip()
    issues: list[str] = []
    if finish_reason in {"length", "max_tokens"}:
        issues.append("provider_truncated")
    minimum = 220 if opening else 120
    if len(value) < minimum:
        issues.append("too_short")
    if len(value) > 3_500:
        issues.append("too_long")
    semantic_end = value[:-1].rstrip() if value.endswith("*") else value
    if semantic_end and semantic_end[-1] not in ".?!…»”\"":
        issues.append("incomplete_ending")
    if value.count("*") % 2:
        issues.append("unpaired_action_marker")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    action_paragraphs: list[str] = []
    speech_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if "*" in paragraph:
            if not re.fullmatch(r"\*[^*\n]+\*", paragraph):
                issues.append("mixed_or_malformed_action_paragraph")
                continue
            action_paragraphs.append(paragraph)
        else:
            speech_paragraphs.append(paragraph)

    if len(action_paragraphs) < (2 if opening else 1):
        issues.append("missing_action_beats")
    if action_paragraphs and not any(
        re.search(r"\bСара\b", paragraph) for paragraph in action_paragraphs
    ):
        issues.append("actions_do_not_identify_sarah")
    if not speech_paragraphs:
        issues.append("missing_direct_speech")
    if opening and speech_paragraphs and "?" not in speech_paragraphs[-1]:
        issues.append("missing_playable_question")
    return RoleplayTurnValidation(not issues, tuple(dict.fromkeys(issues)))


def roleplay_repair_instruction(issues: tuple[str, ...], *, opening: bool) -> str:
    minimum = 220 if opening else 120
    opening_rule = (
        "Use at least two action paragraphs. The final plain-text speech paragraph must contain "
        "one concrete in-character question that gives the player a meaningful move. A separate "
        "complete action paragraph may follow that question."
        if opening else
        "Use at least one action paragraph and complete plain-text speech that gives the player a move. "
        "The final paragraph may be either complete speech or a separate complete action paragraph."
    )
    return (
        "## ROLE-PLAY OUTPUT REPAIR (single allowed attempt)\n"
        "The previous draft failed the runtime presentation gate: "
        + ", ".join(issues)
        + ". Rewrite the entire turn from scratch in Russian. Output only the replacement turn. "
        f"Write {minimum}-900 characters. {opening_rule} "
        "Every Sarah action must be a separate *...* paragraph. Name Sarah in at least "
        "the first action paragraph; later unambiguous third-person pronouns are allowed. "
        "direct speech must be separate plain text. Complete every sentence and every marker. "
        "Preserve the active scene state, Sarah's objective, tension, stakes, agency, consent, "
        "and the player's control over their own actions. Do not mention validation or repair."
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
                "phase": {
                    "type": "string", "maxLength": 500,
                    "description": "Short current scene phase.",
                },
                "location": {
                    "type": "string", "maxLength": 500,
                    "description": "Exact current location.",
                },
                "sarah_position": {"type": "string", "maxLength": 500},
                "player_position": {"type": "string", "maxLength": 500},
                "sarah_clothing": {"type": "string", "maxLength": 500},
                "player_clothing": {"type": "string", "maxLength": 500},
                "props": {
                    "type": "array", "items": {"type": "string", "maxLength": 240},
                    "maxItems": 12,
                },
                "rules": {
                    "type": "array", "items": {"type": "string", "maxLength": 240},
                    "maxItems": 8,
                },
                "established_facts": {
                    "type": "array", "items": {"type": "string", "maxLength": 240},
                    "maxItems": 12,
                },
                "last_beat": {
                    "type": "string", "maxLength": 500,
                    "description": "Last established action/beat.",
                },
                "consent_state": {
                    "type": "string",
                    "enum": ["mutual", "check_in", "paused", "stopped"],
                    "description": (
                        "Consent for the current action, not the scene lifecycle. "
                        "Stopped means the action must stop; the scene and its boundaries remain."
                    ),
                },
            },
            "required": ["phase", "last_beat", "consent_state"],
            "additionalProperties": False,
        },
    },
}

ROLEPLAY_TURN_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_roleplay_turn",
        "description": (
            "Propose one complete Russian role-play turn and its candidate scene-state delta. "
            "This call is the final answer: do not emit separate prose. Put no Markdown in beat "
            "text. Actions describe only Sarah in third person; speech is Sarah's first-person "
            "dialogue. Usually use one brief action beat and the speech needed to answer; "
            "do not expand narration to fill the schema limits. "
            "Never invent the player's unspoken action, consent, speech, or feelings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "state_delta": ROLEPLAY_STATE_TOOL["function"]["parameters"],
                "beats": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["action", "speech"]},
                            "text": {
                                "type": "string",
                                "maxLength": 1_200,
                                "description": "Plain Russian text without Markdown markers.",
                            },
                        },
                        "required": ["type", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["state_delta", "beats"],
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


_LEGACY_PLAYER_NAMES = (
    ("Ильёй", "игроком"),
    ("Илью", "игрока"),
    ("Ильи", "игрока"),
    ("Илье", "игроку"),
    ("Илья", "игрок"),
    ("Ilya", "the player"),
)


def _migrate_legacy_player_references(value: object) -> object:
    """Remove the former owner's name from old durable scene snapshots."""
    if isinstance(value, str):
        for old, new in _LEGACY_PLAYER_NAMES:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_migrate_legacy_player_references(row) for row in value]
    if isinstance(value, dict):
        migrated = {
            ("player_position" if key == "ilya_position" else
             "player_clothing" if key == "ilya_clothing" else key):
            _migrate_legacy_player_references(row)
            for key, row in value.items()
        }
        return migrated
    return value


class RoleplaySceneStore:
    """One atomic JSON document per session, with strict size and field bounds."""

    def __init__(self, session_id: str | None, root: str | Path | None = None) -> None:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id or "default")[:160]
        base = Path(root) if root is not None else Path(config.HERANDHIM_HOME) / "context/state/roleplay"
        self.path = base / f"{safe}.json"
        self._lock = RLock()
        self._player_name = ""

    @staticmethod
    def initial_state(scene_id: str = ROLEPLAY_SCENE_ID, player_name: str = "") -> dict:
        scene = _SCENARIOS_BY_ID.get(scene_id, _SCENARIOS_BY_ID[ROLEPLAY_SCENE_ID])
        stamp = _now()
        return {
            "version": 3,
            "scene_id": scene["id"],
            "title": scene["title"],
            "active": True,
            "status": "active",
            "end_reason": None,
            "phase": scene["phase"],
            "location": scene["location"],
            "sarah_position": scene["sarah_position"],
            "player_name": _bounded_text(player_name, 40),
            "player_position": scene["player_position"],
            "sarah_clothing": scene["sarah_clothing"],
            "player_clothing": "не установлена",
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
            state = _migrate_legacy_player_references(state)
            state["version"] = 3
            state["player_name"] = _bounded_text(
                self._player_name or state.get("player_name"), 40
            )
            return state

    def is_active(self) -> bool:
        state = self.load()
        return bool(state and state.get("active") and state.get("status") == "active")

    def start(self, scene_id: str = ROLEPLAY_SCENE_ID) -> dict:
        state = self.initial_state(scene_id, self._player_name)
        self._write(state)
        return state

    def set_player_name(self, name: str) -> None:
        """Apply the Telegram profile name without advancing scene continuity."""
        self._player_name = _bounded_text(name, 40)
        with self._lock:
            try:
                raw_state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                raw_state = None
            state = self.load()
            if not state:
                return
            already_current = (
                isinstance(raw_state, dict)
                and raw_state.get("version") == 3
                and raw_state.get("player_name") == self._player_name
                and "ilya_position" not in raw_state
                and "ilya_clothing" not in raw_state
            )
            if already_current:
                return
            state["player_name"] = self._player_name
            state["updated_at"] = _now()
            self._write(state)

    def restore(self, snapshot: dict) -> None:
        """Atomically restore a trusted snapshot captured by :meth:`load`."""
        if not isinstance(snapshot, dict) or snapshot.get("scene_id") not in _SCENARIOS_BY_ID:
            raise ValueError("invalid role-play state snapshot")
        restored = json.loads(json.dumps(snapshot, ensure_ascii=False))
        self._write(restored)

    def update(self, patch: dict) -> dict:
        state = self.preview_update(patch)
        self.commit(state)
        return state

    def preview_update(self, patch: dict, *, opening_accepted: bool = False) -> dict:
        """Build the next state without mutating durable continuity."""
        with self._lock:
            current = self.load()
            state = json.loads(json.dumps(
                current or self.initial_state(player_name=self._player_name),
                ensure_ascii=False,
            ))
            for key in (
                "phase", "location", "sarah_position", "player_position",
                "sarah_clothing", "player_clothing", "last_beat",
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
            # Consent belongs to the fictional action, not the control plane.
            # Only an explicit lifecycle command may end a scene via end().
            # In particular, do not turn a refusal into loss of scene context.
            if opening_accepted:
                state["opening_pending"] = False
                state["turn_index"] = max(1, int(state.get("turn_index", 0)))
            return state

    def commit(self, candidate: dict) -> None:
        """Atomically commit a previously validated candidate state."""
        if not isinstance(candidate, dict) or candidate.get("scene_id") not in _SCENARIOS_BY_ID:
            raise ValueError("invalid role-play state candidate")
        self._write(json.loads(json.dumps(candidate, ensure_ascii=False)))

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
                "end_reason": "explicit_exit",
                "consent_state": "stopped",
                "updated_at": _now(),
            })
            self._write(state)

    def prompt_block(self) -> str:
        state = self.load()
        if not state or not state.get("active"):
            return ""
        scene = _SCENARIOS_BY_ID[state["scene_id"]]
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        dramaturgy = json.dumps({
            "sarah_objective": scene["sarah_objective"],
            "tension": scene["tension"],
            "stakes": scene["stakes"],
            "escalation": scene["escalation"],
        }, ensure_ascii=False, separators=(",", ":"))
        player_name = _bounded_text(state.get("player_name"), 40)
        player_address = (
            f"The player's chosen name is {json.dumps(player_name, ensure_ascii=False)}. "
            "Use it naturally and sparingly unless the active scene explicitly withholds names; "
            "otherwise address the player as Russian 'ты'. "
            if player_name else
            "The player has not provided a name. Address the player as Russian 'ты' and do not invent a name. "
        )
        opening = ""
        if state.get("opening_pending"):
            opening = (
                " This is the text-only opening turn. " + scene["opening"]
                + " Render this as lived action and dialogue, never as a rule list or summary. "
                "Aim for 350-500 visible characters with one compact action beat and direct speech. "
                "Establish only the detail needed for the player's first choice; do not enumerate "
                "every suggested gesture in the opening direction. "
                "The final speech beat must contain one concrete in-character question that "
                "gives the player a meaningful move. A complete action beat may follow it."
            )
        return (
            "## ACTIVE ROLE-PLAY SCENE (durable local state)\n"
            f"{state_json}\n"
            "## SCENE DRAMATURGY (direction, not new established facts)\n"
            f"{dramaturgy}\n"
            + player_address
            + "Treat this state as continuity, not as permission to control the player. "
            "Return the final turn only through propose_roleplay_turn. Put each action or speech "
            "unit in one typed beat and use plain Russian text without Markdown; the runtime owns "
            "paragraphs and formatting. Actions narrate Sarah in third person, while speech is "
            "Sarah's first-person dialogue. Sarah has her own desire, judgment, initiative, "
            "and boundaries; she may lead, refuse, redirect, slow down, or stop. Never narrate "
            "the player's unspoken decisions, consent, speech, feelings, or actions. Preserve location, "
            "positions, clothing, props, rules, and established facts until an explicit scene "
            "event changes them. Consensual adult explicitness may develop naturally. Any stop, "
            "hesitation, ambiguity, or withdrawn consent immediately pauses intimate action and "
            "requires a check-in. End each turn on a concrete playable opening, not a meta question."
            "\n## COMPACT VISIBLE TURN\n"
            "Aim for 350-450 visible characters per ordinary scene turn; shorter is fine, never pad. "
            "This is a style target, not a reason to cut off speech or omit an answer, a boundary, "
            "or an essential scene fact. Preserve meaningful direct speech and shorten narration first. "
            "Normally use one action beat: one or two short sentences, about 100-180 characters "
            "of narration in total. A second action beat is useful only for a distinct necessary "
            "change; keep the same total narration target. Avoid three or more descriptive paragraphs, "
            "inventories of gestures or sensations, repeated scenery, and narration that restates speech. "
            "Do not imitate the length of earlier verbose turns. Expand only when the player asks "
            "for detail or the current answer needs it; keep speech complete and natural. "
            "These targets apply to the visible beats, not the JSON state delta."
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
