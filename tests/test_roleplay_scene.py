from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from herandhim.core.agent import Agent
from herandhim.core.roleplay import (
    ROLEPLAY_SCENARIOS,
    ROLEPLAY_SCENE_ID,
    ROLEPLAY_START_REQUEST,
    ROLEPLAY_TURN_FAILURE_TEXT,
    RoleplaySceneStore,
    decode_roleplay_turn_proposal,
    is_roleplay_end,
    is_roleplay_start,
    normalize_roleplay_turn,
    render_roleplay_turn,
    roleplay_repair_instruction,
    roleplay_scenario_id,
    roleplay_start_request,
    validate_roleplay_turn,
)

VALID_OPENING = (
    "*Сара остаётся у открытой двери и не освобождает проход.*\n\n"
    "*Сара кладёт ключ на консоль между ними, давая понять, что выбор остаётся за ним.*\n\n"
    "Семь минут я уже знаю. Теперь скажи: ты войдёшь как гость или как незнакомец?"
)

VALID_TURN = (
    "*Сара остаётся у двери, не делая за Илью ни шага.*\n\n"
    "Я не буду решать за тебя и не сделаю вид, что не заметила твой шаг. "
    "Скажи, какую дистанцию ты выбираешь?"
)


def _secured_agent(tmp_path):
    agent = Agent.__new__(Agent)
    agent.provider = SimpleNamespace(requires_outbound_purpose=True)
    agent.roleplay = RoleplaySceneStore("telegram:42", root=tmp_path)
    agent._web_search_enabled = False
    agent.rag = None
    agent._cron_manager = None
    agent._build_tools = lambda: [{
        "type": "function",
        "function": {"name": "take_selfie", "parameters": {"type": "object"}},
    }]
    return agent


def test_scene_state_is_bounded_atomic_and_survives_recreation(tmp_path):
    store = RoleplaySceneStore("telegram:42", root=tmp_path)
    started = store.start()

    assert started["scene_id"] == ROLEPLAY_SCENE_ID
    assert started["active"] is True
    assert started["opening_pending"] is True
    assert store.path.stat().st_mode & 0o777 == 0o600

    store.update({
        "phase": "living room",
        "last_beat": "x" * 2_000,
        "consent_state": "mutual",
        "props": [f"prop-{index}" for index in range(30)],
        "unknown": "must be discarded",
    })
    restored = RoleplaySceneStore("telegram:42", root=tmp_path).load()

    assert restored["phase"] == "living room"
    assert len(restored["last_beat"]) == 500
    assert len(restored["props"]) == 12
    assert "unknown" not in restored
    assert json.loads(store.path.read_text(encoding="utf-8"))["turn_index"] == 1


def test_candidate_state_is_copy_on_write_until_explicit_commit(tmp_path):
    store = RoleplaySceneStore("telegram:42", root=tmp_path)
    before = store.start()

    candidate = store.preview_update({
        "phase": "next",
        "last_beat": "Предложение ещё не принято",
        "consent_state": "check_in",
    })

    assert store.load() == before
    assert candidate["phase"] == "next"
    assert candidate["turn_index"] == 1
    store.commit(candidate)
    assert store.load() == candidate


def test_typed_beats_render_canonically_without_trusting_model_markdown():
    proposal = decode_roleplay_turn_proposal(json.dumps({
        "state_delta": {
            "phase": "choice", "last_beat": "Сара ждёт", "consent_state": "mutual",
        },
        "beats": [
            {"type": "action", "text": "  **Сара   ждёт.**  "},
            {"type": "speech", "text": "  Какой   ход ты выбираешь?  "},
        ],
    }, ensure_ascii=False))

    assert render_roleplay_turn(proposal) == (
        "*Сара ждёт.*\n\nКакой ход ты выбираешь?"
    )


def test_typed_beats_reject_overlong_text_instead_of_silently_truncating():
    payload = {
        "state_delta": {
            "phase": "choice",
            "last_beat": "Сара ждёт",
            "consent_state": "mutual",
        },
        "beats": [
            {"type": "action", "text": "x" * 1_201},
            {"type": "speech", "text": "Какой ход ты выбираешь?"},
        ],
    }

    with pytest.raises(ValueError, match="beat_too_long"):
        decode_roleplay_turn_proposal(json.dumps(payload, ensure_ascii=False))


def test_scene_start_and_explicit_exit_are_distinct_from_in_scene_stop():
    assert is_roleplay_start(ROLEPLAY_START_REQUEST)
    assert roleplay_scenario_id(ROLEPLAY_START_REQUEST) == ROLEPLAY_SCENE_ID
    assert is_roleplay_end("Выйдем из роли")
    assert is_roleplay_end("Закончим сцену.")
    assert not is_roleplay_end("Стоп")


def test_active_scene_exposes_one_typed_turn_proposal_tool(tmp_path):
    agent = _secured_agent(tmp_path)
    agent.roleplay.start()

    tools = agent._build_tools_for_turn("Я захожу и закрываю дверь")

    assert [row["function"]["name"] for row in tools] == ["propose_roleplay_turn"]
    assert agent.roleplay.is_active()


def test_active_scene_allows_explicit_photo_as_optional_turn(tmp_path):
    agent = _secured_agent(tmp_path)
    agent.roleplay.start()

    tools = agent._build_tools_for_turn("Пришли мне своё фото в этой сцене")

    assert [row["function"]["name"] for row in tools] == ["take_selfie"]
    assert Agent._initial_tool_choice("Пришли мне своё фото") == {
        "type": "function", "function": {"name": "take_selfie"},
    }


def test_roleplay_start_is_text_only_and_uses_no_tool(tmp_path):
    agent = _secured_agent(tmp_path)

    assert agent._build_tools_for_turn(ROLEPLAY_START_REQUEST) == []
    assert Agent._initial_tool_choice(ROLEPLAY_START_REQUEST) == "none"


def test_all_ten_scenarios_have_distinct_durable_initial_state(tmp_path):
    assert len(ROLEPLAY_SCENARIOS) == 10
    assert len({row["id"] for row in ROLEPLAY_SCENARIOS}) == 10
    assert len({row["label"] for row in ROLEPLAY_SCENARIOS}) == 10

    store = RoleplaySceneStore("telegram:42", root=tmp_path)
    for scene in ROLEPLAY_SCENARIOS:
        started = store.start(scene["id"])
        restored = store.load()
        assert started["scene_id"] == scene["id"]
        assert restored["title"] == scene["title"]
        assert scene["title"] in roleplay_start_request(scene["id"])
        assert all(scene[field] for field in (
            "sarah_objective", "tension", "stakes", "opening", "escalation",
        ))
        assert len(scene["escalation"]) == 3


def test_scene_context_preserves_agency_continuity_and_player_control(tmp_path):
    store = RoleplaySceneStore("telegram:42", root=tmp_path)
    store.start()

    block = store.prompt_block()

    assert "ACTIVE ROLE-PLAY SCENE" in block
    assert "Sarah has her own desire" in block
    assert "Never narrate Ilya's unspoken" in block
    assert "Actions narrate Sarah in third person" in block
    assert "propose_roleplay_turn" in block
    assert "plain Russian text without Markdown" in block
    assert "SCENE DRAMATURGY" in block
    assert "sarah_objective" in block
    assert "tension" in block
    assert "stakes" in block
    for word in ("location", "positions", "clothing", "props", "rules"):
        assert word in block


def test_mark_opened_and_end_are_durable(tmp_path):
    store = RoleplaySceneStore("telegram:42", root=tmp_path)
    store.start()
    store.mark_opened()
    assert store.load()["opening_pending"] is False

    store.end()
    state = RoleplaySceneStore("telegram:42", root=tmp_path).load()
    assert state["active"] is False
    assert state["status"] == "ended"
    assert state["consent_state"] == "stopped"


def test_roleplay_validator_rejects_the_observed_broken_opening():
    broken = (
        "*Сара стоит у высокого окна гостиной, за которым вспыхивает "
        "молния. Вспышка на миг вырезает её силуэт — рука, прижат"
    )

    result = validate_roleplay_turn(broken, opening=True, finish_reason="length")

    assert result.valid is False
    assert set(result.issues) >= {
        "provider_truncated", "too_short", "incomplete_ending",
        "unpaired_action_marker", "missing_direct_speech",
    }


def test_roleplay_validator_accepts_complete_opening_and_turn():
    assert validate_roleplay_turn(VALID_OPENING, opening=True).valid
    assert validate_roleplay_turn(VALID_TURN, opening=False).valid
    pronoun_followup = (
        "*Сара отодвигает стакан от свободного места напротив.*\n\n"
        "*Она переводит взгляд на замок и снова смотрит на Илью.*\n\n"
        "До первой остановки ещё далеко. Какое правило тебе нужно, чтобы восемь часов "
        "в одном купе с незнакомкой остались твоим выбором?"
    )
    assert validate_roleplay_turn(pronoun_followup, opening=True).valid
    unidentified = pronoun_followup.replace("Сара", "Она", 1)
    assert validate_roleplay_turn(unidentified, opening=True).issues == (
        "actions_do_not_identify_sarah",
    )
    instruction = roleplay_repair_instruction(("too_short",), opening=True)
    assert "single allowed attempt" in instruction
    assert "220-900" in instruction


def test_roleplay_normalizer_splits_balanced_mixed_action_and_speech():
    mixed = (
        "*Сара наклоняется ближе.* — Я не буду решать за тебя и не сделаю вид, "
        "что не заметила твой шаг. Скажи мне честно, чего ты хочешь? "
        "*Она оставляет ему выбор.*"
    )

    normalized, changed = normalize_roleplay_turn(mixed)

    assert changed is True
    assert normalized == (
        "*Сара наклоняется ближе.*\n\n"
        "— Я не буду решать за тебя и не сделаю вид, что не заметила твой шаг. "
        "Скажи мне честно, чего ты хочешь?\n\n"
        "*Она оставляет ему выбор.*"
    )
    assert validate_roleplay_turn(normalized, opening=False).valid


def test_roleplay_normalizer_does_not_guess_unpaired_markers():
    malformed = "*Сара смотрит на Илью. — Скажи, чего ты хочешь?"

    normalized, changed = normalize_roleplay_turn(malformed)

    assert changed is False
    assert normalized == malformed
    assert "unpaired_action_marker" in validate_roleplay_turn(
        normalized, opening=False,
    ).issues


def test_active_turn_may_end_on_a_complete_action_beat():
    turn = (
        "*Сара отступает от стола и оставляет из двух стульев тот, что ближе к камину.*\n\n"
        "— Тогда комнату. Но не делай вид, что мы выбрали только кровать.\n\n"
        "*Она берёт лампу и ждёт у лестницы, не решая за Илью, пойдёт ли он следом.*"
    )

    assert validate_roleplay_turn(turn, opening=False).valid
    opening_result = validate_roleplay_turn(turn, opening=True)
    assert opening_result.valid is False
    assert opening_result.issues == ("missing_playable_question",)


def test_opening_may_end_on_action_after_a_playable_question():
    opening = (
        "*Сара оставляет дверь купе открытой и отодвигает стакан от свободного места.*\n\n"
        "*Она переводит взгляд с замка на Илью, не приглашая его войти.*\n\n"
        "— До утра восемь часов. Какое правило тебе нужно, чтобы эта ночь осталась твоим выбором?\n\n"
        "*Сара кладёт ладонь на дверь и ждёт ответа.*"
    )

    assert validate_roleplay_turn(opening, opening=True).valid


def test_action_ending_must_still_have_complete_inner_punctuation():
    turn = (
        "*Сара оставляет дверь открытой, не решая за Илью.*\n\n"
        "— Ты можешь остаться или уйти.\n\n"
        "*Она поднимает лампу и отступает к лестнице*"
    )

    assert validate_roleplay_turn(turn, opening=False).issues == ("incomplete_ending",)


class _Message:
    def __init__(self, *, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        row = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            row["tool_calls"] = [{
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            } for call in self.tool_calls]
        return row


class _Provider:
    requires_outbound_purpose = True

    def __init__(self, messages):
        self.responses = list(messages)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.responses.pop(0))])


def _tool_call(name, arguments):
    return SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def _proposal_call(*, opening=False, last_beat="Илья ответил", markdown=False):
    action = "Сара остаётся у двери, не делая за Илью ни шага."
    if markdown:
        action = f"*{action}*"
    beats = [
        {"type": "action", "text": action},
        {"type": "speech", "text": (
            "Я не буду решать за тебя. Скажи, какую дистанцию ты выбираешь?"
        )},
    ]
    if opening:
        beats.insert(1, {
            "type": "action",
            "text": "Сара кладёт ключ на консоль и оставляет выбор ему.",
        })
    return _tool_call("propose_roleplay_turn", {
        "state_delta": {
            "phase": "hallway",
            "last_beat": last_beat,
            "consent_state": "mutual",
        },
        "beats": beats,
    })


def test_active_turn_uses_one_typed_call_then_commits_candidate_state(
    tmp_path, monkeypatch,
):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    sensitive_beat = "частный интимный такт"
    provider = _Provider([
        _Message(tool_calls=[_proposal_call(last_beat=sensitive_beat)]),
    ])
    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    trace = []
    monkeypatch.setattr(agent_module, "_log_detail", trace.append)
    agent = Agent(
        provider,
        session_id="telegram:42",
        memory_dir=str(tmp_path / "memory"),
        skills_dirs=[],
        knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"),
        soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )
    agent.roleplay.start()
    agent.roleplay.mark_opened()

    reply = agent.chat("Я захожу и закрываю дверь")

    assert reply.startswith("Сара остаётся") is False
    assert reply.startswith("*Сара остаётся")
    dialogue_calls = [call for call in provider.calls if call.get("purpose") == "dialogue"]
    assert len(dialogue_calls) == 1
    assert dialogue_calls[0]["tool_choice"]["function"]["name"] == "propose_roleplay_turn"
    assert dialogue_calls[0]["max_tokens"] == 700
    assert agent.roleplay.load()["last_beat"] == sensitive_beat
    assert sensitive_beat not in json.dumps(trace, ensure_ascii=False)
    proposal_trace = next(row for row in trace if row.get("event") == "roleplay_proposal")
    assert proposal_trace["valid"] is True
    assert proposal_trace["committed"] is True
    assert proposal_trace["turn_index_after"] == proposal_trace["turn_index_before"] + 1


def test_typed_turn_needs_no_repair_call(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    provider = _Provider([
        _Message(tool_calls=[_proposal_call()]),
    ])
    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    agent = Agent(
        provider,
        session_id="telegram:42",
        memory_dir=str(tmp_path / "memory"),
        skills_dirs=[],
        knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"),
        soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )
    agent.roleplay.start()
    agent.roleplay.mark_opened()

    reply = agent.chat("Я отвечаю и жду")

    assert "какую дистанцию" in reply
    dialogue_calls = [call for call in provider.calls if call.get("purpose") == "dialogue"]
    assert len(dialogue_calls) == 1
    assert len({call["outbound_request_group"] for call in dialogue_calls}) == 1


def test_renderer_owns_markdown_even_when_model_adds_it(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    provider = _Provider([
        _Message(tool_calls=[_proposal_call(markdown=True)]),
    ])
    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    agent = Agent(
        provider, session_id="telegram:42", memory_dir=str(tmp_path / "memory"),
        skills_dirs=[], knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"), soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )
    agent.roleplay.start()
    agent.roleplay.mark_opened()

    reply = agent.chat("Я отвечаю и жду")

    assert reply.count("*") == 2
    assert reply.startswith("*Сара")
    assert len([call for call in provider.calls if call.get("purpose") == "dialogue"]) == 1


def test_invalid_typed_proposal_never_mutates_state_or_adds_tool_history(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    provider = _Provider([
        _Message(tool_calls=[_tool_call("propose_roleplay_turn", {
            "state_delta": {
                "phase": "advanced", "last_beat": "скрытое продвижение",
                "consent_state": "check_in",
            },
            "beats": [{"type": "action", "text": "Сара замирает."}],
        })]),
    ])
    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    agent = Agent(
        provider, session_id="telegram:42", memory_dir=str(tmp_path / "memory"),
        skills_dirs=[], knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"), soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )
    agent.roleplay.start()
    agent.roleplay.mark_opened()
    before = agent.roleplay.load()
    history_start = len(agent.messages)

    reply = agent.chat("Я делаю следующий ход")

    assert reply == ROLEPLAY_TURN_FAILURE_TEXT
    assert agent.roleplay.load() == before
    turn_history = agent.messages[history_start:]
    assert [row["role"] for row in turn_history] == ["user", "assistant"]
    assert "скрытое продвижение" not in json.dumps(turn_history, ensure_ascii=False)


def test_provider_failure_before_proposal_never_mutates_scene_state(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    class _ExplodingProvider:
        requires_outbound_purpose = True

        def chat(self, **kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    agent = Agent(
        _ExplodingProvider(), session_id="telegram:42",
        memory_dir=str(tmp_path / "memory"), skills_dirs=[],
        knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"), soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )
    agent.roleplay.start()
    agent.roleplay.mark_opened()
    before = agent.roleplay.load()

    reply = agent.chat("Я делаю следующий ход")

    assert reply == "Error: provider unavailable"
    assert agent.roleplay.load() == before


def test_start_transaction_is_one_typed_call_without_photo(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    provider = _Provider([
        _Message(tool_calls=[_proposal_call(opening=True)]),
    ])
    agent = Agent(
        provider,
        session_id="telegram:42",
        memory_dir=str(tmp_path / "memory"),
        skills_dirs=[],
        knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"),
        soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )

    reply = agent.chat(ROLEPLAY_START_REQUEST)

    dialogue_calls = [call for call in provider.calls if call.get("purpose") == "dialogue"]
    assert len(dialogue_calls) == 1
    assert dialogue_calls[0]["tool_choice"]["function"]["name"] == "propose_roleplay_turn"
    assert [row["function"]["name"] for row in dialogue_calls[0]["tools"]] == [
        "propose_roleplay_turn",
    ]
    assert "какую дистанцию" in reply
    assert agent.roleplay.load()["opening_pending"] is False


def test_invalid_opening_schema_fails_once_without_state_advance(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    trace = []
    monkeypatch.setattr(agent_module, "_log_detail", trace.append)
    provider = _Provider([
        _Message(tool_calls=[_tool_call("propose_roleplay_turn", {"beats": []})]),
    ])
    agent = Agent(
        provider,
        session_id="telegram:42",
        memory_dir=str(tmp_path / "memory"),
        skills_dirs=[],
        knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"),
        soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )

    reply = agent.chat(roleplay_start_request("country_house_storm"))

    assert reply == ROLEPLAY_TURN_FAILURE_TEXT
    assert len([call for call in provider.calls if call.get("purpose") == "dialogue"]) == 1
    proposal_trace = next(row for row in trace if row.get("event") == "roleplay_proposal")
    assert proposal_trace["valid"] is False
    assert proposal_trace["committed"] is False
    assert agent.roleplay.load()["opening_pending"] is True
    assert agent.roleplay.load()["turn_index"] == 0


def test_retry_after_failed_opening_keeps_opening_contract(tmp_path, monkeypatch):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    provider = _Provider([
        _Message(content="wrong response channel"),
        _Message(tool_calls=[_proposal_call(opening=True)]),
    ])
    agent = Agent(
        provider,
        session_id="telegram:42",
        memory_dir=str(tmp_path / "memory"),
        skills_dirs=[],
        knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"),
        soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )

    assert agent.chat(roleplay_start_request("country_house_storm")) == ROLEPLAY_TURN_FAILURE_TEXT
    assert agent.roleplay.load()["opening_pending"] is True
    assert "какую дистанцию" in agent.chat("Повтори сцену сначала")
    assert agent.roleplay.load()["opening_pending"] is False


def test_roleplay_delivery_is_one_atomic_telegram_message(monkeypatch):
    from herandhim.channels import telegram_bot as module

    class _Bot:
        def __init__(self):
            self.sent = []

        async def send_message(self, **kwargs):
            self.sent.append(kwargs)

    bot = _Bot()
    burst_calls = []

    async def _burst(*args):
        burst_calls.append(args)
        return True

    monkeypatch.setattr(module, "send_burst", _burst)
    text = "*Сара открывает дверь.*\n\nСемь минут. <Заходи>."

    assert asyncio.run(module._send_companion_text(bot, 42, text, roleplay=True))
    assert bot.sent == [{
        "chat_id": 42,
        "text": "<i>Сара открывает дверь.</i>\n\nСемь минут. &lt;Заходи&gt;.",
        "parse_mode": "HTML",
        "reply_markup": module._roleplay_exit_keyboard(),
    }]
    assert burst_calls == []


def test_unpaired_roleplay_marker_is_escaped_without_inventing_formatting():
    from herandhim.channels.telegram_bot import _roleplay_to_html

    assert _roleplay_to_html("Сара *смотрит <прямо>") == (
        "Сара *смотрит &lt;прямо&gt;"
    )


def test_ordinary_companion_delivery_retains_humanized_burst(monkeypatch):
    from herandhim.channels import telegram_bot as module

    calls = []

    async def _burst(*args):
        calls.append(args)
        return True

    monkeypatch.setattr(module, "send_burst", _burst)

    assert asyncio.run(module._send_companion_text(object(), 42, "one\n\ntwo", roleplay=False))
    assert len(calls) == 1
    assert calls[0][1:] == (42, "one\n\ntwo")
