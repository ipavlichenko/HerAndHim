from __future__ import annotations

import asyncio
import json
from threading import Lock
from types import SimpleNamespace

from herandhim.core.agent import Agent
from herandhim.core.roleplay import (
    ROLEPLAY_SCENE_ID,
    ROLEPLAY_SELFIE_HINT,
    ROLEPLAY_START_REQUEST,
    RoleplaySceneStore,
    is_roleplay_end,
    is_roleplay_start,
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


def test_scene_start_and_explicit_exit_are_distinct_from_in_scene_stop():
    assert is_roleplay_start(ROLEPLAY_START_REQUEST)
    assert is_roleplay_end("Выйдем из роли")
    assert is_roleplay_end("Закончим сцену.")
    assert not is_roleplay_end("Стоп")


def test_active_scene_forces_state_update_tool_and_keeps_full_output_budget(tmp_path):
    agent = _secured_agent(tmp_path)
    agent.roleplay.start()

    tools = agent._build_tools_for_turn("Я захожу и закрываю дверь")

    assert [row["function"]["name"] for row in tools] == ["update_roleplay_scene"]
    assert agent.roleplay.is_active()


def test_roleplay_start_forces_selfie_and_runtime_overrides_model_arguments(
    tmp_path, monkeypatch,
):
    from herandhim.core import agent as agent_module

    captured = []
    monkeypatch.setitem(
        agent_module.AVAILABLE_TOOLS,
        "take_selfie",
        lambda **kwargs: captured.append(kwargs) or "sent",
    )
    agent = _secured_agent(tmp_path)
    agent.session_id = "telegram:42"
    agent.verbose = False
    agent._media_tool_lock = Lock()
    agent._media_tool_used_this_turn = False
    agent._roleplay_started_this_turn = True
    call = SimpleNamespace(function=SimpleNamespace(
        name="take_selfie",
        arguments='{"scene_hint":"model drift","caption":"model drift"}',
    ))

    assert Agent._initial_tool_choice(ROLEPLAY_START_REQUEST) == {
        "type": "function", "function": {"name": "take_selfie"},
    }
    assert agent._execute_tool_call(call) == "sent"
    assert captured == [{
        "scene_hint": ROLEPLAY_SELFIE_HINT,
        "caption": "",
        "session_id": "telegram:42",
    }]


def test_scene_context_preserves_agency_continuity_and_player_control(tmp_path):
    store = RoleplaySceneStore("telegram:42", root=tmp_path)
    store.start()

    block = store.prompt_block()

    assert "ACTIVE ROLE-PLAY SCENE" in block
    assert "Sarah has her own desire" in block
    assert "Never narrate Ilya's unspoken" in block
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


def test_active_turn_updates_state_before_prose_and_redacts_trace(
    tmp_path, monkeypatch,
):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    sensitive_beat = "частный интимный такт"
    provider = _Provider([
        _Message(tool_calls=[_tool_call("update_roleplay_scene", {
            "phase": "hallway",
            "last_beat": sensitive_beat,
            "consent_state": "mutual",
        })]),
        _Message(content="*Сара остаётся у двери.* Заходи."),
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

    assert reply == "*Сара остаётся у двери.* Заходи."
    dialogue_calls = [call for call in provider.calls if call.get("purpose") == "dialogue"]
    assert len(dialogue_calls) == 2
    assert dialogue_calls[0]["tool_choice"]["function"]["name"] == "update_roleplay_scene"
    assert dialogue_calls[1]["tool_choice"] == "none"
    assert dialogue_calls[1]["tools"] == []
    assert agent.roleplay.load()["last_beat"] == sensitive_beat
    assert sensitive_beat not in json.dumps(trace, ensure_ascii=False)
    tool_trace = next(row for row in trace if row.get("event") == "tool_calls")
    assert tool_trace["calls"][0]["args"] == "[redacted-scene-state]"


def test_start_transaction_sends_fixed_door_photo_then_opening_prose(
    tmp_path, monkeypatch,
):
    from herandhim.core import agent as agent_module
    from herandhim.core import quota

    sent = []
    monkeypatch.setitem(
        agent_module.AVAILABLE_TOOLS,
        "take_selfie",
        lambda **kwargs: sent.append(kwargs) or "photo delivered",
    )
    monkeypatch.setattr(agent_module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    provider = _Provider([
        _Message(tool_calls=[_tool_call("take_selfie", {
            "scene_hint": "wrong scene",
            "caption": "wrong caption",
        })]),
        _Message(content="*Я остаюсь у открытой двери.* Ты опоздал на семь минут."),
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
    assert len(dialogue_calls) == 2
    assert dialogue_calls[0]["tool_choice"]["function"]["name"] == "take_selfie"
    assert dialogue_calls[1]["tool_choice"] == "none"
    assert sent == [{
        "scene_hint": ROLEPLAY_SELFIE_HINT,
        "caption": "",
        "session_id": "telegram:42",
    }]
    assert "опоздал на семь" in reply
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
    text = "*Я открываю дверь.*\n\nСемь минут.\n\nЗаходи."

    assert asyncio.run(module._send_companion_text(bot, 42, text, roleplay=True))
    assert bot.sent == [{"chat_id": 42, "text": text}]
    assert burst_calls == []


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
