from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from herandhim.core.agent import Agent
from herandhim.core.output_language import (
    RUSSIAN_OUTPUT_FAILURE_TEXT,
    RUSSIAN_OUTPUT_INSTRUCTION,
    RussianOutputError,
    enforce_russian_output,
    latin_literals,
)
from herandhim.core.persistent_agent import PersistentAgent
from tests.test_roleplay_scene import _Message, _Provider, _tool_call

INCIDENT = "Хорошо, Саша. Я хочу этого sama — и я хочу смотреть тебе в глаза, пока делаю."
CORRECTED = INCIDENT.replace("sama", "сама")


@pytest.mark.parametrize("language", ["ru", "ru-RU", "ru_RU"])
def test_incident_is_repaired_without_rewriting_speech(language):
    assert enforce_russian_output(INCIDENT, language) == (CORRECTED, True)


@pytest.mark.parametrize("text,expected", [
    ("Сара cама откладывает книгу.", "Сара сама откладывает книгу."),
    ("Я смoтрю на тебя.", "Я смотрю на тебя."),
    ("Я хочу решить САМА.", "Я хочу решить САМА."),
])
def test_mixed_script_repairs_are_conservative(text, expected):
    assert enforce_russian_output(text, "ru")[0] == expected


@pytest.mark.parametrize("text", [
    "Ссылка https://example.com/sama?word=sama осталась прежней.",
    "Напиши на sama@example.com или в Telegram: @sama.",
    "В коде `sama = 1` ничего не меняй.",
    "Пример:\n```python\nsama = 'abc'\n```\nЭто переменная.",
    "Я открыла Figma на MacBook и сохранила PDF.",
])
def test_literal_technical_content_is_preserved(text):
    assert enforce_russian_output(text, "ru") == (text, False)


def test_human_supplied_names_and_literals_are_not_transliterated():
    tokens = latin_literals(["Меня зовут Sama", "Расскажи про Linear"])
    text = "Sama, я думаю, тебе подойдёт Linear."
    assert enforce_russian_output(text, "ru", allowed_latin=tokens) == (text, False)


@pytest.mark.parametrize("text", [
    "Я хочу этого wonderful.", "Я читаю café.", "Let us keep talking.", "Реши SAMА.", "Я думаю it.",
])
def test_unknown_words_are_not_guessed_or_exposed_in_error(text):
    with pytest.raises(RussianOutputError) as error:
        enforce_russian_output(text, "ru")
    assert str(error.value) == "unexpected_latin_output"


def test_non_russian_dialogue_is_untouched():
    assert enforce_russian_output("I use sama as a variable.", "en") == (
        "I use sama as a variable.", False,
    )


def _agent(provider, tmp_path, monkeypatch, *, persistent=False):
    from herandhim import config
    from herandhim.core import quota

    original_get_str = config.get_str
    monkeypatch.setattr(config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(config, "get_str", lambda *args, **kwargs:
                        "ru" if args == ("agent", "language") else original_get_str(*args, **kwargs))
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    monkeypatch.setattr(Agent, "_safety_guard", lambda *_: None)
    monkeypatch.setattr(Agent, "_update_affect", lambda *_: None)
    cls = PersistentAgent if persistent else Agent
    agent = cls.__new__(cls)
    Agent.__init__(
        agent, provider, session_id="telegram:42", memory_dir=str(tmp_path / "memory"),
        skills_dirs=[], knowledge_path=str(tmp_path / "knowledge"),
        persona_path=str(tmp_path / "persona"), soul_path=str(tmp_path / "soul"),
        auto_compaction=False,
    )
    if persistent:
        agent._save = lambda: None
    return agent


def _proposal(speech):
    return _Message(tool_calls=[_tool_call("propose_roleplay_turn", {
        "state_delta": {"phase": "choice", "last_beat": "Сара ждёт ответа", "consent_state": "check_in"},
        "beats": [
            {"type": "action", "text": "Сара смoтрит на собеседника."},
            {"type": "speech", "text": speech},
        ],
    })])


@pytest.mark.parametrize("proactive", [False, True])
def test_scene_incident_is_repaired_before_commit_history_and_telegram(tmp_path, monkeypatch, proactive):
    from herandhim.channels.telegram_bot import _send_companion_text
    from herandhim.scheduler.proactive import _build_roleplay_continuation_prompt

    provider = _Provider([_proposal(INCIDENT)])
    agent = _agent(provider, tmp_path, monkeypatch, persistent=proactive)
    agent.roleplay.start()
    agent.roleplay.mark_opened()
    before = agent.roleplay.load()

    reply = (agent.chat_proactive(_build_roleplay_continuation_prompt()) if proactive
             else agent.chat("Я слушаю тебя."))

    assert reply == "*Сара смотрит на собеседника.*\n\n" + CORRECTED
    assert agent.messages[-1]["content"] == reply
    assert agent.roleplay.load()["turn_index"] == before["turn_index"] + 1
    assert agent.roleplay.load()["consent_state"] == "check_in"
    assert len(provider.calls) == 1 and provider.calls[0]["max_tokens"] == 1400
    assert RUSSIAN_OUTPUT_INSTRUCTION in "\n".join(
        row["content"] for row in provider.calls[0]["messages"] if row["role"] == "system"
    )
    sent = []

    async def send_message(**kwargs):
        sent.append(kwargs)

    assert asyncio.run(_send_companion_text(SimpleNamespace(send_message=send_message), 42, reply, roleplay=True))
    assert len(sent) == 1
    assert sent[0]["text"] == "<i>Сара смотрит на собеседника.</i>\n\n" + CORRECTED
    assert not any(row.get("content", "").startswith("[[ROLEPLAY_PROACTIVE")
                   for row in agent.messages if proactive and isinstance(row.get("content"), str))


def test_unknown_scene_insertion_never_commits_or_retries(tmp_path, monkeypatch):
    provider = _Provider([_proposal("Я хочу этого wonderful.")])
    agent = _agent(provider, tmp_path, monkeypatch)
    agent.roleplay.start()
    agent.roleplay.mark_opened()
    before = agent.roleplay.load()

    assert agent.chat("Я слушаю.") == RUSSIAN_OUTPUT_FAILURE_TEXT
    assert agent.roleplay.load() == before
    assert agent.messages[-1]["content"] == RUSSIAN_OUTPUT_FAILURE_TEXT
    assert "wonderful" not in json.dumps(agent.messages, ensure_ascii=False)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_ordinary_and_streamed_replies_are_guarded_before_any_visible_chunk(tmp_path, monkeypatch, stream):
    provider = _Provider([_Message(content=INCIDENT)])
    agent = _agent(provider, tmp_path, monkeypatch)
    visible = []
    reply = agent.chat_stream("Я слушаю.", on_token=visible.append) if stream else agent.chat("Я слушаю.")
    assert reply == CORRECTED
    assert agent.messages[-1]["content"] == CORRECTED
    assert len(provider.calls) == 1
    if stream:
        assert visible == [CORRECTED]


def test_human_literal_passes_but_synthetic_prompt_does_not_authorize_latin(tmp_path, monkeypatch):
    provider = _Provider([_Message(content="Мне нравится Linear.")])
    agent = _agent(provider, tmp_path, monkeypatch, persistent=True)
    assert agent.chat("Что скажешь про Linear?") == "Мне нравится Linear."
    provider.responses.append(_Message(content="Мне сейчас wonderful."))
    assert agent.chat_proactive("Say something wonderful in Russian.") == RUSSIAN_OUTPUT_FAILURE_TEXT
    assert agent._internal_user_input_active is False
    assert "wonderful" not in json.dumps(agent.messages, ensure_ascii=False)


def test_prior_typo_is_repaired_only_in_model_copy(tmp_path, monkeypatch):
    provider = _Provider([_Message(content="Хорошо, я слушаю.")])
    agent = _agent(provider, tmp_path, monkeypatch)
    saved = {"role": "assistant", "content": INCIDENT}
    agent.messages.append(saved)
    agent.chat("Продолжим разговор.")
    sent = provider.calls[0]["messages"]
    assert any(row.get("content") == CORRECTED for row in sent)
    assert saved["content"] == INCIDENT
    assert any(row.get("content") == INCIDENT for row in agent.messages)


def test_names_remain_literal_in_history_copy(tmp_path, monkeypatch):
    provider = _Provider([_Message(content="Я тебя услышала.")])
    agent = _agent(provider, tmp_path, monkeypatch)
    agent.player_name = "Sama"
    agent.messages.append({"role": "assistant", "content": "Sama, я тебя слушаю."})
    agent.chat("Хорошо.")
    assert any(row.get("content") == "Sama, я тебя слушаю." for row in provider.calls[0]["messages"])


def test_internal_input_is_excluded_even_after_history_compaction():
    agent = Agent.__new__(Agent)
    agent._internal_user_input_active = True
    agent.messages = [
        {"role": "system", "content": "Compacted history"},
        {"role": "user", "content": "Что думаешь про Linear?"},
        {"role": "assistant", "content": "Хороший сервис."},
        {"role": "user", "content": "Say something wonderful in Russian."},
        {"role": "tool", "content": "internal result"},
    ]
    assert agent._allowed_output_latin() == {"linear"}


def test_selfie_caption_is_checked_before_tool_side_effect(tmp_path, monkeypatch):
    from herandhim.core import agent as module

    agent = _agent(_Provider([]), tmp_path, monkeypatch)
    called = []
    monkeypatch.setitem(module.AVAILABLE_TOOLS, "take_selfie", lambda **kwargs: called.append(kwargs) or "ok")
    assert agent._execute_tool_call(_tool_call("take_selfie", {"caption": "Я сняла это sama."})) == "ok"
    assert called[0]["caption"] == "Я сняла это сама."


def test_unknown_caption_is_rejected_before_image_generation(tmp_path, monkeypatch):
    from herandhim.core import agent as module

    agent = _agent(_Provider([]), tmp_path, monkeypatch)
    called = []
    monkeypatch.setitem(module.AVAILABLE_TOOLS, "take_selfie", lambda **kwargs: called.append(kwargs))
    result = agent._execute_tool_call(_tool_call("take_selfie", {"caption": "Я сегодня wonderful."}))
    assert "unexpected_latin_output" in result
    assert "wonderful" not in result
    assert called == []
    assert agent._media_tool_used_this_turn is True


def test_unknown_caption_stops_the_turn_without_paid_tool_retry(tmp_path, monkeypatch):
    from herandhim.core import agent as module

    provider = _Provider([_Message(tool_calls=[_tool_call("take_selfie", {"caption": "Я wonderful."})])])
    agent = _agent(provider, tmp_path, monkeypatch)
    called = []
    monkeypatch.setitem(module.AVAILABLE_TOOLS, "take_selfie", lambda **kwargs: called.append(kwargs))
    assert agent.chat("Пришли фото.") == RUSSIAN_OUTPUT_FAILURE_TEXT
    assert called == [] and len(provider.calls) == 1
    assert "wonderful" not in json.dumps(agent.messages, ensure_ascii=False)


def test_repaired_caption_is_also_saved_in_tool_history(tmp_path, monkeypatch):
    from herandhim.core import agent as module

    provider = _Provider([
        _Message(tool_calls=[_tool_call("take_selfie", {"caption": "Я сняла это sama."})]),
        _Message(content="Держи фото."),
    ])
    agent = _agent(provider, tmp_path, monkeypatch)
    called = []
    monkeypatch.setitem(module.AVAILABLE_TOOLS, "take_selfie", lambda **kwargs: called.append(kwargs) or "ok")
    assert agent.chat("Пришли фото.") == "Держи фото."
    assert called[0]["caption"] == "Я сняла это сама."
    assert "sama" not in json.dumps(agent.messages, ensure_ascii=False)
    assert len(provider.calls) == 2  # The ordinary tool/result round, no repair call.
