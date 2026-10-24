from __future__ import annotations

import asyncio
from types import SimpleNamespace

from herandhim.channels.telegram_bot import (
    _COST_CAP_FAILURE_TEXT,
    _MODEL_FAILURE_TEXT,
    _PHOTO_BUTTON_TEXT,
    TelegramBot,
    _expand_quick_action,
    _model_failure_text,
)


class _Message:
    def __init__(self) -> None:
        self.replies: list[str] = []
        self.reply_kwargs: list[dict] = []

    async def reply_text(self, text: str, **kwargs) -> None:
        self.replies.append(text)
        self.reply_kwargs.append(kwargs)


class _Memory:
    @staticmethod
    def list_all() -> list[str]:
        return ["one"]


class _Agent:
    provider = object()
    loaded_skill_names = {"one"}
    memory = _Memory()
    messages = [{"role": "user", "content": "test"}]
    compaction_count = 2

    def compact(self, instruction=None) -> str:
        return "English internal summary must not be exposed"


class _SessionManager:
    def __init__(self) -> None:
        self.agent = _Agent()
        self.reset_ids: list[str] = []

    def get_or_create(self, _sid: str):
        return self.agent

    def reset(self, sid: str) -> None:
        self.reset_ids.append(sid)

    def __len__(self) -> int:
        return 1


def _fixture():
    manager = _SessionManager()
    bot = TelegramBot.__new__(TelegramBot)
    bot._sm = manager
    bot._allowed_users = {42}
    message = _Message()
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=42),
        effective_chat=SimpleNamespace(id=42),
        message=message,
    )
    context = SimpleNamespace(args=[])
    return bot, manager, update, context, message


def test_clear_command_is_russian_only_and_preserves_memory_contract():
    bot, manager, update, context, message = _fixture()

    asyncio.run(bot._cmd_reset(update, context))

    assert manager.reset_ids == ["telegram:42"]
    assert message.replies == [
        "🧹 История текущего диалога очищена.\n"
        "Долгосрочная память, личность Сары и история ваших отношений сохранены."
    ]


def test_all_command_handlers_return_russian_service_messages(monkeypatch):
    bot, _manager, update, context, message = _fixture()
    monkeypatch.setattr("herandhim.config.clear_files", lambda: 3)

    for handler in (bot._cmd_start, bot._cmd_status, bot._cmd_compact, bot._cmd_clear_files):
        asyncio.run(handler(update, context))

    combined = "\n".join(message.replies)
    for forbidden in (
        "Hi!", "Commands:", "Session Status", "Compacting conversation",
        "Compaction complete", "Cleared ", "Chat history", "已清空",
    ):
        assert forbidden not in combined
    assert "Команды:" in combined
    assert "Состояние сессии" in combined
    assert "История диалога сжата" in combined
    assert "Удалено файлов: 3" in combined


def test_telegram_command_menu_descriptions_are_russian():
    descriptions = [command.description for command in TelegramBot._BOT_COMMANDS]

    assert len(descriptions) == 6
    assert all(any("а" <= char.lower() <= "я" for char in text) for text in descriptions)
    assert not any("已清空" in text for text in descriptions)


def test_model_failure_message_is_russian_and_hides_internal_error_code():
    assert "ответить" in _MODEL_FAILURE_TEXT
    assert "provider_empty_response" not in _MODEL_FAILURE_TEXT
    assert "Error:" not in _MODEL_FAILURE_TEXT


def test_cost_cap_message_does_not_invite_a_pointless_retry():
    text = _model_failure_text("Error: daily_cost_cap_exceeded")

    assert text == _COST_CAP_FAILURE_TEXT
    assert "Повторная отправка сейчас не поможет" in text
    assert "сброса дневного лимита" in text
    assert _model_failure_text("Error: provider_empty_response") == _MODEL_FAILURE_TEXT


def test_start_attaches_persistent_photo_button():
    bot, _manager, update, context, message = _fixture()

    asyncio.run(bot._cmd_start(update, context))

    markup = message.reply_kwargs[-1]["reply_markup"]
    assert markup.is_persistent is True
    assert markup.keyboard[0][0].text == _PHOTO_BUTTON_TEXT


def test_photo_button_becomes_context_continuation_request():
    expanded = _expand_quick_action(_PHOTO_BUTTON_TEXT)

    assert "своё фото" in expanded
    assert "текущую сцену" in expanded
    assert _expand_quick_action("обычное сообщение") == "обычное сообщение"
