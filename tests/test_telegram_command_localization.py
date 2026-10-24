from __future__ import annotations

import asyncio
from types import SimpleNamespace

from herandhim.channels.telegram_bot import (
    _COST_CAP_FAILURE_TEXT,
    _MODEL_FAILURE_TEXT,
    _RATE_LIMIT_FAILURE_TEXT,
    TelegramBot,
    _expand_roleplay_request,
    _model_failure_text,
    _roleplay_exit_keyboard,
    _scenario_menu_keyboard,
)
from herandhim.core.roleplay import (
    ROLEPLAY_EXIT_BUTTON_TEXT,
    ROLEPLAY_SCENARIOS,
    roleplay_scenario_id,
)


class _Message:
    def __init__(self, text: str = "") -> None:
        self.replies: list[str] = []
        self.reply_kwargs: list[dict] = []
        self.text = text
        self.caption = None
        self.photo = []
        self.voice = None
        self.audio = None

    async def reply_text(self, text: str, **kwargs) -> None:
        self.replies.append(text)
        self.reply_kwargs.append(kwargs)


class _Memory:
    @staticmethod
    def list_all() -> list[str]:
        return ["one"]


class _Roleplay:
    def __init__(self) -> None:
        self.ended = False

    def end(self) -> None:
        self.ended = True


class _Agent:
    provider = object()
    loaded_skill_names = {"one"}
    memory = _Memory()
    messages = [{"role": "user", "content": "test"}]
    compaction_count = 2

    def __init__(self) -> None:
        self.roleplay = _Roleplay()

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


def test_rate_limit_message_explains_when_retry_is_useful():
    text = _model_failure_text("Error: provider_rate_limited")

    assert text == _RATE_LIMIT_FAILURE_TEXT
    assert "через минуту" in text
    assert "provider_rate_limited" not in text


def test_start_attaches_ten_scenario_buttons_without_photo_button():
    bot, _manager, update, context, message = _fixture()

    asyncio.run(bot._cmd_start(update, context))

    markup = message.reply_kwargs[-1]["reply_markup"]
    assert markup.is_persistent is True
    labels = [button.text for row in markup.keyboard for button in row]
    assert labels == [row["label"] for row in ROLEPLAY_SCENARIOS]
    assert len(labels) == 10
    assert "📸 Хочу твоё фото" not in labels


def test_scenario_buttons_become_typed_text_only_starts():
    for scene in ROLEPLAY_SCENARIOS:
        expanded = _expand_roleplay_request(scene["label"])
        assert roleplay_scenario_id(expanded) == scene["id"]
        assert "без обязательной фотографии" in expanded

    assert _expand_roleplay_request("обычное сообщение") == "обычное сообщение"


def test_scene_and_exit_keyboards_are_separate():
    menu_labels = [
        button.text for row in _scenario_menu_keyboard().keyboard for button in row
    ]
    exit_labels = [
        button.text for row in _roleplay_exit_keyboard().keyboard for button in row
    ]

    assert ROLEPLAY_EXIT_BUTTON_TEXT not in menu_labels
    assert exit_labels == [ROLEPLAY_EXIT_BUTTON_TEXT]


def test_exit_button_ends_scene_and_returns_to_scenario_menu():
    bot, manager, update, context, _message = _fixture()
    message = _Message(ROLEPLAY_EXIT_BUTTON_TEXT)
    update.message = message
    update.update_id = 77
    update.effective_chat.type = "private"
    bot._require_mention = False
    bot._bot_username = None
    bot._transport = SimpleNamespace(accept_update=lambda *_args: True)

    async def _save(_chat_id):
        return None

    bot._maybe_save_chat_id = _save
    context.bot = object()

    asyncio.run(bot._handle_message(update, context))

    assert manager.agent.roleplay.ended is True
    assert message.replies == ["Сцена завершена. Выберите следующий сценарий."]
    markup = message.reply_kwargs[-1]["reply_markup"]
    assert sum(len(row) for row in markup.keyboard) == 10
