"""
Telegram channel for herandhim — the companion's home on your own bot.

Long-polling bot: it connects out to Telegram (no public URL / webhook
needed), so it runs anywhere, including behind a home NAT. Delivery-side
human-feel (voice transcription, emoji reactions, message bursts, absence
awareness) lives in ``core/humanize.py`` and is applied here for companion
sessions.

Telegram is purely a *channel* — it handles sending and receiving messages.
Session lifecycle (which Agent handles which chat) is delegated to the
SessionManager, which is shared across all channels and the cron scheduler.

Session IDs used by this channel: "telegram:{chat_id}"

Commands
--------
  /start          — greeting + usage hint
  /reset          — discard and recreate the current session
  /status         — show session info (provider, skills, memory, tokens, compactions)
  /compact [hint] — compact conversation history
  <text>          — forwarded to Agent.chat(), reply sent back
  <photo>         — image sent to LLM with optional caption
  <voice>         — transcribed (core/stt.py) and answered; per
                    channels.telegram.replyInKindVoice the answer itself is
                    sent back as a voice note (core/tts.py), text on any miss

Access control
--------------
Set TELEGRAM_ALLOWED_USERS to a comma-separated list containing the owner's
integer Telegram user ID.  Empty or unset allowlists fail closed.

Group behaviour
---------------
Set ``channels.telegram.requireMention`` to ``true`` in herandhim.json to
require @bot mention in group chats.  DMs always respond.
"""

from __future__ import annotations

import asyncio
import base64
import html
import logging
import os
import queue as _queue
import re
import tempfile
import time
import unicodedata
from typing import TYPE_CHECKING

from telegram import (
    BotCommand,
    KeyboardButton,
    ReactionTypeEmoji,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .. import config
from ..core.humanize import humanize_gap, maybe_react, reply_delay
from ..core.roleplay import (
    ROLEPLAY_EXIT_BUTTON_TEXT,
    ROLEPLAY_SCENARIOS,
    roleplay_scenario_for_label,
    roleplay_start_request,
)

if TYPE_CHECKING:
    from ..core.telegram_transport import TelegramTransportStore
    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)

_PHOTO_FAILURE_TEXT = (
    "I couldn't safely process that image. Please send a valid photo under 4 MiB."
)
_VOICE_FAILURE_TEXT = "I couldn't safely process that voice note. Please try a shorter recording."
_VOICE_EMPTY_TEXT = "I couldn't recognise speech in that voice note."
_MODEL_FAILURE_TEXT = "Не получилось ответить. Попробуйте отправить сообщение позже."
_RATE_LIMIT_FAILURE_TEXT = (
    "Провайдер временно ограничил запросы. "
    "Повторите ход через минуту."
)
_COST_CAP_FAILURE_TEXT = (
    "На сегодня исчерпан лимит расходов на диалог. Повторная отправка сейчас "
    "не поможет; Сара снова сможет отвечать после сброса дневного лимита."
)
_VOICE_MAX_DURATION_SECONDS = 120
_VOICE_MAX_INPUT_BYTES = 8 * 1024 * 1024
_PLAYER_NAME_PROMPT = (
    "Как тебя зовут? Напиши имя или псевдоним, которым Сара может к тебе обращаться."
)
_PLAYER_NAME_INVALID = (
    "Не получилось распознать имя. Используй 1–40 букв, пробел, дефис или апостроф."
)


def _normalize_player_name(value: str) -> str | None:
    """Return a prompt-safe human name without guessing or transliteration."""
    name = " ".join(unicodedata.normalize("NFC", value or "").split())
    if not name or len(name) > 40:
        return None
    allowed_punctuation = {" ", "-", "'", "’", "ʼ"}
    has_letter = False
    for char in name:
        category = unicodedata.category(char)
        if category.startswith("L"):
            has_letter = True
            continue
        if category.startswith("M") or char in allowed_punctuation:
            continue
        return None
    return name if has_letter else None


def _set_agent_player_name(agent, name: str) -> None:
    setter = getattr(agent, "set_player_name", None)
    if callable(setter):
        setter(name)
        return
    roleplay = getattr(agent, "roleplay", None)
    roleplay_setter = getattr(roleplay, "set_player_name", None)
    if callable(roleplay_setter):
        roleplay_setter(name)


_HIDE_MENU_BUTTON_TEXT = "⌨️ Скрыть меню"


def _scenario_menu_keyboard() -> ReplyKeyboardMarkup:
    buttons = [KeyboardButton(row["label"]) for row in ROLEPLAY_SCENARIOS]
    rows = [buttons[index:index + 2] for index in range(0, len(buttons), 2)]
    rows.append([KeyboardButton(_HIDE_MENU_BUTTON_TEXT)])
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        is_persistent=False,
        input_field_placeholder="Выберите сценарий…",
    )


def _roleplay_exit_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton(ROLEPLAY_EXIT_BUTTON_TEXT)], [KeyboardButton(_HIDE_MENU_BUTTON_TEXT)]],
        resize_keyboard=True,
        is_persistent=False,
        input_field_placeholder="Ваш ход…",
    )


def _expand_roleplay_request(text: str) -> str:
    """Expand an exact scenario-menu selection into a typed start request."""
    scene = roleplay_scenario_for_label(text)
    if scene is not None:
        return roleplay_start_request(scene["id"])
    return text


def _model_failure_text(raw_response: str) -> str:
    """Map internal policy failures to truthful, content-free Russian UX."""
    if raw_response.startswith("Error: ") and "daily_cost_cap_exceeded" in raw_response:
        return _COST_CAP_FAILURE_TEXT
    if raw_response.startswith("Error: ") and "provider_rate_limited" in raw_response:
        return _RATE_LIMIT_FAILURE_TEXT
    return _MODEL_FAILURE_TEXT


def _roleplay_to_html(text: str) -> str:
    """Render paired role-play action markers as safe Telegram italics."""
    parts = text.split("*")
    if len(parts) % 2 == 0:
        return html.escape(text)
    rendered = []
    for index, part in enumerate(parts):
        escaped = html.escape(part)
        rendered.append(f"<i>{escaped}</i>" if index % 2 else escaped)
    return "".join(rendered)


async def _send_companion_text(
    bot, chat_id: int, text: str, *, roleplay: bool, keyboard_hidden: bool = False,
) -> bool:
    """Deliver one logical companion reply without conversational burst splitting.

    Scene state selects the keyboard, never the number of messages. A completed
    scene can still be discussed in ordinary chat; splitting that response makes
    one generated turn look like several unsolicited turns. Only the platform
    length limit may require multiple messages.
    """
    sent = False
    for chunk in _split_message(text):
        markup = {"reply_markup": _roleplay_exit_keyboard()} if roleplay else {}
        if keyboard_hidden:
            markup = {"reply_markup": ReplyKeyboardRemove()}
        await bot.send_message(
            chat_id=chat_id,
            text=_roleplay_to_html(chunk),
            parse_mode="HTML",
            **markup,
        )
        sent = True
    return sent


class TelegramBot:
    """
    Telegram channel — pure I/O layer.

    Receives messages from Telegram and routes them to the appropriate Agent
    via the shared SessionManager.  Does not own or manage Agent instances.
    """

    def __init__(
        self,
        session_manager: "SessionManager",
        token: str,
        allowed_users: list[int] | None = None,
        require_mention: bool = False,
        transport_store: "TelegramTransportStore | None" = None,
    ) -> None:
        from ..core.telegram_transport import TelegramTransportStore

        self._sm = session_manager
        self._token = token
        self._allowed_users: set[int] = set(allowed_users) if allowed_users else set()
        self._transport = transport_store or TelegramTransportStore()
        self._require_mention = require_mention
        self._app: Application | None = None
        self._bot_username: str | None = None
        # Once the chat_id is saved to local settings we don't try again,
        # even across reconnects within this process.
        self._chat_id_saved: bool = False
        # Invoked the first time we capture the chat_id, so the proactive
        # scheduler can register its job without a daemon restart.
        self._on_chat_id_learned = None
        # Per-chat timestamp of the previous inbound message — feeds the
        # companion's absence awareness (agent._gap_note).
        self._last_seen: dict[int, float] = {}

    # ── Session ID convention ─────────────────────────────────────────────────

    def _session_id(self, chat_id: int) -> str:
        """Session id for this chat.

        Partitioned by chat_id so one bot can serve several Telegram chats
        independently — each keeps its own history.
        """
        return f"telegram:{chat_id}"

    # ── Push message (called by cron / heartbeat) ─────────────────────────────

    async def send_message(
        self, chat_id: int, text: str, *, roleplay: bool = False,
    ) -> None:
        """Send a message to a specific chat (used by cron/heartbeat)."""
        if self._app is None:
            logger.warning("[Telegram] send_message called before bot is running")
            return
        await _send_companion_text(
            self._app.bot, chat_id, text, roleplay=roleplay,
            keyboard_hidden=self._transport.keyboard_hidden(self._session_id(chat_id)),
        )

    async def send_photo(
        self,
        chat_id: int,
        path: str,
        caption: str = "",
        *,
        origin: str = "explicit",
    ) -> bool:
        """Accept one owner photo into the durable outbox, then attempt it."""
        if chat_id not in self._allowed_users:
            raise PermissionError("Telegram photo target is not an allowlisted owner")
        session_id = self._session_id(chat_id)
        self._transport.enqueue_media(
            session_id, chat_id, path, caption, origin=origin, kind="photo",
        )
        if self._app is None:
            logger.warning("[Telegram] photo queued before bot is running")
            return False
        return await self._deliver_pending_media(session_id)

    async def _deliver_pending_photo(self, session_id: str) -> bool:
        """Backward-compatible alias for the unified durable media delivery."""
        return await self._deliver_pending_media(session_id)

    async def _deliver_pending_media(self, session_id: str) -> bool:
        """Claim before Telegram I/O; ambiguous failures are never retried."""
        if self._app is None:
            return False
        delivery = self._transport.claim_next_media(session_id)
        if delivery is None:
            return False
        if (delivery.chat_id not in self._allowed_users
                or delivery.session_id != self._session_id(delivery.chat_id)):
            logger.error("[Telegram] rejected non-owner media delivery %s", delivery.id)
            return False
        try:
            with open(delivery.path, "rb") as handle:
                if delivery.kind == "voice":
                    await self._app.bot.send_voice(
                        chat_id=delivery.chat_id,
                        voice=handle,
                        caption=delivery.caption or None,
                    )
                else:
                    await self._app.bot.send_photo(
                        chat_id=delivery.chat_id,
                        photo=handle,
                        caption=delivery.caption or None,
                    )
        except Exception as exc:
            logger.error("[Telegram] %s delivery attempt failed: %s", delivery.kind, exc)
            return False
        self._transport.mark_media_delivered(delivery.id)
        return True

    async def send_voice(
        self,
        chat_id: int,
        voice: bytes | str,
        caption: str = "",
        *,
        format: str | None = None,
        origin: str = "explicit",
    ) -> bool:
        """Accept one owner voice clip into the durable outbox, then attempt it."""
        if chat_id not in self._allowed_users:
            raise PermissionError("Telegram voice target is not an allowlisted owner")
        session_id = self._session_id(chat_id)
        temporary_path: str | None = None
        path = voice if isinstance(voice, str) else None
        if path is None:
            suffix = f".{(format or '').lower()}"
            if suffix not in {".ogg", ".opus", ".mp3", ".m4a"}:
                raise ValueError("raw Telegram voice bytes require an allowed format")
            handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                handle.write(voice)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = handle.name
            finally:
                handle.close()
            path = temporary_path
        try:
            self._transport.enqueue_media(
                session_id, chat_id, path, caption, origin=origin, kind="voice",
            )
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)
        if self._app is None:
            logger.warning("[Telegram] voice queued before bot is running")
            return False
        return await self._deliver_pending_media(session_id)

    # ── Access control ────────────────────────────────────────────────────────

    def _is_allowed(self, user_id: int) -> bool:
        return bool(self._allowed_users) and user_id in self._allowed_users

    async def _remember_their_photo(self, agent, vision_reply: str) -> None:
        """File a short local memory from the already completed vision turn.

        D-029 forbids sending the same inbound image through a second cloud
        vision request merely to create a memory.
        """
        try:
            from datetime import datetime as _dt
            words = str(vision_reply or "").split()
            desc = " ".join(words[:15]).strip()
            if not desc:
                return
            stamp = _dt.now().strftime("%Y-%m-%d_%H%M%S")
            agent.memory.remember(
                f"Photo they sent me on {stamp[:10]}: {desc}",
                key=f"their_photo_{stamp}",
            )
            logger.info("[Telegram] remembered their photo: %s", desc[:60])
        except Exception as exc:
            logger.debug("[Telegram] photo memory skipped: %s", exc)

    async def _maybe_save_chat_id(self, chat_id: int) -> None:
        """Learn and persist the user's Telegram chat_id to the local config
        the first time they message the bot — it's what the proactive
        scheduler uses to send unprompted messages. Idempotent per process."""
        if self._chat_id_saved:
            return
        try:
            from ..config import get as _cfg_get
            if _cfg_get("channels", "telegram", "chatId", default=None):
                self._chat_id_saved = True
                return
            from ..core import local_settings
            local_settings.set_telegram(chat_id=chat_id)  # token untouched
            self._chat_id_saved = True
            logger.info("[Telegram] learned chat_id=%s", chat_id)
            cb = self._on_chat_id_learned
            if cb is not None:
                try:
                    cb(chat_id)
                except Exception:
                    logger.warning("[Telegram] chat_id_learned callback failed",
                                   exc_info=True)
        except Exception as exc:
            logger.warning("[Telegram] chat_id save errored: %s", exc)

    async def _check_access(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        user = update.effective_user
        if user is None or not self._is_allowed(user.id):
            logger.warning("[Telegram] Rejected user_id=%s", user.id if user else "unknown")
            await update.message.reply_text("У вас нет доступа к этому боту.")
            return False
        return True

    def _is_group(self, update: Update) -> bool:
        """Return True if the message is from a group/supergroup."""
        return update.effective_chat.type in ("group", "supergroup")

    def _is_mentioned(self, update: Update) -> bool:
        """Check if the bot is @mentioned in the message text."""
        text = update.message.text or update.message.caption or ""
        if self._bot_username and f"@{self._bot_username}" in text:
            return True
        entities = update.message.entities or update.message.caption_entities or []
        for ent in entities:
            if ent.type == "mention" and self._bot_username:
                mention = text[ent.offset:ent.offset + ent.length]
                if mention.lower() == f"@{self._bot_username.lower()}":
                    return True
        return False

    def _strip_mention(self, text: str) -> str:
        """Remove the @bot mention from message text."""
        if self._bot_username:
            text = text.replace(f"@{self._bot_username}", "").strip()
        return text

    async def _close_roleplay_for_menu(self, sid: str):
        """Atomically leave scene reality before exposing scenario navigation."""
        agent = self._sm.get_or_create(sid)
        async with self._sm.acquire(sid):
            roleplay = getattr(agent, "roleplay", None)
            if roleplay is not None:
                roleplay.end()
            self._transport.discard_pending_reason(sid, "roleplay")
            self._transport.set_keyboard_hidden(sid, False)
        return agent

    # ── Command handlers ──────────────────────────────────────────────────────

    async def _cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show the selector without ending the current scene or changing history."""
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        async with self._sm.acquire(sid):
            self._transport.set_keyboard_hidden(sid, False)
            if not self._transport.player_name(sid) or self._transport.awaiting_player_name(sid):
                self._transport.begin_player_name_onboarding(sid)
                await update.message.reply_text(_PLAYER_NAME_PROMPT, reply_markup=ReplyKeyboardRemove())
                return
            await update.message.reply_text(
                "Выбери сценарий. Если сцена уже идёт, она сохранится до выбора другой.\n"
                "Убрать клавиатуру: «Скрыть меню» или /hide.",
                reply_markup=_scenario_menu_keyboard(),
            )

    async def _cmd_hide(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        async with self._sm.acquire(sid):
            self._transport.set_keyboard_hidden(sid, True)
            await update.message.reply_text(
                "Меню скрыто. Открыть сценарии: /menu.",
                reply_markup=ReplyKeyboardRemove(),
            )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        agent = await self._close_roleplay_for_menu(sid)
        player_name = self._transport.player_name(sid)
        if not player_name:
            self._transport.begin_player_name_onboarding(sid)
            await update.message.reply_text(
                _PLAYER_NAME_PROMPT,
                reply_markup=ReplyKeyboardRemove(),
            )
            return
        _set_agent_player_name(agent, player_name)
        await update.message.reply_text(
            "👋 Это личный Telegram-бот Сары.\n\n"
            "Выберите один из 10 ролевых сценариев на клавиатуре ниже. "
            "Сцена начинается текстом; фотография возможна только как отдельный запрос.\n"
            "Во время игры кнопка «Выйти в меню сценариев» завершает текущую сцену.\n\n"
            "Команды:\n"
            "  /start          — показать эту справку\n"
            "  /menu           — открыть сценарии, сохранив текущую сцену\n"
            "  /hide           — скрыть меню\n"
            "  /clear          — очистить историю, сохранив память\n"
            "  /reset          — то же, что /clear\n"
            "  /status         — показать состояние сессии\n"
            "  /name           — изменить имя игрока\n"
            "  /compact [указание] — сжать историю диалога\n"
            "  /clear_files    — удалить загруженные файлы",
            reply_markup=_scenario_menu_keyboard(),
        )

    async def _cmd_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        self._transport.begin_player_name_onboarding(sid)
        await update.message.reply_text(
            _PLAYER_NAME_PROMPT,
            reply_markup=ReplyKeyboardRemove(),
        )

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Wipe the rolling chat transcript and start a fresh session.

        Long-term memory (MEMORY.md), persona, soul, skills, and learned
        facts are preserved — only this conversation's history is reset.
        Also aliased as ``/clear``.
        """
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        self._sm.reset(sid)
        await update.message.reply_text(
            "🧹 История текущего диалога очищена.\n"
            "Долгосрочная память, личность Сары и история ваших отношений сохранены."
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        agent = self._sm.get_or_create(sid)
        from ..core.compaction import estimate_tokens
        await update.message.reply_text(
            f"📊 Состояние сессии\n"
            f"  ID сессии: {sid}\n"
            f"  Провайдер: {type(agent.provider).__name__}\n"
            f"  Активных навыков: {len(agent.loaded_skill_names)}\n"
            f"  Записей памяти: {len(agent.memory.list_all())}\n"
            f"  Сообщений в истории: {len(agent.messages)}\n"
            f"  Оценка токенов: ~{estimate_tokens(agent.messages):,}\n"
            f"  Сжатий истории: {agent.compaction_count}\n"
            f"  Всего сессий: {len(self._sm)}"
        )

    async def _cmd_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return
        sid = self._session_id(update.effective_chat.id)
        agent = self._sm.get_or_create(sid)
        hint: str | None = " ".join(context.args).strip() or None if context.args else None
        await update.message.reply_text("⏳ Сжимаю историю диалога…")
        try:
            agent.compact(instruction=hint)
        except Exception:
            logger.exception("[Telegram] conversation compaction failed")
            await update.message.reply_text("Не удалось сжать историю диалога.")
            return
        await update.message.reply_text("✅ История диалога сжата. Важный контекст сохранён.")

    async def _cmd_clear_files(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return
        from .. import config as _cfg
        count = _cfg.clear_files()
        await update.message.reply_text(f"🧹 Удалено файлов: {count}.")

    # ── Message handler (text + photos) ───────────────────────────────────────

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_access(update, context):
            return

        if self._is_group(update) and self._require_mention:
            if not self._is_mentioned(update):
                return

        sid = self._session_id(update.effective_chat.id)
        update_id = getattr(update, "update_id", None)
        if update_id is not None and not self._transport.accept_update(update_id, sid):
            logger.info("[Telegram] Ignoring duplicate accepted update_id=%s", update_id)
            return

        # Capture the Telegram chat_id once so the
        # proactive scheduler knows where to message them. Best-effort —
        # any failure just means we'll retry on the next inbound message.
        await self._maybe_save_chat_id(update.effective_chat.id)

        user_text = (update.message.text or update.message.caption or "").strip()
        user_text = self._strip_mention(user_text)
        if user_text == _HIDE_MENU_BUTTON_TEXT:
            await self._cmd_hide(update, context)
            return
        if self._transport.awaiting_player_name(sid):
            player_name = _normalize_player_name(user_text)
            if player_name is None:
                await update.message.reply_text(
                    _PLAYER_NAME_INVALID,
                    reply_markup=ReplyKeyboardRemove(),
                )
                return
            self._transport.save_player_name(sid, player_name)
            agent = self._sm.get_or_create(sid)
            _set_agent_player_name(agent, player_name)
            roleplay = getattr(agent, "roleplay", None)
            is_active = getattr(roleplay, "is_active", None)
            active_scene = bool(callable(is_active) and is_active())
            await update.message.reply_text(
                f"Приятно познакомиться, {player_name}. "
                + ("Продолжим текущую сцену." if active_scene else "Выбери сценарий.")
                + (" Открыть меню: /menu." if self._transport.keyboard_hidden(sid) else ""),
                reply_markup=(
                    ReplyKeyboardRemove() if self._transport.keyboard_hidden(sid)
                    else _roleplay_exit_keyboard() if active_scene else _scenario_menu_keyboard()
                ),
            )
            return
        if user_text == ROLEPLAY_EXIT_BUTTON_TEXT:
            agent = await self._close_roleplay_for_menu(sid)
            _set_agent_player_name(agent, self._transport.player_name(sid))
            await update.message.reply_text(
                "Сцена завершена. Выберите следующий сценарий.",
                reply_markup=_scenario_menu_keyboard(),
            )
            return
        user_text = _expand_roleplay_request(user_text)

        if user_text:
            from ..core.telegram_transport import parse_silence_boundary

            boundary = parse_silence_boundary(user_text)
            if boundary is not None:
                action, expiry = boundary
                if action == "cancel":
                    self._transport.cancel_silence(sid)
                elif expiry is not None:
                    self._transport.set_silence(
                        sid,
                        expiry,
                        source_update_id=int(update_id) if update_id is not None else None,
                    )

        has_photo = bool(update.message.photo)
        has_voice = bool(update.message.voice or update.message.audio)

        if has_voice:
            transcript = await self._transcribe_voice(update)
            if transcript is None:
                return
            user_text = transcript

        if not user_text and not has_photo:
            return

        agent = self._sm.get_or_create(sid)
        _set_agent_player_name(agent, self._transport.player_name(sid))

        # Companion mode (a soul/persona is configured) delivers like a real
        # person texting: selective emoji reactions, absence awareness, a
        # human pause, and one logical reply per message. Bare-assistant
        # sessions keep the streaming/edit-in-place flow below.
        is_companion = bool(getattr(agent, "soul_instruction", "")
                            or getattr(agent, "persona_instruction", ""))

        if is_companion:
            # Absence awareness: how long since THEIR previous message.
            now_ts = time.time()
            prev_ts = self._last_seen.get(update.effective_chat.id)
            self._last_seen[update.effective_chat.id] = now_ts
            gap_note = ""
            if prev_ts:
                mins = (now_ts - prev_ts) / 60.0
                if mins >= 20:
                    gap_note = humanize_gap(mins)
            agent._gap_note = gap_note
            # How many of HER messages went unanswered before this reply — lets
            # her be a little sulky about being ignored, with a real basis.
            # Read before clearing, since this message ends the streak.
            agent._ignored_count = self._transport.clear_proactive_unanswered(sid)
            # Selective reaction — a photo almost always, big feelings
            # sometimes, logistics never. Fired before the reply, like a
            # human seeing the message first.
            asyncio.create_task(maybe_react(
                context.bot, update.effective_chat.id,
                update.message.message_id, has_photo, user_text))
        else:
            try:
                await update.message.set_reaction([ReactionTypeEmoji("\U0001f440")])
            except Exception:
                pass

        # Telegram vision is deliberately current-turn-only.  Older upstream
        # versions queued captionless photos in the Agent until a later text
        # message, which retained base64 in memory and could send a stale image
        # through D-029.  Clear any legacy queue but never include it here.
        stale_attachments = agent.consume_attachments()
        if stale_attachments:
            logger.warning(
                "[Telegram] discarded %d stale queued attachment(s) for %s",
                len(stale_attachments), sid,
            )
        new_parts: list[dict] = []
        if has_photo:
            try:
                built = await self._build_image_input(update, "")
            except Exception as exc:
                from ..core.outbound.policy import PolicyError

                if not isinstance(exc, PolicyError):
                    raise
                logger.warning("[Telegram] rejected inbound photo: %s", exc)
                await update.message.reply_text(_PHOTO_FAILURE_TEXT)
                return
            new_parts = [p for p in built if isinstance(p, dict) and p.get("type") == "image_url"]

        all_attachments = new_parts

        if all_attachments:
            photo_placeholder = ("（我发了一张照片给你，看看然后自然地回应～）"
                                 if is_companion else "What's in this image?")
            chat_input = [
                {"type": "text", "text": user_text or photo_placeholder},
                *all_attachments,
            ]
        else:
            chat_input = user_text or ""

        # Reply-in-kind: they spoke, so she'd rather speak back. Decided per
        # reply by core.tts (mode/length caps); any TTS miss falls back to
        # text below — the reply itself is never at risk.
        from ..core import tts as _tts
        reply_in_kind = has_voice and _tts.reply_in_kind_mode() != "false"

        typing_task = asyncio.create_task(
            self._keep_typing(update.message.chat_id)
        )
        try:
            async with self._sm.acquire(sid):
                loop = asyncio.get_event_loop()
                chat_id = update.effective_chat.id
                self._register_file_sender(loop, chat_id)
                if is_companion or reply_in_kind:
                    # Humanized delivery: full reply, then a length-scaled
                    # pause + one message. (No live edit-in-place — watching
                    # a message rewrite itself reads as a bot, not a partner.)
                    # A voice-note turn also lands here even in assistant
                    # mode: a voice reply needs the finished text, not a
                    # stream.
                    reply = await loop.run_in_executor(None, agent.chat, chat_input)
                    typing_task.cancel()
                    if reply:
                        if reply.startswith("Error: "):
                            await update.message.reply_text(
                                _PHOTO_FAILURE_TEXT if new_parts else _model_failure_text(reply)
                            )
                            return
                        if new_parts:
                            await self._remember_their_photo(agent, reply)
                        text = _clean_response(reply)
                        if is_companion:
                            try:
                                await asyncio.sleep(reply_delay(reply))
                            except Exception:
                                pass
                        sent_voice = False
                        if reply_in_kind:
                            sent_voice = await self._maybe_send_voice_reply(chat_id, text)
                        if not sent_voice:
                            if is_companion:
                                roleplay = getattr(agent, "roleplay", None)
                                await _send_companion_text(
                                    context.bot,
                                    chat_id,
                                    text,
                                    roleplay=bool(roleplay and roleplay.is_active()),
                                    keyboard_hidden=self._transport.keyboard_hidden(sid),
                                )
                            else:
                                for chunk in _split_message(text):
                                    await update.message.reply_text(chunk)
                else:
                    token_queue: _queue.Queue[str] = _queue.Queue()
                    future = loop.run_in_executor(
                        None, agent.chat_stream, chat_input, token_queue.put,
                    )
                    await self._flush_stream(update, token_queue, future)
        except Exception as exc:
            logger.exception("[Telegram] Agent error")
            await update.message.reply_text(f"Sorry, something went wrong: {exc}")
        finally:
            typing_task.cancel()

        # Assistant mode clears its 👀 "reading" reaction when done; a
        # companion's reaction (❤️ on your photo) is a real response — keep it.
        if not is_companion:
            try:
                await update.message.set_reaction([])
            except Exception:
                pass

    _AGENT_TIMEOUT = 600

    async def _maybe_send_voice_reply(self, chat_id: int, text: str) -> bool:
        """Try to answer a voice note with a voice note (reply-in-kind).

        Returns True only when a voice bubble was actually delivered. Any
        miss — the decision says text, no TTS provider is usable, synthesis
        or the send fails, or the local engine could only produce WAV (no
        ffmpeg) — returns False so the caller falls back to text.
        """
        from ..core import tts

        try:
            if not tts.should_voice_reply(text):
                return False
            speech = tts.strip_for_speech(text)
            if not speech:
                return False
            try:
                await self._app.bot.send_chat_action(
                    chat_id=chat_id, action="record_voice")
            except Exception:
                pass
            clip = await tts.synthesize_async(speech)
            if clip is None:
                logger.debug("[Telegram] No usable TTS provider — replying with text")
                return False
            if not clip.telegram_ready:
                logger.warning(
                    "[Telegram] TTS produced %s but Telegram voice notes need "
                    "OGG/Opus or MP3 (install ffmpeg); replying with text",
                    clip.format,
                )
                return False
            delivered = await self.send_voice(
                chat_id, clip.data, format=clip.format, origin="explicit",
            )
            if not delivered:
                return False
            logger.info("[Telegram] Voice reply sent (%d chars → %s, %.1f KB)",
                        len(speech), clip.format, len(clip.data) / 1024)
            return True
        except Exception as exc:
            logger.warning("[Telegram] Voice reply failed (%s); replying with text", exc)
            return False

    async def _flush_stream(
        self,
        update: Update,
        token_queue: "_queue.Queue[str]",
        future: "asyncio.Future[str]",
    ) -> None:
        """Collect streamed tokens and deliver as 2-3 large messages.

        Strategy: accumulate all tokens silently. Tool-call markers are
        stripped but do NOT trigger new messages.  Content is edit-in-place
        updated into a single live message; only when a message hits the
        Telegram 4096 char limit is a new message started.

        No heartbeat / "still working" messages are sent.
        """
        buf: list[str] = []
        live_msg = None
        live_text = ""
        sent_any = False
        THROTTLE = 2.0
        last_edit = time.monotonic()
        start_time = time.monotonic()
        _MARKER = re.compile(r'`\[calling:\s*([^\]]+)\]`')

        while not future.done():
            if (time.monotonic() - start_time) > self._AGENT_TIMEOUT:
                logger.warning(
                    "[Telegram] Agent timeout after %ds", self._AGENT_TIMEOUT,
                )
                break

            drained = False
            while True:
                try:
                    buf.append(token_queue.get_nowait())
                    drained = True
                except _queue.Empty:
                    break

            if not drained:
                await asyncio.sleep(0.4)
                continue

            raw = _MARKER.sub("", "".join(buf))
            text = _clean_response(raw)
            now = time.monotonic()

            # Only show up to the last paragraph break while streaming;
            # the trailing incomplete line is held back to avoid flashing
            # progress narration that will be stripped later.
            last_break = text.rfind("\n\n")
            display = text[:last_break].rstrip() if last_break > 0 else ""

            if display and display != live_text and (now - last_edit) >= THROTTLE:
                try:
                    if live_msg is None:
                        live_msg = await update.message.reply_text(
                            display[:4096],
                        )
                        live_text = display[:4096]
                    elif len(display) <= 4096:
                        await live_msg.edit_text(display)
                        live_text = display
                    else:
                        await live_msg.edit_text(display[:4096])
                        live_msg = None
                        live_text = ""
                        buf = [display[4096:] + text[last_break:]]
                    sent_any = True
                except Exception:
                    pass
                last_edit = now

            await asyncio.sleep(0.4)

        # ── Final drain ───────────────────────────────────────────────
        response = future.result() if future.done() else "(timed out)"
        while True:
            try:
                buf.append(token_queue.get_nowait())
            except _queue.Empty:
                break

        raw = _MARKER.sub("", "".join(buf))
        remaining = _clean_response(raw.strip())
        if remaining and remaining != live_text:
            try:
                if live_msg and len(remaining) <= 4096:
                    await live_msg.edit_text(remaining)
                elif live_msg:
                    await live_msg.edit_text(remaining[:4096])
                    for chunk in _split_message(remaining[4096:]):
                        await update.message.reply_text(chunk)
                else:
                    for chunk in _split_message(remaining):
                        await update.message.reply_text(chunk)
                sent_any = True
            except Exception:
                pass

        if not sent_any:
            raw_response = response or ""
            text = (
                _model_failure_text(raw_response)
                if raw_response.startswith("Error: ") or not raw_response
                else _clean_response(raw_response)
            )
            for chunk in _split_message(text):
                await update.message.reply_text(chunk)

    def _register_file_sender(self, loop: asyncio.AbstractEventLoop, chat_id: int) -> None:
        """Register sync callbacks so the Agent can send files / photos /
        voice notes via Telegram."""
        from ..core.tools import set_file_sender, set_photo_sender, set_voice_sender

        bot_app = self._app

        def _file_sender(path: str, caption: str = "") -> None:
            async def _do_send():
                try:
                    with open(path, "rb") as f:
                        await bot_app.bot.send_document(
                            chat_id=chat_id,
                            document=f,
                            caption=caption[:1024] if caption else None,
                        )
                except Exception as exc:
                    logger.warning("[Telegram] send_file failed: %s", exc)

            future = asyncio.run_coroutine_threadsafe(_do_send(), loop)
            future.result(timeout=60)

        def _photo_sender(path: str, caption: str = "") -> None:
            future = asyncio.run_coroutine_threadsafe(
                self.send_photo(chat_id, path, caption), loop,
            )
            future.result(timeout=60)

        def _voice_sender(path: str, caption: str = "") -> None:
            future = asyncio.run_coroutine_threadsafe(
                self.send_voice(chat_id, path, caption, origin="explicit"), loop,
            )
            future.result(timeout=60)

        set_file_sender(self._session_id(chat_id), _file_sender)
        set_photo_sender(self._session_id(chat_id), _photo_sender)
        set_voice_sender(self._session_id(chat_id), _voice_sender)

    async def _build_image_input(self, update: Update, caption: str) -> list:
        """Download photo and build a multimodal content array."""
        from ..core.outbound.images import normalize_image_bytes

        photo = update.message.photo[-1]  # highest resolution
        file = await photo.get_file()
        data = await file.download_as_bytearray()
        normalized, mime_type, _dimensions = normalize_image_bytes(bytes(data))
        b64 = base64.b64encode(normalized).decode()

        return [
            {"type": "text", "text": caption},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64}",
                },
            },
        ]

    async def _transcribe_voice(self, update: Update) -> str | None:
        """Download a voice/audio message and transcribe it (Deepgram or
        local faster-whisper, per ``stt.provider``).

        Returns the transcript text, or sends a hint to the user and
        returns ``None`` if no STT provider is configured.
        """
        from ..core.stt import no_key_message, transcribe_bytes_async

        voice = update.message.voice or update.message.audio
        duration = getattr(voice, "duration", None)
        file_size = getattr(voice, "file_size", None)
        if not isinstance(duration, (int, float)) or duration <= 0 \
                or duration > _VOICE_MAX_DURATION_SECONDS:
            await update.message.reply_text(_VOICE_FAILURE_TEXT)
            return None
        if not isinstance(file_size, int) or file_size <= 0 \
                or file_size > _VOICE_MAX_INPUT_BYTES:
            await update.message.reply_text(_VOICE_FAILURE_TEXT)
            return None
        tg_file = await voice.get_file()
        audio_bytes = bytes(await tg_file.download_as_bytearray())
        if len(audio_bytes) > _VOICE_MAX_INPUT_BYTES:
            await update.message.reply_text(_VOICE_FAILURE_TEXT)
            return None
        mime = voice.mime_type or "audio/ogg"

        try:
            transcript = await transcribe_bytes_async(
                audio_bytes, mime, duration_seconds=float(duration),
            )
        except Exception as exc:
            logger.warning("[Telegram] Voice transcription failed (%s)", type(exc).__name__)
            await update.message.reply_text(_VOICE_FAILURE_TEXT)
            return None

        if transcript is None:
            await update.message.reply_text(no_key_message())
            return None

        if not transcript.strip():
            await update.message.reply_text(_VOICE_EMPTY_TEXT)
            return None

        logger.info("[Telegram] Voice transcribed (%d chars)", len(transcript))
        return transcript

    async def _keep_typing(self, chat_id: int) -> None:
        """Re-send the 'typing' chat action every 4 s until cancelled."""
        try:
            while True:
                await self._app.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("[Telegram] _keep_typing stopped unexpectedly", exc_info=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    _BOT_COMMANDS = [
        BotCommand("start",       "Показать справку"),
        BotCommand("menu",        "🎭 Открыть сценарии"),
        BotCommand("hide",        "Скрыть меню"),
        BotCommand("clear",       "🧹 Очистить историю, сохранив память"),
        BotCommand("reset",       "То же, что /clear"),
        BotCommand("status",      "Показать состояние сессии"),
        BotCommand("name",        "Изменить имя игрока"),
        BotCommand("compact",     "Сжать историю диалога"),
        BotCommand("clear_files", "Удалить загруженные файлы"),
    ]


    def build_application(self) -> Application:
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("menu", self._cmd_menu))
        app.add_handler(CommandHandler("hide", self._cmd_hide))
        app.add_handler(CommandHandler("reset", self._cmd_reset))
        app.add_handler(CommandHandler("clear", self._cmd_reset))  # alias
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("name", self._cmd_name))
        app.add_handler(CommandHandler("compact", self._cmd_compact))
        app.add_handler(CommandHandler("clear_files", self._cmd_clear_files))
        app.add_handler(MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO)
            & ~filters.COMMAND,
            self._handle_message,
        ))
        self._app = app
        return app

    async def _register_commands(self) -> None:
        """Register slash-commands with Telegram so they appear in the menu."""
        try:
            await self._app.bot.set_my_commands(self._BOT_COMMANDS)
            me = await self._app.bot.get_me()
            self._bot_username = me.username
            logger.info(
                "[Telegram] Registered %d bot commands, username=@%s",
                len(self._BOT_COMMANDS), self._bot_username,
            )
        except Exception:
            logger.warning("[Telegram] Failed to register bot commands", exc_info=True)

    def run_polling(self) -> None:
        """Blocking call — starts the bot using long polling (for standalone use)."""
        app = self.build_application()
        logger.info("[Telegram] Starting bot (polling mode)...")
        app.post_init = lambda _app: self._register_commands()
        app.run_polling(drop_pending_updates=False)

    async def start_async(self) -> None:
        """Non-blocking start — for use inside an existing asyncio event loop."""
        app = self.build_application()
        logger.info("[Telegram] Initialising bot (async mode)...")
        await app.initialize()
        await app.start()
        await self._register_commands()
        await app.updater.start_polling(drop_pending_updates=False)

    async def stop_async(self) -> None:
        if self._app is None:
            return
        logger.info("[Telegram] Stopping bot...")
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()


# ── Utility ───────────────────────────────────────────────────────────────────

_LEAKED_TOOL_RE = re.compile(
    r'<\s*\|?\s*(?:DSML|antml)\s*\|\s*function_calls[^>]*>'
    r'[\s\S]*?'
    r'<\s*/\s*\|?\s*(?:DSML|antml)\s*\|\s*function_calls\s*>',
    re.IGNORECASE,
)


_PROGRESS_LINE_RE = re.compile(r'\n\n.{0,60}[：:]\s*\n\n')


def _clean_response(text: str) -> str:
    """Strip leaked tool-call XML/DSML markup and excess whitespace."""
    text = _LEAKED_TOOL_RE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    for _ in range(10):
        cleaned = _PROGRESS_LINE_RE.sub('\n\n', text)
        if cleaned == text:
            break
        text = cleaned
    return text.strip()


def _split_message(text: str, limit: int = 4096) -> list[str]:
    """Split text into chunks respecting natural boundaries.

    Tries paragraph breaks first, then newlines, then word boundaries,
    and only falls back to a hard character cut as a last resort.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    min_break = limit // 3
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind('\n\n', min_break, limit)
        if split_at < min_break:
            split_at = text.rfind('\n', min_break, limit)
        if split_at < min_break:
            split_at = text.rfind(' ', min_break, limit)
        if split_at < min_break:
            split_at = limit
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    return chunks


def create_bot(session_manager: "SessionManager") -> TelegramBot:
    """Create a TelegramBot from herandhim.json / env vars."""
    token = config.get_str(
        "channels", "telegram", "token", env="TELEGRAM_BOT_TOKEN",
    )
    if not token:
        raise ValueError("Telegram token not set (env TELEGRAM_BOT_TOKEN or channels.telegram.token)")
    allowed_users = config.get_int_list(
        "channels", "telegram", "allowedUsers", env="TELEGRAM_ALLOWED_USERS",
    )
    if not allowed_users:
        raise ValueError(
            "Telegram allowedUsers must contain the owner id; empty allowlist is forbidden"
        )
    require_mention = config.get_bool(
        "channels", "telegram", "requireMention", default=False,
    )
    return TelegramBot(
        session_manager=session_manager,
        token=token,
        allowed_users=allowed_users,
        require_mention=require_mention,
    )


# Backward-compatible alias
create_bot_from_env = create_bot
