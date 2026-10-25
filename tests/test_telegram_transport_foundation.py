from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def test_empty_telegram_allowlist_is_fail_closed(tmp_path):
    from herandhim.channels.telegram_bot import TelegramBot
    from herandhim.core.telegram_transport import TelegramTransportStore

    bot = TelegramBot(
        session_manager=SimpleNamespace(),
        token="synthetic-not-a-real-token",
        allowed_users=None,
        transport_store=TelegramTransportStore(tmp_path / "transport.sqlite3"),
    )
    assert not bot._is_allowed(42)


def test_update_receipt_survives_restart(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    path = tmp_path / "transport.sqlite3"
    first = TelegramTransportStore(path)
    assert first.accept_update(1001, "telegram:42")
    restarted = TelegramTransportStore(path)
    assert not restarted.accept_update(1001, "telegram:42")
    assert restarted.accept_update(1002, "telegram:42")


def test_player_name_onboarding_survives_restart(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    path = tmp_path / "transport.sqlite3"
    first = TelegramTransportStore(path)
    first.begin_player_name_onboarding("telegram:42")
    assert first.awaiting_player_name("telegram:42") is True

    restarted = TelegramTransportStore(path)
    assert restarted.awaiting_player_name("telegram:42") is True
    restarted.save_player_name("telegram:42", "Саша")

    again = TelegramTransportStore(path)
    assert again.player_name("telegram:42") == "Саша"
    assert again.awaiting_player_name("telegram:42") is False


def test_proactive_unanswered_boundary_and_daily_count_survive_restart(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    path = tmp_path / "transport.sqlite3"
    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    first = TelegramTransportStore(path)
    first.record_proactive_delivered("telegram:42", reason="contextual", now=now)
    first.record_proactive_delivered("telegram:42", reason="wishlist", now=now)

    restarted = TelegramTransportStore(path)
    state = restarted.proactive_status("telegram:42", now=now)
    assert state == {
        "unanswered": 2,
        "delivered_today": 2,
        "personal_date_sent_today": False,
    }
    restarted.record_proactive_delivered(
        "telegram:42", reason="personal_date", now=now,
    )
    assert restarted.proactive_status("telegram:42", now=now) == {
        "unanswered": 3,
        "delivered_today": 3,
        "personal_date_sent_today": True,
    }
    assert restarted.clear_proactive_unanswered("telegram:42") == 3
    assert restarted.proactive_status("telegram:42", now=now)["unanswered"] == 0


def test_proactive_daily_counter_resets_without_clearing_unanswered(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    day_one = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    day_two = day_one + timedelta(days=1)
    store.record_proactive_delivered("telegram:42", reason="contextual", now=day_one)
    assert store.proactive_status("telegram:42", now=day_two) == {
        "unanswered": 1,
        "delivered_today": 0,
        "personal_date_sent_today": False,
    }


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.caption = None
        self.photo = []
        self.voice = None
        self.audio = None
        self.entities = []
        self.caption_entities = []
        self.message_id = 7
        self.chat_id = 42
        self.replies: list[str] = []

    async def reply_text(self, text: str):
        self.replies.append(text)

    async def set_reaction(self, *_args, **_kwargs):
        return None


class _FakeAgent:
    soul_instruction = "companion"
    persona_instruction = ""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def consume_attachments(self):
        return []

    def chat(self, text):
        self.inputs.append(text)
        return "reply"


class _FakeSessionManager:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.agent = _FakeAgent()

    def get_or_create(self, session_id: str):
        self.created.append(session_id)
        return self.agent

    @asynccontextmanager
    async def acquire(self, _session_id: str):
        yield


def _update(user_id: int, update_id: int, text: str = "hello"):
    return SimpleNamespace(
        update_id=update_id,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=42, type="private"),
        message=_FakeMessage(text),
    )


def test_only_owner_creates_one_persistent_dialogue_turn_after_restart(tmp_path, monkeypatch):
    from herandhim.channels import telegram_bot as module
    from herandhim.channels.telegram_bot import TelegramBot
    from herandhim.core.telegram_transport import TelegramTransportStore

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr(module, "maybe_react", no_op)
    monkeypatch.setattr(module, "_send_companion_text", no_op)
    monkeypatch.setattr(module, "reply_delay", lambda _reply: 0)

    path = tmp_path / "transport.sqlite3"
    first_sm = _FakeSessionManager()
    first = TelegramBot(
        first_sm,
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(path),
    )
    first._maybe_save_chat_id = no_op
    first._keep_typing = no_op
    first._register_file_sender = lambda *_args: None
    context = SimpleNamespace(bot=SimpleNamespace())

    asyncio.run(first._handle_message(_update(7, 2000), context))
    assert first_sm.created == []
    asyncio.run(first._handle_message(_update(42, 2001), context))
    assert first_sm.created == ["telegram:42"]
    assert first_sm.agent.inputs == ["hello"]

    restarted_sm = _FakeSessionManager()
    restarted = TelegramBot(
        restarted_sm,
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(path),
    )
    restarted._maybe_save_chat_id = no_op
    restarted._keep_typing = no_op
    restarted._register_file_sender = lambda *_args: None
    asyncio.run(restarted._handle_message(_update(42, 2001), context))
    assert restarted_sm.created == []
    assert restarted_sm.agent.inputs == []


def test_explicit_silence_boundary_expires_or_can_be_cancelled(tmp_path):
    from herandhim.core.telegram_transport import (
        TelegramTransportStore,
        parse_silence_boundary,
    )

    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    action, expiry = parse_silence_boundary(
        "не пиши мне до 2026-10-24 09:00", now=now,
    )
    assert action == "set"
    assert expiry is not None

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    store.set_silence("telegram:42", expiry, now=now)
    assert store.silence_active("telegram:42", now=now + timedelta(hours=1))
    assert not store.silence_active("telegram:42", now=expiry + timedelta(seconds=1))

    store.set_silence("telegram:42", expiry, now=now)
    assert parse_silence_boundary("можешь снова писать", now=now) == ("cancel", None)
    assert store.cancel_silence("telegram:42")
    assert not store.silence_active("telegram:42", now=now)


class _DeliveryBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    def _session_id(self, chat_id: int) -> str:
        return f"telegram:{chat_id}"

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


def test_accepted_proactive_delivery_survives_restart_and_is_attempted_once(tmp_path, monkeypatch):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    path = tmp_path / "transport.sqlite3"
    before_restart = TelegramTransportStore(path)
    delivery = before_restart.enqueue("telegram:42", 42, "durable hello")

    bot = _DeliveryBot()
    restarted = TelegramTransportStore(path)
    messenger = ProactiveMessenger(SimpleNamespace(), bot, restarted)
    monkeypatch.setattr(messenger, "_maybe_send_selfie", lambda _chat_id: _async_none())
    assert asyncio.run(messenger._deliver_pending("telegram:42"))
    assert not asyncio.run(messenger._deliver_pending("telegram:42"))
    assert bot.sent == [(42, "durable hello")]
    assert restarted.delivery_status(delivery.id) == "delivered"


async def _async_none():
    return None


def test_kill_switch_and_silence_preserve_pending_state(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    path = tmp_path / "transport.sqlite3"
    store = TelegramTransportStore(path)
    delivery = store.enqueue("telegram:42", 42, "wait for me")
    store.set_delivery_enabled(False)
    assert store.claim_next("telegram:42") is None
    assert store.pending_count("telegram:42") == 1

    restarted = TelegramTransportStore(path)
    assert not restarted.delivery_enabled()
    assert restarted.delivery_status(delivery.id) == "pending"
    restarted.set_delivery_enabled(True)
    now = datetime.now(timezone.utc)
    restarted.set_silence("telegram:42", now + timedelta(minutes=10), now=now)
    assert restarted.claim_next("telegram:42", now=now) is None
    assert restarted.pending_count("telegram:42") == 1
    restarted.cancel_silence("telegram:42")
    assert restarted.claim_next("telegram:42", now=now) is not None


def test_contextual_evaluator_can_return_no_op_without_delivery(tmp_path, monkeypatch):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    messenger = ProactiveMessenger(SimpleNamespace(), _DeliveryBot(), store)
    monkeypatch.setattr("herandhim.scheduler.proactive._todays_personal_dates", lambda: [])
    monkeypatch.setattr("herandhim.scheduler.proactive.random.random", lambda: 0.9)
    decision = messenger._evaluate({"hours_since_last": 0}, threshold=0.1)
    assert decision.action == "no-op"
    assert store.pending_count("telegram:42") == 0


def test_two_unanswered_pause_contextual_but_allow_one_personal_date(tmp_path, monkeypatch):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    messenger = ProactiveMessenger(
        SimpleNamespace(), _DeliveryBot(), TelegramTransportStore(tmp_path / "transport.sqlite3"),
    )
    monkeypatch.setattr(messenger, "_max_unanswered", lambda: 2)
    paused = messenger._evaluate(
        {"hours_since_last": 999}, 1.0, unanswered=2, personal_date=False,
    )
    exception = messenger._evaluate(
        {"hours_since_last": 999}, 0.0, unanswered=2, personal_date=True,
        personal_date_already_sent=False,
    )
    exhausted = messenger._evaluate(
        {"hours_since_last": 999}, 1.0, unanswered=3, personal_date=True,
        personal_date_already_sent=True,
    )
    assert paused.reason == "unanswered_boundary"
    assert exception.reason == "personal_date" and exception.should_generate
    assert exhausted.reason == "unanswered_boundary"


def test_proactive_target_must_be_an_allowlisted_owner(monkeypatch):
    from herandhim.scheduler import proactive

    monkeypatch.setattr(proactive.config, "get_int_list", lambda *_args, **_kwargs: [42])
    monkeypatch.setattr(proactive.config, "get_int", lambda *_args, **_kwargs: 7)
    assert proactive.get_proactive_chat_id() is None

    monkeypatch.setattr(proactive.config, "get_int", lambda *_args, **_kwargs: 42)
    assert proactive.get_proactive_chat_id() == 42


def test_active_silence_blocks_inference_and_pending_delivery(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    class NoInference:
        def get(self, _session_id):
            raise AssertionError("silence must block evaluator context and inference")

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    store.enqueue("telegram:42", 42, "not yet")
    now = datetime.now(timezone.utc)
    store.set_silence("telegram:42", now + timedelta(hours=1), now=now)
    bot = _DeliveryBot()
    messenger = ProactiveMessenger(NoInference(), bot, store)
    asyncio.run(messenger._tick(42))
    assert bot.sent == []
    assert store.pending_count("telegram:42") == 1


def test_active_roleplay_discards_ordinary_pending_but_delivers_scene_pending(
    tmp_path, monkeypatch,
):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    class ActiveAgent:
        roleplay = SimpleNamespace(is_active=lambda: True)

    class Sessions:
        agent = ActiveAgent()

        def get_or_create(self, _session_id):
            return self.agent

        def get(self, _session_id):
            return self.agent

        @asynccontextmanager
        async def acquire(self, _session_id):
            yield

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    store.enqueue("telegram:42", 42, "queued before scene")
    scene_delivery = store.enqueue(
        "telegram:42", 42, "*Sara remains in the room.*", reason="roleplay",
    )

    class SceneBot(_DeliveryBot):
        def __init__(self):
            super().__init__()
            self.scene_flags = []

        async def send_message(self, chat_id: int, text: str, *, roleplay=False) -> None:
            self.sent.append((chat_id, text))
            self.scene_flags.append(roleplay)

    bot = SceneBot()
    messenger = ProactiveMessenger(Sessions(), bot, store)
    monkeypatch.setattr(messenger, "_maybe_send_selfie", lambda _chat_id: _async_none())

    asyncio.run(messenger._tick(42))

    assert bot.sent == [(42, "*Sara remains in the room.*")]
    assert bot.scene_flags == [True]
    assert store.delivery_status(scene_delivery.id) == "delivered"
    assert store.pending_count("telegram:42") == 0


def test_active_roleplay_proactive_generation_continues_scene(tmp_path, monkeypatch):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    prompts = []

    class ActiveAgent:
        roleplay = SimpleNamespace(is_active=lambda: True)

        def chat_proactive(self, prompt):
            prompts.append(prompt)
            return "*Sara stays by the window.*\n\nWill you come closer?"

    class Sessions:
        agent = ActiveAgent()

        def get_or_create(self, _session_id):
            return self.agent

        @asynccontextmanager
        async def acquire(self, _session_id):
            yield

    class SceneBot(_DeliveryBot):
        def __init__(self):
            super().__init__()
            self.scene_flags = []

        async def send_message(self, chat_id: int, text: str, *, roleplay=False) -> None:
            self.sent.append((chat_id, text))
            self.scene_flags.append(roleplay)

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    bot = SceneBot()
    messenger = ProactiveMessenger(Sessions(), bot, store)
    monkeypatch.setattr(messenger, "_maybe_send_selfie", lambda _chat_id: _async_none())

    asyncio.run(messenger._generate_and_send(
        42, datetime(2026, 10, 25, 5, 0, tzinfo=timezone.utc),
        {"sentiment": "negative"}, reason="personal_date",
    ))

    assert len(prompts) == 1
    assert prompts[0].startswith("[[ROLEPLAY_PROACTIVE_CONTINUATION]]")
    assert "durable scene state" in prompts[0]
    assert bot.sent == [(42, "*Sara stays by the window.*\n\nWill you come closer?")]
    assert bot.scene_flags == [True]
    assert store.pending_count("telegram:42") == 0
    assert store.proactive_status("telegram:42")["delivered_today"] == 1


def test_scene_pending_is_discarded_after_owner_leaves_roleplay(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    class InactiveAgent:
        roleplay = SimpleNamespace(is_active=lambda: False)

    class Sessions:
        agent = InactiveAgent()

        def get_or_create(self, _session_id):
            return self.agent

        def get(self, _session_id):
            return self.agent

        @asynccontextmanager
        async def acquire(self, _session_id):
            yield

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    scene_delivery = store.enqueue(
        "telegram:42", 42, "stale scene move", reason="roleplay",
    )
    store.set_delivery_enabled(False)
    messenger = ProactiveMessenger(Sessions(), _DeliveryBot(), store)

    asyncio.run(messenger._tick(42))

    assert store.delivery_status(scene_delivery.id) is None
    assert store.pending_count("telegram:42") == 0


def test_ambiguous_failed_send_is_not_retried(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    class FailingBot(_DeliveryBot):
        async def send_message(self, chat_id: int, text: str) -> None:
            self.sent.append((chat_id, text))
            raise TimeoutError("synthetic ambiguous timeout")

    store = TelegramTransportStore(tmp_path / "transport.sqlite3")
    delivery = store.enqueue("telegram:42", 42, "only one attempt")
    bot = FailingBot()
    messenger = ProactiveMessenger(SimpleNamespace(), bot, store)
    assert not asyncio.run(messenger._deliver_pending("telegram:42"))
    assert not asyncio.run(messenger._deliver_pending("telegram:42"))
    assert bot.sent == [(42, "only one attempt")]
    assert store.delivery_status(delivery.id) == "attempted"
