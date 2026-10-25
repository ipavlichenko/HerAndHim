"""Incoming Telegram photo foundation: gates, exact vision route and persistence."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
from openai import OpenAI
from PIL import Image

from herandhim.channels.telegram_bot import TelegramBot
from herandhim.core.llm.openai_compatible import OpenAICompatibleProvider
from herandhim.core.outbound.costs import CostLedger
from herandhim.core.outbound.images import normalize_image_bytes
from herandhim.core.outbound.policy import OutboundPolicy, PolicyError
from herandhim.core.outbound.trace import OutboundTrace
from herandhim.core.persistent_agent import PersistentAgent
from herandhim.core.session_store import SessionStore
from herandhim.core.telegram_transport import TelegramTransportStore
from herandhim.session_manager import SessionManager


def _vision_policy() -> OutboundPolicy:
    rows = {
        purpose: {"enabled": False}
        for purpose in (
            "dialogue",
            "image_generation",
            "memory_consolidation",
            "relationship_analysis",
        )
    }
    rows["vision"] = {
        "model": "openai/gpt-6.1-sol",
        "allowedProviders": ["azure/eu"],
        "activeProvider": "azure/eu",
        "maxInputTokensEstimated": 6_000,
        "maxOutputTokens": 400,
        "maxCallsPerTurn": 1,
        "connectTimeoutSeconds": 5,
        "firstByteTimeoutSeconds": 30,
        "totalTimeoutSeconds": 90,
        "maxCostUsdPerCall": 0.005,
        "maxCostUsdPerDay": 0.05,
    }
    return OutboundPolicy.from_mapping({
        "version": 1,
        "traceRetentionDays": 30,
        "aggregateRetentionDays": 180,
        "openrouter": {
            "requireAccountPromptLoggingDisabled": True,
            "requireAccountZdr": True,
            "requireNonZeroKeyCreditLimit": True,
        },
        "purposes": rows,
    })


def _jpeg_bytes(*, metadata: bool = False, size: tuple[int, int] = (32, 24)) -> bytes:
    out = io.BytesIO()
    image = Image.new("RGB", size, "purple")
    kwargs = {}
    if metadata:
        exif = Image.Exif()
        exif[0x010E] = "private description"
        kwargs["exif"] = exif
        kwargs["icc_profile"] = b"private color profile"
    image.save(out, format="JPEG", **kwargs)
    return out.getvalue()


class _PhotoFile:
    def __init__(self, payload: bytes, counters: dict[str, int]) -> None:
        self.payload = payload
        self.counters = counters

    async def download_as_bytearray(self) -> bytearray:
        self.counters["download"] += 1
        return bytearray(self.payload)


class _Photo:
    def __init__(self, payload: bytes, counters: dict[str, int]) -> None:
        self.payload = payload
        self.counters = counters

    async def get_file(self) -> _PhotoFile:
        self.counters["get_file"] += 1
        return _PhotoFile(self.payload, self.counters)


class _Message:
    def __init__(self, payload: bytes, counters: dict[str, int], caption: str = "") -> None:
        self.text = None
        self.caption = caption
        self.photo = [_Photo(payload, counters)]
        self.voice = None
        self.audio = None
        self.entities = []
        self.caption_entities = []
        self.message_id = 9
        self.chat_id = 42
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)

    async def set_reaction(self, *_args, **_kwargs) -> None:
        return None


def _update(
    payload: bytes,
    counters: dict[str, int],
    *,
    user_id: int = 42,
    update_id: int = 9001,
    caption: str = "Что ты здесь видишь?",
):
    return SimpleNamespace(
        update_id=update_id,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=42, type="private"),
        message=_Message(payload, counters, caption),
    )


async def _no_op(*_args, **_kwargs) -> None:
    return None


class _NoInferenceSessionManager:
    def __init__(self) -> None:
        self.created = 0

    def get_or_create(self, _session_id: str):
        self.created += 1
        raise AssertionError("access/dedupe gates must run before agent creation")


def test_allowlist_and_restart_dedupe_run_before_photo_download_or_inference(tmp_path):
    path = tmp_path / "transport.sqlite3"
    counters = {"get_file": 0, "download": 0}
    context = SimpleNamespace(bot=SimpleNamespace())

    outsider_sm = _NoInferenceSessionManager()
    outsider_bot = TelegramBot(
        outsider_sm,
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(path),
    )
    asyncio.run(outsider_bot._handle_message(
        _update(_jpeg_bytes(), counters, user_id=7, update_id=9000), context,
    ))
    assert counters == {"get_file": 0, "download": 0}
    assert outsider_sm.created == 0

    assert TelegramTransportStore(path).accept_update(9001, "telegram:42")
    duplicate_sm = _NoInferenceSessionManager()
    restarted = TelegramBot(
        duplicate_sm,
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(path),
    )
    asyncio.run(restarted._handle_message(
        _update(_jpeg_bytes(), counters, update_id=9001), context,
    ))
    assert counters == {"get_file": 0, "download": 0}
    assert duplicate_sm.created == 0


def test_photo_only_turn_is_current_image_only_and_discards_stale_queue(tmp_path, monkeypatch):
    class Agent:
        soul_instruction = ""
        persona_instruction = ""

        def __init__(self) -> None:
            self.inputs = []

        def consume_attachments(self):
            return [{
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,STALE"},
            }]

        def chat_stream(self, value, _on_token):
            self.inputs.append(value)
            return "current image understood"

    class SM:
        def __init__(self) -> None:
            self.agent = Agent()

        def get_or_create(self, _sid):
            return self.agent

        @asynccontextmanager
        async def acquire(self, _sid):
            yield

    sm = SM()
    bot = TelegramBot(
        sm,
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(tmp_path / "transport.sqlite3"),
    )
    bot._maybe_save_chat_id = _no_op
    bot._keep_typing = _no_op
    bot._register_file_sender = lambda *_args: None
    monkeypatch.setattr(bot, "_flush_stream", _no_op)
    counters = {"get_file": 0, "download": 0}
    update = _update(_jpeg_bytes(), counters, caption="")
    asyncio.run(bot._handle_message(update, SimpleNamespace(bot=SimpleNamespace())))

    sent = sm.agent.inputs[0]
    urls = [part["image_url"]["url"] for part in sent if part.get("type") == "image_url"]
    assert len(urls) == 1
    assert "STALE" not in urls[0]
    assert any(part.get("type") == "text" and part.get("text") == "What's in this image?"
               for part in sent)


def test_exact_vision_route_persists_reply_without_image_after_restart(tmp_path, monkeypatch):
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={
            "id": "vision-response",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-6.1-sol",
            "provider": "azure/eu",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Вижу фиолетовый кадр."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    provider = OpenAICompatibleProvider(
        api_key="sk-or-synthetic-photo-key",
        base_url="https://mock.openrouter.test/v1",
        model_name="openai/gpt-6.1-sol",
        supports_images=True,
        outbound_policy=_vision_policy(),
        outbound_trace=OutboundTrace(tmp_path / "trace.jsonl"),
        cost_ledger=CostLedger(tmp_path / "costs.sqlite3"),
    )
    provider.client = OpenAI(
        api_key="synthetic",
        base_url="https://mock.openrouter.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    sessions = SessionStore(str(tmp_path / "sessions"))
    soul = tmp_path / "soul.md"
    soul.write_text("You are a concise companion.", encoding="utf-8")
    memory_dir = tmp_path / "memory"
    skills_dir = tmp_path / "skills"
    memory_dir.mkdir()
    skills_dir.mkdir()

    def factory(session_id: str) -> PersistentAgent:
        agent = PersistentAgent(
            provider=provider,
            memory_dir=str(memory_dir),
            skills_dirs=[str(skills_dir)],
            knowledge_path=None,
            persona_path=None,
            soul_path=str(soul),
            profile_path=None,
            verbose=False,
            auto_compaction=False,
            store=sessions,
            session_id=session_id,
        )
        agent._update_affect = lambda *_args: None
        agent.memory.remember = MagicMock()
        return agent

    sm = SessionManager(factory)
    bot = TelegramBot(
        sm,
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(tmp_path / "transport.sqlite3"),
    )
    bot._maybe_save_chat_id = _no_op
    bot._keep_typing = _no_op
    bot._register_file_sender = lambda *_args: None
    monkeypatch.setattr("herandhim.channels.telegram_bot.maybe_react", _no_op)
    monkeypatch.setattr("herandhim.channels.telegram_bot.reply_delay", lambda _reply: 0)
    delivered: list[str] = []

    async def capture_burst(_bot, _chat_id, text, **_kwargs):
        delivered.append(text)

    monkeypatch.setattr("herandhim.channels.telegram_bot.send_burst", capture_burst)
    counters = {"get_file": 0, "download": 0}
    update = _update(_jpeg_bytes(metadata=True, size=(4096, 64)), counters)
    asyncio.run(bot._handle_message(update, SimpleNamespace(bot=SimpleNamespace())))

    assert delivered == ["Вижу фиолетовый кадр."]
    assert len(captured) == 1
    request = captured[0]
    assert request["model"] == "openai/gpt-6.1-sol"
    assert request["provider"] == {
        "only": ["azure/eu"],
        "order": ["azure/eu"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
        "require_parameters": True,
    }
    image_parts = [
        part
        for message in request["messages"]
        for part in message.get("content", [])
        if isinstance(message.get("content"), list) and part.get("type") == "image_url"
    ]
    assert len(image_parts) == 1
    normalized = base64.b64decode(image_parts[0]["image_url"]["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(normalized)) as image:
        assert max(image.size) <= 2048
        assert not image.getexif()
        assert image.info.get("icc_profile") is None
    assert len(normalized) <= 4 * 1024 * 1024

    live_agent = sm.get("telegram:42")
    live_agent.memory.remember.assert_called_once()
    assert "Вижу фиолетовый кадр" in live_agent.memory.remember.call_args.args[0]
    persisted = (tmp_path / "sessions" / "telegram_42.md").read_text(encoding="utf-8")
    assert "data:image" not in persisted
    assert "[image]" in persisted
    assert "Вижу фиолетовый кадр." in persisted

    restarted = factory("telegram:42")
    restored = repr(restarted.messages)
    assert "data:image" not in restored
    assert "[image]" in restored
    assert "Вижу фиолетовый кадр." in restored


def test_invalid_photo_fails_closed_without_inference(tmp_path):
    class Agent:
        soul_instruction = "companion"
        persona_instruction = ""

        def consume_attachments(self):
            return []

        def chat(self, _value):
            raise AssertionError("invalid image must fail before inference")

    class SM:
        def __init__(self):
            self.agent = Agent()

        def get_or_create(self, _sid):
            return self.agent

        @asynccontextmanager
        async def acquire(self, _sid):
            yield

    bot = TelegramBot(
        SM(),
        "synthetic-not-a-real-token",
        allowed_users=[42],
        transport_store=TelegramTransportStore(tmp_path / "transport.sqlite3"),
    )
    bot._maybe_save_chat_id = _no_op
    counters = {"get_file": 0, "download": 0}
    update = _update(b"not-an-image", counters)
    asyncio.run(bot._handle_message(update, SimpleNamespace(bot=SimpleNamespace())))
    assert update.message.replies == [
        "I couldn't safely process that image. Please send a valid photo under 4 MiB."
    ]


def test_normalized_oversize_photo_is_rejected():
    image = Image.frombytes("RGBA", (1100, 1100), os.urandom(1100 * 1100 * 4))
    source = io.BytesIO()
    image.save(source, format="PNG")
    try:
        normalize_image_bytes(source.getvalue())
    except PolicyError as exc:
        assert str(exc) == "normalized_image_too_large"
    else:
        raise AssertionError("oversize normalized image must fail closed")


def test_mock_transport_vision_failure_is_sanitized_and_not_retried(tmp_path):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="echoed owner photo and sk-or-provider-secret")

    trace_path = tmp_path / "trace.jsonl"
    provider = OpenAICompatibleProvider(
        api_key="sk-or-synthetic-photo-key",
        base_url="https://mock.openrouter.test/v1",
        model_name="openai/gpt-6.1-sol",
        supports_images=True,
        outbound_policy=_vision_policy(),
        outbound_trace=OutboundTrace(trace_path),
        cost_ledger=CostLedger(tmp_path / "costs.sqlite3"),
    )
    provider.client = OpenAI(
        api_key="synthetic",
        base_url="https://mock.openrouter.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    payload = base64.b64encode(_jpeg_bytes()).decode("ascii")
    try:
        provider.chat(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe the current image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{payload}",
                }},
            ]}],
            purpose="vision",
        )
    except PolicyError as exc:
        assert str(exc) == "provider_request_failed"
    else:
        raise AssertionError("mock provider failure must surface as a controlled error")
    assert calls == 1
    trace = trace_path.read_text(encoding="utf-8")
    assert "echoed owner photo" not in trace
    assert "provider-secret" not in trace
