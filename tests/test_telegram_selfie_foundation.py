from __future__ import annotations

import asyncio
import base64
import io
import json
from threading import Lock
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image


def _png_bytes(color: str = "red") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 24), color).save(output, format="PNG")
    return output.getvalue()


def _policy(reference: str, **image_overrides):
    from herandhim.core.outbound.policy import OutboundPolicy
    from tests.test_outbound_contract import policy_mapping

    image = {
        "model": "bytedance-seed/seedream-5-0-flash",
        "transport": "images_api",
        "resolution": "1K",
        "aspectRatio": "2:3",
        "maxImages": 1,
        "allowedReferenceAssets": [reference],
        "maxCostUsdPerCall": 0.03,
        "maxCostUsdPerDay": 0.12,
    }
    image.update(image_overrides)
    return OutboundPolicy.from_mapping(policy_mapping(image_generation=image))


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self.text = "synthetic provider text that must not escape"

    def json(self):
        return self._payload


def _generator(tmp_path, reference, trace=None, **policy_overrides):
    from herandhim.core.image_gen.generator import SeedreamGenerator
    from herandhim.core.outbound.costs import CostLedger

    return SeedreamGenerator(
        api_key="sk-or-synthetic-secret-value",
        provider="openrouter",
        base_url="https://mock.openrouter.test/api/v1",
        model="bytedance-seed/seedream-5-0-flash",
        outbound_policy=_policy(str(reference), **policy_overrides),
        outbound_trace=trace,
        cost_ledger=CostLedger(tmp_path / "costs.sqlite3"),
    )


def test_dedicated_image_adapter_exact_payload_and_metadata_only_trace(tmp_path, monkeypatch):
    from herandhim.core.image_gen import generator as module
    from herandhim.core.outbound.trace import OutboundTrace

    reference = tmp_path / "canonical.png"
    reference.write_bytes(_png_bytes())
    trace_path = tmp_path / "trace.jsonl"
    captured = []

    def post(url, **kwargs):
        captured.append((url, kwargs))
        return _Response({
            "model": "bytedance-seed/seedream-5-0-flash",
            "data": [{"b64_json": base64.b64encode(_png_bytes("blue")).decode()}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "cost": 0.018},
        })

    monkeypatch.setattr(module.requests, "post", post)
    result = _generator(
        tmp_path, reference, OutboundTrace(trace_path),
    ).generate("current scene only", reference_image=str(reference))

    assert len(captured) == 1
    url, request = captured[0]
    assert url == "https://mock.openrouter.test/api/v1/images"
    assert request["json"]["model"] == "bytedance-seed/seedream-5-0-flash"
    assert request["json"]["resolution"] == "1K"
    assert request["json"]["aspect_ratio"] == "2:3"
    assert request["json"]["n"] == 1
    assert len(request["json"]["input_references"]) == 1
    assert not ({"messages", "history", "memory", "provider", "user_media"} & request["json"].keys())
    assert len(result) == 1

    trace_text = trace_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace_text.splitlines()]
    assert events[0]["transport"] == "images_api"
    assert events[-1]["cost_usd"] == 0.018
    assert "current scene only" not in trace_text
    assert request["json"]["input_references"][0]["image_url"]["url"] not in trace_text
    assert "synthetic-secret-value" not in trace_text


def test_image_adapter_rejects_non_allowlisted_reference_before_network(tmp_path, monkeypatch):
    from herandhim.core.image_gen import generator as module

    canonical = tmp_path / "canonical.png"
    other = tmp_path / "user-upload.png"
    canonical.write_bytes(_png_bytes())
    other.write_bytes(_png_bytes("green"))
    calls = []
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: calls.append(1))

    with pytest.raises(module.SeedreamError, match="not allowlisted"):
        _generator(tmp_path, canonical).generate("scene", reference_image=str(other))
    assert calls == []


@pytest.mark.parametrize("change", [
    {"model": "vendor/other"},
    {"transport": "chat_completions"},
    {"resolution": "2K"},
    {"aspectRatio": "1:1"},
    {"maxImages": 2},
    {"allowedReferenceAssets": []},
    {"allowedReferenceAssets": ["/one.png", "/two.png"]},
    {"allowedProviders": ["provider-a"], "activeProvider": "provider-a"},
])
def test_image_policy_rejects_route_reference_and_parameter_drift(tmp_path, change):
    from herandhim.core.outbound.policy import PolicyError

    reference = tmp_path / "canonical.png"
    reference.write_bytes(_png_bytes())
    with pytest.raises(PolicyError):
        _policy(str(reference), **change)


def test_image_adapter_zero_retry_on_invalid_response(tmp_path, monkeypatch):
    from herandhim.core.image_gen import generator as module

    reference = tmp_path / "canonical.png"
    reference.write_bytes(_png_bytes())
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return _Response({"data": []})

    monkeypatch.setattr(module.requests, "post", post)
    with pytest.raises(module.SeedreamError, match="exactly one image"):
        _generator(tmp_path, reference).generate("scene", reference_image=str(reference))
    assert len(calls) == 1


def test_image_daily_cost_cap_rejects_before_network(tmp_path, monkeypatch):
    from herandhim.core.image_gen import generator as module

    reference = tmp_path / "canonical.png"
    reference.write_bytes(_png_bytes())
    generator = _generator(
        tmp_path, reference, maxCostUsdPerCall=0.03, maxCostUsdPerDay=0.03,
    )
    generator.cost_ledger.reserve("image_generation", 0.03, 0.03)
    calls = []
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: calls.append(1))

    with pytest.raises(module.SeedreamError, match="daily_cost_cap_exceeded"):
        generator.generate("scene", reference_image=str(reference))
    assert calls == []


class _TelegramAPI:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.sent = []

    async def send_photo(self, **kwargs):
        self.sent.append((kwargs["chat_id"], kwargs["photo"].read(), kwargs.get("caption")))
        if self.fail:
            raise TimeoutError("synthetic ambiguous timeout")


def _bot(tmp_path, api, *, allowed=(42,)):
    from herandhim.channels.telegram_bot import TelegramBot
    from herandhim.core.telegram_transport import TelegramTransportStore

    bot = TelegramBot(
        SimpleNamespace(), "synthetic-not-a-real-token", allowed_users=list(allowed),
        transport_store=TelegramTransportStore(tmp_path / "telegram.sqlite3"),
    )
    bot._app = SimpleNamespace(bot=api)
    return bot


def test_photo_is_copied_to_durable_outbox_before_telegram_io(tmp_path):
    source = tmp_path / "generated.png"
    content = _png_bytes()
    source.write_bytes(content)
    api = _TelegramAPI()
    bot = _bot(tmp_path, api)

    assert asyncio.run(bot.send_photo(42, str(source), "hello"))
    assert api.sent == [(42, content, "hello")]
    media_dir = tmp_path / "telegram_media"
    durable = list(media_dir.iterdir())
    assert len(durable) == 1 and durable[0].read_bytes() == content


def test_explicit_media_bypasses_proactive_silence_and_kill_switch(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    source = tmp_path / "generated.png"
    source.write_bytes(_png_bytes())
    store = TelegramTransportStore(tmp_path / "telegram.sqlite3")
    store.set_delivery_enabled(False)
    store.set_silence(
        "telegram:42",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )
    accepted = store.enqueue_media(
        "telegram:42", 42, source, "explicit", origin="explicit",
    )

    claimed = store.claim_next_media("telegram:42")
    assert claimed is not None
    assert claimed.id == accepted.id
    assert claimed.origin == "explicit"


def test_proactive_media_obeys_kill_switch_and_silence(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    source = tmp_path / "generated.png"
    source.write_bytes(_png_bytes())
    store = TelegramTransportStore(tmp_path / "telegram.sqlite3")
    accepted = store.enqueue_media(
        "telegram:42", 42, source, "proactive", origin="proactive",
    )
    store.set_delivery_enabled(False)
    assert store.claim_next_media("telegram:42") is None
    assert store.media_status(accepted.id) == "pending"

    store.set_delivery_enabled(True)
    store.set_silence(
        "telegram:42",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert store.claim_next_media("telegram:42") is None
    assert store.media_status(accepted.id) == "pending"


def test_media_restart_and_ambiguous_timeout_are_at_most_once(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    source = tmp_path / "generated.png"
    source.write_bytes(_png_bytes())
    api = _TelegramAPI(fail=True)
    bot = _bot(tmp_path, api)
    assert not asyncio.run(bot.send_photo(42, str(source)))

    restarted_store = TelegramTransportStore(tmp_path / "telegram.sqlite3")
    restarted = _bot(tmp_path, api)
    restarted._transport = restarted_store
    assert not asyncio.run(restarted._deliver_pending_photo("telegram:42"))
    assert len(api.sent) == 1


def test_owner_gate_precedes_media_outbox(tmp_path):
    source = tmp_path / "generated.png"
    source.write_bytes(_png_bytes())
    bot = _bot(tmp_path, _TelegramAPI(), allowed=(42,))

    with pytest.raises(PermissionError):
        asyncio.run(bot.send_photo(7, str(source)))
    assert bot._transport.pending_media_count("telegram:7") == 0


def test_explicit_photo_sender_uses_same_durable_bot_contract(tmp_path):
    from herandhim.core import tools

    source = tmp_path / "generated.png"
    source.write_bytes(_png_bytes())
    api = _TelegramAPI()
    bot = _bot(tmp_path, api)

    async def exercise():
        loop = asyncio.get_running_loop()
        bot._register_file_sender(loop, 42)
        result = await loop.run_in_executor(
            None, lambda: tools.send_photo(str(source), "explicit", session_id="telegram:42"),
        )
        return result

    assert "sent" in asyncio.run(exercise()).lower()
    assert len(api.sent) == 1
    assert bot._transport.pending_media_count("telegram:42") == 0


def test_agent_allows_only_one_photo_generation_attempt_per_user_turn(monkeypatch):
    from herandhim.core import tools
    from herandhim.core.agent import Agent

    calls = []
    monkeypatch.setitem(
        tools.AVAILABLE_TOOLS,
        "take_selfie",
        lambda **kwargs: calls.append(kwargs) or "Error: synthetic provider failure",
    )
    agent = Agent.__new__(Agent)
    agent.verbose = False
    agent.session_id = "telegram:42"
    agent._media_tool_lock = Lock()
    agent._media_tool_used_this_turn = False

    tool_call = MagicMock()
    tool_call.function.name = "take_selfie"
    tool_call.function.arguments = '{"scene_hint":"home","caption":"hi"}'

    first = Agent._execute_tool_call(agent, tool_call)
    second = Agent._execute_tool_call(agent, tool_call)

    assert first == "Error: synthetic provider failure"
    assert "only one photo-generation attempt" in second
    assert len(calls) == 1


@pytest.mark.parametrize("photo_request", [
    "Можешь мне своё фото прислать?",
    "Фото своё можешь прислать?",
    "Пришли мне селфи сейчас",
    "Хочу твоё фото",
])
def test_explicit_photo_request_forces_selfie_tool(photo_request):
    from herandhim.core.agent import Agent

    assert Agent._initial_tool_choice(photo_request) == {
        "type": "function",
        "function": {"name": "take_selfie"},
    }


def test_inbound_photo_discussion_keeps_automatic_tool_choice():
    from herandhim.core.agent import Agent

    assert Agent._initial_tool_choice("Что изображено на этом фото?") == "auto"
    assert Agent._initial_tool_choice([
        {"type": "text", "text": "Посмотри на фото"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}},
    ]) == "auto"


def test_proactive_selfie_uses_same_durable_bot_contract(tmp_path, monkeypatch):
    from herandhim.core import image_gen
    from herandhim.core.image_gen import selfie
    from herandhim.scheduler.proactive import ProactiveMessenger

    source = tmp_path / "generated.png"
    source.write_bytes(_png_bytes())
    api = _TelegramAPI()
    bot = _bot(tmp_path, api)
    result = SimpleNamespace(path=str(source), caption=lambda: "proactive")
    monkeypatch.setattr(image_gen, "take_selfie", lambda: result)
    monkeypatch.setattr(selfie, "is_enabled", lambda: True)
    monkeypatch.setattr("herandhim.scheduler.proactive.config.get", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("herandhim.scheduler.proactive.random.random", lambda: 0.0)

    messenger = ProactiveMessenger(SimpleNamespace(), bot, bot._transport)
    asyncio.run(messenger._maybe_send_selfie(42))
    assert api.sent == [(42, _png_bytes(), "proactive")]
    assert bot._transport.pending_media_count("telegram:42") == 0
