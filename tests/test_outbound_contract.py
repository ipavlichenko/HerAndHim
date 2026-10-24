"""D-029 static and captured-request contract tests (no live calls)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from herandhim.core.llm.openai_compatible import OpenAICompatibleProvider
from herandhim.core.outbound.costs import CostLedger
from herandhim.core.outbound.envelopes import prepare_messages
from herandhim.core.outbound.images import normalize_data_url
from herandhim.core.outbound.policy import OutboundPolicy, PolicyError
from herandhim.core.outbound.redaction import Redactor
from herandhim.core.outbound.trace import OutboundTrace


def policy_mapping(**overrides):
    rows = {}
    for purpose, output, timeout in (
        ("dialogue", 1600, 90), ("vision", 400, 90),
        ("image_generation", 0, 180), ("memory_consolidation", 800, 30),
        ("relationship_analysis", 256, 30),
    ):
        rows[purpose] = {
            "model": ("bytedance-seed/seedream-5-0-flash"
                      if purpose == "image_generation" else f"vendor/{purpose}"),
            "maxInputTokensEstimated": 12_000,
            "maxOutputTokens": output,
            "maxCallsPerTurn": 1,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 20,
            "totalTimeoutSeconds": timeout,
            "maxCostUsdPerCall": 0.10,
            "maxCostUsdPerDay": 1.00,
        }
        if purpose == "image_generation":
            rows[purpose].update({
                "transport": "images_api",
                "resolution": "1K",
                "aspectRatio": "2:3",
                "maxImages": 1,
                "allowedReferenceAssets": ["/canonical/reference.png"],
            })
        else:
            rows[purpose].update({
                "allowedProviders": ["provider-a", "provider-b"],
                "activeProvider": "provider-a",
            })
    for purpose, values in overrides.items():
        rows[purpose].update(values)
    return {
        "version": 1,
        "traceRetentionDays": 30,
        "aggregateRetentionDays": 180,
        "openrouter": {
            "requireAccountPromptLoggingDisabled": True,
            "requireAccountZdr": True,
            "requireNonZeroKeyCreditLimit": True,
        },
        "purposes": rows,
    }


def make_provider(policy=None):
    ledger = CostLedger(Path(tempfile.mkdtemp()) / "costs.sqlite3")
    provider = OpenAICompatibleProvider(
        api_key="sk-or-test-secret-value",
        base_url="https://openrouter.ai/api/v1",
        model_name="vendor/dialogue",
        supports_images=True,
        outbound_policy=policy or OutboundPolicy.from_mapping(policy_mapping()),
        cost_ledger=ledger,
    )
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    return provider


def test_secured_adapter_durably_reserves_and_reconciles_reported_cost(tmp_path):
    ledger = CostLedger(tmp_path / "costs.sqlite3")
    provider = OpenAICompatibleProvider(
        api_key="test", base_url="https://mock.openrouter.test/v1",
        model_name="vendor/dialogue",
        outbound_policy=OutboundPolicy.from_mapping(policy_mapping()),
        cost_ledger=ledger,
    )
    response = MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])
    response.model_dump.return_value = {"usage": {"cost": 0.012}}
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = response
    provider.chat(messages=[{"role": "user", "content": "hello"}], purpose="dialogue")
    with sqlite3.connect(tmp_path / "costs.sqlite3") as conn:
        amount = conn.execute(
            "SELECT amount_usd FROM cost_reservations WHERE purpose='dialogue'",
        ).fetchone()[0]
    assert amount == pytest.approx(0.012)


def test_ambiguous_cost_reservation_blocks_next_call_before_network(tmp_path):
    raw = policy_mapping(dialogue={
        "maxCostUsdPerCall": 0.10, "maxCostUsdPerDay": 0.15,
    })
    provider = OpenAICompatibleProvider(
        api_key="test", base_url="https://mock.openrouter.test/v1",
        model_name="vendor/dialogue",
        outbound_policy=OutboundPolicy.from_mapping(raw),
        cost_ledger=CostLedger(tmp_path / "costs.sqlite3"),
    )
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
    )
    provider.chat(messages=[{"role": "user", "content": "one"}], purpose="dialogue")
    with pytest.raises(PolicyError, match="daily_cost_cap_exceeded"):
        provider.chat(messages=[{"role": "user", "content": "two"}], purpose="dialogue")
    assert provider.client.chat.completions.create.call_count == 1


def test_openrouter_builder_fails_closed_without_policy(monkeypatch):
    from herandhim import config
    from herandhim.main import _build_named_provider

    monkeypatch.setattr(config, "get", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="outboundPolicy"):
        _build_named_provider("openrouter", api_key="k", base_url="https://openrouter.ai/api/v1")


def test_openrouter_builder_uses_policy_model_and_declares_vision(monkeypatch):
    from herandhim import config
    from herandhim.main import _build_named_provider

    raw = policy_mapping()
    monkeypatch.setattr(config, "get", lambda *args, **kwargs: raw)
    provider = _build_named_provider(
        "openrouter", api_key="k", base_url="https://openrouter.ai/api/v1",
    )
    assert provider.model_name == "vendor/dialogue"
    assert provider.supports_images is True
    assert provider.requires_outbound_purpose is True


def test_disabled_image_purpose_does_not_block_dialogue_policy():
    raw = policy_mapping(image_generation={"enabled": False})
    policy = OutboundPolicy.from_mapping(raw)
    assert policy.for_purpose("dialogue").model == "vendor/dialogue"
    with pytest.raises(PolicyError, match="disabled"):
        policy.for_purpose("image_generation")


def png_data_url(size=(8, 8), *, metadata=False):
    out = io.BytesIO()
    image = Image.new("RGB", size, "red")
    kwargs = {"exif": Image.Exif()} if metadata else {}
    if metadata:
        kwargs["exif"][0x010E] = "private description"
    image.save(out, format="PNG", **kwargs)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


@pytest.mark.parametrize("purpose", [
    "dialogue", "vision", "memory_consolidation",
])
def test_openrouter_body_has_exact_privacy_block_for_every_scoped_purpose(purpose):
    provider = make_provider()
    messages = [{"role": "user", "content": "hello"}]
    if purpose == "vision":
        messages[0]["content"] = [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": png_data_url()}},
        ]
    provider.chat(messages=messages, purpose=purpose)
    request = provider.client.chat.completions.create.call_args.kwargs
    assert request["model"] == f"vendor/{purpose}"
    assert request["extra_body"] == {"provider": {
        "only": ["provider-a"], "order": ["provider-a"],
        "allow_fallbacks": False, "data_collection": "deny",
        "zdr": True, "require_parameters": True,
    }}
    assert "models" not in request and "plugins" not in request


@pytest.mark.parametrize("purpose", [
    "dialogue", "vision", "memory_consolidation",
])
def test_actual_http_body_serializes_exact_policy_for_scoped_purposes(purpose):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "mock", "object": "chat.completion", "created": 0,
            "model": captured[-1]["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    provider = OpenAICompatibleProvider(
        api_key="test", base_url="https://mock.openrouter.test/v1",
        model_name="vendor/dialogue", supports_images=True,
        outbound_policy=OutboundPolicy.from_mapping(policy_mapping()),
    )
    provider.client = OpenAI(
        api_key="test", base_url="https://mock.openrouter.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = [{"role": "user", "content": "hello"}]
    if purpose == "vision":
        messages[0]["content"] = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": png_data_url()}},
        ]
    provider.chat(messages=messages, purpose=purpose)
    assert len(captured) == 1
    body = captured[0]
    assert body["model"] == f"vendor/{purpose}"
    assert body["provider"] == {
        "only": ["provider-a"], "order": ["provider-a"],
        "allow_fallbacks": False, "data_collection": "deny",
        "zdr": True, "require_parameters": True,
    }
    assert not ({"models", "plugins"} & set(body))


@pytest.mark.parametrize("purpose", [None, "unknown"])
def test_missing_or_unknown_purpose_fails_before_network(purpose):
    provider = make_provider()
    with pytest.raises(PolicyError):
        provider.chat(messages=[{"role": "user", "content": "hello"}], purpose=purpose)
    provider.client.chat.completions.create.assert_not_called()


def test_vision_text_budget_does_not_count_base64_image_bytes():
    raw = policy_mapping(vision={"maxInputTokensEstimated": 100})
    policy = OutboundPolicy.from_mapping(raw).for_purpose("vision")
    payload = "a" * 100_000
    messages, _tools, _counts = prepare_messages(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the current image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
            ],
        }],
        [],
        policy,
        Redactor(()),
    )
    assert messages[0]["content"][1]["image_url"]["url"].endswith(payload)


@pytest.mark.parametrize("field", ["model", "models", "provider", "extra_body", "plugins"])
def test_caller_cannot_override_secured_fields(field):
    provider = make_provider()
    with pytest.raises(PolicyError):
        provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            purpose="dialogue", **{field: "override"},
        )
    provider.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("change", [
    {"model": "REQUIRED_EXACT_ID"},
    {"model": "openrouter/auto"},
    {"model": "vendor/model:free"},
    {"allowedProviders": []},
    {"activeProvider": "not-allowed"},
    {"maxCostUsdPerCall": 0},
])
def test_policy_rejects_placeholders_routes_and_missing_budgets(change):
    with pytest.raises(PolicyError):
        OutboundPolicy.from_mapping(policy_mapping(dialogue=change))


def test_timeout_and_output_budget_are_purpose_specific():
    provider = make_provider()
    provider.chat(messages=[{"role": "user", "content": "hello"}], purpose="dialogue")
    request = provider.client.chat.completions.create.call_args.kwargs
    assert request["timeout"].connect == 5.0
    assert request["timeout"].read == 20.0
    assert request["timeout"].write == 90.0
    assert request["max_completion_tokens"] == 1600
    with pytest.raises(PolicyError):
        provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            purpose="dialogue", max_tokens=1601,
        )


def test_dialogue_excludes_prior_images_and_bounds_recent_history():
    provider = make_provider()
    messages = [{"role": "system", "content": "persona"}]
    messages += [{"role": "user", "content": f"turn-{i}"} for i in range(10)]
    messages.insert(2, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://forbidden.example/image.jpg"}},
    ]})
    provider.chat(messages=messages, purpose="dialogue")
    sent = provider.client.chat.completions.create.call_args.kwargs["messages"]
    assert len(sent) <= 9
    assert "image_url" not in repr(sent)
    assert "turn-0" not in repr(sent)


def test_dialogue_excludes_upstream_full_volatile_plan_and_memory():
    provider = make_provider()
    provider.chat(messages=[
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "[[VOLATILE]]FULL TODAY PLAN AND MEMORY"},
        {"role": "user", "content": "hello"},
    ], purpose="dialogue")
    sent = provider.client.chat.completions.create.call_args.kwargs["messages"]
    assert "FULL TODAY PLAN" not in repr(sent)


def test_application_owned_planner_prompt_uses_system_envelope():
    provider = make_provider()
    prompt = "plan the companion's day\n" + ("bounded detail " * 600)
    assert len(prompt.encode("utf-8")) > 8_000

    provider.chat(
        messages=[{"role": "system", "content": prompt}],
        purpose="dialogue",
    )

    sent = provider.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent == [{"role": "system", "content": prompt}]


def test_bounded_planner_call_can_disable_reasoning_without_loosening_output_cap():
    provider = make_provider()

    provider.chat(
        messages=[{"role": "system", "content": "plan the companion's day"}],
        purpose="dialogue",
        reasoning_effort="none",
    )

    request = provider.client.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "none"
    assert request["max_completion_tokens"] == 1600


def test_secured_agent_disables_reasoning_for_dialogue_but_not_vision():
    from herandhim.core.agent import Agent

    agent = Agent.__new__(Agent)
    agent.provider = type("Provider", (), {"requires_outbound_purpose": True})()

    dialogue = agent._outbound_purpose_kwargs("hello")
    vision = agent._outbound_purpose_kwargs([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
    ])

    assert dialogue["purpose"] == "dialogue"
    assert dialogue["reasoning_effort"] == "none"
    assert vision["purpose"] == "vision"
    assert "reasoning_effort" not in vision


def test_dialogue_keeps_tool_pairs_atomic_within_budget():
    provider = make_provider()
    messages = [{"role": "user", "content": f"old-{i}"} for i in range(5)] + [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "tc-1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc-1", "content": "result"},
        {"role": "user", "content": "current"},
    ]
    provider.chat(messages=messages, purpose="dialogue")
    sent = provider.client.chat.completions.create.call_args.kwargs["messages"]
    roles = [message["role"] for message in sent]
    assert roles[-3:] == ["assistant", "tool", "user"]


def test_dialogue_prunes_oldest_atomic_history_to_fit_tools_in_global_budget():
    provider = make_provider()
    tools = [{
        "type": "function",
        "function": {
            "name": "large_schema",
            "description": "schema " * 3_500,
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    messages = [
        {"role": "system", "content": "persona " * 700},
        {"role": "system", "content": "[[D029_CONTEXT]]\ncontext " * 250},
        {"role": "user", "content": "old-user " * 700},
        {"role": "assistant", "content": "old-reply " * 700},
        {"role": "user", "content": "current"},
    ]

    provider.chat(messages=messages, tools=tools, purpose="dialogue")

    request = provider.client.chat.completions.create.call_args.kwargs
    sent = request["messages"]
    encoded = json.dumps([sent, request["tools"]], ensure_ascii=False).encode("utf-8")
    assert (len(encoded) + 3) // 4 <= 12_000
    assert sent[-1] == {"role": "user", "content": "current"}
    assert "old-user" not in repr(sent)


def test_dialogue_preserves_current_tool_call_group_while_pruning_history():
    provider = make_provider()
    tools = [{
        "type": "function",
        "function": {
            "name": "large_schema",
            "description": "schema " * 3_500,
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    current_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "current-call",
            "type": "function",
            "function": {"name": "large_schema", "arguments": "{}"},
        }],
    }
    messages = [
        {"role": "system", "content": "persona " * 700},
        {"role": "system", "content": "[[D029_CONTEXT]]\ncontext " * 250},
        {"role": "user", "content": "old-user " * 700},
        {"role": "assistant", "content": "old-reply " * 700},
        current_call,
        {"role": "tool", "tool_call_id": "current-call", "content": "tool-result"},
    ]

    provider.chat(messages=messages, tools=tools, purpose="dialogue")

    sent = provider.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[-2:] == [current_call, messages[-1]]
    assert "old-user" not in repr(sent)


def test_vision_uses_only_current_normalized_image():
    provider = make_provider()
    messages = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": [
            {"type": "text", "text": "current"},
            {"type": "image_url", "image_url": {"url": png_data_url((4096, 64), metadata=True)}},
        ]},
    ]
    provider.chat(messages=messages, purpose="vision")
    sent = provider.client.chat.completions.create.call_args.kwargs["messages"]
    urls = [part["image_url"]["url"] for msg in sent if isinstance(msg.get("content"), list)
            for part in msg["content"] if part.get("type") == "image_url"]
    assert len(urls) == 1
    normalized, size, dimensions = normalize_data_url(urls[0])
    assert normalized == urls[0]
    assert max(dimensions) <= 2048 and size <= 4 * 1024 * 1024


def test_vision_rejects_remote_url_before_network():
    provider = make_provider()
    with pytest.raises(PolicyError, match="remote_image_urls_forbidden"):
        provider.chat(messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.test/a.jpg"}},
        ]}], purpose="vision")
    provider.client.chat.completions.create.assert_not_called()


def test_redactor_covers_nested_tools_urls_pem_and_loaded_secrets():
    value = {"tool": {"password": "plain", "arg": "sk-or-abcdefghijklmnop"},
             "url": "https://x.test/?token=abc", "loaded": "MY-SECRET"}
    clean, counts = Redactor(("MY-SECRET",)).redact(value)
    assert "plain" not in repr(clean) and "MY-SECRET" not in repr(clean)
    assert "abcdefghijklmnop" not in repr(clean) and "token=abc" not in repr(clean)
    assert sum(counts.values()) >= 3
    again, second_counts = Redactor(("MY-SECRET",)).redact(clean)
    assert again == clean and not any(second_counts.values())


def test_memory_secret_reject_makes_zero_network_calls():
    provider = make_provider()
    with pytest.raises(PolicyError, match="memory_secret"):
        provider.chat(messages=[{"role": "user", "content": "sk-or-abcdefghijklmnop"}],
                      purpose="memory_consolidation")
    provider.client.chat.completions.create.assert_not_called()


def test_relationship_analysis_has_its_own_bounded_typed_envelope():
    provider = make_provider()
    provider.chat(
        messages=[
            {"role": "system", "content": "Return relationship JSON."},
            {"role": "user", "content": "User was warm; assistant replied warmly."},
        ],
        tools=[],
        purpose="relationship_analysis",
    )
    request = provider.client.chat.completions.create.call_args.kwargs
    assert request["model"] == "vendor/relationship_analysis"
    assert len(request["messages"]) == 2
    assert request["extra_body"]["provider"]["allow_fallbacks"] is False


def test_call_budget_reject_makes_zero_additional_network_calls():
    provider = make_provider()
    kwargs = {"messages": [{"role": "user", "content": "hello"}],
              "purpose": "dialogue", "outbound_request_group": "turn-1"}
    provider.chat(**kwargs)
    with pytest.raises(PolicyError, match="call_budget_exceeded"):
        provider.chat(**kwargs)
    assert provider.client.chat.completions.create.call_count == 1


def test_provider_failure_is_sanitized_and_never_retried():
    provider = make_provider()
    provider.client.chat.completions.create.side_effect = RuntimeError(
        "echoed prompt and sk-or-abcdefghijklmnop"
    )
    with pytest.raises(PolicyError, match="provider_request_failed") as caught:
        provider.chat(messages=[{"role": "user", "content": "private prompt"}],
                      purpose="dialogue")
    assert "private prompt" not in str(caught.value)
    assert provider.client.chat.completions.create.call_count == 1


def test_empty_paid_response_fails_closed_without_retry():
    provider = make_provider()
    response = MagicMock(choices=[MagicMock(
        message=MagicMock(content=None, tool_calls=None, reasoning_content=None),
        finish_reason="length",
    )])
    response.model_dump.return_value = {"usage": {"cost": 0.004}}
    provider.client.chat.completions.create.return_value = response
    with pytest.raises(PolicyError, match="provider_empty_response"):
        provider.chat(
            messages=[{"role": "user", "content": "hello"}], purpose="dialogue",
        )
    assert provider.client.chat.completions.create.call_count == 1


def test_photo_memory_reuses_vision_reply_without_second_cloud_call():
    from herandhim.channels.telegram_bot import TelegramBot

    agent = MagicMock()
    asyncio.run(TelegramBot._remember_their_photo(
        object(), agent, "A calm beach at sunset with two people walking together",
    ))
    agent.provider.chat.assert_not_called()
    agent.memory.remember.assert_called_once()


def test_trace_rejects_content_and_raw_error_fields(tmp_path):
    trace = OutboundTrace(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError):
        trace.write({"purpose": "dialogue", "prompt": "secret"})
    with pytest.raises(ValueError):
        trace.write({"purpose": "dialogue", "raw_error": "secret"})
    trace.write({"purpose": "dialogue", "budget_decision": "allowed", "retry_count": 0})
    assert "dialogue" in (tmp_path / "trace.jsonl").read_text()


def test_streaming_serializes_same_policy():
    provider = make_provider()
    provider.client.chat.completions.create.return_value = iter(())
    with pytest.raises(PolicyError, match="provider_empty_response"):
        list(provider.chat_stream(messages=[{"role": "user", "content": "hello"}],
                                  purpose="dialogue"))
    request = provider.client.chat.completions.create.call_args.kwargs
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["extra_body"]["provider"]["zdr"] is True
    assert request["extra_body"]["provider"]["allow_fallbacks"] is False
