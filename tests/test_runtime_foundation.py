"""End-to-end dialogue/history/memory/relationship foundation tests."""

from __future__ import annotations

import json
import threading

import httpx
from openai import OpenAI

from herandhim.core.llm.openai_compatible import OpenAICompatibleProvider
from herandhim.core.outbound.costs import CostLedger
from herandhim.core.outbound.policy import OutboundPolicy
from herandhim.core.outbound.trace import OutboundTrace
from herandhim.core.persistent_agent import PersistentAgent
from herandhim.core.session_store import SessionStore


def _policy() -> OutboundPolicy:
    routes = {
        "dialogue": ("mistralai/mistral-large-4-0", "mistral/eu", 12_000, 1_600, 4),
        "vision": ("openai/gpt-6.1-sol", "azure/eu", 6_000, 400, 1),
        "memory_consolidation": (
            "anthropic/claude-haiku-5.5", "google-vertex/europe", 3_000, 800, 1,
        ),
        "relationship_analysis": (
            "anthropic/claude-haiku-5.5", "google-vertex/europe", 2_000, 256, 1,
        ),
    }
    purposes = {}
    for name, (model, provider, input_tokens, output_tokens, calls) in routes.items():
        purposes[name] = {
            "model": model,
            "allowedProviders": [provider],
            "activeProvider": provider,
            "maxInputTokensEstimated": input_tokens,
            "maxOutputTokens": output_tokens,
            "maxCallsPerTurn": calls,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 20,
            "totalTimeoutSeconds": 90,
            "maxCostUsdPerCall": 0.01,
            "maxCostUsdPerDay": 0.10,
        }
    purposes["image_generation"] = {"enabled": False}
    return OutboundPolicy.from_mapping({
        "version": 1,
        "traceRetentionDays": 30,
        "aggregateRetentionDays": 180,
        "openrouter": {
            "requireAccountPromptLoggingDisabled": True,
            "requireAccountZdr": True,
            "requireNonZeroKeyCreditLimit": True,
        },
        "purposes": purposes,
    })


def _response(model: str, *, content: str | None = None, tool_call: dict | None = None):
    message = {"role": "assistant", "content": content}
    if tool_call:
        message["tool_calls"] = [tool_call]
    return {
        "id": "mock-response",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "provider": "mock-exact-provider",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_call else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0001},
    }


def test_runtime_persists_multiturn_memory_and_relationship_across_restart(tmp_path):
    captured: list[dict] = []
    lock = threading.Lock()
    dialogue_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dialogue_calls
        body = json.loads(request.content)
        with lock:
            captured.append(body)
        model = body["model"]
        messages = body["messages"]
        if model == "mistralai/mistral-large-4-0":
            dialogue_calls += 1
            if dialogue_calls == 1:
                return httpx.Response(200, json=_response(model, tool_call={
                    "id": "remember-1",
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "arguments": json.dumps({
                            "key": "user_location",
                            "content": "Илья теперь живёт в Москве",
                        }, ensure_ascii=False),
                    },
                }))
            if dialogue_calls == 2:
                return httpx.Response(200, json=_response(model, content="Запомнила: ты теперь в Москве."))
            return httpx.Response(200, json=_response(model, content="Да, ты живёшь в Москве."))

        first = str(messages[0].get("content", ""))
        if "emotion-analysis sidecar" in first:
            return httpx.Response(200, json=_response(model, content=json.dumps({
                "sentiment": "positive",
                "intensity": 0.8,
                "topic": "shared update",
                "summary": "User shared an important life update",
                "follow_up": False,
            })))
        return httpx.Response(200, json=_response(model, content=json.dumps({
            "op": "UPDATE",
            "key": "user_city",
            "value": "Илья теперь живёт в Москве",
        }, ensure_ascii=False)))

    trace_path = tmp_path / "outbound-trace.jsonl"
    provider = OpenAICompatibleProvider(
        api_key="sk-or-test-runtime-secret",
        base_url="https://mock.openrouter.test/v1",
        model_name="mistralai/mistral-large-4-0",
        outbound_policy=_policy(),
        outbound_trace=OutboundTrace(trace_path),
        cost_ledger=CostLedger(tmp_path / "outbound-costs.sqlite3"),
    )
    provider.client = OpenAI(
        api_key="test",
        base_url="https://mock.openrouter.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    memory_dir = tmp_path / "context" / "memory"
    skills_dir = tmp_path / "context" / "skills"
    memory_dir.mkdir(parents=True)
    skills_dir.mkdir(parents=True)
    store = SessionStore(str(tmp_path / "context" / "sessions"))

    def make_agent() -> PersistentAgent:
        return PersistentAgent(
            provider=provider,
            memory_dir=str(memory_dir),
            skills_dirs=[str(skills_dir)],
            knowledge_path=None,
            persona_path=None,
            soul_path=None,
            profile_path=None,
            verbose=False,
            auto_compaction=False,
            store=store,
            session_id="runtime:foundation",
        )

    first = make_agent()
    first.memory.storage.set("user_city", "Илья живёт в Казани")
    assert "Москве" in first.chat("Запомни: я переехал из Казани в Москву.")
    first._bg_executor.shutdown(wait=True)  # noqa: SLF001

    assert first.memory.storage.get("user_city") == "Илья теперь живёт в Москве"
    relationship_before = first.memory.relationship.get_all()
    assert relationship_before["trust"] > 50

    restarted = make_agent()
    restored_text = "\n".join(str(message.get("content", "")) for message in restarted.messages)
    assert "переехал" in restored_text
    assert restarted.memory.storage.get("user_city") == "Илья теперь живёт в Москве"
    assert restarted.memory.relationship.get_all() == relationship_before
    assert "Москве" in restarted.chat("В каком городе я теперь живу?")
    restarted._bg_executor.shutdown(wait=True)  # noqa: SLF001

    dialogue_bodies = [body for body in captured if body["model"] == "mistralai/mistral-large-4-0"]
    assert len(dialogue_bodies) == 3
    assert all(body["reasoning_effort"] == "none" for body in dialogue_bodies)
    assert all(body["provider"] == {
        "only": ["mistral/eu"],
        "order": ["mistral/eu"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
        "require_parameters": True,
    } for body in dialogue_bodies)
    assert any("Locally retrieved memories" in repr(body["messages"]) for body in dialogue_bodies)
    assert any("user_city" in repr(body["messages"]) for body in dialogue_bodies)

    trace_text = trace_path.read_text(encoding="utf-8")
    assert "переехал" not in trace_text
    assert "Илья теперь" not in trace_text
    assert "sk-or-test-runtime-secret" not in trace_text
    trace_rows = [json.loads(line) for line in trace_text.splitlines()]
    assert {"dialogue", "memory_consolidation", "relationship_analysis"} <= {
        row["purpose"] for row in trace_rows
    }
