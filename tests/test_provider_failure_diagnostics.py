"""Synthetic diagnostic tests; no provider I/O or private text logging."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, OpenAI
from test_outbound_contract import make_provider

from herandhim.channels.telegram_bot import _model_failure_text
from herandhim.core.outbound.failures import classify_failure
from herandhim.core.outbound.policy import PolicyError
from herandhim.core.outbound.trace import OutboundTrace


@pytest.mark.parametrize("status,code", [
    (400, "provider_request_rejected"), (401, "provider_auth_failed"),
    (402, "provider_payment_required"), (403, "provider_access_denied"),
    (404, "provider_request_rejected"), (408, "provider_timeout"),
    (429, "provider_rate_limited"), (500, "provider_unavailable"),
    (503, "provider_unavailable"), (504, "provider_timeout"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_status_failure_is_correlated_safe_and_not_retried(tmp_path, status, code, stream):
    provider = make_provider()
    trace_path = tmp_path / "trace.jsonl"
    provider._trace = OutboundTrace(trace_path)
    exc = RuntimeError("PRIVATE_SECRET echoed body")
    exc.status_code = status
    provider.client.chat.completions.create.side_effect = exc
    with pytest.raises(PolicyError, match=code) as caught:
        if stream:
            list(provider.chat_stream(messages=[{"role": "user", "content": "PRIVATE_PROMPT"}],
                                      purpose="dialogue"))
        else:
            provider.chat(messages=[{"role": "user", "content": "PRIVATE_PROMPT"}],
                          purpose="dialogue")
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    start, failure = events
    assert start["request_id"] == failure["request_id"]
    assert len(failure["request_id"]) == 32
    assert failure["error_code"] == code
    assert failure["http_status"] == status
    assert failure["latency_ms"] >= 0
    assert datetime.fromisoformat(failure["timestamp"]).tzinfo is not None
    assert failure["retry_count"] == 0
    assert provider.client.chat.completions.create.call_count == 1
    request = provider.client.chat.completions.create.call_args.kwargs
    assert "request_id" not in request and "trace_request_id" not in request
    text = _model_failure_text("Error: " + str(caught.value))
    assert "Код: " + failure["request_id"][:12] in text
    assert code not in text
    assert "PRIVATE" not in text + trace_path.read_text() + str(caught.value)


@pytest.mark.parametrize("error_type,phase", [
    (httpx.ConnectTimeout, "connect"), (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"), (httpx.PoolTimeout, "pool"),
])
def test_sdk_timeout_cause_is_classified_without_message(error_type, phase):
    exc = APITimeoutError(request=httpx.Request("POST", "https://invalid.test"))
    exc.__cause__ = error_type("PRIVATE_SECRET")
    assert classify_failure(exc) == {"error_code": "provider_timeout", "timeout_phase": phase}


def test_network_unknown_and_cyclic_causes_are_bounded():
    exc = APIConnectionError(request=httpx.Request("POST", "https://invalid.test"))
    assert classify_failure(exc) == {"error_code": "provider_connection_failed"}
    unknown = RuntimeError("PRIVATE_SECRET")
    unknown.__cause__ = unknown
    assert classify_failure(unknown) == {"error_code": "provider_request_failed"}


def test_mid_stream_timeout_has_same_request_id(tmp_path):
    provider = make_provider()
    trace_path = tmp_path / "trace.jsonl"
    provider._trace = OutboundTrace(trace_path)

    def chunks():
        yield MagicMock(choices=[], model_dump=lambda: {})
        raise httpx.ReadTimeout("PRIVATE_SECRET")

    provider.client.chat.completions.create.return_value = chunks()
    with pytest.raises(PolicyError, match="provider_timeout"):
        list(provider.chat_stream(messages=[{"role": "user", "content": "hello"}],
                                  purpose="dialogue"))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert events[0]["request_id"] == events[-1]["request_id"]
    assert events[-1]["timeout_phase"] == "read"
    assert provider.client.chat.completions.create.call_count == 1


def test_concurrent_calls_do_not_share_request_id(tmp_path):
    provider = make_provider()
    trace_path = tmp_path / "trace.jsonl"
    provider._trace = OutboundTrace(trace_path)
    provider.client.chat.completions.create.side_effect = httpx.ConnectError("PRIVATE_SECRET")

    def call(_):
        with pytest.raises(PolicyError):
            provider.chat(messages=[{"role": "user", "content": "hello"}], purpose="dialogue")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(call, range(2)))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    ids = {row["request_id"] for row in events}
    assert len(ids) == 2
    assert all(sum(e["request_id"] == ident for e in events) == 2 for ident in ids)


def test_telegram_does_not_echo_untrusted_suffix_or_error_text():
    for raw in ["Error: provider_timeout; request_id=<secret>", "Error: PRIVATE_SECRET"]:
        text = _model_failure_text(raw)
        assert "secret" not in text.lower() and "Код:" not in text


@pytest.mark.parametrize("kind", ["timeout", "connection", "503"])
def test_real_sdk_error_chain_is_classified_without_retries(tmp_path, kind):
    provider = make_provider()
    path = tmp_path / "trace.jsonl"
    provider._trace = OutboundTrace(path)
    calls = []

    def handler(request):
        calls.append(request)
        if kind == "timeout":
            raise httpx.ReadTimeout("PRIVATE_SECRET", request=request)
        if kind == "connection":
            raise httpx.ConnectError("PRIVATE_SECRET", request=request)
        return httpx.Response(503, json={"error": {"message": "PRIVATE_SECRET"}})

    provider.client = OpenAI(api_key="test", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(PolicyError) as caught:
            provider.chat(messages=[{"role": "user", "content": "hello"}], purpose="dialogue")
    finally:
        provider.client.close()
    expected = {"timeout": "provider_timeout", "connection": "provider_connection_failed",
                "503": "provider_unavailable"}[kind]
    assert expected in str(caught.value)
    assert len(calls) == 1
    assert "PRIVATE_SECRET" not in path.read_text() + str(caught.value)


def test_failed_scene_call_preserves_checkpoint(tmp_path, monkeypatch):
    from herandhim.core import agent as module
    from herandhim.core import quota

    monkeypatch.setattr(module.config, "HERANDHIM_HOME", tmp_path)
    monkeypatch.setattr(quota, "check_messages", lambda: None)
    monkeypatch.setattr(quota, "record_message", lambda: None)
    provider = make_provider()
    provider.client.chat.completions.create.side_effect = httpx.ReadTimeout("PRIVATE_SECRET")
    agent = module.Agent(provider, session_id="diagnostic-test", skills_dirs=[],
                         memory_dir=str(tmp_path / "memory"), knowledge_path=str(tmp_path / "knowledge"),
                         persona_path=str(tmp_path / "persona"), soul_path=str(tmp_path / "soul"),
                         auto_compaction=False)
    agent.roleplay.start("country_house_storm")
    agent.roleplay.mark_opened()
    before = agent.roleplay.path.read_bytes()
    reply = agent.chat("Ты здесь?")
    assert reply.startswith("Error: provider_timeout; request_id=")
    assert agent.roleplay.path.read_bytes() == before
    assert agent.roleplay.is_active()
    assert "PRIVATE_SECRET" not in reply
