"""OpenAI-compatible LLM provider.

Works with any API that follows the OpenAI chat-completions contract:
DeepSeek, Grok (xAI), Kimi (Moonshot), GLM (Zhipu), and others.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import httpx
from openai import OpenAI

from ..outbound.costs import CostLedger
from ..outbound.envelopes import prepare_messages
from ..outbound.images import normalize_message_images
from ..outbound.policy import OutboundPolicy, PolicyError
from ..outbound.redaction import Redactor
from ..outbound.trace import OutboundTrace
from ..outbound.types import OutboundPurpose
from .base import LLMProvider
from .response import MockChoice, MockFunction, MockMessage, MockResponse, MockToolCall


def _promote_reasoning_to_content(resp: Any) -> None:
    """If a response's `content` is empty but `reasoning_content` is set
    (DeepSeek V4 quirk), copy reasoning into content so callers see the
    actual answer rather than a blank message bubble.
    """
    try:
        msg = resp.choices[0].message
    except (AttributeError, IndexError):
        return
    if msg is None or getattr(msg, "content", None):
        return
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        try:
            msg.content = reasoning
        except Exception:
            pass


def _has_visible_output(resp: Any) -> bool:
    """A paid completion must contain user-visible text or a tool call."""
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError):
        return False
    content = getattr(message, "content", None)
    return bool(content is not None and str(content).strip()) or bool(
        getattr(message, "tool_calls", None)
    )


class OpenAICompatibleProvider(LLMProvider):
    """Thin wrapper around the OpenAI SDK for chat completions."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        supports_images: bool | None = None,
        outbound_policy: OutboundPolicy | None = None,
        loaded_secrets: tuple[str, ...] = (),
        outbound_trace: OutboundTrace | None = None,
        cost_ledger: CostLedger | None = None,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=0 if outbound_policy is not None else 2,
        )
        self.model_name = model_name
        self.outbound_policy = outbound_policy
        self.requires_outbound_purpose = outbound_policy is not None
        self._redactor = Redactor((*loaded_secrets, api_key))
        self._trace = outbound_trace or OutboundTrace()
        self._cost_ledger = (
            cost_ledger or CostLedger() if outbound_policy is not None else None
        )
        self._call_counts: OrderedDict[str, int] = OrderedDict()
        # Auto-detect vision support from the model name. Only explicit
        # OpenAI-compatible vision endpoints make the cut.
        #
        # NOTE on DeepSeek: as of 2026-05-26, the hosted DeepSeek API still
        # 400s on `image_url` content for every published model id we have
        # access to (verified empirically — ``deepseek-chat``,
        # ``deepseek-reasoner`` AND ``deepseek-v4-flash`` all reject it with
        # "unknown variant `image_url`, expected `text`"). The marketing
        # material about V4 being multimodal is ahead of the hosted API.
        # When DeepSeek opens it for real, re-add `deepseek-v4` here.
        if supports_images is None:
            n = (model_name or "").lower()
            supports_images = (
                n.startswith("deepseek-vl")        # self-hosted VL2 family only
                or "vision" in n                   # gpt-4-vision, grok-vision, …
                or n.startswith("qwen-vl") or n.startswith("qwen2-vl") or n.startswith("qwen2.5-vl")
                or n.startswith("llava")           # llava-1.5, llava-next, bakllava…
                or n.startswith("bakllava")
                or n.startswith(("qwen2.5vl", "qwen3-vl", "qwen3vl"))  # Ollama tags drop the hyphen
                or n.startswith("minicpm-v")
                or n.startswith("moondream")
                or n.startswith("pixtral")
                or n.startswith("internvl")
                or n.startswith("yi-vl")
                or n.startswith("moonshot-v1-vision")
            )
        self.supports_images = supports_images

    def _scrub_images_if_unsupported(
        self, messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Defensive: when the model can't see images, walk every message and
        flatten any ``image_url`` content parts that may have leaked in from
        an earlier vision-enabled turn or from session replay. Without this,
        the live agent's in-memory history keeps poisoning subsequent text
        turns with a 400 from the API.
        """
        if self.supports_images:
            return messages
        scrubbed: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                scrubbed.append(msg)
                continue
            text_bits: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    text_bits.append(str(part))
                elif part.get("type") == "text":
                    text_bits.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    text_bits.append("[image attached — vision not enabled on this model]")
            scrubbed.append({**msg, "content": "\n".join(b for b in text_bits if b)})
        return scrubbed

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        purpose: OutboundPurpose | str | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.outbound_policy is not None:
            reservation_id: str | None = None
            try:
                req = self._secured_request(messages, tools, tool_choice, purpose, kwargs)
                policy = self.outbound_policy.for_purpose(purpose)
                assert self._cost_ledger is not None
                reservation_id = self._cost_ledger.reserve(
                    policy.purpose.value,
                    policy.max_cost_usd_per_call,
                    policy.max_cost_usd_per_day,
                )
                started = time.monotonic()
                resp = self.client.chat.completions.create(**req)
            except PolicyError as exc:
                self._trace_reject(purpose, str(exc))
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                error_code = (
                    "provider_rate_limited" if status == 429
                    else "provider_request_failed"
                )
                self._trace_transport_failure(
                    purpose,
                    error_code=error_code,
                    http_status=status if isinstance(status, int) else None,
                )
                raise PolicyError(error_code) from None
            _promote_reasoning_to_content(resp)
            actual_cost = self._reported_cost(resp)
            if actual_cost is not None and reservation_id is not None:
                self._cost_ledger.reconcile(reservation_id, actual_cost)
                if actual_cost > policy.max_cost_usd_per_call + 1e-12:
                    self._trace_success(purpose, resp, started)
                    raise PolicyError("provider_cost_cap_exceeded")
            self._trace_success(purpose, resp, started)
            if not _has_visible_output(resp):
                raise PolicyError("provider_empty_response")
            return resp
        req: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._scrub_images_if_unsupported(messages),
            **kwargs,
        }
        if tools:
            req["tools"] = tools
            req["tool_choice"] = tool_choice

        resp = self.client.chat.completions.create(**req)
        _promote_reasoning_to_content(resp)
        return resp

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        purpose: OutboundPurpose | str | None = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, Any]:
        """Stream OpenAI-compatible responses, yielding text deltas."""
        if self.outbound_policy is not None:
            reservation_id: str | None = None
            try:
                req = self._secured_request(messages, tools, tool_choice, purpose, kwargs)
                policy = self.outbound_policy.for_purpose(purpose)
                assert self._cost_ledger is not None
                reservation_id = self._cost_ledger.reserve(
                    policy.purpose.value,
                    policy.max_cost_usd_per_call,
                    policy.max_cost_usd_per_day,
                )
                req["stream"] = True
                req["stream_options"] = {"include_usage": True}
            except PolicyError as exc:
                self._trace_reject(purpose, str(exc))
                raise
        else:
            req = {
                "model": self.model_name,
                "messages": self._scrub_images_if_unsupported(messages),
                "stream": True,
                **kwargs,
            }
            if tools:
                req["tools"] = tools
                req["tool_choice"] = tool_choice

        content_text = ""
        reasoning_text = ""  # DeepSeek V4 puts the actual answer here when in
                              # thinking mode; we fall back to it if `content`
                              # ends up empty.
        tool_calls_acc: dict[int, dict] = {}
        actual_cost: float | None = None

        try:
            started = time.monotonic()
            chunks = self.client.chat.completions.create(**req)
            for chunk in chunks:
                chunk_cost = self._reported_cost(chunk)
                if chunk_cost is not None:
                    actual_cost = chunk_cost
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    content_text += delta.content
                    yield {"type": "text_delta", "text": delta.content}

                # Some providers (notably DeepSeek V4) stream the visible answer
                # via `reasoning_content` instead of `content`. Capture it so we
                # can fall back if `content_text` is empty at end-of-stream.
                r = getattr(delta, "reasoning_content", None)
                if r:
                    reasoning_text += r
                    yield {"type": "text_delta", "text": r}

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "args": "",
                            }
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["args"] += tc_delta.function.arguments
        except Exception:
            if self.outbound_policy is not None:
                self._trace_transport_failure(purpose)
                raise PolicyError("provider_request_failed") from None
            raise
        if self.outbound_policy is not None:
            if actual_cost is not None and reservation_id is not None:
                self._cost_ledger.reconcile(reservation_id, actual_cost)
                if actual_cost > policy.max_cost_usd_per_call + 1e-12:
                    raise PolicyError("provider_cost_cap_exceeded")
            self._trace.write({
                "purpose": str(getattr(purpose, "value", purpose)),
                "policy_version": self.outbound_policy.version,
                "budget_decision": "allowed",
                "latency_ms": int((time.monotonic() - started) * 1000),
                **({"cost_usd": actual_cost} if actual_cost is not None else {}),
                "retry_count": 0,
            })

        mock_tool_calls = [
            MockToolCall(
                id=v["id"],
                function=MockFunction(name=v["name"], arguments=v["args"] or "{}"),
            )
            for v in sorted(tool_calls_acc.values(), key=lambda x: x["id"])
        ]

        # If the provider only filled reasoning_content (e.g. DeepSeek V4-Flash
        # for tiny prompts), surface that as the visible response.
        final_content = content_text or reasoning_text or None

        if self.outbound_policy is not None and not final_content and not mock_tool_calls:
            raise PolicyError("provider_empty_response")

        return MockResponse(choices=[
            MockChoice(message=MockMessage(
                content=final_content,
                tool_calls=mock_tool_calls or None,
            ))
        ])

    def _trace_reject(self, purpose: object, reason: str) -> None:
        self._trace.write({
            "purpose": str(getattr(purpose, "value", purpose) or "missing"),
            "policy_version": self.outbound_policy.version if self.outbound_policy else 0,
            "budget_decision": "rejected",
            "reason": reason,
            "retry_count": 0,
        })

    @staticmethod
    def _reported_cost(response: Any) -> float | None:
        dumped = response.model_dump() if hasattr(response, "model_dump") else {}
        value = (dumped.get("usage") or {}).get("cost")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
        return None

    def _trace_transport_failure(
        self,
        purpose: object,
        *,
        error_code: str = "provider_request_failed",
        http_status: int | None = None,
    ) -> None:
        event = {
            "purpose": str(getattr(purpose, "value", purpose) or "missing"),
            "policy_version": self.outbound_policy.version if self.outbound_policy else 0,
            "budget_decision": "allowed",
            "error_code": error_code,
            "retry_count": 0,
        }
        if http_status is not None:
            event["http_status"] = http_status
        self._trace.write(event)

    def _trace_success(self, purpose: object, response: Any, started: float) -> None:
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        dumped = response.model_dump() if hasattr(response, "model_dump") else {}
        dumped_usage = dumped.get("usage") or {}
        event = {
            "purpose": str(getattr(purpose, "value", purpose)),
            "policy_version": self.outbound_policy.version if self.outbound_policy else 0,
            "budget_decision": "allowed",
            "latency_ms": int((time.monotonic() - started) * 1000),
            "returned_model": getattr(response, "model", None),
            "returned_provider": dumped.get("provider"),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "cost_usd": dumped_usage.get("cost"),
            "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None,
            "retry_count": 0,
        }
        self._trace.write({key: value for key, value in event.items() if value is not None})

    def _secured_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        purpose: OutboundPurpose | str | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an immutable OpenRouter request or reject it before I/O."""
        assert self.outbound_policy is not None
        request_group = str(kwargs.pop("outbound_request_group", "") or uuid4().hex)
        forbidden = {"model", "models", "provider", "extra_body", "plugins"} & set(kwargs)
        if forbidden:
            raise PolicyError(f"caller cannot override secured fields: {sorted(forbidden)}")
        policy = self.outbound_policy.for_purpose(purpose)
        call_count = self._call_counts.get(request_group, 0) + 1
        if call_count > policy.max_calls:
            raise PolicyError("call_budget_exceeded")
        self._call_counts[request_group] = call_count
        self._call_counts.move_to_end(request_group)
        while len(self._call_counts) > 1024:
            self._call_counts.popitem(last=False)
        clean_messages, clean_tools, redactions = prepare_messages(
            messages, tools, policy, self._redactor,
        )
        clean_messages, image_meta = normalize_message_images(clean_messages)
        trace_messages = json.loads(json.dumps(clean_messages, ensure_ascii=False))
        for message in trace_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"] = {"url": "[normalized-image]"}
        text_bytes = len(json.dumps(
            trace_messages, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        tool_schema_bytes = len(json.dumps(
            clean_tools, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        request: dict[str, Any] = {
            "model": policy.model,
            "messages": clean_messages,
            "extra_body": {"provider": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in policy.provider_block.items()
            }},
            "timeout": httpx.Timeout(
                float(policy.total_timeout_seconds),
                connect=float(policy.connect_timeout_seconds),
                read=float(policy.first_byte_timeout_seconds),
            ),
            **kwargs,
        }
        if policy.max_output_tokens:
            requested = int(request.get("max_completion_tokens", request.pop("max_tokens", policy.max_output_tokens)))
            if requested > policy.max_output_tokens:
                raise PolicyError("output_budget_exceeded")
            request["max_completion_tokens"] = requested
        if clean_tools:
            request["tools"] = clean_tools
            request["tool_choice"] = tool_choice
        self._trace.write({
            "purpose": policy.purpose.value,
            "policy_version": self.outbound_policy.version,
            "transport": "chat_completions",
            "model": policy.model,
            "provider": policy.active_provider,
            "message_count": len(clean_messages),
            "text_bytes": text_bytes,
            "estimated_input_tokens": (text_bytes + tool_schema_bytes + 3) // 4,
            "tool_count": len(clean_tools),
            "tool_schema_bytes": tool_schema_bytes,
            "recent_turns": sum(
                1 for message in clean_messages if message.get("role") != "system"
            ),
            "memory_snippets": sum(
                str(message.get("content", "")).count("\n- key=")
                for message in clean_messages if message.get("role") == "system"
            ),
            "image_count": image_meta["image_count"],
            "image_bytes": image_meta["image_bytes"],
            "redactions": redactions,
            "budget_decision": "allowed",
            "retry_count": 0,
        })
        return request
