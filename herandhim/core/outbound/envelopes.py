"""Purpose-specific message minimisation and hard budgets."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .policy import PolicyError, PurposePolicy
from .redaction import Redactor
from .types import OutboundPurpose


def prepare_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    policy: PurposePolicy,
    redactor: Redactor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    purpose = policy.purpose
    if purpose is OutboundPurpose.DIALOGUE:
        selected = _dialogue(messages)
    elif purpose is OutboundPurpose.VISION:
        selected = _vision(messages)
    elif purpose is OutboundPurpose.MEMORY_CONSOLIDATION:
        selected = deepcopy(messages[:1])
    elif purpose is OutboundPurpose.RELATIONSHIP_ANALYSIS:
        selected = _relationship_analysis(messages)
    elif purpose is OutboundPurpose.IMAGE_GENERATION:
        selected = deepcopy(messages[:1])
    else:  # defensive even though enum parsing precedes this call
        raise PolicyError("unsupported outbound purpose")
    clean_messages, counts = redactor.redact(selected)
    clean_tools, tool_counts = redactor.redact(tools or [])
    for key, count in tool_counts.items():
        counts[key] = counts.get(key, 0) + count
    if purpose in {
        OutboundPurpose.MEMORY_CONSOLIDATION,
        OutboundPurpose.RELATIONSHIP_ANALYSIS,
    } and any(counts.values()):
        raise PolicyError("memory_secret_redaction_rejected")
    # Image bytes have their own hard envelope in ``outbound.images``. Counting
    # a base64 data URL as text here makes every realistic vision/reference
    # image exceed the text-token budget before image normalization can run.
    # Preserve message structure, but exclude the payload from this estimate.
    if purpose is OutboundPurpose.DIALOGUE:
        clean_messages = _fit_dialogue_to_total_budget(
            clean_messages,
            clean_tools,
            policy.max_input_tokens_estimated,
        )
    estimated_messages = _without_image_payloads(clean_messages)
    text_bytes = len(json.dumps([estimated_messages, clean_tools], ensure_ascii=False).encode("utf-8"))
    estimated = (text_bytes + 3) // 4
    if estimated > policy.max_input_tokens_estimated:
        raise PolicyError("input_budget_exceeded")
    return clean_messages, clean_tools, counts


def _without_image_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = deepcopy(messages)
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                part["image_url"] = {"url": "[normalized-image]"}
    return result


def _dialogue(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    systems = [deepcopy(m) for m in messages if m.get("role") == "system"
               and not str(m.get("content", "")).startswith(("[[VOLATILE]]", "[[D029_CONTEXT]]"))]
    contexts = [deepcopy(m) for m in messages if m.get("role") == "system"
                and str(m.get("content", "")).startswith("[[D029_CONTEXT]]")]
    chats = [_without_images(m) for m in messages if m.get("role") != "system"]
    chats = [m for m in chats if m is not None]
    persona = systems[:1]
    context = contexts[-1:]
    if sum(_content_bytes(m) for m in persona) > 16_000:
        raise PolicyError("dialogue_persona_budget_exceeded")
    if sum(_content_bytes(m) for m in context) > 14_000:
        raise PolicyError("dialogue_context_budget_exceeded")
    current = chats[-1:]
    recent = _take_recent_atomic(chats[:-1], 4)
    if current and _content_bytes(current[0]) > 8_000:
        raise PolicyError("dialogue_current_message_budget_exceeded")
    if sum(_content_bytes(m) for m in recent) > 16_000:
        raise PolicyError("dialogue_recent_history_budget_exceeded")
    return persona + context + recent + current


def _relationship_analysis(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the fixed analyzer instruction and the current bounded event."""
    if len(messages) != 2:
        raise PolicyError("relationship_analysis_requires_two_messages")
    selected = deepcopy(messages)
    if selected[0].get("role") != "system" or selected[1].get("role") != "user":
        raise PolicyError("relationship_analysis_invalid_roles")
    if _content_bytes(selected[0]) > 2_000 or _content_bytes(selected[1]) > 4_000:
        raise PolicyError("relationship_analysis_budget_exceeded")
    return selected


def _vision(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_index = next((i for i in range(len(messages) - 1, -1, -1)
                          if _has_image(messages[i])), None)
    if current_index is None:
        raise PolicyError("vision_requires_current_image")
    current = deepcopy(messages[current_index])
    images = [p for p in current.get("content", [])
              if isinstance(p, dict) and p.get("type") == "image_url"]
    if len(images) != 1:
        raise PolicyError("vision_requires_exactly_one_image")
    prior = [_without_images(m) for m in messages[:current_index] if m.get("role") != "system"]
    prior = [m for m in prior if m is not None][-2:]
    systems = [deepcopy(m) for m in messages if m.get("role") == "system"
               and not str(m.get("content", "")).startswith("[[VOLATILE]]")][:1]
    if systems and _content_bytes(systems[0]) > 8_000:
        raise PolicyError("vision_response_card_budget_exceeded")
    if sum(_content_bytes(m) for m in prior) > 8_000:
        raise PolicyError("vision_context_budget_exceeded")
    text_bytes = sum(len(str(p.get("text", "")).encode("utf-8"))
                     for p in current.get("content", [])
                     if isinstance(p, dict) and p.get("type") == "text")
    if text_bytes > 4_000:
        raise PolicyError("vision_caption_budget_exceeded")
    return systems + prior + [current]


def _has_image(message: dict[str, Any]) -> bool:
    return any(isinstance(p, dict) and p.get("type") == "image_url"
               for p in message.get("content", []) if isinstance(message.get("content"), list))


def _without_images(message: dict[str, Any]) -> dict[str, Any] | None:
    result = deepcopy(message)
    content = result.get("content")
    if not isinstance(content, list):
        return result
    parts = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image_url")]
    if not parts:
        return None
    result["content"] = parts
    return result


def _content_bytes(message: dict[str, Any]) -> int:
    return len(json.dumps(message.get("content"), ensure_ascii=False).encode("utf-8"))


def _fit_dialogue_to_total_budget(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens_estimated: int,
) -> list[dict[str, Any]]:
    """Drop only oldest dialogue history until messages + tools fit.

    Per-layer caps protect identity, selected context, and the current turn, but
    their maxima intentionally do not add up to the global request budget. Tool
    schemas also consume that budget. Preserve all leading system messages and
    the latest atomic chat group (including an assistant tool call plus all of
    its tool results), then retain the newest earlier groups that still fit.
    """
    split = 0
    while split < len(messages) and messages[split].get("role") == "system":
        split += 1
    systems = messages[:split]
    groups = _atomic_groups(messages[split:])
    if not groups:
        return messages

    current = groups[-1]
    recent = groups[:-1]
    while recent:
        candidate = systems + [item for group in recent for item in group] + current
        if _estimated_tokens(candidate, tools) <= max_tokens_estimated:
            return candidate
        recent.pop(0)
    candidate = systems + current
    return candidate


def _estimated_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    text_bytes = len(json.dumps([messages, tools], ensure_ascii=False).encode("utf-8"))
    return (text_bytes + 3) // 4


def _atomic_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
        groups.append(group)
    return groups


def _take_recent_atomic(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups = _atomic_groups(messages)
    selected: list[list[dict[str, Any]]] = []
    count = 0
    for group in reversed(groups):
        if count + len(group) > limit:
            break
        selected.append(group)
        count += len(group)
    return [message for group in reversed(selected) for message in group]
