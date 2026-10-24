"""
Context compaction for herandhim.

Compaction summarises older conversation history into a compact summary entry
and keeps recent messages intact — preventing context-window overflows in long
sessions while preserving important information.

Inspired by openclaw's compaction model:
  https://docs.openclaw.ai/concepts/compaction

How it works
------------
1. Split chat history into "old" (to summarise) + "recent" (to keep verbatim).
2. Memory flush — silently ask the LLM to extract key facts from the old
   messages and persist them via the agent's `remember` tool, so nothing
   important is lost permanently.
3. Summarise — call the LLM with a structured, companion-oriented prompt.
   The summary has mandatory sections (relationship thread, confirmed user
   facts, open plans & dates, emotional tone, pending asks & promises,
   do-not-lose) so relationship continuity survives repeated compactions
   instead of eroding into a generic paragraph.
4. Persist — append the summary to `context/compaction/history.jsonl` as an
   audit trail.
5. Replace — swap the old messages with a single [Compaction Summary] system
   message and increment the compaction counter.

Token estimation
----------------
We don't bundle a tokeniser, so we use a conservative character-based
approximation: 1 token ≈ 4 characters.  This is good enough for triggering
auto-compaction; the actual model enforces the hard limit.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm.base import LLMProvider
    from .memory.manager import MemoryManager

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_AUTO_THRESHOLD_TOKENS = 10000  # trigger auto-compaction at ~10k tokens
DEFAULT_RECENT_KEEP = 6                # keep last N chat messages verbatim

# Mandatory sections of the companion-structured summary. Freeform "3–8
# sentence" summaries lose exactly the things a partner would never forget —
# open plans, nicknames, promises — so each compaction is forced to account
# for them explicitly.
SUMMARY_SECTIONS: list[tuple[str, str]] = [
    ("Relationship thread",
     "where things stand between you two — running jokes, nicknames in use, "
     "how you talk to each other"),
    ("User facts (confirmed)",
     "stable facts the user stated: name, city, job, birthday, family, health"),
    ("Open plans & dates",
     "anything scheduled, promised or anticipated, with dates/times"),
    ("Emotional tone",
     "the user's recent mood and how you have been responding to it"),
    ("Pending asks & promises",
     "questions still awaiting an answer; things either of you said you'd do"),
    ("Do-not-lose",
     "anything else it would be jarring for a partner to forget"),
]

# Hard cap so structured summaries can't balloon across compact chains.
MAX_SUMMARY_CHARS = 3000
MAX_COMPACTION_TRANSCRIPT_CHARS = 8_000
MAX_MEMORY_FLUSH_CHARS = 7_000


def _compaction_log_file() -> str:
    from .. import config as _cfg
    return os.path.join(str(_cfg.HERANDHIM_HOME), "context", "compaction", "history.jsonl")


COMPACTION_LOG_FILE = None  # resolved lazily


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(messages: list[dict]) -> int:
    """Rough character-based token estimate for a list of messages."""
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    return total_chars // CHARS_PER_TOKEN


# ── JSONL persistence ─────────────────────────────────────────────────────────

def persist_compaction(summary: str, message_count: int, log_path: str | None = None) -> None:
    """Append one compaction entry to the JSONL audit log."""
    log_path = log_path or _compaction_log_file()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "summarised_messages": message_count,
        "summary": summary,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.debug("[Compaction] Persisted to %s", log_path)


# ── Message → plain text ──────────────────────────────────────────────────────

def messages_to_text(messages: list[dict]) -> str:
    """Convert a message list to a readable transcript for summarisation."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and not content and m.get("tool_calls"):
            content = f"[called tools: {[tc.get('function', {}).get('name') if isinstance(tc, dict) else tc.function.name for tc in m.get('tool_calls', [])]}]"
        if role == "tool":
            content = f"[tool result]: {content[:300]}{'...' if len(content) > 300 else ''}"
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _bounded_summary_history(messages: list[dict]) -> str:
    """Keep the previous summary plus the newest old turns within one call.

    Durable facts are flushed separately across every deterministic chunk.
    The summariser therefore needs the prior relationship summary and the
    newest pre-compaction thread, not an unbounded transcript dump.
    """
    pinned = [
        m for m in messages
        if str(m.get("content") or "").startswith("[Compaction Summary")
    ]
    others = [m for m in messages if m not in pinned]
    pinned_text = messages_to_text(pinned)
    remaining = max(0, MAX_COMPACTION_TRANSCRIPT_CHARS - len(pinned_text))
    selected: list[dict] = []
    used = 0
    for message in reversed(others):
        rendered = messages_to_text([message])
        if not rendered:
            continue
        if used + len(rendered) + 1 > remaining:
            break
        selected.append(message)
        used += len(rendered) + 1
    tail = messages_to_text(list(reversed(selected)))
    return "\n".join(part for part in (pinned_text, tail) if part)


def _memory_flush_chunks(messages: list[dict]) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for message in messages:
        rendered = messages_to_text([message])
        if not rendered:
            continue
        size = len(rendered) + 1
        if current and current_chars + size > MAX_MEMORY_FLUSH_CHARS:
            chunks.append(current)
            current = []
            current_chars = 0
        # A single pathological message is locally clipped; current user
        # content in the live dialogue path is still rejected, never truncated.
        if size > MAX_MEMORY_FLUSH_CHARS:
            clipped = dict(message)
            clipped["content"] = str(message.get("content") or "")[
                :MAX_MEMORY_FLUSH_CHARS
            ]
            chunks.append([clipped])
            continue
        current.append(message)
        current_chars += size
    if current:
        chunks.append(current)
    return chunks


# ── Structured summary prompt ─────────────────────────────────────────────────

def build_summary_prompt(history_text: str, focus: str = "") -> str:
    """Build the companion-structured summarisation prompt.

    Every section is mandatory (write "(none)" when empty) and capped at a few
    short bullets, so summaries stay both complete and small.
    """
    sections = "\n".join(
        f"## {name}\n({hint})" for name, hint in SUMMARY_SECTIONS
    )
    return (
        "You are compacting the older chat history between an AI companion and "
        "their partner. Produce a structured summary with EXACTLY the sections "
        "below, in this order, using the exact '## ' headings. Under each "
        "heading write at most 3 short bullets (each under 25 words). If a "
        f"section has nothing, write '(none)'.{focus}\n\n"
        f"{sections}\n\n"
        f"CONVERSATION:\n{history_text}\n\n"
        "Return only the structured summary:"
    )


def _cap_summary(summary: str) -> str:
    """Hard-cap summary length so compact chains can't balloon the context."""
    if len(summary) <= MAX_SUMMARY_CHARS:
        return summary
    cut = summary[:MAX_SUMMARY_CHARS]
    # Break on a line boundary when one is reasonably close.
    nl = cut.rfind("\n")
    if nl > MAX_SUMMARY_CHARS // 2:
        cut = cut[:nl]
    logger.warning(
        "[Compaction] Summary exceeded %d chars — truncated.", MAX_SUMMARY_CHARS
    )
    return cut.rstrip() + "\n…(truncated)"


# ── Memory flush ──────────────────────────────────────────────────────────────

def memory_flush(
    messages_to_flush: list[dict],
    provider: "LLMProvider",
    memory: "MemoryManager",
) -> int:
    """
    Silent LLM call that extracts key facts from old messages and saves them
    to long-term memory before those messages are discarded.

    Returns the number of facts saved.
    """
    if not messages_to_flush:
        return 0

    saved = 0
    for chunk in _memory_flush_chunks(messages_to_flush):
        history_text = messages_to_text(chunk)
        prompt = (
            "You extract durable memory for a personal companion. Identify only "
            "confirmed facts, preferences, decisions, boundaries, plans, promises, "
            "and relationship context worth keeping. Return a JSON array of objects "
            "with stable snake_case 'key' and concise 'value'. If nothing matters, "
            "return [].\n\n"
            f"TRANSCRIPT:\n{history_text}\n\n"
            "Return ONLY valid JSON, no explanation."
        )
        try:
            purpose_kw = (
                {"purpose": "memory_consolidation"}
                if getattr(provider, "requires_outbound_purpose", False) else {}
            )
            response = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                tool_choice="none",
                max_tokens=600,
                **purpose_kw,
            )
            raw = (response.choices[0].message.content or "[]").strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
            facts: list[dict] = json.loads(raw)
            for fact in facts:
                key = str(fact.get("key", "")).strip()
                value = str(fact.get("value", "")).strip()
                if not key or not value:
                    continue
                store_extracted = getattr(type(memory), "remember_extracted", None)
                if callable(store_extracted):
                    memory.remember_extracted(key=key, content=value)
                else:
                    memory.remember(key=key, content=value)
                saved += 1
        except Exception as exc:
            logger.warning("[Compaction] Memory flush chunk failed (non-fatal): %s", exc)
    logger.info("[Compaction] Memory flush saved %d fact(s).", saved)
    return saved


# ── Core compact function ─────────────────────────────────────────────────────

def compact(
    messages: list[dict],
    provider: "LLMProvider",
    memory: "MemoryManager | None" = None,
    recent_keep: int = DEFAULT_RECENT_KEEP,
    instruction: str | None = None,
    log_path: str | None = None,
    flush_memory: bool = True,
) -> tuple[list[dict], str]:
    """
    Compact conversation history.

    Parameters
    ----------
    messages      : full message list (system + chat)
    provider      : LLM provider used for summarisation
    memory        : MemoryManager for pre-compaction memory flush (optional)
    recent_keep   : number of recent chat messages to keep verbatim
    instruction   : optional extra focus hint for the summarisation prompt
    log_path      : where to persist the compaction JSONL entry

    Returns
    -------
    (new_messages, summary_text)
    """
    prior_summaries = [
        m for m in messages
        if m.get("role") == "system"
        and str(m.get("content") or "").startswith("[Compaction Summary")
    ]
    system_msgs = [
        m for m in messages
        if m.get("role") == "system" and m not in prior_summaries
    ]
    chat_msgs = [m for m in messages if m.get("role") != "system"]

    if len(chat_msgs) <= recent_keep:
        logger.info("[Compaction] Not enough history to compact (%d messages).", len(chat_msgs))
        return messages, ""

    # Fold the previous summary into the new one instead of accumulating one
    # immortal system message per cycle. This keeps compaction bounded while
    # preserving its relationship/fact/commitment evidence for the next pass.
    to_summarise = prior_summaries + chat_msgs[:-recent_keep]
    to_keep      = chat_msgs[-recent_keep:]

    logger.info(
        "[Compaction] Summarising %d message(s), keeping %d recent.",
        len(to_summarise), len(to_keep),
    )

    # 1. Memory flush — save important facts before discarding old messages
    if memory is not None and flush_memory:
        memory_flush(to_summarise, provider, memory)

    # 2. Summarise — structured so relationship continuity survives the chain
    history_text = _bounded_summary_history(to_summarise)
    focus = f"\nAdditional focus: {instruction}" if instruction else ""
    summarise_prompt = build_summary_prompt(history_text, focus)
    try:
        purpose_kw = (
            {"purpose": "memory_consolidation"}
            if getattr(provider, "requires_outbound_purpose", False) else {}
        )
        response = provider.chat(
            messages=[{"role": "user", "content": summarise_prompt}],
            tools=[],
            tool_choice="none",
            max_tokens=800,
            **purpose_kw,
        )
        summary = _cap_summary((response.choices[0].message.content or "").strip())
    except Exception as exc:
        logger.error("[Compaction] Summarisation failed: %s", exc)
        raise

    # 3. Persist
    persist_compaction(summary, len(to_summarise), log_path=log_path)

    # 4. Build new message list
    #
    # Don't stamp the summary with a wall-clock time. The agent's volatile
    # block (added per-turn in _build_volatile_context) is the single source
    # of truth for "what time is it"; a "UTC" or local stamp here just gives
    # the LLM another number to anchor on and frequently desync from the
    # real now (e.g. "Compaction Summary — 11:46 UTC" sitting next to the
    # volatile "20:11 Shanghai" line makes the model split the difference).
    from . import timectx
    summary_system_msg = {
        "role": "system",
        "content": (
            f"[Compaction Summary — {timectx.now_in_bot_tz().strftime('%Y-%m-%d')}]\n"
            f"{summary}"
        ),
    }
    new_messages = system_msgs + [summary_system_msg] + to_keep

    return new_messages, summary
