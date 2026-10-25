"""Local metadata-only trace with a closed schema."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ALLOWED = {
    "timestamp", "request_id", "purpose", "policy_version", "transport", "model",
    "provider", "message_count", "text_bytes", "estimated_input_tokens", "image_count",
    "image_bytes", "tool_count", "tool_schema_bytes", "recent_turns", "memory_snippets",
    "redactions", "budget_decision", "reason", "http_status", "error_code",
    "timeout_phase", "latency_ms", "first_byte_ms", "returned_model", "returned_provider",
    "input_tokens", "output_tokens", "cost_usd", "finish_reason", "retry_count",
    "audio_bytes", "audio_duration_seconds", "transcript_chars", "text_chars",
    "audio_output_bytes", "voice", "audio_encoding",
}


class OutboundTrace:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def write(self, event: dict[str, Any]) -> None:
        forbidden = set(event) - _ALLOWED
        if forbidden:
            raise ValueError(f"forbidden outbound trace fields: {sorted(forbidden)}")
        if event.get("retry_count", 0) != 0:
            raise ValueError("outbound retry_count must be zero")
        if self.path is None:
            return
        event = {"timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"), **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
