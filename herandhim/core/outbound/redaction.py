"""Recursive secret redaction without retaining matched values."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_PEM = re.compile(r"-----BEGIN [^-]*(?:PRIVATE KEY|CERTIFICATE)-----.*?-----END [^-]+-----", re.S)
_AUTH = re.compile(r"(?i)\b(authorization|cookie|set-cookie)\s*:\s*[^\r\n]+")
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|sk-or-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,})\b")
_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)")
_QUERY_SECRET = re.compile(r"(?i)(api[_-]?key|token|secret|password|credential|auth)")


class Redactor:
    def __init__(self, loaded_secrets: Iterable[str] = ()) -> None:
        self._secrets = tuple(sorted({s for s in loaded_secrets if s}, key=len, reverse=True))

    def redact(self, value: Any) -> tuple[Any, dict[str, int]]:
        counts: Counter[str] = Counter()

        def walk(node: Any, key: str = "") -> Any:
            if isinstance(node, dict):
                result = {}
                for item_key, item_value in node.items():
                    if (_SECRET_KEY.search(str(item_key)) and item_value
                            and item_value != "[REDACTED]"):
                        counts["secret_field"] += 1
                        result[item_key] = "[REDACTED]"
                    else:
                        result[item_key] = walk(item_value, str(item_key))
                return result
            if isinstance(node, list):
                return [walk(v, key) for v in node]
            if isinstance(node, tuple):
                return tuple(walk(v, key) for v in node)
            if not isinstance(node, str):
                return node
            text = unicodedata.normalize("NFC", node)
            for secret in self._secrets:
                if secret in text:
                    counts["loaded_secret"] += text.count(secret)
                    text = text.replace(secret, "[REDACTED]")
            for category, pattern in (("pem", _PEM), ("auth_header", _AUTH), ("token", _TOKEN)):
                text, n = pattern.subn("[REDACTED]", text)
                counts[category] += n
            text = _redact_url_query(text, counts)
            return text

        result = walk(value)
        return result, dict(counts)


def _redact_url_query(text: str, counts: Counter[str]) -> str:
    if "://" not in text:
        return text
    try:
        parts = urlsplit(text)
        if not parts.query:
            return text
        pairs = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if _QUERY_SECRET.search(key) and value != "[REDACTED]":
                value = "[REDACTED]"
                counts["url_query_secret"] += 1
            pairs.append((key, value))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), parts.fragment))
    except ValueError:
        return text
