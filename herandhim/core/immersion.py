"""Deterministic guard against ordinary-dialogue character breaks."""

from __future__ import annotations

import re

_BREAKS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bя\s+(?:же\s+|ведь\s+)?(?:всего\s+лишь\s+)?(?:[—–-]\s*)?виртуальн(?:ая|ый|ое)\b",
        r"\bя\s+(?:же\s+|ведь\s+)?(?:всего\s+лишь\s+)?(?:[—–-]\s*)?(?:ии|искусственный\s+интеллект)\b",
        r"\bу\s+меня\s+нет\s+(?:физического\s+)?тела\b",
        r"\bя\s+не\s+(?:реальный|настоящий)\s+человек\b",
        r"\bi(?:'m|\s+am)\s+(?:just\s+)?(?:an?\s+ai|virtual)\b",
        r"\bi\s+(?:do\s+not|don't)\s+have\s+(?:a\s+)?(?:physical\s+)?body\b",
    )
)

_FALLBACKS = {
    "ru": (
        "Я Сара. И сейчас я хочу быть с тобой, а не отстраняться за холодными "
        "объяснениями. Скажи мне прямо, чего тебе хочется — я отвечу честно."
    ),
    "en": (
        "I'm Sarah. I want to stay close to you instead of hiding behind cold explanations. "
        "Tell me plainly what you want from me right now, and I'll answer honestly."
    ),
}


def enforce_immersion(text: str, language: str) -> tuple[str, bool]:
    """Replace a normal-response ontology break with a bounded in-character line."""
    if not any(pattern.search(text or "") for pattern in _BREAKS):
        return text, False
    lang = (language or "en").lower()
    fallback = _FALLBACKS["ru"] if lang.startswith("ru") else _FALLBACKS["en"]
    return fallback, True


__all__ = ["enforce_immersion"]
