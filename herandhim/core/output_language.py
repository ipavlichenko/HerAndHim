"""Cyrillic normalization and a bounded language check for Russian prose.

Never transliterate arbitrary English: repair known Russian misspellings, preserve
literal technical text, and distinguish isolated words from a language switch.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

RUSSIAN_OUTPUT_INSTRUCTION = (
    "Пиши по-русски кириллицей, без транслита и английских вставок. "
    "Сохраняй названия, ссылки, код и цитаты пользователя."
)
RUSSIAN_OUTPUT_FAILURE_TEXT = (
    "Не получилось подготовить ответ. Попробуй отправить сообщение ещё раз."
)
LEGACY_RUSSIAN_OUTPUT_FAILURE_TEXT = (
    "В ответе появилась непонятная латинская вставка. "
    "Этот вариант не принят; автоматического повтора не было."
)
RUSSIAN_OUTPUT_FAILURES = frozenset({RUSSIAN_OUTPUT_FAILURE_TEXT, LEGACY_RUSSIAN_OUTPUT_FAILURE_TEXT})


class RussianOutputError(ValueError):
    def __init__(self) -> None:
        # A safe enum, never the rejected text or token.
        super().__init__("unexpected_latin_output")


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LITERAL = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`|"
    r"(?:https?://|www\.)[^\s<>]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?<!\w)@[A-Za-z][A-Za-z0-9_]*",
    re.IGNORECASE,
)
_BRANDS_AND_TERMS = frozenset(word.casefold() for word in (
    "Telegram", "OpenAI", "ChatGPT", "Codex", "Google", "YouTube", "WhatsApp",
    "Spotify", "Figma", "Notion", "GitHub", "iPhone", "MacBook", "iOS",
    "Android", "Windows", "Linux", "Wi", "Fi", "AIGirla",
))
_ACRONYMS = frozenset({"USB", "PDF", "IT", "UI", "UX", "API", "GPT", "AI", "OK"})
# Deliberately small: the observed word and its unambiguous Russian inflections.
# This is not a general-purpose translator or a phonetic guess for unknown words.
_TRANSLIT = {
    "sama": "сама", "samu": "саму", "samo": "само", "samoy": "самой",
    "samoi": "самой", "samuyu": "самую", "sami": "сами", "samim": "самим",
    "samikh": "самих", "samih": "самих", "samimi": "самими",
    "samogo": "самого", "samomu": "самому",
}
_CONFUSABLES = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
})


def is_russian(language: str) -> bool:
    return (language or "").lower().replace("_", "-").split("-", 1)[0] == "ru"


def _has_latin(value: str) -> bool:
    return any("LATIN" in unicodedata.name(char, "") for char in value)


def latin_literals(sources: Iterable[str]) -> frozenset[str]:
    """Words explicitly supplied by a human or their chosen name, never a model."""
    return frozenset(
        token.casefold()
        for source in sources
        for token in _WORD.findall(source)
        if _has_latin(token)
    )


def enforce_russian_output(
    text: str, language: str, *, allowed_latin: Iterable[str] = (), strict: bool = True,
    tolerate_isolated_latin: bool = False,
) -> tuple[str, bool]:
    """Repair prose only, without rewriting URLs/code/names or guessing translations.

    ``strict=False`` is for repairing known mistakes in an outbound history copy;
    unknown historical wording stays intact rather than losing established facts.
    """
    if not text or not is_russian(language):
        return text, False
    allowed = _BRANDS_AND_TERMS | frozenset(token.casefold() for token in allowed_latin)
    literal_spans = [(match.start(), match.end()) for match in _LITERAL.finditer(text)]
    russian_prose = bool(_CYRILLIC.search(_LITERAL.sub("", text)))
    unresolved: list[str] = []

    def replace(match: re.Match) -> str:
        word = match.group()
        if not _has_latin(word) or word.casefold() in allowed or word in _ACRONYMS:
            return word
        if any(start <= match.start() < end for start, end in literal_spans):
            return word
        if _CYRILLIC.search(word):
            repaired = word.translate(_CONFUSABLES)
            if not _has_latin(repaired):
                return repaired
        elif russian_prose and word.casefold() in _TRANSLIT:
            repaired = _TRANSLIT[word.casefold()]
            return repaired.upper() if word.isupper() else repaired.capitalize() if word.istitle() else repaired
        unresolved.append(word)
        return word

    repaired = _WORD.sub(replace, text)
    if strict and unresolved:
        words = _WORD.findall(_LITERAL.sub("", repaired))
        russian_words = sum(bool(_CYRILLIC.search(word)) and not _has_latin(word) for word in words)
        isolated = (tolerate_isolated_latin and len(unresolved) <= 2
                    and russian_words >= 3 * len(unresolved)
                    and not any(_CYRILLIC.search(word) for word in unresolved))
        if not isolated:
            raise RussianOutputError()
    return repaired, repaired != text
