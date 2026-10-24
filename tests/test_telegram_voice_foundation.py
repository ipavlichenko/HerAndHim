from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class _TelegramAPI:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.sent: list[tuple[int, bytes, str | None]] = []

    async def send_voice(self, **kwargs):
        self.sent.append(
            (kwargs["chat_id"], kwargs["voice"].read(), kwargs.get("caption"))
        )
        if self.fail:
            raise TimeoutError("synthetic provider detail must not trigger a retry")


def _bot(tmp_path, api, *, allowed=(42,)):
    from herandhim.channels.telegram_bot import TelegramBot
    from herandhim.core.telegram_transport import TelegramTransportStore

    bot = TelegramBot(
        SimpleNamespace(),
        "synthetic-not-a-real-token",
        allowed_users=list(allowed),
        transport_store=TelegramTransportStore(tmp_path / "telegram.sqlite3"),
    )
    bot._app = SimpleNamespace(bot=api)
    return bot


def test_voice_is_copied_to_durable_outbox_before_telegram_io(tmp_path):
    content = b"OggSsynthetic-opus"
    api = _TelegramAPI()
    bot = _bot(tmp_path, api)

    assert asyncio.run(bot.send_voice(42, content, "hello", format="ogg"))
    assert api.sent == [(42, content, "hello")]
    durable = list((tmp_path / "telegram_media").iterdir())
    assert len(durable) == 1
    assert durable[0].suffix == ".ogg"
    assert durable[0].read_bytes() == content


def test_voice_restart_and_ambiguous_timeout_are_at_most_once(tmp_path):
    from herandhim.core.telegram_transport import TelegramTransportStore

    api = _TelegramAPI(fail=True)
    bot = _bot(tmp_path, api)
    assert not asyncio.run(bot.send_voice(42, b"OggSclip", format="ogg"))

    restarted = _bot(tmp_path, api)
    restarted._transport = TelegramTransportStore(tmp_path / "telegram.sqlite3")
    assert not asyncio.run(restarted._deliver_pending_media("telegram:42"))
    assert len(api.sent) == 1


def test_voice_owner_gate_precedes_outbox(tmp_path):
    bot = _bot(tmp_path, _TelegramAPI(), allowed=(42,))

    with pytest.raises(PermissionError):
        asyncio.run(bot.send_voice(7, b"OggSclip", format="ogg"))
    assert bot._transport.pending_media_count("telegram:7") == 0


def test_voice_rejects_unknown_raw_format_before_outbox(tmp_path):
    bot = _bot(tmp_path, _TelegramAPI())

    with pytest.raises(ValueError, match="allowed format"):
        asyncio.run(bot.send_voice(42, b"WAV", format="wav"))
    assert bot._transport.pending_media_count("telegram:42") == 0


class _ReplyMessage:
    def __init__(self, voice):
        self.voice = voice
        self.audio = None
        self.replies: list[str] = []

    async def reply_text(self, text):
        self.replies.append(text)


def test_inbound_voice_metadata_gate_precedes_download(tmp_path):
    class _OversizeVoice:
        duration = 121
        file_size = 1
        mime_type = "audio/ogg"

        async def get_file(self):
            raise AssertionError("oversize voice must not reach Telegram file I/O")

    bot = _bot(tmp_path, _TelegramAPI())
    message = _ReplyMessage(_OversizeVoice())
    update = SimpleNamespace(message=message)

    assert asyncio.run(bot._transcribe_voice(update)) is None
    assert len(message.replies) == 1
    assert "shorter" in message.replies[0].lower()


def test_inbound_stt_error_is_sanitized_and_transcript_is_not_logged(
    tmp_path, monkeypatch, caplog,
):
    from herandhim.core import stt

    class _File:
        async def download_as_bytearray(self):
            return bytearray(b"OggSvoice")

    class _Voice:
        duration = 3
        file_size = 9
        mime_type = "audio/ogg"

        async def get_file(self):
            return _File()

    async def explode(*_args, **_kwargs):
        raise RuntimeError("raw-provider-secret-and-audio-text")

    monkeypatch.setattr(stt, "transcribe_bytes_async", explode)
    bot = _bot(tmp_path, _TelegramAPI())
    message = _ReplyMessage(_Voice())

    with caplog.at_level("WARNING"):
        assert asyncio.run(bot._transcribe_voice(SimpleNamespace(message=message))) is None

    assert "raw-provider-secret" not in caplog.text
    assert "raw-provider-secret" not in " ".join(message.replies)
    assert len(message.replies) == 1


def test_reply_in_kind_uses_durable_voice_contract(tmp_path, monkeypatch):
    from herandhim.core import tts

    async def synthesize(_text):
        return tts.VoiceClip(data=b"OggSreply", format="ogg")

    monkeypatch.setattr(tts, "synthesize_async", synthesize)
    monkeypatch.setattr(tts, "should_voice_reply", lambda _text: True)
    api = _TelegramAPI()
    bot = _bot(tmp_path, api)

    assert asyncio.run(bot._maybe_send_voice_reply(42, "Привет"))
    assert api.sent == [(42, b"OggSreply", None)]
    assert bot._transport.pending_media_count("telegram:42") == 0
