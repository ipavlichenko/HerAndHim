"""Installed-package acceptance with synthetic data and zero network calls."""
import asyncio
import json
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from herandhim.catalog.characters import CHARACTERS
from herandhim.catalog.runtime import CatalogRuntime
from herandhim.catalog.telegram import ADULT, CatalogTelegramBot, character_label
from herandhim.core.agent import Agent
from herandhim.core.llm.response import MockChoice, MockFunction, MockMessage, MockResponse, MockToolCall
from herandhim.core.roleplay import ROLEPLAY_SCENARIOS


class Provider:
    requires_outbound_purpose = True

    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        assert kwargs["purpose"] == "dialogue"
        proposal = {
            "state_delta": {"last_beat": "Ника предлагает прогулку", "consent_state": "check_in"},
            "beats": [{"type": "action", "text": "Ника берёт карту города."},
                      {"type": "speech", "text": "Пойдём к реке или заглянем в кафе?"}],
        }
        return MockResponse([MockChoice(MockMessage(None, [MockToolCall(
            "synthetic", MockFunction("propose_roleplay_turn", json.dumps(proposal, ensure_ascii=False)))]))])


class Message:
    web_app_data = None

    def __init__(self, text):
        self.text, self.replies = text, []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))

    async def reply_photo(self, photo, caption, **kwargs):
        self.replies.append((caption, {**kwargs, "photo": photo.read()}))


async def check(root):
    provider = Provider()
    settings = {"catalog": {"legacyOwnerId": 1}, "agent": {"language": "ru"}}
    runtime = CatalogRuntime(provider, root, settings)
    bot = CatalogTelegramBot(runtime, "synthetic", "", public=False, allowed_users=[1])
    counter = 0

    async def send(text):
        nonlocal counter
        counter += 1
        message = Message(text)
        update = SimpleNamespace(effective_message=message,
            effective_user=SimpleNamespace(id=1, is_bot=False),
            effective_chat=SimpleNamespace(id=1, type="private"), update_id=counter)
        await bot._handle_catalog_update(update, None)
        return message

    try:
        assert runtime.state.selected(1) is None
        assert "Как тебя называть" in (await send("/start")).replies[0][0]
        assert "18 лет" in (await send("Павел")).replies[0][0]
        catalog = await send(ADULT)
        assert len(catalog.replies[0][1]["reply_markup"].keyboard) == 10
        for companion in CHARACTERS:
            card = await send(character_label(companion))
            caption, payload = card.replies[0]
            assert companion.description in caption
            with Image.open(BytesIO(payload["photo"])) as image:
                assert image.format == "JPEG"
                image.verify()
        nika = next(c for c in CHARACTERS if c.id == "nika")
        await send(character_label(nika))
        before = runtime.state.selected(1)
        with patch.object(Agent, "_update_affect", lambda *_: None):
            reply = await send(ROLEPLAY_SCENARIOS[0]["label"])
        episode = runtime.state.selected(1)
        assert episode != before and runtime.scene_active(episode)
        assert "Ника берёт карту" in reply.replies[0][0]
        assert provider.calls == 1
        await send("/pause")
        assert runtime.state.active_users() == []
        await send("/hide")
        assert runtime.transport.keyboard_hidden("telegram:1")
        await send("/menu")
        assert not runtime.transport.keyboard_hidden("telegram:1")
        await send(character_label(CHARACTERS[-1]))
        assert not runtime.scene_active(runtime.state.selected(1))
        await send(character_label(nika))
        assert runtime.state.selected(1) == episode
        restarted = CatalogRuntime(provider, root, settings)
        assert restarted.state.selected(1) == episode
        assert restarted.state.name_confirmed(1)
        assert "Ника берёт карту" in restarted.last_reply(episode)
        restarted.close()
        assert provider.calls == 1
        print(json.dumps({"installed_smoke": "PASS", "characters": len(CHARACTERS), "portrait_cards": len(CHARACTERS),
                          "synthetic_completions": provider.calls, "live_calls": 0,
                          "telegram_messages": 0}))
    finally:
        runtime.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="aigirla-smoke-") as temporary:
        asyncio.run(check(Path(temporary)))
