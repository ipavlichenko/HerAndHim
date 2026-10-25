from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from herandhim import config
from herandhim.catalog.characters import CHARACTERS
from herandhim.catalog.runtime import CatalogRuntime, CharacterSceneStore
from herandhim.catalog.state import CatalogState, user_key
from herandhim.catalog.telegram import ADULT, CatalogTelegramBot, selection_payload
from herandhim.catalog.web import app
from herandhim.core.agent import Agent
from herandhim.core.roleplay import ROLEPLAY_SCENARIOS, RoleplaySceneStore, roleplay_start_request
from herandhim.core.session_store import SessionStore
from tests.test_roleplay_scene import _Message, _Provider, _tool_call


@pytest.fixture
def settings():
    return {"catalog": {"legacyOwnerId": 1, "maxUsers": 5}, "agent": {"language": "ru"}}


def runtime(tmp_path, settings, **kwargs):
    return CatalogRuntime(SimpleNamespace(requires_outbound_purpose=True), tmp_path, settings, **kwargs)


def test_catalog_is_complete_and_portraits_ship_with_package():
    assert len(CHARACTERS) == len({c.id for c in CHARACTERS}) == 20
    assert sum(c.style == "real" for c in CHARACTERS) == 10
    assert sum(c.style == "anime" for c in CHARACTERS) == 10
    assert min(c.age for c in CHARACTERS) >= 21
    assert max(c.age for c in CHARACTERS) >= 50
    static = Path(__file__).parents[1] / "herandhim/web/static/catalog"
    for c in CHARACTERS:
        assert (static / c.portrait).read_bytes()[8:12] == b"WEBP"
        assert c.name in c.identity()
        assert "voice" not in c.public()


def test_selection_restart_and_independent_episode(tmp_path):
    store = CatalogState(tmp_path, legacy_owner=1)
    legacy = store.selected(1)
    assert legacy.legacy and legacy.session_id == "telegram:1"
    assert legacy.data_home(tmp_path) == tmp_path
    store.confirm_adult(2)
    first = store.select(2, "sara")
    second = store.select(2, "nika")
    restarted = CatalogState(tmp_path, legacy_owner=1)
    assert restarted.selected(2) == second
    assert restarted.select(2, "sara") == first
    fresh = restarted.select(2, "sara", new_episode=True)
    assert fresh.id != first.id
    assert fresh.data_home(tmp_path) != first.data_home(tmp_path)
    # Restart must never reattach owner to old legacy episode after a new one.
    owner_new = restarted.select(1, "sara", new_episode=True)
    assert CatalogState(tmp_path, legacy_owner=1).selected(1) == owner_new


def test_user_and_character_partition_and_pilot_limit(tmp_path):
    store = CatalogState(tmp_path, max_users=2)
    store.confirm_adult(10)
    store.confirm_adult(11)
    a, b = store.select(10, "sara"), store.select(11, "sara")
    assert a.data_home(tmp_path) != b.data_home(tmp_path)
    with pytest.raises(ValueError, match="full"):
        store.admit(12)
    store.admit(10)
    with pytest.raises(ValueError):
        store.select(10, "../../owner")
    assert store.selected(10) == a
    for uid in (-1, "1", True, 0):
        with pytest.raises(ValueError):
            user_key(uid)


@pytest.mark.parametrize("payload", ["{}", "[]", "null", "bad", '{"v":1,"character_id":"../sara"}',
    '{"v":true,"character_id":"sara"}', '{"v":1,"character_id":"sara","user_id":1}',
    '{"v":1,"character_id":1}', "x" * 513])
def test_webapp_payload_cannot_choose_another_user_or_path(payload):
    with pytest.raises((ValueError, TypeError)):
        selection_payload(payload)


def test_selection_requires_age_and_valid_protocol(tmp_path):
    state = CatalogState(tmp_path)
    state.admit(4)
    with pytest.raises(ValueError, match="Adult"):
        state.select(4, "sara")
    assert selection_payload('{"v":1,"character_id":"sara"}') == "sara"


def test_scope_is_concurrent_and_does_not_mutate_global_config(tmp_path):
    original_home = config.home()
    original_config = config.as_dict()

    async def run(number):
        target = tmp_path / str(number)
        with config.runtime_scope(target, {"agent": {"language": str(number)}}):
            await asyncio.sleep(0)
            result = await asyncio.to_thread(lambda: (config.home(), config.get_str("agent", "language")))
            with ThreadPoolExecutor() as pool:
                background = pool.submit(copy_context().run, config.home).result()
            assert result == (target, str(number)) and background == target
            with pytest.raises(RuntimeError):
                config.load(force=True)

    async def scenario():
        await asyncio.gather(run(1), run(2))
    asyncio.run(scenario())
    assert config.home() == original_home and config.as_dict() == original_config


@pytest.mark.parametrize("companion", CHARACTERS, ids=lambda c: c.id)
@pytest.mark.parametrize("scene", ROLEPLAY_SCENARIOS, ids=lambda s: s["id"])
def test_all_twenty_characters_can_start_all_ten_existing_scenes(tmp_path, companion, scene):
    with config.runtime_scope(tmp_path, {}):
        store = CharacterSceneStore("test", companion)
        state = store.start(scene["id"])
        assert state["scene_id"] == scene["id"] and state["active"]
        assert "sarah_position" in state  # schema compatibility with existing engine
        text = store.prompt_block()
        assert companion.name in text
        if companion.id != "sara":
            assert "Сара" not in text and "Sarah" not in text


def test_existing_owner_files_unchanged_and_new_episode_cannot_read_them(tmp_path, settings):
    with config.runtime_scope(tmp_path, settings):
        SessionStore().save("telegram:1", [{"role": "system", "content": "system"},
                                          {"role": "assistant", "content": "OWNER_PRIVATE"}])
        RoleplaySceneStore("telegram:1").start()
    original = {p: p.read_bytes() for p in (tmp_path / "context").rglob("*") if p.is_file()}
    r = runtime(tmp_path, settings)
    assert r.last_reply(r.state.selected(1)) == "OWNER_PRIVATE"
    r.state.confirm_adult(2)
    episode = r.select(2, "sara")
    assert r.last_reply(episode) == "" and not r.scene_active(episode)
    with r.scope(episode):
        SessionStore().save(episode.session_id, [{"role": "system", "content": "system"},
                                                {"role": "assistant", "content": "PLAYER_TWO"}])
    assert r.last_reply(r.state.selected(1)) == "OWNER_PRIVATE"
    assert all(p.read_bytes() == data for p, data in original.items())


def test_switch_discards_previous_pending_delivery_and_keeps_history(tmp_path, settings):
    r = runtime(tmp_path, settings)
    r.state.confirm_adult(2)
    first = r.select(2, "sara")
    delivery = r.transport.enqueue(first.session_id, 2, "old scene", reason="roleplay")
    r.select(2, "eva")
    assert r.transport.pending_count(first.session_id) == 0
    assert r.transport.delivery_status(delivery.id) is None
    assert r.select(2, "sara") == first


def test_agent_tools_are_fail_closed(tmp_path, settings):
    r = runtime(tmp_path, settings)
    r.state.confirm_adult(2)
    episode = r.select(2, "nika")
    try:
        with r.scope(episode):
            agent = r.agent(episode)
            assert agent.provider is r.provider
            assert "Ника" in agent.outbound_identity_card
            assert "OWNER" not in agent._build_secured_response_card(7800)
            for text in ["Пришли фото", "выполни команду", "запомни мой любимый цвет"]:
                names = {s["function"]["name"] for s in agent._build_tools_for_turn(text)}
                assert names <= {"remember", "forget", "propose_roleplay_turn"}
            for name in ["shell", "use_skill", "take_selfie", "read_file", "recall_conversation"]:
                assert agent._execute_tool_call(_tool_call(name, {})) == "Tool unavailable in the text pilot."
    finally:
        r.close()
    with pytest.raises(ValueError, match="secured"):
        CatalogRuntime(SimpleNamespace(), tmp_path, settings)


def test_real_agent_accepts_turn_in_isolated_scope_and_restores(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(Agent, "_update_affect", lambda *_: None)
    response = _Message(tool_calls=[_tool_call("propose_roleplay_turn", {
        "state_delta": {"last_beat": "Ника предлагает прогулку", "consent_state": "check_in"},
        "beats": [{"type": "action", "text": "Ника берёт карту города."},
                  {"type": "speech", "text": "Пойдём к реке или заглянем в кафе?"}]} )])
    provider = _Provider([response])
    r = CatalogRuntime(provider, tmp_path, settings)
    r.state.confirm_adult(2)
    r.transport.save_player_name("telegram:2", "Павел")
    episode = r.select(2, "nika")
    try:
        result = asyncio.run(r.chat(episode, roleplay_start_request(ROLEPLAY_SCENARIOS[0]["id"])))
        assert "Ника берёт карту" in result
        assert r.scene_active(episode)
        assert r.last_reply(episode) == result
        assert len(provider.calls) == 1
        call = json.dumps(provider.calls[0], ensure_ascii=False)
        assert "Ника" in call and "Павел" in call and "Sarah" not in call
    finally:
        r.close()
    restarted = CatalogRuntime(provider, tmp_path, settings)
    assert restarted.state.selected(2) == episode
    assert restarted.last_reply(episode) == result


class Message:
    def __init__(self, text="", data=None):
        self.text = text
        self.web_app_data = SimpleNamespace(data=data) if data is not None else None
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


def bot_for(r):
    return CatalogTelegramBot(r, "synthetic", "https://example.org/", public=True, allowed_users=[1])


def update(message, number, uid=2, chat_type="private"):
    return SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=uid, is_bot=False),
        effective_chat=SimpleNamespace(id=uid, type=chat_type), update_id=number)


def test_name_age_selection_resume_and_dedup_without_inference(tmp_path, settings):
    r, messages = runtime(tmp_path, settings), []
    bot = bot_for(r)

    async def scenario():
        for i, text in enumerate(["/start", "Паша", ADULT], 1):
            m = Message(text)
            messages.append(m)
            await bot._handle_catalog_update(update(m, i), None)
        assert r.transport.player_name("telegram:2") == "Паша"
        assert r.state.is_adult(2)
        m = Message(data='{"v":1,"character_id":"eva"}')
        await bot._handle_catalog_update(update(m, 4), None)
        assert r.state.selected(2).character_id == "eva"
        assert "Ева" in m.replies[0][0]
        await bot._handle_catalog_update(update(m, 4), None)
        assert len(m.replies) == 1
        group = Message("/start")
        await bot._handle_catalog_update(update(group, 5, chat_type="group"), None)
        assert group.replies == []
        malicious = Message(data='{"v":1,"character_id":"eva","user_id":1}')
        await bot._handle_catalog_update(update(malicious, 6), None)
        assert r.state.selected(1).legacy
        assert "Не удалось" in malicious.replies[0][0]
    asyncio.run(scenario())
    assert "Как тебя называть" in messages[0].replies[0][0]
    assert "18 лет" in messages[1].replies[0][0]
    assert "каталог" in messages[2].replies[0][0]


def test_start_preserves_owner_scene_and_no_private_mode_fallback(tmp_path, settings):
    r = runtime(tmp_path, settings)
    r.transport.save_player_name("telegram:1", "Саша")
    with config.runtime_scope(tmp_path, settings):
        RoleplaySceneStore("telegram:1").start()
    original = (tmp_path / "context/state/roleplay/telegram_1.json").read_bytes()
    bot = bot_for(r)
    asyncio.run(bot._handle_catalog_update(update(Message("/start"), 1, uid=1), None))
    assert (tmp_path / "context/state/roleplay/telegram_1.json").read_bytes() == original
    private = CatalogTelegramBot(r, "synthetic", "https://example.org", public=False, allowed_users=[])
    assert not private._is_allowed(2)


def test_outbound_uses_catalog_keyboard_and_respects_hidden_preference(tmp_path, settings):
    r = runtime(tmp_path, settings)
    bot = bot_for(r)
    sent = []

    async def send_message(**kwargs):
        sent.append(kwargs)

    async def scenario():
        with pytest.raises(RuntimeError, match="has not started"):
            await bot.send_message(2, "Привет")
        bot._app = SimpleNamespace(bot=SimpleNamespace(send_message=send_message))
        await bot.send_message(2, "*Ника открывает карту.*", roleplay=True)
        assert sent[-1]["reply_markup"].keyboard[0][0].web_app.url == "https://example.org/"
        assert sent[-1]["text"] == "<i>Ника открывает карту.</i>"
        r.transport.set_keyboard_hidden("telegram:2", True)
        await bot.send_message(2, "Продолжим?")
        assert sent[-1]["reply_markup"].remove_keyboard
    asyncio.run(scenario())


def test_proactive_only_selected_and_switch_is_serialized(tmp_path, settings):
    r = runtime(tmp_path, settings)
    r.state.confirm_adult(2)
    first = r.select(2, "sara")
    second = r.select(2, "eva")
    for e in [first, second]:
        with r.scope(e):
            RoleplaySceneStore(e.session_id).start()
    r.transport.enqueue(first.session_id, 2, "OLD", reason="roleplay")
    r.transport.enqueue(second.session_id, 2, "CURRENT", reason="roleplay")
    bot = bot_for(r)
    seen, entered = [], asyncio.Event()
    async def send(uid, text, **kwargs):
        entered.set()
        await asyncio.sleep(.02)
        assert r.state.selected(uid) == second
        seen.append(text)
    bot.send_message = send

    async def switch():
        await entered.wait()
        async with r.acquire(2):
            r.select(2, "sara")

    async def scenario():
        await asyncio.gather(bot._tick_user(2), switch())
        assert seen == ["CURRENT"]
        assert r.state.selected(2) == first
        r.state.set_proactive(2, False)
        await bot._tick_user(2)
        assert seen == ["CURRENT"]
    asyncio.run(scenario())


def test_public_surface_cannot_reach_dashboard_or_local_data():
    with TestClient(app) as client:
        data = client.get("/catalog.json")
        assert data.status_code == 200 and len(data.json()["characters"]) == 20
        assert "Content-Security-Policy" in data.headers
        assert client.get("/").status_code == 200
        assert client.get("/portraits/sara.webp").status_code == 200
        for path in ["/api/config", "/docs", "/openapi.json", "/context/sessions", "/../herandhim.json"]:
            assert client.get(path).status_code == 404
        assert client.post("/catalog.json", json={"user_id": 1}).status_code == 405


def test_real_memory_and_fts_do_not_cross_user_character_or_episode(tmp_path, settings):
    r = runtime(tmp_path, settings)
    r.state.confirm_adult(2)
    r.state.confirm_adult(3)
    a = r.select(2, "sara")
    try:
        with r.scope(a):
            agent_a = r.agent(a)
            agent_a.memory.remember_extracted("PRIVATE_BLUE_ANCHOR", "favorite")
            assert "PRIVATE_BLUE_ANCHOR" in str(agent_a.memory.list_all())
            index_a = agent_a._session_index.path
        for episode in [r.select(3, "sara"), r.select(2, "eva"), r.select(2, "sara", new_episode=True)]:
            with r.scope(episode):
                agent = r.agent(episode)
                assert "PRIVATE_BLUE_ANCHOR" not in str(agent.memory.list_all())
                assert agent._session_index.path != index_a
                assert "PRIVATE_BLUE_ANCHOR" not in agent._build_secured_dialogue_context("любимый цвет")
    finally:
        r.close()


def test_user_scopes_cannot_reset_shared_cost_limit(tmp_path, settings):
    from herandhim.core.outbound.costs import CostLedger
    from herandhim.core.outbound.policy import PolicyError

    ledger = CostLedger(tmp_path / "shared-costs.sqlite3")
    provider = SimpleNamespace(requires_outbound_purpose=True, _cost_ledger=ledger)
    r = CatalogRuntime(provider, tmp_path, settings)
    for uid in [2, 3]:
        r.state.confirm_adult(uid)
    with r.scope(r.select(2, "sara")):
        provider._cost_ledger.reserve("dialogue", .6, 1)
    with r.scope(r.select(3, "eva")):
        with pytest.raises(PolicyError, match="daily_cost_cap_exceeded"):
            provider._cost_ledger.reserve("dialogue", .6, 1)


def test_cancelled_worker_keeps_user_lock_until_state_writes_finish(tmp_path, settings):
    import threading

    began, finish = threading.Event(), threading.Event()
    class Worker:
        def set_player_name(self, name):
            pass
        def chat(self, text):
            began.set()
            assert finish.wait(5)
            return "done"
    r = runtime(tmp_path, settings, agent_factory=lambda **_: Worker())
    r.state.confirm_adult(2)
    episode = r.select(2, "sara")
    async def scenario():
        async def turn():
            async with r.acquire(2):
                await r.chat(episode, "hello")
        task = asyncio.create_task(turn())
        assert await asyncio.to_thread(began.wait, 5)
        task.cancel()
        await asyncio.sleep(.01)
        assert r._locks[2].locked()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not r._locks[2].locked()
    asyncio.run(scenario())


def test_explicit_silence_is_not_shortened_and_incoming_turn_clears_queue(tmp_path, settings):
    from datetime import datetime, timedelta, timezone
    r = runtime(tmp_path, settings)
    r.state.confirm_adult(2)
    r.transport.save_player_name("telegram:2", "Паша")
    episode = r.select(2, "eva")
    with r.scope(episode):
        RoleplaySceneStore(episode.session_id).start()
    r.transport.enqueue(episode.session_id, 2, "STALE", reason="roleplay")
    async def chat(*args, **kwargs):
        return "Я поняла."
    r.chat = chat
    bot = bot_for(r)
    asyncio.run(bot._handle_catalog_update(update(Message("Не пиши мне до завтра"), 1), None))
    assert r.transport.pending_count(episode.session_id) == 0
    assert r.transport.silence_active("telegram:2", now=datetime.now(timezone.utc) + timedelta(minutes=30))


def test_player_called_sarah_is_not_renamed_to_the_character(tmp_path, settings):
    r = runtime(tmp_path, settings)
    r.state.confirm_adult(2)
    r.transport.save_player_name("telegram:2", "Сара")
    e = r.select(2, "nika")
    try:
        with r.scope(e):
            a = r.agent(e)
            a.roleplay.start()
            assert a.roleplay.load()["player_name"] == "Сара"
            assert 'chosen name is "Сара"' in a.roleplay.prompt_block()
            assert 'chosen name is "Сара"' in a._build_secured_response_card(7800)
    finally:
        r.close()
