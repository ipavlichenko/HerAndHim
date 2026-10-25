"""Conversation boundary for the catalogue, with a read-only legacy mapping."""
import asyncio
import json
import re
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

from .. import config
from ..core.persistent_agent import PersistentAgent
from ..core.roleplay import RoleplaySceneStore
from ..core.session_store import SessionStore
from ..core.telegram_transport import TelegramTransportStore
from .characters import Character, character
from .state import CatalogState, Episode, user_key


def named(value, companion: Character):
    """Adapt display values only; retain the versioned scene schema keys."""
    if isinstance(value, str):
        name = companion.name
        stem = name[:-1]
        gen = stem + ("и" if name.endswith("я") or stem.endswith(("г", "к", "х", "ж", "ч", "ш", "щ")) else "ы")
        dat = name[:-1] + ("и" if name.endswith("ия") else "е")
        forms = {"Sarah": name, "Сара": name, "Сары": gen, "Саре": dat,
                 "Сару": stem + ("ю" if name.endswith("я") else "у"),
                 "Сарой": stem + ("ей" if name.endswith("я") else "ой")}
        return re.sub(r"\b(?:Sarah|Сара|Сары|Саре|Сару|Сарой)\b", lambda m: forms[m.group()], value)
    if isinstance(value, list):
        return [named(item, companion) for item in value]
    if isinstance(value, dict):
        return {key: item if key == "player_name" else named(item, companion) for key, item in value.items()}
    return value


def named_prompt(text: str, companion: Character, player_name: str):
    clause = "The player's chosen name is " + json.dumps(player_name, ensure_ascii=False) + "."
    return named(text.replace(clause, "\0PLAYER_IDENTITY\0"), companion).replace("\0PLAYER_IDENTITY\0", clause)


class CharacterSceneStore(RoleplaySceneStore):
    def __init__(self, session_id: str, companion: Character):
        super().__init__(session_id)
        self.companion = companion

    def initial_state(self, scene_id, player_name=""):
        return named(super().initial_state(scene_id, player_name), self.companion)

    def prompt_block(self):
        text = super().prompt_block()
        state = self.load()
        if not state:
            return text
        # Durable facts belong to the episode; only template directions are
        # renamed. A player called Sarah must keep their own chosen name.
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        protected = text.replace(state_json, "\0EPISODE_STATE\0", 1)
        return named_prompt(protected, self.companion, state.get("player_name", "")).replace("\0EPISODE_STATE\0", state_json)


class CatalogAgent(PersistentAgent):
    """Text-only adapter. It cannot execute upstream shell/skills/media tools."""
    def __init__(self, *, companion: Character, **kwargs):
        self.companion = companion
        super().__init__(**kwargs)
        self.roleplay = CharacterSceneStore(self.session_id, companion)
        self._needs_onboarding = False

    def _initial_tool_choice(self, user_input):
        choice = super()._initial_tool_choice(user_input)
        if isinstance(choice, dict) and choice["function"]["name"] not in {"remember", "forget"}:
            return "auto"
        return choice

    def _build_tools_for_turn(self, user_input):
        return [named(schema, self.companion) for schema in super()._build_tools_for_turn(user_input)
                if schema.get("function", {}).get("name") in {"propose_roleplay_turn", "remember", "forget"}]

    def _roleplay_turn_tool(self):
        return named(super()._roleplay_turn_tool(), self.companion)

    def _execute_tool_call(self, tool_call):
        if tool_call.function.name not in {"remember", "forget"}:
            return "Tool unavailable in the text pilot."
        return super()._execute_tool_call(tool_call)

    def _build_secured_response_card(self, max_bytes):
        return named_prompt(super()._build_secured_response_card(max_bytes), self.companion, self.player_name)


class CatalogRuntime:
    def __init__(self, provider, root: Path | None = None, settings: dict | None = None,
                 agent_factory=CatalogAgent):
        if not getattr(provider, "requires_outbound_purpose", False):
            raise ValueError("Catalog requires the secured provider and shared budget guard")
        self.root = Path(root or config.home())
        self.settings = deepcopy(settings if settings is not None else config.as_dict())
        options = self.settings.get("catalog", {})
        owner = options.get("legacyOwnerId")
        if owner is None:
            raise ValueError("Explicit catalog.legacyOwnerId is required for existing data migration")
        self.state = CatalogState(self.root, legacy_owner=owner, max_users=options.get("maxUsers", 50))
        self.transport = TelegramTransportStore(self.root / "context/state/telegram_transport.sqlite3")
        self.provider = provider
        self.agent_factory = agent_factory
        self._locks: dict[int, asyncio.Lock] = {}
        self._capacity = asyncio.Semaphore(4)
        self._agents: dict[int, tuple[Episode, CatalogAgent]] = {}

    @asynccontextmanager
    async def acquire(self, user_id: int):
        user_key(user_id)
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            async with self._capacity:
                yield

    def scope(self, episode: Episode):
        settings = deepcopy(self.settings)
        settings["isolation"] = {"perGroup": True}
        settings["heartbeat"] = {"enabled": False}
        settings["tavily"] = {}
        # Never mutate global env or write a copy containing provider credentials.
        return config.runtime_scope(episode.data_home(self.root), settings)

    def agent(self, episode: Episode):
        cached = self._agents.get(episode.user_id)
        if cached and cached[0] == episode:
            return cached[1]
        if cached:
            executor = getattr(cached[1], "_bg_executor", None)
            if executor:
                executor.shutdown(wait=True)
        companion = character(episode.character_id)
        with self.scope(episode):
            home = config.home()
            if not episode.legacy:
                # Empty context prevents upstream auto-init from installing owner
                # or generic companion templates into a new player's episode.
                outbound = home / "context/outbound"
                outbound.mkdir(parents=True, exist_ok=True)
                card = outbound / "dialogue_identity.md"
                if not card.exists():
                    card.write_text(companion.identity(), encoding="utf-8")
            agent = self.agent_factory(provider=self.provider, store=SessionStore(),
                                       session_id=episode.session_id, companion=companion,
                                       verbose=False)
            agent.set_player_name(self.transport.player_name(user_key(episode.user_id)))
        self._agents[episode.user_id] = (episode, agent)
        return agent

    def select(self, user_id: int, character_id: str, *, new_episode=False):
        previous = self.state.selected(user_id)
        selected = self.state.select(user_id, character_id, new_episode=new_episode)
        if previous and previous != selected:
            self.transport.discard_pending(previous.session_id)
        return selected

    def scene_active(self, episode: Episode):
        with self.scope(episode):
            return RoleplaySceneStore(episode.session_id).is_active()

    def last_reply(self, episode: Episode):
        with self.scope(episode):
            messages = SessionStore().load(episode.session_id)
        return next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "assistant" and m.get("content")), "")

    async def chat(self, episode: Episode, text: str, *, proactive=False):
        # to_thread copies ContextVars; the old run_in_executor call did not.
        with self.scope(episode):
            agent = self.agent(episode)
            agent.set_player_name(self.transport.player_name(user_key(episode.user_id)))
            method = agent.chat_proactive if proactive else agent.chat
            task = asyncio.create_task(asyncio.to_thread(method, text))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Cancellation does not stop a Python worker. Keep the caller's
                # user lock until its state writes finish, then propagate it.
                await task
                raise

    def close(self):
        for _, agent in self._agents.values():
            executor = getattr(agent, "_bg_executor", None)
            if executor:
                executor.shutdown(wait=True)
