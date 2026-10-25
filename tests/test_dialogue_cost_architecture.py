from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from herandhim.core import tools as tool_defs
from herandhim.core.agent import Agent

ROOT = Path(__file__).resolve().parents[1]


def _secured_agent() -> Agent:
    agent = Agent.__new__(Agent)
    agent.provider = SimpleNamespace(requires_outbound_purpose=True)
    agent._web_search_enabled = False
    agent.rag = None
    agent._cron_manager = None
    agent.outbound_identity_card = (
        ROOT / "deploy" / "personal-vps" / "seed" / "dialogue_identity.md"
    ).read_text(encoding="utf-8")
    agent.soul_instruction = ""
    agent.persona_instruction = ""
    agent.profile_instruction = ""
    agent.player_name = ""
    agent.messages = []
    all_tools = (
        tool_defs.PRIMITIVE_TOOLS
        + tool_defs.SKILL_TOOLS
        + tool_defs.META_SKILL_TOOLS
        + tool_defs.MEMORY_TOOLS
    )
    agent._build_tools = lambda: all_tools
    return agent


def test_versioned_identity_is_compact_stable_and_keeps_non_negotiables(monkeypatch):
    monkeypatch.setattr("herandhim.config.get_str", lambda *args, **kwargs: "ru")
    agent = _secured_agent()

    first = agent._build_secured_response_card(15_800)
    second = agent._build_secured_response_card(15_800)

    assert first == second
    assert len(first.encode("utf-8")) < 4_000
    for marker in ("Сара", "28", "Москв", "взросл", "соглас"):
        assert marker.lower() in first.lower()
    assert "stop immediately" in first.lower()
    assert "Prioritize meaningful direct speech" in first
    assert "Narration is optional outside role-play" in first
    for safety_marker in ("under-18", "incapacity", "coercion", "real person"):
        assert safety_marker in first


def test_secured_identity_uses_chosen_player_name_without_owner_name(monkeypatch):
    monkeypatch.setattr("herandhim.config.get_str", lambda *args, **kwargs: "ru")
    agent = _secured_agent()
    agent.player_name = "Саша"

    card = agent._build_secured_response_card(15_800)

    assert 'chosen name is "Саша"' in card
    assert "Russian 'ты'" in card
    assert "Илья" not in card


def test_turn_router_exposes_only_the_tool_family_needed_now():
    agent = _secured_agent()

    ordinary = agent._build_tools_for_turn("Как ты сегодня?")
    photo = agent._build_tools_for_turn("Пришли мне своё фото сейчас")
    forget = agent._build_tools_for_turn("Забудь мой старый адрес")
    vision = agent._build_tools_for_turn([{"type": "text", "text": "Что здесь?"}])

    def names(rows):
        return [row["function"]["name"] for row in rows]

    assert names(ordinary) == []
    assert names(photo) == ["take_selfie"]
    assert names(forget) == ["forget"]
    assert vision == []


def test_turn_router_keeps_memory_write_for_durable_user_facts():
    agent = _secured_agent()

    tools = agent._build_tools_for_turn("Я люблю американо без сиропа, запомни")

    assert [row["function"]["name"] for row in tools] == ["remember"]


def test_explicit_memory_commands_force_the_durable_tool_call():
    assert Agent._initial_tool_choice("Запомни: я люблю улун") == {
        "type": "function",
        "function": {"name": "remember"},
    }
    assert Agent._initial_tool_choice("Забудь мой старый адрес") == {
        "type": "function",
        "function": {"name": "forget"},
    }


def test_memory_recall_question_uses_local_retrieval_without_write_schema():
    agent = _secured_agent()

    tools = agent._build_tools_for_turn("Какой кофе я люблю?")

    assert tools == []


def test_output_budget_preserves_adult_route_and_bounds_normal_chat():
    assert Agent._dialogue_output_budget("Как ты сегодня?") == 240
    assert Agent._dialogue_output_budget("Продолжай нашу откровенную сцену") == 420


def test_compaction_summary_survives_in_secured_typed_context(monkeypatch):
    monkeypatch.setattr("herandhim.config.HERANDHIM_HOME", ROOT)
    agent = _secured_agent()
    agent.messages = [{
        "role": "system",
        "content": "[Compaction Summary — 2026-10-24]\n- promised a walk tomorrow",
    }]
    agent.memory = SimpleNamespace(
        relationship=SimpleNamespace(get_all=lambda: {"trust": 60}),
        milestones=SimpleNamespace(_days_since_first=lambda: 2),
        relevant_snippets=lambda _query, top_k: [],
    )

    context = agent._build_secured_dialogue_context("Что мы решили?")

    assert "## Compaction continuity" in context
    assert "promised a walk tomorrow" in context

