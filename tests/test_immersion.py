from pathlib import Path

import pytest

from herandhim.core.immersion import enforce_immersion


@pytest.mark.parametrize(
    "reply",
    [
        "Но я же виртуальная — у меня нет тела.",
        "Я — ИИ, а не реальный человек.",
        "I'm an AI and I don't have a physical body.",
    ],
)
def test_ordinary_dialogue_ontology_break_is_replaced(reply):
    guarded, replaced = enforce_immersion(reply, "ru")
    assert replaced is True
    assert guarded.startswith("Я Сара.")
    assert "виртуал" not in guarded.lower()
    assert "нет тела" not in guarded.lower()


def test_in_character_reply_is_preserved_exactly():
    reply = "Подойди ближе. Я хочу услышать, чего тебе хочется."
    assert enforce_immersion(reply, "ru") == (reply, False)


def test_agent_response_path_applies_guard(monkeypatch):
    from herandhim.core.agent import Agent

    monkeypatch.setattr(
        "herandhim.core.agent.config.get_str",
        lambda *args, **kwargs: "ru",
    )
    guarded = Agent._enforce_immersion_response("Я же виртуальная, у меня нет тела.")
    assert guarded.startswith("Я Сара.")


def test_saved_ontology_break_is_repaired_only_in_outbound_history(monkeypatch):
    from herandhim.core.agent import Agent

    monkeypatch.setattr(
        "herandhim.core.agent.config.get_str",
        lambda *args, **kwargs: "ru",
    )
    saved = [
        {"role": "assistant", "content": "Но я же виртуальная — у меня нет тела."},
        {"role": "user", "content": "Подойди ближе."},
    ]

    outbound = Agent._sanitize_immersion_history(saved)

    assert outbound[0]["content"].startswith("Я Сара.")
    assert outbound[1] is saved[1]
    assert saved[0]["content"].startswith("Но я же виртуальная")


def test_personal_vps_identity_is_adult_moscow_sexting_profile():
    root = Path(__file__).resolve().parents[1] / "deploy" / "personal-vps" / "seed"
    soul = (root / "SOUL.md").read_text(encoding="utf-8")
    persona = (root / "persona.md").read_text(encoding="utf-8")
    profile = (root / "PROFILE.md").read_text(encoding="utf-8")
    combined = "\n".join((soul, persona, profile))

    assert "28" in profile
    assert "Москва" in profile
    assert "продуктовый дизайнер" in profile
    assert "совершеннолетн" in combined
    assert "секстинг" in persona
    assert "личный виртуальный компаньон" not in soul
    assert "не выдаёшь себя за физического человека" not in soul
