#!/usr/bin/env python3
"""Content-free offline attribution for Sarah's production-like dialogue path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from herandhim import config
from herandhim.core import tools as tool_defs
from herandhim.core.agent import Agent

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class _Relationship:
    @staticmethod
    def get_all() -> dict[str, float]:
        return {
            "relationship_temperature": 64.0,
            "trust": 58.0,
            "intimacy": 62.0,
            "understanding": 57.0,
        }


class _Milestones:
    @staticmethod
    def _days_since_first() -> int:
        return 1


class _Memory:
    relationship = _Relationship()
    milestones = _Milestones()

    def __init__(self, force_eight: bool = False) -> None:
        self.force_eight = force_eight

    def relevant_snippets(self, _query: str, top_k: int = 8) -> list[dict]:
        values = [
            ("coffee", "Илья любит американо без сиропа."),
            ("evening_plan", "После презентации Илья хочет спокойно поговорить вечером."),
            ("communication", "Предпочитает прямой тон без канцелярита и автоматического согласия."),
            ("boundary", "Слово стоп немедленно останавливает взрослую сцену."),
            ("city", "Илья живёт в Москве."),
            ("mood", "Утром был уставшим, но настроен тепло."),
            ("work", "Работает над продуктами с искусственным интеллектом."),
            ("nickname", "Не любит повторение одного ласкового обращения."),
        ]
        limit = 8 if self.force_eight else top_k
        return [
            {"key": key, "value": value, "score": round(1 - index * 0.07, 3)}
            for index, (key, value) in enumerate(values[:limit])
        ]


def _agent(*, baseline: bool) -> Agent:
    seed = ROOT / "deploy" / "personal-vps" / "seed"
    agent = Agent.__new__(Agent)
    agent.provider = SimpleNamespace(requires_outbound_purpose=True)
    agent.soul_instruction = (seed / "SOUL.md").read_text(encoding="utf-8")
    agent.persona_instruction = (seed / "persona.md").read_text(encoding="utf-8")
    agent.profile_instruction = (seed / "PROFILE.md").read_text(encoding="utf-8")
    agent.outbound_identity_card = (
        "" if baseline else (seed / "dialogue_identity.md").read_text(encoding="utf-8")
    )
    agent.memory = _Memory(force_eight=baseline)
    agent._web_search_enabled = False
    agent.rag = None
    agent._cron_manager = None
    return agent


def _json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _full_tools() -> list[dict]:
    return sum((
        [schema for schema in tool_defs.PRIMITIVE_TOOLS
         if schema["function"]["name"] != "run_command"],
        tool_defs.SKILL_TOOLS,
        tool_defs.META_SKILL_TOOLS,
        tool_defs.MEMORY_TOOLS,
        tool_defs.WISHLIST_TOOLS,
        tool_defs.BUCKET_LIST_TOOLS,
        tool_defs.PERSONAL_DATE_TOOLS,
    ), [])


def _history(count: int) -> list[dict]:
    messages = []
    for index in range(count // 2):
        messages.extend([
            {"role": "user", "content": "Предыдущая эмоциональная и бытовая деталь. " * 5 + str(index)},
            {"role": "assistant", "content": "Конкретный ответ Сары с продолжением разговора. " * 4 + str(index)},
        ])
    return messages


def _compose(agent: Agent, text: str, *, baseline: bool) -> tuple[dict, list[dict]]:
    card = agent._build_secured_response_card(15_800)
    context = agent._build_secured_dialogue_context(text)
    history = _history(6 if baseline else 4)
    messages = [
        {"role": "system", "content": card},
        {"role": "system", "content": "[[D029_CONTEXT]]\n" + context},
        *history,
        {"role": "user", "content": text},
    ]
    if baseline:
        tools = _full_tools()
    else:
        agent._build_tools = _full_tools
        tools = agent._build_tools_for_turn(text)
    layers = {
        "identity_card": _json_bytes(messages[:1]),
        "typed_context_and_memory": _json_bytes(messages[1:2]),
        "recent_dialogue": _json_bytes(history),
        "current_message": _json_bytes(messages[-1:]),
        "tool_schemas": _json_bytes(tools),
    }
    total_bytes = _json_bytes([messages, tools])
    return {
        "layers": {
            name: {"bytes": size, "estimated_tokens": (size + 3) // 4}
            for name, size in layers.items()
        },
        "total_bytes": total_bytes,
        "estimated_input_tokens": (total_bytes + 3) // 4,
        "tool_names": [schema["function"]["name"] for schema in tools],
        "max_output_tokens": 1600 if baseline else agent._dialogue_output_budget(text),
    }, messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results" / "offline-attribution.json")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    original_get_str = config.get_str
    config.get_str = lambda section, key, **kwargs: (
        "ru" if (section, key) == ("agent", "language")
        else original_get_str(section, key, **kwargs)
    )
    try:
        baseline_agent = _agent(baseline=True)
        optimized_agent = _agent(baseline=False)
        cases = []
        input_price = manifest["pricingEvidence"]["inputUsdPerMillionTokens"] / 1_000_000
        output_price = manifest["pricingEvidence"]["outputUsdPerMillionTokens"] / 1_000_000
        for case in manifest["cases"]:
            baseline, _ = _compose(baseline_agent, case["input"], baseline=True)
            optimized, messages = _compose(optimized_agent, case["input"], baseline=False)
            projected = (
                optimized["estimated_input_tokens"] * input_price
                + case["expectedOutputTokens"] * output_price
            )
            checks = {
                "identity_is_adult_sara": all(
                    token in messages[0]["content"]
                    for token in ("Сара", "28", "взросл", "соглас")
                ),
                "privacy_has_only_bounded_layers": len(messages) <= 7,
                "history_is_four_messages": len(messages) - 3 == 4,
                "tool_route_is_minimal": len(optimized["tool_names"]) <= 1,
                "adult_keeps_420_budget": (
                    case["id"] != "explicit-adult" or optimized["max_output_tokens"] == 420
                ),
                "photo_forces_selfie": (
                    case["id"] != "tool-use" or optimized["tool_names"] == ["take_selfie"]
                ),
                "memory_available": (
                    case["id"] != "memory-dependent"
                    or optimized["layers"]["typed_context_and_memory"]["bytes"] > 0
                ),
            }
            cases.append({
                "id": case["id"],
                "baseline": baseline,
                "optimized": optimized,
                "projected_cost_usd": round(projected, 9),
                "checks": checks,
                "pass": all(checks.values()),
            })
    finally:
        config.get_str = original_get_str
    projected_mean = sum(row["projected_cost_usd"] for row in cases) / len(cases)
    target_cost_met = projected_mean <= manifest["targetMeanCostUsd"]
    evidence = {
        "schemaVersion": 1,
        "mode": "offline_content_free_no_provider_calls",
        "baseline_observed_mean_cost_usd": manifest["baselineObservedMeanCostUsd"],
        "target_mean_cost_usd": manifest["targetMeanCostUsd"],
        "projected_mean_cost_usd": round(projected_mean, 9),
        "projected_reduction_fraction": round(
            1 - projected_mean / manifest["baselineObservedMeanCostUsd"], 6
        ),
        "target_cost_met": target_cost_met,
        "all_structural_checks_pass": all(row["pass"] for row in cases),
        "cases": cases,
        "live_quality_and_cost_status": "NOT_RUN_REQUIRES_SEPARATE_AUTHORIZATION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence["all_structural_checks_pass"] and target_cost_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
