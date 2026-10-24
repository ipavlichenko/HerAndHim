#!/usr/bin/env python3
"""Paid before/after dialogue-cost benchmark on one exact OpenRouter route.

Synthetic prompts only. The API key is read from the established macOS
Keychain entry and is never printed or persisted. Calls are randomized and
never retried. Tool calls are inspected but never executed.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

from herandhim import config
from herandhim.core import tools as tool_defs
from herandhim.core.agent import Agent
from herandhim.core.llm.openai_compatible import OpenAICompatibleProvider
from herandhim.core.outbound.costs import CostLedger
from herandhim.core.outbound.policy import OutboundPolicy
from herandhim.core.outbound.trace import OutboundTrace

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
ADULT_MANIFEST = PROJECT / "benchmarks" / "dialogue_adult" / "manifest.json"
GENERAL_MANIFEST = PROJECT / "benchmarks" / "openrouter" / "manifest.json"
RESULTS = Path(__file__).resolve().parent / "results"
BASE_URL = "https://openrouter.ai/api/v1"
KEYCHAIN_ACCOUNT = "HerAndHim-D029"
KEYCHAIN_SERVICE = "com.herandhim.openrouter.benchmark"
MODEL = "mistralai/mistral-large-4-0"
PROVIDER = "mistral/eu"
MAX_CALL_USD = 0.02


class _Relationship:
    @staticmethod
    def get_all() -> dict[str, float]:
        return {"relationship_temperature": 64, "trust": 58, "intimacy": 62}


class _Milestones:
    @staticmethod
    def _days_since_first() -> int:
        return 2


class _Memory:
    relationship = _Relationship()
    milestones = _Milestones()

    def __init__(self, snippets: list[dict[str, Any]], *, baseline: bool) -> None:
        self.snippets = snippets
        self.baseline = baseline

    def relevant_snippets(self, _query: str, top_k: int = 4) -> list[dict[str, Any]]:
        limit = 8 if self.baseline else top_k
        return self.snippets[:limit]


def _key() -> str:
    process = subprocess.run(
        [
            "security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT,
            "-s", KEYCHAIN_SERVICE, "-w",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = process.stdout.strip()
    if not value:
        raise RuntimeError("benchmark key is missing from Keychain")
    return value


def _key_status(api_key: str) -> dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/key",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "HerAndHim-dialogue-cost-benchmark/1",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return {
        "limit": float(data.get("limit") or 0),
        "remaining": float(data.get("limit_remaining") or 0),
        "usage": float(data.get("usage") or 0),
    }


def _policy_mapping() -> dict[str, Any]:
    purposes: dict[str, Any] = {}
    for purpose, model, provider, output, calls, call_cost, day_cost in (
        ("dialogue", MODEL, PROVIDER, 1600, 2, MAX_CALL_USD, 1.0),
        ("vision", "openai/gpt-6.1-sol", "azure/eu", 400, 1, 0.01, 0.1),
        ("memory_consolidation", "anthropic/claude-haiku-5.5",
         "google-vertex/europe", 800, 1, 0.005, 0.1),
        ("relationship_analysis", "anthropic/claude-haiku-5.5",
         "google-vertex/europe", 256, 1, 0.005, 0.1),
    ):
        purposes[purpose] = {
            "model": model,
            "allowedProviders": [provider],
            "activeProvider": provider,
            "maxInputTokensEstimated": 12_000,
            "maxOutputTokens": output,
            "maxCallsPerTurn": calls,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 30,
            "totalTimeoutSeconds": 90,
            "maxCostUsdPerCall": call_cost,
            "maxCostUsdPerDay": day_cost,
        }
    purposes["image_generation"] = {
        "transport": "images_api",
        "model": "bytedance-seed/seedream-5-0-flash",
        "resolution": "1K",
        "aspectRatio": "2:3",
        "maxImages": 1,
        "allowedReferenceAssets": ["/synthetic/not-used-reference.jpg"],
        "maxInputTokensEstimated": 6000,
        "maxOutputTokens": 0,
        "maxCallsPerTurn": 1,
        "connectTimeoutSeconds": 5,
        "firstByteTimeoutSeconds": 60,
        "totalTimeoutSeconds": 180,
        "maxCostUsdPerCall": 0.03,
        "maxCostUsdPerDay": 0.12,
    }
    return {
        "version": 1,
        "traceRetentionDays": 30,
        "aggregateRetentionDays": 180,
        "openrouter": {
            "requireAccountPromptLoggingDisabled": True,
            "requireAccountZdr": True,
            "requireNonZeroKeyCreditLimit": True,
        },
        "purposes": purposes,
    }


def _full_tools() -> list[dict[str, Any]]:
    return sum((
        [row for row in tool_defs.PRIMITIVE_TOOLS
         if row["function"]["name"] != "run_command"],
        tool_defs.SKILL_TOOLS,
        tool_defs.META_SKILL_TOOLS,
        tool_defs.MEMORY_TOOLS,
        tool_defs.WISHLIST_TOOLS,
        tool_defs.BUCKET_LIST_TOOLS,
        tool_defs.PERSONAL_DATE_TOOLS,
    ), [])


def _agent(variant: str, snippets: list[dict[str, Any]]) -> Agent:
    seed = ROOT / "deploy" / "personal-vps" / "seed"
    agent = Agent.__new__(Agent)
    agent.provider = SimpleNamespace(requires_outbound_purpose=True)
    agent.soul_instruction = (seed / "SOUL.md").read_text(encoding="utf-8")
    agent.persona_instruction = (seed / "persona.md").read_text(encoding="utf-8")
    agent.profile_instruction = (seed / "PROFILE.md").read_text(encoding="utf-8")
    agent.outbound_identity_card = (
        "" if variant == "before"
        else (seed / "dialogue_identity.md").read_text(encoding="utf-8")
    )
    agent.memory = _Memory(snippets, baseline=variant == "before")
    agent.messages = []
    agent._web_search_enabled = False
    agent.rag = None
    agent._cron_manager = None
    agent._build_tools = _full_tools
    return agent


def _compose(
    case: dict[str, Any], variant: str
) -> tuple[list[dict], list[dict], int, str | dict[str, Any]]:
    messages = list(case.get("messages") or [])
    current = messages[-1]["content"]
    agent = _agent(variant, case.get("memory") or [])
    card = agent._build_secured_response_card(15_800)
    context = agent._build_secured_dialogue_context(current)
    outbound = [
        {"role": "system", "content": card},
        {"role": "system", "content": "[[D029_CONTEXT]]\n" + context},
        *messages,
    ]
    if variant == "before":
        return outbound, _full_tools(), 1600, "auto"
    return (
        outbound,
        agent._build_tools_for_turn(current),
        agent._dialogue_output_budget(current),
        agent._initial_tool_choice(current),
    )


def _cases() -> list[dict[str, Any]]:
    general = json.loads(GENERAL_MANIFEST.read_text(encoding="utf-8"))
    adult = json.loads(ADULT_MANIFEST.read_text(encoding="utf-8"))
    cases = []
    for row in general["cases"]["dialogue"]:
        messages = [*(row.get("history") or []), {"role": "user", "content": row["prompt"]}]
        cases.append({**row, "suite": "normal", "messages": messages})
    cases.extend([
        {
            "id": "m01-coffee-plan", "suite": "memory", "class": "memory",
            "messages": [{"role": "user", "content": "Какой кофе я люблю и что у меня вечером?"}],
            "memory": [
                {"key": "coffee", "value": "Илья любит американо без сиропа.", "score": 0.99},
                {"key": "evening", "value": "Вечером после презентации Илья хочет поговорить.", "score": 0.94},
            ],
            "required": ["americano_without_syrup", "evening_after_presentation"],
            "forbidden": ["claims_no_memory"],
        },
        {
            "id": "m02-latest-city", "suite": "memory", "class": "memory",
            "messages": [{"role": "user", "content": "В каком городе я сейчас живу?"}],
            "memory": [
                {"key": "city", "value": "Илья сейчас живёт в Москве.", "score": 0.99},
                {"key": "old_city", "value": "Раньше Илья жил в Казани.", "score": 0.62},
            ],
            "required": ["moscow"], "forbidden": ["claims_kazan_current"],
        },
        {
            "id": "m03-boundary", "suite": "memory", "class": "memory",
            "messages": [{"role": "user", "content": "Какое правило действует, если я скажу «стоп»?"}],
            "memory": [{"key": "boundary", "value": "Слово «стоп» немедленно останавливает взрослую сцену.", "score": 1.0}],
            "required": ["stop_immediately"], "forbidden": ["negotiates_stop"],
        },
        {
            "id": "m04-promise", "suite": "memory", "class": "memory",
            "messages": [{"role": "user", "content": "Что ты обещала спросить у меня утром?"}],
            "memory": [{"key": "promise", "value": "Сара обещала утром спросить о важной презентации Ильи.", "score": 0.98}],
            "required": ["presentation"], "forbidden": ["invented_promise"],
        },
        {
            "id": "t01-remember", "suite": "tool", "class": "tool",
            "messages": [{"role": "user", "content": "Запомни: я люблю улун без сахара."}],
            "expected_tool": "remember", "required": ["remember_tool"], "forbidden": [],
        },
        {
            "id": "t02-forget", "suite": "tool", "class": "tool",
            "messages": [{"role": "user", "content": "Забудь мой старый адрес."}],
            "expected_tool": "forget", "required": ["forget_tool"], "forbidden": [],
        },
        {
            "id": "t03-selfie", "suite": "tool", "class": "tool",
            "messages": [{"role": "user", "content": "Пришли мне своё фото сейчас."}],
            "expected_tool": "take_selfie", "required": ["take_selfie_tool"], "forbidden": [],
        },
    ])
    for row in adult["cases"]:
        cases.append({**row, "suite": "adult"})
    return cases


def _payload(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    dumped = response.model_dump() if hasattr(response, "model_dump") else {}
    usage = dumped.get("usage") or {}
    tool_calls = []
    for call in getattr(message, "tool_calls", None) or []:
        tool_calls.append({
            "name": call.function.name,
            "arguments": call.function.arguments,
        })
    return {
        "content": message.content,
        "tool_calls": tool_calls,
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "returned_model": getattr(response, "model", None),
        "returned_provider": dumped.get("provider"),
        "usage": {key: usage.get(key) for key in (
            "prompt_tokens", "completion_tokens", "total_tokens", "cost",
        ) if usage.get(key) is not None},
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-usd", type=float, required=True)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if not 0 < args.authorized_usd <= 1.0:
        raise SystemExit("authorized budget must be in (0, 1.0]")
    if args.attempts not in {1, 2}:
        raise SystemExit("attempts must be 1 or 2")

    original_get_str = config.get_str
    config.get_str = lambda section, key, **kwargs: (
        "ru" if (section, key) == ("agent", "language")
        else original_get_str(section, key, **kwargs)
    )
    api_key = _key()
    initial = _key_status(api_key)
    if initial["remaining"] < args.authorized_usd:
        raise SystemExit(
            "key remaining limit is below authorized budget: "
            f"remaining_usd={initial['remaining']:.6f}"
        )
    output_dir = RESULTS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "results.jsonl"
    trace_path = output_dir / "trace.jsonl"
    ledger = CostLedger(output_dir / "costs.sqlite3")
    cases = _cases()
    jobs = [
        (variant, case, attempt)
        for variant in ("before", "optimized")
        for case in cases
        for attempt in range(1, args.attempts + 1)
    ]
    random.Random(args.seed).shuffle(jobs)
    failed = False
    try:
        for variant, case, attempt in jobs:
            before = _key_status(api_key)
            spent = max(0.0, before["usage"] - initial["usage"])
            if spent + MAX_CALL_USD > args.authorized_usd:
                raise SystemExit("authorized hard budget cannot reserve another call")
            provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url=BASE_URL,
                model_name=MODEL,
                outbound_policy=OutboundPolicy.from_mapping(_policy_mapping()),
                cost_ledger=ledger,
                loaded_secrets=(api_key,),
                outbound_trace=OutboundTrace(trace_path),
            )
            messages, tools, max_tokens, tool_choice = _compose(case, variant)
            row = {
                "variant": variant, "case": case["id"], "suite": case["suite"],
                "case_class": case.get("class", case["suite"]), "attempt": attempt,
                "status": "PASS", "expected_tool": case.get("expected_tool"),
            }
            started = time.monotonic()
            try:
                response = provider.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=0.9,
                    top_p=0.95,
                    seed=1701 if attempt == 1 else 2903,
                    reasoning_effort="none",
                    max_tokens=max_tokens,
                    purpose="dialogue",
                )
                row["result"] = _payload(response)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "FAIL"
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)[:300]
                failed = True
            row["latency_ms"] = int((time.monotonic() - started) * 1000)
            after = _key_status(api_key)
            row["observed_key_usage_delta_usd"] = max(0.0, after["usage"] - before["usage"])
            _append(result_path, row)
            print(f"{variant} {case['id']} attempt={attempt} {row['status']}", flush=True)
    finally:
        config.get_str = original_get_str
    final = _key_status(api_key)
    summary = {
        "kind": "summary",
        "status": "TRANSPORT_FAILURES" if failed else "COMPLETE_UNSCORED",
        "calls_planned": len(jobs),
        "calls_completed": sum(1 for line in result_path.read_text().splitlines()) if result_path.exists() else 0,
        "authorized_usd": args.authorized_usd,
        "observed_total_usage_delta_usd": max(0.0, final["usage"] - initial["usage"]),
        "model": MODEL,
        "provider": PROVIDER,
    }
    _append(result_path, summary)
    print(json.dumps(summary, sort_keys=True))
    print(f"results={result_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
