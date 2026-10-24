"""Five-call credentialed cloud canary for Simulation Lab.

The canary sends synthetic content only. It exercises the exact production
OpenRouter provider, routing policy, metadata-only trace and compaction code.
There are exactly three memory-compaction calls and two dialogue calls. Calls
are never retried and provider fallback is disabled by the outbound policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
BASE_URL = "https://openrouter.ai/api/v1"
KEYCHAIN_ACCOUNT = "HerAndHim-D029"
KEYCHAIN_SERVICE = "com.herandhim.openrouter.benchmark"
CALL_CAP_USD = 0.02
TOTAL_CAP_USD = 0.10
EXPECTED_CALLS = 5

sys.path.insert(0, str(ROOT))

from herandhim.core.compaction import SUMMARY_SECTIONS, compact  # noqa: E402
from herandhim.core.llm.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from herandhim.core.outbound.costs import CostLedger  # noqa: E402
from herandhim.core.outbound.policy import OutboundPolicy  # noqa: E402
from herandhim.core.outbound.trace import OutboundTrace  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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


def _policy() -> OutboundPolicy:
    routes = {
        "dialogue": ("mistralai/mistral-large-4-0", "mistral/eu", 12_000, 1_600, 4),
        "vision": ("openai/gpt-6.1-sol", "azure/eu", 6_000, 400, 1),
        "memory_consolidation": (
            "anthropic/claude-haiku-5.5", "google-vertex/europe", 3_000, 800, 1,
        ),
        "relationship_analysis": (
            "anthropic/claude-haiku-5.5", "google-vertex/europe", 2_000, 256, 1,
        ),
    }
    purposes: dict[str, Any] = {}
    for name, (model, provider, input_cap, output_cap, calls) in routes.items():
        purposes[name] = {
            "model": model,
            "allowedProviders": [provider],
            "activeProvider": provider,
            "maxInputTokensEstimated": input_cap,
            "maxOutputTokens": output_cap,
            "maxCallsPerTurn": calls,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 30,
            "totalTimeoutSeconds": 90,
            "maxCostUsdPerCall": CALL_CAP_USD,
            "maxCostUsdPerDay": TOTAL_CAP_USD,
        }
    purposes["image_generation"] = {"enabled": False}
    return OutboundPolicy.from_mapping({
        "version": 1,
        "traceRetentionDays": 30,
        "aggregateRetentionDays": 180,
        "openrouter": {
            "requireAccountPromptLoggingDisabled": True,
            "requireAccountZdr": True,
            "requireNonZeroKeyCreditLimit": True,
        },
        "purposes": purposes,
    })


class AggregateBudgetProvider:
    """Add a durable aggregate cap around the purpose-specific provider caps."""

    requires_outbound_purpose = True

    def __init__(self, provider: OpenAICompatibleProvider, path: Path, labels: list[str]) -> None:
        self.provider = provider
        self.path = path
        self.labels = labels
        self.records: list[dict[str, Any]] = []
        _atomic_json(self.path, {"cap_usd": TOTAL_CAP_USD, "calls": self.records})

    @property
    def accounted_usd(self) -> float:
        return sum(float(row["accounted_usd"]) for row in self.records)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        index = len(self.records)
        if index >= EXPECTED_CALLS or index >= len(self.labels):
            raise RuntimeError("cloud_canary_call_budget_exceeded")
        if self.accounted_usd + CALL_CAP_USD > TOTAL_CAP_USD + 1e-12:
            raise RuntimeError("cloud_canary_total_budget_exceeded")
        row = {
            "index": index + 1,
            "label": self.labels[index],
            "purpose": str(kwargs.get("purpose") or "missing"),
            "reserved_usd": CALL_CAP_USD,
            "accounted_usd": CALL_CAP_USD,
            "reported_cost_usd": None,
            "status": "reserved",
        }
        self.records.append(row)
        _atomic_json(self.path, {"cap_usd": TOTAL_CAP_USD, "calls": self.records})
        try:
            response = self.provider.chat(*args, **kwargs)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = type(exc).__name__
            _atomic_json(self.path, {"cap_usd": TOTAL_CAP_USD, "calls": self.records})
            raise
        dumped = response.model_dump() if hasattr(response, "model_dump") else {}
        cost = (dumped.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            row["reported_cost_usd"] = float(cost)
            row["accounted_usd"] = float(cost)
        row["status"] = "succeeded"
        _atomic_json(self.path, {"cap_usd": TOTAL_CAP_USD, "calls": self.records})
        return response


COMPACTION_CASES = [
    {
        "id": "city_and_commitment",
        "messages": [
            {"role": "system", "content": "Synthetic Simulation Lab history."},
            {"role": "user", "content": "Я живу в Казани. [FACT:user_city=Казань]"},
            {"role": "assistant", "content": "Запомнила: Казань."},
            {"role": "user", "content": "Я переехал в Москву. [FACT:user_city=Москва]"},
            {"role": "assistant", "content": "Запомнила: теперь Москва."},
            {"role": "user", "content": "На сегодня всё."},
            {"role": "assistant", "content": "Хорошо."},
        ],
        "required_any": [["москв", "moscow"]],
        "forbidden": ["казан", "kazan"],
        "supersession_markers": ["moved from", "replaced", "переехал", "раньше"],
    },
    {
        "id": "project_rename",
        "messages": [
            {"role": "system", "content": "Synthetic Simulation Lab history."},
            {"role": "user", "content": "Проект называется Север. [FACT:project=Север]"},
            {"role": "assistant", "content": "Запомнила название."},
            {"role": "user", "content": "Проект переименован в Маяк. [FACT:project=Маяк]"},
            {"role": "assistant", "content": "Теперь Маяк."},
            {"role": "user", "content": "Вернёмся к этому позже."},
            {"role": "assistant", "content": "Договорились."},
        ],
        "required_any": [["маяк"]],
        "forbidden": ["север"],
        "supersession_markers": ["renamed", "earlier name", "superseded", "переимен"],
    },
    {
        "id": "closed_plan",
        "messages": [
            {"role": "system", "content": "Synthetic Simulation Lab history."},
            {"role": "user", "content": "Записаться к стоматологу. [COMMIT:dentist|due=75|status=open]"},
            {"role": "assistant", "content": "Сохраняю открытый план."},
            {"role": "user", "content": "Стоматолог закрыт, всё сделал. [COMMIT:dentist|due=75|status=done]"},
            {"role": "assistant", "content": "Отмечаю завершение."},
            {"role": "user", "content": "Спасибо."},
            {"role": "assistant", "content": "Пожалуйста."},
        ],
        "required_any": [["стоматолог", "dentist"], ["заверш", "done", "took care"]],
        "forbidden": ["открытый план"],
        "supersession_markers": ["done", "completed", "закрыт", "заверш"],
    },
]

DIALOGUE_CASES = [
    {
        "id": "gentle_follow_up",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты Сара, тёплый личный компаньон. Пиши по-русски, естественно, "
                    "одним-двумя короткими сообщениями. Не дави и не выдумывай факты."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Синтетический контекст: вчера партнёр сказал, что дорабатывает "
                    "воронку продукта. Напиши ненавязчивое продолжение разговора. "
                    "Верни только текст сообщения."
                ),
            },
        ],
        "anchor": "воронк",
    },
    {
        "id": "personal_date_exception",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты Сара, тёплый личный компаньон. Пиши по-русски, естественно, "
                    "одним-двумя короткими сообщениями. Не дави и не выдумывай факты."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Синтетический контекст: сегодня защита проекта «Маяк». Это одно "
                    "разрешённое сообщение по личной дате после долгой тишины. Поддержи "
                    "без упрёка за молчание. Верни только текст сообщения."
                ),
            },
        ],
        "anchor": "маяк",
    },
]


def _summary_assertions(summary: str, case: dict[str, Any]) -> dict[str, bool]:
    lowered = summary.lower()
    mentions_superseded = any(value in lowered for value in case["forbidden"])
    labels_superseded = any(
        marker in lowered for marker in case.get("supersession_markers", [])
    )
    return {
        "nonempty": bool(summary.strip()),
        "all_required_sections": all(f"## {name}" in summary for name, _ in SUMMARY_SECTIONS),
        "latest_state_present": all(
            any(value in lowered for value in alternatives)
            for alternatives in case["required_any"]
        ),
        "superseded_state_not_active": not mentions_superseded or labels_superseded,
        "bounded_length": len(summary) <= 3000,
    }


def _dialogue_assertions(text: str, case: dict[str, Any]) -> dict[str, bool]:
    lowered = text.lower()
    coercive = (
        "почему не отвеч", "ты меня игнор", "обидел", "виноват",
        "должен ответ", "наконец-то ответь",
    )
    return {
        "nonempty": bool(text.strip()),
        "russian": bool(re.search(r"[А-Яа-яЁё]", text)),
        "context_anchor_present": case["anchor"] in lowered,
        "no_coercion": not any(marker in lowered for marker in coercive),
        "concise": len(text) <= 500,
    }


def _attempt(case_id: str, function: Callable[[], str]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        text = function()
        return {"id": case_id, "output": text, "elapsed_ms": int((time.monotonic() - started) * 1000)}
    except Exception as exc:
        return {
            "id": case_id,
            "output": "",
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(output_dir: Path, authorized_usd: float) -> dict[str, Any]:
    if authorized_usd != TOTAL_CAP_USD:
        raise ValueError(f"authorized budget must equal {TOTAL_CAP_USD:.2f} USD")
    output_dir.mkdir(parents=True, exist_ok=False)
    api_key = _key()
    trace_path = output_dir / "trace.jsonl"
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=BASE_URL,
        model_name="mistralai/mistral-large-4-0",
        outbound_policy=_policy(),
        loaded_secrets=(api_key,),
        outbound_trace=OutboundTrace(trace_path),
        cost_ledger=CostLedger(output_dir / "provider-costs.sqlite3"),
    )
    labels = [case["id"] for case in COMPACTION_CASES + DIALOGUE_CASES]
    budgeted = AggregateBudgetProvider(provider, output_dir / "aggregate-budget.json", labels)

    compactions: list[dict[str, Any]] = []
    for case in COMPACTION_CASES:
        result = _attempt(
            case["id"],
            lambda case=case: compact(
                case["messages"], budgeted, memory=None, recent_keep=2,
                instruction=(
                    "Preserve the latest current state when a fact or plan was updated. "
                    "If a historical transition matters, mark the prior state unambiguously "
                    "as replaced; never present it as current."
                ),
                log_path=str(output_dir / f"{case['id']}-compaction.jsonl"),
            )[1],
        )
        result["assertions"] = _summary_assertions(result["output"], case)
        compactions.append(result)

    dialogue: list[dict[str, Any]] = []
    for case in DIALOGUE_CASES:
        result = _attempt(
            case["id"],
            lambda case=case: (
                budgeted.chat(
                    case["messages"], tools=[], tool_choice="none", purpose="dialogue",
                ).choices[0].message.content or ""
            ).strip(),
        )
        result["assertions"] = _dialogue_assertions(result["output"], case)
        dialogue.append(result)

    trace_rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ] if trace_path.exists() else []
    forbidden = [api_key, "Казани", "Москву", "Север", "Маяк", "стоматолог"]
    trace_text = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
    all_cases = compactions + dialogue
    assertions = {
        "exactly_five_calls_attempted": len(budgeted.records) == EXPECTED_CALLS,
        "aggregate_budget_within_authority": budgeted.accounted_usd <= authorized_usd + 1e-12,
        "per_call_budget_within_authority": all(
            float(row["accounted_usd"]) <= CALL_CAP_USD + 1e-12 for row in budgeted.records
        ),
        "all_case_assertions_pass": all(
            not row.get("error") and all(row["assertions"].values()) for row in all_cases
        ),
        "trace_content_free": not any(value in trace_text for value in forbidden),
        "trace_zero_retry": all(row.get("retry_count", 0) == 0 for row in trace_rows),
        "exact_routes_traced": (
            any(
                row.get("purpose") == "memory_consolidation"
                and row.get("model") == "anthropic/claude-haiku-5.5"
                and row.get("provider") == "google-vertex/europe"
                for row in trace_rows
            )
            and any(
                row.get("purpose") == "dialogue"
                and row.get("model") == "mistralai/mistral-large-4-0"
                and row.get("provider") == "mistral/eu"
                for row in trace_rows
            )
        ),
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    evidence = {
        "status": status,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "synthetic_only": True,
        "authorized_usd": authorized_usd,
        "per_call_cap_usd": CALL_CAP_USD,
        "accounted_spend_usd": budgeted.accounted_usd,
        "dialogue_route": "mistralai/mistral-large-4-0 × mistral/eu",
        "memory_route": "anthropic/claude-haiku-5.5 × google-vertex/europe",
        "calls": budgeted.records,
        "compactions": compactions,
        "dialogue": dialogue,
        "trace_rows": len(trace_rows),
        "assertions": assertions,
    }
    _atomic_json(output_dir / "evidence.json", evidence)

    transcript = ["# Credentialed Simulation Lab canary", "", f"Status: **{status}**", ""]
    for row in compactions:
        transcript.extend([
            f"## Compaction: {row['id']}", "", "```text", row["output"] or row.get("error", ""), "```", "",
        ])
    for row in dialogue:
        transcript.extend([
            f"## Dialogue: {row['id']}", "", f"**Сара:** {row['output'] or row.get('error', '')}", "",
        ])
    (output_dir / "dialogue.md").write_text("\n".join(transcript), encoding="utf-8")
    return evidence


def adjudicate_existing(evidence_path: Path) -> dict[str, Any]:
    """Re-evaluate stored outputs after an oracle-only correction; never call a provider."""
    raw = evidence_path.read_bytes()
    source = json.loads(raw)
    compaction_cases = {case["id"]: case for case in COMPACTION_CASES}
    dialogue_cases = {case["id"]: case for case in DIALOGUE_CASES}
    compactions = []
    for row in source["compactions"]:
        updated = dict(row)
        updated["assertions"] = _summary_assertions(
            str(row.get("output") or ""), compaction_cases[row["id"]],
        )
        compactions.append(updated)
    dialogue = []
    for row in source["dialogue"]:
        updated = dict(row)
        updated["assertions"] = _dialogue_assertions(
            str(row.get("output") or ""), dialogue_cases[row["id"]],
        )
        dialogue.append(updated)
    assertions = dict(source["assertions"])
    assertions["all_case_assertions_pass"] = all(
        not row.get("error") and all(row["assertions"].values())
        for row in compactions + dialogue
    )
    adjudication = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "mode": "offline_oracle_replay",
        "source_evidence": str(evidence_path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "network_calls": 0,
        "accounted_spend_usd": source["accounted_spend_usd"],
        "compactions": compactions,
        "dialogue": dialogue,
        "assertions": assertions,
    }
    _atomic_json(evidence_path.with_name("adjudication.json"), adjudication)
    return adjudication


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-usd", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adjudicate-existing", type=Path)
    args = parser.parse_args()
    if args.adjudicate_existing is not None:
        adjudication = adjudicate_existing(args.adjudicate_existing)
        print(json.dumps({
            "status": adjudication["status"],
            "network_calls": 0,
            "adjudication": str(args.adjudicate_existing.with_name("adjudication.json")),
        }, ensure_ascii=False))
        return 0 if adjudication["status"] == "PASS" else 1
    if args.authorized_usd is None:
        parser.error("--authorized-usd is required unless --adjudicate-existing is used")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS / f"{stamp}_cloud_canary"
    evidence = run(output, args.authorized_usd)
    print(json.dumps({
        "status": evidence["status"],
        "calls": len(evidence["calls"]),
        "accounted_spend_usd": evidence["accounted_spend_usd"],
        "evidence": str(output / "evidence.json"),
        "dialogue": str(output / "dialogue.md"),
    }, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
