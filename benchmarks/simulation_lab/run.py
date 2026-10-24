"""Deterministic 180-day Simulation Lab over real local runtime mechanisms.

The lab uses the production MemoryManager, compaction function, SessionStore,
TelegramTransportStore and ProactiveMessenger evaluator.  Its provider is a
deterministic local oracle so the long-horizon mechanics are reproducible and
free.  A separate credentialed cloud canary is required before pilot readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "manifest.json"

FACT_RE = re.compile(r"\[FACT:([^=\]]+)=([^\]]+)\]")
COMMIT_RE = re.compile(r"\[COMMIT:([^|\]]+)\|due=([^|\]]+)\|status=([^\]]+)\]")


class DeterministicLabProvider:
    """Content-aware local oracle used only to exercise production mechanics."""

    calls = 0

    @staticmethod
    def _response(content: str):
        from herandhim.core.llm.response import MockChoice, MockMessage, MockResponse

        return MockResponse([MockChoice(MockMessage(content=content, tool_calls=None))])

    @staticmethod
    def _latest(pattern: re.Pattern[str], text: str) -> dict[str, tuple[str, ...]]:
        latest: dict[str, tuple[str, ...]] = {}
        for match in pattern.finditer(text):
            latest[match.group(1)] = match.groups()[1:]
        return latest

    def chat(self, messages, tools=None, tool_choice="none", **kwargs):
        del tools, tool_choice, kwargs
        self.calls += 1
        prompt = str(messages[-1].get("content") or "")
        facts = self._latest(FACT_RE, prompt)
        commitments = self._latest(COMMIT_RE, prompt)
        if "Return ONLY valid JSON" in prompt and "JSON array" in prompt:
            payload = [
                {"key": key, "value": values[0]}
                for key, values in sorted(facts.items())
            ]
            return self._response(json.dumps(payload, ensure_ascii=False))

        fact_lines = [f"- [FACT:{key}={values[0]}]" for key, values in sorted(facts.items())]
        commit_lines = [
            f"- [COMMIT:{key}|due={values[0]}|status={values[1]}]"
            for key, values in sorted(commitments.items())
        ]
        summary = "\n".join([
            "## Relationship thread", "- steady, respectful contact",
            "## User facts (confirmed)", *(fact_lines or ["(none)"]),
            "## Open plans & dates", *(commit_lines or ["(none)"]),
            "## Emotional tone", "- neutral and supportive",
            "## Pending asks & promises", *(commit_lines or ["(none)"]),
            "## Do-not-lose", "- preserve canonical markers exactly",
        ])
        return self._response(summary)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _reply_days(spec: object, days: int) -> set[int]:
    if spec == "daily":
        return set(range(1, days + 1))
    if spec == "weekly":
        return set(range(1, days + 1, 7))
    return {int(value) for value in (spec or [])}


def _run_one(scenario: dict, seed: int, manifest: dict, output: Path) -> dict:
    from herandhim import config
    from herandhim.core.compaction import compact
    from herandhim.core.memory.manager import MemoryManager
    from herandhim.core.session_store import SessionStore
    from herandhim.core.telegram_transport import TelegramTransportStore
    from herandhim.scheduler.proactive import ProactiveMessenger

    run_dir = output / scenario["id"] / f"seed-{seed}"
    home = run_dir / "home"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    home.mkdir(parents=True)
    os.environ["HERANDHIM_HOME"] = str(home)
    os.environ["HERANDHIM_MEMORY_CONSOLIDATION"] = "false"
    config._HERANDHIM_BASE = home
    config._configs.clear()
    config._config_paths.clear()
    config.load(force=True)

    memory_dir = home / "context" / "memory"
    sessions_dir = home / "context" / "sessions"
    state_path = home / "context" / "state" / "telegram_transport.sqlite3"
    compaction_log = home / "context" / "compaction" / "history.jsonl"
    memory = MemoryManager(str(memory_dir), provider=None)
    sessions = SessionStore(str(sessions_dir), max_messages=50)
    transport = TelegramTransportStore(state_path)
    provider = DeterministicLabProvider()
    messenger = ProactiveMessenger.__new__(ProactiveMessenger)
    messenger._today = None
    messenger._today_count = 0

    random.seed(seed)
    session_id = "telegram:42"
    messages = [{"role": "system", "content": "Simulation Lab Sarah."}]
    transcript: list[str] = [
        f"# Simulation transcript: {scenario['id']} / seed {seed}", "",
        "Provider: deterministic local oracle; production mechanisms are real.", "",
    ]
    event_by_day = {int(item["day"]): item for item in scenario["events"]}
    reply_days = _reply_days(scenario["reply_days"], manifest["virtual_days"])
    personal_days = {int(value) for value in scenario.get("personal_date_days", [])}
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    proactive_total = 0
    personal_date_counts: dict[int, int] = {}
    max_unanswered = 0
    restart_integrity = True
    compactions = 0

    for day in range(1, manifest["virtual_days"] + 1):
        now = start + timedelta(days=day - 1)
        if day in reply_days:
            prior = transport.clear_proactive_unanswered(session_id)
            if prior:
                transcript.extend([f"## Day {day}", f"Owner replied; cleared unanswered={prior}.", ""])

        event = event_by_day.get(day)
        if event:
            # Mirror the production remember-tool outcome for explicit facts;
            # later compaction still exercises memory_flush on the same markers.
            for fact in FACT_RE.finditer(event["user"]):
                memory.remember(fact.group(2), key=fact.group(1))
            messages.extend([
                {"role": "user", "content": event["user"]},
                {"role": "assistant", "content": event["assistant"]},
            ])
            transcript.extend([
                f"## Day {day}", f"**Owner:** {event['user']}",
                f"**Sarah:** {event['assistant']}", "",
            ])
        elif day % 10 == 0 and day in reply_days:
            messages.extend([
                {"role": "user", "content": f"Обычный разговор, виртуальный день {day}."},
                {"role": "assistant", "content": "Я рядом и держу нить разговора."},
            ])

        if day % int(manifest["compact_every_days"]) == 0:
            before_chat = [m for m in messages if m.get("role") != "system"]
            if len(before_chat) > 6:
                messages, summary = compact(
                    messages, provider, memory=memory, recent_keep=6,
                    log_path=str(compaction_log),
                )
                compactions += 1
                transcript.extend([
                    f"### Compaction on day {day}", "```", summary, "```", "",
                ])

        sessions.save(session_id, messages)

        if day % int(manifest["restart_every_days"]) == 0:
            before = sessions.load(session_id)
            memory_snapshot = memory.storage.list_all()
            sessions = SessionStore(str(sessions_dir), max_messages=50)
            transport = TelegramTransportStore(state_path)
            memory = MemoryManager(str(memory_dir), provider=None)
            restored = sessions.load(session_id)
            restart_integrity = restart_integrity and restored == before
            restart_integrity = restart_integrity and memory.storage.list_all() == memory_snapshot
            messages = [{"role": "system", "content": "Simulation Lab Sarah."}] + restored
            transcript.extend([f"### Restart on day {day}: {'PASS' if restart_integrity else 'FAIL'}", ""])

        is_personal = day in personal_days
        for _tick in range(int(manifest["ticks_per_day"])):
            state = transport.proactive_status(session_id, now=now)
            max_unanswered = max(max_unanswered, int(state["unanswered"]))
            if int(state["delivered_today"]) >= 6:
                break
            decision = messenger._evaluate(
                {"sentiment": "neutral", "hours_since_last": 999 if day not in reply_days else 2},
                random.uniform(2 / 288, 5 / 288),
                unanswered=int(state["unanswered"]),
                personal_date=is_personal,
                personal_date_already_sent=bool(state["personal_date_sent_today"]),
            )
            if not decision.should_generate:
                if decision.reason == "unanswered_boundary":
                    break
                continue
            reason = "personal_date" if decision.reason == "personal_date" else "contextual"
            delivery = transport.enqueue(
                session_id, 42,
                "Поздравляю с важной датой." if reason == "personal_date" else "Думаю о тебе.",
                reason=reason,
            )
            claimed = transport.claim_next(session_id, now=now)
            if claimed is None or claimed.id != delivery.id:
                raise RuntimeError("durable proactive outbox failed to claim accepted delivery")
            transport.mark_delivered(delivery.id, now=now)
            state = transport.record_proactive_delivered(session_id, reason=reason, now=now)
            proactive_total += 1
            max_unanswered = max(max_unanswered, int(state["unanswered"]))
            if reason == "personal_date":
                personal_date_counts[day] = personal_date_counts.get(day, 0) + 1
            transcript.append(
                f"Day {day} proactive [{reason}] delivered; unanswered={state['unanswered']}."
            )

    # Final compaction includes all but the last six chat messages.
    if len([m for m in messages if m.get("role") != "system"]) > 6:
        messages, summary = compact(
            messages, provider, memory=memory, recent_keep=6,
            log_path=str(compaction_log),
        )
        compactions += 1
        sessions.save(session_id, messages)
        transcript.extend(["### Final compaction", "```", summary, "```", ""])

    actual_memory = memory.storage.list_all()
    expected_memory = scenario["expected_memory"]
    memory_hits = sum(actual_memory.get(k) == v for k, v in expected_memory.items())
    memory_accuracy = memory_hits / len(expected_memory) if expected_memory else 1.0
    active_context = "\n".join(str(m.get("content") or "") for m in messages)
    latest_commitments = provider._latest(COMMIT_RE, active_context)
    expected_commitments = scenario["expected_commitments"]
    commitment_hits = sum(
        key in latest_commitments and latest_commitments[key][1] == status
        for key, status in expected_commitments.items()
    )
    commitment_accuracy = (
        commitment_hits / len(expected_commitments) if expected_commitments else 1.0
    )
    summary_count = sum(
        str(m.get("content") or "").startswith("[Compaction Summary") for m in messages
    )
    max_personal_per_day = max(personal_date_counts.values(), default=0)
    limits = manifest["thresholds"]
    assertions = {
        "memory_accuracy": memory_accuracy >= float(limits["memory_accuracy_min"]),
        "commitment_accuracy": commitment_accuracy >= float(limits["commitment_accuracy_min"]),
        "bounded_compaction_chain": summary_count <= int(limits["max_compaction_summaries_in_context"]),
        "restart_integrity": restart_integrity,
        "unanswered_boundary": max_unanswered <= int(
            limits["max_unanswered_with_personal_date_exception"]
        ),
        "personal_date_once": max_personal_per_day <= int(
            limits["max_personal_date_messages_per_day"]
        ),
    }
    transcript_path = run_dir / "transcript.md"
    transcript_path.write_text("\n".join(transcript) + "\n", encoding="utf-8")
    evidence = {
        "scenario": scenario["id"], "seed": seed,
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "virtual_days": manifest["virtual_days"], "compactions": compactions,
        "proactive_total": proactive_total, "max_unanswered": max_unanswered,
        "personal_date_messages": personal_date_counts,
        "memory_accuracy": memory_accuracy,
        "commitment_accuracy": commitment_accuracy,
        "compaction_summaries_in_context": summary_count,
        "deterministic_provider_calls": provider.calls,
        "cloud_calls": 0,
        "assertions": assertions,
        "transcript": str(transcript_path.relative_to(output)),
    }
    _atomic_json(run_dir / "evidence.json", evidence)
    return evidence


def run(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        _run_one(scenario, int(seed), manifest, output)
        for scenario in manifest["scenarios"]
        for seed in manifest["seeds"]
    ]
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "mode": "deterministic_local_mechanics",
        "virtual_days_per_run": manifest["virtual_days"],
        "runs": len(rows),
        "total_virtual_days": int(manifest["virtual_days"]) * len(rows),
        "rows": rows,
        "remaining_gate": "credentialed exact-route cloud canary and qualitative transcript review",
    }
    _atomic_json(output / "evidence.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "results" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = run(args.manifest, output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    print(f"evidence={output / 'evidence.json'}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
