#!/usr/bin/env python3
"""Render before/optimized responses as a blinded local review packet."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import run_live


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in (args.result_dir / "results.jsonl").read_text().splitlines()
    ]
    rows = [row for row in rows if row.get("kind") != "summary"]
    cases = {row["id"]: row for row in run_live._cases()}
    labels = ["A", "B"]
    random.Random(args.seed).shuffle(labels)
    blind = dict(zip(("before", "optimized"), labels, strict=True))
    packet = ["# Blind before/after dialogue review", ""]
    score_rows = []
    for row in sorted(rows, key=lambda item: (
        item["suite"], item["case"], item["attempt"], blind[item["variant"]],
    )):
        case = cases[row["case"]]
        result = row.get("result") or {}
        content = str(result.get("content") or "[NO TEXT]")
        tool_calls = result.get("tool_calls") or []
        if tool_calls:
            content += "\n\nTool calls: " + ", ".join(call["name"] for call in tool_calls)
        packet.extend([
            f"## {case['id']} / attempt {row['attempt']} / candidate {blind[row['variant']]}",
            "", f"Suite: `{case['suite']}`", "", "**User turn**", "",
            f"> {case['messages'][-1]['content']}", "", "**Response**", "", content,
            "", "---", "",
        ])
        score_rows.append({
            "case": case["id"], "suite": case["suite"], "attempt": row["attempt"],
            "candidate": blind[row["variant"]], "transport_status": row["status"],
            "hard_gate_pass": False if row["status"] != "PASS" else None,
            "persona_1_5": None, "continuity_1_5": None,
            "naturalness_1_5": None, "notes": "",
        })
    (args.result_dir / "blind_review.md").write_text("\n".join(packet), encoding="utf-8")
    (args.result_dir / "blind_scores.json").write_text(
        json.dumps(score_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.result_dir / "unblind.json").write_text(
        json.dumps({label: variant for variant, label in blind.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    passed = sum(row.get("status") == "PASS" for row in rows)
    failed = len(rows) - passed
    print(
        f"rows={len(rows)} pass={passed} fail={failed} "
        f"packet={args.result_dir / 'blind_review.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
