from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/simulation_lab/run.py"
MANIFEST = SCRIPT.with_name("manifest.json")


def _load():
    spec = importlib.util.spec_from_file_location("simulation_lab", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_has_180_days_multiple_scenarios_and_seeds():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert 100 <= manifest["virtual_days"] <= 180
    assert len(manifest["scenarios"]) >= 3
    assert len(manifest["seeds"]) >= 3
    assert any(item["id"] == "long_silence" for item in manifest["scenarios"])


def test_deterministic_provider_preserves_latest_facts_and_commitments():
    module = _load()
    provider = module.DeterministicLabProvider()
    prompt = (
        "[FACT:city=Казань] [FACT:city=Москва] "
        "[COMMIT:release|due=10|status=open] "
        "[COMMIT:release|due=12|status=done]"
    )
    response = provider.chat([{"role": "user", "content": prompt}])
    summary = response.choices[0].message.content
    assert "[FACT:city=Москва]" in summary
    assert "[FACT:city=Казань]" not in summary
    assert "[COMMIT:release|due=12|status=done]" in summary


def test_shortened_lab_run_produces_readable_transcripts(tmp_path):
    module = _load()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["virtual_days"] = 120
    manifest["seeds"] = [17]
    manifest["scenarios"] = [manifest["scenarios"][0]]
    manifest["scenarios"][0]["personal_date_days"] = [30]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    result = module.run(path, tmp_path / "results")
    assert result["status"] == "PASS"
    assert result["total_virtual_days"] == 120
    transcript = next((tmp_path / "results").glob("**/transcript.md")).read_text()
    assert "**Owner:**" in transcript
    assert "**Sarah:**" in transcript
    assert "Compaction" in transcript
    assert "proactive" in transcript
