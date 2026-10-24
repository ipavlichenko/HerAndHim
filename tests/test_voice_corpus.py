from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_ROOT = PROJECT_ROOT / "benchmarks" / "voice"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_voice_corpus_passes_offline_verifier():
    verifier = _load_module("voice_corpus_verifier", VOICE_ROOT / "verify_corpus.py")
    assert verifier.verify() == []


def test_frozen_reference_texts_match_manifest_exactly():
    manifest = json.loads((VOICE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((VOICE_ROOT / "corpus" / "index.json").read_text(encoding="utf-8"))
    expected = {case["id"]: (case["text"], case["critical"]) for case in manifest["stt_cases"]}
    actual = {entry["case_id"]: (entry["reference_text"], entry["critical"]) for entry in index["entries"]}
    assert actual == expected
