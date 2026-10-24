"""Offline semantic QA for the frozen corpus using local faster-whisper.

Run with:
    uv run --with faster-whisper python benchmarks/voice/machine_listen.py

The model may be downloaded before the run. Inference itself is local and this
script makes no provider/API request.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-zа-я0-9]+", text.casefold().replace("ё", "е")))


def _decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> int:
    index_path = ROOT / "corpus" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    results = []
    failed = False
    for entry in index["entries"]:
        derivative = ROOT / entry["telegram_derivative"]["path"]
        segments, info = model.transcribe(
            _decode(derivative),
            language="ru",
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        candidates = [entry["reference_text"]]
        if pronunciation := entry.get("pronunciation_only_synthesis_text"):
            candidates.append(pronunciation)
        similarity = max(
            difflib.SequenceMatcher(None, _normalize(candidate), _normalize(transcript)).ratio()
            for candidate in candidates
        )
        passed = bool(transcript) and info.language_probability >= 0.9 and similarity >= 0.55
        failed |= not passed
        results.append(
            {
                "case_id": entry["case_id"],
                "derivative_sha256": _sha256(derivative),
                "transcript": transcript,
                "language": info.language,
                "language_probability": round(info.language_probability, 6),
                "normalized_character_similarity": round(similarity, 6),
                "status": "pass_gross_content_integrity" if passed else "review_required",
            }
        )
    report = {
        "version": 1,
        "scope": "offline machine-listening QA; not a human naturalness/persona rating",
        "model": "faster-whisper medium",
        "faster_whisper_version": importlib.metadata.version("faster-whisper"),
        "ctranslate2_version": importlib.metadata.version("ctranslate2"),
        "network_during_inference": False,
        "thresholds": {"language_probability_min": 0.9, "normalized_character_similarity_min": 0.55},
        "results": results,
    }
    (ROOT / "corpus" / "qa_machine_listening.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
