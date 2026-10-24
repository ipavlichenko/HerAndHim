"""Verify frozen voice-corpus identity and technical audio invariants offline."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one audio stream: {path}")
    stream = streams[0]
    return {
        "duration_seconds": round(float(stream["duration"]), 6),
        "codec": stream["codec_name"],
        "sample_rate_hz": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


def _wav_stats(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError(f"master must be mono pcm_s16le: {path}")
        raw = reader.readframes(reader.getnframes())
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    peak = max(abs(value) for value in samples)
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    return peak / 32767, rms / 32767


def verify() -> list[str]:
    errors: list[str] = []
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    materialization = manifest["materialization"]
    if materialization["status"] != "frozen":
        errors.append("manifest materialization status is not frozen")
    index_path = ROOT / materialization["corpus_index"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["stt_cases"]}
    entries = {entry["case_id"]: entry for entry in index["entries"]}
    if set(entries) != set(cases):
        errors.append("corpus case IDs do not exactly match manifest")
    if len(index["voices"]) < materialization["voices_minimum"]:
        errors.append("fewer voices than manifest minimum")
    if len({entry["voice_id"] for entry in index["entries"]}) < materialization["voices_minimum"]:
        errors.append("not all required independent voices are used")
    if any(voice["dataset_license"].lower() == "unknown" for voice in index["voices"]):
        errors.append("voice with unknown dataset licence")
    qa_path = ROOT / "corpus" / "qa_machine_listening.json"
    if not qa_path.is_file():
        errors.append("missing offline machine-listening QA report")
        qa_results = {}
    else:
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa_results = {result["case_id"]: result for result in qa["results"]}

    for case_id, case in cases.items():
        entry = entries.get(case_id)
        if not entry:
            continue
        if entry["reference_text"] != case["text"] or entry["critical"] != case["critical"]:
            errors.append(f"{case_id}: frozen reference text/critical spans drifted")
        if entry["condition"] != case["condition"]:
            errors.append(f"{case_id}: condition drifted")
        for asset_name in ("clean_master", "telegram_derivative"):
            recorded = entry[asset_name]
            path = ROOT / recorded["path"]
            if not path.is_file():
                errors.append(f"{case_id}: missing {asset_name}")
                continue
            if path.stat().st_size != recorded["bytes"] or _sha256(path) != recorded["sha256"]:
                errors.append(f"{case_id}: {asset_name} bytes/hash mismatch")
                continue
            try:
                actual = _probe(path)
            except (ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                errors.append(f"{case_id}: corrupt {asset_name}: {exc}")
                continue
            for key in ("codec", "sample_rate_hz", "channels", "duration_seconds"):
                if actual[key] != recorded[key]:
                    errors.append(f"{case_id}: {asset_name} {key} drifted")
        derivative = entry["telegram_derivative"]
        if derivative["codec"] != "opus" or derivative["sample_rate_hz"] != 48000 or derivative["channels"] != 1:
            errors.append(f"{case_id}: derivative is not mono 48 kHz Opus")
        if not 0 < derivative["duration_seconds"] <= 120:
            errors.append(f"{case_id}: derivative duration outside Telegram benchmark boundary")
        qa_result = qa_results.get(case_id)
        if not qa_result or qa_result["status"] != "pass_gross_content_integrity":
            errors.append(f"{case_id}: machine-listening QA missing or not passed")
        elif qa_result["derivative_sha256"] != derivative["sha256"]:
            errors.append(f"{case_id}: machine-listening QA belongs to different derivative")
        master_path = ROOT / entry["clean_master"]["path"]
        if master_path.is_file():
            try:
                peak, rms = _wav_stats(master_path)
            except (ValueError, wave.Error) as exc:
                errors.append(f"{case_id}: invalid master PCM: {exc}")
            else:
                if peak >= 1.0:
                    errors.append(f"{case_id}: clipped clean master")
                if rms < 0.001:
                    errors.append(f"{case_id}: silent clean master")

    return errors


def main() -> int:
    errors = verify()
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    index = json.loads((ROOT / "corpus" / "index.json").read_text(encoding="utf-8"))
    print(f"PASS: {len(index['entries'])} frozen STT cases, {len(index['voices'])} licensed voices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
