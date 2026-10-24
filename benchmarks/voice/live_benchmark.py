"""Credentialed, zero-retry cloud voice benchmark.

Secrets are read from files outside the repository.  Product traces are
metadata-only; content-bearing transcripts/audio stay in the explicitly
selected benchmark result directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import secrets
import statistics
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
INDEX_PATH = ROOT / "corpus" / "index.json"
DEFAULT_KEYS = {
    "deepgram": Path("/private/tmp/herandhim-deepgram.key"),
    "openai": Path("/private/tmp/herandhim-openai.key"),
    "google_cloud": Path("/private/tmp/herandhim-google.key"),
}


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class BudgetError(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    path: Path
    total_ceiling: float
    per_call_ceiling: float

    def __post_init__(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "version": 1,
                "total_ceiling_usd": self.total_ceiling,
                "per_call_ceiling_usd": self.per_call_ceiling,
                "reservations": [],
            }
            _atomic_json(self.path, self.data)
        if self.data["total_ceiling_usd"] != self.total_ceiling:
            raise BudgetError("ledger total ceiling mismatch")
        if self.data["per_call_ceiling_usd"] != self.per_call_ceiling:
            raise BudgetError("ledger per-call ceiling mismatch")

    @property
    def reserved_total(self) -> float:
        return sum(float(item["reserved_usd"]) for item in self.data["reservations"])

    def reserve(self, *, request_id: str, estimated_usd: float, route: str) -> None:
        if estimated_usd <= 0 or estimated_usd > self.per_call_ceiling:
            raise BudgetError(f"call estimate {estimated_usd:.8f} outside per-call ceiling")
        if self.reserved_total + estimated_usd > self.total_ceiling:
            raise BudgetError("aggregate benchmark ceiling would be exceeded")
        self.data["reservations"].append({
            "request_id": request_id,
            "route": route,
            "reserved_usd": round(estimated_usd, 8),
            "status": "reserved",
        })
        _atomic_json(self.path, self.data)

    def settle(self, request_id: str, *, status: str, observed_usd: float | None = None) -> None:
        for item in self.data["reservations"]:
            if item["request_id"] == request_id:
                item["status"] = status
                if observed_usd is not None:
                    item["observed_usd"] = round(observed_usd, 8)
                _atomic_json(self.path, self.data)
                return
        raise BudgetError(f"unknown request reservation: {request_id}")


def _key(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"empty credential file: {path}")
    return value


def _request_id() -> str:
    return secrets.token_hex(12)


def _normalise(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    spoken_numbers = {
        "двадцать третьего октября": "23 октября",
        "восемнадцать сорок": "18 40",
        "триста пятьдесят тысяч": "350 тысяч",
        "двадцать": "20",
    }
    for spoken, canonical in spoken_numbers.items():
        value = value.replace(spoken, canonical)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return [token for token in value.split() if token]


def _edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        curr = [i]
        for j, b in enumerate(right, start=1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = _normalise(reference)
    return _edit_distance(ref, _normalise(hypothesis)) / max(1, len(ref))


_CRITICAL_EQUIVALENTS = {
    "двадцать": ("двадцать", "20"),
    "23 октября": ("23 октября", "23 его октября", "двадцать третьего октября"),
    "18 40": ("18 40", "восемнадцать сорок"),
    "350000": ("350000", "350 000", "350 тысяч", "триста пятьдесят тысяч"),
}


def critical_hit(span: str, hypothesis: str) -> bool:
    normal_span = " ".join(_normalise(span))
    normal_hypothesis = " ".join(_normalise(hypothesis))
    options = _CRITICAL_EQUIVALENTS.get(normal_span, (normal_span,))
    return any(" ".join(_normalise(option)) in normal_hypothesis for option in options)


def _trace(
    path: Path,
    *,
    request_id: str,
    purpose: str,
    provider: str,
    model: str,
    route_meta: dict[str, Any],
    input_count: int,
    output_count: int,
    latency_ms: int,
    status: str,
    cost_usd: float,
    error: str | None = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "route": route_meta,
        "input_count": input_count,
        "output_count": output_count,
        "latency_ms": latency_ms,
        "status": status,
        "error_enum": error,
        "cost_usd": round(cost_usd, 8),
        "retry_count": 0,
    }
    _append_jsonl(path, record)


def _stt_call(candidate: dict[str, Any], audio: bytes, filename: str, keys: dict[str, Path]) -> tuple[str, dict]:
    provider = candidate["provider"]
    model = candidate["model"]
    with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=False) as client:
        if provider == "deepgram":
            params = {
                "model": model,
                "language": candidate["language"],
                "mip_opt_out": "true",
                "smart_format": "true",
                "punctuate": "true",
            }
            response = client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                content=audio,
                headers={"Authorization": f"Token {_key(keys[provider])}", "Content-Type": "audio/ogg"},
            )
            response.raise_for_status()
            payload = response.json()
            transcript = payload["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript, {"request_id": response.headers.get("dg-request-id")}
        if provider == "openai":
            data = {"model": model, "language": "ru", "response_format": "json"}
            if candidate.get("prompt_profile"):
                data["prompt"] = candidate["prompt_profile"]
            response = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {_key(keys[provider])}"},
                data=data,
                files={"file": (filename, audio, "audio/ogg")},
            )
            response.raise_for_status()
            payload = response.json()
            return payload["text"], {"request_id": response.headers.get("x-request-id"), "usage": payload.get("usage")}
        if provider == "google_cloud":
            project = candidate["project_id"]
            region = candidate["region"]
            url = (
                f"https://{region}-speech.googleapis.com/v2/projects/{project}/locations/{region}/"
                "recognizers/_:recognize"
            )
            response = client.post(
                url,
                params={"key": _key(keys[provider])},
                json={
                    "config": {
                        "autoDecodingConfig": {},
                        "languageCodes": [candidate["language"]],
                        "model": model,
                        "features": {"enableAutomaticPunctuation": True},
                    },
                    "content": base64.b64encode(audio).decode("ascii"),
                },
            )
            response.raise_for_status()
            payload = response.json()
            transcript = " ".join(
                result["alternatives"][0]["transcript"].strip()
                for result in payload.get("results", [])
                if result.get("alternatives")
            ).strip()
            return transcript, {"request_id": response.headers.get("x-request-id")}
    raise RuntimeError(f"unsupported STT provider: {provider}")


def _tts_call(candidate: dict[str, Any], text: str, keys: dict[str, Path]) -> bytes:
    if candidate["provider"] != "google_cloud":
        raise RuntimeError(f"unsupported TTS provider: {candidate['provider']}")
    body = {
        "input": {"text": text},
        "voice": {"languageCode": "ru-RU", "name": candidate["voice_id"]},
        "audioConfig": {"audioEncoding": "OGG_OPUS"},
    }
    with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=False) as client:
        response = client.post(
            "https://eu-texttospeech.googleapis.com/v1/text:synthesize",
            params={"key": _key(keys["google_cloud"])},
            json=body,
        )
        response.raise_for_status()
        return base64.b64decode(response.json()["audioContent"], validate=True)


def _terminal_route_error(exc: Exception) -> bool:
    """Stop a candidate after auth/config failures; never multiply bad calls."""
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {400, 401, 403, 404}
    return False


def run_stt(
    manifest: dict,
    index: dict,
    output: Path,
    ledger: BudgetLedger,
    keys: dict[str, Path],
    selected: set[str] | None = None,
) -> None:
    candidates = [item for item in manifest["stt_candidates"] if not item.get("eligibility_gate")]
    if selected:
        candidates = [item for item in candidates if item["id"] in selected]
    cases = {item["case_id"]: item for item in index["entries"]}
    results_path = output / "stt_results.jsonl"
    trace_path = output / "metadata_trace.jsonl"
    for candidate in candidates:
        for case in manifest["stt_cases"]:
            entry = cases[case["id"]]
            audio_path = ROOT / entry["telegram_derivative"]["path"]
            audio = audio_path.read_bytes()
            duration = float(entry["telegram_derivative"]["duration_seconds"])
            estimated = max(0.000001, duration / 60 * float(candidate["public_price_usd_per_minute"]))
            request_id = _request_id()
            route = f"{candidate['provider']}:{candidate['model']}"
            ledger.reserve(request_id=request_id, estimated_usd=estimated, route=route)
            started = time.perf_counter()
            try:
                transcript, usage = _stt_call(candidate, audio, audio_path.name, keys)
                latency_ms = round((time.perf_counter() - started) * 1000)
                spans = {span: critical_hit(span, transcript) for span in case["critical"]}
                record = {
                    "request_id": request_id,
                    "candidate_id": candidate["id"],
                    "case_id": case["id"],
                    "reference": case["text"],
                    "transcript": transcript,
                    "wer": round(wer(case["text"], transcript), 6),
                    "critical_spans": spans,
                    "latency_ms": latency_ms,
                    "estimated_cost_usd": round(estimated, 8),
                    "provider_usage": usage,
                    "status": "pass",
                }
                _append_jsonl(results_path, record)
                ledger.settle(request_id, status="settled", observed_usd=estimated)
                _trace(trace_path, request_id=request_id, purpose="stt", provider=candidate["provider"],
                       model=candidate["model"], route_meta={"language": "ru", "retention": "zero"},
                       input_count=len(audio), output_count=len(transcript), latency_ms=latency_ms,
                       status="success", cost_usd=estimated)
            except Exception as exc:
                latency_ms = round((time.perf_counter() - started) * 1000)
                ledger.settle(request_id, status="ambiguous_or_failed")
                _append_jsonl(results_path, {"request_id": request_id, "candidate_id": candidate["id"],
                                              "case_id": case["id"], "status": "failed",
                                              "error_enum": type(exc).__name__})
                _trace(trace_path, request_id=request_id, purpose="stt", provider=candidate["provider"],
                       model=candidate["model"], route_meta={"language": "ru", "retention": "zero"},
                       input_count=len(audio), output_count=0, latency_ms=latency_ms, status="failed",
                       cost_usd=estimated, error=type(exc).__name__)
                if _terminal_route_error(exc):
                    break


def run_tts(manifest: dict, output: Path, ledger: BudgetLedger, keys: dict[str, Path]) -> None:
    candidates = [item for item in manifest["tts_candidates"] if not item.get("eligibility_gate")]
    randomised = list(candidates)
    secrets.SystemRandom().shuffle(randomised)
    labels = {chr(65 + idx): candidate for idx, candidate in enumerate(randomised)}
    private_map = {label: candidate["id"] for label, candidate in labels.items()}
    _atomic_json(output / ".unblind.json", private_map)
    os.chmod(output / ".unblind.json", 0o600)
    _atomic_json(output / "tts_blind_manifest.json", {
        "labels": list(labels),
        "cases": manifest["tts_cases"],
        "rating_scale": {"naturalness": "1-5", "persona_fit": "1-5", "pronunciation": "1-5", "prosody": "1-5"},
    })
    trace_path = output / "metadata_trace.jsonl"
    results_path = output / "tts_results.jsonl"
    for label, candidate in labels.items():
        for case in manifest["tts_cases"]:
            text = case["text"]
            estimated = max(0.000001, len(text) / 1000 * float(candidate["public_ceiling_usd_per_1000_chars"]))
            request_id = _request_id()
            route = f"{candidate['provider']}:{candidate['model']}:{candidate['voice_id']}:{candidate['region']}"
            ledger.reserve(request_id=request_id, estimated_usd=estimated, route=route)
            started = time.perf_counter()
            try:
                audio = _tts_call(candidate, text, keys)
                latency_ms = round((time.perf_counter() - started) * 1000)
                target = output / "tts_blind" / label / f"{case['id']}.ogg"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(audio)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
                _append_jsonl(results_path, {"request_id": request_id, "label": label, "case_id": case["id"],
                                              "status": "pass", "output_bytes": len(audio),
                                              "latency_ms": latency_ms, "estimated_cost_usd": round(estimated, 8)})
                ledger.settle(request_id, status="settled", observed_usd=estimated)
                _trace(trace_path, request_id=request_id, purpose="tts", provider=candidate["provider"],
                       model=candidate["model"], route_meta={"voice_id": candidate["voice_id"],
                       "region": candidate["region"], "retention": "logging_disabled"},
                       input_count=len(text), output_count=len(audio), latency_ms=latency_ms,
                       status="success", cost_usd=estimated)
            except Exception as exc:
                latency_ms = round((time.perf_counter() - started) * 1000)
                ledger.settle(request_id, status="ambiguous_or_failed")
                _append_jsonl(results_path, {"request_id": request_id, "label": label, "case_id": case["id"],
                                              "status": "failed", "error_enum": type(exc).__name__})
                _trace(trace_path, request_id=request_id, purpose="tts", provider=candidate["provider"],
                       model=candidate["model"], route_meta={"voice_id": candidate["voice_id"],
                       "region": candidate["region"], "retention": "logging_disabled"},
                       input_count=len(text), output_count=0, latency_ms=latency_ms, status="failed",
                       cost_usd=estimated, error=type(exc).__name__)


def summarise_stt(output: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in (output / "stt_results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary: dict[str, Any] = {}
    for candidate in sorted({record["candidate_id"] for record in records}):
        rows = [record for record in records if record["candidate_id"] == candidate]
        passed = [row for row in rows if row["status"] == "pass"]
        hits = [hit for row in passed for hit in row["critical_spans"].values()]
        latencies = sorted(row["latency_ms"] for row in passed)
        p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else 0
        summary[candidate] = {
            "calls": len(rows),
            "successes": len(passed),
            "mean_wer": round(statistics.mean(row["wer"] for row in passed), 6) if passed else None,
            "critical_span_accuracy": round(sum(hits) / len(hits), 6) if hits else None,
            "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
            "p95_latency_ms": latencies[p95_index] if latencies else None,
            "estimated_cost_usd": round(sum(row.get("estimated_cost_usd", 0) for row in passed), 8),
        }
    _atomic_json(output / "stt_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stt", "tts", "all"), default="all")
    parser.add_argument("--output")
    parser.add_argument("--authorized-total-usd", type=float, required=True)
    parser.add_argument("--authorized-per-call-usd", type=float, required=True)
    parser.add_argument("--deepgram-key", type=Path, default=DEFAULT_KEYS["deepgram"])
    parser.add_argument("--openai-key", type=Path, default=DEFAULT_KEYS["openai"])
    parser.add_argument("--google-key", type=Path, default=DEFAULT_KEYS["google_cloud"])
    parser.add_argument("--stt-candidate", action="append")
    args = parser.parse_args()
    if args.authorized_total_usd != 1.0 or args.authorized_per_call_usd != 0.05:
        raise SystemExit("refusing: this task is authorised only for $1.00 total / $0.05 per call")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output) if args.output else ROOT / "results" / stamp
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    keys = {"deepgram": args.deepgram_key, "openai": args.openai_key, "google_cloud": args.google_key}
    ledger = BudgetLedger(output / "budget_ledger.json", args.authorized_total_usd, args.authorized_per_call_usd)
    _atomic_json(output / "run.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                                        "stage": args.stage, "automatic_retries": 0,
                                        "corpus_sha256": index["reference_texts_sha256"]})
    if args.stage in ("stt", "all"):
        run_stt(manifest, index, output, ledger, keys, set(args.stt_candidate or []))
        summarise_stt(output)
    if args.stage in ("tts", "all"):
        run_tts(manifest, output, ledger, keys)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
