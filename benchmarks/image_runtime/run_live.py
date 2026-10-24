"""One-call live gate for the production OpenRouter image adapter.

Synthetic canonical identity only. The production ImageGenerator enforces the
exact Seedream route, one reference, one output, durable cost reservation and
zero retry/fallback. The key is read from the existing macOS Keychain item.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
REFERENCE = ROOT / "assets" / "samples" / "realistic-european.jpg"
REFERENCE_SHA256 = "456dfe60752870674eca8ded6b3689a3046ed7aa3085b27f0338f24f84ad56fc"
BASE_URL = "https://openrouter.ai/api/v1"
KEYCHAIN_ACCOUNT = "HerAndHim-D029"
KEYCHAIN_SERVICE = "com.herandhim.openrouter.benchmark"
MODEL = "bytedance-seed/seedream-5-0-flash"
CALL_CAP_USD = 0.05

sys.path.insert(0, str(ROOT))

from herandhim.core.image_gen.generator import ImageGenerator  # noqa: E402
from herandhim.core.outbound.costs import CostLedger  # noqa: E402
from herandhim.core.outbound.policy import OutboundPolicy  # noqa: E402
from herandhim.core.outbound.trace import OutboundTrace  # noqa: E402


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
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "HerAndHim-image-runtime/1"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return {
        "remaining": float(data.get("limit_remaining") or 0),
        "usage": float(data.get("usage") or 0),
        "expires_at": data.get("expires_at"),
    }


def _policy(reference: Path) -> OutboundPolicy:
    purposes: dict[str, Any] = {
        "dialogue": {
            "model": "mistralai/mistral-large-4-0",
            "allowedProviders": ["mistral/eu"],
            "activeProvider": "mistral/eu",
            "maxInputTokensEstimated": 12_000,
            "maxOutputTokens": 1_600,
            "maxCallsPerTurn": 4,
            "maxCostUsdPerCall": CALL_CAP_USD,
            "maxCostUsdPerDay": CALL_CAP_USD,
        },
        "vision": {
            "model": "openai/gpt-6.1-sol",
            "allowedProviders": ["azure/eu"],
            "activeProvider": "azure/eu",
            "maxInputTokensEstimated": 6_000,
            "maxOutputTokens": 400,
            "maxCallsPerTurn": 1,
            "maxCostUsdPerCall": CALL_CAP_USD,
            "maxCostUsdPerDay": CALL_CAP_USD,
        },
        "image_generation": {
            "transport": "images_api",
            "model": MODEL,
            "resolution": "1K",
            "aspectRatio": "2:3",
            "maxImages": 1,
            "allowedReferenceAssets": [str(reference.resolve())],
            "maxInputTokensEstimated": 6_000,
            "maxOutputTokens": 0,
            "maxCallsPerTurn": 1,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 60,
            "totalTimeoutSeconds": 180,
            "maxCostUsdPerCall": CALL_CAP_USD,
            "maxCostUsdPerDay": CALL_CAP_USD,
        },
        "memory_consolidation": {
            "model": "anthropic/claude-haiku-5.5",
            "allowedProviders": ["google-vertex/europe"],
            "activeProvider": "google-vertex/europe",
            "maxInputTokensEstimated": 3_000,
            "maxOutputTokens": 800,
            "maxCallsPerWrite": 1,
            "maxCostUsdPerCall": CALL_CAP_USD,
            "maxCostUsdPerDay": CALL_CAP_USD,
        },
        "relationship_analysis": {
            "model": "anthropic/claude-haiku-5.5",
            "allowedProviders": ["google-vertex/europe"],
            "activeProvider": "google-vertex/europe",
            "maxInputTokensEstimated": 2_000,
            "maxOutputTokens": 256,
            "maxCallsPerTurn": 1,
            "maxCostUsdPerCall": CALL_CAP_USD,
            "maxCostUsdPerDay": CALL_CAP_USD,
        },
    }
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


def run(output_dir: Path, authorized_usd: float) -> dict[str, Any]:
    if authorized_usd != CALL_CAP_USD:
        raise ValueError(f"authorized budget must equal {CALL_CAP_USD:.2f} USD")
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("canonical reference checksum mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    api_key = _key()
    before = _key_status(api_key)
    if before["remaining"] < authorized_usd:
        raise RuntimeError("key has insufficient remaining authority")
    trace_path = output_dir / "trace.jsonl"
    generator = ImageGenerator(
        api_key=api_key,
        base_url=BASE_URL,
        model=MODEL,
        provider="openrouter",
        outbound_policy=_policy(REFERENCE),
        outbound_trace=OutboundTrace(trace_path),
        cost_ledger=CostLedger(output_dir / "costs.sqlite3"),
    )
    prompt = (
        "Create one new photorealistic candid smartphone selfie of the same adult "
        "synthetic woman in the canonical reference. Preserve recognizable facial "
        "identity, facial proportions, eye color, nose, lips, freckles, and natural "
        "skin texture. She is beside an apartment window in soft overcast daylight, "
        "wearing a simple dark green sweater, with a relaxed subtle smile. Natural "
        "handheld framing, anatomically correct hands, no text, no watermark."
    )
    started = time.monotonic()
    rows = generator.generate(
        prompt, n=1, reference_image=str(REFERENCE), model=MODEL,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    if len(rows) != 1 or not isinstance(rows[0].get("b64"), str):
        raise RuntimeError("adapter did not return exactly one inline image")
    image_bytes = base64.b64decode(rows[0]["b64"], validate=True)
    image_path = output_dir / "seedream-live.jpg"
    image_path.write_bytes(image_bytes)
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        dimensions = list(image.size)
        image_format = image.format
    after = _key_status(api_key)
    trace_text = trace_path.read_text(encoding="utf-8")
    trace_rows = [json.loads(line) for line in trace_text.splitlines()]
    reported_cost = sum(float(row.get("cost_usd") or 0) for row in trace_rows)
    key_delta = max(0.0, after["usage"] - before["usage"])
    accounted = max(reported_cost, key_delta)
    assertions = {
        "exactly_one_image": len(rows) == 1,
        "valid_image": bool(image_bytes) and image_format in {"JPEG", "PNG", "WEBP"},
        "portrait_2_to_3": abs(dimensions[0] / dimensions[1] - 2 / 3) < 0.02,
        "exact_model_traced": any(
            row.get("model") == MODEL and row.get("transport") == "images_api"
            for row in trace_rows
        ),
        "one_reference_traced": any(row.get("image_count") == 1 for row in trace_rows),
        "budget_within_authority": accounted <= authorized_usd + 1e-12,
        "zero_retry": all(row.get("retry_count", 0) == 0 for row in trace_rows),
        "trace_content_free": (
            prompt not in trace_text and api_key not in trace_text and "base64" not in trace_text
        ),
    }
    evidence = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "synthetic_only": True,
        "authorized_usd": authorized_usd,
        "reported_cost_usd": reported_cost,
        "observed_key_delta_usd": key_delta,
        "accounted_spend_usd": accounted,
        "latency_ms": latency_ms,
        "model": MODEL,
        "transport": "images_api",
        "reference_sha256": REFERENCE_SHA256,
        "image": {
            "path": str(image_path),
            "bytes": len(image_bytes),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "format": image_format,
            "dimensions": dimensions,
        },
        "trace_rows": len(trace_rows),
        "assertions": assertions,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-usd", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS / stamp
    evidence = run(output, args.authorized_usd)
    print(json.dumps({
        "status": evidence["status"],
        "accounted_spend_usd": evidence["accounted_spend_usd"],
        "image": evidence["image"]["path"],
        "evidence": str(output / "evidence.json"),
    }, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
