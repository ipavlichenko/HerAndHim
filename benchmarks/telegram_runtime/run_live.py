"""Three-message live Telegram transport smoke for the owner-only MVP.

The runner reads the bot token and owner chat id from macOS Keychain, sends
exactly one text, one existing OGG/Opus voice note and one existing JPEG, and
never invokes a model/provider.  Each attempt is journalled before network I/O
so a partial run is evidence, not a reason to retry automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
BOT_ACCOUNT = "sara_herandhim_bot"
BOT_SERVICE = "com.herandhim.telegram.bot"
OWNER_SERVICE = "com.herandhim.telegram.owner"
VOICE = ROOT / "benchmarks/voice/results/20261023T_voice_live/tts_blind/B_remaining_montage.ogg"
IMAGE = ROOT / "benchmarks/image_runtime/results/20261024T053016Z/seedream-live.jpg"
EXPECTED_MESSAGES = 3


def _secret(service: str) -> str:
    process = subprocess.run(
        [
            "security", "find-generic-password", "-a", BOT_ACCOUNT,
            "-s", service, "-w",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = process.stdout.strip()
    if not value:
        raise RuntimeError(f"missing Keychain item for {service}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _run(output_dir: Path, authorized_messages: int) -> dict[str, Any]:
    if authorized_messages != EXPECTED_MESSAGES:
        raise ValueError(f"authorized message count must equal {EXPECTED_MESSAGES}")
    for path in (VOICE, IMAGE):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing smoke fixture: {path}")

    token = _secret(BOT_SERVICE)
    owner_text = _secret(OWNER_SERVICE)
    if not owner_text.isdecimal():
        raise RuntimeError("owner chat id in Keychain is invalid")
    owner_id = int(owner_text)

    from herandhim.channels.telegram_bot import TelegramBot
    from herandhim.core.telegram_transport import TelegramTransportStore

    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = output_dir / "evidence.json"
    evidence: dict[str, Any] = {
        "status": "RUNNING",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "bot_username": BOT_ACCOUNT,
        "authorized_messages": authorized_messages,
        "attempts": [],
        "model_calls": 0,
        "provider_calls": 0,
        "automatic_retries": 0,
        "fallbacks": 0,
        "fixtures": {
            "voice": {"bytes": VOICE.stat().st_size, "sha256": _sha256(VOICE)},
            "image": {"bytes": IMAGE.stat().st_size, "sha256": _sha256(IMAGE)},
        },
    }
    _atomic_json(evidence_path, evidence)

    transport = TelegramTransportStore(output_dir / "telegram_transport.sqlite3")
    bot = TelegramBot(
        session_manager=None,
        token=token,
        allowed_users=[owner_id],
        transport_store=transport,
    )
    app = bot.build_application()
    await app.initialize()

    async def attempt(kind: str, send) -> None:
        row = {
            "ordinal": len(evidence["attempts"]) + 1,
            "kind": kind,
            "status": "attempted",
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        evidence["attempts"].append(row)
        _atomic_json(evidence_path, evidence)
        try:
            delivered = await send()
            if delivered is False:
                raise RuntimeError(f"{kind} transport returned false")
        except Exception as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            evidence["status"] = "FAIL"
            _atomic_json(evidence_path, evidence)
            raise
        row["status"] = "delivered"
        row["finished_at_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(evidence_path, evidence)

    try:
        await attempt(
            "text",
            lambda: bot.send_message(
                owner_id,
                "[HerAndHim smoke 1/3] Текстовый канал готов.",
            ),
        )
        await attempt(
            "voice",
            lambda: bot.send_voice(
                owner_id,
                str(VOICE),
                "[HerAndHim smoke 2/3] Голосовой канал.",
                origin="explicit",
            ),
        )
        await attempt(
            "image",
            lambda: bot.send_photo(
                owner_id,
                str(IMAGE),
                "[HerAndHim smoke 3/3] Канал изображений.",
                origin="explicit",
            ),
        )
    finally:
        await app.shutdown()

    evidence["status"] = "PASS" if all(
        row["status"] == "delivered" for row in evidence["attempts"]
    ) and len(evidence["attempts"]) == EXPECTED_MESSAGES else "FAIL"
    evidence["finished_at_utc"] = datetime.now(UTC).isoformat()
    _atomic_json(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-messages", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or RESULTS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    evidence = asyncio.run(_run(output_dir, args.authorized_messages))
    print(json.dumps({
        "status": evidence["status"],
        "attempts": [row["status"] for row in evidence["attempts"]],
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
