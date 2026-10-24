#!/usr/bin/env python3
"""Render the private single-owner pilot config without exposing secrets.

Run as root on the VPS. Secrets are read from separate mode-0600 files and
never accepted through argv, environment variables, stdout, or image layers.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import stat
import tempfile

SECRET_NAMES = (
    "openrouter_api_key",
    "openai_audio_api_key",
    "google_cloud_tts_api_key",
    "telegram_bot_token",
    "telegram_owner_chat_id",
)


def _read_secret(directory: pathlib.Path, name: str) -> str:
    path = directory / name
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"secret path is not a regular file: {path}")
    if info.st_mode & 0o077:
        raise PermissionError(f"secret file must be mode 0600 or stricter: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"secret file is empty: {path}")
    return value


def build_config(secret_dir: pathlib.Path) -> dict:
    values = {name: _read_secret(secret_dir, name) for name in SECRET_NAMES}
    try:
        owner_id = int(values["telegram_owner_chat_id"])
    except ValueError as exc:
        raise ValueError("telegram_owner_chat_id must contain one integer") from exc

    dashboard_secret = secret_dir / "dashboard_access_token"
    if dashboard_secret.exists():
        dashboard_token = _read_secret(secret_dir, "dashboard_access_token")
    else:
        dashboard_token = secrets.token_urlsafe(36)
        fd = os.open(dashboard_secret, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(dashboard_token + "\n")

    openrouter_key = values["openrouter_api_key"]
    purposes = {
        "dialogue": {
            "model": "mistralai/mistral-large-4-0",
            "allowedProviders": ["mistral/eu"],
            "activeProvider": "mistral/eu",
            "maxInputTokensEstimated": 6000,
            "maxOutputTokens": 1600,
            "maxCallsPerTurn": 2,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 20,
            "totalTimeoutSeconds": 90,
            "maxCostUsdPerCall": 0.02,
            "maxCostUsdPerDay": 0.30,
        },
        "vision": {
            "model": "openai/gpt-6.1-sol",
            "allowedProviders": ["azure/eu"],
            "activeProvider": "azure/eu",
            "maxInputTokensEstimated": 6000,
            "maxOutputTokens": 400,
            "maxCallsPerTurn": 1,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 30,
            "totalTimeoutSeconds": 90,
            "maxCostUsdPerCall": 0.01,
            "maxCostUsdPerDay": 0.10,
        },
        "image_generation": {
            "transport": "images_api",
            "model": "bytedance-seed/seedream-5-0-flash",
            "resolution": "1K",
            "aspectRatio": "2:3",
            "maxImages": 1,
            "allowedReferenceAssets": ["/data/context/appearance/reference.jpg"],
            "maxInputTokensEstimated": 6000,
            "maxOutputTokens": 0,
            "maxCallsPerTurn": 1,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 60,
            "totalTimeoutSeconds": 180,
            "maxCostUsdPerCall": 0.03,
            "maxCostUsdPerDay": 0.12,
        },
        "memory_consolidation": {
            "model": "anthropic/claude-haiku-5.5",
            "allowedProviders": ["google-vertex/europe"],
            "activeProvider": "google-vertex/europe",
            "maxInputTokensEstimated": 3000,
            "maxOutputTokens": 800,
            "maxCallsPerWrite": 1,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 15,
            "totalTimeoutSeconds": 30,
            "maxCostUsdPerCall": 0.01,
            "maxCostUsdPerDay": 0.10,
        },
        "relationship_analysis": {
            "model": "anthropic/claude-haiku-5.5",
            "allowedProviders": ["google-vertex/europe"],
            "activeProvider": "google-vertex/europe",
            "maxInputTokensEstimated": 2000,
            "maxOutputTokens": 256,
            "maxCallsPerTurn": 1,
            "connectTimeoutSeconds": 5,
            "firstByteTimeoutSeconds": 15,
            "totalTimeoutSeconds": 30,
            "maxCostUsdPerCall": 0.005,
            "maxCostUsdPerDay": 0.05,
        },
    }

    return {
        "llm": {
            "provider": "openrouter",
            "openrouter": {
                "apiKey": openrouter_key,
                "model": "mistralai/mistral-large-4-0",
                "baseUrl": "https://openrouter.ai/api/v1",
            },
            "vision": {"provider": "openrouter", "model": "openai/gpt-6.1-sol"},
        },
        "outboundPolicy": {
            "version": 1,
            "traceRetentionDays": 30,
            "aggregateRetentionDays": 180,
            "openrouter": {
                "requireAccountPromptLoggingDisabled": True,
                "requireAccountZdr": True,
                "requireNonZeroKeyCreditLimit": True,
            },
            "purposes": purposes,
        },
        "stt": {"provider": "openai"},
        "openaiAudio": {
            "apiKey": values["openai_audio_api_key"],
            "model": "gpt-transcribe",
            "language": "ru",
            "promptProfile": "v2",
            "maxCostUsdPerCall": 0.01,
            "maxCostUsdPerDay": 0.10,
        },
        "tts": {"provider": "google_cloud"},
        "googleCloudTts": {
            "apiKey": values["google_cloud_tts_api_key"],
            "region": "eu",
            "voice": "ru-RU-Chirp3-HD-Zephyr",
            "audioEncoding": "OGG_OPUS",
            "maxCostUsdPerCall": 0.012,
            "maxCostUsdPerDay": 0.50,
        },
        "web": {"host": "0.0.0.0", "port": 7788, "accessToken": dashboard_token},
        "tools": {"runCommand": False},
        "channels": {
            "telegram": {
                "token": values["telegram_bot_token"],
                "allowedUsers": [owner_id],
                "replyInKindVoice": "auto",
                "voiceReplyMaxChars": 350,
            }
        },
        "memory": {
            "consolidation": True,
            "overnightConsolidation": False,
            "overnightArchiveDays": 0,
        },
        "storage": {
            "eventsRetentionDays": 90,
            "turnsRetentionDays": 365,
            "autoVacuumOnPrune": True,
        },
        "skills": {
            "image": {"provider": "openrouter"},
            "openrouter": {
                "apiKey": openrouter_key,
                "model": "bytedance-seed/seedream-5-0-flash",
            },
        },
        "selfie": {
            "enabled": True,
            "schedule": [],
            "chatId": owner_id,
            "model": "bytedance-seed/seedream-5-0-flash",
            "maxDaily": 4,
            "proactiveProbability": 0.15,
        },
        "proactive": {"enabled": True, "chatId": owner_id, "maxDaily": 6},
        "heartbeat": {"enabled": False, "intervalSec": 60, "alertChatId": None},
        "agent": {"language": "ru", "autoCompactThreshold": 0, "verbose": False},
        "companion": {
            "companionName": "Сара",
            "companionAge": 28,
            "companionCountry": "RU",
            "companionRegion": "Moscow",
            "companionOccupation": "product designer",
        },
        "user": {"timezone": "Europe/Moscow"},
        "persona": {"timezone": "Europe/Moscow"},
        "quota": {"diskMb": 2048, "dailyMessages": 100, "dailyPhotos": 4},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret-dir", type=pathlib.Path, default=pathlib.Path("/etc/herandhim/secrets"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/srv/sara/data/herandhim.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing config: {args.output}")

    config = build_config(args.secret_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".herandhim.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.chown(temporary, 10001, 10001)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"pilot config written securely to {args.output} (secret values not logged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

