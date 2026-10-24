"""Offline validator for the voice benchmark manifest.

This script performs no network I/O and never reads credentials.  Credentialed
provider/account checks are a separate, explicitly authorised live stage.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path(__file__).with_name("manifest.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["language"] == "ru-RU"
    assert data["materialization"]["status"] == "frozen"
    index_path = path.parent / data["materialization"]["corpus_index"]
    assert index_path.is_file()
    assert len(data["stt_candidates"]) >= 2
    assert len(data["tts_candidates"]) >= 2
    assert len(data["stt_cases"]) >= 10
    assert len(data["tts_cases"]) >= 8
    assert len({c["id"] for c in data["stt_cases"]}) == len(data["stt_cases"])
    assert len({c["id"] for c in data["tts_cases"]}) == len(data["tts_cases"])
    budget = data["hard_budget"]
    assert 0 < budget["stt_usd_per_call"] <= budget["total_usd"]
    assert 0 < budget["tts_usd_per_call"] <= budget["total_usd"]
    assert budget["automatic_retries"] == 0
    serialised = json.dumps(data, ensure_ascii=False).lower()
    for forbidden in ("api_key", "telegram_token", "owner_audio", "personal_data"):
        assert forbidden not in serialised
    print(f"PASS: {len(data['stt_cases'])} STT cases, {len(data['tts_cases'])} TTS cases, frozen offline corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
