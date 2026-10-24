"""Materialize the frozen Russian STT corpus without network I/O.

The caller must provide the three pinned Piper model/config pairs. Model
download and licence review are intentionally outside this script so running it
cannot silently change provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
CORPUS = ROOT / "corpus"
MODEL_REVISION = "e9ef9dd26adb6a40a9b47198817cd8e96d01e7cb"

VOICES = (
    {
        "id": "piper_ru_RU_denis_medium",
        "model_stem": "ru_RU-denis-medium",
        "dataset": "OHF-Voice/voice-datasets",
        "dataset_license": "CC0-1.0",
        "model_card": (
            "https://huggingface.co/rhasspy/piper-voices/blob/"
            f"{MODEL_REVISION}/ru/ru_RU/denis/medium/MODEL_CARD"
        ),
        "model_sha256": "15fab56e11a097858ee115545d0f697fc2a316c41a291a5362349fb870411b0a",
        "config_sha256": "831c860dac0b5073eaa81610a0a638ec23d90a6cf8e5f871b4485c2cec3767c8",
    },
    {
        "id": "piper_ru_RU_dmitri_medium",
        "model_stem": "ru_RU-dmitri-medium",
        "dataset": "OHF-Voice/voice-datasets",
        "dataset_license": "CC0-1.0",
        "model_card": (
            "https://huggingface.co/rhasspy/piper-voices/blob/"
            f"{MODEL_REVISION}/ru/ru_RU/dmitri/medium/MODEL_CARD"
        ),
        "model_sha256": "f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb",
        "config_sha256": "667ef3117bc642c2892dff7690d8bdc8ca4228aeaa783b2dc1416df632855e0d",
    },
    {
        "id": "piper_ru_RU_ruslan_medium",
        "model_stem": "ru_RU-ruslan-medium",
        "dataset": "RUSLAN speech corpus",
        "dataset_license": "CC-BY-NC-SA-4.0",
        "model_card": (
            "https://huggingface.co/rhasspy/piper-voices/blob/"
            f"{MODEL_REVISION}/ru/ru_RU/ruslan/medium/MODEL_CARD"
        ),
        "model_sha256": "72a5f88e0b20928064eb45d88e1daa21f8af62d18613580d32cbb4aed48dcf7f",
        "config_sha256": "706a4fb17bc304abd07809b552deae615e64dcbffbfbd09854ba37ca59e88117",
    },
)

CONDITION_RECIPES = {
    "clean": "No acoustic post-processing before Opus encoding.",
    "casual_fast": "Deterministic atempo=1.12 post-processing.",
    "hesitation": "Lexical pauses/self-correction are encoded in the frozen reference text; no acoustic filter.",
    "quiet": "Deterministic volume=0.22 post-processing.",
    "cafe_noise_snr_10db": (
        "Deterministic algorithmic coloured-noise proxy, seed=20261023, mixed at measured 10 dB SNR; "
        "not a field recording."
    ),
    "ogg_opus_32k": "No acoustic filter; Telegram-like mono OGG/Opus CBR 32 kbps encoding.",
    "subdued": "Deterministic volume=0.62 and lowpass=4800 Hz post-processing.",
    "expressive": "Punctuation-driven synthesis from frozen text; no acoustic filter.",
}

VOICE_OVERRIDES = {
    # Medium Whisper QA found this pairing materially clearer while preserving
    # the four-cases-per-voice balance of the round-robin assignment.
    "numbers_dates": "piper_ru_RU_denis_medium",
    "names_terms": "piper_ru_RU_dmitri_medium",
    "self_correction": "piper_ru_RU_ruslan_medium",
}

SYNTHESIS_TEXT_OVERRIDES = {
    "names_terms": "На ревью обсудим лесс, скрам и карту причинно-следственных связей с Александром.",
    "code_switch": "Я обновил роудмэп, но deployment пока не запускай — сначала нужен регрэшн чек.",
    "telegram_codec": "Это проверка голосового сообщения после перекодирования в телеграмный Опус.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, input_text: str | None = None) -> None:
    subprocess.run(command, input=input_text, text=True, check=True)


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
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "duration_seconds": round(float(stream["duration"]), 6),
        "codec": stream["codec_name"],
        "sample_rate_hz": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _mix_noise_at_snr(source: Path, target: Path, *, snr_db: float, seed: int) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        assert params.nchannels == 1 and params.sampwidth == 2
        raw = reader.readframes(params.nframes)
    samples = list(struct.unpack(f"<{len(raw) // 2}h", raw))
    signal_rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    rng = random.Random(seed)
    state = 0.0
    noise: list[float] = []
    for _ in samples:
        white = rng.uniform(-1.0, 1.0)
        state = 0.93 * state + 0.07 * white
        noise.append(state)
    raw_noise_rms = math.sqrt(sum(value * value for value in noise) / len(noise))
    scale = noise_rms / raw_noise_rms
    mixed = [max(-32767, min(32767, round(signal + value * scale))) for signal, value in zip(samples, noise)]
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(struct.pack(f"<{len(mixed)}h", *mixed))


def _conditioned_input(master: Path, condition: str, temporary: Path, case_index: int) -> Path:
    if condition == "cafe_noise_snr_10db":
        _mix_noise_at_snr(master, temporary, snr_db=10.0, seed=20261023 + case_index)
        return temporary
    return master


def _opus_filter(condition: str) -> str | None:
    return {
        "casual_fast": "atempo=1.12",
        "quiet": "volume=0.22",
        "subdued": "volume=0.62,lowpass=f=4800",
    }.get(condition)


def materialize(models_dir: Path, piper_command: str) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    masters = CORPUS / "masters"
    telegram = CORPUS / "telegram"
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    masters.mkdir(parents=True)
    telegram.mkdir(parents=True)

    command_prefix = shlex.split(piper_command)
    voice_records = []
    for voice in VOICES:
        model = models_dir / f"{voice['model_stem']}.onnx"
        config = models_dir / f"{voice['model_stem']}.onnx.json"
        if _sha256(model) != voice["model_sha256"] or _sha256(config) != voice["config_sha256"]:
            raise RuntimeError(f"Pinned model/config hash mismatch: {voice['id']}")
        voice_records.append({key: value for key, value in voice.items() if key != "model_stem"})

    entries = []
    with tempfile.TemporaryDirectory(prefix="voice-corpus-") as temp_name:
        temp = Path(temp_name)
        for case_index, case in enumerate(data["stt_cases"]):
            voice = VOICES[case_index % len(VOICES)]
            if override := VOICE_OVERRIDES.get(case["id"]):
                voice = next(candidate for candidate in VOICES if candidate["id"] == override)
            model = models_dir / f"{voice['model_stem']}.onnx"
            config = models_dir / f"{voice['model_stem']}.onnx.json"
            master = masters / f"{case['id']}.wav"
            raw_master = temp / f"{case['id']}-raw.wav"
            derivative = telegram / f"{case['id']}.ogg"
            synthesis_text = SYNTHESIS_TEXT_OVERRIDES.get(case["id"], case["text"])
            _run(
                command_prefix
                + ["--model", str(model), "--config", str(config), "--output_file", str(raw_master)],
                input_text=f"{synthesis_text}\n",
            )
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(raw_master),
                    "-af",
                    "volume=0.85",
                    "-map_metadata",
                    "-1",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(master),
                ]
            )
            conditioned = _conditioned_input(
                master,
                case["condition"],
                temp / f"{case['id']}-conditioned.wav",
                case_index,
            )
            ffmpeg = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(conditioned)]
            audio_filter = _opus_filter(case["condition"])
            if audio_filter:
                ffmpeg += ["-af", audio_filter]
            ffmpeg += [
                "-map_metadata",
                "-1",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "libopus",
                "-application",
                "voip",
                "-frame_duration",
                "20",
                "-b:a",
                "32k",
                "-vbr",
                "off",
                str(derivative),
            ]
            _run(ffmpeg)
            entry = {
                "case_id": case["id"],
                "reference_text": case["text"],
                "critical": case["critical"],
                "voice_id": voice["id"],
                "condition": case["condition"],
                "condition_recipe": CONDITION_RECIPES[case["condition"]],
                "clean_master": _probe(master),
                "telegram_derivative": _probe(derivative),
            }
            if synthesis_text != case["text"]:
                entry["pronunciation_only_synthesis_text"] = synthesis_text
            entries.append(entry)

    reference_payload = [
        {"id": case["id"], "text": case["text"], "critical": case["critical"]}
        for case in data["stt_cases"]
    ]
    reference_sha = hashlib.sha256(
        json.dumps(reference_payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    index = {
        "version": 1,
        "language": "ru-RU",
        "frozen_on": "2026-10-23",
        "reference_texts_sha256": reference_sha,
        "generator": {
            "engine": "piper-tts",
            "engine_version": "1.8.0",
            "engine_license": "GPL-3.0-or-later",
            "ffmpeg_version": "9.0.1",
            "clean_master_normalization": "uniform volume=0.85 headroom; no other acoustic processing",
            "model_repository": "rhasspy/piper-voices",
            "model_revision": MODEL_REVISION,
            "network_during_synthesis": False,
        },
        "licence_boundary": (
            "Corpus is for the project's non-commercial benchmark. Files made with the Ruslan model remain "
            "subject to CC-BY-NC-SA-4.0 attribution/share-alike constraints; do not relicense the corpus as a whole."
        ),
        "voices": voice_records,
        "entries": entries,
    }
    (CORPUS / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--piper-command", default="piper")
    args = parser.parse_args()
    materialize(args.models_dir, args.piper_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
