# Synthetic Russian voice corpus provenance

## Boundary

This corpus contains only locally synthesized speech. It contains no owner audio,
recording supplied by another natural person, or voice clone. Synthesis performs
no network I/O; model files are supplied separately and verified against pinned
SHA-256 values before generation.

## Engine

- Piper TTS `1.8.0`, GPL-3.0-or-later, used only as an offline build tool.
- FFmpeg `9.0.1` for deterministic mono OGG/Opus derivatives.
- Piper model repository revision:
  `e9ef9dd26adb6a40a9b47198817cd8e96d01e7cb`.
- Model weights are not vendored into this repository.

## Voices and dataset licences

| Voice | Dataset provenance | Dataset licence | Use in corpus |
|---|---|---|---|
| `ru_RU-denis-medium` | OHF-Voice `voice-datasets` | CC0-1.0 | 3 cases |
| `ru_RU-dmitri-medium` | OHF-Voice `voice-datasets` | CC0-1.0 | 5 cases |
| `ru_RU-ruslan-medium` | RUSLAN speech corpus | CC BY-NC-SA 4.0 | 4 cases |

The exact model/config hashes and model-card URLs are frozen in
`corpus/index.json`. The Ruslan-derived files are used only for this
non-commercial research benchmark and retain the attribution,
non-commercial-use and share-alike boundary of CC BY-NC-SA 4.0. Do not infer
that the complete corpus is available under the upstream repository's AGPL
licence or relicense it as a homogeneous dataset.

`irina` was deliberately excluded because its Piper model card labels the
training-dataset licence as unknown. macOS system voices were also excluded
because the installed host exposed only one Russian voice and did not satisfy
the three-voice independence criterion.

## Acoustic conditions

Every case has a clean Piper WAV master with one uniform `volume=0.85` headroom
step to prevent raw-model clipping, plus a mono 48 kHz OGG/Opus 32 kbps
derivative. The derivative applies only its declared condition. The café-noise
case uses deterministic algorithmic coloured noise at measured 10 dB SNR; it is
a controlled proxy, not a field recording. This avoids introducing unlicensed
background speakers or an irreproducible external recording.

Piper is not reliably multilingual inside a Russian voice. Three cases therefore
store an explicit `pronunciation_only_synthesis_text` in the index, transliterating
`LeSS`, `Scrum`, `roadmap`, `regression check` and `Opus` while leaving the frozen
reference text and semantics unchanged. `deployment` remains Latin because the
selected voice pronounced it more clearly that way. These overrides are part of
the frozen recipe, not hidden normalization by the future STT scorer.

## QA boundary

`corpus/qa_machine_listening.json` records a local faster-whisper `medium`
transcription of every final derivative and binds each result to the derivative
SHA-256. It is a machine-listening gross-content check, not a human judgement of
naturalness, persona fit or provider quality. The future blinded TTS benchmark
still requires human ratings; this STT input freeze does not claim them.
