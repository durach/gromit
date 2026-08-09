# CLAUDE.md

## Project Overview

Gromit is a privacy-first, local AI-powered transcription tool that converts audio and video files into speaker-attributed text. All processing occurs on the user's machine. The only network access in `src/` is first-run model download — Whisper and pyannote from HuggingFace, EasyOCR's weights from its own host (see `docs/ARCHITECTURE.md` §1). Nothing else is transmitted; keep it that way, and if you add a network call, update that table and the README's opening paragraph in the same change.

Docs for humans: [`README.md`](README.md) is the user manual,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the internals reference, and
[`docs/PIPELINE.md`](docs/PIPELINE.md) is the end-to-end runbook. Read the
architecture doc before changing anything structural — it is written against the
current source and names the real modules and functions.

## Quick Reference

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Tasks (via taskipy)
uv run task test        # Run fast tests
uv run task test-slow   # Include slow tests (requires HF_TOKEN)
uv run task test-e2e    # Run e2e tests only
uv run task test-all    # Run all tests
uv run task lint        # Check linting
uv run task fix         # Auto-fix lint issues
uv run task format      # Format code
uv run task check       # Lint + test

# Run CLI — audio transcription
gromit transcribe <file.mp3>                    # Single file
gromit transcribe <file1.mp4> <file2.mp4>       # Multiple files (concatenated)
gromit transcribe <file.mp4> --glossary g.yaml  # Glossary hotwords + <stem>.gromit.json

# Run CLI — transcript-quality pipeline (see docs/PIPELINE.md for the full flow)
gromit crosscheck <stem>.gromit.json --meet <stem>-uk-asr.vtt --glossary g.yaml -o <stem>.flags.json
gromit review <stem>.flags.json --video <stem>.mp4 --named <stem>.named.vtt -o review/
gromit glossary-merge corrections.yaml --glossary g.yaml

# Run CLI — video name-tagging
gromit nametag <meeting-dir> --guest "Name"     # <stem>.named.vtt + .named.txt

# Optional: transcribe on a remote CUDA host instead of locally
tools/wsl-transcribe.sh <meeting-dir> --language uk
```

## Project Structure

```
src/gromit/
├── cli.py              # Typer CLI: transcribe, crosscheck, review, glossary-merge, nametag
├── config.py           # TranscriptionConfig, ModelSize, Device enums
├── exceptions.py       # GromitError hierarchy
├── orchestrator.py     # Audio-pipeline coordination (for `transcribe`)
├── glossary.py         # Per-project glossary load/validate (hotword_list, misheard_index)
├── glossary_merge.py   # `glossary-merge`: fold corrections back (ruamel round-trip)
├── audio/              # Audio loading (ffmpeg extraction + soundfile decode)
├── transcription/      # Whisper backends (faster-whisper)
├── diarization/        # Speaker ID (pyannote)
├── alignment/          # Temporal alignment
├── output/             # Text formatting + .gromit.json writer
├── crosscheck/         # `crosscheck`: align two engines → flags.json (normalize, align, signals)
├── review/             # `review`: rank spans, cut clips, render self-contained HTML
├── nametag/            # Video name-tagging: tile detection, name-strip crop, OCR, roster match
└── utils/              # Device detection
```

Two independent pipelines share the package: an **audio** one (`transcribe`,
using faster-whisper + pyannote) and a **video-only** one (`nametag`, using
OpenCV + OCR). They do not call each other.

## Transcript-quality pipeline (`crosscheck` / `review` / `glossary-merge`)

Closes the loop from raw recording to reviewed corrections that improve the next
meeting's ASR. **Full runbook: [`docs/PIPELINE.md`](docs/PIPELINE.md); internals:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §§4–5.** Four stages, each
independently useful:

1. **Transcribe** — `gromit transcribe --glossary` → `<stem>.gromit.{txt,json}`
   (JSON carries word-level timing + confidence). Runs locally; on a CPU-only
   machine `tools/wsl-transcribe.sh` can push the job to any ssh-reachable CUDA
   host instead (push→run→pull→delete). Note the measured warning in
   `docs/PIPELINE.md` about transcribe-time hotwords.
2. **`gromit crosscheck`** — aligns the gromit transcript with the Google Meet
   `-uk-asr.vtt` by time overlap and flags spans → `flags.json`. Signals:
   `divergence` (asymmetric token-containment vs Meet), `low_confidence` (low-word
   cluster / avg_logprob), `misheard_match` (known `misheard` → auto-suggests canonical).
3. **`gromit review`** — ranks spans, cuts a ~480p clip each (ffmpeg re-encode),
   renders a self-contained `file://` HTML page with per-row correction inputs +
   a client-side `corrections.yaml` export.
4. **`gromit glossary-merge`** — folds corrections back into the glossary via
   ruamel round-trip (comment-preserving, idempotent, conflict-detecting).

Glossary is per-project (`--glossary PATH`); gromit has no hardcoded project
knowledge. Loader: `glossary.py`. Thresholds: `crosscheck/signals.py::Thresholds`
— starting points tuned on one real meeting, meant to be retuned, not defended.

## Video name-tagging (`nametag`)

`gromit nametag` reads each Google Meet participant's on-screen name from the
video and attributes Meet's own caption cues to those names. It is a
**video-only pipeline, parallel to audio** — no pyannote, no diarization, the
audio track is never opened. Inputs are a meeting folder holding `<stem>.mp4`
plus Meet's `<stem>.vtt`, and a candidate list from `--roster` and/or `--guest`;
outputs are `<stem>.named.vtt` (`Name: ` prefix) and `<stem>.named.txt`
(`[HH:MM:SS] Name:` annotation).

Three stages, all in `src/gromit/nametag/` (driver: `run.py::attribute_meeting`):

- **Stage 1** — per-frame tile/layout detection, classical CV, no model
  (`heuristic.py::detect`, types in `schema.py`).
- **Stage 2a** — crop the bottom-left **name strip** of the speaker's tile
  (`name_region.py::name_band`, `frame_speaker.py::speaker_tile`).
- **Stage 2b** — OCR the strip and resolve it to an identity:
  - `name_ocr.py` — EasyOCR (Latin-only) reader + white-text preprocessing;
    `keep_leftmost_cluster` drops stray non-name text (logos/role labels).
  - `vision_ocr.py` — Apple Vision via the optional `ocrmac` extra; on macOS it
    is the stronger engine, and `name_resolve.py` matches on the **best of both**,
    falling back gracefully to EasyOCR-only where `ocrmac` is unavailable.
  - `roster.py` — front-anchored `difflib` match against `roster.yaml`
    (permanent members) plus `--guest` names, threshold 0.80, with an open-set
    verbatim fallback for people who are on neither list.
- **Stage 3** — cue attribution: sample ≥7 frames per cue (≥1 per 500 ms), read
  the on-screen name in each, and vote. Modules: `vtt.py`, `sampling.py`
  (`cue_frame_times`/`extract_frames_at`), `attribution.py`
  (`vote_cue`/`attribute_cue`), `vtt_output.py`.

A `research/` tree of validation tooling (bake-off harnesses, OCR contact
sheets, a correction tool) was removed once the detector choice it existed to
settle was settled and nothing shipped invoked it.

## Key Patterns

- **TDD**: Write failing test first, then implement
- **Explicit subcommand**: `gromit transcribe FILE` runs the audio pipeline. Bare `gromit FILE` does not work — the subcommand is required.
- **Pluggable backends**: BaseTranscriber/BaseDiarizer abstract classes
- **Sequential GPU processing**: Load one model at a time to manage VRAM
- **Lazy heavy imports**: torch/cv2/easyocr are imported inside command bodies so `--help` stays instant

## Testing

- **Fast tests**: Run by default with `pytest`
- **Slow tests**: Marked with `@pytest.mark.slow`, require `--run-slow` flag
- **E2E tests**: Marked with `@pytest.mark.e2e`, require `--run-e2e` flag
- Pyannote tests: Also require `HF_TOKEN` environment variable

## E2E Tests

End-to-end tests use TTS-generated audio (edge-tts) to validate the full pipeline.

- Located in `tests/e2e/`
- Skipped by default, run with `--run-e2e`
- Requires completed pipeline implementation
- Requires HF_TOKEN for pyannote diarization
- Audio fixtures cached in `tests/e2e_fixtures/` (gitignored)
- Tests English, Ukrainian, and Russian with single and two-speaker scenarios

See [`docs/E2E-TESTING-DESIGN.md`](docs/E2E-TESTING-DESIGN.md) for full architecture.

## Dependencies

- **Runtime**: typer, rich, torch, faster-whisper, pyannote.audio, numpy, soundfile, librosa, opencv-python-headless, easyocr, pyyaml, ruamel.yaml
- **Optional (macOS, nametag)**: ocrmac (Apple Vision OCR — better than EasyOCR on these crops), installed via the `vision` extra
- **Dev**: pytest, pytest-cov, ruff, edge-tts, pydub, taskipy, socksio
- **System**: ffmpeg (audio/video extraction, frame extraction, review clips)

Audio loading is **ffmpeg + soundfile**, not librosa: video containers and `.m4a`
are extracted to a temporary 16 kHz mono WAV with ffmpeg, and everything else is
decoded by soundfile. `librosa` is imported lazily in one place only —
`audio/processor.py` — to resample a file that ffmpeg did not already normalise.

## External Requirements

- HuggingFace token for pyannote models (one-time setup)
- Accept model licenses at huggingface.co/pyannote/speaker-diarization-3.1

## Troubleshooting

### Slow tests fail with `DiarizationError: ... HF_TOKEN ... license ...`

If the underlying message is `Using SOCKS proxy, but the 'socksio' package is not installed`, the root cause is the SOCKS proxy in `$ALL_PROXY` plus a missing optional `httpx` dep — *not* an HF token / license issue. `socksio>=1.0` is now a dev dep; `uv pip install -e ".[dev]"` is enough to fix it. Always read the underlying error rather than trusting the `DiarizationError` wrapper.

Alternatives if you don't want `socksio`:
- Per-run: `ALL_PROXY= all_proxy= uv run task test-slow` (host must reach `huggingface.co` directly).
- Offline: `HF_HUB_OFFLINE=1 uv run task test-slow` if models are already cached.
