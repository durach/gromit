# Gromit

Gromit turns a meeting recording into a transcript that says who spoke each
line. It runs Whisper for the words and pyannote for speaker diarization,
entirely on your own machine — the audio and video never leave it, and no part
of the pipeline calls a transcription API. Gromit makes network calls only to
download model weights, once each: Whisper and pyannote from HuggingFace on
your first `transcribe`, and EasyOCR's detection and recognition weights from
EasyOCR's own host on your first `nametag`. Once those are cached everything
runs offline. (The one other thing that touches the network is optional and
yours: `tools/wsl-transcribe.sh` copies a recording over SSH to a GPU machine
*you* nominate, and `docs/PIPELINE.md`'s optional stage 2½ sends transcript
text to whatever LLM you point it at.)

It also does the unglamorous part that follows a transcript: finding the spans
a human should actually check, showing you the video clip for each one, and
folding your corrections into a glossary so the next meeting comes out better.

## What it does

- Transcribes audio (MP3, WAV, M4A, …) and video (MP4, MKV, MOV, WebM, AVI,
  FLV, WMV — audio is extracted with ffmpeg first).
- Labels speakers automatically (`Speaker 1`, `Speaker 2`, …) via diarization.
- Handles English, Ukrainian, and Russian; auto-detects by default.
- Uses your GPU when there is one — CUDA or Apple Silicon MPS — and falls back
  to CPU.
- Reads the on-screen participant names out of a Google Meet recording and
  attaches real names to caption cues (`gromit nametag`).
- Flags likely transcription errors and builds an offline review page with one
  video clip per flagged span (`gromit crosscheck`, `gromit review`).
- Keeps a per-project glossary of names and terms that ASR gets wrong, and
  merges reviewed corrections back into it (`gromit glossary-merge`).

## What it does not do

- **Not real-time.** Gromit processes finished files. There is no live or
  streaming mode.
- **`nametag` is Google Meet-specific.** It expects a Meet grid recording plus
  the WebVTT caption file Meet produces. It reads the name strip in the corner
  of each tile. Zoom, Teams, and other layouts are not supported.
- **Diarization needs a HuggingFace token** and you must accept the
  `pyannote/speaker-diarization-3.1` licence first. There is no bundled
  fallback that skips this.
- **Diarization gives you `Speaker 1`, not names.** Real names come either
  from `nametag` (Meet video) or from you.
- **Large models are slow on CPU.** The default `large-v3` on a CPU-only
  machine can take longer than the meeting itself. See `--model` and `--device`.
- **No cloud, no daemon, no telemetry.** No `gromit` command uploads your
  recording anywhere, which also means no server does the heavy lifting for
  you. (If you want a server to, `tools/wsl-transcribe.sh` will copy the file
  to a GPU host you name — that is you choosing, not Gromit.)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (any Python installer works, but the
  commands below use `uv`)
- ffmpeg — used to extract audio from video and to cut review clips
- A HuggingFace account and access token, for the pyannote diarization models

### Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### HuggingFace token

Pyannote's models are gated: you must accept their licence with your account
before the download works.

1. Create an account at [huggingface.co](https://huggingface.co).
2. Accept the licence at
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
3. Create a token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Make it visible to Gromit:

```bash
# Option 1: environment variable
export HF_TOKEN="your_token_here"

# Option 2: a .env file in the Gromit checkout
cp .env.example .env   # then set HF_TOKEN in it

# Option 3: HuggingFace CLI login
pip install huggingface_hub
huggingface-cli login
```

`HF_TOKEN` is the only environment variable Gromit reads — everything else is a
command-line flag. Gromit does load a `.env` (python-dotenv), but the search
starts where the `gromit` package was imported from and walks upward, so with
the editable install above `.env` belongs in the Gromit checkout; one dropped
into an unrelated working directory is ignored. Option 1 has no such caveat: an
exported shell variable is always read, and always wins over `.env`.

Only `transcribe` needs the token. `nametag`, `crosscheck`, `review`, and
`glossary-merge` do not touch pyannote.

## Install

```bash
git clone https://github.com/durach/gromit.git
cd gromit

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

The `[dev]` extra pulls in the test and lint tooling. It also pulls in
`socksio`, which you need if your shell sets a SOCKS proxy — see
[Troubleshooting](#troubleshooting).

On macOS, `uv pip install -e ".[vision]"` adds `ocrmac`, which lets `nametag`
use Apple Vision for OCR alongside EasyOCR. It is optional; without it
`nametag` runs EasyOCR only.

Check the install:

```bash
gromit --help
```

## Your first transcript

```bash
gromit transcribe meeting.mp4
```

The first run downloads the Whisper and pyannote models (several GB) and is
therefore much slower than later ones. When it finishes you get two files
beside the input:

- **`meeting.gromit.txt`** — the readable transcript, grouped into blocks of
  consecutive speech by the same speaker:

  ```
  Speaker 1:
  Thanks for joining. I want to go through the timeline before we decide anything.

  Speaker 2:
  Sure. The design phase slipped by about a week, but the rest still holds.

  Speaker 1:
  Good. Let's start there.
  ```

- **`meeting.gromit.json`** — the same transcript as structured data:
  per-segment start/end times, speaker, `avg_logprob`, and word-level timings
  with per-word probabilities. `gromit crosscheck` consumes this file; nothing
  else in the pipeline reads the `.txt`.

Useful first adjustments:

```bash
gromit transcribe meeting.mp4 --language uk        # skip auto-detection
gromit transcribe meeting.mp4 --speakers 3         # you know the head count
gromit transcribe meeting.mp4 --model small        # faster, less accurate
gromit transcribe meeting.mp4 --duration 120 -v    # first 2 minutes, verbose
```

Note that `transcribe` is a required subcommand — bare `gromit meeting.mp4`
does not work.

## Commands

### `gromit transcribe`

Audio/video in, speaker-attributed transcript out.

```
gromit transcribe [OPTIONS] [INPUT_FILES]...
```

Multiple files are concatenated in order and transcribed as one recording, so
a call split into parts keeps a single consistent speaker numbering:

```bash
gromit transcribe part1.mp4 part2.mp4 part3.mp4
gromit transcribe --from-file parts.txt -o full_day.txt
```

| Flag | Meaning |
| --- | --- |
| `-f`, `--from-file PATH` | Read the input list from a text file, one path per line |
| `-o`, `--output PATH` | Transcript path (default `<input>.gromit.txt`) |
| `-l`, `--language TEXT` | `en`, `uk`, `ru`, or `auto` (default `auto`) |
| `-m`, `--model TEXT` | `tiny`, `base`, `small`, `medium`, `large-v3` (default `large-v3`) |
| `-s`, `--speakers INTEGER` | Expected speaker count — improves diarization when you know it |
| `--device TEXT` | `auto`, `cuda`, `mps`, `cpu` (default `auto`) |
| `-v`, `--verbose` | Progress detail |
| `--duration FLOAT` | Process only the first N seconds — for trying things out |
| `--glossary PATH` | Glossary YAML whose canonical forms become Whisper hotwords (repeatable). **Read the warning in [the pipeline section](#the-transcript-quality-pipeline) before using this.** |

The `.gromit.json` is written alongside the `.txt` regardless of `-o`: an
explicit `-o transcript.txt` still produces `transcript.gromit.json`.

### `gromit nametag`

Google Meet recording plus Meet's caption file in, the same captions with real
speaker names out.

```
gromit nametag [OPTIONS] FOLDER
```

`FOLDER` must hold two files: the recording as an `.mp4`, and a **Google Meet
WebVTT caption file**. Meet does not create the caption file for you — when
Meet has recorded a meeting with captions on, open the recording in the Google
Drive player and download its caption track; that `.vtt` is the input here.
Gromit does not generate it and cannot run `nametag` without it: the cues are
what it labels.

There are two ways to point `nametag` at them.

**Name them explicitly** — the reliable form when the recording and its captions
do not share a stem, which is how Meet exports them:

```bash
gromit nametag \
  --video "/path/to/meetings/Team sync - Recording.mp4" \
  --vtt   "/path/to/meetings/Team sync - Recording-uk-asr.vtt" \
  --roster /path/to/meetings/roster.yaml
```

The two must be given together, and results are written beside the video (or
into `FOLDER`, if you also pass one).

**Or give a folder** and let it find them. How that picks, exactly (`cli.py`):

- **Video** — every `*.mp4` in `FOLDER`, sorted by name, first one wins.
- **Captions** — every `*.vtt` in `FOLDER` *except* `*.named.vtt` (which is
  `nametag`'s own output, so re-runs do not eat their own results), sorted by
  name, first one wins.
- Missing either one is an error: `need one .mp4 and one .vtt in <folder>`.
- The two stems need **not** match, and nothing warns you when a folder holds
  more than one candidate — the alphabetically first simply wins, silently. When
  that ambiguity matters, use `--video`/`--vtt` instead of relying on sort
  order. If
  you keep both `2026-01-15-board.vtt` and `2026-01-15-board-uk-asr.vtt` in the
  folder, `-` sorts before `.`, so the `-uk-asr.vtt` is the one used. Keep one
  `.mp4` and one non-`.named` `.vtt` per meeting folder and the rule never
  matters.

For every caption cue, Gromit samples frames from the video, reads the name
strip on the speaking participant's tile with OCR, votes across the frames, and
resolves the winner against your candidate names.

```bash
gromit nametag /path/to/meetings/2026-01-15-board \
  --roster /path/to/meetings/roster.yaml \
  --guest "Solomia Kravets"
```

Candidates come from `--roster` (people who attend regularly) and `--guest`
(one-offs). At least one of the two is required. Outputs:

- `<stem>.named.vtt` — the original cues with a `Name: ` prefix on each
- `<stem>.named.txt` — a flat readable form annotated `[HH:MM:SS] Name:`

| Flag | Meaning |
| --- | --- |
| `-g`, `--guest TEXT` | An occasional attendee's name (repeatable) |
| `--roster PATH` | `roster.yaml` of permanent members |
| `-v`, `--verbose` | Print the vote per cue instead of a progress bar |
| `--early-stop` / `--no-early-stop` | Stop sampling a cue once a roster name has a majority (default: on). Turn off for a full vote on every cue |
| `--keep-cache` | Keep the extracted frame cache after the run (also kept automatically when cues need review) |

On macOS with `ocrmac` installed, `nametag` reads each crop with both EasyOCR
and Apple Vision and takes the better match; elsewhere it uses EasyOCR alone.
Names that match nobody in your candidate list are kept verbatim rather than
snapped to the nearest roster entry — an unlisted attendee should never be
silently relabelled as someone who was there.

### `gromit crosscheck`

A transcript in, a list of spans worth a human's attention out.

```
gromit crosscheck [OPTIONS] GROMIT_JSON
```

```bash
gromit crosscheck meeting.gromit.json \
  --meet meeting-uk-asr.vtt \
  --glossary /path/to/meetings/glossary.yaml \
  -o meeting.flags.json
```

Three signals produce a flag, and the reason is recorded on each span:

- `divergence` — the span disagrees with Google Meet's own captions for the
  same stretch of time (asymmetric token containment). Requires `--meet`.
- `low_confidence` — a cluster of low-probability words, or a poor segment
  `avg_logprob`.
- `misheard_match` — the span contains a spelling listed under `misheard` in
  your glossary, so the canonical form is suggested automatically.

| Flag | Meaning |
| --- | --- |
| `-o`, `--output PATH` | Where to write `flags.json` (**required**) |
| `--meet PATH` | Google Meet WebVTT to compare against (optional; without it you get no `divergence` signal) |
| `--glossary PATH` | Glossary YAML for `misheard` matching (repeatable) |

It prints the span count and a breakdown by reason, e.g.
`Wrote meeting.flags.json — 34 spans ({'low_confidence': 21, 'divergence': 15, 'misheard_match': 6})`.

### `gromit review`

Flags in, an offline review page out.

```
gromit review [OPTIONS] FLAGS_JSON
```

```bash
gromit review meeting.flags.json \
  --video meeting.mp4 \
  --named meeting.named.vtt \
  -o review/
```

This ranks the spans (mishearings first, then divergence, then low
confidence), cuts a short ~480p clip for each one with ffmpeg, and writes
`review/index.html` plus `review/clips/NNN.mp4`. The page is self-contained
and opens straight from `file://` — no server, no network. Each row gives you
the clip, the transcript text, and a correction box; when you are done, the
page exports a `corrections.yaml` client-side.

| Flag | Meaning |
| --- | --- |
| `--video PATH` | The recording the flags refer to (**required**) |
| `-o`, `--output PATH` | Output `review/` directory (**required**) |
| `--named PATH` | A `nametag` `.named.vtt`, to label each span with who was speaking |
| `--limit INTEGER` | Only the top-N ranked spans (default: all) |

### `gromit glossary-merge`

Corrections in, an updated glossary out.

```
gromit glossary-merge [OPTIONS] CORRECTIONS
```

```bash
gromit glossary-merge ~/Downloads/corrections.yaml \
  --glossary /path/to/meetings/glossary.yaml
```

Takes the `corrections.yaml` the review page exported and folds it into the
glossary **in place**. The merge is a round-trip edit, so your comments and
formatting survive, and it is idempotent — running it twice reports everything
as unchanged rather than duplicating entries. Conflicting edits to the same
term are reported instead of silently applied.

| Flag | Meaning |
| --- | --- |
| `--glossary PATH` | The glossary to update in place (**required**) |

It prints each new entry and each new `misheard` spelling, then a summary line:
`Merged glossary.yaml — 2 new, 5 misheard added, 41 unchanged`.

## The transcript-quality pipeline

The four commands above are a loop. Each meeting you review makes the glossary
better, and a better glossary flags more of the next meeting's real errors.

**Stage 1 — transcribe.** Produce `<stem>.gromit.json`. On a CPU-only laptop
this is the slow step; `tools/wsl-transcribe.sh` will push the video to a
remote CUDA machine over SSH, run `gromit transcribe` there, pull the results
back and delete the remote copy. Point it at your own host with
`WSL_HOST=user@gpu-worker`.

> **Do not pass `--glossary` at this stage.** Measured on a real meeting:
> hotwords of *any* size — a handful of terms was as bad as a whole glossary — collapsed transcript
> coverage from 76% to 27% on some recordings, with whole segments dropped by
> Whisper's logprob and compression-ratio filters. The cause is not understood,
> so treat transcribe-time hotwords as unsafe. The glossary does its real work
> in stage 2, where `misheard` repair needs no hotwords at all.

**Stage 2 — crosscheck.** Compare the transcript against Google Meet's own
caption track and against the glossary's `misheard` lists, and write
`flags.json`. Meet's captions are a genuinely independent second engine, which
is what makes `divergence` informative.

**Stage 3 — review.** Turn the flags into a page of clips. Watching four
seconds of video is much faster than re-reading the transcript, and it is the
only way to settle what was actually said. Correct what is wrong, tick the
terms worth remembering, export.

**Stage 4 — merge.** Fold the export back into the glossary. Next meeting,
`crosscheck` recognises those mishearings on sight.

Worked end to end, using the example glossary in this repo. `$DIR` holds two
files you put there yourself: `2026-01-15-board.mp4` (the Meet recording) and
`2026-01-15-board-uk-asr.vtt` (Meet's caption track, downloaded from the Drive
player — `-uk-asr` is just this document's naming convention).

```bash
DIR=/path/to/meetings/2026-01-15-board
GLOSS=examples/glossary.yaml

# 1. transcribe (no --glossary; see the warning above)
gromit transcribe "$DIR/2026-01-15-board.mp4" --language uk

# 2. optional: real speaker names from the video
#    nametag finds the .mp4 and the .vtt in $DIR itself — here that is the
#    -uk-asr.vtt, the only non-.named .vtt in the folder
gromit nametag "$DIR" --roster examples/roster.yaml --guest "Solomia Kravets"

# 3. flag the spans worth checking
gromit crosscheck "$DIR/2026-01-15-board.gromit.json" \
  --meet "$DIR/2026-01-15-board-uk-asr.vtt" \
  --glossary "$GLOSS" \
  -o "$DIR/2026-01-15-board.flags.json"

# 4. build the review page and open it
gromit review "$DIR/2026-01-15-board.flags.json" \
  --video "$DIR/2026-01-15-board.mp4" \
  --named "$DIR/2026-01-15-board.named.vtt" \
  -o "$DIR/review/"
open "$DIR/review/index.html"

# 5. after exporting corrections.yaml from the page
gromit glossary-merge ~/Downloads/corrections.yaml --glossary "$GLOSS"
```

The full runbook — per-stage checks, thresholds, what to do when a stage
misbehaves — is in [`docs/PIPELINE.md`](docs/PIPELINE.md). For how the pieces
fit together internally, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## File formats

Gromit ships with no built-in knowledge of your project, your people, or your
vocabulary. Both files below are yours, live wherever you keep your meetings,
and are passed in explicitly.

### Glossary — [`examples/glossary.yaml`](examples/glossary.yaml)

A list of terms ASR gets wrong. Each entry carries the `canonical` spelling, a
`category` (`person` / `company` / `product` / `term`, which only sets hotword
priority), an optional human `note`, and the `misheard` spellings the engine
has actually produced for it. `crosscheck` matches against `misheard` and
suggests the canonical form; `glossary-merge` appends new mishearings as you
confirm them.

The example file is commented line by line — read it rather than a schema.

### Roster — [`examples/roster.yaml`](examples/roster.yaml)

Who `nametag` should expect to see on screen. It has one section, `permanent`:
the people who attend regularly, offered as candidates in every video. A
one-off attendee does not go in the file — pass them on the command line with
`--guest "Name"` (repeatable) for that run only, which keeps a guest from one
meeting out of the candidate list of an unrelated one. `nametag` needs at least
one candidate from `--roster`, `--guest`, or both.

Write full names, not what Meet shows: Meet truncates long names on the tile
("Yaroslav Vyshn…") and Gromit recovers the full form by front-anchored prefix
match. Anyone not listed is kept verbatim and flagged, never snapped to the
closest listed name.

## Development

```bash
uv run task test        # fast unit tests
uv run task test-slow   # + slow tests (model loading; needs HF_TOKEN)
uv run task test-e2e    # end-to-end tests only
uv run task test-all    # everything

uv run task lint        # ruff check
uv run task fix         # ruff check --fix
uv run task format      # ruff format
uv run task check       # lint + test
```

Or with pytest directly: slow tests need `--run-slow` and e2e tests need
`--run-e2e`; both are skipped otherwise.

```bash
pytest -v
pytest -v --run-slow
pytest tests/e2e/ -v --run-e2e
```

The e2e tests synthesise their own audio with edge-tts (English, Ukrainian and
Russian; single- and two-speaker), cache the fixtures locally, and run the
whole pipeline against them — so they need `HF_TOKEN` and a lot of patience.
See [`docs/E2E-TESTING-DESIGN.md`](docs/E2E-TESTING-DESIGN.md).

Development follows TDD: a failing test first, then the implementation.

## Troubleshooting

### `ffmpeg` not found

Video input, and every review clip, goes through ffmpeg. Install it (see
[Prerequisites](#prerequisites)) and make sure `ffmpeg -version` works in the
same shell you run `gromit` in.

### `DiarizationError` mentioning HF_TOKEN or licences

Two different causes wear the same error message. Read the underlying
exception before assuming which one you have.

1. **You really have not accepted the licence.** Visit
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   while logged in, accept, then confirm `HF_TOKEN` is exported in the shell
   that runs `gromit`.

2. **The underlying message says `Using SOCKS proxy, but the 'socksio' package
   is not installed`.** Then it is neither the token nor the licence: your
   shell sets `ALL_PROXY` to a SOCKS proxy and `httpx` needs an optional
   dependency to use it. `socksio` is a dev dependency, so
   `uv pip install -e ".[dev]"` fixes it. Alternatives: run with the proxy
   unset for one command (`ALL_PROXY= all_proxy= gromit transcribe …`, which
   requires your host to reach `huggingface.co` directly), or run offline with
   `HF_HUB_OFFLINE=1` if the models are already cached.

### `CUDA driver version is insufficient for CUDA runtime version`

`torch.cuda.is_available()` can return `True` on a machine whose NVIDIA driver
is older than the CUDA runtime PyTorch was built against — common under WSL,
where PyTorch targets CUDA 12.x but the Windows driver exposes something
older.

Gromit already guards against this: device detection runs a real tensor
operation on CUDA and MPS before selecting them, and falls back to CPU if that
fails. If you still see the error, force the device explicitly with
`--device cpu`, or update the NVIDIA driver on the host.

### Warnings during a run

Some warning noise from dependencies is expected and harmless:

- **`PySoundFile failed. Trying audioread instead.`** and the librosa
  deprecation notice — these came from loading video containers through
  librosa's deprecated fallback, and are fixed: video is now extracted to a
  temporary WAV with ffmpeg first. If you still see them, you are on an old
  build.
- **`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected`** and **`std(): degrees of
  freedom is <= 0`** — both come from inside pyannote (its model loading uses
  `weights_only=False`, and short audio segments trip the std warning). They
  cannot be fixed here and will go away when pyannote updates.
- **`torchcodec is not installed correctly so built-in audio decoding will
  fail`**, sometimes with a `libtorchcodec_core*.so` or `libnvrtc.so` traceback.
  `torchcodec` is pulled in by pyannote, declares no version constraint against
  torch, and so routinely resolves to a build compiled for a different torch.
  The traceback looks alarming but is not fatal — pyannote falls back to another
  decoder, and transcription and diarization both work normally.

### `nametag` says "no candidate names"

`nametag` will not guess. Pass `--roster roster.yaml`, `--guest "Name"`, or
both.

### Transcription is very slow

Check which device was chosen with `-v`. On CPU, drop to `--model medium` or
`--model small`, or move the job to a CUDA machine with
`tools/wsl-transcribe.sh`.

## License

MIT — see [`LICENSE`](LICENSE).
