# Architecture

This is a reference for people changing Gromit — where the code lives, what
flows between the pieces, and why several non-obvious choices are the way they
are. For how to *use* the commands, read [`README.md`](../README.md); for the
operational runbook of the quality loop, read [`PIPELINE.md`](PIPELINE.md).

---

## 1. Overview and design principles

Gromit is four largely independent subsystems behind one Typer CLI
(`src/gromit/cli.py`):

| Subsystem | Command | Input | Output |
| --- | --- | --- | --- |
| Audio pipeline | `transcribe` | audio/video | `<stem>.gromit.txt` + `<stem>.gromit.json` |
| Nametag (video) | `nametag` | Meet `.mp4` + Meet `.vtt` | `<stem>.named.vtt` + `<stem>.named.txt` |
| Crosscheck | `crosscheck` | `.gromit.json` (+ Meet `.vtt`, glossary) | `flags.json` |
| Review / merge | `review`, `glossary-merge` | `flags.json` + video; `corrections.yaml` | `review/index.html`; updated glossary |

They are not layered on top of each other. `crosscheck` reads the audio
pipeline's JSON, and `review` optionally reads nametag's `.named.vtt`, but each
runs as its own process and each can be used alone.

### Design principles

**Local only.** No transcription API, no telemetry, no upload. Every network
access in `src/` is a first-run model download, three of them:

| Where | What it fetches | From | Triggered by |
| --- | --- | --- | --- |
| `src/gromit/transcription/faster_whisper.py` (`WhisperModel(...)`) | Whisper weights for the chosen `--model` | HuggingFace Hub, via `huggingface_hub` inside faster-whisper | first `transcribe` per model size |
| `src/gromit/diarization/pyannote.py` (`Pipeline.from_pretrained`) | `pyannote/speaker-diarization-3.1` (gated, needs `HF_TOKEN`) | HuggingFace Hub | first `transcribe` |
| `src/gromit/nametag/name_ocr.py` (`easyocr.Reader(...)`) | EasyOCR detection + recognition weights | EasyOCR's own download host, not HuggingFace | first `nametag` |

All three cache to disk and are not repeated. Everything else in `src/` —
ffmpeg, CTranslate2, OpenCV, Apple Vision, the generated `review/index.html`
(self-contained, no CDN reference) — runs on the machine and opens no socket.

Outside `src/` there are two opt-in exceptions, and both belong to the user:
`tools/remote-transcribe.sh` rsyncs a recording over SSH to a GPU host named by
`GROMIT_CUDA_WORKER`, and [`PIPELINE.md`](PIPELINE.md)'s stage 2½ has a human hand
transcript text to an LLM of their choosing. Neither is invoked by any `gromit`
command.

**Pluggable backends.** ASR and diarization sit behind the abstract classes
`BaseTranscriber` (`src/gromit/transcription/base.py`) and `BaseDiarizer`
(`src/gromit/diarization/base.py`). The dataclasses they exchange —
`TranscriptSegment`, `Word`, `SpeakerSegment` — are the pipeline's real
contract; the concrete backends are replaceable. See
[§6 Extension points](#6-extension-points).

**Sequential GPU model loading, to bound VRAM.** `Orchestrator.process()` never
holds the Whisper model and the pyannote pipeline at the same time: it
transcribes, calls `_clear_gpu_memory()`, diarizes, then clears again. Both
models are large; loading them concurrently is the difference between running
and OOM-ing on a consumer GPU. That is why the pipeline is a strict sequence of
steps that each fully consume their input rather than a streaming design.

**Lazy imports at the CLI boundary.** `cli.py` imports only `typer` and
`gromit.config` at module level. `Orchestrator`, `nametag.run`,
`crosscheck.core` and `review.core` are imported inside their command
functions, so `gromit nametag --help` does not pay for torch, cv2 and easyocr —
and does not print their startup chatter.

**Anything project-specific is a file you pass in.** Gromit has no built-in
knowledge of your people or vocabulary. The glossary (`--glossary`) and the
roster (`--roster`) are per-project YAML, loaded by `src/gromit/glossary.py`
and `src/gromit/nametag/roster.py`. See
[`examples/glossary.yaml`](../examples/glossary.yaml) and
[`examples/roster.yaml`](../examples/roster.yaml).

### Repository layout

```
src/gromit/
├── cli.py                  Typer app: transcribe, nametag, crosscheck, review, glossary-merge
├── config.py               TranscriptionConfig, ModelSize, Device
├── exceptions.py           GromitError hierarchy
├── orchestrator.py         Audio-pipeline coordination
├── glossary.py             Glossary load/validate → hotword_list(), misheard_index()
├── glossary_merge.py       Fold corrections back into the glossary (ruamel round-trip)
├── audio/processor.py      Decode/extract/concatenate → float32 16 kHz mono
├── transcription/          base.py (interface + dataclasses), faster_whisper.py
├── diarization/            base.py (interface + dataclasses), pyannote.py
├── alignment/temporal.py   Overlap-based speaker attribution
├── output/                 formatter.py (.txt), json_writer.py (.gromit.json)
├── nametag/                Video-only speaker attribution (Stages 1–3)
├── crosscheck/             Two-engine comparison → flags.json
├── review/                 Ranking, clip cutting, self-contained HTML page
└── utils/                  device.py (detection), progress.py (ETA + Rich columns)
```

---

## 2. The audio pipeline

```
cli.transcribe
   └─ TranscriptionConfig
        └─ Orchestrator.process()
             1. audio/processor.py      paths        → np.ndarray  (float32, 16 kHz, mono)
             2. transcription/          ndarray      → list[TranscriptSegment]
                (clear GPU memory)
             3. diarization/            ndarray      → list[SpeakerSegment]
                (clear GPU memory)
             4. alignment/temporal.py   both lists   → list[AlignedSegment]
             5. output/formatter.py     aligned      → str  (written to .gromit.txt)
                output/json_writer.py   aligned      → dict (written to .gromit.json)
```

### The config object

`src/gromit/config.py` holds everything the run needs:

- `ModelSize` — `TINY`, `BASE`, `SMALL`, `MEDIUM`, `LARGE_V3` (values are the
  faster-whisper model names).
- `Device` — `AUTO`, `CUDA`, `MPS`, `CPU`. `AUTO` is resolved by
  `utils/device.py::resolve_device`, which does not trust
  `torch.cuda.is_available()`: it allocates a one-element tensor on the device
  and falls back to CPU if that raises. A WSL host whose NVIDIA driver predates
  the CUDA runtime PyTorch was built against reports CUDA as available and then
  fails on first use, so availability alone is not a usable signal.
- `TranscriptionConfig` — `input_paths`, `output_path`, `language`,
  `model_size`, `device`, `num_speakers`, `verbose`, `max_duration`,
  `from_file_path`, `glossary_paths`, plus two derived properties:
  `effective_output_path` (`<stem>.gromit.txt`, or `<stem>_combined.gromit.txt`
  for multi-file input) and `json_output_path`, which shares the txt's base name
  so `-o X.txt` and `-o X.gromit.txt` both yield `X.gromit.json` — never a
  doubled `.gromit.gromit`.

The `.gromit` infix exists because a meeting folder accumulates artifacts from
several subsystems; `<stem>.gromit.txt` must not collide with nametag's
`<stem>.named.txt`.

### Step 1 — audio loading (`src/gromit/audio/processor.py`)

`AudioProcessor.load()` returns a float32 mono array at 16 kHz — the only audio
representation the rest of the pipeline knows.

- Video containers (`.mp4 .mkv .avi .webm .mov .flv .wmv`) and `.m4a` are
  extracted with ffmpeg to a temporary PCM WAV already at 16 kHz mono, then read
  with `soundfile`. Going through ffmpeg rather than librosa's audioread
  fallback removed a class of decode warnings and a lot of wall time.
- Other formats are read directly with `soundfile`; librosa is used only for
  resampling when the file's rate is not 16 kHz.
- `load_multiple()` concatenates several inputs through ffmpeg's concat demuxer
  and returns a single array, so a call recorded in three parts gets one
  consistent speaker numbering rather than three independent ones.
- `get_file_boundaries()` uses `ffprobe` to build `[(filename, start_offset)]`
  for that concatenated timeline; the formatter uses it to re-split the output
  per file and restart timestamps at zero in each section.
- `is_valid_audio()` rejects silence (RMS below 0.001) with `AudioLoadError`
  before any model loads.

### Step 2 — transcription (`src/gromit/transcription/`)

`FasterWhisperTranscriber` wraps faster-whisper's `WhisperModel`
(`compute_type="float16"` on CUDA, `int8` otherwise; MPS falls back to the CPU
backend, which is what faster-whisper supports). It transcribes with
`beam_size=5`, `vad_filter=True` and `word_timestamps=True`, and returns
`list[TranscriptSegment]` where each segment carries `start`, `end`, `text`,
`avg_logprob` and `words: list[Word(w, start, end, p)]` — per-word text, timing
and probability. Those word probabilities are not decoration: they are the
`low_confidence` signal in [§4](#4-crosscheck).

`progress_callback(progress, audio_position)` is invoked per segment; the
orchestrator feeds it into `utils/progress.py::SpeedTracker`, which measures
real throughput (audio seconds per wall-clock second) and only starts
displaying an ETA once it has seen at least 10 seconds of audio. The measured
speed ratio is then reused as the initial estimate for the diarization bar,
which has no natural progress signal of its own.

**The hotword budget.** Whisper's decoder context is 448 tokens
(`n_text_ctx`) — prompt plus generated text, an architectural limit.
faster-whisper's `get_prompt()` caps hotwords and previous-text context at
`448 // 2 - 1 = 223` tokens *each, independently*, but both live in that same
448-token prompt. A glossary large enough to hit its own cap therefore produces
a 450-token prompt, and CTranslate2 fails the entire run with "The maximum
decoding length must be > 0" — minutes in, once previous-text context has filled
up. `HOTWORD_TOKEN_BUDGET = 221` is the largest hotword allowance that keeps the
total at or under 448, and `_fit_hotwords()` truncates on *term* boundaries
(counting with faster-whisper's own tokenizer) so every surviving hotword is a
whole word, warning about what it dropped.

That machinery works, but transcribe-time hotwords are still not recommended —
see the warning in the README. The glossary earns its keep in `crosscheck`,
where `misheard` repair needs no hotwords at all.

### Step 3 — diarization (`src/gromit/diarization/`)

`PyannoteDiarizer` loads `pyannote/speaker-diarization-3.1` via
`Pipeline.from_pretrained` with `HF_TOKEN` from the environment, moves it to the
resolved device, and returns `list[SpeakerSegment(start, end, speaker)]` with
opaque labels (`SPEAKER_00`, …). Two details worth knowing before you touch it:

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is set around the `from_pretrained` call
  and restored afterwards. PyTorch 2.6 flipped `weights_only` to default-on and
  pyannote's checkpoints do not load under it.
- `diarize()` handles both pyannote output shapes — the older `itertracks()` API
  and the newer `DiarizeOutput.speaker_diarization` — and raises
  `DiarizationError` on anything else, rather than silently returning no
  speakers.

Any failure during construction is wrapped in `DiarizationError` with advice
about the token and licence. That wrapper is lossy: when the real cause is
something else (a SOCKS proxy without `socksio`, for example) the message still
talks about tokens. Read the chained exception, not the wrapper.

### Step 4 — alignment (`src/gromit/alignment/temporal.py`)

`TemporalAligner.align()` is deliberately simple: for each `TranscriptSegment`,
pick the `SpeakerSegment` with the largest time overlap; if nothing overlaps at
all, fall back to the speaker whose segment boundary is nearest the transcript
segment's midpoint; with no speakers at all, `"UNKNOWN"`. The result is
`list[AlignedSegment(start, end, speaker, text, avg_logprob, words)]` — the
transcript with a speaker label attached, word data carried through untouched.

There is no resegmentation: Whisper's segmentation wins, and diarization only
labels it. That keeps the text identical to what the ASR produced, which is what
`crosscheck` later compares.

### Step 5 — output (`src/gromit/output/`)

- `formatter.py::TextFormatter.format()` maps `SPEAKER_00 → Speaker 1`, groups
  consecutive segments by the same speaker, and emits
  `[HH:MM:SS] Speaker N:` blocks. With more than one `file_boundaries` entry it
  emits a `--- filename ---` header per input file and rebases timestamps to that
  file's start.
- `json_writer.py::build_transcript_json()` produces the `.gromit.json` payload:
  `language`, `model`, `hotwords_from` (the glossary paths used), and `segments`
  with `start`, `end`, `speaker`, `text`, `avg_logprob` and the full `words`
  array. It is written with `ensure_ascii=False`, so Cyrillic stays readable.

The `.txt` is for humans. The `.gromit.json` is the machine-readable artifact —
`crosscheck` reads it and nothing in the pipeline reads the `.txt`.

### Optional: remote GPU

`tools/remote-transcribe.sh` pushes a meeting folder to a CUDA host over SSH
(`GROMIT_CUDA_WORKER=user@host`, required), runs `gromit transcribe` there, pulls the results
back and deletes the remote copy. It is a convenience wrapper around the same
CLI, not a different code path.

---

## 3. The `nametag` video pipeline

**`nametag` is a video-only pipeline that runs parallel to the audio one. It
does not use pyannote, does not diarize, and never touches the audio track.** It
reads each Google Meet participant's name as it is rendered on screen and
attributes Meet's own caption cues to those names. You can run `nametag` on a
recording you never transcribe, and `transcribe` on a recording you never
nametag.

Why do it this way at all? Diarization gives you `Speaker 1`, not
`Yaroslav Vyshnevetsky`. Meet already draws the answer in the corner of every
tile, and it is right by construction — no voice-print enrolment, no mapping
step, no confusion between two people with similar voices.

The trade-off: it is Meet-specific (tile geometry and the bottom-left name
strip), it needs the WebVTT caption file Meet produces, and it depends on the
speaker actually being on screen.

```
cli.nametag  →  nametag/run.py::attribute_meeting
   vtt.py            Meet .vtt              → list[Cue(index, start, end, text)]
   for each cue:
     sampling.py     cue_frame_times        → list[float]           (≥7 times, ≤0.5 s apart)
     sampling.py     extract_frames_at      → list[(time, Path)]    (ffmpeg, cached per ms)
     for each frame  (make_frame_reader):
       heuristic.py  detect(frame)          → FrameResult(tiles, layout)     Stage 1
       frame_speaker speaker_tile(result)   → Tile | None
       name_region   name_band(box, w, h)   → normalized Box                 Stage 2a
       name_resolve  resolve_name(crop, …)  → NameMatch | None               Stage 2b
     attribution.py  vote_cue(matches)      → CueResult(name, votes, frames_used)  Stage 3
   vtt_output.py     cues + names           → <stem>.named.vtt, <stem>.named.txt
```

### Stage 1 — tile and layout detection (`nametag/heuristic.py`, `nametag/schema.py`)

`detect(frame_bgr) -> FrameResult` is classical CV — no training, no downloaded
weights. It trims near-black letterbox/pillarbox borders
(`geometry.py::active_canvas_px`), then in order:

1. A **pillarbox guard**: symmetric black bands on both sides mean a portrait
   speaker, not a screen share, so the PIP branch is skipped.
2. **Structural right-PIP detection** — the sole PIP classifier. It profiles the
   rightmost column for a tile flush right, letterboxed above and below, with
   shared content to its left. Face detection is deliberately *not* used here:
   Haar misses small camera tiles and hallucinates faces on slides, and either
   error used to force a wrong verdict.
3. **Face-based full-frame**: no structural PIP, but a Haar face anywhere →
   `FULL_FRAME` covering the whole frame.
4. **Content fallback**: no face but the canvas is not blank (mean > 8) →
   `FULL_FRAME`, kind `AVATAR` (a camera-off participant's coloured initial
   disk).
5. Dark or blank frame → no tiles.

`schema.py` is the output contract: `Tile(kind, box, confidence)` where `box` is
`(x, y, w, h)` **normalized to 0.0–1.0**, so it is resolution-independent;
`TileKind` is `CAMERA` or `AVATAR`; `derive_layout()` infers `Layout`
(`FULL_FRAME` / `SCREEN_SHARE_PIP` / `GALLERY` / `UNKNOWN`) from the tile set.
Tile detection is deliberately classical CV rather than a learned model. A
learned alternative was prototyped, lost the bake-off to the heuristic on these
recordings, and has since been removed rather than left in as an unused stub.

`frame_speaker.py::speaker_tile()` then picks the active speaker — and its rule
is one line: return the tile **only if there is exactly one**. Recordings of the
kind this targets show one participant at a time, so a single tile is
unambiguously the speaker.
Zero tiles (a pure shared slide) or two or more (a gallery) make the frame
abstain, because a wrong name is worse than no name and cue-level voting will
recover the cue from the clean frames.

### Stage 2a — the name strip (`nametag/name_region.py`)

Meet renders the participant name as white text at the bottom-left of a tile.
`name_band(tile_box, frame_w, frame_h)` returns that sub-rectangle, sized to
*contain* the label rather than hug it: height is a fraction of tile height
clamped to 24–64 px (Meet's font height is near-constant and the source is
1080p), width is the full tile width capped at 560 px. Surplus video inside the
band is harmless — the white-text masking in Stage 2b removes it.

### Stage 2b — OCR and roster resolution (`name_ocr.py`, `vision_ocr.py`, `name_resolve.py`, `roster.py`)

`name_ocr.py` runs EasyOCR. Two decisions are load-bearing:

- **Latin-only reader (`["en"]`).** Meet display names are commonly Latin-script
  even for non-English speakers, and enabling the Cyrillic model made the recogniser substitute
  homoglyphs into Latin names (`Kravets` → `Кгаvеtѕ`), which broke matching
  outright.
- **White-text isolation first.** `isolate_white_text()` thresholds on luminance
  (>170) and closes with a small ellipse, lifting the glyphs off the video
  behind them; if that yields nothing, OCR retries on the raw colour crop.

`keep_leftmost_cluster()` (shared by both OCR engines) drops fragments separated
from the name by a horizontal gap wider than 12% of the crop — a t-shirt logo, a
role label. The Meet name is always left-anchored, so the leftmost run is the
name.

`vision_ocr.py` is the Apple Vision adapter (macOS, via the optional `ocrmac`
extra). It is graceful by construction: if `ocrmac` cannot be imported,
`vision_available()` is `False` and `recognize_vision()` returns an empty
reading, so the pipeline degrades to EasyOCR-only rather than failing. Both
engines return the same `NameReading(text, confidence)`.

`name_resolve.py::resolve_name()` is the "best of both": where Vision is
available it runs **both** engines and keeps the higher-scoring roster match,
with Vision tried first so it wins score ties (it is the stronger engine on
these crops); elsewhere EasyOCR alone. It returns `NameMatch | None` — `None`
only when no engine produced any text.

`roster.py::match_name(reading, candidates, threshold=0.80)` turns a reading
into an identity, and two properties matter:

- **Front-anchored similarity.** `_prefix_similarity` compares the *shorter*
  string against the same-length prefix of the longer one, via
  `difflib.SequenceMatcher`. Meet truncates long names from the right
  (`Yaroslav Vyshn…`), and stray text the crop caught after the name also lands
  on the right (`Mykola H LOGO`), so the first letters are the reliable part
  and the tail is not. A reading shorter than 4 characters scores 0.0 — too
  little lead to bind safely.
- **Open set with a verbatim fallback.** A reading at or above the threshold
  resolves to the canonical roster name; anything else is kept verbatim with
  `matched=False`. If `Nadiya Ivanets` drops into a meeting and is not on the
  roster, she appears as herself and is flagged for review — she is never
  snapped to the nearest listed member. Silently relabelling a stranger as a
  regular attendee would be an invisible, unrecoverable error.

`load_roster()` parses `roster.yaml` into `Roster(permanent)` — a single list of
regular attendees, and nothing else. `cli.py` builds its candidate list as
`load_roster(roster).permanent` plus any `--guest` names, so `--guest` is the
mechanism for scoping a one-off attendee to a single run.

An earlier design also carried a `temporary:` block keyed by video stem, with a
`candidates_for(stem)` accessor. It was never wired into the CLI, so a
`temporary:` block silently did nothing; both have been removed rather than left
as a feature that looks available and is not. Occasional attendees go in with
`--guest`.

`rank_candidates()` exposes the ranked scores for debugging what `match_name`
collapsed into one verdict.

### Stage 3 — cue attribution (`vtt.py`, `sampling.py`, `attribution.py`, `vtt_output.py`, `run.py`)

`vtt.py::parse_vtt()` reads Meet's WebVTT into `list[Cue(index, start, end,
text)]` — timed text with **no speaker**, which is exactly the gap this stage
fills. Original line breaks survive so the writer can re-emit them, and
`parse_header()` preserves the `WEBVTT / Kind / Language` block.

`sampling.py::cue_frame_times(start, end)` guarantees *both* constraints: at
least 7 frames per cue and no gap wider than 0.5 s, endpoints included.
`extract_frames_at()` extracts each one with a seek-then-single-frame ffmpeg
call, caching by millisecond so overlapping cues share frames, and forcing
`-pix_fmt yuvj420p` because many Meet recordings are full-range YUV that the
default mjpeg encoder refuses. An unreadable frame is skipped, not fatal.

`attribution.py::vote_cue(matches)` collapses the per-frame `NameMatch` values
into one `CueResult`. Roster-matched readings bucket by canonical name;
off-roster verbatim readings merge when their leading letters agree at ≥0.80, so
OCR wobble does not split an unlisted guest's votes across three spellings.
Matched and unmatched buckets are kept separate even for the same person,
precisely so a guest is never absorbed into a prefix-similar roster name. The
winner is the largest bucket, ties broken by summed score; no usable frame gives
`"Unknown"`.

On top of that sits the **single-match rule**: if exactly one roster member
matched anywhere in the cue, they win even when sub-threshold garbles out-count
them. A name drawn over a busy graphic reads as many near-miss verbatims plus a
few clean hits — and those garbles are that same person. A genuine off-roster
guest produces no match at all, so they still surface verbatim; two or more
matched names fall back to the count. The rule never lowers the match threshold.

`attribute_cue(..., early_stop=True)` stops reading a cue's frames once a
roster-matched name holds a strict majority of the *planned* frames — the
default, and a large speed win on long meetings.

`run.py::attribute_meeting()` drives the whole thing and writes:

- `<stem>.named.vtt` — the cues unchanged except for a `Name: ` prefix on the
  first text line. A plain prefix, not the WebVTT `<v>` voice tag, because `<v>`
  renders invisibly in VLC.
- `<stem>.named.txt` — `[HH:MM:SS] Name:` blocks with consecutive same-name cues
  grouped, mirroring the audio pipeline's formatter.

Frames are scratch: `cache.py::cache_dir_for()` puts them under the system temp
dir keyed by `<stem>-<sha1[:8] of resolved path>` (so two meetings sharing a
stem do not collide), and the run deletes the cache afterwards **unless**
`--keep-cache` is set or some cue resolved to `Unknown` or to an off-roster
name — in which case the frames are exactly what a human needs to check it.
Errors leave the cache in place.

`run.py` also carries the quiet-startup machinery: `silence_warnings()` filters
the per-cue torch `pin_memory` warning, and `preload_quiet()` imports `av` and
`cv2` under an fd-2 redirect. Both bundle a `libavdevice`, and whichever loads
second makes the dynamic loader print an objc duplicate-class warning at the C
level, where a Python `warnings` filter cannot reach it.

One supporting module sits off the Stage-3 path: `segmentation.py`, which groups
frames into stable layout runs.

---

## 4. `crosscheck`

`crosscheck` finds spans a human should check, by asking two independent
questions: *did a second engine hear the same thing?* and *does the model itself
sound unsure?*

```
cli.crosscheck → crosscheck/core.py::run_crosscheck
   gromit_json.py  .gromit.json     → GromitTranscript(language, model, hotwords_from, segments)
   nametag/vtt.py  Meet .vtt        → list[Cue]
   align.py        overlap_fraction → sanity guard
   per segment:
     align.py      meet_text_for    → the Meet text overlapping this segment
     signals.py    segment_flags    → (reasons, suggestion)
   signals.py      merge_spans      → list[Span]
   output.py       write_flags_json → flags.json
```

**Alignment is pure interval overlap** (`crosscheck/align.py`) — no DTW. Both
transcripts are timestamped against the same recording, so there is nothing to
warp. The wrinkle is that Meet emits a *rolling* caption window whose cues
overlap each other, so one gromit segment typically spans several Meet cues;
`meet_text_for()` joins the unique overlapping cue texts in order.

Before comparing anything, `overlap_fraction()` checks that at least 20%
(`MIN_OVERLAP_FRACTION`) of Meet cues overlap some gromit segment. Below that,
you have almost certainly paired the wrong two files, and `CrosscheckError` says
so — much better than a flags file where every span is "divergent".

### The three signals (`crosscheck/signals.py`)

`segment_flags()` returns `(reasons, suggestion)` for one segment, in this order:

- **`divergence`** — `token_containment(gromit_text, meet_text) <
  divergence_max`. Containment is `|A∩B| / |A|` over normalized, filler-stripped
  tokens (`crosscheck/normalize.py`), and it is **asymmetric on purpose**: the
  gromit segment is narrower than the Meet caption window it aligns to, so a
  symmetric metric like Jaccard would score agreement as divergence merely
  because Meet's window carries extra words. The question being asked is "did
  the gromit segment's words show up in Meet at all?" Requires `--meet`.
- **`low_confidence`** — a *cluster* of at least `low_word_min` words below
  `word_p_min`, or a whole-segment `avg_logprob` below `seg_logprob_min`. One
  uncertain word is normal speech; two or more clustered is a signal. This is
  the only signal that works with no second engine.
- **`misheard_match`** — the segment (or its Meet text) contains a spelling
  listed under `misheard` in your glossary. This one also fills in
  `suggestion` with the canonical form, which pre-fills the review page's
  correction box.

Thresholds live in one place: **`crosscheck/signals.py::Thresholds`**, a frozen
dataclass with `divergence_max=0.5`, `word_p_min=0.4`, `low_word_min=2`,
`seg_logprob_min=-0.8`, `merge_gap=2.0`. They were tuned on one meeting and are
meant to be retuned — `run_crosscheck()` takes a `Thresholds` argument, so you
can sweep them without editing the module.

Flagged segments are then merged: `merge_spans()` joins spans no more than
`merge_gap` seconds apart into one, unioning reasons and concatenating both
sides' text, so a stretch of trouble becomes one review clip instead of six.

Output is `flags.json`: `{"spans": [{start, end, meet_text, gromit_text,
reasons, suggestion}, …]}`, UTF-8 with `ensure_ascii=False`.

---

## 5. `review` and `glossary-merge`

### `review` (`src/gromit/review/`)

```
cli.review → review/core.py::run_review
   flags.py    flags.json         → list[FlagSpan], sorted by rank_key
   names.py    .named.vtt         → list[NamedCue] → speaker per span
   clips.py    video + span       → review/clips/NNN.mp4
   diff.py     both texts         → (meet_html, gromit_html) with <mark>
   render.py   list[ReviewRow]    → review/index.html
```

**Ranking** (`review/flags.py`): `rank_key` sorts by `REASON_PRIORITY` —
`misheard_match` (0), `divergence` (1), `low_confidence` (2) — then by start
time. Mishearings come first because they are the ones with an actionable
suggestion attached and the ones that feed back into the glossary. `--limit N`
truncates after ranking, so you get the best N, not the first N.

**Clips** (`review/clips.py`): each span becomes a ~480p H.264 clip with 5 s of
padding on both sides, produced by a **re-encode**, not a stream copy. A copy
would have to cut on keyframes and could drift the boundary by seconds, which
defeats the point of the clip. `-ss` goes before `-i` for a fast input seek.
`extract_clip()` never raises: a failed clip yields `False`, the row renders
"clip unavailable", and the rest of the page still builds.

**Speaker labels** (`review/names.py`): if you pass `--named`, each span is
labelled with the `.named.vtt` cue that overlaps it most. This is the one place
the video and audio subsystems meet, and it is optional in both directions.

**Diff** (`review/diff.py`): `highlight()` whitespace-tokenizes both readings,
runs `difflib.SequenceMatcher` case-insensitively, and wraps non-equal tokens in
`<mark>` on both sides — so your eye goes straight to the disagreement. Both
sides are HTML-escaped first.

**The page** (`review/render.py`): `render_html(rows, title)` emits one
document with the CSS and JS inlined and no external references of any kind.
It opens from `file://` — no server, no network, nothing that could exfiltrate a
meeting. Each row is a `<video>` element pointing at its relative clip, the two
readings, and a correction form (canonical / heard-as / category / "add to
glossary"). The `Export corrections.yaml` button builds the YAML in the browser
from a `Blob` and downloads it; nothing is posted anywhere. The canonical field
is pre-filled from the span's `suggestion` and the heard-as field from the
gromit reading, so a confirmed mishearing is two clicks.

### `glossary-merge` (`src/gromit/glossary_merge.py`)

`load_corrections()` parses the exported `corrections.yaml` into
`list[Correction(canonical, heard, category)]`; `merge_corrections()` folds them
into the glossary **in place** and returns a `MergeSummary(added_entries,
added_misheard, unchanged)`.

The merge uses **ruamel.yaml in round-trip mode** (`YAML()` with
`preserve_quotes = True`), not `yaml.safe_load` + `dump`. The glossary is a
hand-curated file full of comments explaining *why* a term is there; a
load-and-dump cycle would silently delete every one of them and reorder the
rest. Round-trip mode preserves comments, ordering and quoting style, and new
values are written as `DoubleQuotedScalarString` so they match the existing
file's style.

Two behaviours matter operationally:

- **Idempotent.** A `heard` spelling already present under its canonical is
  counted as `unchanged`, not appended again. Re-running the same export is
  safe.
- **Conflict-detecting.** A `heard` string already mapped to a *different*
  canonical raises `GlossaryError` naming both. One spelling cannot mean two
  things, and guessing which one you meant would corrupt the glossary quietly.

### The glossary itself (`src/gromit/glossary.py`)

`load_glossary()` / `load_glossaries()` validate and merge one or more YAML
files into a `Glossary` of `GlossaryEntry(canonical, category, note, misheard)`.
Categories are limited to `term`, `person`, `company`, `product`. Duplicate
canonicals, and a `misheard` string claimed by two entries, are `GlossaryError`s
— within a file and across files alike.

The `Glossary` exposes exactly two views, one per consumer:

- `hotword_list()` — canonical forms ordered by `HOTWORD_CATEGORY_PRIORITY`
  (`person` → `company` → `product` → `term`), file order preserved within each
  group. Because the tail is what gets dropped at the token budget, this
  ordering decides what survives: names the model has never seen (a person, or a
  company like `Acme`) are worth biasing; generic terms are both less mangled
  and still repairable afterwards through their `misheard` lists.
- `misheard_index()` — `{lowercased misheard → canonical}`, consumed by
  `crosscheck`'s `misheard_match` signal.

---

## 6. Extension points

Two abstract base classes define the audio pipeline's replaceable parts. Both
live beside the dataclasses they exchange, and both are what you subclass to
swap in a different engine.

### `BaseTranscriber` — `src/gromit/transcription/base.py`

```python
class BaseTranscriber(ABC):
    def __init__(self, model_size: str, device: str, language: str) -> None: ...

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        progress_callback: ProgressCallback = None,
        hotwords: Sequence[str] | None = None,
    ) -> list[TranscriptSegment]: ...
```

- `__init__` is concrete and just stores `model_size`, `device`, `language` on
  `self`; `transcribe()` is the only abstract method.
- `audio` is float32, 16 kHz, mono. Nothing else is ever passed.
- `device` is one of `"cuda"`, `"mps"`, `"cpu"` — already resolved, never
  `"auto"`.
- `language` is a code such as `"en"`, `"uk"`, `"ru"`, or the literal `"auto"`;
  it is your backend's job to interpret `"auto"` as detection.
- `ProgressCallback = Callable[[float, float], None] | None`, called with
  `(progress 0–1, audio_position_seconds)`. Optional but recommended — the
  orchestrator uses it for the progress bar *and* for the speed estimate it
  hands to diarization.
- `hotwords` arrives most-important-first; a backend may drop the tail to fit
  its own prompt budget (see `FasterWhisperTranscriber._fit_hotwords`).
- The return value is `list[TranscriptSegment(start, end, text, confidence,
  avg_logprob, words)]` with `words: list[Word(w, start, end, p)]`. Word-level
  timing and probability are optional for the `.txt` output but **required** for
  the `low_confidence` signal in `crosscheck`; a backend that returns no words
  silently degrades that signal to `avg_logprob` alone.
- Setting a `detected_language` attribute is not part of the ABC, but
  `Orchestrator.transcript_json()` reads it (via `getattr`) to record the
  language in `.gromit.json`.

### `BaseDiarizer` — `src/gromit/diarization/base.py`

```python
class BaseDiarizer(ABC):
    def __init__(self, device: str, num_speakers: int | None = None) -> None: ...

    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        progress_callback: ProgressCallback = None,
    ) -> list[SpeakerSegment]: ...
```

- Same audio contract; same resolved `device` values.
- `num_speakers` is `None` for auto-detection, or the count the user passed with
  `--speakers`.
- `ProgressCallback` here is `Callable[[float], None] | None` — one argument, a
  fraction. It is *not* the same type as the transcriber's.
- Returns `list[SpeakerSegment(start, end, speaker)]`. Labels are opaque
  strings; `TextFormatter` renders anything matching `SPEAKER_(\d+)` as
  `Speaker N+1` and passes other labels through unchanged.

### Wiring a backend in

`Orchestrator` currently constructs `FasterWhisperTranscriber` and
`PyannoteDiarizer` by name (`orchestrator.py`, in `_transcribe_with_progress`
and `_diarize_with_progress`). There is no plugin registry — substituting a
backend means changing those construction sites, and if it is a GPU model,
keeping the sequential ordering and the `_clear_gpu_memory()` calls between the
steps intact.

The other subsystems have narrower seams rather than ABCs, but the same idea:
`attribute_cue()` takes a `frame_reader` callable `(path, candidates) ->
NameMatch | None` (see `nametag/run.py::make_frame_reader`), `run_crosscheck()`
takes a `Thresholds` instance, and `sampling.extract_frames_at()` takes an
injectable `_run` so tests do not shell out to ffmpeg.

---

## 7. Error model

Everything Gromit raises deliberately descends from one base
(`src/gromit/exceptions.py`), so a caller embedding the library can catch a
single type:

```
GromitError
├── AudioLoadError        file missing/unreadable, ffmpeg failed, audio is silent
├── TranscriptionError    transcription failed
├── DiarizationError      diarization failed (incl. HF token / licence problems)
├── GlossaryError         glossary missing, malformed, or self-contradictory
└── CrosscheckError       crosscheck input missing/malformed, or wrong file pairing
```

How they are used in practice:

- Each CLI command catches, prints `Error: <message>` in red, and exits 1 —
  no traceback for an expected failure. `transcribe` catches broadly
  (`Exception`) since backend libraries raise their own types; `crosscheck`,
  `review` and `glossary-merge` catch `GromitError` specifically.
- **Bad flag values count as expected failures.** `--model` and `--device` are
  declared as plain `str` rather than as the `ModelSize`/`Device` enums (the
  enum repr would bloat `--help`), so Typer validates neither. `cli.parse_choice()`
  does it instead, inside the same `try` as the input-file checks, raising
  `typer.BadParameter`: `gromit transcribe m.mp4 --model largev3` prints
  `Error: Invalid --model: 'largev3'. Choose one of: tiny, base, small, medium,
  large-v3` and exits 1. Keep any future enum-valued option on this path.
- **Wrapping is lossy, so always read the chained cause.** `DiarizationError`
  from `PyannoteDiarizer.__init__` blames the HF token and licence, because that
  is the common case — but the same wrapper fires when the real cause is a SOCKS
  proxy without `socksio` installed. The original exception is preserved with
  `raise ... from e`.
- **Some failures are deliberately not exceptions.** `extract_clip()` returns
  `False` and `recognize_vision()` returns an empty reading rather than raising:
  in a batch over hundreds of spans or thousands of frames, one bad item must
  not destroy the run. Likewise `extract_frames_at()` skips a frame ffmpeg
  cannot decode. The rule of thumb: a wrong *input pairing* is fatal, one
  unusable *item* is not.
- `GlossaryError` is raised for contradictions, not just malformed YAML —
  a duplicate canonical, or a `misheard` spelling claimed by two entries. These
  are silent-corruption risks, so they fail loudly at load time.

---

## Appendix: tests

`tests/` mirrors the package: `tests/nametag/`, `tests/crosscheck/`,
`tests/review/` plus flat modules for the audio pipeline. Slow tests
(`--run-slow`, real model loading, needs `HF_TOKEN`) and end-to-end tests
(`--run-e2e`, TTS-generated fixtures — see
[`E2E-TESTING-DESIGN.md`](E2E-TESTING-DESIGN.md)) are skipped by default.
Development is test-first: a failing test, then the implementation.

The repository once also carried a `research/` tree of validation tooling —
tile-detection bake-off harnesses, HTML contact sheets for eyeballing OCR crops,
a draft-roster builder, and a two-pass correction tool. All of it has been
removed: the detector choice it existed to settle was settled, and nothing in
the shipped pipeline invoked it. The tuned constants it produced live on in
`nametag/name_region.py` and `nametag/heuristic.py`.
