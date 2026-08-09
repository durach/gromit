# Transcript-quality pipeline — runbook

The end-to-end flow that turns a raw Google Meet recording into a
speaker-attributed transcript, locates the spans worth a human's attention, and
feeds every correction back into a per-project glossary so the next meeting
transcribes better.

**What stays local, and what does not.** Every stage that gromit itself runs is
local: the recording and its audio are read, transcribed, diarized and clipped
on machines you control, and no gromit command uploads media anywhere. Two
things in this runbook are exceptions you opt into explicitly, and both are
called out where they appear: `tools/wsl-transcribe.sh` (stage 1) copies the
video over SSH to a GPU host **you** nominate, and stage 2½ below hands
transcript **text** to whatever LLM you choose to point at it. Audio never
leaves your machines either way — but stage 2½ is text, and if the LLM you use
is a hosted one, that text goes to a third party. Read stage 2½ before running
it.

How the pieces work internally: [`ARCHITECTURE.md`](ARCHITECTURE.md).
Command-by-command reference: [`../README.md`](../README.md).

## The four stages

| Stage | Command | Produces |
|-------|---------|----------|
| 1. Transcribe | `gromit transcribe` (optionally via `tools/wsl-transcribe.sh`) | `<stem>.gromit.{txt,json}` |
| 2. Cross-check | `gromit crosscheck` | `<stem>.flags.json` |
| 2½. Triage (LLM session, optional — sends transcript text to an LLM) | — (no command; see below) | `corrections.yaml` + a small ask-batch |
| 3. Review | `gromit review` | `review/index.html` + `review/clips/` |
| 4. Merge corrections | `gromit glossary-merge` | updated `glossary.yaml` |

Each stage is useful on its own and can be re-run independently.

## Stage 2½ — LLM triage of flags (optional; the author's usual path)

> **What this stage sends where.** There is no `gromit` command for it. You open
> an LLM session and give it `flags.json`, **both full transcripts**, and the
> project glossary. That is complete meeting text — everything said in the
> meeting, by name where `nametag` ran. If the LLM you use is a hosted service,
> that text leaves your machine and goes to that provider under their terms; if
> you run a local model (Ollama, llama.cpp, LM Studio — anything that reads the
> files on disk), nothing leaves. Gromit neither picks the model nor makes the
> call, so this is entirely your choice, and it is the only stage in the
> pipeline where transcript content can go anywhere.
>
> **The local-only alternative is stage 3 as it stands.** Skip 2½ entirely, run
> `gromit review`, and work the rows by hand on the offline page: same
> `corrections.yaml` out, nothing sent anywhere, at the cost of the 1–2 hours
> that 2½ exists to save. Nothing downstream can tell which route produced the
> file. If the meeting is confidential, take this route or use a local model.

A meeting produces far more flagged spans than glossary material — on an
hour-long call, expect a few hundred. Working through them by hand on the
review page takes 1–2 hours; an LLM session driving the pipeline does the
judgment instead, and the
human answers only a small batch. Manual row-by-row review remains the fallback
— the review page is unchanged, and nothing in gromit requires this stage.

The session reads `flags.json` + both transcripts + the project glossary and
puts every span in exactly one bucket:

- **auto** → `corrections.yaml` directly. Only for new misheard spellings of
  canonicals **already in** the glossary; `heard` trimmed to the minimal
  misheard fragment; deduplicated; never a form colliding with an ordinary
  word (collision rule: note-only, no auto-replacement).
- **ask** → one batched question round (1–4 per call, timecode + quote + clip
  link). New canonicals, who-said-what, numbers/dates/decisions,
  collision-suspects, real source conflicts.
- **skip** → filler-word divergence, caption-repetition noise. Counted in the
  report, never silently dropped.

Answers append to the same `corrections.yaml`, then stage 4 merges it. The
session should end with a summary: every auto item listed, ask/skip counts, and
an idempotent re-merge ("unchanged") as the check.

## Inputs (manual)

Keep one directory per meeting, named however you like —
`/path/to/meetings/<YYYY-MM-DD-slug>/` is the convention used throughout this
document. Drop two files into it:

- `<stem>.mp4` — the recording, as Google Meet named it (typically
  `<Title> - <date> <time> <TZ> - Recording.mp4`);
- a **Meet WebVTT caption file** — Meet's auto-caption track, downloaded from
  the Drive player (Meet does not put it in the folder for you). Optional but
  recommended: it powers the `divergence` signal in stage 2, and it is the cue
  source for `gromit nametag`, which cannot run without it.

  `<stem>-uk-asr.vtt` is the naming convention used throughout this document
  (`uk-asr` = Ukrainian ASR track); it is a convention, not a requirement.
  `gromit crosscheck --meet PATH` takes the path explicitly, while
  `gromit nametag` either takes both paths explicitly with `--video` and `--vtt`
  — the reliable form when the recording and its captions do not share a stem —
  or, given a folder, globs it for `*.vtt`, excludes its own `*.named.vtt`
  output, sorts, and takes the first. See the `gromit nametag` section of
  [`../README.md`](../README.md) for the full selection rule.

Everything the pipeline writes — `.gromit.json`, `.flags.json`, `.named.vtt`,
`review/` — lands in the same directory alongside them.

## The per-project glossary

`crosscheck`, `review`, and `transcribe` all take a `--glossary PATH`. It is a
file you keep with your own project — gromit has no built-in knowledge of any
project, any organisation, or any person. A worked example with comments is
[`../examples/glossary.yaml`](../examples/glossary.yaml). Format:

```yaml
terms:
  - canonical: "release checklist"     # correct written form
    category: term                     # term | person | company | product
    note: "кроки перед релізом"        # optional
    misheard: ["реліз чекліст"]        # accumulated ASR mistakes
```

## Full flow (copy-paste)

```bash
# --- setup ---
source ~/projects/gromit/.venv/bin/activate
GROMIT=~/projects/gromit
GLOSS=/path/to/meetings/glossary.yaml
DIR="/path/to/meetings/2026-01-15-board"
STEM="Team sync - 2026_01_15 10_00 UTC - Recording"

# --- 1. Transcribe (NO --glossary: see warning) ---
gromit transcribe "$DIR/$STEM.mp4" --language uk -o "$DIR/$STEM.gromit.txt"
#   check: ls "$DIR/$STEM.gromit.json"
#   ⚠️ Do NOT pass --glossary here. Measured on a real meeting recording:
#   hotwords of ANY size (a handful of terms as bad as a whole glossary)
#   collapse coverage on some recordings — 76% → 27%, whole segments dropped
#   by the logprob/compression filters. Unexplained; until it is, hotwords are
#   unsafe. The glossary does
#   its real work in stages 2–2½ (misheard repair), which don't need hotwords.
#   Health check: a good transcript has ≈12 chars per spoken second.
#
#   Faster on a CPU-only machine — hand this one step to a CUDA box you can ssh
#   into (see "Optional: a remote GPU worker" below):
#   WSL_HOST=user@gpu-worker "$GROMIT/tools/wsl-transcribe.sh" "$DIR" --language uk

# --- 2. (optional) speaker labels for the review page ---
gromit nametag "$DIR" --guest "Solomiya Verbytska" --guest "Yaroslav Vyshnevetsky"
#   check: ls "$DIR/$STEM.named.vtt"

# --- 3. cross-check against Google Meet's captions ---
gromit crosscheck "$DIR/$STEM.gromit.json" \
  --meet "$DIR/$STEM-uk-asr.vtt" \
  --glossary "$GLOSS" \
  -o "$DIR/$STEM.flags.json"
#   prints span count + reason breakdown

# --- 4. build the review page ---
gromit review "$DIR/$STEM.flags.json" \
  --video "$DIR/$STEM.mp4" \
  --named "$DIR/$STEM.named.vtt" \
  -o "$DIR/review/"
#   drop --named if you skipped step 2
open "$DIR/review/index.html"

# --- (in the browser) correct rows, tick "add to glossary", click Export ---

# --- 5. fold corrections back into the glossary ---
gromit glossary-merge ~/Downloads/corrections.yaml --glossary "$GLOSS"
#   idempotent + comment-preserving; re-run reports everything "unchanged"
```

## Optional: a remote GPU worker

Stage 1 is the only expensive step, and it is the only one that benefits from a
GPU. Nothing in the pipeline requires one — `gromit transcribe` runs on CPU and
on Apple Silicon, just more slowly.

If you do have a CUDA machine you can `ssh` into, `tools/wsl-transcribe.sh`
automates the round trip: it pushes the meeting's video (and any `--glossary`
files) to a temporary directory on the worker, runs `gromit transcribe` there,
pulls `<stem>.gromit.{txt,json}` back into the meeting folder, and deletes the
remote copy.

```bash
WSL_HOST=user@gpu-worker WSL_REPO=~/projects/gromit \
  tools/wsl-transcribe.sh /path/to/meetings/2026-01-15-board --language uk
```

- `WSL_HOST` (default `user@gpu-worker`) — any ssh target with a CUDA-capable
  gromit install. There is nothing WSL-, VPN- or vendor-specific about it; the
  name is historical.
- `WSL_REPO` (default `~/projects/gromit`) — where the checkout and its `.venv`
  live on the worker.
- The script fails loudly and early if the host is unreachable, rather than
  silently falling back.

## Notes & options

- **`--meet` is optional** in `crosscheck`; without it you still get
  `low_confidence` + `misheard_match` (no `divergence`). A Meet VTT whose
  timeline barely overlaps the transcript (<20 % of cues) is a hard error
  (wrong file pairing).
- **`gromit review --limit N`** cuts only the top-N ranked spans (mishears
  first, then divergence, then low confidence). Handy when a meeting flags many
  spans and you only want the highest-value clips.
- The review page is **self-contained** — it also opens straight from `file://`.
- **Flag signal thresholds** live in `src/gromit/crosscheck/signals.py`
  (`Thresholds`): starting points tuned on one real meeting, not constants to
  defend — retune on your own recordings if flags are too many/few.

## What crosscheck flags

- `misheard_match` — either engine's text contains a known `misheard` string;
  the correction is auto-suggested. Highest-value, most precise signal.
- `divergence` — the two engines' text disagrees over the same time window
  (asymmetric token-containment, so Meet's wider caption window doesn't count as
  disagreement).
- `low_confidence` — a cluster of low-probability words, or a low segment
  `avg_logprob`.
