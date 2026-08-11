# Driving gromit from an AI agent

For an autonomous or semi-autonomous agent asked to run gromit on someone's
recording. It covers what an agent gets wrong that a human reading the README
does not: what to check before committing to a long job, where the pipeline
stops and a human has to take over, and which alarming-looking output is normal
and must not be "fixed".

It is **not** the runbook. The copy-paste command sequence lives in
[`PIPELINE.md`](PIPELINE.md) — read it before running the pipeline, and follow
it rather than reconstructing commands from memory. Flag-by-flag reference is
[`../README.md`](../README.md); internals are [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Orientation

Gromit is local-first: no gromit command uploads media anywhere, and the only
network access is a first-run model download. Two independent pipelines share
the package and never call each other:

- **audio** — `gromit transcribe` (faster-whisper + pyannote) → a transcript
  with `Speaker 1`/`Speaker 2` labels;
- **video** — `gromit nametag` (OpenCV + OCR) → Google Meet caption cues tagged
  with the participants' real on-screen names. Never opens the audio track.

`crosscheck` → `review` → `glossary-merge` then form the quality loop over the
audio pipeline's output. Each command is independently useful; you do not have
to run all five.

## Preflight

`transcribe` can run for longer than the meeting itself on a CPU-only machine.
Check all of this **before** starting, not thirty minutes in:

```bash
gromit --help                 # installed and importable at all
ffmpeg -version               # required by transcribe (video), review, nametag
echo "${HF_TOKEN:0:4}"        # transcribe only — see below
```

- **`HF_TOKEN` is needed by `transcribe` alone**, for pyannote diarization. The
  user must also have accepted the licence for `pyannote/speaker-diarization-3.1`.
  `nametag`, `crosscheck`, `review` and `glossary-merge` never touch pyannote —
  do not block them on a missing token.
- **Confirm the device** with `gromit transcribe … -v`. If it reports CPU and
  the recording is long, say so and agree a plan (smaller `--model`, or the
  remote GPU worker in PIPELINE.md) before starting, rather than silently
  committing the user to hours.
- **Locate the real files.** Meet names recordings
  `<Title> - <date> <time> <TZ> - Recording.mp4`; resolve the actual stem with
  `ls`, never assume a bare name. `nametag` additionally needs Meet's `.vtt`
  caption track, which **gromit cannot generate** — the user downloads it from
  the Drive player. Without it, `nametag` cannot run at all.

## The commands

| Command | Needs | Writes | Long? |
|---|---|---|---|
| `transcribe FILE…` | ffmpeg (video), `HF_TOKEN` | `<stem>.gromit.{txt,json}` | **yes — minutes to hours** |
| `nametag FOLDER` or `--video/--vtt` | ffmpeg, `.mp4` + Meet `.vtt`, `--roster`/`--guest` | `<stem>.named.{vtt,txt}` | yes — minutes |
| `crosscheck GROMIT_JSON -o …` | the `.gromit.json` | `<stem>.flags.json` | no |
| `review FLAGS_JSON --video … -o …` | ffmpeg | `review/index.html` + clips | minutes (re-encodes) |
| `glossary-merge CORRECTIONS --glossary …` | — | **edits the glossary in place** | no |

Every command exits 1 on failure and prints a single `Error: …` line — there is
no traceback to parse. `transcribe` is a required subcommand; bare
`gromit file.mp4` does not work.

Run the two long stages **in the background**, and note that `transcribe` and
`nametag` are independent — they can run in parallel on the same meeting, since
they share only read access to the `.mp4`. `crosscheck` needs both to finish.

## Where you must hand back to a human

**`gromit review` builds a page for a person to watch.** It cuts one short video
clip per flagged span, and settling what was actually said means watching them.
You can build the page; you cannot fill it in. Build it, tell the user where
`index.html` is, and stop — the corrections come back from them as a
`corrections.yaml` exported by the page.

There is an agent-shaped alternative, **stage 2½ in [`PIPELINE.md`](PIPELINE.md)**:
you read `flags.json` plus the transcripts and triage the spans yourself into
auto / ask / skip, so the human answers one small batch instead of working
several hundred rows for 1–2 hours. Read that section before attempting it, and
note two things:

1. **It is opt-in and it has a privacy cost.** Triage means reading the complete
   meeting text. Gromit is local-first precisely so that content stays on the
   user's machine; if you are a hosted model, reading it sends it to your
   provider. Ask first, in those terms. If the meeting is confidential, the
   manual review page is the local-only route.
2. **Auto-apply only new misheard spellings of canonicals already in the
   glossary.** New canonical terms, who-said-what, numbers, dates and decisions
   go in the ask batch — they are exactly the cases where a plausible guess is
   worst.

`glossary-merge` **edits the user's glossary in place**. Confirm before running
it. It is idempotent and comment-preserving, so a repeat run is safe and reports
everything as `unchanged` — that re-run is the natural check that a merge landed.

## Do not "fix" these

Each of these looks like a bug an agent should solve, and is not:

- **A `torchcodec … Could not load this library` traceback** during `transcribe`.
  Harmless noise from a pyannote dependency; pyannote falls back to another
  decoder and the run is fine. Do not reinstall or pin anything.
- **`nametag` reporting *N* cues need review.** A normal outcome, not a failure:
  either no on-screen name was readable (labelled `Unknown`) or the name read is
  not in the candidate list (kept verbatim, never snapped to a roster entry — an
  unlisted guest must not be relabelled as someone who was there). Usually the
  fix is a missing `--guest "Name"`, which is the user's to supply.
- **`DiarizationError` mentioning `HF_TOKEN` or licences.** Read the underlying
  message first. If it says `Using SOCKS proxy, but the 'socksio' package is not
  installed`, it is neither the token nor the licence — see the README's
  Troubleshooting section.
- **Warnings from pyannote** (`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, `std(): degrees
  of freedom is <= 0`). Expected; not actionable here.

And one thing to actively avoid doing:

> **Never pass `--glossary` to `transcribe`.** The flag exists and is documented,
> but hotwords of *any* size — a handful of terms was as bad as a whole glossary
> — were measured collapsing transcript coverage from 76 % to 27 % on some
> recordings, with whole segments dropped by Whisper's own filters. The cause is
> not understood. The glossary does its real work in `crosscheck`, which needs no
> hotwords. If a user asks for it explicitly, quote this warning first.

## Checking a stage actually worked

- **transcribe** — `<stem>.gromit.json` exists and is non-trivial. Health check:
  a good transcript runs ≈12 characters per spoken second. Far below that
  suggests dropped segments, so check whether hotwords were used.
- **nametag** — `<stem>.named.vtt` exists; the run prints the cue count, the OCR
  engine, and any needs-review count.
- **crosscheck** — prints the span count and a breakdown by reason. Zero spans on
  a real meeting means something is wrong, most likely a mispaired `--meet` file.
- **review** — `review/index.html` plus `review/clips/NNN.mp4`; the run prints
  how many clips were extracted, which can be fewer than the span count.
- **glossary-merge** — prints `N new, M misheard added, K unchanged`.

## Conventions this project does not have

Gromit ships **no** built-in knowledge of any project, organisation or person.
The glossary and roster are the user's files, passed explicitly with
`--glossary` / `--roster`. Never borrow another project's glossary as a default,
and never invent roster entries: an unlisted name is a question for the user, not
a gap to fill in.
