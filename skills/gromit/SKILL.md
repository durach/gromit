---
name: gromit
description: Use when running gromit — transcribing an audio or video recording to speaker-attributed text, tagging Google Meet speakers with their real on-screen names (nametag), flagging likely ASR errors against Meet's captions (crosscheck), building the offline clip review page (review), or merging reviewed corrections into a project glossary (glossary-merge). Also for processing a meeting recording end to end.
---

# Driving gromit

Gromit is a local-first transcription tool: audio and video in, speaker-attributed
transcript out, with a quality loop that flags likely ASR errors and folds
corrections back into a per-project glossary. Nothing it runs uploads media.

**Read `docs/AGENTS.md` in the gromit checkout before doing anything.** It is the
operating manual for agents — preflight checks, where the pipeline hands back to
a human, and which alarming output is normal. From there, `docs/PIPELINE.md` is
the copy-paste runbook and `README.md` is the flag reference. Follow the runbook;
do not reconstruct commands from memory.

This skill is symlinked out of the checkout, so it can locate itself:

```bash
GROMIT_REPO="$(dirname "$(dirname "$(readlink -f ~/.claude/skills/gromit)")")"
ls "$GROMIT_REPO/docs/AGENTS.md"
```

If that fails (the skill was copied rather than symlinked), ask the user where
the gromit checkout is rather than guessing.

## The three rules that matter before you read anything

- **Never pass `--glossary` to `transcribe`.** Measured: hotwords of any size
  collapsed transcript coverage from 76 % to 27 % on some recordings. The
  glossary does its work in `crosscheck` instead. `AGENTS.md` has the detail.
- **`gromit review` is where you stop.** It builds a page of video clips for a
  *human* to watch and correct. Build it, hand over the path, and wait. The
  agent-driven alternative (stage 2½ of `PIPELINE.md`) means reading the entire
  meeting transcript — if you are a hosted model, that sends it to your provider,
  so it is the user's explicit call, not yours.
- **`transcribe` is a required subcommand** and can run for hours on CPU. Check
  the device with `-v`, agree a plan, and run it in the background.
