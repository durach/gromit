#!/usr/bin/env bash
# Push a meeting's video to a CUDA worker, run `gromit transcribe`, pull
# the .gromit.{txt,json} back, and delete the remote copy (worker keeps nothing).
#
# The worker is any machine you can ssh into that has a CUDA-capable gromit
# checkout at GROMIT_CUDA_WORKER_REPO. This script is just ssh, rsync and a
# remote temp dir — it holds no assumptions about the far end beyond that.
#
# Usage:  GROMIT_CUDA_WORKER=user@host tools/remote-transcribe.sh <meeting-dir> [extra gromit transcribe args…]
# Example: GROMIT_CUDA_WORKER=user@host tools/remote-transcribe.sh /path/to/meetings/2026-01-15-board \
#            --glossary /path/to/meetings/glossary.yaml --language uk
#
# Env: GROMIT_CUDA_WORKER (required) — ssh target of the worker.
#      GROMIT_CUDA_WORKER_REPO (default ~/projects/gromit) — gromit checkout there.
set -euo pipefail

GROMIT_CUDA_WORKER_REPO="${GROMIT_CUDA_WORKER_REPO:-~/projects/gromit}"

die() { echo "remote-transcribe: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: remote-transcribe.sh <meeting-dir> [transcribe args…]"
MEETING_DIR="$1"; shift
[ -d "$MEETING_DIR" ] || die "meeting dir not found: $MEETING_DIR"

# Exactly one .mp4 in the meeting dir.
shopt -s nullglob
mp4s=("$MEETING_DIR"/*.mp4)
shopt -u nullglob
[ "${#mp4s[@]}" -eq 1 ] || die "need exactly one .mp4 in $MEETING_DIR (found ${#mp4s[@]})"
MP4="${mp4s[0]}"
STEM="$(basename "$MP4" .mp4)"

# Split incoming args: intercept --glossary PATH (rsync those files), pass the rest through.
declare -a GLOSSARY_LOCAL=()
declare -a PASS_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --glossary) GLOSSARY_LOCAL+=("$2"); shift 2 ;;
    --glossary=*) GLOSSARY_LOCAL+=("${1#--glossary=}"); shift ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

# Reachability check (loud failure before touching anything).
[ -n "${GROMIT_CUDA_WORKER:-}" ] \
  || die "GROMIT_CUDA_WORKER is not set — point it at an ssh-reachable CUDA worker (e.g. GROMIT_CUDA_WORKER=user@host)"
ssh -o ConnectTimeout=15 "$GROMIT_CUDA_WORKER" true 2>/dev/null \
  || die "worker unreachable: $GROMIT_CUDA_WORKER (check SSH connectivity)"

REMOTE="$(ssh "$GROMIT_CUDA_WORKER" 'mktemp -d /tmp/gromit-remote.XXXXXX')"
[ -n "$REMOTE" ] || die "could not create remote temp dir"
cleanup() { ssh "$GROMIT_CUDA_WORKER" "rm -rf '$REMOTE'" 2>/dev/null || true; }
trap cleanup EXIT

echo "remote-transcribe: → $GROMIT_CUDA_WORKER:$REMOTE (uploading video…)"
rsync -a "$MP4" "$GROMIT_CUDA_WORKER:$REMOTE/recording.mp4"

# rsync each glossary to a safe remote name and rewrite the remote --glossary args.
declare -a REMOTE_GLOSS_ARGS=()
i=0
for g in "${GLOSSARY_LOCAL[@]:-}"; do
  [ -n "$g" ] || continue
  [ -f "$g" ] || die "glossary not found: $g"
  rsync -a "$g" "$GROMIT_CUDA_WORKER:$REMOTE/glossary_${i}.yaml"
  REMOTE_GLOSS_ARGS+=(--glossary "$REMOTE/glossary_${i}.yaml")
  i=$((i + 1))
done

# Run transcribe on the worker (CUDA). Proxy cleared so model/HF traffic works.
echo "remote-transcribe: transcribing on CUDA…"
ssh "$GROMIT_CUDA_WORKER" \
  "cd $GROMIT_CUDA_WORKER_REPO && ALL_PROXY= all_proxy= .venv/bin/gromit transcribe \
     '$REMOTE/recording.mp4' ${REMOTE_GLOSS_ARGS[*]:-} ${PASS_ARGS[*]:-} \
     --device cuda -o '$REMOTE/recording.gromit.txt'" \
  || die "remote transcribe failed — nothing pulled back, remote cleaned on exit"

# Pull results back under the meeting's own name.
echo "remote-transcribe: ← results"
rsync -a "$GROMIT_CUDA_WORKER:$REMOTE/recording.gromit.txt"  "$MEETING_DIR/$STEM.gromit.txt"
rsync -a "$GROMIT_CUDA_WORKER:$REMOTE/recording.gromit.json" "$MEETING_DIR/$STEM.gromit.json"

# Rewrite hotwords_from from the ephemeral remote paths to the local glossary paths.
if [ "${#GLOSSARY_LOCAL[@]}" -gt 0 ]; then
  python3 - "$MEETING_DIR/$STEM.gromit.json" "${GLOSSARY_LOCAL[@]}" <<'PY'
import json, sys
path, locals_ = sys.argv[1], sys.argv[2:]
d = json.load(open(path))
d["hotwords_from"] = locals_
json.dump(d, open(path, "w"), ensure_ascii=False, indent=2)
PY
fi

echo "remote-transcribe: done → $MEETING_DIR/$STEM.gromit.{txt,json}"
