#!/usr/bin/env bash
# Push a meeting's video to a CUDA worker, run `gromit transcribe`, pull
# the .gromit.{txt,json} back, and delete the remote copy (worker keeps nothing).
#
# The worker was originally a WSL install — hence the script name and the WSL_*
# variables, which are historical. Nothing here is WSL-specific: any ssh-reachable
# machine with a CUDA-capable gromit checkout at WSL_REPO works.
#
# Usage:  WSL_HOST=user@host tools/wsl-transcribe.sh <meeting-dir> [extra gromit transcribe args…]
# Example: WSL_HOST=user@host tools/wsl-transcribe.sh /path/to/meetings/2026-01-15-board \
#            --glossary /path/to/meetings/glossary.yaml --language uk
#
# Env: WSL_HOST (required) — ssh target of the worker.
#      WSL_REPO (default ~/projects/gromit) — the gromit checkout on that worker.
set -euo pipefail

WSL_REPO="${WSL_REPO:-~/projects/gromit}"

die() { echo "wsl-transcribe: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: wsl-transcribe.sh <meeting-dir> [transcribe args…]"
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
[ -n "${WSL_HOST:-}" ] \
  || die "WSL_HOST is not set — point it at an ssh-reachable CUDA worker (e.g. WSL_HOST=user@host)"
ssh -o ConnectTimeout=15 "$WSL_HOST" true 2>/dev/null \
  || die "worker unreachable: $WSL_HOST (check SSH connectivity)"

REMOTE="$(ssh "$WSL_HOST" 'mktemp -d /tmp/gromit-wsl.XXXXXX')"
[ -n "$REMOTE" ] || die "could not create remote temp dir"
cleanup() { ssh "$WSL_HOST" "rm -rf '$REMOTE'" 2>/dev/null || true; }
trap cleanup EXIT

echo "wsl-transcribe: → $WSL_HOST:$REMOTE (uploading video…)"
rsync -a "$MP4" "$WSL_HOST:$REMOTE/recording.mp4"

# rsync each glossary to a safe remote name and rewrite the remote --glossary args.
declare -a REMOTE_GLOSS_ARGS=()
i=0
for g in "${GLOSSARY_LOCAL[@]:-}"; do
  [ -n "$g" ] || continue
  [ -f "$g" ] || die "glossary not found: $g"
  rsync -a "$g" "$WSL_HOST:$REMOTE/glossary_${i}.yaml"
  REMOTE_GLOSS_ARGS+=(--glossary "$REMOTE/glossary_${i}.yaml")
  i=$((i + 1))
done

# Run transcribe on the worker (CUDA). Proxy cleared so model/HF traffic works.
echo "wsl-transcribe: transcribing on CUDA…"
ssh "$WSL_HOST" \
  "cd $WSL_REPO && ALL_PROXY= all_proxy= .venv/bin/gromit transcribe \
     '$REMOTE/recording.mp4' ${REMOTE_GLOSS_ARGS[*]:-} ${PASS_ARGS[*]:-} \
     --device cuda -o '$REMOTE/recording.gromit.txt'" \
  || die "remote transcribe failed — nothing pulled back, remote cleaned on exit"

# Pull results back under the meeting's own name.
echo "wsl-transcribe: ← results"
rsync -a "$WSL_HOST:$REMOTE/recording.gromit.txt"  "$MEETING_DIR/$STEM.gromit.txt"
rsync -a "$WSL_HOST:$REMOTE/recording.gromit.json" "$MEETING_DIR/$STEM.gromit.json"

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

echo "wsl-transcribe: done → $MEETING_DIR/$STEM.gromit.{txt,json}"
