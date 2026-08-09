"""Cut a small, seekable ~480p clip per flagged span (ffmpeg re-encode)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def build_clip_command(
    video: Path, start: float, end: float, out_path: Path, pad: float = 5.0
) -> list[str]:
    """ffmpeg argv for a padded, re-encoded ~480p clip. Start clamped to >= 0.

    Re-encode (not stream copy): keyframe-aligned copies would drift the cut by
    seconds. `-ss` before `-i` for a fast input seek.
    """
    clip_start = max(0.0, start - pad)
    duration = (end + pad) - clip_start
    return [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(video),
        "-t", f"{duration:.3f}",
        "-vf", "scale=-2:480",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(out_path),
    ]


def extract_clip(
    video: Path, start: float, end: float, out_path: Path, pad: float = 5.0
) -> bool:
    """Run ffmpeg; return True iff the clip was produced. Never raises."""
    try:
        subprocess.run(
            build_clip_command(video, start, end, out_path, pad),
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    return out_path.exists() and out_path.stat().st_size > 0
