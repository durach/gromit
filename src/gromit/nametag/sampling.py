"""Sample frames from a video at a fixed interval via ffmpeg, caching to disk."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path


def sample_frames(video_path, interval: float = 0.5, out_dir=None,
                  duration: float | None = None, offset: float = 0.0):
    """Extract 1 frame every `interval` seconds. Returns sorted
    list[(timestamp_seconds, frame_path)]. Idempotent: reuses existing files.

    Args:
        video_path: Path to input video.
        interval:   Seconds between extracted frames.
        out_dir:    Directory for cached frame JPEGs.
        duration:   Optional cap in seconds.  When given, only the first
                    *duration* seconds of the video are sampled.  Has no effect
                    when frames are already cached in *out_dir*.
        offset:     Seconds to skip before the first frame, shifting the whole
                    sampling grid (e.g. interval=5, offset=2 -> frames at 2, 7,
                    12 ... s).  Lets a second pass select the same NUMBER of
                    DIFFERENT frames.  Has no effect when frames are already
                    cached in *out_dir*.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir) if out_dir else video_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / interval
    pattern = out_dir / "frame_%06d.jpg"
    if not any(out_dir.glob("frame_*.jpg")):
        cmd = ["ffmpeg", "-nostdin", "-v", "error"]
        if offset:
            cmd += ["-ss", str(offset)]  # input seek before decode
        cmd += ["-i", str(video_path)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-vf", f"fps={fps}", "-q:v", "3", str(pattern)]
        subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    # ffmpeg fps filter emits frame N at output t = N*interval; the -ss seek
    # shifts the source time, so the source timestamp is N*interval + offset.
    return [(idx * interval + offset, fp) for idx, fp in enumerate(frames)]


def cue_frame_times(start: float, end: float,
                    min_frames: int = 7, max_interval: float = 0.5) -> list[float]:
    """Timestamps to sample within ``[start, end]``.

    Guarantees BOTH constraints from the spec: at least ``min_frames`` frames,
    and no gap wider than ``max_interval`` seconds. Endpoints are included.
    """
    dur = end - start
    if dur <= 0:
        return [start]
    n = max(min_frames, math.ceil(dur / max_interval) + 1)
    step = dur / (n - 1)
    return [start + i * step for i in range(n)]


def extract_frames_at(video_path, times, cache_dir, _run=subprocess.run):
    """Extract one video frame at each timestamp, caching by millisecond.

    Returns ``list[(time, Path)]``. Overlapping cues share the cache (frames at
    the same millisecond are extracted once). ``_run`` is injectable for tests.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t in times:
        fp = cache_dir / f"t_{round(t * 1000):09d}.jpg"
        if not fp.exists():
            # -pix_fmt yuvj420p: many Meet recordings are full-range (JPEG-range)
            # YUV, which the default mjpeg encoder rejects ("Non full-range YUV is
            # non-standard"); forcing the jpeg-range format lets those frames encode.
            cmd = ["ffmpeg", "-nostdin", "-v", "error",
                   "-ss", f"{t:.3f}", "-i", str(video_path),
                   "-frames:v", "1", "-pix_fmt", "yuvj420p", "-q:v", "3", str(fp)]
            try:
                _run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue  # seek past EOF / unreadable frame -> skip, don't abort the run
        if fp.exists():
            out.append((t, fp))
    return out
