"""Frame-cache path policy for the nametag pipeline.

Sampled frames are scratch data: they live under the system temp dir, keyed per
meeting, so they never touch the (tidy) meeting folders and are reclaimed on
reboot. ``run.attribute_meeting`` decides per run whether to keep or discard the
directory (kept when cues still need review, or when ``--keep-cache`` is set).
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path


def cache_dir_for(video: str | Path) -> Path:
    """Per-meeting frame-cache directory under the system temp dir.

    Named ``<stem>-<hash8>`` where the hash is over the *resolved* video path, so
    two different videos that happen to share a stem (across folders) get
    distinct directories. Does not create the directory — callers extract frames
    into it lazily (``sampling.extract_frames_at`` mkdirs as needed).
    """
    video = Path(video)
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8"),
                          usedforsecurity=False).hexdigest()[:8]
    return Path(tempfile.gettempdir()) / "gromit-nametag" / f"{video.stem}-{digest}"


def discard(cache_dir: str | Path) -> None:
    """Remove a cache directory and its contents; a no-op if it is absent."""
    shutil.rmtree(Path(cache_dir), ignore_errors=True)
