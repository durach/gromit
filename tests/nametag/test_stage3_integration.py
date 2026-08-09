"""Slow end-to-end seam test: real ffmpeg extraction feeding the vote loop.

Guards the past-EOF tolerance in extract_frames_at against a real video.
"""
from __future__ import annotations

import subprocess

import pytest

from gromit.nametag.attribution import UNKNOWN, attribute_cue
from gromit.nametag.sampling import cue_frame_times, extract_frames_at


@pytest.mark.slow
def test_extraction_to_vote_survives_past_eof(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", "color=c=black:s=320x240:d=2", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    cache = tmp_path / "cache"
    times = cue_frame_times(1.5, 4.0)          # window runs PAST the 2s clip
    frames = extract_frames_at(clip, times, cache)
    assert 0 < len(frames) < len(times)        # in-bounds frames kept, past-EOF skipped
    res = attribute_cue(frames, ["Nobody"], lambda path, candidates: None)
    assert res.name == UNKNOWN                 # no crash; cue with no readings -> Unknown
