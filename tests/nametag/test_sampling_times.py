"""Unit tests for per-cue frame-time selection and cached extraction."""
from __future__ import annotations

import itertools

from gromit.nametag.sampling import cue_frame_times, extract_frames_at


def test_at_least_seven_frames_even_for_short_cues():
    times = cue_frame_times(10.0, 12.0)  # 2.0s cue
    assert len(times) == 7
    assert times[0] == 10.0 and abs(times[-1] - 12.0) < 1e-9


def test_density_at_least_one_per_500ms_for_long_cues():
    times = cue_frame_times(0.0, 10.0)  # 10s -> ceil(20)+1 = 21 points, step 0.5
    assert len(times) == 21
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert max(gaps) <= 0.5 + 1e-9


def test_zero_length_cue_yields_single_time():
    assert cue_frame_times(5.0, 5.0) == [5.0]


def test_extract_frames_at_caches_and_dedups(tmp_path):
    calls = []

    def fake_run(cmd, check, **kw):  # **kw absorbs stdout/stderr suppression           # stand in for subprocess.run
        out = cmd[-1]                   # last arg is the output path
        open(out, "w").close()          # "extract" by touching the file
        calls.append(out)

    out = extract_frames_at("v.mp4", [1.0, 1.0004, 2.0], tmp_path, _run=fake_run)
    # 1.0 and 1.0004 round to the same millisecond -> one extraction
    assert len(calls) == 2
    assert len(out) == 3  # but all three requested times map to a cached frame

    # second call reuses the cache -> no new extractions
    calls.clear()
    extract_frames_at("v.mp4", [2.0], tmp_path, _run=fake_run)
    assert calls == []


def test_extract_frames_at_skips_failed_extraction(tmp_path):
    import subprocess

    def fake_run(cmd, check, **kw):  # **kw absorbs stdout/stderr suppression
        raise subprocess.CalledProcessError(234, cmd)  # e.g. ffmpeg seek past EOF

    out = extract_frames_at("v.mp4", [1.0, 2.0], tmp_path, _run=fake_run)
    assert out == []  # both failed -> skipped, no exception raised
