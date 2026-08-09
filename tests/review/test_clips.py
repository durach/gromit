"""Tests for ffmpeg clip extraction."""

import shutil
import subprocess
from pathlib import Path

import pytest

from gromit.review.clips import build_clip_command, extract_clip


def test_build_command_clamps_start_and_sets_scale():
    cmd = build_clip_command(Path("v.mp4"), 3.0, 8.0, Path("out.mp4"), pad=5.0)
    # start 3-5 -> clamped to 0; duration = (8+5) - 0 = 13
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "0.000"
    assert cmd[cmd.index("-t") + 1] == "13.000"
    assert "scale=-2:480" in cmd
    assert cmd[-1] == "out.mp4"


def test_build_command_pads_symmetrically_midfile():
    cmd = build_clip_command(Path("v.mp4"), 100.0, 103.0, Path("o.mp4"), pad=5.0)
    assert cmd[cmd.index("-ss") + 1] == "95.000"
    assert cmd[cmd.index("-t") + 1] == "13.000"


def test_extract_clip_missing_video_returns_false(tmp_path):
    ok = extract_clip(tmp_path / "nope.mp4", 0.0, 1.0, tmp_path / "out.mp4")
    assert ok is False


@pytest.mark.slow
def test_extract_clip_real(tmp_path):
    """Generate a 3s test video, cut a clip, assert it plays out."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
        capture_output=True, check=True,
    )
    out = tmp_path / "clip.mp4"
    assert extract_clip(src, 1.0, 2.0, out) is True
    assert out.exists() and out.stat().st_size > 0
