import shutil
import subprocess

import pytest

from gromit.nametag.sampling import sample_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _make_video(path, seconds=2):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size=320x180:rate=10",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_sample_frames_interval(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, seconds=2)
    out = tmp_path / "frames"
    frames = sample_frames(video, interval=0.5, out_dir=out)
    # 2s at 0.5s -> ~4 frames (allow +/-1 for boundary)
    assert 3 <= len(frames) <= 5
    ts, fp = frames[0]
    assert fp.exists()
    assert ts == pytest.approx(0.0, abs=0.01)
