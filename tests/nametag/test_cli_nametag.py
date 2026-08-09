"""CLI tests for `gromit nametag` (folder resolution + candidate wiring)."""
from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from gromit import cli

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_meeting(tmp_path):
    (tmp_path / "m.mp4").write_bytes(b"x")
    (tmp_path / "m.vtt").write_text("WEBVTT\n", encoding="utf-8")


def test_nametag_resolves_folder_and_passes_candidates(tmp_path, monkeypatch):
    _make_meeting(tmp_path)
    captured = {}

    def fake_attr(video, vtt, out_dir, candidates, **kw):
        captured.update(video=Path(video).name, vtt=Path(vtt).name, candidates=candidates)
        return {"stem": "m", "cues": 0, "out_dir": str(out_dir),
                "use_vision": False, "named_vtt": str(Path(out_dir) / "m.named.vtt"),
                "cache_dir": "/tmp/gromit-nametag/m-deadbeef", "kept": False, "needs_review": 0}

    monkeypatch.setattr("gromit.nametag.run.attribute_meeting", fake_attr)
    result = runner.invoke(cli.app, ["nametag", str(tmp_path),
                                     "--guest", "Yaroslav Vyshnevetsky", "--guest", "Mykola H", "--verbose"])
    assert result.exit_code == 0, result.output
    assert captured["video"] == "m.mp4" and captured["vtt"] == "m.vtt"
    assert captured["candidates"] == ["Yaroslav Vyshnevetsky", "Mykola H"]


def test_nametag_errors_without_candidates(tmp_path):
    _make_meeting(tmp_path)
    result = runner.invoke(cli.app, ["nametag", str(tmp_path)])
    assert result.exit_code == 1
    assert "no candidate names" in result.output


def test_nametag_errors_when_folder_has_no_media(tmp_path):
    result = runner.invoke(cli.app, ["nametag", str(tmp_path), "--guest", "X"])
    assert result.exit_code == 1
    assert "need one .mp4 and one .vtt" in result.output


def test_nametag_accepts_explicit_video_and_vtt_with_unrelated_stems(tmp_path, monkeypatch):
    # Meet exports the recording and its captions under names that do NOT share a
    # stem; the explicit form must handle that without renaming or symlinking.
    (tmp_path / "Team sync - Recording.mp4").write_bytes(b"x")
    (tmp_path / "Team sync - Recording-uk-asr.vtt").write_text("WEBVTT\n", encoding="utf-8")
    captured = {}

    def fake_attr(video, vtt, out_dir, candidates, **kw):
        captured.update(video=Path(video).name, vtt=Path(vtt).name, out_dir=str(out_dir))
        return {"stem": "s", "cues": 0, "out_dir": str(out_dir), "use_vision": False,
                "named_vtt": str(Path(out_dir) / "s.named.vtt"), "cache_dir": "/tmp/x",
                "kept": False, "needs_review": 0}

    monkeypatch.setattr("gromit.nametag.run.attribute_meeting", fake_attr)
    result = runner.invoke(cli.app, [
        "nametag",
        "--video", str(tmp_path / "Team sync - Recording.mp4"),
        "--vtt", str(tmp_path / "Team sync - Recording-uk-asr.vtt"),
        "--guest", "X",
    ])
    assert result.exit_code == 0, result.output
    assert captured["video"] == "Team sync - Recording.mp4"
    assert captured["vtt"] == "Team sync - Recording-uk-asr.vtt"
    # output lands beside the video when no folder is given
    assert captured["out_dir"] == str(tmp_path)


def test_nametag_errors_when_only_one_of_video_vtt_given(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    result = runner.invoke(cli.app, ["nametag", "--video", str(tmp_path / "a.mp4"), "--guest", "X"])
    assert result.exit_code == 1
    assert "--video and --vtt must be given together" in _strip_ansi(result.output)


def test_nametag_errors_when_neither_folder_nor_explicit_paths(tmp_path):
    result = runner.invoke(cli.app, ["nametag", "--guest", "X"])
    assert result.exit_code == 1
    assert "give a meeting FOLDER or --video with --vtt" in _strip_ansi(result.output)


def test_nametag_keep_cache_flag_forwards(tmp_path, monkeypatch):
    _make_meeting(tmp_path)
    captured = {}

    def fake_attr(video, vtt, out_dir, candidates, **kw):
        captured.update(kw)
        return {"stem": "m", "cues": 0, "out_dir": str(out_dir),
                "use_vision": False, "named_vtt": str(Path(out_dir) / "m.named.vtt"),
                "cache_dir": "/tmp/x", "kept": True, "needs_review": 0}

    monkeypatch.setattr("gromit.nametag.run.attribute_meeting", fake_attr)
    result = runner.invoke(cli.app, ["nametag", str(tmp_path), "--guest", "X", "--keep-cache"])
    assert result.exit_code == 0, result.output
    assert captured["keep_cache"] is True

    # and the default is off — the cache is only kept when asked, or when the
    # run itself decides cues need review
    captured.clear()
    result = runner.invoke(cli.app, ["nametag", str(tmp_path), "--guest", "X"])
    assert result.exit_code == 0, result.output
    assert captured["keep_cache"] is False


def test_nametag_prints_review_hint_when_cues_need_review(tmp_path, monkeypatch):
    _make_meeting(tmp_path)

    def fake_attr(video, vtt, out_dir, candidates, **kw):
        return {"stem": "m", "cues": 3, "out_dir": str(out_dir),
                "use_vision": False, "named_vtt": str(Path(out_dir) / "m.named.vtt"),
                "cache_dir": "/tmp/gromit-nametag/m-abc12345", "kept": True, "needs_review": 2}

    monkeypatch.setattr("gromit.nametag.run.attribute_meeting", fake_attr)
    result = runner.invoke(cli.app, ["nametag", str(tmp_path), "--guest", "X"])
    assert result.exit_code == 0, result.output
    plain = _strip_ansi(result.output)
    assert "2 cues need review" in plain
    # the kept frame cache is the whole point of the hint — name it, so the user
    # knows where to look at the cues that could not be resolved
    assert "/tmp/gromit-nametag/m-abc12345" in plain
