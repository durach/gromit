"""Unit tests for the shared nametag run pipeline (run.py)."""
from __future__ import annotations

import numpy as np

from gromit.nametag import run
from gromit.nametag.roster import NameMatch
from gromit.nametag.schema import FrameResult, Tile, TileKind
from gromit.nametag.vtt import Cue


def test_frame_reader_returns_none_when_no_tile(monkeypatch):
    monkeypatch.setattr(run, "detect", lambda frame: FrameResult(tiles=[]))
    reader = run.make_frame_reader(image_loader=lambda p: np.zeros((1080, 1920, 3), np.uint8))
    assert reader("anything.jpg", ["Solomiya Verbytska"]) is None


def test_frame_reader_delegates_to_resolve_name(monkeypatch):
    one = Tile(kind=TileKind.CAMERA, box=(0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(run, "detect", lambda frame: FrameResult(tiles=[one]))
    sentinel = NameMatch("Solomiya Verbytska", 0.97, True)
    monkeypatch.setattr(run, "resolve_name",
                        lambda crop, candidates, easy_reader=None, use_vision=None: sentinel)
    reader = run.make_frame_reader(image_loader=lambda p: np.ones((1080, 1920, 3), np.uint8),
                                   use_vision=False)
    assert reader("f.jpg", ["Solomiya Verbytska"]) is sentinel


def test_attribute_meeting_writes_named_files(tmp_path, monkeypatch):
    # fake the per-frame extraction + reader so no video/OCR is needed
    cues = [Cue(0, 0.0, 2.0, "hi"), Cue(1, 2.0, 4.0, "there")]
    monkeypatch.setattr(run, "parse_vtt", lambda p: cues)
    monkeypatch.setattr(run, "parse_header", lambda p: "WEBVTT")
    monkeypatch.setattr(run, "extract_frames_at", lambda v, times, cache: [(t, "f") for t in times])
    monkeypatch.setattr(run, "make_frame_reader",
                        lambda use_vision=None: (lambda path, cands: NameMatch("Mykola H", 1.0, True)))

    seen: list[str] = []
    out = tmp_path / "out"
    summary = run.attribute_meeting(tmp_path / "m.mp4", "m.vtt", out, ["Mykola H"],
                                    use_vision=False, on_cue=lambda i, n, name: seen.append(name))

    assert summary["cues"] == 2 and len(seen) == 2
    assert (out / "m.named.vtt").exists() and (out / "m.named.txt").exists()
    assert "Mykola H:" in (out / "m.named.txt").read_text(encoding="utf-8")


def _clean_fakes(monkeypatch, reader_match):
    cues = [Cue(0, 0.0, 2.0, "hi"), Cue(1, 2.0, 4.0, "there")]
    monkeypatch.setattr(run, "parse_vtt", lambda p: cues)
    monkeypatch.setattr(run, "parse_header", lambda p: "WEBVTT")
    monkeypatch.setattr(run, "extract_frames_at", lambda v, times, cache: [(t, "f") for t in times])
    monkeypatch.setattr(run, "make_frame_reader",
                        lambda use_vision=None: (lambda path, cands: reader_match))


def test_clean_run_deletes_cache(tmp_path, monkeypatch):
    _clean_fakes(monkeypatch, NameMatch("Mykola H", 1.0, True))
    cache = tmp_path / "cache"
    cache.mkdir()
    summary = run.attribute_meeting(tmp_path / "m.mp4", "m.vtt", tmp_path / "out",
                                    ["Mykola H"], use_vision=False, cache_dir=cache)
    assert summary["needs_review"] == 0
    assert summary["kept"] is False
    assert summary["cache_dir"] == str(cache)
    assert not cache.exists()


def test_dirty_run_keeps_cache(tmp_path, monkeypatch):
    # reader returns None for every frame -> each cue resolves to UNKNOWN
    _clean_fakes(monkeypatch, None)
    cache = tmp_path / "cache"
    cache.mkdir()
    summary = run.attribute_meeting(tmp_path / "m.mp4", "m.vtt", tmp_path / "out",
                                    ["Mykola H"], use_vision=False, cache_dir=cache)
    assert summary["needs_review"] >= 1
    assert summary["kept"] is True
    assert cache.exists()


def test_keep_cache_flag_keeps_clean_cache(tmp_path, monkeypatch):
    _clean_fakes(monkeypatch, NameMatch("Mykola H", 1.0, True))
    cache = tmp_path / "cache"
    cache.mkdir()
    summary = run.attribute_meeting(tmp_path / "m.mp4", "m.vtt", tmp_path / "out",
                                    ["Mykola H"], use_vision=False, cache_dir=cache,
                                    keep_cache=True)
    assert summary["needs_review"] == 0
    assert summary["kept"] is True
    assert cache.exists()
