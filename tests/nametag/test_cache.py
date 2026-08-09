"""Tests for the nametag frame-cache path policy (cache.py)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from gromit.nametag.cache import cache_dir_for, discard


def test_cache_dir_is_under_system_temp_and_named_by_stem():
    d = cache_dir_for("/some/where/2026-01-15-board.mp4")
    root = Path(tempfile.gettempdir()) / "gromit-nametag"
    assert root in d.parents
    assert d.name.startswith("2026-01-15-board-")


def test_same_stem_different_paths_get_distinct_dirs(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    a = cache_dir_for(tmp_path / "x" / "m.mp4")
    b = cache_dir_for(tmp_path / "y" / "m.mp4")
    assert a != b
    assert a.name.startswith("m-") and b.name.startswith("m-")


def test_same_path_is_stable():
    p = "/some/where/m.mp4"
    assert cache_dir_for(p) == cache_dir_for(p)


def test_discard_removes_existing_dir(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "f.jpg").write_bytes(b"x")
    discard(d)
    assert not d.exists()


def test_discard_missing_dir_is_noop(tmp_path):
    discard(tmp_path / "does-not-exist")  # must not raise
