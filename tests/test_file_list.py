"""Tests for file list resolution."""

from pathlib import Path

import pytest
import typer

from gromit.cli import resolve_file_list


def test_resolve_file_list_basic(tmp_path):
    """Resolve relative paths against list file's parent directory."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("part1.mp4\npart2.mp4\n")
    result = resolve_file_list(list_file)
    assert result == [tmp_path / "part1.mp4", tmp_path / "part2.mp4"]


def test_resolve_file_list_skips_blank_lines_and_comments(tmp_path):
    """Blank lines and # comments are ignored."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("# Day 1 files\npart1.mp4\n\npart2.mp4\n# end\n")
    result = resolve_file_list(list_file)
    assert result == [tmp_path / "part1.mp4", tmp_path / "part2.mp4"]


def test_resolve_file_list_absolute_paths(tmp_path):
    """Absolute paths are used as-is."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("/absolute/path/file.mp4\n")
    result = resolve_file_list(list_file)
    assert result == [Path("/absolute/path/file.mp4")]


def test_resolve_file_list_mixed_paths(tmp_path):
    """Mix of relative and absolute paths."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("relative.mp4\n/absolute/file.mp4\n")
    result = resolve_file_list(list_file)
    assert result == [tmp_path / "relative.mp4", Path("/absolute/file.mp4")]


def test_resolve_file_list_windows_absolute(tmp_path):
    """Windows-style drive letter paths treated as absolute."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("C:\\Users\\test\\file.mp4\n")
    result = resolve_file_list(list_file)
    assert result == [Path("C:\\Users\\test\\file.mp4")]


def test_resolve_file_list_empty_file_raises(tmp_path):
    """Empty file (or only comments/blanks) raises error."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("# just a comment\n\n")
    with pytest.raises(typer.BadParameter, match="no file entries"):
        resolve_file_list(list_file)


def test_resolve_file_list_strips_whitespace(tmp_path):
    """Leading/trailing whitespace on lines is stripped."""
    list_file = tmp_path / "files.txt"
    list_file.write_text("  part1.mp4  \n  part2.mp4\n")
    result = resolve_file_list(list_file)
    assert result == [tmp_path / "part1.mp4", tmp_path / "part2.mp4"]
