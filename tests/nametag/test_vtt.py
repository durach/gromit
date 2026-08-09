"""Unit tests for Stage 3 WebVTT cue parsing."""
from __future__ import annotations

from gromit.nametag.vtt import Cue, format_timestamp, parse_header, parse_timestamp, parse_vtt

SAMPLE = """WEBVTT
Kind: captions
Language: uk

00:00:10.000 --> 00:00:14.500
Перший рядок.
Другий рядок.

00:00:20.000 --> 00:00:25.000
третій рядок,
"""


def test_parse_timestamp_hms_and_ms():
    assert parse_timestamp("00:00:10.500") == 10.5
    assert parse_timestamp("01:30:00.250") == 5400.25


def test_format_timestamp_roundtrips():
    assert format_timestamp(10.5) == "00:00:10.500"
    assert format_timestamp(5400.25) == "01:30:00.250"


def test_parse_vtt_reads_cues(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text(SAMPLE, encoding="utf-8")
    cues = parse_vtt(p)
    assert len(cues) == 2
    assert cues[0] == Cue(index=0, start=10.0, end=14.5, text="Перший рядок.\nДругий рядок.")
    assert cues[1].text == "третій рядок,"
    assert cues[1].start == 20.0


def test_parse_vtt_ignores_header_and_blank_blocks(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text("WEBVTT\n\nNOTE just a note\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")
    cues = parse_vtt(p)
    assert len(cues) == 1 and cues[0].text == "hi"


def test_parse_vtt_skips_cue_identifier_lines(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nhello\n", encoding="utf-8")
    cues = parse_vtt(p)
    assert len(cues) == 1 and cues[0].text == "hello"


def test_parse_vtt_ignores_note_block_with_arrow(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text(
        "WEBVTT\n\nNOTE arrows a --> b are overlays\n\n"
        "00:00:01.000 --> 00:00:02.000\nhi\n",
        encoding="utf-8",
    )
    cues = parse_vtt(p)
    assert len(cues) == 1 and cues[0].text == "hi"


def test_parse_header_preserves_kind_and_language(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text(SAMPLE, encoding="utf-8")
    assert parse_header(p) == "WEBVTT\nKind: captions\nLanguage: uk"
