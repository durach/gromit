"""Tests for speaker-name lookup from a nametag .named.vtt."""

from gromit.review.names import NamedCue, load_named_cues, name_for

NAMED = """WEBVTT
Kind: captions
Language: uk

00:00:01.000 --> 00:00:05.000
Yaroslav Vyshnevetsky: Перший рядок.

00:00:06.000 --> 00:00:10.000
Mykola: Другий рядок.
"""


def test_load_named_cues_extracts_name_prefix(tmp_path):
    p = tmp_path / "r.named.vtt"
    p.write_text(NAMED)
    cues = load_named_cues(p)
    assert cues[0] == NamedCue(1.0, 5.0, "Yaroslav Vyshnevetsky")
    assert cues[1].name == "Mykola"


def test_name_for_picks_best_overlap():
    named = [NamedCue(1.0, 5.0, "Yaroslav Vyshnevetsky"), NamedCue(6.0, 10.0, "Mykola")]
    assert name_for(6.5, 9.0, named) == "Mykola"


def test_name_for_empty_when_no_overlap():
    named = [NamedCue(1.0, 5.0, "Yaroslav Vyshnevetsky")]
    assert name_for(50.0, 51.0, named) == ""


def test_name_for_no_named_cues():
    assert name_for(1.0, 2.0, []) == ""
