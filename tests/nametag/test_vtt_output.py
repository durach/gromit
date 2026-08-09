"""Unit tests for the named-VTT and annotation writers."""
from __future__ import annotations

from gromit.nametag.vtt import Cue
from gromit.nametag.vtt_output import write_annotation, write_named_vtt


def cues():
    return [
        Cue(0, 10.0, 14.5, "Перший рядок.\nДругий рядок."),
        Cue(1, 20.0, 25.0, "третій рядок,"),
    ]


def test_named_vtt_prefixes_name_on_first_line(tmp_path):
    out = tmp_path / "x.named.vtt"
    write_named_vtt(cues(), ["Solomiya Verbytska", "Unknown"], out)
    body = out.read_text(encoding="utf-8")
    assert body.startswith("WEBVTT")
    assert "00:00:10.000 --> 00:00:14.500" in body
    assert "Solomiya Verbytska: Перший рядок." in body
    assert "\nДругий рядок." in body   # second line kept, not prefixed
    assert "Unknown: третій рядок," in body


def test_annotation_groups_consecutive_speakers(tmp_path):
    out = tmp_path / "x.named.txt"
    cs = [Cue(0, 13.0, 15.0, "a"), Cue(1, 15.0, 17.0, "b"), Cue(2, 17.0, 19.0, "c")]
    write_annotation(cs, ["Solomiya Verbytska", "Solomiya Verbytska", "Yaroslav Vyshnevetsky"], out)
    body = out.read_text(encoding="utf-8")
    assert "[00:00:13] Solomiya Verbytska:\na b" in body
    assert "[00:00:17] Yaroslav Vyshnevetsky:\nc" in body
