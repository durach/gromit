"""Tests for loading the .gromit.json transcript."""

import json

import pytest

from gromit.crosscheck.gromit_json import load_gromit_json
from gromit.exceptions import CrosscheckError

SAMPLE = {
    "language": "uk",
    "model": "large-v3",
    "hotwords_from": ["data/glossary.yaml"],
    "segments": [
        {
            "start": 0.21, "end": 1.81, "speaker": "SPEAKER_02",
            "text": "Соломія Вербицька", "avg_logprob": -0.30,
            "words": [{"w": " Соломія", "start": 0.21, "end": 1.81, "p": 0.40}],
        }
    ],
}


def _write(tmp_path, obj):
    p = tmp_path / "x.gromit.json"
    p.write_text(json.dumps(obj, ensure_ascii=False))
    return p


def test_load_valid(tmp_path):
    t = load_gromit_json(_write(tmp_path, SAMPLE))
    assert t.language == "uk"
    assert t.model == "large-v3"
    assert t.hotwords_from == ("data/glossary.yaml",)
    assert len(t.segments) == 1
    seg = t.segments[0]
    assert seg.speaker == "SPEAKER_02"
    assert seg.text == "Соломія Вербицька"
    assert seg.avg_logprob == -0.30
    assert seg.words[0].w == " Соломія"
    assert seg.words[0].p == 0.40


def test_missing_file_errors(tmp_path):
    with pytest.raises(CrosscheckError, match="not found"):
        load_gromit_json(tmp_path / "nope.json")


def test_missing_segments_key_errors(tmp_path):
    with pytest.raises(CrosscheckError, match="segments"):
        load_gromit_json(_write(tmp_path, {"language": "uk"}))
