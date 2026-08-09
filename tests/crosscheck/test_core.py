"""Tests for the crosscheck orchestrator + flags.json writer."""

import json

import pytest

from gromit.crosscheck.core import run_crosscheck, write_flags_json
from gromit.exceptions import CrosscheckError


def _gromit(tmp_path, segments):
    p = tmp_path / "r.gromit.json"
    p.write_text(json.dumps(
        {"language": "uk", "model": "large-v3", "hotwords_from": [], "segments": segments},
        ensure_ascii=False,
    ))
    return p


def _vtt(tmp_path, body):
    p = tmp_path / "r.vtt"
    p.write_text("WEBVTT\nKind: captions\nLanguage: uk\n\n" + body)
    return p


def _glossary(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text('terms:\n  - canonical: "release checklist"\n    misheard: ["реліз чекліст"]\n')
    return p


def test_divergence_span_from_real_disagreement(tmp_path):
    gp = _gromit(tmp_path, [
        {"start": 0.0, "end": 3.0, "speaker": "S", "text": "соломія вербицька",
         "avg_logprob": -0.1, "words": [{"w": "соломія", "start": 0.0, "end": 3.0, "p": 0.9}]},
    ])
    vp = _vtt(tmp_path, "00:00:00.000 --> 00:00:03.000\nсьогодні тепло і сонячно\n")
    spans = run_crosscheck(gp, vp, [])
    assert len(spans) == 1
    assert "divergence" in spans[0].reasons
    assert spans[0].gromit_text == "соломія вербицька"
    assert "сьогодні" in spans[0].meet_text


def test_misheard_match_with_glossary_no_meet(tmp_path):
    gp = _gromit(tmp_path, [
        {"start": 5.0, "end": 7.0, "speaker": "S", "text": "у нас реліз чекліст",
         "avg_logprob": -0.1, "words": [{"w": "у", "start": 5.0, "end": 7.0, "p": 0.9}]},
    ])
    spans = run_crosscheck(gp, None, [_glossary(tmp_path)])
    assert len(spans) == 1
    assert spans[0].reasons == ["misheard_match"]
    assert spans[0].suggestion == "release checklist"


def test_clean_segment_produces_no_span(tmp_path):
    gp = _gromit(tmp_path, [
        {"start": 0.0, "end": 3.0, "speaker": "S", "text": "release checklist",
         "avg_logprob": -0.1, "words": [{"w": "m", "start": 0.0, "end": 3.0, "p": 0.95}]},
    ])
    vp = _vtt(tmp_path, "00:00:00.000 --> 00:00:03.000\nrelease checklist\n")
    assert run_crosscheck(gp, vp, []) == []


def test_bad_pairing_raises(tmp_path):
    # gromit segment near t=0, Meet cue far away -> ~0% overlap
    gp = _gromit(tmp_path, [
        {"start": 0.0, "end": 3.0, "speaker": "S", "text": "щось",
         "avg_logprob": -0.1, "words": [{"w": "щось", "start": 0.0, "end": 3.0, "p": 0.9}]},
    ])
    vp = _vtt(tmp_path, "01:00:00.000 --> 01:00:03.000\nзовсім інша розмова\n")
    with pytest.raises(CrosscheckError, match="overlap"):
        run_crosscheck(gp, vp, [])


def test_write_flags_json_schema(tmp_path):
    gp = _gromit(tmp_path, [
        {"start": 0.0, "end": 3.0, "speaker": "S", "text": "соломія вербицька",
         "avg_logprob": -0.1, "words": [{"w": "соломія", "start": 0.0, "end": 3.0, "p": 0.9}]},
    ])
    vp = _vtt(tmp_path, "00:00:00.000 --> 00:00:03.000\nсьогодні тепло і сонячно\n")
    spans = run_crosscheck(gp, vp, [])
    out = tmp_path / "flags.json"
    write_flags_json(out, spans)
    payload = json.loads(out.read_text())
    assert set(payload) == {"spans"}
    s = payload["spans"][0]
    assert set(s) == {"start", "end", "meet_text", "gromit_text", "reasons", "suggestion"}
    assert "соломія" in s["gromit_text"]  # Cyrillic not escaped
