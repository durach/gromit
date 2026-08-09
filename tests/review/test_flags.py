"""Tests for loading + ranking flags.json."""

import json

import pytest

from gromit.exceptions import CrosscheckError
from gromit.review.flags import FlagSpan, load_flags, rank_key


def _write(tmp_path, spans):
    p = tmp_path / "flags.json"
    p.write_text(json.dumps({"spans": spans}, ensure_ascii=False))
    return p


def test_load_flags(tmp_path):
    p = _write(tmp_path, [
        {"start": 1.0, "end": 2.0, "meet_text": "m", "gromit_text": "g",
         "reasons": ["divergence"], "suggestion": None},
    ])
    spans = load_flags(p)
    assert len(spans) == 1
    assert spans[0] == FlagSpan(1.0, 2.0, "m", "g", ("divergence",), None)


def test_load_flags_malformed(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"nope": 1}')
    with pytest.raises(CrosscheckError, match="spans"):
        load_flags(p)


def test_rank_key_orders_misheard_before_divergence_before_lowconf():
    a = FlagSpan(9.0, 9.0, "", "", ("low_confidence",), None)
    b = FlagSpan(50.0, 51.0, "", "", ("misheard_match",), "x")
    c = FlagSpan(3.0, 4.0, "", "", ("divergence",), None)
    ordered = sorted([a, b, c], key=rank_key)
    assert [s.reasons[0] for s in ordered] == ["misheard_match", "divergence", "low_confidence"]


def test_rank_key_uses_best_reason_then_start():
    both = FlagSpan(80.0, 81.0, "", "", ("divergence", "misheard_match"), "x")
    assert rank_key(both) == (0, 80.0)  # misheard_match wins the priority
