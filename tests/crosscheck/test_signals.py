"""Tests for crosscheck flag signals and span merging."""

from gromit.crosscheck.gromit_json import GSegment, GWord
from gromit.crosscheck.signals import (
    Span,
    Thresholds,
    find_misheard,
    merge_spans,
    segment_flags,
)

TH = Thresholds()
MISHEARD = {"реліз чекліст": "release checklist", "attention": "extension"}


def _seg(text, avg_logprob=-0.1, word_ps=(0.9,)):
    words = tuple(GWord(w=text, start=0.0, end=1.0, p=p) for p in word_ps)
    return GSegment(0.0, 1.0, "S", text, avg_logprob, words)


def test_find_misheard_substring_returns_canonical():
    assert find_misheard("у нас реліз чекліст і", MISHEARD) == "release checklist"


def test_find_misheard_none_when_absent():
    assert find_misheard("нічого такого", MISHEARD) is None


def test_divergence_flag_when_texts_disagree():
    seg = _seg("соломія вербицька")
    reasons, _ = segment_flags(seg, "сьогодні тепло і сонячно", {}, True, TH)
    assert reasons == ["divergence"]


def test_no_divergence_when_texts_agree():
    seg = _seg("release checklist")
    reasons, _ = segment_flags(seg, "release checklist", {}, True, TH)
    assert "divergence" not in reasons


def test_low_confidence_from_word_cluster():
    # >= 2 low words trips it; a single low word does not (avoids over-flagging).
    seg = _seg("щось невиразне", avg_logprob=-0.1, word_ps=(0.2, 0.3))
    reasons, _ = segment_flags(seg, "щось невиразне", {}, True, TH)
    assert "low_confidence" in reasons


def test_single_low_word_does_not_flag():
    seg = _seg("майже добре", avg_logprob=-0.1, word_ps=(0.9, 0.2))
    reasons, _ = segment_flags(seg, "майже добре", {}, True, TH)
    assert "low_confidence" not in reasons


def test_low_confidence_from_avg_logprob():
    seg = _seg("щось", avg_logprob=-0.9, word_ps=(0.95,))
    reasons, _ = segment_flags(seg, "щось", {}, True, TH)
    assert "low_confidence" in reasons


def test_misheard_match_sets_suggestion():
    seg = _seg("у нас реліз чекліст")
    reasons, suggestion = segment_flags(seg, "", MISHEARD, False, TH)
    assert "misheard_match" in reasons
    assert suggestion == "release checklist"


def test_no_divergence_when_meet_absent():
    seg = _seg("соломія вербицька")
    reasons, _ = segment_flags(seg, "", {}, False, TH)
    assert "divergence" not in reasons


def test_reasons_order_is_stable():
    seg = _seg("реліз чекліст", avg_logprob=-0.9, word_ps=(0.2,))
    reasons, _ = segment_flags(seg, "зовсім інше", MISHEARD, True, TH)
    assert reasons == ["divergence", "low_confidence", "misheard_match"]


def _span(a, b, reasons, suggestion=None):
    return Span(a, b, "m", "g", list(reasons), suggestion)


def test_merge_joins_within_gap():
    spans = [_span(0.0, 2.0, ["divergence"]), _span(3.5, 5.0, ["low_confidence"])]
    merged = merge_spans(spans, merge_gap=2.0)  # gap 1.5 <= 2.0
    assert len(merged) == 1
    assert merged[0].start == 0.0 and merged[0].end == 5.0
    assert merged[0].reasons == ["divergence", "low_confidence"]


def test_merge_keeps_apart_beyond_gap():
    spans = [_span(0.0, 2.0, ["divergence"]), _span(10.0, 11.0, ["low_confidence"])]
    assert len(merge_spans(spans, merge_gap=2.0)) == 2


def test_merge_keeps_first_suggestion():
    spans = [_span(0.0, 2.0, ["misheard_match"], "release checklist"),
             _span(2.5, 3.0, ["misheard_match"], "extension")]
    merged = merge_spans(spans, merge_gap=2.0)
    assert len(merged) == 1
    assert merged[0].suggestion == "release checklist"


def test_merge_empty():
    assert merge_spans([], merge_gap=2.0) == []
