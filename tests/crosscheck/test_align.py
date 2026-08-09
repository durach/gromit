"""Tests for Meet-cue ↔ gromit-segment time alignment."""

from gromit.crosscheck.align import meet_text_for, overlap_fraction
from gromit.crosscheck.gromit_json import GSegment
from gromit.nametag.vtt import Cue


def _cue(i, a, b, text):
    return Cue(index=i, start=a, end=b, text=text)


def _seg(a, b):
    return GSegment(start=a, end=b, speaker="S", text="", avg_logprob=0.0, words=())


def test_meet_text_gathers_overlapping_cues_in_order():
    cues = [_cue(0, 0.0, 5.0, "перший"), _cue(1, 4.0, 9.0, "другий"),
            _cue(2, 20.0, 25.0, "далеко")]
    # window 3..6 overlaps cue0 and cue1, not cue2
    assert meet_text_for(3.0, 6.0, cues) == "перший другий"


def test_meet_text_dedupes_identical_cue_text():
    cues = [_cue(0, 0.0, 5.0, "той самий"), _cue(1, 4.0, 9.0, "той самий")]
    assert meet_text_for(3.0, 6.0, cues) == "той самий"


def test_meet_text_joins_multiline_cue():
    cues = [_cue(0, 0.0, 5.0, "рядок один\nрядок два")]
    assert meet_text_for(1.0, 2.0, cues) == "рядок один рядок два"


def test_meet_text_empty_when_no_overlap():
    assert meet_text_for(100.0, 101.0, [_cue(0, 0.0, 5.0, "x")]) == ""


def test_overlap_fraction_counts_cues_hitting_any_segment():
    cues = [_cue(0, 0.0, 2.0, "a"), _cue(1, 3.0, 4.0, "b"),
            _cue(2, 50.0, 51.0, "c")]  # c overlaps nothing
    segs = [_seg(0.0, 5.0)]
    assert overlap_fraction(cues, segs) == 2 / 3


def test_overlap_fraction_no_cues_is_zero():
    assert overlap_fraction([], [_seg(0.0, 5.0)]) == 0.0
