"""Unit tests for cue-level name voting."""
from __future__ import annotations

from gromit.nametag.attribution import UNKNOWN, CueResult, attribute_cue, vote_cue
from gromit.nametag.roster import NameMatch


def m(name, score=1.0, matched=True):
    return NameMatch(name=name, score=score, matched=matched)


def test_majority_roster_name_wins():
    r = vote_cue([m("Solomiya Verbytska"), m("Solomiya Verbytska"), m("Solomiya Verbytska")])
    assert isinstance(r, CueResult)
    assert r.name == "Solomiya Verbytska"
    assert r.frames_used == 3


def test_tie_broken_by_summed_score():
    r = vote_cue([m("A", score=0.85), m("B", score=0.95)])  # 1 vote each
    assert r.name == "B"


def test_all_none_is_unknown():
    assert vote_cue([None, None]).name == UNKNOWN


def test_offroster_spelling_variants_bucket_together():
    # an unlisted guest read slightly differently across frames must not split
    reads = [m("Nadiya Ivanets", score=0.6, matched=False),
             m("Nadiya Ivanet", score=0.6, matched=False),
             m("Nadiya Ivanets", score=0.6, matched=False)]
    r = vote_cue(reads)
    assert r.name.startswith("Nadiya Ivan")
    assert r.frames_used == 3


def test_matched_canonical_beats_unmatched_noise():
    reads = [m("Yaroslav Vyshnevetsky"), m("Yaroslav Vyshnevetsky"),
             m("xZ9", score=0.0, matched=False)]
    assert vote_cue(reads).name == "Yaroslav Vyshnevetsky"


def test_single_roster_match_owns_cue_over_garbles():
    # the "name over a busy graphic" case: 1 clean match + many sub-threshold
    # garbles that out-count it. Exactly one roster member matched -> they win.
    reads = [m("Yaroslav Vyshnevetsky", 0.85, True)]
    reads += [m(f"Yaros{c}hivvyshn", 0.7, False) for c in "abcde"]  # near-identical -> one bucket
    assert vote_cue(reads).name == "Yaroslav Vyshnevetsky"


def test_two_roster_matches_fall_back_to_count():
    # >=2 roster members matched (genuine ambiguity) -> count/score decides, not the rule
    r = vote_cue([m("Solomiya Verbytska"), m("Solomiya Verbytska"), m("Yaroslav Vyshnevetsky")])
    assert r.name == "Solomiya Verbytska"


def test_offroster_guest_not_forced_to_roster_when_no_match():
    # no >=0.80 match anywhere -> guest stays verbatim (the single-match rule must NOT fire)
    reads = [m("Nadiya Ivanets", 0.6, False)] * 3
    assert vote_cue(reads).name.startswith("Nadiya Ivan")


def reader_from(table):
    """Return a frame_reader that maps a frame path -> a preset NameMatch|None."""
    def read(path, candidates):
        return table.get(str(path))
    return read


def test_attribute_cue_votes_over_frames():
    frames = [(0.0, "a"), (0.5, "b"), (1.0, "c")]
    table = {
        "a": NameMatch("Solomiya Verbytska", 1.0, True),
        "b": NameMatch("Solomiya Verbytska", 1.0, True),
        "c": None,
    }
    res = attribute_cue(frames, ["Solomiya Verbytska"], reader_from(table))
    assert res.name == "Solomiya Verbytska" and res.frames_used == 2


def test_attribute_cue_early_stop_short_circuits_reads():
    seen = []

    def read(path, candidates):
        seen.append(path)
        return NameMatch("Yaroslav Vyshnevetsky", 1.0, True)

    frames = [(i * 0.5, f"f{i}") for i in range(7)]
    res = attribute_cue(frames, ["Yaroslav Vyshnevetsky"], read, early_stop=True)
    assert res.name == "Yaroslav Vyshnevetsky"
    assert len(seen) == 4  # strict majority of 7 reached after the 4th matching read
