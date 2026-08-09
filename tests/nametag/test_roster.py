"""Unit tests for Stage 2b roster matching + loading."""
from __future__ import annotations

from gromit.nametag.roster import clean_name, load_roster, match_name, rank_candidates

PERM = [
    "Yaroslav Vyshnevetsky",
    "Solomiya Verbytska",
    "Mykola H",
    "Ярослав Вишневецький",
]


def test_exact_match_scores_one():
    m = match_name("Yaroslav Vyshnevetsky", PERM)
    assert m.matched and m.name == "Yaroslav Vyshnevetsky" and m.score == 1.0


def test_truncated_unicode_ellipsis_recovers_full_name():
    m = match_name("Yaroslav Vyshn…", PERM)
    assert m.matched and m.name == "Yaroslav Vyshnevetsky"


def test_truncated_ascii_ellipsis_recovers_full_name():
    m = match_name("Yaroslav Vyshn...", PERM)
    assert m.matched and m.name == "Yaroslav Vyshnevetsky"


def test_ocr_misread_within_threshold_matches():
    m = match_name("Yarcslav Vyshnevetsky", PERM)  # o->c OCR slip
    assert m.matched and m.name == "Yaroslav Vyshnevetsky"


def test_off_roster_falls_back_verbatim():
    m = match_name("Zachary Quinto", PERM)
    assert not m.matched and m.name == "Zachary Quinto"


def test_cyrillic_name_matches():
    m = match_name("Ярослав Вишневецький", PERM)
    assert m.matched and m.name == "Ярослав Вишневецький"


def test_confusable_pair_resolves_to_correct_name():
    cands = ["Solomiya Verbytska", "Solomia Kravets"]
    assert match_name("Solomia Kravets", cands).name == "Solomia Kravets"
    assert match_name("Solomiya Verbytska", cands).name == "Solomiya Verbytska"


def test_verbatim_fallback_is_ellipsis_stripped():
    m = match_name("Nadiya Iva…", ["Yaroslav Vyshnevetsky"])
    assert not m.matched and m.name == "Nadiya Iva"


def test_trailing_junk_after_name_still_matches():
    # the crop caught extra text after the name (a t-shirt logo / role label); the
    # leading name must still win (front-anchored similarity ignores the tail).
    assert match_name("Mykola H LOGO", ["Mykola H", "Solomiya Verbytska"]).name == "Mykola H"
    assert match_name("Yaroslav Vyshnevetsky SPEAKER", PERM).name == "Yaroslav Vyshnevetsky"


def test_trailing_junk_does_not_rescue_a_wrong_name():
    # garbled leading text must still fall through to verbatim (front errors count)
    assert not match_name("Zxcvb Qwerty LOGO", PERM).matched


def test_clean_name_collapses_ws_and_strips_ellipsis():
    assert clean_name("  Yaroslav   Vyshnevetsky … ") == "Yaroslav Vyshnevetsky"


def test_rank_candidates_orders_by_score():
    ranked = rank_candidates("Yaroslav Vyshn", PERM, top=2)
    assert ranked[0][0] == "Yaroslav Vyshnevetsky" and ranked[0][1] == 1.0  # truncation
    assert len(ranked) == 2 and ranked[0][1] >= ranked[1][1]


def test_very_short_prefix_does_not_false_match():
    # a stray/truncated 1-2 char reading must NOT bind to a roster name by prefix
    assert not match_name("Y", PERM).matched
    assert not match_name("So", PERM).matched


def test_load_roster_reads_permanent(tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text(
        "permanent:\n  - Yaroslav Vyshnevetsky\n  - Solomia Kravets\n",
        encoding="utf-8",
    )
    r = load_roster(p)
    assert r.permanent == ["Yaroslav Vyshnevetsky", "Solomia Kravets"]
    assert match_name("Solomia Kravets", r.permanent).matched
