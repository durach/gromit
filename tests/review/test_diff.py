"""Tests for token diff highlighting."""

from gromit.review.diff import highlight


def test_identical_has_no_marks():
    m, g = highlight("release checklist", "release checklist")
    assert "<mark>" not in m and "<mark>" not in g


def test_differing_token_is_marked_on_both_sides():
    m, g = highlight("реліз чекліст", "release checklist")
    assert "<mark>реліз</mark>" in m
    assert "<mark>release</mark>" in g


def test_shared_token_not_marked():
    m, g = highlight("у нас extension там", "у нас attention там")
    assert "<mark>extension</mark>" in m
    assert "у нас" in g and "<mark>у</mark>" not in g  # shared words stay plain


def test_html_escaped():
    m, _g = highlight("a <b>", "c <b>")
    assert "&lt;b&gt;" in m  # angle brackets escaped, not raw
    assert "<mark>a</mark>" in m


def test_empty_meet_side():
    m, g = highlight("", "щось нове")
    assert m == ""
    assert "<mark>щось</mark>" in g  # everything on gromit side differs
