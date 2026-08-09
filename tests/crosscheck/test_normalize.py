"""Tests for crosscheck text normalization."""

from gromit.crosscheck.normalize import normalize_text, token_containment, tokens


def test_normalize_lowercases_strips_punct_and_newlines():
    assert normalize_text("Привіт,\nсвіт!  ") == "привіт світ"


def test_normalize_keeps_digits():
    assert normalize_text("+500 на 225 місці") == "500 на 225 місці"


def test_tokens_drop_fillers():
    # "ее" and "ну" are fillers; real words stay
    assert tokens("теплим ее вітром ну добре") == ["теплим", "вітром", "добре"]


def test_token_containment_identical():
    assert token_containment("release checklist", "release checklist") == 1.0


def test_token_containment_gromit_contained_in_wider_meet():
    # The key asymmetry: a short gromit segment fully inside a wider Meet
    # window must NOT read as divergence — all of A's tokens are in B.
    assert token_containment("соломія тут", "сьогодні соломія тут разом з усіма і") == 1.0


def test_token_containment_disjoint():
    assert token_containment("соломія вербицька", "сьогодні тепло і сонячно") == 0.0


def test_token_containment_partial():
    # A={a,b,c,d}, present in B={a,b,x}: 2/4 = 0.5
    assert token_containment("a b c d", "a b x") == 0.5


def test_token_containment_empty_a_is_one():
    assert token_containment("", "будь-що") == 1.0
