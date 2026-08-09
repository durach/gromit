"""Tests for glossary loading and validation."""

from pathlib import Path

import pytest

from gromit.exceptions import GlossaryError
from gromit.glossary import load_glossaries, load_glossary

VALID = """\
terms:
  - canonical: "release checklist"
    category: term
    note: "кроки перед релізом"
    misheard: ["реліз чекліст", "реліс чикліст"]
  - canonical: "Вишневецький"
    category: person
    misheard: ["Вишневський", "Кишневецький"]
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "glossary.yaml"
    p.write_text(text)
    return p


def test_load_valid_glossary(tmp_path):
    g = load_glossary(_write(tmp_path, VALID))
    assert len(g.entries) == 2
    assert g.entries[0].canonical == "release checklist"
    assert g.entries[0].note == "кроки перед релізом"
    assert g.entries[1].category == "person"


def test_hotword_list_puts_proper_nouns_first(tmp_path):
    # Whisper truncates hotwords hard, so names must outrank generic terms.
    g = load_glossary(_write(tmp_path, VALID))
    assert g.hotword_list() == ["Вишневецький", "release checklist"]


def test_hotword_list_orders_by_category_then_file_order(tmp_path):
    text = (
        "terms:\n"
        '  - canonical: "стендап"\n    category: term\n'
        '  - canonical: "Dealflow"\n    category: product\n'
        '  - canonical: "Соломія"\n    category: person\n'
        '  - canonical: "Acme"\n    category: company\n'
        '  - canonical: "Nimbus"\n    category: product\n'
        '  - canonical: "Вербицька"\n    category: person\n'
    )
    g = load_glossary(_write(tmp_path, text))
    assert g.hotword_list() == [
        "Соломія",
        "Вербицька",
        "Acme",
        "Dealflow",
        "Nimbus",
        "стендап",
    ]


def test_misheard_index_lowercased(tmp_path):
    g = load_glossary(_write(tmp_path, VALID))
    idx = g.misheard_index()
    assert idx["реліз чекліст"] == "release checklist"
    assert idx["вишневський"] == "Вишневецький"


def test_missing_file_errors(tmp_path):
    with pytest.raises(GlossaryError, match="not found"):
        load_glossary(tmp_path / "nope.yaml")


def test_missing_terms_key_errors(tmp_path):
    with pytest.raises(GlossaryError, match="terms"):
        load_glossary(_write(tmp_path, "other: 1\n"))


def test_entry_missing_canonical_errors(tmp_path):
    text = "terms:\n  - category: term\n"
    with pytest.raises(GlossaryError, match="canonical"):
        load_glossary(_write(tmp_path, text))


def test_unknown_category_errors(tmp_path):
    text = 'terms:\n  - canonical: "x"\n    category: bogus\n'
    with pytest.raises(GlossaryError, match="category"):
        load_glossary(_write(tmp_path, text))


def test_duplicate_canonical_errors(tmp_path):
    text = 'terms:\n  - canonical: "Foo"\n  - canonical: "foo"\n'
    with pytest.raises(GlossaryError, match="duplicate canonical"):
        load_glossary(_write(tmp_path, text))


def test_misheard_under_two_entries_errors(tmp_path):
    text = (
        "terms:\n"
        '  - canonical: "A"\n    misheard: ["x"]\n'
        '  - canonical: "B"\n    misheard: ["X"]\n'
    )
    with pytest.raises(GlossaryError, match="misheard"):
        load_glossary(_write(tmp_path, text))


def test_load_glossaries_merges(tmp_path):
    p1 = tmp_path / "a.yaml"
    p1.write_text('terms:\n  - canonical: "A"\n')
    p2 = tmp_path / "b.yaml"
    p2.write_text('terms:\n  - canonical: "B"\n')
    g = load_glossaries([p1, p2])
    assert g.hotword_list() == ["A", "B"]


def test_load_glossaries_cross_file_duplicate_errors(tmp_path):
    p1 = tmp_path / "a.yaml"
    p1.write_text('terms:\n  - canonical: "Dup"\n')
    p2 = tmp_path / "b.yaml"
    p2.write_text('terms:\n  - canonical: "dup"\n')
    with pytest.raises(GlossaryError, match="[Dd]uplicate canonical"):
        load_glossaries([p1, p2])
