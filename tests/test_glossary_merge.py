"""Tests for corrections loading and glossary merging."""

import pytest

from gromit.exceptions import GlossaryError
from gromit.glossary import load_glossary
from gromit.glossary_merge import Correction, load_corrections, merge_corrections

GLOSSARY = """\
# Curated glossary — keep this comment!
terms:
  - canonical: "release checklist"
    category: term
    misheard: ["реліз чекліст"]
  - canonical: "Вишневецький"
    category: person
    misheard: []
"""


def _glossary(tmp_path):
    p = tmp_path / "glossary.yaml"
    p.write_text(GLOSSARY)
    return p


def _corr(tmp_path, body):
    p = tmp_path / "corrections.yaml"
    p.write_text(body)
    return p


def test_load_corrections(tmp_path):
    p = _corr(tmp_path,
        'corrections:\n  - canonical: "extension"\n    heard: "ітеншн"\n    category: term\n')
    got = load_corrections(p)
    assert got == [Correction("extension", "ітеншн", "term")]


def test_merge_appends_to_existing_entry(tmp_path):
    gp = _glossary(tmp_path)
    summary = merge_corrections(gp, [Correction("Вишневецький", "Кишневецьки", "person")])
    assert ("Вишневецький", "Кишневецьки") in summary.added_misheard
    g = load_glossary(gp)
    entry = next(e for e in g.entries if e.canonical == "Вишневецький")
    assert "Кишневецьки" in entry.misheard
    assert "keep this comment" in gp.read_text()  # comments preserved


def test_merge_creates_new_entry(tmp_path):
    gp = _glossary(tmp_path)
    summary = merge_corrections(gp, [Correction("extension", "ітеншн", "term")])
    assert "extension" in summary.added_entries
    g = load_glossary(gp)
    assert any(e.canonical == "extension" and "ітеншн" in e.misheard for e in g.entries)


def test_merge_is_idempotent(tmp_path):
    gp = _glossary(tmp_path)
    c = [Correction("release checklist", "реліз чекліст", "term")]  # already present
    summary = merge_corrections(gp, c)
    assert summary.added_misheard == []
    assert summary.unchanged == 1


def test_merge_conflicting_heard_errors(tmp_path):
    gp = _glossary(tmp_path)
    # "реліз чекліст" already maps to "release checklist"
    with pytest.raises(GlossaryError, match="conflict|different"):
        merge_corrections(gp, [Correction("Вишневецький", "реліз чекліст", "person")])
