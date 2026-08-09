"""Glossary loading and validation for ASR hotwords and mishearing repair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from gromit.exceptions import GlossaryError

ALLOWED_CATEGORIES = {"term", "person", "company", "product"}

# Whisper's decoder context caps hotwords well below a real glossary's size
# (see HOTWORD_TOKEN_BUDGET), so the budget goes to the categories ASR mangles
# most: names it has never seen. Generic terms go last — they are also the ones
# `crosscheck` can still repair afterwards via their `misheard` lists.
HOTWORD_CATEGORY_PRIORITY = {"person": 0, "company": 1, "product": 2, "term": 3}


@dataclass(frozen=True)
class GlossaryEntry:
    """A single glossary term: canonical form + known mishearings."""

    canonical: str
    category: str
    note: str | None
    misheard: tuple[str, ...]


@dataclass
class Glossary:
    """A validated collection of glossary entries."""

    entries: list[GlossaryEntry]

    def hotword_list(self) -> list[str]:
        """Canonical forms for ASR hotwords, most-worth-biasing first.

        Ordered by category (person, company, product, term), keeping file
        order within each. Whatever does not fit the model's prompt budget is
        dropped from the tail, so this ordering decides what survives.
        """
        return [
            e.canonical
            for e in sorted(
                self.entries,
                key=lambda e: HOTWORD_CATEGORY_PRIORITY.get(e.category, 3),
            )
        ]

    def misheard_index(self) -> dict[str, str]:
        """Map each lowercased misheard string to its canonical form."""
        index: dict[str, str] = {}
        for entry in self.entries:
            for m in entry.misheard:
                index[m.lower()] = entry.canonical
        return index


def load_glossary(path: Path) -> Glossary:
    """Parse and validate a single glossary YAML file.

    Raises:
        GlossaryError: file missing, malformed, or containing a duplicate
            canonical / a misheard string mapped under two entries.
    """
    if not path.exists():
        raise GlossaryError(f"Glossary file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise GlossaryError(f"Glossary YAML parse error in {path}: {e}") from e

    if not isinstance(data, dict) or "terms" not in data:
        raise GlossaryError(f"Glossary {path} must have a top-level 'terms:' list")
    terms = data["terms"]
    if not isinstance(terms, list):
        raise GlossaryError(f"Glossary {path}: 'terms' must be a list")

    entries: list[GlossaryEntry] = []
    seen_canonical: dict[str, str] = {}
    seen_misheard: dict[str, str] = {}

    for i, raw in enumerate(terms):
        if not isinstance(raw, dict) or "canonical" not in raw:
            raise GlossaryError(f"Glossary {path}: entry #{i} missing 'canonical'")
        canonical = str(raw["canonical"])
        category = str(raw.get("category", "term"))
        if category not in ALLOWED_CATEGORIES:
            raise GlossaryError(
                f"Glossary {path}: entry '{canonical}' has unknown category "
                f"'{category}' (allowed: {sorted(ALLOWED_CATEGORIES)})"
            )
        note_raw = raw.get("note")
        note = str(note_raw) if note_raw is not None else None

        misheard_raw = raw.get("misheard", []) or []
        if not isinstance(misheard_raw, list):
            raise GlossaryError(
                f"Glossary {path}: 'misheard' for '{canonical}' must be a list"
            )
        misheard = tuple(str(m) for m in misheard_raw)

        key = canonical.lower()
        if key in seen_canonical:
            raise GlossaryError(f"Glossary {path}: duplicate canonical '{canonical}'")
        seen_canonical[key] = canonical

        for m in misheard:
            mk = m.lower()
            if mk in seen_misheard and seen_misheard[mk] != canonical:
                raise GlossaryError(
                    f"Glossary {path}: misheard '{m}' maps to both "
                    f"'{seen_misheard[mk]}' and '{canonical}'"
                )
            seen_misheard[mk] = canonical

        entries.append(GlossaryEntry(canonical, category, note, misheard))

    return Glossary(entries)


def load_glossaries(paths: list[Path]) -> Glossary:
    """Load and merge multiple glossary files into one validated Glossary.

    Cross-file duplicate canonicals and conflicting misheard mappings are
    errors, exactly as within a single file.
    """
    merged: list[GlossaryEntry] = []
    seen_canonical: dict[str, str] = {}
    seen_misheard: dict[str, str] = {}
    for path in paths:
        g = load_glossary(path)
        for entry in g.entries:
            key = entry.canonical.lower()
            if key in seen_canonical:
                raise GlossaryError(
                    f"Duplicate canonical '{entry.canonical}' across glossary files"
                )
            seen_canonical[key] = entry.canonical
            for m in entry.misheard:
                mk = m.lower()
                if mk in seen_misheard and seen_misheard[mk] != entry.canonical:
                    raise GlossaryError(
                        f"Misheard '{m}' maps to both '{seen_misheard[mk]}' "
                        f"and '{entry.canonical}' across glossary files"
                    )
                seen_misheard[mk] = entry.canonical
            merged.append(entry)
    return Glossary(merged)
