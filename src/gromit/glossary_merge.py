"""Merge review corrections back into a per-project glossary.yaml.

Uses ruamel.yaml round-trip so the curated file keeps its comments and order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ

from gromit.exceptions import GlossaryError


@dataclass(frozen=True)
class Correction:
    canonical: str
    heard: str
    category: str = "term"


@dataclass
class MergeSummary:
    added_entries: list[str] = field(default_factory=list)
    added_misheard: list[tuple[str, str]] = field(default_factory=list)
    unchanged: int = 0


def load_corrections(path: Path) -> list[Correction]:
    """Parse a corrections.yaml into Correction objects."""
    if not path.exists():
        raise GlossaryError(f"corrections file not found: {path}")
    yaml = YAML(typ="safe")
    data = yaml.load(path.read_text()) or {}
    items = data.get("corrections") or []
    out: list[Correction] = []
    for it in items:
        if not it.get("canonical") or not it.get("heard"):
            raise GlossaryError(f"{path}: each correction needs canonical + heard")
        out.append(Correction(str(it["canonical"]), str(it["heard"]),
                               str(it.get("category", "term"))))
    return out


def merge_corrections(glossary_path: Path, corrections: list[Correction]) -> MergeSummary:
    """Fold corrections into glossary_path (in place), preserving comments/order."""
    yaml = YAML()  # round-trip
    yaml.preserve_quotes = True
    doc = yaml.load(glossary_path.read_text())
    terms = doc.setdefault("terms", [])

    by_canonical = {str(e["canonical"]): e for e in terms}
    # heard(lower) -> canonical, across the whole glossary (conflict detection)
    heard_owner: dict[str, str] = {}
    for e in terms:
        for m in e.get("misheard") or []:
            heard_owner[str(m).lower()] = str(e["canonical"])

    summary = MergeSummary()
    for c in corrections:
        owner = heard_owner.get(c.heard.lower())
        if owner is not None and owner != c.canonical:
            raise GlossaryError(
                f"conflict: heard '{c.heard}' already maps to '{owner}', "
                f"not '{c.canonical}'"
            )

        entry = by_canonical.get(c.canonical)
        if entry is None:
            entry = {
                "canonical": DQ(c.canonical),
                "category": c.category,
                "misheard": [DQ(c.heard)],
            }
            terms.append(entry)
            by_canonical[c.canonical] = entry
            heard_owner[c.heard.lower()] = c.canonical
            summary.added_entries.append(c.canonical)
            summary.added_misheard.append((c.canonical, c.heard))
            continue

        misheard = entry.setdefault("misheard", [])
        if any(str(m).lower() == c.heard.lower() for m in misheard):
            summary.unchanged += 1
        else:
            misheard.append(DQ(c.heard))
            heard_owner[c.heard.lower()] = c.canonical
            summary.added_misheard.append((c.canonical, c.heard))

    yaml.dump(doc, glossary_path)
    return summary
