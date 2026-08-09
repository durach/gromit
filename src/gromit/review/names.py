"""Resolve a span's speaker from the nametag .named.vtt (Name: prefix)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gromit.nametag.vtt import parse_vtt


@dataclass(frozen=True)
class NamedCue:
    start: float
    end: float
    name: str


def load_named_cues(path: Path) -> list[NamedCue]:
    """Parse a .named.vtt, taking each cue's leading 'Name: ' as the speaker."""
    out: list[NamedCue] = []
    for cue in parse_vtt(path):
        first = cue.text.split("\n", 1)[0]
        name = first.split(": ", 1)[0].strip() if ": " in first else ""
        out.append(NamedCue(cue.start, cue.end, name))
    return out


def name_for(start: float, end: float, named: list[NamedCue]) -> str:
    """Name of the cue with the largest overlap of [start, end]; '' if none."""
    best_name = ""
    best_overlap = 0.0
    for c in named:
        overlap = min(end, c.end) - max(start, c.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = c.name
    return best_name
