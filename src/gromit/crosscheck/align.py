"""Time-axis alignment between Google Meet cues and gromit segments.

Both are timestamped against the same recording, so alignment is pure interval
overlap — no DTW. Google Meet emits a *rolling* caption window, so cues overlap
each other; a single gromit segment can therefore span several Meet cues.
"""

from __future__ import annotations

from gromit.nametag.vtt import Cue


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def meet_text_for(seg_start: float, seg_end: float, cues: list[Cue]) -> str:
    """Space-joined text of the (unique, in-order) Meet cues overlapping the window."""
    parts: list[str] = []
    for c in cues:
        if _overlaps(seg_start, seg_end, c.start, c.end):
            t = " ".join(c.text.split())  # flatten newlines/extra spaces
            if t and t not in parts:
                parts.append(t)
    return " ".join(parts)


def overlap_fraction(cues: list[Cue], segments) -> float:
    """Fraction of Meet cues overlapping at least one gromit segment (0.0 if none)."""
    if not cues:
        return 0.0
    hit = sum(
        1
        for c in cues
        if any(_overlaps(c.start, c.end, s.start, s.end) for s in segments)
    )
    return hit / len(cues)
