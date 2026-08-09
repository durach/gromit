"""Load and rank the Step-2 flags.json for the review page."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gromit.exceptions import CrosscheckError

REASON_PRIORITY: dict[str, int] = {
    "misheard_match": 0,
    "divergence": 1,
    "low_confidence": 2,
}


@dataclass(frozen=True)
class FlagSpan:
    start: float
    end: float
    meet_text: str
    gromit_text: str
    reasons: tuple[str, ...]
    suggestion: str | None


def load_flags(path: Path) -> list[FlagSpan]:
    """Parse flags.json into FlagSpan objects. Raises CrosscheckError if malformed."""
    if not path.exists():
        raise CrosscheckError(f"flags.json not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CrosscheckError(f"flags.json parse error in {path}: {e}") from e
    if not isinstance(data, dict) or "spans" not in data:
        raise CrosscheckError(f"{path}: not a flags file ('spans' missing)")
    return [
        FlagSpan(
            start=s["start"],
            end=s["end"],
            meet_text=s.get("meet_text", ""),
            gromit_text=s.get("gromit_text", ""),
            reasons=tuple(s.get("reasons", [])),
            suggestion=s.get("suggestion"),
        )
        for s in data["spans"]
    ]


def rank_key(span: FlagSpan) -> tuple[int, float]:
    """Sort key: best (lowest) reason priority, then start time."""
    best = min((REASON_PRIORITY.get(r, 9) for r in span.reasons), default=9)
    return (best, span.start)
