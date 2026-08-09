"""Load the Step-1 .gromit.json transcript into typed structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gromit.exceptions import CrosscheckError


@dataclass(frozen=True)
class GWord:
    w: str
    start: float
    end: float
    p: float


@dataclass(frozen=True)
class GSegment:
    start: float
    end: float
    speaker: str
    text: str
    avg_logprob: float
    words: tuple[GWord, ...]


@dataclass(frozen=True)
class GromitTranscript:
    language: str
    model: str
    hotwords_from: tuple[str, ...]
    segments: tuple[GSegment, ...]


def load_gromit_json(path: Path) -> GromitTranscript:
    """Parse a .gromit.json file. Raises CrosscheckError if missing/malformed."""
    if not path.exists():
        raise CrosscheckError(f"gromit JSON not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CrosscheckError(f"gromit JSON parse error in {path}: {e}") from e
    if not isinstance(data, dict) or "segments" not in data:
        raise CrosscheckError(f"{path}: not a gromit transcript ('segments' missing)")

    segments = []
    for s in data["segments"]:
        words = tuple(
            GWord(w=w["w"], start=w["start"], end=w["end"], p=w["p"])
            for w in s.get("words", [])
        )
        segments.append(
            GSegment(
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker", "UNKNOWN"),
                text=s.get("text", ""),
                avg_logprob=s.get("avg_logprob", 0.0),
                words=words,
            )
        )
    return GromitTranscript(
        language=data.get("language", ""),
        model=data.get("model", ""),
        hotwords_from=tuple(data.get("hotwords_from", [])),
        segments=tuple(segments),
    )
