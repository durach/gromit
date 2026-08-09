"""Stage 3: parse a Google Meet WebVTT captions file into timed cues.

A cue is one ``HH:MM:SS.mmm --> HH:MM:SS.mmm`` block plus its text lines (the
original line breaks are preserved so the named-VTT writer can re-emit them).
Header lines (WEBVTT / Kind / Language / NOTE) and blank blocks are skipped.
The file carries no speaker information — that is what Stage 3 supplies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIMING = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}\s*-->")


@dataclass(frozen=True)
class Cue:
    index: int
    start: float  # seconds from video t=0
    end: float    # seconds
    text: str     # original line breaks preserved ("\n"-joined)


def parse_timestamp(ts: str) -> float:
    """``HH:MM:SS.mmm`` or ``MM:SS.mmm`` -> seconds (float)."""
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS.mmm`` (the WebVTT timestamp form)."""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_header(path: str | Path) -> str:
    """Return the WebVTT header block — the lines before the first cue, e.g.
    ``WEBVTT\\nKind: captions\\nLanguage: uk``. Falls back to ``WEBVTT``."""
    raw = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    first = raw.split("\n\n", 1)[0].strip()
    return first if first.startswith("WEBVTT") else "WEBVTT"


def parse_vtt(path: str | Path) -> list[Cue]:
    """Read a WebVTT file and return its cues in order (header/NOTE blocks skipped)."""
    raw = Path(path).read_text(encoding="utf-8")
    cues: list[Cue] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        # find the timing line (skip an optional cue-id line before it)
        time_idx = next((i for i, ln in enumerate(lines) if _TIMING.match(ln)), None)
        if time_idx is None:
            continue  # header / NOTE / blank block
        start_s, _, end_s = lines[time_idx].partition("-->")
        text = "\n".join(lines[time_idx + 1:])
        cues.append(Cue(
            index=len(cues),
            start=parse_timestamp(start_s),
            end=parse_timestamp(end_s),
            text=text,
        ))
    return cues
