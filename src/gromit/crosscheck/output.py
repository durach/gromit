"""Write the crosscheck flags.json (spans needing human review)."""

from __future__ import annotations

import json
from pathlib import Path

from gromit.crosscheck.signals import Span


def write_flags_json(path: Path, spans: list[Span]) -> None:
    """Serialize flagged spans to `path` (UTF-8, Cyrillic preserved)."""
    payload = {
        "spans": [
            {
                "start": s.start,
                "end": s.end,
                "meet_text": s.meet_text,
                "gromit_text": s.gromit_text,
                "reasons": s.reasons,
                "suggestion": s.suggestion,
            }
            for s in spans
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
