"""Stage 3 outputs: a named WebVTT and a grouped annotation transcript.

* ``write_named_vtt`` re-emits the cues unchanged except that the resolved name
  is prepended to each cue's first text line as ``Name: ...`` (a plain prefix —
  the WebVTT ``<v>`` voice tag renders invisibly in VLC).
* ``write_annotation`` mirrors ``gromit.output.formatter`` (``[HH:MM:SS] Name:``
  then text), grouping consecutive same-name cues. Cue text is used verbatim
  (newlines collapsed to spaces for the prose form).
"""

from __future__ import annotations

from pathlib import Path

from gromit.nametag.vtt import Cue, format_timestamp


def write_named_vtt(cues: list[Cue], names: list[str], path, header: str = "WEBVTT") -> None:
    lines = [header, ""]
    for cue, name in zip(cues, names):
        lines.append(f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}")
        text_lines = cue.text.split("\n") if cue.text else [""]
        first = text_lines[0]
        lines.append(f"{name}: {first}".rstrip() if first else f"{name}:")
        lines.extend(text_lines[1:])
        lines.append("")
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"[{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}]"


def write_annotation(cues: list[Cue], names: list[str], path) -> None:
    groups: list[list] = []  # [name, start, [texts]]
    for cue, name in zip(cues, names):
        flat = " ".join(cue.text.split())
        if groups and groups[-1][0] == name:
            groups[-1][2].append(flat)
        else:
            groups.append([name, cue.start, [flat]])
    parts = [
        f"{_hms(start)} {name}:\n{' '.join(t for t in texts if t)}"
        for name, start, texts in groups
    ]
    Path(path).write_text(("\n\n".join(parts) + "\n") if parts else "", encoding="utf-8")
