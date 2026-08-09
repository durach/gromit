"""Structured JSON output for transcripts (the .gromit.json file)."""

from __future__ import annotations

from gromit.alignment.temporal import AlignedSegment


def build_transcript_json(
    segments: list[AlignedSegment],
    *,
    language: str,
    model: str,
    hotwords_from: list[str],
) -> dict:
    """Build the .gromit.json payload from aligned segments + metadata."""
    return {
        "language": language,
        "model": model,
        "hotwords_from": hotwords_from,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "words": [
                    {"w": w.w, "start": w.start, "end": w.end, "p": w.p}
                    for w in seg.words
                ],
            }
            for seg in segments
        ],
    }
