"""Crosscheck orchestration: align two engines, run signals, emit flagged spans."""

from __future__ import annotations

from pathlib import Path

from gromit.crosscheck.align import meet_text_for, overlap_fraction
from gromit.crosscheck.gromit_json import load_gromit_json
from gromit.crosscheck.output import write_flags_json
from gromit.crosscheck.signals import Span, Thresholds, merge_spans, segment_flags
from gromit.exceptions import CrosscheckError
from gromit.glossary import load_glossaries
from gromit.nametag.vtt import parse_vtt

MIN_OVERLAP_FRACTION = 0.20

# Shared default. Thresholds is a frozen dataclass, so one module-level instance is
# safe to reuse as an argument default (and keeps B008 honest without a suppression).
DEFAULT_THRESHOLDS = Thresholds()

__all__ = ["DEFAULT_THRESHOLDS", "Span", "Thresholds", "run_crosscheck", "write_flags_json"]


def run_crosscheck(
    gromit_path: Path,
    meet_path: Path | None,
    glossary_paths: list[Path],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[Span]:
    """Align the gromit transcript with the Meet VTT and return flagged spans.

    Without `meet_path`, only low_confidence + misheard_match run. Raises
    CrosscheckError if the Meet timeline barely overlaps the gromit one.
    """
    transcript = load_gromit_json(gromit_path)
    misheard_index = (
        load_glossaries(glossary_paths).misheard_index() if glossary_paths else {}
    )

    has_meet = meet_path is not None
    cues = parse_vtt(meet_path) if has_meet else []
    if has_meet:
        frac = overlap_fraction(cues, transcript.segments)
        if frac < MIN_OVERLAP_FRACTION:
            raise CrosscheckError(
                f"Only {frac:.0%} of Meet cues overlap gromit segments — "
                f"wrong file pairing? ({meet_path})"
            )

    flagged: list[Span] = []
    for seg in transcript.segments:
        meet_text = meet_text_for(seg.start, seg.end, cues) if has_meet else ""
        reasons, suggestion = segment_flags(
            seg, meet_text, misheard_index, has_meet, thresholds
        )
        if reasons:
            flagged.append(
                Span(seg.start, seg.end, meet_text, seg.text, reasons, suggestion)
            )

    return merge_spans(flagged, thresholds.merge_gap)
