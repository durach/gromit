"""Flag signals (divergence / low_confidence / misheard_match) and span merging."""

from __future__ import annotations

from dataclasses import dataclass

from gromit.crosscheck.gromit_json import GSegment
from gromit.crosscheck.normalize import normalize_text, token_containment


@dataclass
class Span:
    """A flagged time span needing human review."""

    start: float
    end: float
    meet_text: str
    gromit_text: str
    reasons: list[str]
    suggestion: str | None


@dataclass(frozen=True)
class Thresholds:
    """Tunable crosscheck thresholds (design §3 starting points)."""

    divergence_max: float = 0.5
    word_p_min: float = 0.4
    low_word_min: int = 2  # need a cluster of low-prob words, not a single one
    seg_logprob_min: float = -0.8
    merge_gap: float = 2.0


def find_misheard(text: str, misheard_index: dict[str, str]) -> str | None:
    """Canonical form for the first known misheard substring in `text`, else None."""
    norm = normalize_text(text)
    for misheard, canonical in misheard_index.items():
        if normalize_text(misheard) in norm:
            return canonical
    return None


def segment_flags(
    seg: GSegment,
    meet_text: str,
    misheard_index: dict[str, str],
    has_meet: bool,
    thresholds: Thresholds,
) -> tuple[list[str], str | None]:
    """Return (reasons, suggestion) for one gromit segment.

    reasons is ordered: divergence, low_confidence, misheard_match.
    """
    reasons: list[str] = []
    suggestion: str | None = None

    # Divergence: how much of the gromit segment's wording is missing from the
    # aligned Meet text. Containment (not Jaccard) so Meet's wider caption
    # window doesn't read as disagreement (see token_containment).
    if has_meet and token_containment(seg.text, meet_text) < thresholds.divergence_max:
        reasons.append("divergence")

    # A single uncertain word is normal; require a cluster (or a low
    # whole-segment avg_logprob) before calling a segment low-confidence.
    n_low = sum(1 for w in seg.words if w.p < thresholds.word_p_min)
    if seg.avg_logprob < thresholds.seg_logprob_min or n_low >= thresholds.low_word_min:
        reasons.append("low_confidence")

    hit = find_misheard(seg.text, misheard_index)
    if hit is None and has_meet:
        hit = find_misheard(meet_text, misheard_index)
    if hit is not None:
        reasons.append("misheard_match")
        suggestion = hit

    return reasons, suggestion


def merge_spans(spans: list[Span], merge_gap: float) -> list[Span]:
    """Merge spans whose gap ≤ merge_gap into one, unioning reasons and text.

    Sorts by start. Reasons keep first-seen order; the first non-null
    suggestion in the group wins.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start)
    merged: list[Span] = [
        Span(s.start, s.end, s.meet_text, s.gromit_text, list(s.reasons), s.suggestion)
        for s in ordered[:1]
    ]
    for s in ordered[1:]:
        cur = merged[-1]
        if s.start - cur.end <= merge_gap:
            cur.end = max(cur.end, s.end)
            for r in s.reasons:
                if r not in cur.reasons:
                    cur.reasons.append(r)
            cur.meet_text = " ".join(t for t in (cur.meet_text, s.meet_text) if t)
            cur.gromit_text = " ".join(t for t in (cur.gromit_text, s.gromit_text) if t)
            if cur.suggestion is None:
                cur.suggestion = s.suggestion
        else:
            merged.append(
                Span(s.start, s.end, s.meet_text, s.gromit_text, list(s.reasons), s.suggestion)
            )
    return merged
