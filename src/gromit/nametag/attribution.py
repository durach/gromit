"""Stage 3: collapse per-frame name matches for a cue into one decision.

Each sampled frame yields a ``roster.NameMatch`` (or ``None`` when the frame had
no usable tile / no reading). ``vote_cue`` buckets those matches and returns the
winner: roster-matched names aggregate by their canonical string; off-roster
verbatim readings are merged when their leading letters agree (so OCR spelling
wobble does not split an unlisted guest's votes). The winner is the bucket with
the most frames, ties broken by summed match score. No usable frame -> Unknown.

Single-match rule: if exactly one roster member matched (>= threshold) anywhere
in the cue, that member wins even if sub-threshold garbles out-count them — a
name over a busy graphic reads as many near-miss verbatims plus a few clean hits,
and those garbles are that same person. A genuine off-roster guest produces no
match, so they still surface verbatim; >= 2 matched names fall back to the count.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gromit.nametag.roster import NameMatch, clean_name, prefix_similarity

UNKNOWN = "Unknown"


@dataclass
class CueResult:
    name: str               # resolved name, or UNKNOWN
    votes: dict[str, float]  # bucket name -> summed match score
    frames_used: int        # frames that produced a usable reading


def vote_cue(matches: list[NameMatch | None], bucket_threshold: float = 0.80) -> CueResult:
    buckets: list[dict] = []  # {name, count, score, matched}
    used = 0
    for mt in matches:
        if mt is None or not mt.name.strip():
            continue
        used += 1
        label = mt.name if mt.matched else clean_name(mt.name)
        target = None
        # Matched (roster-canonical) and unmatched (verbatim) readings are kept in
        # separate buckets even for the same person; in practice the matched bucket
        # wins by frame count. This deliberately avoids merging an off-roster guest
        # into a prefix-similar roster name.
        for b in buckets:
            if b["matched"] and mt.matched:
                if b["name"] == label:
                    target = b
                    break
            # The nesting below is deliberate: the outer if/elif picks the bucket
            # *kind* (matched vs verbatim), the inner test asks whether the name is
            # equivalent *within* that kind. Collapsing only this arm would break
            # the symmetry with the matched arm above (which ruff cannot collapse
            # because it owns the elif) and bury the kind check inside an already
            # multi-line boolean.
            elif not b["matched"] and not mt.matched:  # noqa: SIM102 — see comment above
                if (b["name"].casefold() == label.casefold()
                        or prefix_similarity(b["name"].casefold(), label.casefold()) >= bucket_threshold):
                    target = b
                    break
        if target is None:
            buckets.append({"name": label, "count": 1, "score": mt.score, "matched": mt.matched})
        else:
            target["count"] += 1
            target["score"] += mt.score
    if not buckets:
        return CueResult(name=UNKNOWN, votes={}, frames_used=used)
    buckets.sort(key=lambda b: (b["count"], b["score"]), reverse=True)
    votes: dict[str, float] = {}
    for b in buckets:
        votes[b["name"]] = round(votes.get(b["name"], 0.0) + b["score"], 4)
    # Single-match rule: exactly one roster member matched anywhere -> they own the
    # cue (sub-threshold garbles are their corrupted reads). 0 or >=2 matches fall
    # back to the count/score winner. Never lowers the match threshold.
    matched_names = {b["name"] for b in buckets if b["matched"]}
    winner = next(iter(matched_names)) if len(matched_names) == 1 else buckets[0]["name"]
    return CueResult(name=winner, votes=votes, frames_used=used)


def _decided(matches: list, total: int) -> bool:
    """True once a roster-matched name holds a strict majority of *total* frames."""
    r = vote_cue(matches)
    if r.name == UNKNOWN:
        return False
    # matched names pass through vote_cue unchanged, so r.name == m.name for roster hits
    agree = sum(1 for m in matches if m is not None and m.matched and m.name == r.name)
    return agree * 2 > total


def attribute_cue(
    frames: list[tuple[float, Any]],
    candidates: list[str],
    frame_reader: Callable[[Any, list[str]], NameMatch | None],
    early_stop: bool = False,
) -> CueResult:
    """Read each sampled frame and vote.

    *frames* is ``list[(time, path)]``; *frame_reader* is
    ``(path, candidates) -> NameMatch | None``. With *early_stop*, stops reading
    once a roster name already holds a strict majority of the planned frames.
    """
    matches: list = []
    total = len(frames)
    for _t, path in frames:
        matches.append(frame_reader(path, candidates))
        if early_stop and _decided(matches, total):
            break
    return vote_cue(matches)
