"""Stage 2b: resolve an OCR'd name reading to a participant identity.

Open-set matching with a verbatim fallback: a reading close enough to a known
roster name (>= a similarity threshold) resolves to that canonical name —
recovering Google Meet's own truncation (``Yaroslav Vyshn…`` ->
``Yaroslav Vyshnevetsky``); anything else is kept verbatim and flagged, so an unlisted
(occasional) participant is never mislabeled as the nearest roster member.

Scoring is truncation-tolerant prefix similarity via stdlib ``difflib``: a
shorter (truncated) reading is compared against the same-length prefix of each
candidate. The roster file is parsed here, but ``match_name`` takes a plain list
of candidate names, so it stays decoupled from the YAML schema.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import yaml

DEFAULT_THRESHOLD = 0.80

_TRAILING_ELLIPSIS = re.compile(r"\s*(?:…|\.{2,})\s*$")  # trailing … or ...
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class NameMatch:
    name: str       # canonical roster name if matched, else the cleaned reading
    score: float    # best similarity in [0, 1]
    matched: bool   # True iff score >= threshold


@dataclass
class Roster:
    permanent: list[str] = field(default_factory=list)


def clean_name(s: str) -> str:
    """NFC-normalize, strip a trailing ellipsis, collapse whitespace, trim."""
    s = unicodedata.normalize("NFC", s)
    s = _TRAILING_ELLIPSIS.sub("", s)
    return _WS.sub(" ", s).strip()


def load_roster(path: str | Path) -> Roster:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Roster(permanent=list(data.get("permanent") or []))


_MIN_PREFIX_CHARS = 4  # a truncated reading shorter than this is too ambiguous to bind by prefix


def _prefix_similarity(reading: str, candidate: str) -> float:
    """Front-anchored similarity: compare only the shared LEADING run.

    Names are front-loaded and read left-to-right; the unreliable part is the tail
    — Meet truncation (a short reading), or stray text the crop caught *after* the
    name (a t-shirt logo / role label, e.g. ``Mykola H LOGO``). So we compare the
    shorter string against the same-length prefix of the longer one, making the
    first letters decisive and ignoring trailing junk on either side.
    """
    if not reading or not candidate:
        return 0.0
    n = min(len(reading), len(candidate))
    if n < _MIN_PREFIX_CHARS:  # too short a lead to bind safely
        return 0.0
    return SequenceMatcher(None, reading[:n], candidate[:n]).ratio()


def match_name(reading: str, candidates: list[str],
               threshold: float = DEFAULT_THRESHOLD) -> NameMatch:
    cleaned = clean_name(reading)
    norm_reading = cleaned.casefold()
    best_name, best_score = "", 0.0
    for cand in candidates:
        score = _prefix_similarity(norm_reading, clean_name(cand).casefold())
        # strict >: ties resolve to the first (permanent-first) candidate
        if score > best_score:
            best_name, best_score = cand, score
    if best_score >= threshold and best_name:
        return NameMatch(name=best_name, score=best_score, matched=True)
    return NameMatch(name=cleaned, score=best_score, matched=False)


def rank_candidates(reading: str, candidates: list[str],
                    top: int = 3) -> list[tuple[str, float]]:
    """Best-matching roster candidates for *reading*, highest score first.

    Unlike ``match_name`` (one verdict that hides the nearest name when nothing
    clears the threshold), this exposes the ranked candidates + scores for
    debugging/review. Same front-anchored similarity; empty list if no candidates.
    """
    nr = clean_name(reading).casefold()
    scored = [(c, _prefix_similarity(nr, clean_name(c).casefold())) for c in candidates]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:top]


def prefix_similarity(a: str, b: str) -> float:
    """Public front-anchored similarity between two strings (see ``_prefix_similarity``).

    Used by Stage 3 vote bucketing to merge off-roster OCR spelling variants of
    the same unlisted name. Inputs should already be cleaned/casefolded by the
    caller. Returns 0.0 when either lead is shorter than ``_MIN_PREFIX_CHARS``.
    """
    return _prefix_similarity(a, b)
