"""Text normalization + token-set similarity for engine comparison."""

from __future__ import annotations

import re

# Hesitation / discourse fillers dropped before comparison. Conservative set —
# only tokens that carry no lexical content (design §3 names «е-е», «ну»).
FILLERS: frozenset[str] = frozenset(
    {"е", "ее", "еее", "ееее", "эм", "ем", "мм", "ммм", "хм", "ну"}
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)  # keep letters/digits/underscore + space
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation, collapse newlines/whitespace."""
    s = s.replace("\n", " ").lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    """Normalized tokens with fillers removed."""
    return [t for t in normalize_text(s).split() if t not in FILLERS]


def token_containment(a: str, b: str) -> float:
    """Fraction of a's tokens present in b: |A∩B| / |A|.

    Asymmetric on purpose: gromit segments are narrower than the Google Meet
    rolling-caption window they align to, so a symmetric metric (Jaccard) would
    flag agreement as divergence merely because Meet's window carries extra
    tokens. Containment asks "did the gromit segment's words show up in Meet?"
    Returns 1.0 when a has no tokens (nothing can be missing).
    """
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta:
        return 1.0
    return len(ta & tb) / len(ta)
