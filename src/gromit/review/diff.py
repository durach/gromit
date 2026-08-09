"""Token-level diff highlighting between the two engine readings."""

from __future__ import annotations

import html
from difflib import SequenceMatcher


def _render(tokens: list[str], changed: set[int]) -> str:
    out = []
    for i, tok in enumerate(tokens):
        esc = html.escape(tok)
        out.append(f"<mark>{esc}</mark>" if i in changed else esc)
    return " ".join(out)


def highlight(meet_text: str, gromit_text: str) -> tuple[str, str]:
    """Return (meet_html, gromit_html) with differing tokens wrapped in <mark>.

    Whitespace-tokenized; matching is case-insensitive but original casing is
    shown. Both sides are HTML-escaped.
    """
    m_tokens = meet_text.split()
    g_tokens = gromit_text.split()
    sm = SequenceMatcher(
        a=[t.lower() for t in m_tokens], b=[t.lower() for t in g_tokens]
    )
    m_changed: set[int] = set()
    g_changed: set[int] = set()
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            m_changed.update(range(i1, i2))
            g_changed.update(range(j1, j2))
    return _render(m_tokens, m_changed), _render(g_tokens, g_changed)
