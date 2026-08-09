"""Resolve a name-strip crop to a roster identity using the best available OCR.

On macOS, runs BOTH EasyOCR and Apple Vision and keeps the higher-scoring roster
match (Vision wins ties — it is the stronger engine on these crops); elsewhere,
EasyOCR alone. This is the Stage 2b "best of both" promoted for Stage 3 use.
"""

from __future__ import annotations

import numpy as np

from gromit.nametag.name_ocr import recognize_name
from gromit.nametag.roster import NameMatch, match_name
from gromit.nametag.vision_ocr import recognize_vision, vision_available


def resolve_name(crop_bgr: np.ndarray, candidates: list[str], *,
                 easy_reader=None, use_vision: bool | None = None) -> NameMatch | None:
    """Best roster match for *crop_bgr* across the available OCR engines.

    Returns ``None`` when no engine produced any text. With Vision available, both
    engines run and the higher-scoring match wins (Vision is tried first, so it
    wins score ties). *use_vision* defaults to ``vision_available()``.
    """
    if use_vision is None:
        use_vision = vision_available()
    readings = []
    if use_vision:
        readings.append(recognize_vision(crop_bgr))   # first -> wins score ties
    readings.append(recognize_name(crop_bgr, reader=easy_reader))
    best: NameMatch | None = None
    for r in readings:
        if not r.text.strip():
            continue
        m = match_name(r.text, candidates)
        if best is None or m.score > best.score:       # strict > -> Vision wins ties
            best = m
    return best
