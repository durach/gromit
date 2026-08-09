"""Apple Vision OCR adapter for name-strip crops (macOS only, via ocrmac).

Graceful by design: where ``ocrmac`` cannot be imported (i.e. not macOS, or the
optional [vision] extra is not installed), ``vision_available()`` is False and
``recognize_vision`` returns an empty reading, so callers fall back to EasyOCR.
Takes an in-memory BGR crop rather than a file path.
"""

from __future__ import annotations

import cv2
import numpy as np

from gromit.nametag.name_ocr import NameReading, keep_leftmost_cluster

_available: bool | None = None


def vision_available() -> bool:
    """True iff Apple Vision (ocrmac) can be imported (cached after first call)."""
    global _available
    if _available is None:
        try:
            from ocrmac import ocrmac  # noqa: F401
            _available = True
        except Exception:  # noqa: BLE001
            _available = False
    return _available


def recognize_vision(crop_bgr: np.ndarray) -> NameReading:
    """OCR a name-strip crop with Apple Vision; empty reading if unavailable/empty.

    Keeps only the left-anchored name cluster (drops a t-shirt logo / role label
    elsewhere in the tile), matching the EasyOCR path.
    """
    if crop_bgr is None or crop_bgr.size == 0 or not vision_available():
        return NameReading("", 0.0)
    from ocrmac import ocrmac
    from PIL import Image
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        anns = ocrmac.OCR(Image.fromarray(rgb), language_preference=["en-US"]).recognize()
    except Exception:  # noqa: BLE001 - one unreadable crop must not abort the run
        return NameReading("", 0.0)
    # Vision bbox = (x, y, w, h) normalized to the crop; build (left, right, text, conf).
    frags = [(x, x + w, t.strip(), float(c)) for (t, c, (x, y, w, h)) in anns if t.strip()]
    frags = keep_leftmost_cluster(frags)
    if not frags:
        return NameReading("", 0.0)
    text = " ".join(f[2] for f in frags)
    conf = sum(f[3] for f in frags) / len(frags)
    return NameReading(text=text, confidence=conf)
