"""Stage 2b: read the participant name from a name-strip crop with EasyOCR.

The Meet label is always near-white text, so the glyphs are first lifted off the
noisy video background by thresholding on luminance (``isolate_white_text``)
before OCR; if that yields nothing we retry on the raw colour crop. EasyOCR is
markedly better than Tesseract on this kind of low-resolution in-video text,
which is a book-OCR engine's weakest case. The reader is
configured for Latin/English only: Meet display names are commonly Latin-script
even for non-English speakers, and enabling the Cyrillic model made the recogniser
emit look-alike Cyrillic glyphs for Latin letters (e.g. ``Kravets`` -> ``Кгаvеtѕ``),
which broke matching. A Cyrillic-script roster needs the reader reconfigured.
The reader is expensive to build, so a module-level singleton is reused across
crops.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Latin/English only. Meet display names are usually Latin-script; enabling
# uk/ru made the recogniser substitute look-alike Cyrillic glyphs into Latin
# names (homoglyph substitution), so we keep a single Latin reader.
_LANGS = ["en"]
_WHITE_LUMA = 170  # pixels brighter than this (0-255) are candidate glyphs

_reader = None


def get_reader():
    """Lazily build + cache the EasyOCR reader (heavy: imports torch + models)."""
    global _reader
    if _reader is None:
        import easyocr  # local import keeps module import cheap

        _reader = easyocr.Reader(_LANGS, gpu=False, verbose=False)  # silence the "Using CPU" banner
    return _reader


@dataclass(frozen=True)
class NameReading:
    text: str
    confidence: float


def isolate_white_text(crop_bgr: np.ndarray) -> np.ndarray:
    """Binarize to white glyphs on black, keying on high luminance.

    Returns a 3-channel uint8 image (white text on black) ready for EasyOCR.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, _WHITE_LUMA, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


_NAME_GAP_FRAC = 0.12  # drop fragments separated from the name by a wider gap


def keep_leftmost_cluster(frags: list[tuple[float, float, str, float]]):
    """Keep the bottom-left name label, dropping unrelated text elsewhere in the tile.

    The Meet name is always left-anchored; other on-screen text (a t-shirt logo, a
    role label) sits after a large horizontal gap. *frags* is a list of
    ``(left, right, text, conf)`` with left/right normalized to the crop width;
    returns the leftmost run, stopping at the first gap wider than ``_NAME_GAP_FRAC``.
    """
    frags = sorted(frags, key=lambda f: f[0])
    kept: list[tuple[float, float, str, float]] = []
    right = None
    for left, r, text, conf in frags:
        if right is not None and left - right > _NAME_GAP_FRAC:
            break
        kept.append((left, r, text, conf))
        right = r if right is None else max(right, r)
    return kept


def _read(reader, img: np.ndarray) -> NameReading:
    # detail=1 -> list of (bbox, text, confidence); bbox = 4 [x, y] points (px).
    w = img.shape[1] or 1
    frags = []
    for bbox, text, conf in reader.readtext(img, detail=1, paragraph=False):
        if not text.strip():
            continue
        xs = [p[0] for p in bbox]
        frags.append((min(xs) / w, max(xs) / w, text.strip(), float(conf)))
    frags = keep_leftmost_cluster(frags)
    if not frags:
        return NameReading("", 0.0)
    text = " ".join(f[2] for f in frags)
    conf = float(np.mean([f[3] for f in frags]))
    return NameReading(text=text, confidence=conf)


def recognize_name(crop_bgr: np.ndarray, reader=None) -> NameReading:
    """OCR a name-strip crop; returns the joined text + mean confidence."""
    if crop_bgr is None or crop_bgr.size == 0:
        return NameReading("", 0.0)
    reader = reader or get_reader()
    reading = _read(reader, isolate_white_text(crop_bgr))
    if not reading.text:  # binarization wiped it -> retry on the raw colour crop
        reading = _read(reader, crop_bgr)
    return reading
