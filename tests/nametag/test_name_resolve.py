"""Unit tests for best-of-both name resolution (OCR engines monkeypatched)."""
from __future__ import annotations

import numpy as np

import gromit.nametag.name_resolve as nr
from gromit.nametag.name_ocr import NameReading

CROP = np.ones((10, 40, 3), np.uint8)


def test_higher_scoring_engine_wins(monkeypatch):
    # EasyOCR garbles the name; Vision reads it cleanly -> Vision's match wins.
    monkeypatch.setattr(nr, "recognize_name", lambda c, reader=None: NameReading("Solomiya hytska", 0.9))
    monkeypatch.setattr(nr, "recognize_vision", lambda c: NameReading("Solomiya Verbytska", 0.95))
    m = nr.resolve_name(CROP, ["Solomiya Verbytska"], use_vision=True)
    assert m is not None and m.name == "Solomiya Verbytska" and m.matched


def test_easyocr_only_when_vision_disabled(monkeypatch):
    seen = {"vision": 0}

    def vision(c):
        seen["vision"] += 1
        return NameReading("Whoever", 1.0)

    monkeypatch.setattr(nr, "recognize_vision", vision)
    monkeypatch.setattr(nr, "recognize_name", lambda c, reader=None: NameReading("Yaroslav Vyshnevetsky", 1.0))
    m = nr.resolve_name(CROP, ["Yaroslav Vyshnevetsky"], use_vision=False)
    assert m.name == "Yaroslav Vyshnevetsky" and seen["vision"] == 0  # Vision never called


def test_no_text_returns_none(monkeypatch):
    monkeypatch.setattr(nr, "recognize_name", lambda c, reader=None: NameReading("", 0.0))
    monkeypatch.setattr(nr, "recognize_vision", lambda c: NameReading("", 0.0))
    assert nr.resolve_name(CROP, ["Yaroslav Vyshnevetsky"], use_vision=True) is None
