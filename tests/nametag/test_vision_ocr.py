"""Unit tests for the Apple Vision adapter (graceful)."""
from __future__ import annotations

import numpy as np

from gromit.nametag.name_ocr import NameReading
from gromit.nametag.vision_ocr import recognize_vision, vision_available


def test_vision_available_returns_bool():
    assert isinstance(vision_available(), bool)


def test_recognize_vision_empty_on_empty_or_none_crop():
    assert recognize_vision(np.zeros((0, 0, 3), np.uint8)) == NameReading("", 0.0)
    assert recognize_vision(None) == NameReading("", 0.0)
