"""Unit tests for Stage 2b OCR preprocessing + the recognize_name wrapper.

End-to-end OCR quality is validated visually (model-dependent); here we test the
deterministic preprocessing and the no-model guard paths. A heavy real-OCR smoke
test is marked ``slow``.
"""
from __future__ import annotations

import numpy as np
import pytest

from gromit.nametag.name_ocr import (
    NameReading,
    isolate_white_text,
    keep_leftmost_cluster,
    recognize_name,
)


def test_keep_leftmost_cluster_drops_far_logo_fragment():
    # a name at the left of the strip, an unrelated logo far to the right
    frags = [(0.012, 0.251, "Solomiya Verbytska", 1.0), (0.582, 0.649, "ACME", 0.5)]
    assert [f[2] for f in keep_leftmost_cluster(frags)] == ["Solomiya Verbytska"]


def test_keep_leftmost_cluster_keeps_normal_word_gap():
    # a name split into two words with an ordinary space stays intact (any input order)
    frags = [(0.14, 0.25, "Verbytska", 1.0), (0.01, 0.12, "Solomiya", 1.0)]
    assert [f[2] for f in keep_leftmost_cluster(frags)] == ["Solomiya", "Verbytska"]


def test_isolate_white_text_keeps_bright_drops_dark():
    crop = np.full((40, 200, 3), (60, 70, 80), dtype=np.uint8)  # dark background
    crop[10:30, 20:120] = (255, 255, 255)                       # bright "glyph"
    out = isolate_white_text(crop)
    assert out.shape == crop.shape
    assert set(np.unique(out)).issubset({0, 255})  # binary
    assert out[20, 70, 0] == 255                    # bright region -> white
    assert out[2, 2, 0] == 0                        # dark region -> black


def test_recognize_name_empty_crop_returns_blank_without_model():
    # size==0 must short-circuit before any (expensive) EasyOCR import/build.
    assert recognize_name(np.zeros((0, 0, 3), dtype=np.uint8)) == NameReading("", 0.0)


@pytest.mark.slow
def test_recognize_name_returns_namereading_on_real_image():
    import cv2

    img = np.zeros((60, 400, 3), dtype=np.uint8)
    cv2.putText(img, "Yaroslav", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 2, cv2.LINE_AA)
    r = recognize_name(img)
    assert isinstance(r, NameReading)
    assert isinstance(r.text, str)
