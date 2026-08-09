import numpy as np

from gromit.nametag.geometry import active_canvas_px, iou, normalize_box, to_px


def test_iou_identical_is_one():
    assert iou((0.0, 0.0, 0.5, 0.5), (0.0, 0.0, 0.5, 0.5)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)) == 0.0


def test_iou_half_overlap():
    # two unit-area-ish boxes overlapping in half their union
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.1, 0.0, 0.2, 0.2)
    # intersection = 0.1*0.2=0.02; union = 0.08+0.08-0.02=0.06 -> 1/3
    assert abs(iou(a, b) - (0.02 / 0.06)) < 1e-9


def test_normalize_and_to_px_roundtrip():
    box_px = (192, 108, 480, 270)
    norm = normalize_box(box_px, width=1920, height=1080)
    assert norm == (0.1, 0.1, 0.25, 0.25)
    assert to_px(norm, 1920, 1080) == box_px


def test_active_canvas_trims_black_border():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[10:90, 20:180] = 200  # bright content inset by a black border
    x, y, w, h = active_canvas_px(frame)
    assert 18 <= x <= 22 and 8 <= y <= 12
    assert 158 <= w <= 162 and 78 <= h <= 82
