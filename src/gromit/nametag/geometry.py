"""Box math and active-canvas (letterbox trim) helpers."""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def normalize_box(box_px, width: int, height: int) -> Box:
    x, y, w, h = box_px
    return (x / width, y / height, w / width, h / height)


def to_px(box: Box, width: int, height: int):
    x, y, w, h = box
    return (round(x * width), round(y * height), round(w * width), round(h * height))


def active_canvas_px(frame: np.ndarray, dark: int = 16):
    """Bounding box (x, y, w, h in px) of non-near-black content after trimming
    uniform letterbox/pillarbox borders."""
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    mask = gray > dark
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if cols.size == 0 or rows.size == 0:
        h, w = gray.shape[:2]
        return (0, 0, w, h)
    x1, x2 = int(cols[0]), int(cols[-1])
    y1, y2 = int(rows[0]), int(rows[-1])
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
