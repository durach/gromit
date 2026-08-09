"""Unit tests for the Stage 2a name-strip sub-region geometry."""
from __future__ import annotations

from gromit.nametag.geometry import to_px
from gromit.nametag.name_region import name_band

W, H = 1920, 1080


def _px(tile):
    """name_band as integer pixel (x, y, w, h) on a 1080p frame."""
    return to_px(name_band(tile, W, H), W, H)


def test_full_frame_band_is_bottom_left_strip():
    bx, by, bw, bh = _px((0.0, 0.0, 1.0, 1.0))
    assert bx == 0                    # flush to the tile/frame left edge
    assert bh == 64                   # height hits the max clamp
    assert bw == 560                  # width hits the cap
    assert abs((by + bh) - H) <= 1    # bottom-aligned to the tile bottom


def test_pip_band_inside_tile():
    tile = (0.78, 0.06, 0.20, 0.30)
    bx, by, bw, bh = _px(tile)
    tx, ty, tw, th = to_px(tile, W, H)
    assert bx == tx                          # left edge of the tile
    assert abs((by + bh) - (ty + th)) <= 1   # bottom-aligned to the tile bottom
    assert bx + bw <= tx + tw                # never spills past the tile right edge
    assert bh == 64                          # 0.22*324=71 -> clamped to 64
    assert by >= 0 and by + bh <= H          # inside the frame


def test_band_never_exceeds_frame_at_bottom_right_corner():
    tile = (0.90, 0.85, 0.10, 0.15)   # tile in the bottom-right corner
    bx, by, bw, bh = _px(tile)
    tx, _, tw, _ = to_px(tile, W, H)
    assert bx + bw <= W
    assert by + bh <= H
    assert bx + bw <= tx + tw         # within the tile width (192px)
    assert bh == 36                   # 0.22*162 = 35.6 -> 36 (proportional, unclamped)


def test_tiny_width_tile_clamps_band_to_tile_width():
    tile = (0.10, 0.50, 0.05, 0.20)   # tile only 96px wide
    _, _, bw, _ = _px(tile)
    _, _, tw, _ = to_px(tile, W, H)
    assert bw == tw                   # band width capped to the tile width (96)


def test_short_tile_caps_band_height_to_tile_height():
    tile = (0.10, 0.50, 0.30, 0.015)  # tile only ~16px tall
    _, _, _, bh = _px(tile)
    _, _, _, th = to_px(tile, W, H)
    assert bh == th                   # height capped to tile height (< min clamp)


def test_narrow_portrait_tile_uses_full_tile_width():
    # A narrow portrait PIP tile (~154px wide): the band must span the FULL
    # tile width — not get clipped by a fixed floor below the tile width
    # (Google Meet left-aligns the name across nearly the whole narrow tile).
    tile = (0.50, 0.40, 0.08, 0.30)   # 154px wide, 324px tall
    bx, _by, bw, _bh = _px(tile)
    tx, _ty, tw, _th = to_px(tile, W, H)
    assert bw == tw                   # full tile width captured (no right-edge clip)
    assert bx == tx                   # anchored at the tile's left edge
