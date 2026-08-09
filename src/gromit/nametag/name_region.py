"""Stage 2a: the bottom-left name-strip sub-region of a Stage-1 tile box.

Google Meet always renders the participant name as white text at the
bottom-left of a tile. Given a (validated) tile box, this returns the
sub-rectangle that *contains* that name label, sized to contain rather than
hug it (tight glyph localisation is a later escalation, not needed here).

Geometry (the "absolute-px height / proportional width" hybrid):
  * height: an absolute pixel band (Meet's label is ~constant font height and
    the targeted source resolution is 1920x1080), clamped to [MIN, MAX] px and
    capped at the tile's own height;
  * width: the full tile width (the Meet label is left-aligned and never wider
    than its tile), capped at MAX px so a large full-frame tile doesn't yield a
    huge band; surrounding video is harmless and gets removed by white-text
    masking in Stage 2b.
Constants were tuned by eye against sampled frames across several recordings.
"""

from __future__ import annotations

from gromit.nametag.geometry import Box, normalize_box

_NAME_H_FRAC = 0.22       # band height as a fraction of tile height...
_NAME_H_MIN_PX = 24       # ...clamped to this floor...
_NAME_H_MAX_PX = 64       # ...and this cap (~label height + margin at 1080p).
_NAME_W_MAX_PX = 560      # band width = full tile width, capped here for big tiles.


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def name_band(tile_box: Box, frame_w: int, frame_h: int) -> Box:
    """Bottom-left name-strip sub-region of *tile_box*, as a normalized Box."""
    tx_px = round(tile_box[0] * frame_w)
    ty_px = round(tile_box[1] * frame_h)
    tw_px = round(tile_box[2] * frame_w)
    th_px = round(tile_box[3] * frame_h)

    band_h = _clamp(round(_NAME_H_FRAC * th_px), _NAME_H_MIN_PX, _NAME_H_MAX_PX)
    band_h = min(band_h, th_px)
    band_w = min(tw_px, _NAME_W_MAX_PX)

    band_x = tx_px
    band_y = ty_px + th_px - band_h

    # Defensive clamp so the band always lies inside the frame.
    band_x = int(_clamp(band_x, 0, max(0, frame_w - band_w)))
    band_y = int(_clamp(band_y, 0, max(0, frame_h - band_h)))

    return normalize_box((band_x, band_y, int(band_w), int(band_h)), frame_w, frame_h)
