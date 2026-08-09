"""Shared output contract for Stage-1 tile/layout detection.

Boxes are (x, y, w, h) normalized to the frame size (0.0-1.0) so the schema is
resolution-independent. Only participant tiles (camera/avatar) are represented;
the shared-content region is not a tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

Box = tuple[float, float, float, float]


class Layout(Enum):
    FULL_FRAME = "full_frame"
    SCREEN_SHARE_PIP = "screen_share_pip"
    GALLERY = "gallery"
    UNKNOWN = "unknown"


class TileKind(Enum):
    CAMERA = "camera"
    AVATAR = "avatar"


@dataclass(frozen=True)
class Tile:
    kind: TileKind
    box: Box
    confidence: float = 1.0


# A tile covering this fraction of the frame area is treated as "full-bleed".
FULL_FRAME_AREA = 0.6


def derive_layout(tiles: list[Tile]) -> Layout:
    """Infer the frame layout from the set of detected participant tiles.

    A single tile is FULL_FRAME when:
    * its area (w*h, normalised) meets the FULL_FRAME_AREA threshold, OR
    * it spans nearly the full frame height (h >= 0.95) — the portrait-pillarbox
      case where the active video column is tall but narrow.  A PIP tile is
      never full-height, so the height check is unambiguous.
    """
    if not tiles:
        return Layout.UNKNOWN
    if len(tiles) == 1:
        _, _, w, h = tiles[0].box
        if w * h >= FULL_FRAME_AREA or h >= 0.95:
            return Layout.FULL_FRAME
        return Layout.SCREEN_SHARE_PIP
    return Layout.GALLERY


@dataclass
class FrameResult:
    tiles: list[Tile] = field(default_factory=list)
    layout: Layout = field(init=False)

    def __post_init__(self) -> None:
        self.layout = derive_layout(self.tiles)
