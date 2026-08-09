"""Stage 3: pick the active-speaker tile in one frame.

Meet recordings of the kind this stage targets show one participant at a time:
FULL_FRAME (a single person) or SCREEN_SHARE_PIP (one camera box on the right).
Either way there is
exactly ONE tile, which is the speaker (camera or camera-off avatar). Zero tiles
(a pure shared slide) or a gallery (>=2 tiles, rare/edge) are ambiguous, so the
frame abstains — a wrong name is worse than no name, and cue-level voting
recovers the cue from the clean frames.
"""

from __future__ import annotations

from gromit.nametag.schema import FrameResult, Tile


def speaker_tile(frame: FrameResult) -> Tile | None:
    return frame.tiles[0] if len(frame.tiles) == 1 else None
