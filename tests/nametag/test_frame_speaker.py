"""Unit tests for the per-frame active-speaker tile pick."""
from __future__ import annotations

from gromit.nametag.frame_speaker import speaker_tile
from gromit.nametag.schema import FrameResult, Tile, TileKind

CAM = Tile(kind=TileKind.CAMERA, box=(0.0, 0.0, 1.0, 1.0))
AV = Tile(kind=TileKind.AVATAR, box=(0.7, 0.4, 0.25, 0.2))


def test_single_camera_tile_is_the_speaker():
    assert speaker_tile(FrameResult(tiles=[CAM])) is CAM


def test_single_avatar_tile_counts():
    assert speaker_tile(FrameResult(tiles=[AV])) is AV


def test_no_tile_abstains():
    assert speaker_tile(FrameResult(tiles=[])) is None


def test_gallery_abstains():
    assert speaker_tile(FrameResult(tiles=[CAM, AV])) is None
