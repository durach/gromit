from gromit.nametag.schema import FrameResult, Layout, Tile, TileKind, derive_layout


def _tile(x, y, w, h, kind=TileKind.CAMERA, conf=1.0):
    return Tile(kind=kind, box=(x, y, w, h), confidence=conf)


def test_tile_box_is_normalized_tuple():
    t = _tile(0.1, 0.2, 0.3, 0.4)
    assert t.box == (0.1, 0.2, 0.3, 0.4)
    assert t.kind is TileKind.CAMERA


def test_derive_layout_full_frame_when_one_big_tile():
    assert derive_layout([_tile(0.0, 0.0, 1.0, 1.0)]) is Layout.FULL_FRAME


def test_derive_layout_pip_when_one_small_tile():
    assert derive_layout([_tile(0.72, 0.05, 0.25, 0.25)]) is Layout.SCREEN_SHARE_PIP


def test_derive_layout_gallery_when_several_comparable_tiles():
    tiles = [_tile(0.0, 0.0, 0.5, 0.5), _tile(0.5, 0.0, 0.5, 0.5),
             _tile(0.0, 0.5, 0.5, 0.5)]
    assert derive_layout(tiles) is Layout.GALLERY


def test_derive_layout_unknown_when_empty():
    assert derive_layout([]) is Layout.UNKNOWN


def test_frameresult_layout_autoderives():
    fr = FrameResult(tiles=[_tile(0.0, 0.0, 1.0, 1.0)])
    assert fr.layout is Layout.FULL_FRAME
