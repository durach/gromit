import numpy as np

from gromit.nametag.heuristic import (
    _find_right_pip_tile,
    _tighten_camera_box,
    detect,
)
from gromit.nametag.schema import Layout, TileKind


def _full_frame(h=180, w=320):
    # photographic-ish: mid-gray with noise across the whole frame
    rng = np.random.default_rng(0)
    return rng.integers(40, 210, size=(h, w, 3), dtype=np.uint8)


def _screen_share_pip(h=180, w=320):
    rng = np.random.default_rng(2)
    frame = np.zeros((h, w, 3), dtype=np.uint8)  # black right column
    frame[:, :220] = 245  # bright "document" on the LEFT
    # sparse short dark text tokens (thin, low-solidity) scattered on the doc
    for _ in range(40):
        ty = rng.integers(20, 150)
        tx = rng.integers(10, 200)
        frame[ty:ty + 2, tx:tx + rng.integers(8, 30)] = 30
    # solid photographic PIP tile, top-right, letterboxed in the black column
    frame[12:72, 244:312] = rng.integers(40, 200, size=(60, 68, 3), dtype=np.uint8)
    return frame


def _bright_slide_with_left_graphic(h=360, w=640):
    """Bright document background with a big solid graphic on the LEFT and a
    small solid photographic camera tile on the RIGHT.  The LEFT graphic is
    large and colourful — the old heuristic would grab it.  The new one must
    select only the RIGHT tile.
    """
    rng = np.random.default_rng(7)
    frame = np.zeros((h, w, 3), dtype=np.uint8)  # black right column
    frame[:, :450] = 250  # bright document fills the LEFT

    # Big solid LEFT graphic (solid orange block — large area, high extent)
    frame[40:280, 20:200] = (30, 100, 230)  # BGR orange

    # Sparse text strokes on the document area (right side of slide)
    for _ in range(30):
        ty = rng.integers(30, 300)
        tx = rng.integers(220, 440)
        frame[ty:ty + 2, tx:tx + rng.integers(10, 40)] = 20

    # Camera tile: photographic block flush-right, letterboxed (black above/below)
    tile_x, tile_y, tile_w, tile_h = 488, 120, 140, 120
    frame[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w] = rng.integers(
        50, 180, size=(tile_h, tile_w, 3), dtype=np.uint8
    )
    return frame


def _dark_slide_with_right_pip(h=360, w=640):
    """Dark slide background with a small solid photographic camera tile on
    the RIGHT.  The whole left side is a dark-toned presentation (low
    brightness) — the old heuristic saw this as FULL_FRAME.  The new one
    must classify it as SCREEN_SHARE_PIP with the tile on the right.
    """
    rng = np.random.default_rng(11)
    # Dark slide background
    frame = np.full((h, w, 3), 25, dtype=np.uint8)

    # Slide content: slightly lighter text/graphics on the dark background
    for _ in range(20):
        ty = rng.integers(40, 300)
        tx = rng.integers(20, 420)
        frame[ty:ty + 3, tx:tx + rng.integers(20, 80)] = rng.integers(
            80, 160, size=(3, 1, 3), dtype=np.uint8
        ).reshape(3, 1, 3)

    # Camera tile: mid-bright photographic block at right, mid-height
    tile_x, tile_y, tile_w, tile_h = 470, 110, 155, 140
    frame[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w] = rng.integers(
        60, 200, size=(tile_h, tile_w, 3), dtype=np.uint8
    )
    # Embed a face-like region in the camera tile
    face_cx = tile_x + tile_w // 2
    face_cy = tile_y + tile_h // 3
    frame[face_cy - 18:face_cy + 28, face_cx - 22:face_cx + 22] = (115, 155, 185)
    frame[face_cy - 6:face_cy + 6, face_cx - 16:face_cx - 5] = 25
    frame[face_cy - 6:face_cy + 6, face_cx + 5:face_cx + 16] = 25

    return frame


def test_full_frame_detected():
    fr = detect(_full_frame())
    assert fr.layout is Layout.FULL_FRAME
    assert len(fr.tiles) == 1
    _x, _y, w, h = fr.tiles[0].box
    assert w > 0.8 and h > 0.8


def test_pip_detected_top_right():
    fr = detect(_screen_share_pip())
    assert fr.layout is Layout.SCREEN_SHARE_PIP
    assert len(fr.tiles) == 1
    x, y, w, h = fr.tiles[0].box
    assert x > 0.6 and y < 0.3       # top-right
    assert 0.1 < w < 0.45 and 0.1 < h < 0.5


def test_bright_slide_left_graphic_tile_is_on_right():
    """The large left-side solid graphic must NOT be selected; tile must be on
    the RIGHT side (x > 0.55)."""
    fr = detect(_bright_slide_with_left_graphic())
    assert fr.layout is Layout.SCREEN_SHARE_PIP
    assert len(fr.tiles) == 1
    x, _y, _w, _h = fr.tiles[0].box
    assert x > 0.55, f"tile x={x:.3f} should be on the right (>0.55)"


def test_dark_slide_pip_is_screen_share_and_tile_on_right():
    """A dark-background slide with a right PIP must be SCREEN_SHARE_PIP and
    the tile must be on the right (x > 0.55), not full-frame."""
    fr = detect(_dark_slide_with_right_pip())
    assert fr.layout is Layout.SCREEN_SHARE_PIP, (
        f"expected SCREEN_SHARE_PIP, got {fr.layout}"
    )
    assert len(fr.tiles) == 1
    x, _y, _w, _h = fr.tiles[0].box
    assert x > 0.55, f"tile x={x:.3f} should be on the right (>0.55)"


# ---------------------------------------------------------------------------
# Gap-1 test: screen-share with camera-OFF avatar tile on the right
# ---------------------------------------------------------------------------

def _screen_share_avatar_pip(h=360, w=640):
    """Bright document on the left, dark avatar tile on the right.

    The right tile has NO photographic texture — just a flat dark background
    with a solid coloured disk (avatar).  The heuristic must still detect and
    return the right-side tile (x > 0.55) even without a face.
    """
    rng = np.random.default_rng(17)
    frame = np.zeros((h, w, 3), dtype=np.uint8)  # black right column
    frame[:, :int(w * 0.62)] = 245  # bright document on the LEFT

    # Sparse dark text strokes on the document (left two-thirds).
    for _ in range(40):
        ty = rng.integers(20, h - 20)
        tx = rng.integers(10, int(w * 0.55))
        frame[ty:ty + 2, tx:tx + rng.integers(8, 35)] = 30

    # Avatar tile letterboxed in the black right column: a coloured disk (no
    # face structure) near the right edge plus a bright name label below it.
    cx_a, cy_a, radius = int(w * 0.90), h // 2, 36
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - cy_a) ** 2 + (xx - cx_a) ** 2 <= radius * radius
    frame[disk] = (40, 100, 170)  # brown-orange (BGR)
    # bright name label at the tile's bottom-left
    frame[cy_a + radius + 8:cy_a + radius + 24, int(w * 0.66):int(w * 0.86)] = 230

    return frame


def test_screen_share_avatar_pip_tile_on_right():
    """Gap-1: avatar tile (camera-off) on the right must be detected.

    The returned tile must have x > 0.55 (right side). The kind may be
    AVATAR. Layout must be SCREEN_SHARE_PIP.
    """
    fr = detect(_screen_share_avatar_pip())
    assert len(fr.tiles) == 1, f"expected 1 tile, got {len(fr.tiles)}"
    x, _y, _w, _h = fr.tiles[0].box
    assert x > 0.55, f"tile x={x:.3f} should be on the right (>0.55)"
    assert fr.layout is Layout.SCREEN_SHARE_PIP, (
        f"expected SCREEN_SHARE_PIP, got {fr.layout}"
    )


# ---------------------------------------------------------------------------
# Gap-2 test: portrait-pillarbox full-frame speaker
# ---------------------------------------------------------------------------

def _portrait_pillarbox(h=360, w=640):
    """Portrait speaker filling the canvas, pillarboxed with black left/right bands.

    The active video is a portrait column in the centre of the landscape frame.
    No screen-share content — left and right bands are pure black.
    The heuristic must return FULL_FRAME (NOT SCREEN_SHARE_PIP) and the tile
    box must cover the centre column (x roughly 0.2–0.45, h roughly >= 0.9).
    """
    rng = np.random.default_rng(23)
    frame = np.zeros((h, w, 3), dtype=np.uint8)  # black pillarbox background

    # Portrait video column: photographic content in the middle ~37% of width.
    col_x = int(w * 0.29)
    col_w = int(w * 0.42)
    frame[:, col_x:col_x + col_w] = rng.integers(
        40, 200, size=(h, col_w, 3), dtype=np.uint8
    )

    return frame


def test_portrait_pillarbox_is_full_frame():
    """Gap-2: portrait-pillarbox speaker must be FULL_FRAME, not SCREEN_SHARE_PIP.

    The tile box must cover the portrait column (x in [0.15, 0.45]) and the
    layout must be FULL_FRAME.
    """
    fr = detect(_portrait_pillarbox())
    assert fr.layout is Layout.FULL_FRAME, (
        f"expected FULL_FRAME for portrait pillarbox, got {fr.layout}"
    )
    assert len(fr.tiles) == 1, f"expected 1 tile, got {len(fr.tiles)}"
    x, _y, _w, h = fr.tiles[0].box
    # The tile should cover the centre column (not the full black frame).
    assert 0.15 <= x <= 0.45, f"tile x={x:.3f} should be in the centre column"
    assert h >= 0.9, f"tile height={h:.3f} should span almost the full frame height"


# ---------------------------------------------------------------------------
# Structural right-PIP detector: small camera / black-avatar tile on a bright
# document that fills most of the frame.  This is the real-world failure mode
# (Google Meet "present + camera") that the face-gated branch misclassified as
# FULL_FRAME.
# ---------------------------------------------------------------------------

def _present_small_camera_right(h=1080, w=1920):
    """Bright shared document filling the left ~78 %, a near-black right column,
    and a SMALL photographic camera tile flush to the right.
    """
    rng = np.random.default_rng(31)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright document on the left (value ~205) with sparse dark text.
    doc_right = 1500
    frame[:, :doc_right] = 205
    for _ in range(120):
        ty = rng.integers(20, h - 20)
        tx = rng.integers(10, doc_right - 40)
        frame[ty:ty + 3, tx:tx + rng.integers(20, 80)] = 25
    # Small photographic camera tile, flush-rightish, vertically centred.
    tx, ty, tw, th = 1600, 405, 160, 270
    frame[ty:ty + th, tx:tx + tw] = rng.integers(
        45, 200, size=(th, tw, 3), dtype=np.uint8
    )
    return frame


def _present_black_avatar_right(h=1080, w=1920):
    """Bright document on the left, a near-black right column with a camera-off
    avatar: a small coloured disk plus a bright name label at the tile's
    bottom-left.  No photographic texture, no face.
    """
    rng = np.random.default_rng(41)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    doc_right = 1450
    frame[:, :doc_right] = 210
    for _ in range(120):
        ty = rng.integers(20, h - 20)
        tx = rng.integers(10, doc_right - 40)
        frame[ty:ty + 3, tx:tx + rng.integers(20, 80)] = 25
    # Avatar disk (centre of the right column).
    cx_a, cy_a, radius = 1690, 520, 40
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - cy_a) ** 2 + (xx - cx_a) ** 2 <= radius * radius
    frame[disk] = (40, 100, 170)
    # Bright name label at the tile's bottom-left.
    frame[640:662, 1470:1600] = 230
    return frame


def test_find_right_pip_tile_camera():
    """The structural detector returns a CAMERA tile on the right."""
    canvas = _present_small_camera_right()
    out = _find_right_pip_tile(canvas)
    assert out is not None, "expected a right PIP tile, got None"
    (x, _y, tw, _th), kind = out
    assert kind is TileKind.CAMERA
    assert x > 0.55 * canvas.shape[1], f"tile x={x} should be on the right"
    assert tw < 0.45 * canvas.shape[1]


def test_find_right_pip_tile_black_avatar():
    """The structural detector returns an AVATAR tile (camera-off) on the right."""
    canvas = _present_black_avatar_right()
    out = _find_right_pip_tile(canvas)
    assert out is not None, "expected a right PIP tile, got None"
    (x, _y, _tw, _th), kind = out
    assert kind is TileKind.AVATAR
    assert x > 0.55 * canvas.shape[1], f"tile x={x} should be on the right"


def test_find_right_pip_tile_none_on_full_frame():
    """A full-frame talking head must NOT be matched as a right PIP."""
    assert _find_right_pip_tile(_full_frame(1080, 1920)) is None


def test_present_small_camera_is_pip():
    """End-to-end: bright doc + small right camera → SCREEN_SHARE_PIP, tile right."""
    fr = detect(_present_small_camera_right())
    assert fr.layout is Layout.SCREEN_SHARE_PIP, (
        f"expected SCREEN_SHARE_PIP, got {fr.layout}"
    )
    x, _y, _w, _h = fr.tiles[0].box
    assert x > 0.55, f"tile x={x:.3f} should be on the right"


def _present_short_band_portrait_camera(h=1080, w=1920):
    """Screen-share + a PORTRAIT camera tile flush right whose BRIGHT band (face +
    window) is short (~0.14*ch) with dark clothing below it.

    This band-height regime sits just under the old 0.15*ch CAMERA gate, so the
    feed was mis-routed to AVATAR (and boxed as the wide landscape container).
    It must be a CAMERA, boxed portrait.
    """
    rng = np.random.default_rng(83)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :1450] = 205  # bright document on the left
    for _ in range(120):
        ty = rng.integers(20, h - 20)
        tx = rng.integers(10, 1410)
        frame[ty:ty + 3, tx:tx + rng.integers(20, 80)] = 25
    tx, tw = 1600, 160
    # Bright upper feed band ~143 px (0.13*ch), then dark clothing (dim, not black).
    frame[405:548, tx:tx + tw] = rng.integers(60, 200, size=(143, tw, 3), dtype=np.uint8)
    frame[548:675, tx:tx + tw] = 18
    frame[650:670, tx + 5:tx + 120] = 230  # name label at the tile's bottom-left
    return frame


def test_short_band_portrait_camera_is_camera_portrait():
    """A portrait feed with a short bright band must be CAMERA (not AVATAR) and
    boxed portrait (taller than wide), not the wide landscape container."""
    fr = detect(_present_short_band_portrait_camera())
    assert fr.layout is Layout.SCREEN_SHARE_PIP
    tile = fr.tiles[0]
    assert tile.kind is TileKind.CAMERA, f"expected CAMERA, got {tile.kind}"
    x, _y, bw, bh = tile.box
    assert x > 0.70, f"tile x={x:.3f} should be the flush-right feed"
    assert bh > bw, f"box should be portrait (h>{bw:.3f}), got w={bw:.3f} h={bh:.3f}"


def test_present_black_avatar_is_pip():
    """End-to-end: bright doc + camera-off avatar → SCREEN_SHARE_PIP, tile right."""
    fr = detect(_present_black_avatar_right())
    assert fr.layout is Layout.SCREEN_SHARE_PIP, (
        f"expected SCREEN_SHARE_PIP, got {fr.layout}"
    )
    x, _y, _w, _h = fr.tiles[0].box
    assert x > 0.55, f"tile x={x:.3f} should be on the right"


def _meta_share_narrow_gutter(h=1080, w=1920):
    """A busy Meet window shared on the left, abutting a flush-right camera tile
    across a NARROW (~1 % of width) black gutter.

    The localiser's smoothing washes the gutter out, so the run spans content +
    tile (too wide); only the raw-profile retry can split them.  Mirrors the
    real 'someone shares their whole Meet window beside their own camera' frame.
    """
    rng = np.random.default_rng(71)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Busy shared content (bright, gallery-like) filling the band, up to the gutter.
    frame[300:760, 20:1424] = rng.integers(60, 220, size=(460, 1404, 3), dtype=np.uint8)
    frame[:, 1424:1443] = 0  # narrow black gutter (~19 px)
    # Flush-right camera tile (photographic), letterboxed top/bottom.
    frame[406:675, 1443:1920] = rng.integers(60, 210, size=(269, 477, 3), dtype=np.uint8)
    frame[648:668, 1455:1600] = 235  # name label at the tile's bottom-left
    return frame


def test_meta_share_narrow_gutter_isolates_right_tile():
    """The flush-right camera tile must be isolated from the shared content even
    when only a narrow gutter separates them (raw-profile retry), not swallowed
    into a too-wide box nor dropped to FULL_FRAME."""
    fr = detect(_meta_share_narrow_gutter())
    assert fr.layout is Layout.SCREEN_SHARE_PIP, (
        f"expected SCREEN_SHARE_PIP, got {fr.layout}"
    )
    x, _y, w, _h = fr.tiles[0].box
    assert x > 0.70, f"tile x={x:.3f} should be the flush-right tile"
    assert w < 0.30, f"tile w={w:.3f} should be just the tile, not content+tile"


# ---------------------------------------------------------------------------
# Trim camera box to the visible feed (cut grey container bars / black padding)
# ---------------------------------------------------------------------------

def test_tighten_trims_side_bars_and_black_padding():
    """Grey pillarbox bars on the sides AND pure-black padding below are trimmed
    to the visible feed."""
    g = np.zeros((1080, 1920), dtype=np.uint8)
    # grey 16:9 container at x[1450,1900] y[405,675]
    g[405:675, 1450:1900] = 17
    # bright portrait feed pillarboxed inside: x[1600,1760] y[405,674]
    g[405:674, 1600:1760] = 180
    # box loosely includes grey bars on the sides + black padding below the feed
    x, y, w, h = _tighten_camera_box(g, 1590, 405, 180, 290)
    assert 1595 <= x <= 1605, f"left {x} should hug the feed (~1600)"
    assert x + w <= 1765, f"right {x + w} should hug the feed (~1760)"
    assert y == 405, f"top should not move (feed starts at 405), got {y}"
    assert y + h <= 678, f"bottom {y + h} should drop the black padding (~675)"


def test_tighten_drops_left_border_keeps_flush_right_feed():
    """A bright document gridline + grey border sit LEFT of the feed (the feed is
    flush to the tile's right edge).  The box must hug the feed, dropping the
    isolated border on the left — a plain min/max-column trim would keep it."""
    g = np.zeros((1080, 1920), dtype=np.uint8)
    g[406:675, 1420] = 250          # 1px bright document gridline
    g[406:675, 1421:1443] = 17      # grey border (gap between doc and feed)
    g[406:675, 1443:1920] = 120     # landscape feed flush to the right edge
    x, _y, w, _h = _tighten_camera_box(g, 1420, 406, 500, 269)
    assert 1440 <= x <= 1446, f"left {x} should hug the feed (~1443), not the gridline"
    assert x + w >= 1916, f"right {x + w} should reach the frame edge (~1920)"


def test_tighten_keeps_dark_feed_rows():
    """Stability: the feed's dark lower region (clothing in shadow, ~12 — above
    pure black but below the feed level) is NOT trimmed, so the same camera keeps
    a stable height/aspect frame-to-frame."""
    g = np.zeros((1080, 1920), dtype=np.uint8)
    g[405:530, 1600:1760] = 180     # bright upper feed
    g[530:675, 1600:1760] = 12      # dark lower feed (clothing) — keep this
    _x, y, _w, h = _tighten_camera_box(g, 1600, 405, 160, 270)
    assert y == 405 and h >= 268, f"dark feed rows must be kept, got y={y} h={h}"


def test_tighten_noop_on_full_bright_tile():
    """A tile that is bright edge-to-edge (landscape feed filling the box) is
    essentially unchanged."""
    g = np.zeros((1080, 1920), dtype=np.uint8)
    g[300:600, 1500:1900] = 150
    x, y, w, h = _tighten_camera_box(g, 1500, 300, 400, 300)
    assert (x, y) == (1500, 300)
    assert abs(w - 400) <= 2 and abs(h - 300) <= 2


# ---------------------------------------------------------------------------
# Dark-slide screen-share with a camera-off avatar on the right.  The shared
# content is a DARK letterboxed slide (not a bright document), separated from a
# flush-right grey avatar container by a black gutter.  The old bright-sheet
# column walk grabbed the whole right half here because the dark slide never
# reached the plateau level; the dim-level letterbox gate isolates the
# container.
# ---------------------------------------------------------------------------

def _dark_slide_with_avatar_right(h=1080, w=1920):
    """Dark letterboxed slide centre-left + a camera-off avatar in a flush-right
    GREY container (disk + name), separated by a black gutter.
    """
    rng = np.random.default_rng(53)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Thin full-height left rail (Meet chrome) so the active canvas spans the
    # whole frame and is not trimmed to the slide bounds (matches real captures).
    frame[:, 0:14] = 120
    # Dark slide (gray ~24) centred-left, with a black gutter to its right.
    frame[262:820, 222:1215] = 24
    # Bright title block + accent bar (like a real slide heading).
    frame[330:392, 300:312] = (40, 130, 230)  # orange accent (BGR)
    frame[330:392, 330:900] = 210             # white title text band
    for _ in range(140):  # denser body text on the slide
        ty = rng.integers(430, 800)
        tx = rng.integers(300, 1150)
        frame[ty:ty + 4, tx:tx + rng.integers(40, 160)] = 170
    # Avatar container: dark-grey rounded rect flush right, y[405,675].
    frame[405:675, 1410:1920] = 17
    # Bright initial disk (short band → routes to the AVATAR branch).
    cx_a, cy_a, radius = 1665, 540, 58
    yy, xx = np.ogrid[:h, :w]
    frame[(yy - cy_a) ** 2 + (xx - cx_a) ** 2 <= radius * radius] = (180, 120, 40)
    # Bright name label at the container's bottom-left.
    frame[640:662, 1440:1570] = 230
    return frame


def test_dark_slide_avatar_is_landscape_container():
    """The avatar tile must be the flush-right LANDSCAPE container (wider than
    tall), not the whole right half of the frame."""
    canvas = _dark_slide_with_avatar_right()
    out = _find_right_pip_tile(canvas)
    assert out is not None, "expected a right avatar tile, got None"
    (x, _y, tw, th), kind = out
    assert kind is TileKind.AVATAR, f"expected AVATAR, got {kind}"
    assert x > 0.55 * canvas.shape[1], f"tile x={x} should be flush right"
    assert tw > th, f"avatar container should be landscape (w={tw} > h={th})"
    assert th < 0.5 * canvas.shape[0], f"tile height {th} should not span the frame"
    fr = detect(canvas)
    assert fr.layout is Layout.SCREEN_SHARE_PIP, (
        f"expected SCREEN_SHARE_PIP, got {fr.layout}"
    )


# ---------------------------------------------------------------------------
# Portrait pillarbox whose active canvas is a few rows short of full height (a
# thin top/bottom letterbox).  The pillarbox rule must still box the portrait
# COLUMN as FULL_FRAME rather than falling back to the whole landscape frame —
# the previous `ch == H` guard was too strict (Round 7).
# ---------------------------------------------------------------------------

def _portrait_pillarbox_letterboxed(h=1080, w=1920):
    rng = np.random.default_rng(67)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    col_x, col_w = int(w * 0.34), int(w * 0.32)
    top, bot = 16, h - 16  # thin top/bottom letterbox -> ch < H
    frame[top:bot, col_x:col_x + col_w] = rng.integers(
        40, 200, size=(bot - top, col_w, 3), dtype=np.uint8
    )
    return frame


def test_portrait_pillarbox_with_letterbox_is_full_frame():
    """A pillarbox portrait a few rows short of full height must still be
    FULL_FRAME with the narrow portrait-column box, not the whole frame."""
    fr = detect(_portrait_pillarbox_letterboxed())
    assert fr.layout is Layout.FULL_FRAME, f"expected FULL_FRAME, got {fr.layout}"
    x, _y, w, h = fr.tiles[0].box
    assert 0.20 <= x <= 0.45, f"tile x={x:.3f} should be the portrait column"
    assert w < 0.5, f"tile w={w:.3f} should be the narrow column, not the full frame"
    assert h >= 0.95, f"tile h={h:.3f} should span almost the full height"
