"""Solution A: classical-CV tile/layout detector (no training).

Algorithm
---------
1.  Trim near-black letterbox/pillarbox borders to get the active canvas
    (`active_canvas_px`).
2.  **Pillarbox guard (`_is_pillarbox`):** detect a **portrait pillarbox**
    (symmetric near-black bands on BOTH sides, each ≥ 10 % of frame width).
    A portrait speaker padded this way is FULL_FRAME, so the PIP-detection
    branch is skipped for it.
3.  **Structural right-PIP detection (`_find_right_pip_tile`) — the SOLE PIP
    classifier:**  profile the rightmost-strip row profile to find a
    camera/avatar tile flush to the RIGHT that is letterboxed by near-black
    bands above AND below it, with shared content to the LEFT.  Works for
    bright documents and dark slides alike.  A full-frame speaker fills the
    right column top-to-bottom, so it is not matched.  When a tile is found →
    SCREEN_SHARE_PIP, with the localised tile box (CAMERA or AVATAR).  No Haar
    cascade is involved, so small camera tiles (which Haar misses) and slide
    false-positive "faces" (which Haar invents) no longer force a wrong verdict.
4.  **Face-based full-frame:**  not a structural screen-share.  Run the Haar
    cascade over the whole canvas; if any face is detected → FULL_FRAME, tile =
    the whole canvas box.
5.  **Whole-canvas fallback:**  no face, but the canvas has some content
    (mean > 8) → FULL_FRAME / avatar.  For a portrait pillarbox the tile box is
    the actual portrait column; otherwise it is the full-frame box (0, 0, 1, 1)
    so derive_layout always sees a full-frame area.
6.  **Dark / blank frame:**  canvas mean ≤ 8 → no tiles.

Key design choices
------------------
* No hard-coded tile coordinates — everything is derived per frame.
* The Haar cascade ships with OpenCV (no download).
* Works for both bright and dark slide backgrounds.
* The structural right-PIP detector is the SOLE PIP classifier; the old
  face-gated / blob PIP branches were removed because they mis-fired on
  full-frame speakers in busy rooms.
* Pillarbox guard prevents portrait speakers from triggering the PIP branch.
"""

from __future__ import annotations

import cv2
import numpy as np

from gromit.nametag.geometry import active_canvas_px, normalize_box
from gromit.nametag.schema import FrameResult, Tile, TileKind

# ---------------------------------------------------------------------------
# Haar face cascade (loaded once at module import time)
# ---------------------------------------------------------------------------
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Pixels at or below this grayscale value are treated as "black" (letterbox
# gutter / empty tile background) when profiling the screen-share structure in
# _find_right_pip_tile.
_PIP_BLACK_LEVEL: int = 28

# A camera-off avatar tile's *container* is a dark-grey rounded rect (~15-25
# gray) that _PIP_BLACK_LEVEL treats as black.  This lower threshold is used to
# recover the full container (disk + name) when boxing an avatar tile.
_PIP_DIM_LEVEL: int = 14

# A Meet camera tile is a 16:9 dark-grey container; a portrait phone feed is
# often pillarboxed inside it (grey bars ~17 on the sides, black above/below).
# We box the visible FEED, so a camera box is trimmed to pixels brighter than
# this — above the grey container/bars (~17), below lit feed content.
_PIP_FEED_LEVEL: int = 28

# Rows whose mean is below this are PURE-BLACK padding (above/below the feed) and
# are trimmed vertically.  It sits BELOW the feed's own dark lower region (a
# person's clothing in shadow reads ~10-15), so trimming is stable frame-to-frame
# — only genuine black padding (~0-3) is removed, never dark feed content.
_PIP_PAD_LEVEL: int = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) face detections on a grayscale image."""
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    if len(faces) == 0:
        return []
    return [tuple(int(v) for v in f) for f in faces]


def _is_pillarbox(frame_bgr: np.ndarray, cx: int, cw: int) -> bool:
    """Return True when the frame has significant black pillarbox bands on BOTH sides.

    A portrait speaker fills a narrow vertical strip centred in the landscape
    frame; Meet pads BOTH the left AND the right with black.  A screen-share
    frame only has a black border on one side at most (the slide fills the rest).

    Criteria:
    * The left black bar  (cx)                  is at least 10 % of frame width.
    * The right black bar (W - cx - cw)          is at least 10 % of frame width.
    * Both bars are near-black (mean < 20).

    Requiring BOTH sides prevents a thin recording-artifact border + wide
    content region (screen-share with dark slide) from being misidentified.
    """
    W = frame_bgr.shape[1]
    left_bar = cx
    right_bar = W - cx - cw
    min_bar = int(W * 0.10)
    if left_bar < min_bar or right_bar < min_bar:
        return False  # bars are not wide enough on one or both sides
    left_strip = frame_bgr[:, :left_bar]
    right_strip = frame_bgr[:, cx + cw:]
    left_mean = float(left_strip.mean())
    right_mean = float(right_strip.mean())
    return left_mean < 20.0 and right_mean < 20.0


def _smooth1d(a: np.ndarray, k: int) -> np.ndarray:
    """Box-smooth a 1-D signal with a window of at least 3 samples."""
    k = max(3, k)
    return np.convolve(a, np.ones(k) / k, mode="same")


def _pip_localize_bandrow(
    bw: np.ndarray, cw: int, sw: int, ty0: int, ty1: int
) -> tuple[int, int] | None:
    """Localise the tile's horizontal extent using the tile-band rows only.

    Within the tile's vertical band the camera/avatar tile is the flush-right
    run of non-black columns, separated from the shared content by a dark gutter
    or (for a dark slide) by a brightness contrast.  Robust whenever such a
    separation exists (sharp gutter, centred-slide gap, or dark content).

    The leftward walk tolerates near-black gaps shorter than ``gut_max`` columns
    (so a dark band inside the feed does not split the tile).  If the resulting
    run is too WIDE — the shared content abuts the tile across a gutter that the
    smoothing washed out (e.g. a Meet window shared next to its own camera tile,
    separated by a ~1 %-of-width gutter) — the walk is retried on the *raw*
    (unsmoothed) column profile, whose narrow gutters stay sharp.  The retry only
    fires on the too-wide failure, so it can only rescue a tile the default walk
    would otherwise reject; a genuinely gutter-less wide band still returns None.
    """
    cnb = _smooth1d(bw[ty0:ty1 + 1, :].mean(axis=0), int(cw * 0.01))
    cnb_raw = bw[ty0:ty1 + 1, :].mean(axis=0)
    floor = max(0.10, 0.40 * float(cnb[cw - sw:].max()))
    tr = cw - 1
    while tr > 0 and cnb[tr] < floor:
        tr -= 1
    if tr < cw * 0.5:
        return None

    def _walk_left(prof: np.ndarray, gut_max: int) -> int:
        tl = tr
        gut = 0
        while tl > 0:
            if prof[tl - 1] < floor * 0.5:
                gut += 1
                if gut >= gut_max:
                    break
            else:
                gut = 0
            tl -= 1
        while tl < tr and prof[tl] < floor * 0.5:
            tl += 1
        return tl

    for prof, coeff in ((cnb, 0.012), (cnb_raw, 0.006)):
        tl = _walk_left(prof, max(6, int(cw * coeff)))
        if cw * 0.03 <= tr - tl + 1 <= cw * 0.45:
            return (tl, tr)
    return None


def _pip_localize_bright(bw: np.ndarray, cw: int) -> tuple[int, int] | None:
    """Localise the tile when bright content is directly adjacent (no gutter).

    Uses the full-height column profile: the shared document is a high plateau,
    the tile a lower one flush right; the boundary is where the profile rises
    back to content level.  Complements _pip_localize_bandrow for the gutterless
    bright-document case.
    """
    full = _smooth1d(bw.mean(axis=0), int(cw * 0.01))
    if not (full >= 0.40).any():
        return None
    content_level = max(float(np.median(full[full >= 0.40])), 0.40)
    content_stop = content_level * 0.55
    floor = 0.04
    tr = cw - 1
    while tr > 0 and full[tr] < floor:
        tr -= 1
    if tr < cw * 0.5:
        return None
    tl = tr
    gut = 0
    gut_max = max(6, int(cw * 0.012))
    while tl > 0:
        v = full[tl - 1]
        if v >= content_stop:
            break
        if v < floor:
            gut += 1
            if gut >= gut_max:
                break
        else:
            gut = 0
        tl -= 1
    while tl < tr and full[tl] < floor:
        tl += 1
    return (tl, tr) if cw * 0.03 <= tr - tl + 1 <= cw * 0.45 else None


def _find_right_pip_tile(
    canvas: np.ndarray,
) -> tuple[tuple[int, int, int, int], TileKind] | None:
    """Detect a right-side PIP tile *structurally*, independent of face detection.

    A Google-Meet "present + camera" frame places the camera/avatar tile in a
    column flush to the RIGHT, **letterboxed by near-black bands above and
    below** it (the right column is taller than the tile).  A full-frame speaker,
    by contrast, fills the right column top-to-bottom.  So the decisive,
    content-agnostic signal is the *row profile of the rightmost strip*: a band
    of non-black rows bounded by black above AND below ⇒ PIP tile; a full-height
    fill ⇒ not PIP.  This works for bright documents AND dark slides, and rejects
    full-frame speakers in busy rooms (whose right side is filled top-to-bottom).

    Why not Haar faces: face detection both *misses* small camera faces and
    *hallucinates* faces on slides, which made the old face-gated branch
    misclassify these screenshares.  This is the sole PIP classifier in detect().

    Returns
    -------
    ((x, y, w, h), TileKind)
        Tile rectangle in canvas-pixel coords plus CAMERA (photographic) or
        AVATAR (camera-off).
    None
        Not a content-left / tile-right screen-share (full-frame speaker, blank
        frame, lone centred avatar, etc.).
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    if cw < 20 or ch < 20:
        return None
    bw = gray >= _PIP_BLACK_LEVEL

    # (A) Letterbox gate: in the flush-right strip, find the band of non-black
    #     rows (the tile) and require black bands above AND below it.
    sw = max(8, round(cw * 0.13))
    rprof = _smooth1d(bw[:, cw - sw:].mean(axis=1), int(ch * 0.012))
    peak = float(rprof.max())
    if peak < 0.18:
        return None  # right edge essentially black — no flush-right tile
    row_thr = max(0.12, 0.5 * peak)
    ys = np.where(rprof >= row_thr)[0]
    if ys.size == 0:
        return None
    ty0, ty1 = int(ys[0]), int(ys[-1])
    th = ty1 - ty0 + 1
    if th > ch * 0.90:
        return None  # fills the right column top-to-bottom → full-frame speaker
    if (ty0 + (ch - 1 - ty1)) < ch * 0.08:
        return None  # not letterboxed (no meaningful black bands) → full-frame

    # (B) Horizontal localisation: try the band-row localiser first (handles dark
    #     slides, gutters and centred-slide gaps), then the bright-adjacent one.
    loc = _pip_localize_bandrow(bw, cw, sw, ty0, ty1) or _pip_localize_bright(bw, cw)
    if loc is None:
        return None
    tile_left, tile_right = loc
    tile_w = tile_right - tile_left + 1

    # (C) Content-on-left sanity: a real screen-share has content left of the
    #     tile; reject a lone tile / centred avatar on an otherwise-black frame.
    if tile_left > cw * 0.04 and float(bw[:, :tile_left].mean()) < 0.015:
        return None

    fill = float(bw[ty0:ty1 + 1, tile_left:tile_right + 1].mean())
    # CAMERA vs AVATAR by the bright-band height: a camera-off avatar's bright
    # region is just the centred initial disk (band ≈ 0.10*ch, very consistent),
    # while a photographic feed's bright band is taller (≈ 0.15-0.25*ch) — even a
    # portrait feed whose lower half is dark clothing still clears ≈ 0.147*ch.
    # The 0.125*ch gate sits in the middle of that gap (disk ≈ 109 px vs camera
    # ≥ 159 px at 1080p), so a short-but-real portrait feed is not mis-routed.
    if fill >= 0.55 and th >= ch * 0.125 and tile_w >= cw * 0.05:
        # Photographic camera tile.  The rightmost-strip row band captures only
        # the bright upper tile (window / face); dark clothing and the name
        # label below it fall under the black threshold and get clipped.  Extend
        # the box DOWN to the lowest row (within a bounded window) whose tile
        # columns still carry content at a dim threshold, so the name at the
        # tile's bottom-left is inside the crop.
        tcols = gray[:, tile_left:tile_right + 1] >= _PIP_DIM_LEVEL
        fillrows = tcols.sum(axis=1) > max(2, int(0.05 * tile_w))
        win_hi = min(ch, ty1 + 1 + int(0.20 * ch))
        active = np.where(fillrows[ty0:win_hi])[0]
        cy1 = ty0 + int(active.max()) if active.size else ty1
        cy1 = min(cy1, ty0 + int(ch * 0.92))
        return (int(tile_left), int(ty0), int(tile_w), int(cy1 - ty0 + 1)), TileKind.CAMERA

    # Camera-off avatar: the bright band found above is only the initial disk.
    # The tile CONTAINER is a dark-grey rounded rect (≈15-25 gray, below
    # _PIP_BLACK_LEVEL) holding the disk and the name at its bottom-left.
    # Re-detect it with the SAME letterbox structure as the camera path but at the
    # DIM level, where the grey container reads as content: a flush-right band of
    # dim rows (the container) letterboxed by black above AND below, localised
    # horizontally by the black gutter separating it from the shared content.
    # Works whether the shared content is a bright document OR a dark slide — the
    # old bright-sheet column walk grabbed the whole right half when a dark slide
    # never reached the plateau level.
    dim = gray >= _PIP_DIM_LEVEL
    drow = _smooth1d(dim[:, cw - sw:].mean(axis=1), int(ch * 0.012))
    if float(drow.max()) >= 0.18:
        dys = np.where(drow >= max(0.12, 0.5 * float(drow.max())))[0]
        if dys.size:
            ay0, ay1 = int(dys[0]), int(dys[-1])
            aloc = (
                _pip_localize_bandrow(dim, cw, sw, ay0, ay1)
                if (ay1 - ay0 + 1) <= ch * 0.90
                else None
            )
            if aloc is not None:
                al, aright = aloc
                aw = aright - al + 1
                content_left = al <= cw * 0.04 or float(dim[:, :al].mean()) >= 0.015
                if content_left and cw * 0.03 <= aw <= cw * 0.45:
                    return (int(al), int(ay0), int(aw),
                            int(ay1 - ay0 + 1)), TileKind.AVATAR

    # Fallback: bright-sheet column walk (a bright document directly abutting the
    # container with no separating gutter).  At the dim level the document is a
    # tall column plateau (dcol ≈ 1.0) while the container is a lower one flush
    # right; walk left through the container and stop at the plateau.
    dcol = _smooth1d(dim.mean(axis=0), int(cw * 0.01))
    sheet_lvl = 0.60
    dtr = cw - 1
    while dtr > 0 and dcol[dtr] < 0.04:
        dtr -= 1  # trim trailing black margin
    dtl = dtr
    while dtl > 0 and dcol[dtl - 1] < sheet_lvl:
        dtl -= 1  # through container, stop at the bright content plateau
    while dtl < dtr and dcol[dtl] < 0.04:
        dtl += 1  # trim the leading near-black gutter
    aw = dtr - dtl + 1
    arows = np.where(dim[:, dtl:dtr + 1].sum(axis=1) > max(3, int(0.05 * aw)))[0]
    if arows.size and cw * 0.03 <= aw <= cw * 0.45:
        by0, by1 = int(arows.min()), int(arows.max())
        ah = min(by1 - by0 + 1, int(ch * 0.92))
        return (int(dtl), by0, int(aw), int(ah)), TileKind.AVATAR
    # Last resort: tight box around the disk band.
    return (int(tile_left), int(ty0), int(tile_w), int(th)), TileKind.AVATAR


def _tighten_camera_box(
    gray: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """Trim a camera-tile box to the visible feed.

    Around the feed the detected box may include: flat grey pillarbox bars (a
    portrait phone feed, ~17), a bright document edge / grey border on the LEFT
    (the feed is flush to the tile's right edge), or pure-black padding ABOVE and
    BELOW (the feed is shorter than the detected band).

    Horizontal — keep the flush-RIGHT run of FEED columns (a small fraction above
    ``_PIP_FEED_LEVEL``): walk left from the right edge through feed columns and
    stop at a sustained non-feed gap.  This drops a grey/black border or a bright
    document gridline sitting left of the feed (which a plain min/max-column trim
    would glue back on), and the left/right grey pillarbox bars.

    Vertical — trim only PURE-BLACK rows (row mean < ``_PIP_PAD_LEVEL``) off the
    top and bottom.  Black padding goes, but the feed's dark lower region
    (clothing in shadow, ~10-15) stays, so the height is stable frame-to-frame
    (brightness-based vertical trimming made the same camera wobble 9:16 ↔ 3:5).

    Trim only; returns the box unchanged if no feed columns are found.
    """
    if w <= 0 or h <= 0:
        return (x, y, w, h)
    feedcol = (gray[y:y + h, x:x + w] >= _PIP_FEED_LEVEL).mean(axis=0) > 0.04
    if not feedcol.any():
        return (x, y, w, h)
    # Right edge = rightmost feed column (drops a trailing grey bar).
    right = w - 1
    while right > 0 and not feedcol[right]:
        right -= 1
    # Walk left through the feed, stopping at a sustained non-feed gap.
    gap = 0
    gap_max = max(4, int(0.03 * w))
    left = right
    while left > 0:
        if not feedcol[left - 1]:
            gap += 1
            if gap >= gap_max:
                break
        else:
            gap = 0
        left -= 1
    while left < right and not feedcol[left]:  # skip into the feed past the gap
        left += 1
    nx, nw = x + left, right - left + 1
    # Trim pure-black padding rows from the top and bottom (keep dark feed).
    rowmean = gray[y:y + h, nx:nx + nw].mean(axis=1)
    top = 0
    while top < h - 1 and rowmean[top] < _PIP_PAD_LEVEL:
        top += 1
    bot = h - 1
    while bot > top and rowmean[bot] < _PIP_PAD_LEVEL:
        bot -= 1
    return (nx, y + top, nw, bot - top + 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(frame_bgr: np.ndarray) -> FrameResult:
    """Detect camera tile(s) in *frame_bgr* and return a FrameResult.

    Implements the algorithm described in the module docstring.
    """
    H, W = frame_bgr.shape[:2]
    cx, cy, cw, ch = active_canvas_px(frame_bgr)
    canvas = frame_bgr[cy : cy + ch, cx : cx + cw]

    # --- 0.  Pillarbox guard — portrait speaker, not screen-share ---
    # If the frame has narrow black bars on both sides (pillarbox), the canvas IS
    # the portrait column.  Any face detected in its right portion would
    # spuriously trigger the PIP branch.  Skip straight to FULL_FRAME.
    is_pillarbox = _is_pillarbox(frame_bgr, cx, cw)

    # --- 0a. Structural right-PIP detection (takes precedence over faces) ---
    # A content-left / tile-right screen-share has an unmistakable column
    # signature.  Detect it directly — BEFORE any Haar logic — so that small
    # camera tiles (which Haar misses) and hallucinated faces on slides (which
    # Haar invents) can no longer drive the frame to a wrong FULL_FRAME verdict.
    if not is_pillarbox:
        pip = _find_right_pip_tile(canvas)
        if pip is not None:
            (tx, ty, tw, th), kind = pip
            if kind is TileKind.CAMERA:
                # Trim the box to the visible feed — the Meet tile is a 16:9
                # grey container with the (often portrait) feed pillarboxed
                # inside; cut the grey bars / black padding off all four edges.
                gray_canvas_pip = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                tx, ty, tw, th = _tighten_camera_box(gray_canvas_pip, tx, ty, tw, th)
            box = normalize_box((cx + tx, cy + ty, tw, th), W, H)
            return FrameResult(tiles=[Tile(kind=kind, box=box, confidence=0.8)])

    # --- 0b. Face-based FULL_FRAME ---
    # Not a structural screen-share (step 0a returned None).  If a face is
    # present anywhere the frame is a full-frame speaker; return the whole canvas
    # as the tile.  (The structural detector in 0a is the SOLE PIP classifier —
    # the old face-right / blob PIP branches were removed because they
    # mis-fired on full-frame speakers in busy rooms.)
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    faces_all = _detect_faces(gray_canvas)
    if faces_all:
        # Full-frame box.  Do NOT use the trimmed active-canvas box: a lone
        # camera-off avatar (a face-bearing profile photo on an otherwise-black
        # frame) trims to a small sub-region whose area < 0.6 would be
        # mis-derived as SCREEN_SHARE_PIP.  The Meet name label is at the
        # bottom-left of the WHOLE frame for a single full-frame participant.
        if is_pillarbox:
            # The active canvas IS the portrait feed column (the wide black
            # pillars are trimmed away); box it directly so the crop is the
            # portrait video, not the whole landscape frame.  A small top/bottom
            # letterbox trim (ch slightly < H) is fine — derive_layout keys
            # full-frame on h >= 0.95, so a pillarbox column still reads
            # FULL_FRAME rather than being mis-classified as a PIP tile.
            box = normalize_box((cx, cy, cw, ch), W, H)
        else:
            box = (0.0, 0.0, 1.0, 1.0)
        return FrameResult(
            tiles=[Tile(kind=TileKind.CAMERA, box=box, confidence=0.9)]
        )

    # --- 4.  Whole canvas has some content → FULL_FRAME ---
    # Threshold: mean > 8 (previously 20).  A full-frame avatar on a black
    # background has a coloured disk that contributes ~10–20 average brightness
    # across the canvas.  Lowering the threshold catches that case while still
    # rejecting truly blank/transition frames (mean ≈ 0–5).
    if gray_canvas.mean() > 8:
        # Choose the tile bounding box carefully:
        # * Portrait pillarbox (is_pillarbox=True, cy==0, ch==H): use the
        #   canvas box so the name crop targets the portrait column precisely.
        #   derive_layout sees h ≈ 1.0 and classifies as FULL_FRAME.
        # * All other cases (letterboxed avatar, normal full-frame): return
        #   the full-frame box (0, 0, 1, 1).  The canvas bounding box in these
        #   cases is a sub-frame region whose area < 0.6 would be mis-classified
        #   as SCREEN_SHARE_PIP; the Meet name label is at the bottom-left of
        #   the WHOLE FRAME for full-frame speakers.
        if is_pillarbox:
            # The active canvas IS the portrait feed column (the wide black
            # pillars are trimmed away); box it directly so the crop is the
            # portrait video, not the whole landscape frame.  A small top/bottom
            # letterbox trim (ch slightly < H) is fine — derive_layout keys
            # full-frame on h >= 0.95, so a pillarbox column still reads
            # FULL_FRAME rather than being mis-classified as a PIP tile.
            box = normalize_box((cx, cy, cw, ch), W, H)
        else:
            box = (0.0, 0.0, 1.0, 1.0)
        return FrameResult(
            tiles=[Tile(kind=TileKind.AVATAR, box=box, confidence=0.3)]
        )

    # --- 5.  Dark / blank frame: nothing found ---
    return FrameResult(tiles=[])
