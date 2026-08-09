"""Run the Stage 3 attribution pipeline over one meeting: video + VTT -> named files.

Drives the ``gromit nametag`` CLI. Quiet by default
(the per-cue torch ``pin_memory`` warning is filtered here; the EasyOCR banner is
off via ``name_ocr``; ffmpeg chatter is suppressed in ``sampling``). Pass
``on_cue`` to observe per-cue progress.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path

import cv2

from gromit.nametag.attribution import UNKNOWN, attribute_cue
from gromit.nametag.cache import cache_dir_for, discard
from gromit.nametag.frame_speaker import speaker_tile
from gromit.nametag.geometry import to_px
from gromit.nametag.heuristic import detect
from gromit.nametag.name_region import name_band
from gromit.nametag.name_resolve import resolve_name
from gromit.nametag.sampling import cue_frame_times, extract_frames_at
from gromit.nametag.vision_ocr import vision_available
from gromit.nametag.vtt import parse_header, parse_vtt
from gromit.nametag.vtt_output import write_annotation, write_named_vtt


def silence_warnings() -> None:
    """Filter the per-cue torch dataloader warning that floods batch runs."""
    warnings.filterwarnings("ignore", message=r".*pin_memory.*")
    warnings.filterwarnings("ignore", message=r".*degrees of freedom.*")


@contextlib.contextmanager
def _suppress_fd2():
    """Redirect C-level stderr (fd 2) to /dev/null, then restore.

    The objc/dyld "Class ... implemented in both" warning from cv2 and av both
    bundling libavdevice is printed by the dynamic loader at the C level, so a
    Python ``warnings`` filter can't catch it — only an fd-level redirect can.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def preload_quiet() -> None:
    """Force cv2 + av to load now with the dyld duplicate-class chatter silenced.

    ``av`` (PyAV, pulled in transitively under torch) and ``cv2`` each bundle a
    ``libavdevice``; whichever loads second triggers a one-time objc warning.
    Loading both here under the fd-2 redirect makes that warning fire silently,
    so later imports (e.g. torch during OCR) find them already loaded.
    """
    with _suppress_fd2(), contextlib.suppress(Exception):
        import av  # noqa: F401
        import cv2  # noqa: F401


def make_frame_reader(reader=None, image_loader=None, use_vision=None):
    """Build a ``(path, candidates) -> NameMatch | None`` reader (best-of-both on macOS)."""
    load = image_loader or (lambda p: cv2.imread(str(p)))
    if use_vision is None:
        use_vision = vision_available()

    def read(path, candidates):
        frame = load(path)
        if frame is None:
            return None
        tile = speaker_tile(detect(frame))
        if tile is None:
            return None
        h, w = frame.shape[:2]
        nx, ny, nbw, nbh = to_px(name_band(tile.box, w, h), w, h)
        crop = frame[ny:ny + nbh, nx:nx + nbw]
        if crop.size == 0:
            return None
        return resolve_name(crop, candidates, easy_reader=reader, use_vision=use_vision)

    return read


def attribute_meeting(video, vtt, out_dir, candidates, *, use_vision=None,
                      early_stop=False, cache_dir=None, keep_cache=False,
                      on_cue=None) -> dict:
    """Attribute every VTT cue and write ``<stem>.named.vtt`` / ``.named.txt`` to *out_dir*.

    *on_cue(index, total, name)* is invoked after each cue (for a progress display).
    Returns a summary dict. Caller supplies *candidates* (roster + guests).

    The frame cache defaults to a per-meeting dir under the system temp
    (``cache.cache_dir_for``). After a successful run the cache is deleted unless
    *keep_cache* is set or some cue still needs review (resolved to ``UNKNOWN`` or
    to a name outside *candidates*). On error the cache is left in place.
    """
    silence_warnings()
    preload_quiet()
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    cues = parse_vtt(vtt)
    cache = Path(cache_dir) if cache_dir else cache_dir_for(video)
    if use_vision is None:
        use_vision = vision_available()
    reader = make_frame_reader(use_vision=use_vision)

    names: list[str] = []
    for cue in cues:
        frames = extract_frames_at(video, cue_frame_times(cue.start, cue.end), cache)
        res = attribute_cue(frames, candidates, reader, early_stop=early_stop)
        names.append(res.name)
        if on_cue is not None:
            on_cue(cue.index, len(cues), res.name)

    named_vtt = out_dir / f"{stem}.named.vtt"
    write_named_vtt(cues, names, named_vtt, header=parse_header(vtt))
    write_annotation(cues, names, out_dir / f"{stem}.named.txt")

    candidate_set = set(candidates)
    needs_review = sum(1 for n in names if n == UNKNOWN or n not in candidate_set)
    kept = keep_cache or needs_review > 0
    if not kept:
        discard(cache)
    return {"stem": stem, "cues": len(cues), "out_dir": str(out_dir),
            "use_vision": use_vision, "named_vtt": str(named_vtt),
            "cache_dir": str(cache), "kept": kept, "needs_review": needs_review}
