"""Review orchestration: rank spans, cut clips, render the review page."""

from __future__ import annotations

from pathlib import Path

from gromit.review.clips import extract_clip
from gromit.review.diff import highlight
from gromit.review.flags import load_flags, rank_key
from gromit.review.names import load_named_cues, name_for
from gromit.review.render import ReviewRow, render_html


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"[{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}]"


def run_review(
    flags_path: Path,
    video: Path,
    named_path: Path | None,
    out_dir: Path,
    limit: int | None = None,
) -> dict:
    """Cut a clip per ranked span and write out_dir/index.html + clips/NNN.mp4."""
    spans = sorted(load_flags(flags_path), key=rank_key)
    if limit is not None:
        spans = spans[:limit]

    named = load_named_cues(named_path) if named_path else []

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ReviewRow] = []
    clips_ok = 0
    for i, sp in enumerate(spans):
        clip_path = clips_dir / f"{i:03d}.mp4"
        ok = extract_clip(video, sp.start, sp.end, clip_path)
        clips_ok += 1 if ok else 0
        meet_html, gromit_html = highlight(sp.meet_text, sp.gromit_text)
        rows.append(
            ReviewRow(
                index=i,
                clip_rel=f"clips/{i:03d}.mp4" if ok else None,
                timestamp=_hms(sp.start),
                speaker=name_for(sp.start, sp.end, named) if named else "",
                reasons=sp.reasons,
                meet_html=meet_html,
                gromit_html=gromit_html,
                suggestion=sp.suggestion,
            )
        )

    (out_dir / "index.html").write_text(
        render_html(rows, title=out_dir.parent.name or "review"), encoding="utf-8"
    )
    return {"spans": len(rows), "clips_ok": clips_ok, "out_dir": out_dir}
