"""Tests for the review orchestrator (clip extraction stubbed)."""

import json
from unittest.mock import patch

from gromit.review.core import run_review


def _flags(tmp_path, spans):
    p = tmp_path / "flags.json"
    p.write_text(json.dumps({"spans": spans}, ensure_ascii=False))
    return p


def test_run_review_writes_html_and_orders_by_priority(tmp_path):
    fp = _flags(tmp_path, [
        {"start": 10.0, "end": 11.0, "meet_text": "a", "gromit_text": "b",
         "reasons": ["low_confidence"], "suggestion": None},
        {"start": 90.0, "end": 91.0, "meet_text": "реліз чекліст", "gromit_text": "release checklist",
         "reasons": ["misheard_match"], "suggestion": "release checklist"},
    ])
    out = tmp_path / "review"
    # Stub ffmpeg: pretend every clip extracts fine.
    with patch("gromit.review.core.extract_clip", return_value=True):
        summary = run_review(fp, tmp_path / "v.mp4", None, out)
    assert summary["spans"] == 2
    assert summary["clips_ok"] == 2
    html = (out / "index.html").read_text()
    # misheard span (priority 0) must render before the low_confidence one
    assert html.index("release checklist") < html.index('data-i="1"')


def test_run_review_marks_failed_clip_but_keeps_row(tmp_path):
    fp = _flags(tmp_path, [
        {"start": 1.0, "end": 2.0, "meet_text": "x", "gromit_text": "y",
         "reasons": ["divergence"], "suggestion": None},
    ])
    out = tmp_path / "review"
    with patch("gromit.review.core.extract_clip", return_value=False):
        summary = run_review(fp, tmp_path / "v.mp4", None, out)
    assert summary["spans"] == 1
    assert summary["clips_ok"] == 0
    assert "clip unavailable" in (out / "index.html").read_text().lower()


def test_run_review_limit(tmp_path):
    spans = [
        {"start": float(i), "end": i + 0.5, "meet_text": "m", "gromit_text": "g",
         "reasons": ["divergence"], "suggestion": None}
        for i in range(5)
    ]
    fp = _flags(tmp_path, spans)
    out = tmp_path / "review"
    with patch("gromit.review.core.extract_clip", return_value=True):
        summary = run_review(fp, tmp_path / "v.mp4", None, out, limit=2)
    assert summary["spans"] == 2
