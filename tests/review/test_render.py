"""Tests for the self-contained review HTML render."""

from gromit.review.render import ReviewRow, render_html


def _row(**kw):
    base = {"index": 0, "clip_rel": "clips/000.mp4", "timestamp": "[00:12:34]",
            "speaker": "Yaroslav Vyshnevetsky", "reasons": ("misheard_match",),
            "meet_html": "<mark>реліз</mark> чекліст", "gromit_html": "release checklist",
            "suggestion": "release checklist"}
    base.update(kw)
    return ReviewRow(**base)


def test_html_is_self_contained():
    html = render_html([_row()], title="Team sync 2026-01-15")
    assert "<!doctype html>" in html.lower()
    assert "http://" not in html and "https://" not in html  # no external assets
    assert "<script" in html and "<style" in html


def test_row_references_relative_clip():
    html = render_html([_row(clip_rel="clips/007.mp4")], title="t")
    assert 'src="clips/007.mp4"' in html


def test_clip_unavailable_marked():
    html = render_html([_row(clip_rel=None)], title="t")
    assert "clip unavailable" in html.lower()
    assert "<video" not in html


def test_suggestion_prefills_correction_input():
    html = render_html([_row(suggestion="release checklist")], title="t")
    assert 'value="release checklist"' in html


def test_speaker_and_readings_present():
    html = render_html([_row()], title="t")
    assert "Yaroslav Vyshnevetsky" in html
    assert "<mark>реліз</mark> чекліст" in html
    assert "release checklist" in html


def test_export_button_and_yaml_builder_present():
    html = render_html([_row()], title="t")
    assert "Export" in html
    assert "corrections.yaml" in html  # JS names the download
    assert "corrections:" in html      # JS builds the YAML body
