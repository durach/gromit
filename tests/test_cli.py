"""Tests for CLI module."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from gromit.cli import app, parse_choice
from gromit.config import Device, ModelSize

runner = CliRunner()


def unwrapped(output: str) -> str:
    """Collapse Rich's console line-wrapping so a message can be matched whole.

    Rich hard-wraps to the console width, so `Choose one of: tiny, base, small,
    medium, large-v3` arrives split across two lines. Asserting on the raw
    output would make these tests depend on terminal width.
    """
    return " ".join(output.split())


def test_cli_import_does_not_pull_pyannote():
    """Importing gromit.cli must not transitively import pyannote / torch.

    A module-level import of gromit.orchestrator in cli.py would drag in
    pyannote and torch (and the torchcodec UserWarning + objc duplicate-class
    chatter) even for `gromit transcribe --help`. cli.py imports the
    orchestrator lazily inside the command body to keep startup clean. A
    subprocess gives us a clean interpreter so we measure first-import
    behavior, not state cached by other tests in this file.
    """
    code = (
        "import gromit.cli, sys; "
        "loaded = sorted(m for m in sys.modules "
        "if m == 'torch' or m.startswith(('pyannote.', 'torch.', 'torchcodec'))); "
        "print('LOADED=' + ','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = result.stdout.strip().removeprefix("LOADED=").split(",") if "LOADED=" in result.stdout else []
    leaked = [m for m in loaded if m]
    assert not leaked, (
        f"gromit.cli pulled heavy audio-pipeline deps at import time: {leaked}"
    )


def test_transcribe_command_exists():
    """The transcribe command should exist and show help."""
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "INPUT_FILES" in result.output or "input_files" in result.output.lower()


def test_transcribe_requires_input_file():
    """Transcribe should error when no input file provided."""
    result = runner.invoke(app, [])
    assert result.exit_code != 0


@patch("gromit.orchestrator.Orchestrator")
def test_transcribe_calls_orchestrator(mock_orchestrator_cls, tmp_path):
    """Transcribe command should create orchestrator and process."""
    # Create a test file
    test_file = tmp_path / "test.mp3"
    test_file.touch()

    # Setup mock
    mock_orchestrator = MagicMock()
    mock_orchestrator.process.return_value = "Speaker 1:\nHello world"
    mock_orchestrator.transcript_json.return_value = {
        "language": "en",
        "model": "large-v3",
        "hotwords_from": [],
        "segments": [],
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    runner.invoke(app, ["transcribe", str(test_file)])

    # Should have called orchestrator
    mock_orchestrator_cls.assert_called_once()
    mock_orchestrator.process.assert_called_once()


def test_transcribe_validates_file_exists():
    """Transcribe should error for nonexistent file."""
    result = runner.invoke(app, ["transcribe", "nonexistent.mp3"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


@patch("gromit.orchestrator.Orchestrator")
def test_transcribe_multiple_files(mock_orchestrator_cls, tmp_path):
    """Transcribe should accept multiple input files."""
    # Create test files
    file1 = tmp_path / "part1.mp3"
    file2 = tmp_path / "part2.mp3"
    file1.touch()
    file2.touch()

    # Setup mock
    mock_orchestrator = MagicMock()
    mock_orchestrator.process.return_value = "Speaker 1:\nHello world"
    mock_orchestrator.transcript_json.return_value = {
        "language": "en",
        "model": "large-v3",
        "hotwords_from": [],
        "segments": [],
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    result = runner.invoke(app, ["transcribe", str(file1), str(file2)])

    assert result.exit_code == 0
    # Verify config has both files
    call_args = mock_orchestrator_cls.call_args
    config = call_args[0][0]
    assert len(config.input_paths) == 2


def test_transcribe_validates_all_files_exist(tmp_path):
    """Transcribe should error if any file doesn't exist."""
    file1 = tmp_path / "exists.mp3"
    file1.touch()

    result = runner.invoke(app, ["transcribe", str(file1), "nonexistent.mp3"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


@patch("gromit.orchestrator.Orchestrator")
def test_transcribe_from_file(mock_orchestrator_cls, tmp_path):
    """--from-file should read paths from text file and transcribe."""
    # Create media files
    file1 = tmp_path / "part1.mp4"
    file2 = tmp_path / "part2.mp4"
    file1.touch()
    file2.touch()

    # Create list file
    list_file = tmp_path / "day1.txt"
    list_file.write_text("part1.mp4\npart2.mp4\n")

    # Setup mock
    mock_orchestrator = MagicMock()
    mock_orchestrator.process.return_value = "Speaker 1:\nHello"
    mock_orchestrator.transcript_json.return_value = {
        "language": "en",
        "model": "large-v3",
        "hotwords_from": [],
        "segments": [],
    }
    mock_orchestrator_cls.return_value = mock_orchestrator

    result = runner.invoke(app, ["transcribe", "--from-file", str(list_file)])

    assert result.exit_code == 0
    config = mock_orchestrator_cls.call_args[0][0]
    assert len(config.input_paths) == 2
    assert config.from_file_path == list_file


def test_transcribe_from_file_and_positional_args_errors(tmp_path):
    """Using both --from-file and positional args should error."""
    file1 = tmp_path / "part1.mp4"
    file1.touch()
    list_file = tmp_path / "day1.txt"
    list_file.write_text("part1.mp4\n")

    result = runner.invoke(app, ["transcribe", str(file1), "--from-file", str(list_file)])
    assert result.exit_code != 0
    assert "cannot use both" in result.output.lower() or "mutually exclusive" in result.output.lower()


def test_transcribe_neither_files_nor_from_file():
    """No positional args and no --from-file should show help / error."""
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_transcribe_from_file_nonexistent_list():
    """--from-file with nonexistent list file should error."""
    result = runner.invoke(app, ["transcribe", "--from-file", "nonexistent.txt"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


def test_transcribe_from_file_validates_media_files(tmp_path):
    """--from-file should validate that listed media files exist."""
    list_file = tmp_path / "day1.txt"
    list_file.write_text("nonexistent.mp4\n")

    result = runner.invoke(app, ["transcribe", "--from-file", str(list_file)])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


@patch("gromit.orchestrator.Orchestrator")
def test_transcribe_writes_json_and_passes_glossary(mock_orchestrator_cls, tmp_path):
    """--glossary reaches the config; a .gromit.json is written beside the txt."""
    media = tmp_path / "Recording.mp4"
    media.touch()
    gloss = tmp_path / "g.yaml"
    gloss.write_text('terms:\n  - canonical: "release checklist"\n')

    mock_orch = MagicMock()
    mock_orch.process.return_value = "Speaker 1:\nHello"
    mock_orch.transcript_json.return_value = {
        "language": "uk",
        "model": "large-v3",
        "hotwords_from": [str(gloss)],
        "segments": [],
    }
    mock_orchestrator_cls.return_value = mock_orch

    result = runner.invoke(app, ["transcribe", str(media), "--glossary", str(gloss)])
    assert result.exit_code == 0, result.output

    config = mock_orchestrator_cls.call_args[0][0]
    assert config.glossary_paths == [gloss]

    json_path = tmp_path / "Recording.gromit.json"
    assert json_path.exists()
    import json as _json

    payload = _json.loads(json_path.read_text())
    assert payload["hotwords_from"] == [str(gloss)]


def test_crosscheck_writes_flags(tmp_path):
    """crosscheck reads a .gromit.json + Meet VTT and writes flags.json."""
    import json

    gp = tmp_path / "r.gromit.json"
    gp.write_text(json.dumps({
        "language": "uk", "model": "large-v3", "hotwords_from": [],
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "S", "text": "соломія вербицька",
             "avg_logprob": -0.1,
             "words": [{"w": "соломія", "start": 0.0, "end": 3.0, "p": 0.9}]},
        ],
    }, ensure_ascii=False))
    vp = tmp_path / "r.vtt"
    vp.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:03.000\nсьогодні тепло і сонячно\n")
    out = tmp_path / "flags.json"

    result = runner.invoke(
        app, ["crosscheck", str(gp), "--meet", str(vp), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["spans"]
    assert "divergence" in payload["spans"][0]["reasons"]


def test_crosscheck_without_meet_runs(tmp_path):
    """crosscheck works with no --meet (low_confidence + misheard only)."""
    import json

    gp = tmp_path / "r.gromit.json"
    gp.write_text(json.dumps({
        "language": "uk", "model": "large-v3", "hotwords_from": [],
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "S", "text": "щось",
             "avg_logprob": -0.9,
             "words": [{"w": "щось", "start": 0.0, "end": 3.0, "p": 0.9}]},
        ],
    }, ensure_ascii=False))
    out = tmp_path / "flags.json"

    result = runner.invoke(app, ["crosscheck", str(gp), "-o", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert "low_confidence" in payload["spans"][0]["reasons"]


def test_review_cli_writes_page(tmp_path):
    """review builds index.html from flags.json (clip extraction stubbed)."""
    import json
    from unittest.mock import patch

    fp = tmp_path / "flags.json"
    fp.write_text(json.dumps({"spans": [
        {"start": 1.0, "end": 2.0, "meet_text": "реліз чекліст",
         "gromit_text": "release checklist", "reasons": ["misheard_match"],
         "suggestion": "release checklist"},
    ]}, ensure_ascii=False))
    out = tmp_path / "review"

    with patch("gromit.review.core.extract_clip", return_value=True):
        result = runner.invoke(
            app, ["review", str(fp), "--video", str(tmp_path / "v.mp4"), "-o", str(out)]
        )
    assert result.exit_code == 0, result.output
    assert (out / "index.html").exists()
    assert "release checklist" in (out / "index.html").read_text()


def test_glossary_merge_cli(tmp_path):
    """glossary-merge appends a correction's heard-as to the glossary."""
    gp = tmp_path / "glossary.yaml"
    gp.write_text('terms:\n  - canonical: "Вишневецький"\n    category: person\n    misheard: []\n')
    cp = tmp_path / "corrections.yaml"
    cp.write_text('corrections:\n  - canonical: "Вишневецький"\n    heard: "Кишневецьки"\n    category: person\n')

    result = runner.invoke(app, ["glossary-merge", str(cp), "--glossary", str(gp)])
    assert result.exit_code == 0, result.output
    assert "Кишневецьки" in gp.read_text()


# --- --model / --device validation (cli.parse_choice) -----------------------
#
# These options are declared as plain `str` rather than as the ModelSize/Device
# enums, so Typer validates nothing for them. `parse_choice` does it instead,
# inside `transcribe`'s existing try block, which is what keeps the promise in
# ARCHITECTURE.md §7 that an expected failure prints `Error: …` and exits 1
# rather than raising a bare ValueError through Rich's traceback handler.


@pytest.mark.parametrize(
    ("value", "choices", "expected"),
    [
        ("tiny", ModelSize, ModelSize.TINY),
        ("large-v3", ModelSize, ModelSize.LARGE_V3),
        ("auto", Device, Device.AUTO),
        ("cpu", Device, Device.CPU),
    ],
)
def test_parse_choice_accepts_exact_enum_values(value, choices, expected):
    """A permitted value comes back as the enum member, not the string."""
    assert parse_choice(value, choices, "--flag") is expected


def test_parse_choice_rejects_unknown_model_and_names_value_and_choices():
    """The message must name the bad value AND every permitted one."""
    with pytest.raises(typer.BadParameter) as excinfo:
        parse_choice("largev3", ModelSize, "--model")

    message = str(excinfo.value)
    assert "largev3" in message
    assert "--model" in message
    for allowed in ("tiny", "base", "small", "medium", "large-v3"):
        assert allowed in message


def test_parse_choice_rejects_unknown_device_and_names_value_and_choices():
    """Same contract for --device, which routes through the same helper."""
    with pytest.raises(typer.BadParameter) as excinfo:
        parse_choice("gpu", Device, "--device")

    message = str(excinfo.value)
    assert "gpu" in message
    assert "--device" in message
    for allowed in ("auto", "cuda", "mps", "cpu"):
        assert allowed in message


@pytest.mark.parametrize("value", ["TINY", "Tiny", "LARGE-V3"])
def test_parse_choice_is_case_sensitive(value):
    """Matching is exact: enum lookup is by value, so case is not folded.

    Documenting the real behaviour rather than the convenient one — if this is
    ever softened to accept `--model TINY`, this test is the place to say so.
    """
    with pytest.raises(typer.BadParameter):
        parse_choice(value, ModelSize, "--model")


@pytest.mark.parametrize("value", [" tiny", "tiny ", " tiny "])
def test_parse_choice_does_not_strip_surrounding_whitespace(value):
    """Padding is not trimmed either; `!r` quoting makes it visible in the error."""
    with pytest.raises(typer.BadParameter) as excinfo:
        parse_choice(value, ModelSize, "--model")

    assert repr(value) in str(excinfo.value)


def test_transcribe_rejects_bad_model_without_traceback(tmp_path):
    """A mistyped --model exits 1 with `Error: …` and no traceback."""
    media = tmp_path / "meeting.mp4"
    media.touch()

    result = runner.invoke(app, ["transcribe", str(media), "--model", "largev3"])

    assert result.exit_code == 1
    output = unwrapped(result.output)
    assert "Error: Invalid --model: 'largev3'." in output
    assert "Choose one of: tiny, base, small, medium, large-v3" in output
    assert "Traceback" not in result.output
    assert "ValueError" not in result.output


def test_transcribe_rejects_bad_device_without_traceback(tmp_path):
    """Same for a mistyped --device."""
    media = tmp_path / "meeting.mp4"
    media.touch()

    result = runner.invoke(app, ["transcribe", str(media), "--device", "gpu"])

    assert result.exit_code == 1
    output = unwrapped(result.output)
    assert "Error: Invalid --device: 'gpu'." in output
    assert "Choose one of: auto, cuda, mps, cpu" in output
    assert "Traceback" not in result.output
    assert "ValueError" not in result.output


@patch("gromit.orchestrator.Orchestrator")
def test_transcribe_passes_valid_model_and_device_to_config(mock_orchestrator_cls, tmp_path):
    """The happy path still reaches the config as enum members."""
    media = tmp_path / "meeting.mp4"
    media.touch()

    mock_orch = MagicMock()
    mock_orch.process.return_value = "Speaker 1:\nHello"
    mock_orch.transcript_json.return_value = {
        "language": "en",
        "model": "small",
        "hotwords_from": [],
        "segments": [],
    }
    mock_orchestrator_cls.return_value = mock_orch

    result = runner.invoke(
        app, ["transcribe", str(media), "--model", "small", "--device", "cpu"]
    )

    assert result.exit_code == 0, result.output
    config = mock_orchestrator_cls.call_args[0][0]
    assert config.model_size is ModelSize.SMALL
    assert config.device is Device.CPU
