"""Tests for orchestrator module."""

from unittest.mock import patch

import numpy as np
import pytest

from gromit.config import Device, ModelSize, TranscriptionConfig
from gromit.orchestrator import Orchestrator


@pytest.fixture
def mock_config(tmp_path):
    """Create a test config."""
    test_file = tmp_path / "test.wav"
    test_file.touch()
    return TranscriptionConfig(
        input_paths=[test_file],
        model_size=ModelSize.TINY,
        device=Device.CPU,
        language="en",
    )


def test_orchestrator_initialization(mock_config):
    """Orchestrator should initialize with config."""
    orchestrator = Orchestrator(mock_config)
    assert orchestrator.config == mock_config


@patch("gromit.orchestrator.AudioProcessor")
@patch("gromit.orchestrator.FasterWhisperTranscriber")
@patch("gromit.orchestrator.PyannoteDiarizer")
def test_orchestrator_process_calls_components(
    mock_diarizer_cls,
    mock_transcriber_cls,
    mock_processor_cls,
    mock_config,
):
    """Orchestrator.process should call all pipeline components."""
    # Setup mocks
    mock_audio = np.zeros(16000, dtype=np.float32)
    mock_processor_cls.return_value.load_multiple.return_value = mock_audio
    mock_processor_cls.return_value.is_valid_audio.return_value = True

    mock_transcriber_cls.return_value.transcribe.return_value = []
    mock_diarizer_cls.return_value.diarize.return_value = []

    orchestrator = Orchestrator(mock_config)
    result = orchestrator.process()

    # Verify components were called
    mock_processor_cls.return_value.load_multiple.assert_called_once()
    mock_transcriber_cls.return_value.transcribe.assert_called_once()
    mock_diarizer_cls.return_value.diarize.assert_called_once()

    assert isinstance(result, str)


@patch("gromit.orchestrator.AudioProcessor")
@patch("gromit.orchestrator.FasterWhisperTranscriber")
@patch("gromit.orchestrator.PyannoteDiarizer")
@patch("gromit.orchestrator.TextFormatter")
def test_orchestrator_passes_file_boundaries_to_formatter(
    mock_formatter_cls,
    mock_diarizer_cls,
    mock_transcriber_cls,
    mock_processor_cls,
    mock_config,
):
    """Orchestrator should pass file boundaries to the formatter."""
    mock_audio = np.zeros(16000, dtype=np.float32)
    mock_processor_cls.return_value.load_multiple.return_value = mock_audio
    mock_processor_cls.return_value.is_valid_audio.return_value = True
    mock_processor_cls.return_value.get_file_boundaries.return_value = [
        ("test.wav", 0.0),
    ]

    mock_transcriber_cls.return_value.transcribe.return_value = []
    mock_diarizer_cls.return_value.diarize.return_value = []
    mock_formatter_cls.return_value.format.return_value = ""

    orchestrator = Orchestrator(mock_config)
    orchestrator.process()

    # Verify formatter was called with file_boundaries
    mock_formatter_cls.return_value.format.assert_called_once()
    call_kwargs = mock_formatter_cls.return_value.format.call_args
    assert "file_boundaries" in call_kwargs.kwargs
    assert call_kwargs.kwargs["file_boundaries"] == [("test.wav", 0.0)]


def test_build_hotwords_from_glossary(tmp_path):
    from pathlib import Path

    from gromit.config import TranscriptionConfig
    from gromit.orchestrator import Orchestrator

    g = tmp_path / "g.yaml"
    g.write_text(
        'terms:\n  - canonical: "release checklist"\n'
        '  - canonical: "Вишневецький"\n    category: person\n'
    )
    cfg = TranscriptionConfig(input_paths=[Path("/x/a.mp4")], glossary_paths=[g])
    orch = Orchestrator(cfg)
    hotwords, hotwords_from = orch._build_hotwords()
    # a list of terms, proper nouns first — the transcriber fits them to budget
    assert hotwords == ["Вишневецький", "release checklist"]
    assert hotwords_from == [str(g)]


def test_build_hotwords_empty_without_glossary():
    from pathlib import Path

    from gromit.config import TranscriptionConfig
    from gromit.orchestrator import Orchestrator

    cfg = TranscriptionConfig(input_paths=[Path("/x/a.mp4")])
    orch = Orchestrator(cfg)
    assert orch._build_hotwords() == (None, [])
