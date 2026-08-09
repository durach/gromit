"""Tests for speaker diarization module."""

import os
from pathlib import Path

import pytest

from gromit.audio.processor import AudioProcessor
from gromit.diarization.base import SpeakerSegment
from gromit.diarization.pyannote import PyannoteDiarizer

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_speaker_segment_dataclass():
    """SpeakerSegment should store speaker timing data."""
    segment = SpeakerSegment(
        start=0.0,
        end=5.0,
        speaker="SPEAKER_00",
    )
    assert segment.start == 0.0
    assert segment.end == 5.0
    assert segment.speaker == "SPEAKER_00"


@pytest.fixture
def audio_processor():
    return AudioProcessor()


@pytest.fixture
def test_audio(audio_processor):
    """Load test audio fixture."""
    return audio_processor.load(FIXTURES_DIR / "test_tone.wav")


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for pyannote models",
)
def test_diarizer_initialization():
    """Diarizer should initialize with model."""
    diarizer = PyannoteDiarizer(device="cpu")
    assert diarizer.pipeline is not None


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for pyannote models",
)
def test_diarize_returns_segments(test_audio):
    """Diarizing audio should return speaker segment list."""
    diarizer = PyannoteDiarizer(device="cpu")
    segments = diarizer.diarize(test_audio)
    assert isinstance(segments, list)
    # All items should be SpeakerSegment
    for seg in segments:
        assert isinstance(seg, SpeakerSegment)
