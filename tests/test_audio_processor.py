"""Tests for audio processor module."""

from pathlib import Path

import numpy as np
import pytest

from gromit.audio.processor import AudioProcessor
from gromit.exceptions import AudioLoadError

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_wav_returns_numpy_array():
    """Loading audio should return a numpy array."""
    processor = AudioProcessor()
    audio = processor.load(FIXTURES_DIR / "test_tone.wav")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32


def test_load_wav_resamples_to_16khz():
    """Audio should be resampled to 16kHz."""
    processor = AudioProcessor()
    audio = processor.load(FIXTURES_DIR / "test_tone.wav")
    # 2 seconds at 16kHz = 32000 samples
    assert len(audio) == 32000


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("audio.mp4", True),
        ("audio.mkv", True),
        ("audio.m4a", True),
        ("audio.wav", False),
        ("audio.mp3", False),
        ("audio.flac", False),
    ],
)
def test_needs_ffmpeg_extraction(filename, expected):
    """M4a and video containers should be routed through ffmpeg."""
    processor = AudioProcessor()
    assert processor._needs_ffmpeg_extraction(Path(filename)) == expected


def test_load_nonexistent_file_raises_error():
    """Loading nonexistent file should raise AudioLoadError."""
    processor = AudioProcessor()
    with pytest.raises(AudioLoadError):
        processor.load(Path("nonexistent.mp3"))


def test_load_validates_audio_not_silent():
    """Processor should detect if audio has meaningful content (not silent)."""
    processor = AudioProcessor()
    audio = processor.load(FIXTURES_DIR / "test_tone.wav")
    # Our test tone should not be silent
    assert processor.is_valid_audio(audio)


def test_load_with_max_duration():
    """Loading with max_duration should truncate audio."""
    processor = AudioProcessor()
    # Load only first 1 second of 2-second file
    audio = processor.load(FIXTURES_DIR / "test_tone.wav", max_duration=1.0)
    # 1 second at 16kHz = 16000 samples
    assert len(audio) == 16000


def test_load_max_duration_longer_than_file():
    """If max_duration exceeds file length, return full file."""
    processor = AudioProcessor()
    audio = processor.load(FIXTURES_DIR / "test_tone.wav", max_duration=10.0)
    # Full 2-second file = 32000 samples
    assert len(audio) == 32000


def test_load_multiple_single_file():
    """load_multiple with single file should work like load."""
    processor = AudioProcessor()
    audio = processor.load_multiple([FIXTURES_DIR / "test_tone.wav"])
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 32000  # 2 seconds at 16kHz


def test_load_multiple_concatenates_files():
    """load_multiple should concatenate multiple files."""
    processor = AudioProcessor()
    # Load same file twice - should get double the samples
    audio = processor.load_multiple([
        FIXTURES_DIR / "test_tone.wav",
        FIXTURES_DIR / "test_tone.wav",
    ])
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 64000  # 4 seconds at 16kHz (2 + 2)


def test_load_multiple_empty_list_raises_error():
    """load_multiple with empty list should raise AudioLoadError."""
    processor = AudioProcessor()
    with pytest.raises(AudioLoadError):
        processor.load_multiple([])


def test_load_multiple_nonexistent_file_raises_error():
    """load_multiple should raise AudioLoadError if any file missing."""
    processor = AudioProcessor()
    with pytest.raises(AudioLoadError):
        processor.load_multiple([
            FIXTURES_DIR / "test_tone.wav",
            Path("nonexistent.mp3"),
        ])


def test_get_file_boundaries_returns_offsets():
    """get_file_boundaries should return (filename, offset) tuples."""
    processor = AudioProcessor()
    paths = [FIXTURES_DIR / "test_tone.wav"]  # 2-second fixture
    boundaries = processor.get_file_boundaries(paths)

    assert len(boundaries) == 1
    assert boundaries[0][0] == "test_tone.wav"
    assert boundaries[0][1] == 0.0


def test_get_file_boundaries_multiple_files():
    """Multiple files should have cumulative offsets."""
    processor = AudioProcessor()
    # Use same file twice to test offset accumulation
    paths = [FIXTURES_DIR / "test_tone.wav", FIXTURES_DIR / "test_tone.wav"]
    boundaries = processor.get_file_boundaries(paths)

    assert len(boundaries) == 2
    assert boundaries[0] == ("test_tone.wav", 0.0)
    # Second file offset should be ~2.0 (duration of first file)
    assert boundaries[1][0] == "test_tone.wav"
    assert abs(boundaries[1][1] - 2.0) < 0.1
