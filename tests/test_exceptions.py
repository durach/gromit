"""Tests for custom exceptions."""

import pytest

from gromit.exceptions import (
    AudioLoadError,
    DiarizationError,
    GromitError,
    TranscriptionError,
)


def test_gromit_error_is_base_exception():
    """All custom exceptions should inherit from GromitError."""
    assert issubclass(AudioLoadError, GromitError)
    assert issubclass(TranscriptionError, GromitError)
    assert issubclass(DiarizationError, GromitError)


def test_exceptions_can_be_raised_with_message():
    """Exceptions should accept and store messages."""
    with pytest.raises(AudioLoadError) as exc_info:
        raise AudioLoadError("Failed to load audio.mp3")
    assert "Failed to load audio.mp3" in str(exc_info.value)
