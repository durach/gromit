"""Tests for transcription base interface."""


import numpy as np
import pytest

from gromit.transcription.base import BaseTranscriber, TranscriptSegment


def test_transcript_segment_dataclass():
    """TranscriptSegment should store segment data."""
    segment = TranscriptSegment(
        start=0.0,
        end=1.5,
        text="Hello world",
        confidence=0.95,
    )
    assert segment.start == 0.0
    assert segment.end == 1.5
    assert segment.text == "Hello world"
    assert segment.confidence == 0.95


def test_base_transcriber_is_abstract():
    """BaseTranscriber should not be instantiable directly."""
    with pytest.raises(TypeError):
        BaseTranscriber(model_size="tiny", device="cpu", language="en")


class MockTranscriber(BaseTranscriber):
    """Concrete implementation for testing."""

    def transcribe(self, audio: np.ndarray) -> list[TranscriptSegment]:
        return [TranscriptSegment(start=0.0, end=1.0, text="test", confidence=1.0)]


def test_concrete_transcriber_can_be_instantiated():
    """Concrete implementations should be instantiable."""
    transcriber = MockTranscriber(model_size="tiny", device="cpu", language="en")
    assert transcriber.model_size == "tiny"
    assert transcriber.device == "cpu"
    assert transcriber.language == "en"


def test_transcript_segment_carries_words_and_logprob():
    from gromit.transcription.base import TranscriptSegment, Word

    seg = TranscriptSegment(
        start=0.0,
        end=1.0,
        text="hi",
        avg_logprob=-0.3,
        words=[Word(w="hi", start=0.0, end=0.5, p=0.9)],
    )
    assert seg.avg_logprob == -0.3
    assert seg.words[0].p == 0.9
    # Defaults preserved for existing callers:
    assert TranscriptSegment(0.0, 1.0, "x").words == []
    assert TranscriptSegment(0.0, 1.0, "x").avg_logprob == 0.0
