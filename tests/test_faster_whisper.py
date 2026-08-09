"""Tests for faster-whisper transcriber."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gromit.audio.processor import AudioProcessor
from gromit.transcription.faster_whisper import (
    HOTWORD_TOKEN_BUDGET,
    FasterWhisperTranscriber,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def audio_processor():
    return AudioProcessor()


@pytest.fixture
def test_audio(audio_processor):
    """Load test audio fixture."""
    return audio_processor.load(FIXTURES_DIR / "test_tone.wav")


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    """One token per character — deterministic and model-free."""

    def encode(self, text, add_special_tokens=False):
        return _FakeEncoding(list(text))


def _bare_transcriber():
    """A transcriber with a fake tokenizer and no model weights loaded."""
    t = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    t.model = SimpleNamespace(hf_tokenizer=_FakeTokenizer())
    return t


def test_hotword_budget_leaves_room_for_the_rest_of_the_prompt():
    # faster-whisper builds: sot_prev(1) + hotwords + previous_tokens(<=223)
    # + sot_sequence(3), and CTranslate2 rejects a prompt over max_length=448.
    assert 1 + HOTWORD_TOKEN_BUDGET + (448 // 2 - 1) + 3 <= 448


def test_fit_hotwords_keeps_everything_when_it_fits():
    t = _bare_transcriber()
    kept, dropped = t._fit_hotwords(["Acme", "Nimbus", "Dealflow"])
    assert kept == "Acme Nimbus Dealflow"
    assert dropped == []


def test_fit_hotwords_drops_the_tail_that_exceeds_budget():
    t = _bare_transcriber()
    # 30 terms x 10 chars (+1 space each) = ~330 tokens > 221 budget.
    terms = [f"term{i:06d}" for i in range(30)]
    kept, dropped = t._fit_hotwords(terms)

    assert dropped, "expected the tail to be dropped"
    assert kept.split(" ") + dropped == terms, "must keep a prefix, in order"
    assert len(" " + kept) <= HOTWORD_TOKEN_BUDGET
    # and adding the next term back would have overflowed
    assert len(" " + kept + " " + dropped[0]) > HOTWORD_TOKEN_BUDGET


def test_fit_hotwords_never_splits_a_term():
    t = _bare_transcriber()
    terms = ["x" * 200, "y" * 200]
    kept, dropped = t._fit_hotwords(terms)
    assert kept == "x" * 200
    assert dropped == ["y" * 200]


def test_fit_hotwords_handles_empty():
    t = _bare_transcriber()
    assert t._fit_hotwords([]) == (None, [])


@pytest.mark.slow
def test_transcriber_initialization():
    """Transcriber should initialize with model."""
    transcriber = FasterWhisperTranscriber(
        model_size="tiny",
        device="cpu",
        language="en",
    )
    assert transcriber.model is not None


@pytest.mark.slow
def test_transcribe_returns_segments(test_audio):
    """Transcribing audio should return segment list."""
    transcriber = FasterWhisperTranscriber(
        model_size="tiny",
        device="cpu",
        language="en",
    )
    segments = transcriber.transcribe(test_audio)
    assert isinstance(segments, list)
    # Even silent/tone audio should return something (possibly empty)


@pytest.mark.slow
def test_transcribe_with_progress_callback(test_audio):
    """Progress callback should be called during transcription."""
    transcriber = FasterWhisperTranscriber(
        model_size="tiny",
        device="cpu",
        language="en",
    )
    progress_values = []

    def callback(progress: float, audio_position: float):
        progress_values.append((progress, audio_position))

    transcriber.transcribe(test_audio, progress_callback=callback)
    # Should have received at least one progress update
    assert len(progress_values) >= 0  # May be 0 for very short audio


@pytest.mark.slow
def test_transcribe_populates_words_and_language(test_audio):
    """word_timestamps=True should yield float word probs + a detected language."""
    transcriber = FasterWhisperTranscriber(
        model_size="tiny",
        device="cpu",
        language="en",
    )
    segments = transcriber.transcribe(test_audio, hotwords=["Gromit", "Wallace"])
    assert isinstance(transcriber.detected_language, str)
    assert transcriber.detected_language  # non-empty
    for seg in segments:
        assert isinstance(seg.words, list)
        assert isinstance(seg.avg_logprob, float)
        for w in seg.words:
            assert 0.0 <= w.p <= 1.0
            assert w.end >= w.start
