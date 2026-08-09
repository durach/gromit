"""Base interface for transcription backends."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Word:
    """A single word with timing and probability."""

    w: str  # Word text (verbatim from the ASR, may include a leading space)
    start: float  # Start time in seconds
    end: float  # End time in seconds
    p: float  # Probability (0-1)


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing information."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str  # Transcribed text
    confidence: float = 1.0  # Confidence score (0-1)
    avg_logprob: float = 0.0  # Segment average log-probability (from Whisper)
    words: list[Word] = field(default_factory=list)  # Word-level timing/probs


# Progress callback receives (progress: float 0-1, audio_position: float seconds)
ProgressCallback = Callable[[float, float], None] | None


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends."""

    def __init__(self, model_size: str, device: str, language: str) -> None:
        """Initialize transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Compute device (cuda, mps, cpu)
            language: Language code (en, uk, ru, auto)
        """
        self.model_size = model_size
        self.device = device
        self.language = language

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        progress_callback: ProgressCallback = None,
        hotwords: Sequence[str] | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio to text segments.

        Args:
            audio: Audio data as float32 numpy array at 16kHz mono
            progress_callback: Optional callback for progress updates
            hotwords: Optional hotword terms to bias decoding, most important
                first; a backend may drop the tail to fit its prompt budget

        Returns:
            List of transcribed segments with timing
        """
