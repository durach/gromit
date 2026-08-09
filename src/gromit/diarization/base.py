"""Base interface for speaker diarization."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SpeakerSegment:
    """A segment attributed to a specific speaker."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker: str  # Speaker label (SPEAKER_00, SPEAKER_01, etc.)


# Progress callback type
ProgressCallback = Callable[[float], None] | None


class BaseDiarizer(ABC):
    """Abstract base class for diarization backends."""

    def __init__(self, device: str, num_speakers: int | None = None) -> None:
        """Initialize diarizer.

        Args:
            device: Compute device (cuda, mps, cpu)
            num_speakers: Expected number of speakers (None = auto-detect)
        """
        self.device = device
        self.num_speakers = num_speakers

    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        progress_callback: ProgressCallback = None,
    ) -> list[SpeakerSegment]:
        """Identify speakers in audio.

        Args:
            audio: Audio data as float32 numpy array at 16kHz mono
            progress_callback: Optional callback for progress updates

        Returns:
            List of speaker segments with timing
        """
