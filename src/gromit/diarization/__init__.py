"""Speaker diarization module."""

from gromit.diarization.base import SpeakerSegment
from gromit.diarization.pyannote import PyannoteDiarizer

__all__ = ["PyannoteDiarizer", "SpeakerSegment"]
