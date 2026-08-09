"""Transcription module."""

from gromit.transcription.base import BaseTranscriber, TranscriptSegment
from gromit.transcription.faster_whisper import FasterWhisperTranscriber

__all__ = ["BaseTranscriber", "FasterWhisperTranscriber", "TranscriptSegment"]
