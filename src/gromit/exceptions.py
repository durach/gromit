"""Custom exceptions for Gromit."""


class GromitError(Exception):
    """Base exception for all Gromit errors."""


class AudioLoadError(GromitError):
    """Failed to load or process audio file."""


class TranscriptionError(GromitError):
    """Transcription processing failed."""


class DiarizationError(GromitError):
    """Speaker diarization failed."""


class GlossaryError(GromitError):
    """Glossary file is missing, malformed, or has conflicting entries."""


class CrosscheckError(GromitError):
    """Crosscheck input is missing or malformed, or the file pairing is wrong."""
