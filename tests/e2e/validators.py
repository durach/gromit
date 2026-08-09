"""Validators for e2e test results."""

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of a validation check."""

    success: bool
    error: str = ""
    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def validate_phrases(transcript: str, expected_phrases: list[str]) -> ValidationResult:
    """Check that each phrase appears in transcript (case-insensitive).

    Args:
        transcript: The transcript text to search
        expected_phrases: List of phrases that should appear

    Returns:
        ValidationResult with found/missing phrases
    """
    transcript_lower = transcript.lower()
    found = []
    missing = []

    for phrase in expected_phrases:
        if phrase.lower() in transcript_lower:
            found.append(phrase)
        else:
            missing.append(phrase)

    success = len(missing) == 0
    error = f"Missing phrases: {missing}" if missing else ""

    return ValidationResult(success=success, error=error, found=found, missing=missing)


def validate_structure(transcript: str, expected_speaker_count: int) -> ValidationResult:
    """Validate transcript structure and speaker count.

    Args:
        transcript: The transcript text to validate
        expected_speaker_count: Expected number of unique speakers

    Returns:
        ValidationResult with validation status
    """
    if not transcript or not transcript.strip():
        return ValidationResult(success=False, error="Transcript is empty")

    # Find all speaker labels (e.g., "Speaker 1:", "Speaker 2:")
    speaker_pattern = r"Speaker\s+(\d+):"
    matches = re.findall(speaker_pattern, transcript)

    if not matches:
        return ValidationResult(
            success=False,
            error="No speaker labels found (expected 'Speaker N:' format)",
        )

    unique_speakers = set(matches)
    actual_count = len(unique_speakers)

    if actual_count != expected_speaker_count:
        return ValidationResult(
            success=False,
            error=f"Speaker count mismatch: expected {expected_speaker_count}, found {actual_count}",
        )

    return ValidationResult(success=True)
