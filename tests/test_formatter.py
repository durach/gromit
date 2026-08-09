"""Tests for output formatter module."""


from gromit.alignment.temporal import AlignedSegment
from gromit.output.formatter import TextFormatter


def test_format_single_speaker():
    """Formatter should output speaker label and text."""
    segments = [
        AlignedSegment(start=0.0, end=5.0, speaker="SPEAKER_00", text="Hello world"),
    ]

    formatter = TextFormatter()
    result = formatter.format(segments)

    assert "Speaker 1:" in result
    assert "Hello world" in result


def test_format_multiple_speakers():
    """Formatter should label each speaker correctly."""
    segments = [
        AlignedSegment(start=0.0, end=2.0, speaker="SPEAKER_00", text="Hello"),
        AlignedSegment(start=2.0, end=4.0, speaker="SPEAKER_01", text="Hi there"),
        AlignedSegment(start=4.0, end=6.0, speaker="SPEAKER_00", text="How are you"),
    ]

    formatter = TextFormatter()
    result = formatter.format(segments)

    assert "Speaker 1:" in result
    assert "Speaker 2:" in result
    assert "Hello" in result
    assert "Hi there" in result


def test_format_groups_consecutive_same_speaker():
    """Consecutive segments from same speaker should be grouped."""
    segments = [
        AlignedSegment(start=0.0, end=2.0, speaker="SPEAKER_00", text="First part."),
        AlignedSegment(start=2.0, end=4.0, speaker="SPEAKER_00", text="Second part."),
        AlignedSegment(start=4.0, end=6.0, speaker="SPEAKER_01", text="Different speaker."),
    ]

    formatter = TextFormatter()
    result = formatter.format(segments)

    # Should have only 2 speaker labels, not 3
    assert result.count("Speaker 1:") == 1
    assert result.count("Speaker 2:") == 1
    # Text should be combined
    assert "First part." in result
    assert "Second part." in result


def test_format_empty_input():
    """Formatter should handle empty input."""
    formatter = TextFormatter()
    result = formatter.format([])
    assert result == ""


def test_format_cleans_speaker_labels():
    """SPEAKER_00 should become Speaker 1, etc."""
    segments = [
        AlignedSegment(start=0.0, end=1.0, speaker="SPEAKER_02", text="Third speaker"),
    ]

    formatter = TextFormatter()
    result = formatter.format(segments)

    assert "SPEAKER_02" not in result
    assert "Speaker 3:" in result


def test_format_includes_timestamps():
    """Each speaker turn should have a [HH:MM:SS] timestamp."""
    segments = [
        AlignedSegment(start=0.0, end=2.0, speaker="SPEAKER_00", text="Hello"),
        AlignedSegment(start=2.0, end=4.0, speaker="SPEAKER_01", text="Hi there"),
        AlignedSegment(start=4.0, end=6.0, speaker="SPEAKER_00", text="How are you"),
    ]

    formatter = TextFormatter()
    result = formatter.format(segments)

    assert "[00:00:00] Speaker 1:" in result
    assert "[00:00:02] Speaker 2:" in result
    assert "[00:00:04] Speaker 1:" in result


def test_format_timestamp_over_one_hour():
    """Timestamps should handle durations over one hour."""
    segments = [
        AlignedSegment(start=3661.5, end=3670.0, speaker="SPEAKER_00", text="Late segment"),
    ]

    formatter = TextFormatter()
    result = formatter.format(segments)

    assert "[01:01:01] Speaker 1:" in result


def test_format_multi_file_headers_and_restarting_timestamps():
    """Multi-file input should show file headers with restarting timecodes."""
    segments = [
        AlignedSegment(start=0.0, end=5.0, speaker="SPEAKER_00", text="In file one."),
        AlignedSegment(start=10.0, end=15.0, speaker="SPEAKER_01", text="In file two."),
    ]
    file_boundaries = [("memo1.m4a", 0.0), ("memo2.m4a", 10.0)]

    formatter = TextFormatter()
    result = formatter.format(segments, file_boundaries=file_boundaries)

    assert "--- memo1.m4a ---" in result
    assert "--- memo2.m4a ---" in result
    assert "[00:00:00] Speaker 1:" in result
    assert result.count("[00:00:00]") == 2


def test_format_single_file_no_header():
    """Single-file boundary list should not produce a file header."""
    segments = [
        AlignedSegment(start=0.0, end=5.0, speaker="SPEAKER_00", text="Hello"),
    ]
    file_boundaries = [("only_file.m4a", 0.0)]

    formatter = TextFormatter()
    result = formatter.format(segments, file_boundaries=file_boundaries)

    assert "---" not in result
    assert "[00:00:00] Speaker 1:" in result
