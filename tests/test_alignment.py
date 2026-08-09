"""Tests for temporal alignment module."""


from gromit.alignment.temporal import AlignedSegment, TemporalAligner
from gromit.diarization.base import SpeakerSegment
from gromit.transcription.base import TranscriptSegment


def test_aligned_segment_dataclass():
    """AlignedSegment should store combined data."""
    segment = AlignedSegment(
        start=0.0,
        end=5.0,
        speaker="SPEAKER_00",
        text="Hello world",
    )
    assert segment.start == 0.0
    assert segment.text == "Hello world"


def test_align_simple_case():
    """Aligner should assign speaker to transcript based on overlap."""
    transcript = [
        TranscriptSegment(start=0.0, end=2.0, text="Hello"),
        TranscriptSegment(start=2.5, end=4.0, text="World"),
    ]
    speakers = [
        SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.5, end=5.0, speaker="SPEAKER_01"),
    ]

    aligner = TemporalAligner()
    aligned = aligner.align(transcript, speakers)

    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].text == "Hello"
    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].text == "World"


def test_align_with_overlap():
    """When transcript overlaps multiple speakers, assign to max overlap."""
    transcript = [
        TranscriptSegment(start=1.0, end=4.0, text="Overlapping speech"),
    ]
    speakers = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),  # 1.0s overlap
        SpeakerSegment(start=2.0, end=5.0, speaker="SPEAKER_01"),  # 2.0s overlap
    ]

    aligner = TemporalAligner()
    aligned = aligner.align(transcript, speakers)

    assert len(aligned) == 1
    assert aligned[0].speaker == "SPEAKER_01"  # More overlap


def test_align_no_overlap_uses_nearest():
    """When no overlap, assign to nearest speaker segment."""
    transcript = [
        TranscriptSegment(start=5.0, end=6.0, text="After speakers"),
    ]
    speakers = [
        SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=3.0, end=4.5, speaker="SPEAKER_01"),  # Nearest
    ]

    aligner = TemporalAligner()
    aligned = aligner.align(transcript, speakers)

    assert len(aligned) == 1
    assert aligned[0].speaker == "SPEAKER_01"


def test_align_empty_inputs():
    """Aligner should handle empty inputs gracefully."""
    aligner = TemporalAligner()

    # Empty transcript
    assert aligner.align([], [SpeakerSegment(0, 1, "S")]) == []

    # Empty speakers - should still return transcript with unknown speaker
    result = aligner.align(
        [TranscriptSegment(0, 1, "text")],
        [],
    )
    assert len(result) == 1
    assert result[0].speaker == "UNKNOWN"


def test_align_carries_words_and_logprob():
    from gromit.transcription.base import Word

    tseg = TranscriptSegment(
        start=0.0, end=1.0, text="hi",
        avg_logprob=-0.4, words=[Word(w="hi", start=0.0, end=0.5, p=0.8)],
    )
    speakers = [SpeakerSegment(start=0.0, end=1.0, speaker="SPEAKER_00")]
    aligned = TemporalAligner().align([tseg], speakers)
    assert aligned[0].avg_logprob == -0.4
    assert aligned[0].words[0].w == "hi"
    assert aligned[0].speaker == "SPEAKER_00"
