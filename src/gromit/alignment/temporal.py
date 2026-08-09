"""Temporal alignment of transcripts with speaker segments."""

from dataclasses import dataclass, field

from gromit.diarization.base import SpeakerSegment
from gromit.transcription.base import TranscriptSegment, Word


@dataclass
class AlignedSegment:
    """A transcript segment with speaker attribution."""

    start: float
    end: float
    speaker: str
    text: str
    avg_logprob: float = 0.0
    words: list[Word] = field(default_factory=list)


class TemporalAligner:
    """Align transcript segments with speaker segments using temporal overlap."""

    def align(
        self,
        transcript: list[TranscriptSegment],
        speakers: list[SpeakerSegment],
    ) -> list[AlignedSegment]:
        """Align transcript segments to speakers based on temporal overlap.

        Args:
            transcript: List of transcribed segments with timing
            speakers: List of speaker segments with timing

        Returns:
            List of aligned segments with speaker attribution
        """
        if not transcript:
            return []

        result = []
        for tseg in transcript:
            speaker = self._find_speaker(tseg, speakers)
            result.append(
                AlignedSegment(
                    start=tseg.start,
                    end=tseg.end,
                    speaker=speaker,
                    text=tseg.text,
                    avg_logprob=tseg.avg_logprob,
                    words=tseg.words,
                )
            )

        return result

    def _find_speaker(
        self,
        tseg: TranscriptSegment,
        speakers: list[SpeakerSegment],
    ) -> str:
        """Find the speaker with maximum overlap to this transcript segment.

        Args:
            tseg: Transcript segment to match
            speakers: List of speaker segments

        Returns:
            Speaker label, or "UNKNOWN" if no speakers
        """
        if not speakers:
            return "UNKNOWN"

        best_speaker = None
        best_overlap = -1.0

        for sseg in speakers:
            overlap = self._calculate_overlap(tseg, sseg)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg.speaker

        # If no overlap found, use nearest speaker
        if best_overlap <= 0:
            best_speaker = self._find_nearest_speaker(tseg, speakers)

        return best_speaker or "UNKNOWN"

    def _calculate_overlap(
        self,
        tseg: TranscriptSegment,
        sseg: SpeakerSegment,
    ) -> float:
        """Calculate overlap duration between transcript and speaker segment."""
        overlap_start = max(tseg.start, sseg.start)
        overlap_end = min(tseg.end, sseg.end)
        return max(0.0, overlap_end - overlap_start)

    def _find_nearest_speaker(
        self,
        tseg: TranscriptSegment,
        speakers: list[SpeakerSegment],
    ) -> str | None:
        """Find speaker segment nearest to transcript segment."""
        if not speakers:
            return None

        tseg_mid = (tseg.start + tseg.end) / 2
        nearest = min(
            speakers,
            key=lambda s: min(abs(s.start - tseg_mid), abs(s.end - tseg_mid)),
        )
        return nearest.speaker
