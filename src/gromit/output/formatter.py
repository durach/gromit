"""Output formatting for transcription results."""

import re

from gromit.alignment.temporal import AlignedSegment


class TextFormatter:
    """Format aligned segments as plain text with speaker labels."""

    def format(
        self,
        segments: list[AlignedSegment],
        file_boundaries: list[tuple[str, float]] | None = None,
    ) -> str:
        """Format segments as speaker-attributed text.

        Groups consecutive segments from the same speaker and formats as:

            [00:00:00] Speaker 1:
            Text from speaker 1...

            [00:02:15] Speaker 2:
            Text from speaker 2...

        Args:
            segments: List of aligned segments with speaker attribution
            file_boundaries: Optional list of (filename, start_offset) tuples
                for multi-file input. When provided with >1 entry, inserts
                file headers and restarts timestamps per file.

        Returns:
            Formatted text string
        """
        if not segments:
            return ""

        speaker_labels = self._build_speaker_labels(segments)
        multi_file = file_boundaries is not None and len(file_boundaries) > 1

        if not multi_file:
            groups = self._group_by_speaker(segments)
            offset = file_boundaries[0][1] if file_boundaries else 0.0
            return self._format_groups(groups, speaker_labels, offset)

        # Multi-file: split segments by file boundary and format each section
        sections = []
        for i, (filename, offset) in enumerate(file_boundaries):
            next_offset = file_boundaries[i + 1][1] if i + 1 < len(file_boundaries) else float("inf")
            file_segments = [s for s in segments if offset <= s.start < next_offset]
            if not file_segments:
                continue
            groups = self._group_by_speaker(file_segments)
            header = f"--- {filename} ---"
            body = self._format_groups(groups, speaker_labels, offset)
            sections.append(f"{header}\n\n{body}")

        return "\n\n".join(sections)

    def _format_groups(
        self,
        groups: list[tuple[str, float, list[str]]],
        speaker_labels: dict[str, str],
        time_offset: float = 0.0,
    ) -> str:
        """Format speaker groups as timestamped text."""
        output_parts = []
        for speaker, start_time, texts in groups:
            label = speaker_labels.get(speaker, speaker)
            timestamp = self._format_timestamp(start_time - time_offset)
            combined_text = " ".join(texts)
            output_parts.append(f"{timestamp} {label}:\n{combined_text}")
        return "\n\n".join(output_parts)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as [HH:MM:SS]."""
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"[{h:02d}:{m:02d}:{s:02d}]"

    def _group_by_speaker(
        self,
        segments: list[AlignedSegment],
    ) -> list[tuple[str, float, list[str]]]:
        """Group consecutive segments by speaker.

        Returns list of (speaker, start_time, [texts]) tuples.
        """
        if not segments:
            return []

        groups = []
        current_speaker = segments[0].speaker
        current_start = segments[0].start
        current_texts = [segments[0].text]

        for seg in segments[1:]:
            if seg.speaker == current_speaker:
                current_texts.append(seg.text)
            else:
                groups.append((current_speaker, current_start, current_texts))
                current_speaker = seg.speaker
                current_start = seg.start
                current_texts = [seg.text]

        # Don't forget the last group
        groups.append((current_speaker, current_start, current_texts))

        return groups

    def _build_speaker_labels(
        self,
        segments: list[AlignedSegment],
    ) -> dict[str, str]:
        """Build mapping from internal speaker IDs to friendly labels.

        SPEAKER_00 -> Speaker 1
        SPEAKER_01 -> Speaker 2
        etc.
        """
        # Find unique speakers in order of appearance
        seen = set()
        speakers_ordered = []
        for seg in segments:
            if seg.speaker not in seen:
                seen.add(seg.speaker)
                speakers_ordered.append(seg.speaker)

        # Build mapping
        labels = {}
        for i, speaker in enumerate(speakers_ordered):
            # Try to extract number from SPEAKER_XX format
            match = re.search(r"SPEAKER_(\d+)", speaker)
            if match:
                num = int(match.group(1)) + 1  # 0-indexed to 1-indexed
            else:
                num = i + 1
            labels[speaker] = f"Speaker {num}"

        return labels
