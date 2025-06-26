"""Format transcription with speaker information."""

from typing import List, Dict
from gromit.diarizer import merge_speaker_segments


def format_conversation(
    transcription: Dict,
    speaker_segments: List[Dict],
    merge_same_speaker: bool = True
) -> str:
    """
    Format transcription with speaker labels.
    
    Args:
        transcription: Transcription result from whisper
        speaker_segments: Speaker diarization segments
        merge_same_speaker: Whether to merge consecutive same-speaker segments
        
    Returns:
        Formatted conversation text
    """
    if not transcription["segments"]:
        return "No transcription available."
    
    # Merge speaker segments if requested
    if merge_same_speaker:
        speaker_segments = merge_speaker_segments(speaker_segments)
    
    # If no speaker segments or only one speaker, format simply
    if not speaker_segments or len(speaker_segments) == 1:
        return _format_single_speaker(transcription)
    
    # Align transcription segments with speaker segments
    aligned_segments = _align_segments(
        transcription["segments"], 
        speaker_segments
    )
    
    # Format output
    lines = []
    current_speaker = None
    current_text = []
    
    for segment in aligned_segments:
        speaker = segment["speaker"]
        text = segment["text"].strip()
        
        if not text:
            continue
        
        if speaker != current_speaker:
            # Output previous speaker's text
            if current_speaker and current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
                lines.append("")  # Empty line between speakers
            
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)
    
    # Don't forget the last speaker
    if current_speaker and current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return "\n".join(lines)


def _format_single_speaker(transcription: Dict) -> str:
    """Format transcription with a single speaker."""
    text_segments = [
        segment["text"].strip() 
        for segment in transcription["segments"]
        if segment["text"].strip()
    ]
    
    # Join segments into paragraphs
    return "Speaker 1: " + " ".join(text_segments)


def _align_segments(
    transcription_segments: List[Dict],
    speaker_segments: List[Dict]
) -> List[Dict]:
    """
    Align transcription segments with speaker segments.
    
    Returns list of segments with both text and speaker information.
    """
    aligned = []
    
    for trans_seg in transcription_segments:
        # Find the speaker segment that contains this transcription
        trans_mid = (trans_seg["start"] + trans_seg["end"]) / 2
        
        speaker = "Unknown"
        for speaker_seg in speaker_segments:
            if speaker_seg["start"] <= trans_mid <= speaker_seg["end"]:
                speaker = speaker_seg["speaker"]
                break
        
        aligned.append({
            "start": trans_seg["start"],
            "end": trans_seg["end"],
            "text": trans_seg["text"],
            "speaker": speaker
        })
    
    return aligned


def format_with_timestamps(
    transcription: Dict,
    speaker_segments: List[Dict]
) -> str:
    """
    Format transcription with timestamps and speaker labels.
    
    Returns formatted text with timestamps.
    """
    aligned_segments = _align_segments(
        transcription["segments"], 
        speaker_segments
    )
    
    lines = []
    for segment in aligned_segments:
        start = _format_timestamp(segment["start"])
        end = _format_timestamp(segment["end"])
        speaker = segment["speaker"]
        text = segment["text"].strip()
        
        if text:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
    
    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"