"""Audio generator using edge-tts for e2e test fixtures."""

import io
from pathlib import Path

import edge_tts
from pydub import AudioSegment

# Voice mappings by language
VOICES = {
    "en": {
        "voice1": "en-US-GuyNeural",
        "voice2": "en-US-JennyNeural",
    },
    "uk": {
        "voice1": "uk-UA-OstapNeural",
        "voice2": "uk-UA-PolinaNeural",
    },
    "ru": {
        "voice1": "ru-RU-DmitryNeural",
        "voice2": "ru-RU-SvetlanaNeural",
    },
}


async def generate_single_speaker(
    text: str,
    voice: str,
    output_path: Path,
) -> None:
    """Generate single-speaker audio using edge-tts.

    Args:
        text: Text to synthesize
        voice: Edge-tts voice name
        output_path: Path to save the MP3 file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


async def generate_two_speakers(
    exchanges: list[tuple[str, str]],
    silence_gap: float,
    output_path: Path,
) -> None:
    """Generate two-speaker audio by concatenating TTS segments.

    Args:
        exchanges: List of (text, voice) tuples
        silence_gap: Silence duration in seconds between exchanges
        output_path: Path to save the final MP3 file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate each segment
    segments = []
    silence = AudioSegment.silent(duration=int(silence_gap * 1000))

    for i, (text, voice) in enumerate(exchanges):
        # Generate to bytes buffer
        communicate = edge_tts.Communicate(text, voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        # Load as AudioSegment
        segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        segments.append(segment)

        # Add silence between segments (not after last)
        if i < len(exchanges) - 1:
            segments.append(silence)

    # Concatenate all segments
    combined = segments[0]
    for segment in segments[1:]:
        combined = combined + segment

    # Export
    combined.export(str(output_path), format="mp3")
