import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import soundfile as sf
from pydub import AudioSegment


def is_supported_by_soundfile(file_path: str) -> bool:
    """Check if the file format is directly supported by soundfile."""
    try:
        info = sf.info(file_path)
        return True
    except Exception:
        return False


def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> Tuple[str, bool]:
    """
    Convert audio file to WAV format if needed.
    
    Args:
        input_file: Path to the input audio file
        output_file: Optional path for the output WAV file. If None, creates a temporary file.
        
    Returns:
        Tuple of (output_file_path, is_temporary) where is_temporary indicates if a temp file was created
    """
    input_path = Path(input_file)
    
    # Check if file is already supported by soundfile
    if is_supported_by_soundfile(input_file):
        return input_file, False
    
    # Need to convert the file
    if output_file is None:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        output_file = temp_path
        is_temporary = True
    else:
        is_temporary = False
    
    # Load audio using pydub (supports MP4, MP3, etc via ffmpeg)
    audio = AudioSegment.from_file(input_file)
    
    # Export as WAV
    audio.export(output_file, format="wav")
    
    return output_file, is_temporary


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of an audio file in seconds.
    Supports all formats that pydub can handle.
    """
    if is_supported_by_soundfile(file_path):
        info = sf.info(file_path)
        return info.duration
    else:
        # Use pydub for unsupported formats
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds