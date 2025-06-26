"""Transcription module using faster-whisper."""

from pathlib import Path
import torch
from faster_whisper import WhisperModel
from typing import Dict, List, Optional
import warnings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_device(device: str = "auto") -> tuple[str, bool]:
    """
    Determine the best available device for faster-whisper.
    
    Returns:
        Tuple of (device_to_use, fallback_occurred)
    """
    fallback_occurred = False
    
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda", False
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # For Apple Silicon Macs - faster-whisper doesn't support MPS
            return "cpu", False  # This is expected, not a fallback
        else:
            return "cpu", False
    
    # Handle explicit device requests
    if device == "mps":
        # MPS is not supported by faster-whisper/ctranslate2
        print("⚠️  Warning: MPS device not supported by faster-whisper. Falling back to CPU for transcription.")
        return "cpu", True
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Falling back to CPU for transcription.")
        return "cpu", True
    
    return device, False


def transcribe_audio(
    audio_path: Path, 
    language: str = "en",
    device: str = "auto",
    model_size: str = None,
    verbose: bool = False,
    max_seconds: Optional[float] = None
) -> Dict:
    """
    Transcribe audio file using faster-whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en' for English, 'uk' for Ukrainian)
        device: Device to use ('auto', 'cpu', 'cuda')
        model_size: Whisper model size
        verbose: Enable verbose output
        
    Returns:
        Dictionary with transcription results including segments and text
    """
    device, fallback_occurred = get_device(device)
    
    # Use model size from env if not specified
    if model_size is None:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
    
    if verbose:
        print(f"Loading {model_size} model on {device}...")
    
    # Initialize model with optimal settings for the device
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(
        model_size, 
        device=device,
        compute_type=compute_type,
        download_root=Path.home() / ".cache" / "whisper"
    )
    
    # Handle max_seconds by creating temporary clip
    audio_to_process = str(audio_path)
    if max_seconds is not None:
        import tempfile
        import soundfile as sf
        
        if verbose:
            print(f"Creating temporary clip: first {max_seconds} seconds...")
        
        # Read original audio
        data, sample_rate = sf.read(str(audio_path))
        max_samples = int(max_seconds * sample_rate)
        
        # Trim to max_seconds
        if len(data) > max_samples:
            data = data[:max_samples]
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, data, sample_rate)
        audio_to_process = temp_file.name
        
        if verbose:
            print(f"Temporary file created: {audio_to_process}")
    
    if verbose:
        print(f"Transcribing {audio_to_process}...")
    
    # Transcribe with word timestamps for better alignment with speakers
    segments, info = model.transcribe(
        audio_to_process,
        language=language,
        word_timestamps=True,
        vad_filter=True,  # Voice activity detection
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Minimum silence for speaker change
            threshold=0.5
        )
    )
    
    # Convert generator to list and extract information
    segments_list = []
    full_text = []
    
    for segment in segments:
        segment_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": []
        }
        
        # Add word-level timestamps if available
        if hasattr(segment, "words") and segment.words:
            segment_dict["words"] = [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word.strip()
                }
                for word in segment.words
            ]
        
        segments_list.append(segment_dict)
        full_text.append(segment.text.strip())
    
    # Cleanup temporary file if created
    if max_seconds is not None and audio_to_process != str(audio_path):
        try:
            os.unlink(audio_to_process)
            if verbose:
                print(f"Cleaned up temporary file: {audio_to_process}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not cleanup temporary file: {e}")
    
    result = {
        "text": " ".join(full_text),
        "segments": segments_list,
        "language": info.language,
        "duration": info.duration,
    }
    
    if verbose:
        print(f"Detected language: {info.language}")
        print(f"Duration: {info.duration:.1f}s")
        print(f"Total segments: {len(segments_list)}")
    
    return result