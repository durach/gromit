"""Speaker diarization module using pyannote.audio."""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import warnings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")


def diarize_audio(
    audio_path: Path,
    device: str = "auto",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
    max_seconds: Optional[float] = None
) -> List[Dict]:
    """
    Perform speaker diarization on audio file.
    
    Args:
        audio_path: Path to audio file
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        hf_token: Hugging Face token for model access
        
    Returns:
        List of speaker segments with start/end times and speaker labels
    """
    # Import here to avoid issues if user hasn't set up HF token yet
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for speaker diarization. "
            "Please ensure it's installed."
        ) from e
    
    # Get HF token from environment if not provided
    if hf_token is None:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", None)
    
    if hf_token is None:
        # Return simple fallback if no token
        print("Warning: No Hugging Face token found. Using simple speaker detection.")
        return _simple_speaker_segments(audio_path, max_seconds)
    
    # Debug mode
    if os.getenv("DEBUG", "false").lower() == "true":
        print(f"Using HF token: {hf_token[:10]}..." if hf_token else "No token")
    
    # Handle max_seconds by creating temporary clip
    audio_to_process = str(audio_path)
    if max_seconds is not None:
        import tempfile
        import soundfile as sf
        
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"Creating temporary clip for diarization: first {max_seconds} seconds...")
        
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
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("cpu")  # MPS can have issues with pyannote
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    try:
        # Load pretrained pipeline
        if os.getenv("DEBUG", "false").lower() == "true":
            print("Loading speaker diarization model...")
            
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to device
        pipeline.to(device)
        
        # Set number of speakers if provided
        diarization_params = {}
        if num_speakers is not None:
            diarization_params["num_speakers"] = num_speakers
        elif min_speakers is not None or max_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
            diarization_params["max_speakers"] = max_speakers
        
        # Run diarization
        diarization = pipeline(audio_to_process, **diarization_params)
        
        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })
        
        # Sort by start time
        segments.sort(key=lambda x: x["start"])
        
        # Normalize speaker labels to Speaker 1, Speaker 2, etc.
        speaker_mapping = {}
        for segment in segments:
            if segment["speaker"] not in speaker_mapping:
                speaker_mapping[segment["speaker"]] = f"Speaker {len(speaker_mapping) + 1}"
            segment["speaker"] = speaker_mapping[segment["speaker"]]
        
        # Cleanup temporary file if created
        if max_seconds is not None and audio_to_process != str(audio_path):
            try:
                os.unlink(audio_to_process)
                if os.getenv("DEBUG", "false").lower() == "true":
                    print(f"Cleaned up diarization temporary file: {audio_to_process}")
            except Exception as e:
                if os.getenv("DEBUG", "false").lower() == "true":
                    print(f"Warning: Could not cleanup diarization temp file: {e}")
        
        return segments
        
    except Exception as e:
        print(f"\nWarning: Speaker diarization failed: {e}")
        print("\nPossible solutions:")
        print("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   and accept the model license agreement")
        print("2. Verify your token has 'read' permissions")
        print("3. Check that your token is correctly set in .env file")
        print("\nFalling back to simple speaker detection.\n")
        
        # Cleanup temporary file if created before falling back
        if max_seconds is not None and 'audio_to_process' in locals() and audio_to_process != str(audio_path):
            try:
                os.unlink(audio_to_process)
            except:
                pass
                
        return _simple_speaker_segments(audio_path, max_seconds)


def _simple_speaker_segments(audio_path: Path, max_seconds: Optional[float] = None) -> List[Dict]:
    """
    Simple fallback speaker segmentation based on silence.
    Returns a single speaker for the entire audio.
    """
    import soundfile as sf
    
    # Get audio duration
    info = sf.info(str(audio_path))
    full_duration = info.duration
    
    # Use max_seconds if specified
    duration = min(max_seconds, full_duration) if max_seconds else full_duration
    
    # Return single speaker segment
    return [{
        "start": 0.0,
        "end": duration,
        "speaker": "Speaker 1",
        "duration": duration
    }]


def merge_speaker_segments(segments: List[Dict], min_gap: float = 0.5) -> List[Dict]:
    """
    Merge consecutive segments from the same speaker.
    
    Args:
        segments: List of speaker segments
        min_gap: Minimum gap (in seconds) to keep segments separate
        
    Returns:
        Merged segments
    """
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for segment in segments[1:]:
        # If same speaker and gap is small, merge
        if (segment["speaker"] == current["speaker"] and 
            segment["start"] - current["end"] < min_gap):
            current["end"] = segment["end"]
            current["duration"] = current["end"] - current["start"]
        else:
            merged.append(current)
            current = segment.copy()
    
    merged.append(current)
    return merged