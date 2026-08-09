"""Audio loading and preprocessing."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from gromit.exceptions import AudioLoadError

# Whisper expects 16kHz mono audio
TARGET_SAMPLE_RATE = 16000

# Video container extensions that need ffmpeg extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv"}

# Audio formats that soundfile can't decode natively and need ffmpeg extraction
AUDIO_EXTRACT_EXTENSIONS = {".m4a"}


class AudioProcessor:
    """Load and preprocess audio files for transcription."""

    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE) -> None:
        """Initialize processor with target sample rate."""
        self.target_sr = target_sr

    def _needs_ffmpeg_extraction(self, path: Path) -> bool:
        """Check if file needs audio extraction via ffmpeg."""
        return path.suffix.lower() in VIDEO_EXTENSIONS | AUDIO_EXTRACT_EXTENSIONS

    def _extract_audio_from_video(self, video_path: Path, max_duration: float | None) -> Path:
        """Extract audio from video file using ffmpeg.

        Args:
            video_path: Path to video file
            max_duration: Maximum duration in seconds (None = full file)

        Returns:
            Path to temporary WAV file with extracted audio

        Raises:
            AudioLoadError: If ffmpeg extraction fails
        """
        # Reserve a temp path for the extracted audio. ffmpeg writes the file, so
        # the handle only claims the name and is closed straight away.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", str(self.target_sr),  # Target sample rate
            "-ac", "1",  # Mono
        ]

        if max_duration is not None:
            cmd.extend(["-t", str(max_duration)])

        cmd.extend([
            "-y",  # Overwrite output
            str(temp_path),
        ])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            temp_path.unlink(missing_ok=True)
            raise AudioLoadError(
                f"ffmpeg failed to extract audio from {video_path}: {e.stderr}"
            ) from e
        except FileNotFoundError:
            temp_path.unlink(missing_ok=True)
            raise AudioLoadError(
                "ffmpeg not found. Please install ffmpeg to process video files."
            )

        return temp_path

    def load(self, path: Path, max_duration: float | None = None) -> np.ndarray:
        """Load audio file and return resampled numpy array.

        Args:
            path: Path to audio file (mp3, wav, mp4, etc.)
            max_duration: Maximum duration in seconds to load (None = full file)

        Returns:
            Audio data as float32 numpy array at 16kHz mono

        Raises:
            AudioLoadError: If file cannot be loaded
        """
        if not path.exists():
            raise AudioLoadError(f"File not found: {path}")

        temp_path = None

        try:
            if self._needs_ffmpeg_extraction(path):
                # Extract audio from video using ffmpeg
                temp_path = self._extract_audio_from_video(path, max_duration)
                load_path = temp_path
                # Audio already at target sample rate and mono from ffmpeg
                audio, sr = sf.read(load_path, dtype="float32")
            else:
                # Load audio file directly with soundfile
                load_path = path
                audio, sr = sf.read(load_path, dtype="float32")

                # Resample if needed
                if sr != self.target_sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Apply duration limit if specified
                if max_duration is not None:
                    max_samples = int(max_duration * self.target_sr)
                    audio = audio[:max_samples]

            return audio.astype(np.float32)

        except AudioLoadError:
            raise
        except Exception as e:
            raise AudioLoadError(f"Failed to load audio from {path}: {e}") from e
        finally:
            # Clean up temp file
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def load_multiple(
        self, paths: list[Path], max_duration: float | None = None
    ) -> np.ndarray:
        """Load and concatenate multiple audio files.

        Args:
            paths: List of paths to audio/video files (in order)
            max_duration: Maximum total duration in seconds (None = full files)

        Returns:
            Concatenated audio data as float32 numpy array at 16kHz mono

        Raises:
            AudioLoadError: If any file cannot be loaded or list is empty
        """
        if not paths:
            raise AudioLoadError("No input files provided")

        if len(paths) == 1:
            return self.load(paths[0], max_duration=max_duration)

        # Validate all files exist first
        for path in paths:
            if not path.exists():
                raise AudioLoadError(f"File not found: {path}")

        # Concatenate using ffmpeg
        return self._concatenate_with_ffmpeg(paths, max_duration)

    def _concatenate_with_ffmpeg(
        self, paths: list[Path], max_duration: float | None
    ) -> np.ndarray:
        """Concatenate multiple files using ffmpeg.

        Args:
            paths: List of paths to audio/video files
            max_duration: Maximum total duration in seconds

        Returns:
            Concatenated audio as numpy array
        """
        # Reserve temp paths for the concat list and the ffmpeg output. Both
        # handles only claim a name; the files are written later (the list below,
        # the output by ffmpeg), so each is closed immediately.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_list_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Write concat list file (ffmpeg concat demuxer format). Inside the
            # try so a write failure still hits the finally that unlinks both.
            with concat_list_path.open("w", encoding="utf-8") as concat_list:
                for path in paths:
                    # Use absolute paths and escape single quotes
                    abs_path = str(path.absolute()).replace("'", "'\\''")
                    concat_list.write(f"file '{abs_path}'\n")

            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", str(self.target_sr),
                "-ac", "1",  # Mono
            ]

            if max_duration is not None:
                cmd.extend(["-t", str(max_duration)])

            cmd.extend(["-y", str(output_path)])

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Load the concatenated audio
            audio, _ = sf.read(output_path, dtype="float32")
            return audio.astype(np.float32)

        except subprocess.CalledProcessError as e:
            raise AudioLoadError(
                f"ffmpeg failed to concatenate audio files: {e.stderr}"
            ) from e
        except FileNotFoundError:
            raise AudioLoadError(
                "ffmpeg not found. Please install ffmpeg to process multiple files."
            )
        finally:
            concat_list_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def is_valid_audio(self, audio: np.ndarray, silence_threshold: float = 0.001) -> bool:
        """Check if audio has meaningful content (not silent).

        Args:
            audio: Audio data as numpy array
            silence_threshold: RMS threshold below which audio is considered silent

        Returns:
            True if audio has content above threshold
        """
        rms = np.sqrt(np.mean(audio**2))
        return rms > silence_threshold

    def get_file_boundaries(self, paths: list[Path]) -> list[tuple[str, float]]:
        """Get file boundaries as (filename, start_offset) tuples.

        Uses ffprobe to determine each file's duration and computes
        cumulative offsets for the concatenated timeline.

        Args:
            paths: List of input file paths

        Returns:
            List of (filename, start_offset) tuples
        """
        boundaries = []
        offset = 0.0
        for path in paths:
            boundaries.append((path.name, offset))
            offset += self._probe_duration(path)
        return boundaries

    def _probe_duration(self, path: Path) -> float:
        """Get audio duration in seconds using ffprobe.

        Args:
            path: Path to audio/video file

        Returns:
            Duration in seconds

        Raises:
            AudioLoadError: If ffprobe fails
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError) as e:
            raise AudioLoadError(f"Failed to probe duration of {path}: {e}") from e
