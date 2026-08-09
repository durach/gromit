"""Pipeline orchestration for transcription workflow."""

import time
import warnings

import torch
from rich.console import Console
from rich.progress import Progress

from gromit.alignment.temporal import TemporalAligner
from gromit.audio.processor import AudioProcessor
from gromit.config import TranscriptionConfig
from gromit.diarization.pyannote import PyannoteDiarizer
from gromit.exceptions import AudioLoadError
from gromit.glossary import load_glossaries
from gromit.output.formatter import TextFormatter
from gromit.output.json_writer import build_transcript_json
from gromit.transcription.faster_whisper import FasterWhisperTranscriber
from gromit.utils.device import resolve_device
from gromit.utils.progress import SpeedTracker, create_progress_columns


class Orchestrator:
    """Coordinate the transcription pipeline."""

    def __init__(self, config: TranscriptionConfig) -> None:
        """Initialize orchestrator with configuration.

        Args:
            config: Transcription configuration
        """
        self.config = config
        self.console = Console()
        self.device = resolve_device(config.device.value)
        self._aligned: list = []
        self._transcriber: FasterWhisperTranscriber | None = None
        self._hotwords: list[str] | None = None
        self._hotwords_from: list[str] = []

    def _build_hotwords(self) -> tuple[list[str] | None, list[str]]:
        """Build the ordered hotword terms + source paths from glossaries.

        Terms come back most-worth-biasing first; the transcriber trims them to
        whatever its prompt budget allows.
        """
        if not self.config.glossary_paths:
            return None, []
        glossary = load_glossaries(self.config.glossary_paths)
        words = glossary.hotword_list()
        return (words or None), [str(p) for p in self.config.glossary_paths]

    def transcript_json(self) -> dict:
        """Structured JSON payload for the last process() run."""
        language = (
            getattr(self._transcriber, "detected_language", None)
            or self.config.language
        )
        return build_transcript_json(
            self._aligned,
            language=language,
            model=self.config.model_size.value,
            hotwords_from=self._hotwords_from,
        )

    def process(self) -> str:
        """Run the full transcription pipeline.

        Returns:
            Formatted transcript as string

        Raises:
            AudioLoadError: If audio cannot be loaded
            TranscriptionError: If transcription fails
            DiarizationError: If diarization fails
        """
        # Suppress noisy warnings from dependencies
        warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
        warnings.filterwarnings("ignore", message=".*TensorFloat-32.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*", category=UserWarning)

        # Build hotwords from any glossaries before transcription.
        self._hotwords, self._hotwords_from = self._build_hotwords()

        # Step 1: Load audio (spinner only - fast step)
        with Progress(
            *create_progress_columns(with_bar=False),
            console=self.console,
            disable=not self.config.verbose,
        ) as progress:
            progress.add_task("Loading audio...", total=None)
            audio, file_boundaries = self._load_audio()

        audio_duration = len(audio) / 16000  # seconds

        # Step 2: Transcribe (with progress bar)
        transcript, transcribe_speed = self._transcribe_with_progress(audio, audio_duration)

        # Clear GPU memory before diarization
        self._clear_gpu_memory()

        # Step 3: Diarize (with progress bar, using transcribe speed as estimate)
        speakers = self._diarize_with_progress(audio, audio_duration, transcribe_speed)

        # Clear GPU memory after diarization
        self._clear_gpu_memory()

        # Step 4 & 5: Align and Format (spinner only - fast steps)
        with Progress(
            *create_progress_columns(with_bar=False),
            console=self.console,
            disable=not self.config.verbose,
        ) as progress:
            progress.add_task("Aligning...", total=None)
            aligned = self._align(transcript, speakers)
            self._aligned = aligned

            progress.add_task("Formatting...", total=None)
            result = self._format(aligned, file_boundaries)

        return result

    def _load_audio(self):
        """Load and validate audio file(s)."""
        processor = AudioProcessor()
        audio = processor.load_multiple(
            self.config.input_paths,
            max_duration=self.config.max_duration,
        )

        if not processor.is_valid_audio(audio):
            raise AudioLoadError("Audio appears to be silent")

        boundaries = processor.get_file_boundaries(self.config.input_paths)
        return audio, boundaries

    def _transcribe(self, audio):
        """Transcribe audio to text segments."""
        transcriber = FasterWhisperTranscriber(
            model_size=self.config.model_size.value,
            device=self.device,
            language=self.config.language,
        )
        self._transcriber = transcriber
        return transcriber.transcribe(audio, hotwords=self._hotwords)

    def _transcribe_with_progress(self, audio, audio_duration: float) -> tuple[list, float | None]:
        """Transcribe audio with progress bar display.

        Returns:
            Tuple of (transcript segments, speed_ratio or None)
        """
        tracker = SpeedTracker(total_duration=audio_duration)

        with Progress(
            *create_progress_columns(with_bar=True),
            console=self.console,
            disable=not self.config.verbose,
        ) as progress:
            task_id = progress.add_task("Transcribing...", total=100)

            def on_progress(prog: float, audio_pos: float) -> None:
                tracker.update(audio_pos)
                if tracker.has_estimate:
                    progress.update(task_id, completed=prog * 100)

            transcriber = FasterWhisperTranscriber(
                model_size=self.config.model_size.value,
                device=self.device,
                language=self.config.language,
            )
            self._transcriber = transcriber
            result = transcriber.transcribe(
                audio, progress_callback=on_progress, hotwords=self._hotwords
            )

        return result, tracker.speed_ratio

    def _diarize(self, audio):
        """Identify speakers in audio."""
        diarizer = PyannoteDiarizer(
            device=self.device,
            num_speakers=self.config.num_speakers,
        )
        return diarizer.diarize(audio)

    def _diarize_with_progress(self, audio, audio_duration: float, speed_hint: float | None) -> list:
        """Diarize audio with progress bar display.

        Args:
            audio: Audio data
            audio_duration: Total duration in seconds
            speed_hint: Speed ratio from transcription (for initial estimate)

        Returns:
            List of speaker segments
        """
        # Use transcription speed or conservative default
        estimated_speed = speed_hint if speed_hint else (3.0 if self.device == "cuda" else 1.0)
        estimated_time = audio_duration / estimated_speed

        with Progress(
            *create_progress_columns(with_bar=True),
            console=self.console,
            disable=not self.config.verbose,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                "Identifying speakers...",
                total=estimated_time,
            )

            start = time.time()

            diarizer = PyannoteDiarizer(
                device=self.device,
                num_speakers=self.config.num_speakers,
            )
            result = diarizer.diarize(audio)

            # Update to 100% when done
            elapsed = time.time() - start
            progress.update(task_id, completed=elapsed)

        return result

    def _align(self, transcript, speakers):
        """Align transcript with speaker segments."""
        aligner = TemporalAligner()
        return aligner.align(transcript, speakers)

    def _format(self, aligned, file_boundaries):
        """Format aligned segments as text."""
        formatter = TextFormatter()
        return formatter.format(aligned, file_boundaries=file_boundaries)

    def _clear_gpu_memory(self):
        """Release GPU memory between pipeline steps."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS doesn't have explicit cache clearing yet
            pass
