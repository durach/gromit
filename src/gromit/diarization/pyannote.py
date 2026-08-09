"""Pyannote speaker diarization backend."""

import os

import numpy as np
import torch
from pyannote.audio import Pipeline

from gromit.diarization.base import BaseDiarizer, ProgressCallback, SpeakerSegment
from gromit.exceptions import DiarizationError


class PyannoteDiarizer(BaseDiarizer):
    """Speaker diarizer using pyannote.audio."""

    def __init__(self, device: str, num_speakers: int | None = None) -> None:
        """Initialize pyannote diarizer.

        Args:
            device: Compute device (cuda, mps, cpu)
            num_speakers: Expected number of speakers (None = auto-detect)

        Raises:
            DiarizationError: If HF token not configured or model unavailable
        """
        super().__init__(device, num_speakers)

        try:
            # Disable weights_only for pyannote models (PyTorch 2.6+ security change)
            # Pyannote models from HuggingFace are trusted and require this workaround
            original_weights_only = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=os.environ.get("HF_TOKEN"),
                )
            finally:
                # Restore original setting
                if original_weights_only is None:
                    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
                else:
                    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = original_weights_only

            # Move to appropriate device
            torch_device = torch.device(
                "cuda" if device == "cuda" and torch.cuda.is_available()
                else "mps" if device == "mps" and torch.backends.mps.is_available()
                else "cpu"
            )
            self.pipeline.to(torch_device)

        except Exception as e:
            raise DiarizationError(
                f"Failed to load pyannote model. Ensure HF_TOKEN is set and you've "
                f"accepted the model license at huggingface.co/pyannote/speaker-diarization-3.1. "
                f"Error: {e}"
            ) from e

    def diarize(
        self,
        audio: np.ndarray,
        progress_callback: ProgressCallback = None,
    ) -> list[SpeakerSegment]:
        """Identify speakers using pyannote.

        Args:
            audio: Audio data as float32 numpy array at 16kHz mono
            progress_callback: Optional callback for progress updates

        Returns:
            List of speaker segments with timing
        """
        # Pyannote expects dict with "waveform" and "sample_rate"
        audio_dict = {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": 16000,
        }

        # Run diarization
        diarization_kwargs = {}
        if self.num_speakers is not None:
            diarization_kwargs["num_speakers"] = self.num_speakers

        diarization = self.pipeline(audio_dict, **diarization_kwargs)

        # Convert to SpeakerSegment list
        result = []

        # Handle both old API (itertracks) and new API (speaker_diarization attribute)
        if hasattr(diarization, "itertracks"):
            # Old pyannote API
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                result.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                    )
                )
        elif hasattr(diarization, "speaker_diarization"):
            # New pyannote API (DiarizeOutput)
            for turn, speaker in diarization.speaker_diarization:
                result.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                    )
                )
        else:
            raise DiarizationError(
                f"Unknown diarization output format: {type(diarization)}"
            )

        # Report completion
        if progress_callback:
            progress_callback(1.0)

        return result
