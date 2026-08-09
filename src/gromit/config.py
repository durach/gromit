"""Configuration dataclasses for Gromit."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelSize(Enum):
    """Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"


class Device(Enum):
    """Compute device options."""

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class TranscriptionConfig:
    """Configuration for a transcription job."""

    input_paths: list[Path]
    output_path: Path | None = None
    language: str = "auto"
    model_size: ModelSize = ModelSize.LARGE_V3
    device: Device = Device.AUTO
    num_speakers: int | None = None
    verbose: bool = False
    max_duration: float | None = None
    from_file_path: Path | None = None
    glossary_paths: list[Path] = field(default_factory=list)

    @property
    def effective_output_path(self) -> Path:
        """Get output path, defaulting based on input paths.

        Transcribe outputs are namespaced `.gromit.txt` (paired with the
        `.gromit.json`), distinguishing them from other artifacts in a
        meeting folder such as `.named.txt` from nametag.
        """
        if self.output_path is not None:
            return self.output_path
        if self.from_file_path is not None:
            return self.from_file_path.with_name(
                f"{self.from_file_path.stem}.gromit.txt"
            )
        if len(self.input_paths) == 1:
            return self.input_paths[0].with_name(
                f"{self.input_paths[0].stem}.gromit.txt"
            )
        # Multiple files: use first file with _combined suffix
        first = self.input_paths[0]
        return first.with_name(f"{first.stem}_combined.gromit.txt")

    @property
    def json_output_path(self) -> Path:
        """Path for the structured .gromit.json, beside the txt output.

        Shares the txt's base name: an explicit `-o X.gromit.txt` and a plain
        `-o X.txt` both yield `X.gromit.json` (no double `.gromit` suffix).
        """
        txt = self.effective_output_path
        stem = txt.stem  # drops only the final ".txt"
        stem = stem.removesuffix(".gromit")
        return txt.with_name(f"{stem}.gromit.json")
