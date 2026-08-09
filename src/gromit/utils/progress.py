"""Progress tracking utilities with time estimation."""

import time
from dataclasses import dataclass, field

from rich.progress import (
    BarColumn,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


@dataclass
class SpeedTracker:
    """Track processing speed and estimate remaining time.

    Measures actual processing speed by comparing audio position
    to wall-clock time, then estimates ETA for completion.
    """

    total_duration: float  # Total audio duration in seconds
    min_audio_for_estimate: float = 10.0  # Min audio seconds before estimating

    _audio_position: float = field(default=0.0, init=False)
    _elapsed_seconds: float = field(default=0.0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)

    def update(self, audio_position: float, elapsed_seconds: float | None = None) -> None:
        """Update tracker with current position.

        Args:
            audio_position: Current position in audio (seconds)
            elapsed_seconds: Wall-clock time elapsed (auto-calculated if None)
        """
        self._audio_position = audio_position
        if elapsed_seconds is not None:
            self._elapsed_seconds = elapsed_seconds
        else:
            self._elapsed_seconds = time.time() - self._start_time

    @property
    def progress(self) -> float:
        """Current progress as fraction (0.0 to 1.0)."""
        if self.total_duration <= 0:
            return 0.0
        return min(self._audio_position / self.total_duration, 1.0)

    @property
    def has_estimate(self) -> bool:
        """Whether we have enough data to estimate."""
        return self._audio_position >= self.min_audio_for_estimate

    @property
    def speed_ratio(self) -> float | None:
        """Processing speed as multiple of realtime (e.g., 5.0 = 5x realtime)."""
        if not self.has_estimate or self._elapsed_seconds <= 0:
            return None
        return self._audio_position / self._elapsed_seconds

    def reset(self) -> None:
        """Reset tracker for reuse."""
        self._audio_position = 0.0
        self._elapsed_seconds = 0.0
        self._start_time = time.time()


def create_progress_columns(with_bar: bool = True) -> list[ProgressColumn]:
    """Create Rich progress columns for display.

    Args:
        with_bar: If True, include progress bar and ETA. If False, spinner only.

    Returns:
        List of Rich ProgressColumn instances
    """
    if with_bar:
        return [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ]
    else:
        return [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]
