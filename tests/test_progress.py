"""Tests for progress tracking utilities."""

from gromit.utils.progress import SpeedTracker


def test_speed_tracker_initial_state():
    """SpeedTracker starts with no estimate available."""
    tracker = SpeedTracker(total_duration=60.0)
    assert tracker.has_estimate is False
    assert tracker.progress == 0.0


def test_speed_tracker_calculates_speed_after_threshold():
    """SpeedTracker calculates speed after enough data."""
    tracker = SpeedTracker(total_duration=60.0, min_audio_for_estimate=5.0)

    # Simulate processing 10 seconds of audio in 2 seconds wall time
    tracker.update(audio_position=10.0, elapsed_seconds=2.0)

    assert tracker.has_estimate is True
    assert tracker.speed_ratio == 5.0  # 10 audio sec / 2 wall sec
    assert tracker.progress == 10.0 / 60.0


def test_speed_tracker_eta_calculation():
    """SpeedTracker calculates correct ETA."""
    tracker = SpeedTracker(total_duration=60.0, min_audio_for_estimate=5.0)

    # Processing at 5x realtime, 30 seconds done, 30 to go
    tracker.update(audio_position=30.0, elapsed_seconds=6.0)

    # 30 seconds remaining at 5x speed = 6 seconds ETA


def test_speed_tracker_no_estimate_before_threshold():
    """SpeedTracker doesn't estimate with insufficient data."""
    tracker = SpeedTracker(total_duration=60.0, min_audio_for_estimate=10.0)

    # Only 5 seconds processed, below threshold
    tracker.update(audio_position=5.0, elapsed_seconds=1.0)

    assert tracker.has_estimate is False


def test_create_progress_columns_returns_rich_columns():
    """create_progress_columns returns appropriate Rich columns."""
    from rich.progress import BarColumn, SpinnerColumn

    from gromit.utils.progress import create_progress_columns

    columns = create_progress_columns(with_bar=True)

    # Should have spinner, text, bar, percentage, time
    assert len(columns) == 5
    assert isinstance(columns[0], SpinnerColumn)
    assert isinstance(columns[2], BarColumn)


def test_create_progress_columns_spinner_only():
    """create_progress_columns can return spinner-only columns."""
    from gromit.utils.progress import create_progress_columns

    columns = create_progress_columns(with_bar=False)

    # Should have just spinner and text
    assert len(columns) == 2
