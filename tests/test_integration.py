"""Integration tests for full pipeline."""

import os
from pathlib import Path

import pytest

from gromit.config import Device, ModelSize, TranscriptionConfig
from gromit.orchestrator import Orchestrator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for pyannote models",
)
def test_full_pipeline_with_audio():
    """Full pipeline should process audio and return formatted transcript."""
    config = TranscriptionConfig(
        input_paths=[FIXTURES_DIR / "test_tone.wav"],
        model_size=ModelSize.TINY,
        device=Device.CPU,
        language="en",
        verbose=True,
    )

    orchestrator = Orchestrator(config)
    result = orchestrator.process()

    # Should return a string (may be empty for pure tone)
    assert isinstance(result, str)
