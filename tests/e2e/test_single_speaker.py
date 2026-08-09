"""Single speaker e2e tests for each language."""

import pytest

from tests.e2e.validators import validate_phrases, validate_structure

# Expected phrases for each language (subset that should reliably transcribe with tiny model)
EXPECTED_PHRASES = {
    "en": ["hello", "test", "weather"],
    "uk": ["привіт", "тест", "погода"],  # Simpler words for tiny model
    "ru": ["тест", "погода"],  # Simpler words for tiny model
}


def _run_transcription(audio_path) -> str:
    """Run the gromit transcription pipeline on an audio file.

    Returns the transcript as a string.
    """
    from gromit.config import Device, ModelSize, TranscriptionConfig
    from gromit.orchestrator import Orchestrator

    config = TranscriptionConfig(
        input_paths=[audio_path],
        model_size=ModelSize.TINY,  # Use tiny for faster tests
        device=Device.CPU,
        language="auto",
    )

    orchestrator = Orchestrator(config)
    return orchestrator.process()


@pytest.mark.e2e
class TestSingleSpeakerEnglish:
    def test_english_single_speaker(self, audio_fixtures, test_scripts):
        """Test English single speaker transcription."""
        audio_path = audio_fixtures["en_single"]
        transcript = _run_transcription(audio_path)

        # Validate phrases
        phrase_result = validate_phrases(transcript, EXPECTED_PHRASES["en"])
        assert phrase_result.success, f"Missing phrases: {phrase_result.missing}"

        # Validate structure (single speaker)
        struct_result = validate_structure(transcript, expected_speaker_count=1)
        assert struct_result.success, struct_result.error


@pytest.mark.e2e
class TestSingleSpeakerUkrainian:
    def test_ukrainian_single_speaker(self, audio_fixtures, test_scripts):
        """Test Ukrainian single speaker transcription."""
        audio_path = audio_fixtures["uk_single"]
        transcript = _run_transcription(audio_path)

        # Validate phrases
        phrase_result = validate_phrases(transcript, EXPECTED_PHRASES["uk"])
        assert phrase_result.success, f"Missing phrases: {phrase_result.missing}"

        # Validate structure (single speaker)
        struct_result = validate_structure(transcript, expected_speaker_count=1)
        assert struct_result.success, struct_result.error


@pytest.mark.e2e
class TestSingleSpeakerRussian:
    def test_russian_single_speaker(self, audio_fixtures, test_scripts):
        """Test Russian single speaker transcription."""
        audio_path = audio_fixtures["ru_single"]
        transcript = _run_transcription(audio_path)

        # Validate phrases
        phrase_result = validate_phrases(transcript, EXPECTED_PHRASES["ru"])
        assert phrase_result.success, f"Missing phrases: {phrase_result.missing}"

        # Validate structure (single speaker)
        struct_result = validate_structure(transcript, expected_speaker_count=1)
        assert struct_result.success, struct_result.error
