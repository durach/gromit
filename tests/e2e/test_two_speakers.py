"""Two speaker e2e tests for each language."""

import pytest

from tests.e2e.validators import validate_phrases, validate_structure

# Expected phrases for each language (from both speakers, simplified for tiny model)
EXPECTED_PHRASES = {
    "en": ["speaker", "project"],
    "uk": ["перший", "проект"],  # Simpler words for tiny model
    "ru": ["проект", "план"],  # Simpler words for tiny model
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
        num_speakers=2,  # Hint for diarization
    )

    orchestrator = Orchestrator(config)
    return orchestrator.process()


@pytest.mark.e2e
class TestTwoSpeakersEnglish:
    def test_english_two_speakers(self, audio_fixtures, test_scripts):
        """Test English two speaker transcription and diarization."""
        audio_path = audio_fixtures["en_two_speakers"]
        transcript = _run_transcription(audio_path)

        # Validate phrases from both speakers
        phrase_result = validate_phrases(transcript, EXPECTED_PHRASES["en"])
        assert phrase_result.success, f"Missing phrases: {phrase_result.missing}"

        # Validate structure (two speakers)
        struct_result = validate_structure(transcript, expected_speaker_count=2)
        assert struct_result.success, struct_result.error


@pytest.mark.e2e
class TestTwoSpeakersUkrainian:
    def test_ukrainian_two_speakers(self, audio_fixtures, test_scripts):
        """Test Ukrainian two speaker transcription and diarization."""
        audio_path = audio_fixtures["uk_two_speakers"]
        transcript = _run_transcription(audio_path)

        # Validate phrases from both speakers
        phrase_result = validate_phrases(transcript, EXPECTED_PHRASES["uk"])
        assert phrase_result.success, f"Missing phrases: {phrase_result.missing}"

        # Validate structure (two speakers)
        struct_result = validate_structure(transcript, expected_speaker_count=2)
        assert struct_result.success, struct_result.error


@pytest.mark.e2e
class TestTwoSpeakersRussian:
    def test_russian_two_speakers(self, audio_fixtures, test_scripts):
        """Test Russian two speaker transcription and diarization."""
        audio_path = audio_fixtures["ru_two_speakers"]
        transcript = _run_transcription(audio_path)

        # Validate phrases from both speakers
        phrase_result = validate_phrases(transcript, EXPECTED_PHRASES["ru"])
        assert phrase_result.success, f"Missing phrases: {phrase_result.missing}"

        # Validate structure (two speakers)
        struct_result = validate_structure(transcript, expected_speaker_count=2)
        assert struct_result.success, struct_result.error
