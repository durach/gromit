"""Tests for e2e validators."""


from tests.e2e.validators import validate_phrases, validate_structure


class TestValidatePhrases:
    def test_all_phrases_found(self):
        transcript = "Hello world, this is a test of the system."
        result = validate_phrases(transcript, ["hello", "test", "system"])
        assert result.success
        assert result.found == ["hello", "test", "system"]
        assert result.missing == []

    def test_some_phrases_missing(self):
        transcript = "Hello world"
        result = validate_phrases(transcript, ["hello", "goodbye"])
        assert not result.success
        assert result.found == ["hello"]
        assert result.missing == ["goodbye"]

    def test_case_insensitive(self):
        transcript = "HELLO World"
        result = validate_phrases(transcript, ["hello", "world"])
        assert result.success

    def test_cyrillic_phrases(self):
        transcript = "Привіт, це тест системи"
        result = validate_phrases(transcript, ["привіт", "тест"])
        assert result.success


class TestValidateStructure:
    def test_valid_single_speaker(self):
        transcript = "Speaker 1:\nHello world, this is a test."
        result = validate_structure(transcript, expected_speaker_count=1)
        assert result.success

    def test_valid_two_speakers(self):
        transcript = """Speaker 1:
Hello, I'm speaker one.

Speaker 2:
Nice to meet you."""
        result = validate_structure(transcript, expected_speaker_count=2)
        assert result.success

    def test_wrong_speaker_count(self):
        transcript = "Speaker 1:\nOnly one speaker here."
        result = validate_structure(transcript, expected_speaker_count=2)
        assert not result.success
        assert "expected 2" in result.error.lower()

    def test_empty_transcript(self):
        result = validate_structure("", expected_speaker_count=1)
        assert not result.success
        assert "empty" in result.error.lower()

    def test_missing_speaker_label(self):
        transcript = "Just some text without speaker labels"
        result = validate_structure(transcript, expected_speaker_count=1)
        assert not result.success
