"""Tests for audio generator."""

import asyncio

import pytest

from tests.e2e.audio_generator import (
    VOICES,
    generate_single_speaker,
    generate_two_speakers,
)


@pytest.mark.e2e
class TestAudioGenerator:
    def test_voices_defined_for_all_languages(self):
        """Voice mappings should exist for en, uk, ru."""
        assert "en" in VOICES
        assert "uk" in VOICES
        assert "ru" in VOICES
        for lang in ["en", "uk", "ru"]:
            assert "voice1" in VOICES[lang]
            assert "voice2" in VOICES[lang]

    def test_generate_single_speaker(self, tmp_path):
        """Should generate an MP3 file."""
        output_path = tmp_path / "test_single.mp3"

        asyncio.run(
            generate_single_speaker(
                text="Hello, this is a test.",
                voice=VOICES["en"]["voice1"],
                output_path=output_path,
            )
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_two_speakers(self, tmp_path):
        """Should generate an MP3 file with multiple speakers."""
        output_path = tmp_path / "test_two.mp3"

        asyncio.run(
            generate_two_speakers(
                exchanges=[
                    ("Hello, I'm speaker one.", VOICES["en"]["voice1"]),
                    ("Hi, I'm speaker two.", VOICES["en"]["voice2"]),
                ],
                silence_gap=0.5,
                output_path=output_path,
            )
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
