"""E2E test configuration and fixtures."""

import asyncio
import hashlib
from pathlib import Path

import pytest

from tests.e2e.audio_generator import VOICES, generate_single_speaker, generate_two_speakers

# Test scripts for each language
TEST_SCRIPTS = {
    "en": {
        "single": "Hello, this is a test of the transcription system. The weather today is sunny and warm.",
        "two_speaker": [
            ("Hello, I'm the first speaker talking to you today.", VOICES["en"]["voice1"]),
            ("Nice to meet you. I'm the second speaker in this conversation.", VOICES["en"]["voice2"]),
            ("Let's discuss the project timeline and next steps.", VOICES["en"]["voice1"]),
        ],
    },
    "uk": {
        "single": "Привіт, це тест системи транскрипції. Сьогодні гарна сонячна погода.",
        "two_speaker": [
            ("Привіт, я перший спікер і говорю з вами сьогодні.", VOICES["uk"]["voice1"]),
            ("Радий знайомству. Я другий спікер у цій розмові.", VOICES["uk"]["voice2"]),
            ("Давайте обговоримо план проекту та наступні кроки.", VOICES["uk"]["voice1"]),
        ],
    },
    "ru": {
        "single": "Здравствуйте, это тест системы транскрипции. Сегодня хорошая солнечная погода.",
        "two_speaker": [
            ("Здравствуйте, я первый спикер и говорю с вами сегодня.", VOICES["ru"]["voice1"]),
            ("Приятно познакомиться. Я второй спикер в этом разговоре.", VOICES["ru"]["voice2"]),
            ("Давайте обсудим план проекта и следующие шаги.", VOICES["ru"]["voice1"]),
        ],
    },
}

FIXTURES_DIR = Path(__file__).parent.parent / "e2e_fixtures"


def _get_cache_hash() -> str:
    """Generate hash of test scripts for cache invalidation."""
    content = str(TEST_SCRIPTS).encode()
    return hashlib.md5(content).hexdigest()[:12]


def _cache_valid() -> bool:
    """Check if cached fixtures are still valid."""
    version_file = FIXTURES_DIR / ".cache_version"
    if not version_file.exists():
        return False
    return version_file.read_text().strip() == _get_cache_hash()


def _write_cache_version() -> None:
    """Write cache version file."""
    version_file = FIXTURES_DIR / ".cache_version"
    version_file.write_text(_get_cache_hash())


async def _generate_all_fixtures() -> dict[str, Path]:
    """Generate all audio fixtures."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = {}

    for lang, scripts in TEST_SCRIPTS.items():
        # Single speaker
        single_path = FIXTURES_DIR / f"{lang}_single.mp3"
        if not single_path.exists():
            await generate_single_speaker(
                text=scripts["single"],
                voice=VOICES[lang]["voice1"],
                output_path=single_path,
            )
        fixtures[f"{lang}_single"] = single_path

        # Two speakers
        two_path = FIXTURES_DIR / f"{lang}_two_speakers.mp3"
        if not two_path.exists():
            await generate_two_speakers(
                exchanges=scripts["two_speaker"],
                silence_gap=1.0,
                output_path=two_path,
            )
        fixtures[f"{lang}_two_speakers"] = two_path

    _write_cache_version()
    return fixtures


@pytest.fixture(scope="session")
def audio_fixtures() -> dict[str, Path]:
    """Generate or load cached audio fixtures."""
    if _cache_valid():
        # Load from cache
        fixtures = {}
        for lang in TEST_SCRIPTS:
            fixtures[f"{lang}_single"] = FIXTURES_DIR / f"{lang}_single.mp3"
            fixtures[f"{lang}_two_speakers"] = FIXTURES_DIR / f"{lang}_two_speakers.mp3"
        return fixtures

    # Generate fresh
    return asyncio.run(_generate_all_fixtures())


@pytest.fixture(scope="session")
def test_scripts() -> dict:
    """Return test scripts for phrase validation."""
    return TEST_SCRIPTS
