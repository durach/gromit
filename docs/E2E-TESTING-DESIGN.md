# E2E Testing Architecture Design

**Date:** January 2026
**Status:** Approved

## Overview

End-to-end integration tests that exercise the real transcription pipeline using TTS-generated audio with known content. These are periodic validation tests (not CI gate) that verify the complete pipeline works correctly.

## Goals

- Validate transcription accuracy across English, Ukrainian, and Russian
- Verify speaker diarization correctly identifies multiple speakers
- Catch regressions in the full pipeline integration
- Provide confidence that the system works end-to-end

## Directory Structure

```
tests/
├── e2e/
│   ├── __init__.py
│   ├── conftest.py          # Fixtures: audio generation, temp dirs
│   ├── audio_generator.py   # TTS wrapper using edge-tts
│   ├── validators.py        # Phrase matching + structural checks
│   │
│   ├── test_single_speaker.py    # 3 tests (en, uk, ru)
│   └── test_two_speakers.py      # 3 tests (en, uk, ru)
│
└── e2e_fixtures/            # Generated audio cached here (gitignored)
```

## Audio Generator

The `audio_generator.py` module wraps edge-tts to create test audio files.

### Single Speaker Generation

```python
generate_single_speaker(
    text="Hello, this is a test of the transcription system.",
    language="en",
    voice="en-US-GuyNeural",
    output_path="fixtures/en_single.mp3"
)
```

### Two Speaker Generation

```python
generate_two_speakers(
    exchanges=[
        ("Hello, I'm the first speaker.", "en-US-GuyNeural"),
        ("Nice to meet you, I'm speaker two.", "en-US-JennyNeural"),
        ("Let's discuss the project timeline.", "en-US-GuyNeural"),
    ],
    silence_gap=1.0,  # seconds between speakers
    output_path="fixtures/en_two_speakers.mp3"
)
```

### Voice Mapping by Language

| Language | Voice 1 (male) | Voice 2 (female) |
|----------|----------------|------------------|
| English | en-US-GuyNeural | en-US-JennyNeural |
| Ukrainian | uk-UA-OstapNeural | uk-UA-PolinaNeural |
| Russian | ru-RU-DmitryNeural | ru-RU-SvetlanaNeural |

The 1-second silence gap between speakers helps pyannote detect speaker changes reliably.

## Validators

The `validators.py` module provides two types of checks.

### Phrase Validation

Verifies expected content appears in transcript:

```python
def validate_phrases(transcript: str, expected_phrases: list[str]) -> ValidationResult:
    """
    Check that each phrase appears in transcript (case-insensitive).
    Returns which phrases were found/missing.
    """
```

### Structural Validation

Verifies output format is correct:

```python
def validate_structure(
    transcript: str,
    expected_speaker_count: int,
) -> ValidationResult:
    """
    Checks:
    - Transcript is non-empty
    - Contains expected number of unique "Speaker N:" labels
    - Each speaker label is followed by text content
    - No malformed output (e.g., empty speaker blocks)
    """
```

### Usage in Tests

```python
def test_english_two_speakers():
    result = run_transcription("en_two_speakers.mp3")

    # Content check
    assert validate_phrases(result, ["first speaker", "speaker two", "project"])

    # Structure check
    assert validate_structure(result, expected_speaker_count=2)
```

Validators return detailed results (not just pass/fail) so test failures show exactly what was missing or malformed.

## Test Scenarios

### Single Speaker Tests (`test_single_speaker.py`)

| Test | Language | Script | Key Phrases |
|------|----------|--------|-------------|
| `test_english_single` | en | "Hello, this is a test of the transcription system. The weather today is sunny and warm." | "hello", "transcription", "weather", "sunny" |
| `test_ukrainian_single` | uk | "Привіт, це тест системи транскрипції. Сьогодні гарна сонячна погода." | "привіт", "тест", "погода", "сонячна" |
| `test_russian_single` | ru | "Здравствуйте, это тест системы транскрипции. Сегодня хорошая солнечная погода." | "здравствуйте", "тест", "погода", "солнечная" |

### Two Speaker Tests (`test_two_speakers.py`)

Each test has 3 exchanges (~20 seconds total), alternating voices:
- Exchange 1: Speaker A introduces themselves
- Exchange 2: Speaker B responds
- Exchange 3: Speaker A mentions a topic (project/weather/meeting)

Validates:
- Both speakers' key phrases present
- Exactly 2 speaker labels in output
- Proper grouping (consecutive same-speaker lines merged)

## Fixture Management & Caching

### Audio Caching Strategy

Generated audio is slow (TTS + network). Cache locally:

```
tests/e2e_fixtures/           # gitignored
├── en_single.mp3
├── en_two_speakers.mp3
├── uk_single.mp3
├── uk_two_speakers.mp3
├── ru_single.mp3
├── ru_two_speakers.mp3
└── .cache_version            # Invalidate when test scripts change
```

### Fixture Lifecycle

```python
@pytest.fixture(scope="session")
def audio_fixtures():
    """Generate all test audio once per test session."""
    cache_dir = Path("tests/e2e_fixtures")

    if cache_valid(cache_dir):
        return load_cached(cache_dir)

    # Generate fresh fixtures
    fixtures = generate_all_fixtures(cache_dir)
    write_cache_version(cache_dir)
    return fixtures
```

### Cache Invalidation

- Hash of the test script content stored in `.cache_version`
- If phrases or voices change, cache regenerates automatically
- Manual: delete `tests/e2e_fixtures/` to force regeneration

## Pytest Configuration

### Marker Registration

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end integration tests")
```

### Skip by Default

```python
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-e2e"):
        skip = pytest.mark.skip(reason="need --run-e2e to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip)

def pytest_addoption(parser):
    parser.addoption("--run-e2e", action="store_true", help="run e2e tests")
```

## Running Tests

```bash
# Unit tests only (fast, default)
pytest -v

# E2E tests only
pytest tests/e2e/ -v --run-e2e

# All tests including e2e
pytest -v --run-e2e --run-slow

# Single language for quick check
pytest tests/e2e/test_single_speaker.py::test_english_single -v --run-e2e
```

## Dependencies

Add to `pyproject.toml` dev dependencies:

```toml
[project.optional-dependencies]
dev = [
    # ... existing deps ...
    "edge-tts>=6.1.0",
]
```

## Summary

| Aspect | Decision |
|--------|----------|
| TTS Provider | edge-tts (free, multi-language, distinct voices) |
| Languages | English, Ukrainian, Russian |
| Scenarios | Single speaker + two speakers |
| Duration | 15-30 seconds per clip |
| Verification | Key phrase matching + structural validation |
| Caching | Local fixture cache with version invalidation |
| Running | `--run-e2e` flag, skipped by default |
