"""Tests for configuration module."""

from pathlib import Path

from gromit.config import Device, ModelSize, TranscriptionConfig


def test_config_creation_with_defaults():
    """Config should have sensible defaults."""
    config = TranscriptionConfig(input_paths=[Path("test.mp3")])
    assert config.language == "auto"
    assert config.model_size == ModelSize.LARGE_V3
    assert config.device == Device.AUTO


def test_config_output_path_default():
    """Output path should default to input path with .gromit.txt extension."""
    config = TranscriptionConfig(input_paths=[Path("/path/to/meeting.mp4")])
    assert config.effective_output_path == Path("/path/to/meeting.gromit.txt")


def test_config_output_path_explicit():
    """Explicit output path should be used when provided."""
    config = TranscriptionConfig(
        input_paths=[Path("meeting.mp4")],
        output_path=Path("transcript.txt"),
    )
    assert config.effective_output_path == Path("transcript.txt")


def test_model_size_enum_values():
    """ModelSize enum should have expected values."""
    assert ModelSize.TINY.value == "tiny"
    assert ModelSize.LARGE_V3.value == "large-v3"


def test_config_with_single_path():
    """Config should work with single path (backward compatible)."""
    config = TranscriptionConfig(input_paths=[Path("test.mp3")])
    assert config.input_paths == [Path("test.mp3")]


def test_config_with_multiple_paths():
    """Config should accept multiple input paths."""
    paths = [Path("part1.mp4"), Path("part2.mp4"), Path("part3.mp4")]
    config = TranscriptionConfig(input_paths=paths)
    assert config.input_paths == paths
    assert len(config.input_paths) == 3


def test_config_output_path_default_single_file():
    """Output path should default to input path with .gromit.txt for single file."""
    config = TranscriptionConfig(input_paths=[Path("/path/to/meeting.mp4")])
    assert config.effective_output_path == Path("/path/to/meeting.gromit.txt")


def test_config_output_path_default_multiple_files():
    """Output path should default to first file with _combined.gromit.txt suffix."""
    config = TranscriptionConfig(
        input_paths=[Path("/path/to/part1.mp4"), Path("/path/to/part2.mp4")]
    )
    assert config.effective_output_path == Path("/path/to/part1_combined.gromit.txt")


def test_config_output_path_from_file():
    """Output path from --from-file should derive from list filename."""
    config = TranscriptionConfig(
        input_paths=[Path("/path/to/part1.mp4")],
        from_file_path=Path("/lists/day1.txt"),
    )
    assert config.effective_output_path == Path("/lists/day1.gromit.txt")


def test_config_output_path_from_file_explicit_overrides():
    """Explicit -o should override --from-file derivation."""
    config = TranscriptionConfig(
        input_paths=[Path("/path/to/part1.mp4")],
        from_file_path=Path("/lists/day1.txt"),
        output_path=Path("/output/custom.txt"),
    )
    assert config.effective_output_path == Path("/output/custom.txt")


def test_json_output_path_beside_txt():
    from pathlib import Path

    from gromit.config import TranscriptionConfig

    cfg = TranscriptionConfig(input_paths=[Path("/x/Recording.mp4")])
    # effective_output_path -> /x/Recording.gromit.txt
    assert cfg.json_output_path == Path("/x/Recording.gromit.json")


def test_json_output_path_honors_explicit_output():
    from pathlib import Path

    from gromit.config import TranscriptionConfig

    cfg = TranscriptionConfig(
        input_paths=[Path("/x/a.mp4")], output_path=Path("/out/mtg.txt")
    )
    assert cfg.json_output_path == Path("/out/mtg.gromit.json")


def test_json_output_path_no_double_gromit_suffix():
    """An explicit -o already ending in .gromit.txt must not double-suffix."""
    from pathlib import Path

    from gromit.config import TranscriptionConfig

    cfg = TranscriptionConfig(
        input_paths=[Path("/x/a.mp4")],
        output_path=Path("/out/Recording.gromit.txt"),
    )
    assert cfg.json_output_path == Path("/out/Recording.gromit.json")


def test_glossary_paths_default_empty():
    from pathlib import Path

    from gromit.config import TranscriptionConfig

    cfg = TranscriptionConfig(input_paths=[Path("/x/a.mp4")])
    assert cfg.glossary_paths == []
