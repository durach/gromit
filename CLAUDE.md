# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gromit is a local AI transcription tool that converts audio/video files to text with speaker diarization. It uses OpenAI's Whisper model for transcription and pyannote.audio for speaker identification.

## Key Commands

### Development Commands
- `uv sync` - Install dependencies and sync environment
- `uv run gromit transcribe <file>` - Run transcription on audio/video file
- `uv run gromit transcribe <file> --max-seconds 30 --verbose` - Debug mode with limited duration
- `uv run python -m pytest test_cli.py` - Run CLI tests
- `uv run python -m pytest test_token.py` - Run token validation tests

### Environment Setup
- Copy `.env.example` to `.env` and configure `HUGGING_FACE_HUB_TOKEN`
- Model downloads to `~/.cache/whisper/` (first run requires ~10GB space)

## Architecture

### Core Components
1. **CLI (`cli.py`)** - Click-based interface with Rich progress bars
2. **Transcriber (`transcriber.py`)** - Uses faster-whisper for audio-to-text
3. **Diarizer (`diarizer.py`)** - Uses pyannote.audio for speaker detection
4. **Formatter (`formatter.py`)** - Merges transcription with speaker segments
5. **Audio Utils (`audio_utils.py`)** - Handles format conversion and duration

### Processing Pipeline
1. Convert input to WAV format if needed
2. Run speaker diarization (pyannote.audio)
3. Run transcription (faster-whisper)
4. Format output by merging speaker segments with transcription

### Device Support
- **Diarization**: CPU, CUDA, MPS supported
- **Transcription**: CPU, CUDA only (faster-whisper limitation)
- Auto-detection with fallback to CPU for MPS requests on transcription

### Configuration
- Environment variables in `.env` file
- `WHISPER_MODEL_SIZE` (default: large-v3)
- `DEFAULT_LANGUAGE` (default: en)
- `DEFAULT_DEVICE` (default: auto)
- `HUGGING_FACE_HUB_TOKEN` (required for optimal diarization)

## Important Notes

### Dependencies
- Python 3.9-3.11 (uses pyproject.toml with UV package manager)
- FFmpeg required for video/audio conversion
- CUDA optional for GPU acceleration

### Error Handling
- Graceful fallback to simple speaker detection if HF token missing
- Temporary file cleanup for max_seconds processing
- Device fallback (MPS → CPU for transcription)

### Known Issues
- MPS not supported by faster-whisper (see TODO.md for investigation items)
- First run downloads large models (~1.5GB Whisper + diarization models)