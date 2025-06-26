# Gromit

Gromit – Local AI Tool for Transcribing Meetings

## Features

- Transcribes audio and video files (MP4, WAV, MP3, etc.) to text
- Identifies different speakers in the conversation
- Supports 99+ languages via OpenAI Whisper model
- Uses state-of-the-art Whisper model for accurate transcription
- Supports GPU acceleration on CUDA and Apple Silicon

## Prerequisites

- Python 3.9-3.11
- 10GB free disk space (for model download on first run)
- 8GB+ RAM recommended for large model
- UV package manager (install from https://docs.astral.sh/uv/getting-started/installation/)
- FFmpeg (for video/audio conversion - install with `brew install ffmpeg` on macOS)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gromit

# Install with UV
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your Hugging Face token
```

## Usage

Basic usage:
```bash
uv run gromit transcribe video.mp4
uv run gromit transcribe audio.wav
uv run gromit transcribe music.mp3
```

This will create `input_transcript.txt` with the transcribed conversation.

### Options

```bash
uv run gromit transcribe video.mp4 --output conversation.txt --language en --verbose
```

- `-o, --output`: Output file path (default: input_transcript.txt)
- `-l, --language`: Language code (default: en, supports 99+ languages)
- `--device`: Device to use (auto/cpu/cuda/mps, default: auto)
- `-v, --verbose`: Enable verbose logging
- `--max-seconds FLOAT`: Only transcribe first X seconds (for debugging/testing)

### Debug Mode

For quick testing with long audio files:

```bash
# Process only the first 30 seconds
uv run gromit transcribe video.mp4 --max-seconds 30 --verbose

# Process first 2 minutes for testing
uv run gromit transcribe audio.wav --max-seconds 120
```

## Configuration

### Environment Variables (.env file)

The app uses a `.env` file for configuration. Copy `.env.example` to `.env` and configure:

```bash
# Required for speaker diarization
HUGGING_FACE_HUB_TOKEN=your_token_here

# Optional settings
WHISPER_MODEL_SIZE=large-v3  # Model size (tiny, base, small, medium, large-v3)
DEFAULT_LANGUAGE=en          # Default language code
DEFAULT_DEVICE=auto          # Device preference (auto, cpu, cuda, mps)
```

### Getting a Hugging Face Token

For optimal speaker diarization:

1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with **read-only** permissions
4. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
5. Add the token to your `.env` file

**Privacy Note**: Your audio files are NEVER uploaded. The token is only used to download the model once. All processing happens locally on your machine.

Without a token, the tool will still work but use simplified speaker detection.

## Example Output

```
Speaker 1: Hello, how are you doing today?

Speaker 2: I'm doing well, thanks for asking. How about you?

Speaker 1: Pretty good, thanks. I wanted to discuss the project with you.

Speaker 2: Sure, that sounds great. What did you want to talk about?
```

## Hardware Requirements

- GPU acceleration supported (CUDA, Apple Silicon)
- Minimum 8GB RAM for optimal performance

## Performance

- First run will download the Whisper model (~1.5GB)
- Processing speed depends on audio length and hardware
- GPU acceleration significantly improves speed
- Apple Silicon Macs use optimized CPU processing

## License

MIT