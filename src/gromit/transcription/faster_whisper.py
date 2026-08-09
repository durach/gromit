"""Faster-Whisper transcription backend."""

import warnings
from collections.abc import Sequence

import numpy as np
from faster_whisper import WhisperModel

from gromit.transcription.base import (
    BaseTranscriber,
    ProgressCallback,
    TranscriptSegment,
    Word,
)

# Whisper's decoder context is 448 tokens (`n_text_ctx`, the positional
# embedding table) — a hard architectural limit, prompt plus generated text.
# faster-whisper's get_prompt() fills it with:
#
#     sot_prev(1) + hotwords(<=223) + previous_tokens(<=223) + sot_sequence(3)
#
# It caps hotwords and previous_tokens at `max_length // 2 - 1 = 223` each,
# independently — but both live in the same prompt, so a glossary big enough to
# hit its own cap yields 450 tokens and CTranslate2 fails the whole run with
# "The maximum decoding length must be > 0", several minutes in, once the
# previous-text context has filled up.
#
# 221 is the largest hotword budget that keeps the total at or under 448.
HOTWORD_TOKEN_BUDGET = 448 - (448 // 2 - 1) - 3 - 1  # == 221


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using faster-whisper (CTranslate2 backend)."""

    def __init__(self, model_size: str, device: str, language: str) -> None:
        """Initialize faster-whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Compute device (cuda, mps, cpu)
            language: Language code (en, uk, ru, auto)
        """
        super().__init__(model_size, device, language)

        # Map device names to faster-whisper compute types
        compute_type = "float16" if device == "cuda" else "int8"

        # faster-whisper uses "cuda" or "cpu" (mps falls back to cpu)
        fw_device = "cuda" if device == "cuda" else "cpu"

        self.model = WhisperModel(
            model_size,
            device=fw_device,
            compute_type=compute_type,
        )
        self.detected_language: str | None = None

    def _count_tokens(self, text: str) -> int:
        """Token length of `text` as faster-whisper's own tokenizer sees it."""
        return len(
            self.model.hf_tokenizer.encode(text, add_special_tokens=False).ids
        )

    def _fit_hotwords(
        self, terms: Sequence[str], budget: int = HOTWORD_TOKEN_BUDGET
    ) -> tuple[str | None, list[str]]:
        """Keep the longest prefix of `terms` that fits the prompt budget.

        Truncating on term boundaries (rather than mid-token, as faster-whisper
        would) keeps every surviving hotword a real word.

        Returns:
            (space-joined kept terms or None, dropped terms)
        """
        kept: list[str] = []
        for i, term in enumerate(terms):
            candidate = " ".join([*kept, term])
            # faster-whisper encodes " " + hotwords.strip(); match that exactly.
            if self._count_tokens(" " + candidate) > budget:
                return (" ".join(kept) if kept else None), list(terms[i:])
            kept.append(term)
        return (" ".join(kept) if kept else None), []

    def transcribe(
        self,
        audio: np.ndarray,
        progress_callback: ProgressCallback = None,
        hotwords: Sequence[str] | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio using faster-whisper.

        Args:
            audio: Audio data as float32 numpy array at 16kHz mono
            progress_callback: Optional callback for progress updates
            hotwords: Optional hotword terms to bias decoding, most important
                first; trimmed to what the model's prompt budget allows

        Returns:
            List of transcribed segments with word-level timing and probability
        """
        # Determine language parameter
        lang = None if self.language == "auto" else self.language

        hotword_str, dropped = self._fit_hotwords(hotwords or [])
        if dropped:
            warnings.warn(
                f"glossary exceeds the {HOTWORD_TOKEN_BUDGET}-token hotword "
                f"budget: using {len(hotwords) - len(dropped)} of "
                f"{len(hotwords)} terms. Dropped: {', '.join(dropped)}",
                stacklevel=2,
            )

        segments_iter, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            hotwords=hotword_str,
        )
        self.detected_language = info.language

        result = []
        total_duration = len(audio) / 16000  # seconds

        for segment in segments_iter:
            words = [
                Word(w=w.word, start=w.start, end=w.end, p=w.probability)
                for w in (segment.words or [])
            ]
            avg_logprob = getattr(segment, "avg_logprob", 0.0)
            result.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=avg_logprob,
                    avg_logprob=avg_logprob,
                    words=words,
                )
            )

            # Report progress based on segment end time
            if progress_callback and total_duration > 0:
                progress = min(segment.end / total_duration, 1.0)
                progress_callback(progress, segment.end)

        return result
