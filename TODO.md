# TODO

## High Priority

### MPS Support Investigation
- [ ] Research MPS support in faster-whisper/ctranslate2
- [ ] Investigate alternative Whisper implementations that support Apple Silicon MPS:
  - [ ] OpenAI's whisper (original) with PyTorch backend
  - [ ] whisper.cpp with Metal support
  - [ ] transformers library with MPS support
- [ ] Evaluate performance trade-offs between CPU fallback vs alternative implementations
- [ ] Consider implementing device-specific model loading (MPS for diarization, best available for transcription)

## Medium Priority

### Device Support Improvements
- [ ] Add device capability detection at startup
- [ ] Implement per-component device selection (diarization vs transcription)
- [ ] Add device performance benchmarking option

### User Experience
- [ ] Improve device selection help text with capability matrix
- [ ] Add --list-devices command to show supported devices per component

## Notes

### Current Device Support Status
- **Diarization (pyannote.audio)**: CPU, CUDA, MPS ✅
- **Transcription (faster-whisper)**: CPU, CUDA ❌ (no MPS)

### References
- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [ctranslate2 device support](https://opennmt.net/CTranslate2/python_api.html#devices)
- [PyTorch MPS support](https://pytorch.org/docs/stable/notes/mps.html)