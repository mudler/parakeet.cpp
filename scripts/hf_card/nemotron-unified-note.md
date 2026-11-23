> One checkpoint for offline and buffered streaming transcription, in 25 European languages. Pass `--lang <locale>` (for example `en`, `de`), or leave it at the default `auto`. Offline it decodes with full context; `--stream` re-encodes a sliding window (default 1.12 s latency, other windows with `--stream-latency` or `--stream-context`). The WER column is agreement with NeMo, offline, on en and de (and `auto` for f16). In the `scene` command its word times come from the token emission frames and are corrected, see [sound.md](https://github.com/mudler/parakeet.cpp/blob/master/docs/sound.md#asr-models-in-the-scene). The small prompt layers, the LSTM, the norms and the biases stay F32 in every quantization.

```bash
huggingface-cli download mudler/parakeet-cpp-gguf nemotron-labs-unified-asr-0.6b-q8_0.gguf --local-dir models/
build/examples/cli/parakeet-cli transcribe --model models/nemotron-labs-unified-asr-0.6b-q8_0.gguf --input audio.wav --lang de
```
