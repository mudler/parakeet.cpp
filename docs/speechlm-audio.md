# Speech LLM audio embeddings (Nemotron 3.5 Transcribe)

[nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
is a speech LLM: an audio side feeds a 52-layer Nemotron-H language model that writes the
transcript. parakeet.cpp runs the audio side only. The result is not a transcript. It is
one embedding row per 80 ms of audio, in the LLM embedding space, ready to be used as the
audio tokens of a prompt. The language model, the tokenizer and the decoding loop are not
part of this library.

## What runs

```
16 kHz PCM
  -> log-mel (128 bins, n_fft 512, win 400, hop 160, preemphasis 0.97)
  -> ASR encoder branch                     -> Sortformer speaker branch
     per-feature normalisation                  raw log-mel
     8x feature stacking + Linear               31-layer RoPE transformer
     32 pre-norm blocks, RoPE on half           speaker head -> 8 probabilities per 10 ms
     of each head, per-head QK LayerNorm,       mean over the 8 sub-frames, > 0.5
     exact GELU, run in windows of 500
     frames with 50 frames of context
  -> asr_norm(hidden) + diar_norm(activity) @ S      (S: fixed sinusoidal table [8, 1280])
  -> Linear 1280 -> 2688                              rows = ceil(floor(samples / 160) / 8)
```

The encoder is the `DiarizationEncoder` pre-norm transformer, with two GGUF switches that
default to the old behaviour: `parakeet.encoder.qk_norm` (a LayerNorm with weight and bias on
q and k, per head, before RoPE) and `parakeet.encoder.activation` (`gelu_erf` is the exact
GELU, `gelu` is ggml's tanh form). The speaker branch is the existing `StreamingDiarization`
in the offline mode of the checkpoint: chunks of 500 frames with 50 frames of context, a FIFO
of 40 frames, a speaker cache of 188 frames, cache update every 300 frames.

Two rules come from the reference implementation:

* The spectrogram is trimmed to `floor(samples / hop)` frames (the centred STFT makes one more),
  then zero-padded to a multiple of 8. A group of 8 frames with no valid frame does not exist,
  so an audio length that is a multiple of 1280 samples gives no extra token.
* Chunking is part of the model. Audio longer than 500 encoder frames (40 s) must be encoded
  in windows; one pass over the whole clip gives different values.

## Convert

The checkpoint is one 67 GB safetensors file. The converter reads only the `perception.*`
tensors (about 1.5 GB, one byte range at the end of the file), from a local file or by HTTP
range request, and needs only `numpy` and `gguf`:

```
python scripts/convert_nemotron_speechlm_audio.py --hf --output-dir out/ [--dtype f16]
python scripts/convert_nemotron_speechlm_audio.py --safetensors model.safetensors \
    --config config.json --output-dir out/
```

It writes `<name>-audio.gguf` (arch `speechlm_audio`: mel constants, ASR encoder, fusion norms,
adapter) and `<name>-diar.gguf` (arch `diarization`: the speaker encoder and head). `--dtype
f16` or `q8_0` stores the large linear weights in that type; norms, biases, QK norms and mel
constants stay F32. Both files can go into one bundle GGUF as components of kind
`speechlm_audio` and `diar` ([bundle.md](bundle.md)). The model is under the OpenMDW-1.1
licence: a distributed GGUF has to keep the licence text and the NVIDIA notice.

## Use

C++:

```cpp
auto m = pk::SpeechLmAudio::load("audio.gguf", "", "diar.gguf", "", &err);
std::vector<float> rows; int n_tokens = 0;
m->embed_pcm(pcm16k, rows, n_tokens);          // rows: [n_tokens, m->dim()] row-major
```

C (additive, ABI v10 is unchanged):

```c
parakeet_ctx* ctx = parakeet_capi_load_speechlm_audio("audio.gguf", NULL, "diar.gguf", NULL);
float* rows; int n, dim;
if (parakeet_capi_audio_embed_pcm(ctx, pcm, n_samples, 16000, &rows, &n, &dim) == 0) {
    /* n x dim floats */
    parakeet_capi_free_floats(rows);
}
parakeet_capi_free(ctx);
```

`parakeet_capi_audio_embed_path` takes a file. The context reports
`PARAKEET_MODEL_KIND_SPEECHLM_AUDIO`; the transcription calls refuse it.

## Parity

Reference: Hugging Face transformers 5.22.0, fp32, CPU (`Nemotron3_5TranscribeAudioModel` and the
adapter `nn.Linear`, built from the checkpoint tensors by
`scripts/gen_speechlm_audio_baseline.py`). Largest absolute difference per stage, F32 GGUFs:

| Clip | log-mel | ASR encoder (max 17.7) | speaker probs | fused | adapter output (max 3.3) |
|---|---|---|---|---|---|
| 7.4 s speech, 93 tokens | 7.9e-5 | 4.8e-5 | 1.5e-6 | 3.2e-5 | 8.9e-6 |
| 62 s, 776 tokens, two encoder windows, speaker cache compressed | 1.0e-3 | 1.3e-4 | 8.0e-6 (last frame 1.5e-3) | 5.4e-4 | 4.7e-5 |

The log-mel difference is largest where a bin is near the log floor. No speaker flips on the
0.5 threshold on either clip.

Known differences:

* **Last speaker frame.** When the valid mel length is a multiple of 8, the reference pads its
  spectrogram with one more group that is masked as a key, but the speaker head's convolution at
  the last valid frame still sees that group's output. This engine pads that edge with zeros. The
  last frame's speaker probability can differ (up to 2.2e-2 on a 2 s clip, 1.5e-3 on 62 s); the
  other frames match, and the activity bit did not change in any test. This was not checked
  against NeMo itself.
* **Ill-conditioned input.** The repository's 2 s `clip.wav` fixture gave an encoder difference
  of 1e-3 and a speaker difference of 1e-2, against 5e-5 and 1e-6 on speech. The cause was not
  investigated; real speech is the target.
* **Quantisation.** On the same clips F16 gives encoder differences of 2.1e-3 (7.4 s) and 1.1e-2
  (62 s), adapter 5.7e-4 and 1.1e-2; Q8_0 gives 7.8e-2 and 2.5e-1 (encoder) and 2.6e-2 and 2.1e-1
  (adapter). Mean differences are 6e-5 (F16) and 3e-3 (Q8_0). Whether that costs transcript quality
  can only be judged with the language model.
* The reference runs the audio side in fp32 (`_keep_in_fp32_modules_strict`); other deployments
  may use bf16, which will differ more.

## Tests

* `test_encoder_gates` (CPU only): each switch against a double-precision implementation.
* `check_speechlm_convert` (CPU only): the converter on a tiny synthetic safetensors.
* `test_capi_speechlm_audio` (argument checks run everywhere).
* `test_speechlm_audio`, and the model part of `test_capi_speechlm_audio`: set
  `PARAKEET_TEST_SPEECHLM_AUDIO_GGUF` and `PARAKEET_TEST_SPEECHLM_DIAR_GGUF`, and make the baseline
  with `scripts/gen_speechlm_audio_baseline.py --wav tests/fixtures/speechlm_audio_2s.wav
  --output models/speechlm_audio_baseline.gguf`. Both are labelled `model`.

## Not included

The language model, tokenizer, prompt template and decoding; the CTC word-timestamp head; the
multi-token-prediction block; a CLI command; speed and GPU measurements; publishing the GGUFs.
Audio longer than 62 s, and several speaker-cache compressions in one run, were not tested.
