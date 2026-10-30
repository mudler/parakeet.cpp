# C API (`libparakeet.so`)

The README has the overview and the ABI note. This page has the longer examples.
The current ABI version is 10 (`parakeet_capi_abi_version()`); all additions since v5 are additive.
The full symbol list and the exact signatures are in `include/parakeet_capi.h`.

`include/parakeet_capi.h` defines a flat, exception-free C-API meant for `dlopen` / FFI / LocalAI integration. Build the shared library with `-DPARAKEET_SHARED=ON`:

```c
#include "parakeet_capi.h"

parakeet_ctx *ctx = parakeet_capi_load("model.gguf");  // load ONCE
if (!ctx) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); return 1; }

char *text = parakeet_capi_transcribe_path(ctx, "audio.wav", 0 /*default*/);
if (text) { printf("%s\n", text); parakeet_capi_free_string(text); }

parakeet_capi_free(ctx);
```

In-memory PCM:
```c
char *text = parakeet_capi_transcribe_pcm(ctx, samples, n_samples,
                                          sample_rate, 0 /*default*/);
```

Timestamps and confidence as JSON (matches NeMo `timestamps=True` + `max_prob`):
```c
char *json = parakeet_capi_transcribe_path_json(ctx, "audio.wav", 0 /*default*/);
// {"text":"...",
//  "frame_sec":0.080000,
//  "words":[{"w":"Well,","start":0.480,"end":0.640,"conf":0.7859}, ...],
//  "tokens":[{"id":639,"t":0.480,"conf":0.9969}, ...]}
if (json) { printf("%s\n", json); parakeet_capi_free_string(json); }
```
`start`/`end`/`t` are in seconds; `conf` is the rescaled softmax probability of the emitted token in `(0,1]` (a word's `conf` is the `min` over its tokens). `frame_sec` is the encoder frame stride in seconds (`hop x subsampling / sample_rate`); multiply a frame-unit segment gap threshold (NeMo's `segment_gap_threshold`) by it to get the seconds gap between words when forming segments.

## Streaming (cache-aware EOU model)

For `parakeet_realtime_eou_120m-v1`, a streaming session decodes 16 kHz mono f32 PCM as it arrives, returning newly-finalized text and signalling EOU/EOB events:

```c
parakeet_stream *s = parakeet_capi_stream_begin(ctx);
int eou = 0;
char *t = parakeet_capi_stream_feed(s, pcm, n_samples, &eou); // "" if none yet
if (t) { printf("%s", t); parakeet_capi_free_string(t); }
if (eou) printf(" [EOU]");
// ...feed more chunks...
char *tail = parakeet_capi_stream_finalize(s);                // flush the tail
if (tail) { printf("%s\n", tail); parakeet_capi_free_string(tail); }
parakeet_capi_stream_free(s);
```

`<EOU>` (end-of-utterance) and `<EOB>` (backchannel) are stripped from the text and surfaced via `*eou_out` (the CLI `--stream` prints them as `[EOU @ <t>s]` markers). Other special pieces, such as the language tags a prompt-conditioned model emits, are stripped from the text and the words. The streaming transcript matches NeMo's cache-aware streaming exactly, and `finalize` flushes the end-of-stream tail without fabricating an `<EOU>` that NeMo would not emit.

## Buffered streaming (unified checkpoints, ABI v11)

A unified checkpoint (`nvidia/nemotron-labs-unified-asr-0.6b`) streams by re-encoding a window of audio at each step: `[left | chunk | right]` in 80 ms frames. The text it commits is final and is never revised. A buffered stream raises no `<EOU>`/`<EOB>`, so `eou` and `eob` stay 0 and `parakeet_capi_stream_drain_events` returns no events.

The `parakeet_capi_stream_*` calls above work for these models too. `parakeet_capi_stream_begin` and `_begin_lang` use the default window (`parakeet_capi_stream_presets_json` lists them, with `"default":true`). Use `parakeet_capi_stream_begin_opts` to choose another preset, or a custom window, and the language:

```c
parakeet_stream_opts o;
parakeet_capi_stream_opts_default(&o);
o.target_lang = "de";                       // NULL or "" = the model default
o.preset = PARAKEET_STREAM_PRESET_CUSTOM;   // or an index from the presets JSON, or PARAKEET_STREAM_PRESET_DEFAULT
o.left_frames = 70; o.chunk_frames = 7; o.right_frames = 7;
parakeet_stream *s = parakeet_capi_stream_begin_opts(ctx, &o);
if (!s) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); }
```

`parakeet_capi_stream_kind(ctx)` returns `PARAKEET_STREAM_KIND_BUFFERED` for these models and `PARAKEET_STREAM_KIND_CACHE_AWARE` for the EOU and nemotron-3.5 models. `parakeet_capi_stream_info_json(s)` reports the window and the latency. `parakeet_capi_stream_reset(s)` starts a new utterance on the same stream. `parakeet_stream_opts.size` must be `sizeof(parakeet_stream_opts)`, and a cache-aware model accepts only the default preset.

The text does not depend on the feed size: the same audio fed in blocks of 1, 160, 1280 or 7777 samples gives the same tokens (see `test_capi_stream_unified` and `test_unified_stream`, which need the checkpoint). The window per step is set by the preset: latency is `(chunk + right) x 0.08 s`. The default is 1.12 s (`70,7,7`). `left + chunk + right` must be at most 4000 frames (320 s), `chunk` at least 1 and the other two at least 0; a window outside these limits is refused with `NULL` and `parakeet_capi_last_error`.

**CPU speed.** On a laptop-class CPU only the default 1.12 s preset keeps up with real time. The shorter presets fall behind, because each step re-encodes the whole `left + chunk + right` window, so the cost per step does not shrink with the chunk. Measure the GPU before relying on a shorter preset there.

The LocalAI backend (in the LocalAI repo) dlopens `libparakeet.so` and uses these symbols directly: the offline `parakeet_capi_transcribe_*` / `parakeet_capi_transcribe_path_json` and the streaming `parakeet_capi_stream_*`. See `include/parakeet_capi.h` for the full API. The C++ streaming session (`pk::StreamingSession`) also exposes per-word timestamps and confidence as words finalize, via `drain_words()` alongside the EOU events, which the CLI `--stream --timestamps` path prints.
