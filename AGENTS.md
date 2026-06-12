# AGENTS.md

Durable reference for humans and agents maintaining parakeet.cpp.

## AI-assisted contributions

This project follows the Linux kernel project's
[guidelines for AI coding assistants](https://docs.kernel.org/process/coding-assistants.html)
(the same policy LocalAI uses). Key rules for commits:

- **No `Signed-off-by` from AI.** Only a human submitter may sign off on the
  Developer Certificate of Origin.
- **No `Co-Authored-By: <AI>` trailers.** The human contributor owns the change.
- **Use an `Assisted-by:` trailer** to attribute AI involvement. Format:
  `Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2]`
  (e.g. `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`).
- The human submitter is responsible for reviewing, testing, and understanding
  every line of generated code.

## What this project is

parakeet.cpp is a C++17/ggml inference port of NVIDIA NeMo Parakeet ASR.
It targets CPU (GPU backends are wired but not exercised in CI) and is designed
for parity with the NeMo reference: a Python converter turns a NeMo checkpoint
into a metadata-driven GGUF, and a C++ model loader + conformer inference engine
run the same computation natively, with no Python dependency at inference time.

The public surface ships as a flat C-API (`include/parakeet_capi.h` +
`libparakeet.so`) suitable for `dlopen`/FFI/LocalAI integration.

Current status: Phase 5 complete.  Supports all offline Parakeet families -
CTC, RNNT, TDT, and hybrid TDT-CTC (0.6B/1.1B/110M, EN + multilingual v3) -
validated at WER 0 vs NeMo on every published checkpoint.  Quantization
(F16/Q8_0/K-quants) validated at WER 0.  Cache-aware streaming + EOU decoding
(`parakeet_realtime_eou_120m-v1`) is implemented: `pk::StreamingEncoder`
(per-layer conv/attention caches) + `pk::StreamingSession` (carried RNN-T
state) + `<EOU>`/`<EOB>` timed events, exposed via `parakeet_capi_stream_*` and
`parakeet-cli transcribe --stream`.  The streaming transcript matches NeMo's
cache-aware streaming byte-for-byte.

## Performance invariants (do not regress)

These are measured wins. An agent "simplifying" them has caused real regressions
before, so do not change them without an A/B benchmark that proves parity.

- **Keep the persistent `ggml_gallocr`** in `src/backend.cpp`. Reusing one
  allocator across the many tiny per-utterance graphs (no per-call alloc/free) is
  the core throughput lever on CPU and GPU. Do NOT replace it with
  `ggml_backend_sched` on the fast path: sched re-plans the graph split on every
  call and regressed CUDA by 7-23% when it did. The scheduler is used ONLY as a
  per-graph fallback, when the active GPU backend lacks a kernel for some op
  (so the unsupported op can run on CPU); when every op is supported, the fast
  gallocr path runs. If you think gallocr can go, you are about to reintroduce
  that regression.
- **Zero-copy weights.** `clone_weight` returns loader tensors directly so the
  same device buffer is reused every utterance; do not copy weights per call.

## Repository layout

```
include/             public C/C++ headers
                       parakeet.h         , C++ API
                       parakeet_capi.h    , flat C-API for FFI / dlopen
src/                 libparakeet implementation
                       model.hpp/cpp      , load-once pk::Model
                       parakeet.cpp       , thin transcribe() wrapper
                       parakeet_capi.cpp  , flat C-API implementation
                       common.hpp/cpp     , logging helpers
                       audio_io.hpp/cpp   , dr_wav load + linear resample to 16k
                       model_loader.hpp/cpp, GGUF -> ParakeetConfig + name->tensor
                       mel.cpp            , log-mel frontend
                       encoder.cpp / conformer.cpp / relpos_attention.cpp
                       ctc_decoder.cpp    , CTC head + greedy decode
                       prediction.cpp     , stacked LSTM prediction net
                       joint.cpp          , joint network
                       tdt.cpp / rnnt.cpp , TDT / RNNT greedy loops
                       streaming_encoder.hpp/cpp, cache-aware streaming FastConformer encoder
                       streaming.hpp/cpp  , pk::StreamingSession (carried RNN-T + EOU events) + run_stream_over_pcm
examples/cli/        parakeet-cli binary
                       subcommands: info, transcribe (+ --stream), quantize
scripts/             Python tooling
                       convert_parakeet_to_gguf.py, .nemo/.hf -> GGUF (--dtype f32|f16|q8_0)
                       gen_nemo_baseline.py        , NeMo intermediates -> baseline.gguf
                       gen_stream_baseline.py      , NeMo cache-aware streaming encode+decode -> stream baseline.gguf
                       validate_vs_nemo.py         , WER parity gate vs NeMo
                       publish_hf.py               , convert+quantize -> HF upload (dry-run default)
                       requirements.txt            , nemo_toolkit[asr] + gguf
tests/               ctest targets
                       test_smoke.cpp          , version string (model-independent)
                       test_audio_io.cpp       , wav load + resample (model-independent)
                       test_fft.cpp            , FFT cross-check (model-independent)
                       test_model_loader.cpp   , config + tensor map (model-dependent)
                       test_capi.cpp           , C-API load -> transcribe -> free (model-dependent)
                       test_transcribe_speech.cpp, end-to-end CTC transcript (model-dependent)
                       test_transcribe_tdt.cpp , TDT transcript on speech fixture (model-dependent)
                       test_transcribe_0_6b.cpp, regression gate for 0.6B model (model-dependent)
                       test_transcribe_ctc.cpp , standalone CTC regression (model-dependent)
                       test_transcribe_rnnt.cpp, RNNT regression (model-dependent)
                       test_transcribe_eou.cpp , offline EOU model transcript + token ids (PARAKEET_TEST_GGUF_EOU)
                       test_streaming_encoder.cpp, cache-aware streaming encoder == offline + NeMo
                       test_streaming_decode.cpp , streaming RNN-T tokens == NeMo cache-aware streaming
                       test_streaming_eou_reset.cpp, multi-utterance streaming: decoder resets on <EOU>, transcript == NeMo reset-on-EOU (issue #13; PARAKEET_TEST_BASELINE_EOU_RESET)
                       test_capi_stream.cpp    , streaming C-API transcript == NeMo streaming (PARAKEET_TEST_BASELINE_EOU_STREAM)
                       python/check_convert.py , converter round-trip (model-dependent)
                       python/check_baseline.py, baseline dumper (model-dependent)
                       fixtures/clip.wav       , 2 s 16 kHz mono WAV for stage parity tests
                       fixtures/speech.wav     , LibriSpeech 2086-149220-0033, ~7.4 s
third_party/         vendored deps
                       ggml/     , submodule pinned at v0.13.0
                       dr_wav.h  , vendored single header
models/              output dir for converted GGUFs (gitignored;
                       MANIFEST.md tracks the expected published set)
docs/
  conversion.md     , GGUF schema reference
  quantization.md   , quantization allowlist, policy, measured size + WER per type
  parity.md         , full model coverage matrix + per-stage tensor parity
.github/workflows/
  ci.yml            , build job (per-push) + closed-loop job (pull_request + dispatch)
```

## Build

```
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
```

### CMake options

| Option                   | Default | Purpose                                    |
| ------------------------ | ------- | ------------------------------------------ |
| `PARAKEET_BUILD_TESTS`   | OFF     | Compile and register ctest targets         |
| `PARAKEET_BUILD_CLI`     | ON      | Build `parakeet-cli`                       |
| `PARAKEET_SHARED`        | OFF     | Build libparakeet as a shared library      |
| `PARAKEET_GGML_CUDA`     | OFF     | Forward GGML_CUDA to the submodule         |
| `PARAKEET_GGML_METAL`    | OFF     | Forward GGML_METAL to the submodule        |
| `PARAKEET_GGML_VULKAN`   | OFF     | Forward GGML_VULKAN to the submodule       |
| `PARAKEET_GGML_HIPBLAS`  | OFF     | Forward GGML_HIPBLAS to the submodule      |

Use `-DGGML_NATIVE=OFF` when building for CI or portable binaries.

## Running tests

### Model-independent (run anywhere, no checkpoint needed)

```
ctest --test-dir build --output-on-failure -LE model
```

Expected: `test_smoke`, `test_audio_io`, `test_fft` PASS.

### Model-dependent (need Python venv + cached checkpoint)

```
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf
export PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
export PARAKEET_TEST_BASELINE_SPEECH=/tmp/baseline_speech.gguf
ctest --test-dir build --output-on-failure
```

Tests return exit code 77 (ctest SKIP) when the venv or checkpoint is absent,
so they never break a CI environment that lacks them.

### Test labels

| Label   | Tests                                                        | Needs              |
| ------- | ------------------------------------------------------------ | ------------------ |
| (none)  | `test_smoke`, `test_audio_io`, `test_fft`                    | nothing            |
| `model` | `test_model_loader`, `test_capi`, `test_transcribe_*`, `check_*` | venv + checkpoint  |

## Converting a model

Set up the Python venv once:

```
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r scripts/requirements.txt   # nemo_toolkit[asr] + gguf
```

NeMo 2.7.3 is the validated version.  The anchor checkpoint is
`nvidia/parakeet-tdt_ctc-110m` (~440 MB, auto-downloaded by NeMo on first use).

Convert (HuggingFace id or local `.nemo`):

```
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --dtype q8_0 \
    --output models/parakeet-tdt_ctc-110m.gguf
```

Featurizer window and filterbank are lifted from the checkpoint at runtime;
mel/fft parameters do not need to be specified manually.

## Quantization policy

See `docs/quantization.md` for the full policy. Summary:

Only **linear `ggml_mul_mat`-consumed weights** are quantized:
- Encoder per-layer FFN + attention projections (`feed_forward*.linear*.weight`,
  `self_attn.linear_{q,k,v,out,pos}.weight`)
- Subsampling output projection (`encoder.pre_encode.out.weight`)
- Joint enc/pred projections (`joint.enc.weight`, `joint.pred.weight`)

Everything else stays F32: conv kernels, LSTM weights/biases, mel featurizer,
batch_norm stats, LayerNorm gain/bias, all `*.bias`, pos_bias, embeddings, the
joint output projection (`joint.joint_net.2.weight`, hand-rolled loop), and the
CTC head (stored `[1, V]`, block quantization impossible without transpose).

Supported `--dtype` values for the converter: `f32` (default), `f16`, `q8_0`.

For K-quants (`q4_k`, `q5_k`, `q6_k`), re-quantize an F32 GGUF with the CLI:

```
parakeet-cli quantize <in.gguf> <out.gguf> <type>
```

All variants of the 110m anchor hold WER 0.0 vs NeMo at F16, Q8_0, Q6_K, and
Q4_K. See `docs/quantization.md` for size figures.

## CLI

The binary is at `build/examples/cli/parakeet-cli`.

```
parakeet-cli info <model.gguf>
parakeet-cli transcribe --model <model.gguf> --input <audio.wav> [--decoder ctc|tdt] [--stream] [--timestamps] [--json]
parakeet-cli quantize <in.gguf> <out.gguf> <type>
```

`--timestamps` prints one `<start>-<end>  <word>  (<conf>)` line per word (also
works with `--stream`, where words print as they finalize); `--json` prints the
`parakeet_capi_transcribe_path_json` document (text + per-word/per-token
timestamps + confidence).

## C-API and LocalAI integration

`include/parakeet_capi.h` defines the flat C-API.  Build `libparakeet.so` with
`-DPARAKEET_SHARED=ON`.  Verify exports with `nm -D build-shared/libparakeet.so | grep parakeet_capi`.

The LocalAI backend lives in the LocalAI repo and dlopens `libparakeet.so`.
Symbols the LocalAI side depends on, do not remove or change any signature
without a coordinated bump on the LocalAI side:

```
parakeet_capi_abi_version
parakeet_capi_load
parakeet_capi_free
parakeet_capi_transcribe_path
parakeet_capi_transcribe_pcm
parakeet_capi_transcribe_path_json   # text + per-word/per-token timestamps + confidence as JSON
parakeet_capi_free_string
parakeet_capi_last_error
# streaming (cache-aware EOU model parakeet_realtime_eou_120m-v1):
parakeet_capi_stream_begin
parakeet_capi_stream_feed       # 16k mono f32 PCM -> newly-finalized text; *eou_out = event bitmask (ABI v5)
parakeet_capi_stream_finalize   # flush the end-of-stream tail
parakeet_capi_stream_free
```

`parakeet_capi_transcribe_path_json(ctx, wav, decoder)` returns malloc'd UTF-8
JSON `{"text":..,"words":[{"w","start","end","conf"}],"tokens":[{"id","t","conf"}]}`
(times in seconds, conf in `(0,1]`), built from
`pk::Model::transcribe_path_with_timestamps`.  Confidence is NeMo's `max_prob`
method, the rescaled softmax probability of the emitted (argmax) token over the
same logit slice NeMo log-softmaxes (`conf = (N·p_max − 1)/(N − 1)`, N = classes);
per-word `conf` is the `min` aggregate over the word's tokens.  Word offsets +
confidence match NeMo `transcribe(timestamps=True)` exactly (see `docs/parity.md`).

`parakeet_capi_abi_version` returns an integer that LocalAI can check for
compatibility; bump it on any breaking change to the above signatures or
semantics. Additive changes (new functions) are fine without bumping.

Streaming semantics: `parakeet_capi_stream_feed` buffers PCM, decodes encoder
chunks as audio arrives (carried encoder/decoder caches), and returns the
newly-finalized text (`<EOU>`/`<EOB>` STRIPPED, surfaced via `*eou_out`).
Since ABI v5 `*eou_out` is a bitmask — `PARAKEET_EVENT_EOU` (end of utterance:
respond) | `PARAKEET_EVENT_EOB` (backchannel: do not treat as a turn) — and the
streaming JSON documents carry separate `"eou"`/`"eob"` 0/1 flags (in v4 a
single conflated any-event flag). Per-event timestamps come from
`parakeet_capi_stream_drain_events`
(`parakeet_stream_event{token,is_eob,encoder_frame,time_sec}`, free with
`parakeet_capi_free_events`) or the `"events"` array
(`{"type":"eou"|"eob","frame","t"}`) in the `stream_feed_json` /
`stream_finalize_json` documents. The event queue is shared between the typed
drain and the JSON entry points — use one style per stream.
`parakeet_capi_stream_finalize` flushes the streaming tail and does NOT
fabricate an `<EOU>` NeMo's cache-aware streaming would not emit (for a final
chunk whose right context is incomplete, the trailing `<EOU>` is dropped exactly
as NeMo does).  Internally these wrap `pk::StreamingSession` (`src/streaming.*`):
`feed_mel_chunk` (token ids, used by `test_streaming_decode`), `take_new_text`,
`drain_events` (`pk::EouEvent{token,is_eob,encoder_frame,time_sec}`),
`drain_words` (`pk::Word{text,start,end,conf}` for words finalized since the last
drain, a word finalizes when the next `▁`-token arrives, the last word on
`finalize()`; reuses the offline `pk::group_words` grouping),
`last_chunk_had_eou`, `finalize`.  The CLI `--stream` path uses
`pk::run_stream_over_pcm` (full-clip mel + the model's chunk schedule); its
`on_chunk` callback now also receives the per-chunk finalized `pk::Word`s, which
`--stream --timestamps` prints.

## Dumping NeMo baselines

Used by Phase 1 parity tests.  Requires the venv and a 16 kHz mono WAV.

```
.venv/bin/python scripts/gen_nemo_baseline.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --audio tests/fixtures/clip.wav \
    --output /tmp/baseline.gguf
.venv/bin/python scripts/gen_nemo_baseline.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --audio tests/fixtures/speech.wav \
    --output /tmp/baseline_speech.gguf
```

## Publishing models to HuggingFace

`scripts/publish_hf.py` converts the anchor to the full variant set (F16,
Q8_0, Q4_K) and uploads each to a HF repo.  Dry-run by default, add `--upload`
to actually push.  Requires an HF token at `~/.cache/huggingface/token`
(`huggingface-cli login`).

```
.venv/bin/python scripts/publish_hf.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --repo mudler/parakeet.cpp-110m
# add --upload to actually push
```

See `models/MANIFEST.md` for the expected set of published GGUFs per checkpoint.

## CI workflow

`.github/workflows/ci.yml` has two jobs:

1. **build** (every push + pull_request): cmake build + `ctest -LE model`. Fast.
2. **closed-loop** (pull_request + `workflow_dispatch`): converts the 110m
   checkpoint and asserts `parakeet-cli transcribe --decoder tdt` matches the
   reference transcript below. Heavy (NeMo download, ~60 min); not on every push.

### Reference transcript

`tests/fixtures/speech.wav` on the 110m TDT head decodes (WER 0.0 vs NeMo) to
exactly the following. This is the closed-loop assertion and the quickest smoke
test that a build is correct on any backend (CPU, Metal, CUDA):

> Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.

## GGUF schema

See `docs/conversion.md` for the authoritative schema.  Quick summary:

- `general.architecture = "parakeet"`
- All metadata keys use the `parakeet.*` prefix.
- **Tensor names are verbatim NeMo `state_dict` keys**, no remapping, no
  prefix stripping.  This convention is load-bearing: the C++ model loader maps
  `name -> ggml_tensor*` by exact string.  Never remap tensor names at
  conversion time.

## ggml submodule

Pinned at v0.13.0 in `third_party/ggml`.  No local patches.  To bump:
1. Update the submodule SHA.
2. Run `ctest --test-dir build --output-on-failure`.
3. Fix any API breakage in `src/model_loader.cpp`.

## Common maintenance tasks

### Add support for a new Parakeet checkpoint

1. Convert + run `parakeet-cli info` to inspect the GGUF metadata.
2. Run `scripts/validate_vs_nemo.py` to get a WER figure.
3. If it passes (WER 0), add a row to `docs/parity.md` and `models/MANIFEST.md`.

The C++ loader is metadata-driven (arch, d_model, layers, mel params, vocab,
pred LSTM layers, xscaling, optional biases all read from GGUF KV); no source
changes are typically needed.

### Update to a newer NeMo version

1. Bump the venv and re-run the converter on the anchor checkpoint.
2. Regenerate baselines via `scripts/gen_nemo_baseline.py`.
3. Run the full test suite.  Any parity drift will surface in the `test_*`
   targets.

### Update to a newer ggml

1. Update the submodule SHA.
2. Run `ctest --output-on-failure`.
3. Fix any API breakage in `src/model_loader.cpp` (gguf/ggml C API).

### Add a new quantization type

1. Extend `examples/cli/main.cpp` `cmd_quantize` with the new type mapping.
2. Update the `should_quantize` heuristic in `scripts/convert_parakeet_to_gguf.py`
   if the new type has a different block size requirement.
3. Run `scripts/validate_vs_nemo.py` on the quantized GGUF and record WER + size
   in `docs/quantization.md`.
