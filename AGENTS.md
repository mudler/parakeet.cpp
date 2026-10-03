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

- **The backend pool is opt-in.** With K = 1 (the default) every call uses the
  process-global backend and its mutex, exactly as before. A request on a pool
  borrows one backend for its whole duration through a thread-local route
  (`PoolLease`); code below `run_graph` does not know about the pool. Weights
  are realized once on the global backend and shared. Do not create weights per
  backend, and keep every mutable member of a shared object (decoder objects,
  loader) behind a lock or `std::once_flag`.

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
- **Ternary weights stay packed in the GGUF and are repacked once per loader; do not dequantize per call.**
  The packed Redux form is what makes it 6.8x smaller than F16 and what the
  CPU kernels in `src/ternary*.cpp` read.
- **Zero-copy weights.** `clone_weight` returns loader tensors directly so the
  same device buffer is reused every utterance; do not copy weights per call.

## Repository layout

```
include/             public C/C++ headers
                       parakeet.h         , C++ API
                       parakeet_capi.h    , flat C-API for FFI / dlopen
src/                 libparakeet implementation
                       backend.hpp/cpp    , pk::Backend: one CPU (or GPU) backend + persistent gallocr
                       backend_pool.hpp/cpp, pk::BackendPool + PoolLease: K CPU backends borrowed per request (thread-local routing in run_graph)
                       model.hpp/cpp      , load-once pk::Model (+ set_concurrency for the pool)
                       parakeet.cpp       , thin transcribe() wrapper
                       parakeet_capi.cpp  , flat C-API implementation
                       common.hpp/cpp     , logging helpers
                       audio_io.hpp/cpp   , dr_wav load + linear resample to 16k
                       model_loader.hpp/cpp, GGUF -> ParakeetConfig + name->tensor
                       bundle.hpp/cpp     , bundle GGUF header: components, licences, default component (docs/bundle.md)
                       mel.cpp            , log-mel frontend
                       encoder.cpp / conformer.cpp / relpos_attention.cpp
                       ctc_decoder.cpp    , CTC head + greedy decode
                       prediction.cpp     , stacked LSTM prediction net
                       exact_matvec.hpp/cpp, multi-column CPU matmul bitwise equal to the one-column ggml matmul (batched decode)
                       joint.cpp          , joint network
                       tdt.cpp / rnnt.cpp , TDT / RNNT greedy loops
                       streaming_encoder.hpp/cpp, cache-aware streaming FastConformer encoder
                       streaming.hpp/cpp  , pk::StreamingSession (carried RNN-T + EOU events) + run_stream_over_pcm
                       diarization.hpp/cpp, pk::DiarizationModel: offline speaker diarization (Sortformer)
                       diarization_encoder/head.*, RoPE Transformer encoder + speaker head
                       diarization_streaming.*, NeMo cache-aware streaming diarization (speaker cache + FIFO)
                       sas_merge.hpp/cpp  , ASR words x speaker segments -> speaker-attributed utterances
                       asr_committer.hpp/cpp, shared "finalize a word/utterance once" logic used by SAS and SceneStream
                       ced_tagger.hpp/cpp , pk::CedTagger: loads a CED GGUF (ced.cpp) into a tagger context, pk::SoundScorer
                       sound_stream.hpp/cpp, pk::SoundStream: sliding-window sound-event detection over live PCM
                       scene_stream.hpp/cpp, pk::SceneStream: combined ASR + diarization + sound-event stream
                       ternary.hpp/cpp    , packed ternary (moondream/parakeet-redux) linears: repack once, int8 activations, scalar ref + two-op ggml custom op
                       ternary_kernels.hpp, ternary_kernels_x86.cpp (AVX-512 VNNI / AVX2), ternary_kernels_neon.cpp
                       vad_head.hpp/cpp   , voice-activity head of Ultra/Redux (plain C++ loops), Model::vad_probabilities
                       vad_segmenter.hpp/cpp, pk::segment_by_vad: cut long audio at VAD pauses (SegmenterOpts), used by `transcribe --vad`
                       scene_render.hpp/cpp, pk::SceneRenderer + format_span/is_speech_label: `parakeet-cli scene` text rendering
                       speaker_registry.hpp/cpp, pk::SpeakerRegistry: enrolled voices (centroid per name), match, binary save/load
                       speaker_identifier.hpp/cpp, pk::SpeakerIdentifier: names diarization slots from their clean audio; identify_offline
                       speaker_encoder.hpp/cpp, pk::SpeakerEncoder: the only code that talks to voice-detect.cpp (voicedetect_capi.h)
examples/cli/        parakeet-cli binary
                       subcommands: info, transcribe (+ --stream), quantize, scene (ASR + diar + sound, one time-ordered feed)
                       sound-window-eval: measures CED short-window accuracy vs whole-clip top-1
                     diarize binary: diarize <diar.gguf> <wav> [--stream]
scripts/             Python tooling
                       convert_parakeet_to_gguf.py, .nemo/.hf -> GGUF (--dtype f32|f16|q8_0)
                       bundle_gguf.py              , merge GGUFs into one bundle; --list/--verify/--notice (docs/bundle.md)
                       convert_hf_parakeet_to_gguf.py, HF safetensors (moondream/parakeet-ultra, -redux) to GGUF (--template, --ternary keep|dequant, --vad keep|drop)
                       gen_nemo_baseline.py        , NeMo intermediates -> baseline.gguf
                       gen_stream_baseline.py      , NeMo cache-aware streaming encode+decode -> stream baseline.gguf
                       gen_diar_baseline.py        , NeMo offline + streaming diarization -> diar baseline.gguf
                       validate_vs_nemo.py         , WER parity gate vs NeMo
                       publish_hf.py               , convert+quantize -> HF upload (dry-run default)
                       requirements.txt            , nemo_toolkit[asr] + gguf
tests/               ctest targets
                       test_smoke.cpp          , version string (model-independent)
                       test_audio_io.cpp       , wav load + resample (model-independent)
                       test_fft.cpp            , FFT cross-check (model-independent)
                       test_backend_pool.cpp   , backend pool routing, limits, reentrancy, shutdown (model-independent)
                       test_capi_concurrency.cpp, set_concurrency + concurrent last_error writes (PARAKEET_TEST_GGUF)
                       test_pool_stress.cpp    , N threads x K backends == single-thread results; PARAKEET_STRESS_QUICK=1 for slow builds (PARAKEET_TEST_GGUF, _ULTRA, _REDUX_KEEP)
                       test_pool_shutdown.cpp  , pool create/destroy leaves no threads, results equal (PARAKEET_TEST_GGUF)
                       test_model_loader.cpp   , config + tensor map (model-dependent)
                       test_bundle.cpp         , bundle header, selection, partial read, Silero component, C-API (model-independent, synthetic files)
                       test_bundle_models.cpp  , real bundle == single-model files: transcript, VAD head, Silero (PARAKEET_TEST_BUNDLE, _ASR, _SILERO)
                       python/check_bundle.py  , bundle_gguf.py build/verify/refusals (model-independent)
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
                       test_diarization_accuracy.cpp, offline diarization == NeMo (PARAKEET_TEST_BASELINE_DIAR)
                       test_streaming_diarization.cpp, streaming diarization == NeMo streaming, every latency mode (same baseline)
                       test_combined_offline.cpp, SAS + streaming diarization/SAS through the C-API
                       test_sas_merge.cpp      , SAS merge/grouping (model-independent)
                       test_ternary.cpp        , ternary repack, int8 quant, every kernel == scalar (model-independent)
                       test_ternary_model.cpp  , packed Redux == dequantized Redux transcript (PARAKEET_TEST_GGUF_REDUX_KEEP + _DEQ)
                       test_model_loader_ternary.cpp, ternary + VAD flags from GGUF KVs
                       bench_ternary.cpp       , single-thread throughput of each ternary kernel (not a ctest)
                       test_vad_head.cpp       , VAD head probabilities (PARAKEET_TEST_GGUF_ULTRA)
                       test_vad_segmenter.cpp  , segmenter cut rules, 32 ms grid, Silero defaults, event tracker (model-independent)
                       test_vad_options.cpp    , VAD option parser, NULL and bad-file C-API paths (model-independent)
                       test_capi_vad_silero.cpp, Silero via the C-API: JSON, options, threads, stream (PARAKEET_TEST_SILERO_GGUF)
                       test_transcribe_vad_silero.cpp, transcribe with Silero segments on a model without a head (PARAKEET_TEST_SILERO_GGUF, PARAKEET_TEST_GGUF)
                       test_transcribe_vad.cpp , --vad path vs plain pass on long audio (PARAKEET_TEST_GGUF_ULTRA, PARAKEET_TEST_GGUF)
                       test_asr_committer.cpp  , shared word/utterance finalize logic (model-independent)
                       test_vad_batched.cpp    , grouped VAD decode == per-segment decode, bit for bit (PARAKEET_TEST_GGUF_ULTRA, _REDUX_KEEP, _REDUX_DEQ)
                       test_exact_batch.cpp    , batched decode == per-item decode, bit for bit, batch sizes 1 to 16 (PARAKEET_TEST_GGUF, _ULTRA, _REDUX_KEEP, _REDUX_DEQ)
                       bench_batch_decode.cpp  , timing of per-item vs batched decode and of the VAD path (not a ctest)
                       test_ced_parity.cpp     , CedTagger scores == ced.cpp PyTorch baseline (PARAKEET_TEST_CED_GGUF f32 + PARAKEET_TEST_CED_BASELINE)
                       test_sound_stream.cpp   , pk::SoundStream windowing/on-off-min_duration logic (model-independent)
                       test_sound_capi.cpp     , sound_stream_* C-API (PARAKEET_TEST_CED_GGUF)
                       test_scene_stream.cpp   , pk::SceneStream / scene_stream_* C-API, all three models together (PARAKEET_TEST_GGUF + PARAKEET_TEST_DIAR_GGUF + PARAKEET_TEST_CED_GGUF)
                       test_scene_render.cpp   , SceneRenderer / format_span / is_speech_label (model-independent)
                       test_speaker_registry.cpp, SpeakerRegistry enroll/match/serialize (model-independent)
                       test_speaker_identifier.cpp, SpeakerIdentifier with a fake embedder (model-independent)
                       test_speaker_encoder.cpp, SpeakerEncoder vs voice-detect reference embedding (PARAKEET_TEST_VD_GGUF + PARAKEET_TEST_VD_REF_WAV + PARAKEET_TEST_VD_REF_JSON)
                       test_speaker_identify.cpp, scene stream names both fixture voices (PARAKEET_TEST_DIAR_GGUF + PARAKEET_TEST_VD_GGUF; PARAKEET_TEST_GGUF adds the named-utterance block)
                       test_capi_speaker.cpp   , speaker C-API v9 (PARAKEET_TEST_DIAR_GGUF + PARAKEET_TEST_VD_GGUF; PARAKEET_TEST_GGUF optional; PARAKEET_TEST_VD_GGUF_ALT for the size-mismatch check)
                       python/check_convert.py , converter round-trip (model-dependent)
                       python/check_baseline.py, baseline dumper (model-dependent)
                       fixtures/clip.wav       , 2 s 16 kHz mono WAV for stage parity tests
                       fixtures/speech.wav     , LibriSpeech 2086-149220-0033, ~7.4 s
                       fixtures/two_speakers.wav, LibriSpeech 1272 + 2086 alternating A-B-A-B, 23.6 s
third_party/         vendored deps
                       ggml/     , submodule pinned at v0.13.0
                       ced.cpp/  , submodule, CED sound-event tagger (PARAKEET_WITH_CED, on by default);
                                   built as a static `ced` target linked into libparakeet, not a separate
                                   process; dr_wav is shared via CED_EXTERNAL_DR_WAV so there is one
                                   DR_WAV_IMPLEMENTATION in the whole build
                       voice-detect.cpp/, submodule, speaker encoders (PARAKEET_WITH_VOICEDETECT, on by default);
                                   static `voicedetect` target linked into libparakeet, dr_wav shared via
                                   VOICEDETECT_EXTERNAL_DR_WAV
                       dr_wav.h  , vendored single header
models/              output dir for converted GGUFs (gitignored;
                       MANIFEST.md tracks the expected published set)
docs/
  conversion.md     , GGUF schema reference
  bundle.md         , bundle GGUF format: several models in one file, per-component licences
  quantization.md   , quantization allowlist, policy, measured size + WER per type
  parity.md         , full model coverage matrix + per-stage tensor parity
  concurrency.md    , backend pool: concurrent requests, thread rules, measured throughput, TSan recipe
  ternary.md        , packed ternary Redux: GGUF form, kernels, limits, measured speed
  diarization.md    , speaker diarization + speaker-attributed ASR: parity, C-API, speed
  sound.md          , sound-event detection (CED) and the combined scene stream
  speaker.md        , speaker identification: enroll, scene naming, C-API v9 and v10, measured numbers
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
| `PARAKEET_WITH_CED`      | ON      | Sound-event detection through ced.cpp      |
| `PARAKEET_WITH_VOICEDETECT` | ON   | Speaker identification through voice-detect.cpp |

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

Ultra/Redux tests read `PARAKEET_TEST_GGUF_ULTRA` (F16 Ultra),
`PARAKEET_TEST_GGUF_REDUX_KEEP` (packed ternary) and `PARAKEET_TEST_GGUF_REDUX_DEQ`
(dequantized Redux); they skip (77) when unset.

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

## Ternary GGUF flags

Two optional GGUF flags, both read into `ParakeetConfig`:

- `parakeet.ternary.present` (with `parakeet.ternary.group_size` = 128): the
  encoder linears are stored as `<name>.qweight` (I8) + `<name>.scales` (F16).
  CPU only, offline only (no streaming). `PARAKEET_TERNARY_KERNEL=scalar|avx2|vnni|neon`
  forces a kernel. See `docs/ternary.md`.
- `parakeet.vad.present` (with `parakeet.vad.d_in/hidden/kernel/frame_sec`): the
  file carries `vad_head.*` tensors.

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
parakeet-cli bench --model <m.gguf> --manifest <file> [--threads N] [--concurrency K] [--json <out>]   # K workers over a pool of K backends
parakeet-cli transcribe --model <asr.gguf> --input <long.wav> --vad [--vad-model <silero.gguf>] [--vad-threshold F] [--vad-min-pause SEC] [--vad-min-speech SEC] [--vad-max-seg SEC]
parakeet-cli vad --model <ultra-or-redux-or-silero.gguf> --input <wav|-> [--mode speech|segments] [--probabilities]   # speech regions as JSON
parakeet-cli vad-probe --model <m.gguf> --input <wav|-> [--variant N]   # dump VAD head probabilities as t_sec,p
parakeet-cli scene [--model <asr.gguf>] [--diar <diar.gguf>] [--sound <ced.gguf>] --input <audio.wav> [--latency model|low|very_low|ultra_low] [--chunk-ms N] [--show-speech] [--json]
parakeet-cli scene ... --speakers <speaker.gguf> --registry <file> [--speaker-threshold F]   # names diarized speakers
parakeet-cli enroll --model <speaker.gguf> --name <name> --input <wav> [--input <wav> ...] --registry <file>
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

Concurrent requests (additive, ABI unchanged; not used by LocalAI yet). Default is one
backend, as before. See `docs/concurrency.md` for the rules (cores, `last_error`):

```
parakeet_capi_set_concurrency   # K CPU backends for one ASR ctx; concurrent transcribe_* calls then run in parallel
```

VAD segmentation (additive, ABI unchanged; not used by LocalAI yet). Needs a GGUF
with a VAD head (Ultra/Redux); see `docs/ternary.md`:

```
parakeet_capi_transcribe_path_json_vad   # same JSON as _json, long audio cut at VAD pauses
parakeet_capi_vad_pcm_json / _path_json  # speech regions as JSON; ctx is a VAD-head model or a Silero GGUF
parakeet_capi_transcribe_path_json_vad_with # cut with a Silero ctx, so any ASR model can segment
parakeet_capi_vad_stream_begin / _feed_json / _reset / _free  # streaming Silero: probabilities + speech events
```

A Silero VAD GGUF loads into a `parakeet_ctx` of kind `PARAKEET_MODEL_KIND_VAD`; see `docs/vad.md`.

Speaker diarization (ABI v7, additive; not used by LocalAI yet). A
diarization GGUF loads into its own `parakeet_ctx`; see `docs/diarization.md`:

```
parakeet_capi_diarize_path / _pcm              # offline, JSON segments
parakeet_capi_transcribe_and_diarize(_json)    # speaker-attributed ASR (two contexts)
parakeet_capi_free_sas_results                 # frees the array and every .text
parakeet_capi_diarize_stream_begin / _begin_latency / _feed / _active / _time / _free / _chunk_samples
parakeet_capi_free_diar_segments
parakeet_capi_sas_stream_begin / _begin_latency / _feed / _free
```

Sound-event detection (ABI v8, additive; not used by LocalAI yet). A CED GGUF
(ced.cpp) loads into its own `parakeet_ctx` kind (a "tagger") through the same
`parakeet_capi_load`; see `docs/sound.md`:

```
parakeet_capi_sound_opts_default
parakeet_capi_sound_stream_begin / _feed / _active / _drain_scores_json / _free
parakeet_capi_free_sound_segments
parakeet_capi_num_classes
parakeet_capi_class_label
parakeet_capi_model_kind        # which kind of ctx (NONE/ASR/DIARIZATION/SOUND/SPEAKER)
```

Speaker identification (ABI v9, additive; not used by LocalAI yet). A
voice-detect.cpp speaker GGUF loads into a fourth `parakeet_ctx`
kind (`PARAKEET_MODEL_KIND_SPEAKER`, 4) through the same `parakeet_capi_load`;
see `docs/speaker.md`:

```
parakeet_capi_speaker_dim
parakeet_capi_speaker_registry_new / _free / _size / _last_error
parakeet_capi_speaker_enroll
parakeet_capi_speaker_registry_save / _load
parakeet_capi_speaker_identify_pcm_json
parakeet_capi_scene_stream_begin_speaker
parakeet_capi_transcribe_and_diarize_named_json
```

```
# v10 (additive; not used by LocalAI yet)
parakeet_capi_speaker_registry_add_embedding   # add an embedding computed by the host
parakeet_capi_diarize_named_pcm_json           # diarization + "names", no ASR model
```

Combined scene stream (ABI v8, additive; not used by LocalAI yet). One stream
that carries any mix of an ASR context, a diarization context, and a tagger
context, and emits speaker-attributed words/utterances plus sound-event
segments in one time-ordered JSON document per feed; see `docs/sound.md`:

```
parakeet_capi_scene_opts_default
parakeet_capi_scene_stream_begin
parakeet_capi_scene_stream_feed_json
parakeet_capi_scene_stream_drain_scores_json
parakeet_capi_scene_stream_last_error
parakeet_capi_scene_stream_free
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

Diarization (needs NeMo main / >= 3.1: NeMo 3.0 cannot load the RoPE
encoder of nvidia/Nemotron-3-Diarization):

```
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/Nemotron-3-Diarization --output /tmp/diar.gguf
.venv/bin/python scripts/gen_diar_baseline.py \
    --model nvidia/Nemotron-3-Diarization \
    --audio tests/fixtures/two_speakers.wav --output /tmp/diar_baseline.gguf
PARAKEET_TEST_DIAR_GGUF=/tmp/diar.gguf PARAKEET_TEST_BASELINE_DIAR=/tmp/diar_baseline.gguf \
    ctest --test-dir build -R diar --output-on-failure
```

Quantized diarization GGUFs keep the same segments but move probabilities
more; set `PARAKEET_TEST_DIAR_PROB_TOL=0.15` for Q8_0. The baseline also
holds each low-latency streaming mode (`--modes`, NeMo is slow on them).

ASR:

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

# Optional TDT beam/N-best parity data:
.venv/bin/python scripts/gen_nemo_baseline.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --audio tests/fixtures/speech.wav \
    --tdt-beam-size 4 \
    --output /tmp/tdt_nbest_baseline.gguf
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

Pinned at v0.13.0 in `third_party/ggml`.  CMake applies the patches in
`third_party/ggml-patches` in-tree at configure time (`scripts/apply_ggml_patches.sh`),
so the submodule shows as modified.  To bump:
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
