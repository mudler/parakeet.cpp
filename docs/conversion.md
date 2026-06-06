# GGUF conversion schema

`scripts/convert_parakeet_to_gguf.py` turns a NeMo Parakeet checkpoint (HF id or
local `.nemo`) into a single **GGUF v3** file consumed by the C++ `ModelLoader`.

The design rule is **fully metadata-driven**: every configuration value lives in
GGUF KV, and **tensor names are kept verbatim from the NeMo `state_dict`** (no
renaming). This makes the C++ port a 1:1 mapping of the reference model — new
checkpoints/variants require no C++ changes, only new KV/tensors.

```
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --output model.gguf
```

The acceptance check is `tests/python/check_convert.py` (registered as the
`check_convert` ctest, label `model`; skips with exit 77 if `gguf`/NeMo or the
checkpoint are unavailable).

## GGUF header

- `general.architecture` = `"parakeet"` (the GGUFWriter arch; what the reader keys on).
- `general.name` = the `--model` argument (HF id or `.nemo` path).

## KV metadata (`parakeet.*`)

All keys below are emitted for the hybrid anchor `parakeet-tdt_ctc-110m`. CTC-only
checkpoints omit the `parakeet.decoder.*` / `parakeet.joint.*` / `parakeet.tdt.*`
keys; non-TDT transducers omit `parakeet.tdt.durations`. The
`parakeet.encoder.att_context_*` / `causal_*` / `conv_causal` and
`parakeet.streaming.*` keys are emitted **only for cache-aware streaming models**
(`att_context_style != "regular"`, e.g. `nvidia/parakeet_realtime_eou_120m-v1`);
offline checkpoints omit them entirely (so they keep converting byte-identically)
and the C++ loader falls back to offline-safe defaults (`att_context [-1,-1]`,
style `regular`, causal flags `false`, no streaming block).

The `parakeet.prompt.*` keys and `parakeet.encoder.use_bias` /
`parakeet.encoder.att_context_presets` are emitted **only for the
prompt-conditioned multilingual model** `nvidia/nemotron-3.5-asr-streaming-0.6b`
(`model_defaults.initialize_prompt_feature == true`). Every other checkpoint omits
them and the loader defaults `prompt.present=false` (the prompt stage is skipped)
and `use_bias=true`. nemotron stores `att_context_size` as a **list of presets**
(`[[56,3],[56,0],[56,6],[56,13]]`, the first is the default 320 ms preset) rather
than a single `[left,right]`, so the converter records all of them in
`att_context_presets` and uses the first pair for the scalar left/right keys.

| Key | GGUF type | Meaning | Source | 110m value |
| --- | --- | --- | --- | --- |
| `parakeet.arch` | STRING | One of `ctc` / `rnnt` / `tdt` / `hybrid_rnnt_ctc` / `hybrid_tdt_ctc` | arch detection (below) | `hybrid_tdt_ctc` |
| `parakeet.encoder.feat_in` | UINT32 | Encoder input feature dim (= n_mels) | `cfg.encoder.feat_in` | 80 |
| `parakeet.encoder.d_model` | UINT32 | Encoder model dim | `cfg.encoder.d_model` | 512 |
| `parakeet.encoder.n_layers` | UINT32 | Conformer layer count | `cfg.encoder.n_layers` | 17 |
| `parakeet.encoder.n_heads` | UINT32 | Attention heads | `cfg.encoder.n_heads` | 8 |
| `parakeet.encoder.ff_dim` | UINT32 | FFN hidden dim = `d_model * ff_expansion_factor` | `cfg.encoder` | 2048 |
| `parakeet.encoder.conv_kernel` | UINT32 | Depthwise conv kernel size | `cfg.encoder.conv_kernel_size` | 9 |
| `parakeet.encoder.conv_norm_type` | STRING | Conv-module norm: `batch_norm` or `layer_norm` | `cfg.encoder.conv_norm_type` | `batch_norm` |
| `parakeet.encoder.subsampling_factor` | UINT32 | Time downsample factor | `cfg.encoder.subsampling_factor` | 8 |
| `parakeet.encoder.subsampling_conv_channels` | UINT32 | Subsampling conv channels | `cfg.encoder.subsampling_conv_channels` | 256 |
| `parakeet.encoder.xscaling` | BOOL | Scale embeddings by √d_model | `cfg.encoder.xscaling` | `false` |
| `parakeet.encoder.pos_emb_max_len` | UINT32 | Max relative-pos length | `cfg.encoder.pos_emb_max_len` | 5000 |
| `parakeet.encoder.att_context_left` | INT32 | Attention left context (`att_context_size[0]`); `-1` = unbounded. **Streaming only.** | `cfg.encoder.att_context_size` | (n/a) |
| `parakeet.encoder.att_context_right` | INT32 | Attention right context (`att_context_size[1]`); `-1` = unbounded. **Streaming only.** | `cfg.encoder.att_context_size` | (n/a) |
| `parakeet.encoder.att_context_style` | STRING | `regular` or `chunked_limited`. **Streaming only** (offline loader defaults `regular`). | `cfg.encoder.att_context_style` | (n/a) |
| `parakeet.encoder.causal_downsampling` | BOOL | Causal (left-pad) subsampling. **Streaming only** (default `false`). | `cfg.encoder.causal_downsampling` | (n/a) |
| `parakeet.encoder.conv_causal` | BOOL | Causal depthwise conv (`conv_context_size=="causal"`). **Streaming only** (default `false`). | `cfg.encoder.conv_context_size` | (n/a) |
| `parakeet.streaming.chunk_size` | ARRAY\<INT32\> | Per-step encoder chunk frames `[first, rest]`. **Streaming only.** | `encoder.streaming_cfg.chunk_size` | (n/a) |
| `parakeet.streaming.shift_size` | ARRAY\<INT32\> | Per-step shift frames `[first, rest]`. **Streaming only.** | `encoder.streaming_cfg.shift_size` | (n/a) |
| `parakeet.streaming.cache_drop_size` | INT32 | Steps dropped from the cache. **Streaming only.** | `encoder.streaming_cfg.cache_drop_size` | (n/a) |
| `parakeet.streaming.last_channel_cache_size` | INT32 | Attention left-context cache size. **Streaming only.** | `encoder.streaming_cfg.last_channel_cache_size` | (n/a) |
| `parakeet.streaming.valid_out_len` | INT32 | Valid encoder frames per step. **Streaming only.** | `encoder.streaming_cfg.valid_out_len` | (n/a) |
| `parakeet.streaming.pre_encode_cache_size` | ARRAY\<INT32\> | Pre-encode (mel) cache frames `[first, rest]`. **Streaming only.** | `encoder.streaming_cfg.pre_encode_cache_size` | (n/a) |
| `parakeet.streaming.drop_extra_pre_encoded` | INT32 | Steps dropped after pre-encode. **Streaming only.** | `encoder.streaming_cfg.drop_extra_pre_encoded` | (n/a) |
| `parakeet.encoder.use_bias` | BOOL | Whether the encoder linear layers carry a bias. `false` for nemotron (`use_bias=false`); the loader reads biases optionally and tolerates their absence. Defaults `true`. | `cfg.encoder.use_bias` | `true` |
| `parakeet.encoder.att_context_presets` | ARRAY\<INT32\> | Flattened `[l,r,l,r,...]` list of all `att_context_size` presets when the model stores a **list** of `[left,right]` pairs (multi-latency streaming, e.g. nemotron `[[56,3],[56,0],[56,6],[56,13]]`). The first pair is the default and is also written to `att_context_left`/`att_context_right`. **Streaming, multi-context only.** | `cfg.encoder.att_context_size` | (n/a) |
| `parakeet.prompt.present` | BOOL | Marks a prompt-conditioned multilingual model (nemotron). When `true` the C++ engine inserts the `prompt_kernel` (Linear, ReLU, Linear) on the encoder output, selected by a per-utterance language one-hot. Absent/`false` for every other model (which skip the stage entirely). | `model_defaults.initialize_prompt_feature` | (n/a) |
| `parakeet.prompt.num_prompts` | UINT32 | Width of the language one-hot appended to the encoder output (`prompt_kernel.0` input = `d_model + num_prompts`). **Prompt only.** | `model_defaults.num_prompts` | 128 |
| `parakeet.prompt.default_lang` | STRING | Locale used when no `--lang`/`target_lang` is given (nemotron: `auto`, prompt index 101). **Prompt only.** | derived (`auto` if present) | `auto` |
| `parakeet.prompt.dictionary.keys` | ARRAY\<STRING\> | Locale strings (e.g. `en`, `en-US`, `de`, `es`, `ja-JP`, `auto`) parallel to `dictionary.values`. The loader resolves a `target_lang` to its prompt index by lookup. **Prompt only.** | `model_defaults.prompt_dictionary` keys | len 121 |
| `parakeet.prompt.dictionary.values` | ARRAY\<INT32\> | Prompt index for each parallel key (multiple locales may share an index, e.g. `en` and `en-US` both map to 0). **Prompt only.** | `model_defaults.prompt_dictionary` values | (n/a) |
| `parakeet.preprocessor.sample_rate` | UINT32 | Audio sample rate | `featurizer.sample_rate` | 16000 |
| `parakeet.preprocessor.n_mels` | UINT32 | Mel filterbank count | `featurizer.nfilt` | 80 |
| `parakeet.preprocessor.n_fft` | UINT32 | FFT size | `featurizer.n_fft` | 512 |
| `parakeet.preprocessor.win_length` | UINT32 | STFT window length (samples) | `featurizer.win_length` | 400 |
| `parakeet.preprocessor.hop_length` | UINT32 | STFT hop length (samples) | `featurizer.hop_length` | 160 |
| `parakeet.preprocessor.preemph` | FLOAT32 | Pre-emphasis coefficient (0 = off) | `featurizer.preemph` | 0.97 |
| `parakeet.preprocessor.mag_power` | FLOAT32 | Spectrogram magnitude power | `featurizer.mag_power` | 2.0 |
| `parakeet.preprocessor.normalize` | STRING | Feature normalization mode | `featurizer.normalize` | `per_feature` |
| `parakeet.preprocessor.log_zero_guard` | FLOAT32 | Additive log zero-guard (default 2⁻²⁴) | `featurizer.log_zero_guard_value` | 5.96e-08 |
| `parakeet.vocab_size` | UINT32 | Tokenizer vocab size | `m.tokenizer.vocab_size` | 1024 |
| `parakeet.blank_id` | UINT32 | Blank token id (always == vocab_size) | derived | 1024 |
| `parakeet.tokenizer.pieces` | ARRAY\<STRING\> | SentencePiece pieces, index = token id (`▁` = leading space) | `tokenizer.ids_to_tokens([i])[0]` | len 1024 |
| `parakeet.decoder.pred_hidden` | UINT32 | RNNT prediction-net hidden size | `cfg.decoder.prednet.pred_hidden` | 640 |
| `parakeet.decoder.pred_rnn_layers` | UINT32 | RNNT prediction-net LSTM layers | `cfg.decoder.prednet.pred_rnn_layers` | 1 |
| `parakeet.joint.joint_hidden` | UINT32 | Joint-net hidden size | `cfg.joint.jointnet.joint_hidden` | 640 |
| `parakeet.joint.activation` | STRING | Joint-net activation | `cfg.joint.jointnet.activation` | `relu` |
| `parakeet.tdt.durations` | ARRAY\<INT32\> | TDT duration buckets | `cfg.decoding.durations` (fallback `model_defaults.tdt_durations`) | `[0,1,2,3,4]` |

**Effective preprocessor values:** restored configs frequently leave
`mag_power` / `preemph` / `log_zero_guard_value` as `None`, so NeMo falls back to
`FilterbankFeatures` defaults. The converter therefore reads the **instantiated**
`m.preprocessor.featurizer` attributes (effective runtime values), not the raw
cfg.

## Arch detection (`parakeet.arch`)

`detect_arch()` mirrors spec §2.1:

```
1. cfg.aux_ctc present:
     loss.loss_name == "tdt" OR decoding.durations non-empty  → hybrid_tdt_ctc
     else                                                      → hybrid_rnnt_ctc
2. else cfg.joint present:
     decoding.durations non-empty OR joint.num_extra_outputs>0 → tdt
     else                                                      → rnnt
3. else                                                        → ctc
```

For `parakeet-tdt_ctc-110m`: `aux_ctc` present + `loss.loss_name == "tdt"` →
`hybrid_tdt_ctc`.

## Tensors (verbatim NeMo names, F32)

Every tensor is written under its **exact `state_dict` key**, converted to F32
(no quantization in this converter; quantization is a separate CLI step). Rules:

- Skip everything under `preprocessor.*` **except** the two featurizer buffers
  (see below).
- Skip 0-dimensional scalar tensors (e.g. `…batch_norm.num_batches_tracked`).

State-dict prefixes present in the hybrid anchor (690 tensors total):

| Module | Prefix | Example |
| --- | --- | --- |
| Mel filterbank buffer | `preprocessor.featurizer.fb` | shape numpy `(1, 80, 257)` = `(1, n_mels, n_fft/2+1)` |
| Hann window buffer | `preprocessor.featurizer.window` | shape `(win_length,)` = `(400,)` |
| Subsampling front end | `encoder.pre_encode.*` | `encoder.pre_encode.out.weight` |
| Conformer layers | `encoder.layers.<i>.*` | `encoder.layers.0.norm_feed_forward1.weight`, `encoder.layers.0.self_attn.pos_bias_u` |
| CTC head (hybrid aux CTC) | `ctc_decoder.decoder_layers.0.*` | `ctc_decoder.decoder_layers.0.weight` shape `(vocab+1, d_model, 1)` |
| Prediction net (LSTM) | `decoder.prediction.*` | `decoder.prediction.embed.weight`, `decoder.prediction.dec_rnn.lstm.weight_ih_l0` |
| Joint net | `joint.{enc,pred,joint_net}.*` | `joint.joint_net.2.weight` shape `(vocab+1+D, joint_hidden)` |
| Prompt kernel (nemotron only) | `prompt_kernel.{0,2}.*` | `prompt_kernel.0.weight` `(2048, d_model+num_prompts)`, `prompt_kernel.0.bias` `(2048,)`, `prompt_kernel.2.weight` `(d_model, 2048)`, `prompt_kernel.2.bias` `(d_model,)` |

> The `prompt_kernel.*` projection weights are written verbatim by the generic
> tensor loop (no special handling); only the `parakeet.prompt.*` KV metadata is
> added by the converter. Like the LSTM prediction net and the featurizer buffers,
> the prompt kernel is **not** on the quantization allowlist, so it stays F32 in
> every quantized variant.

> Pure-CTC checkpoints (`EncDecCTCModelBPE`) put the CTC head under `decoder.*`
> instead of `ctc_decoder.*`; the verbatim rule preserves whatever the checkpoint
> uses.

### Featurizer buffers

The mel filterbank is a **buffer in the checkpoint** (`preprocessor.featurizer.fb`,
numpy shape `(1, n_mels, n_fft/2+1)`), so the converter lifts it directly rather
than recomputing librosa — exact mel parity for free. Note GGUF stores dimensions
reversed relative to numpy, so the GGUF tensor shape reads `[257, 80, 1]`. The
Hann `window` buffer (`preprocessor.featurizer.window`, shape `(win_length,)`) is
exported the same way.

## Worked example — `parakeet-tdt_ctc-110m`

```
wrote model.gguf: arch=hybrid_tdt_ctc vocab=1024 tensors=690
```

`EncDecHybridRNNTCTCBPEModel`. Encoder: feat_in 80, d_model 512, n_layers 17,
n_heads 8, ff_dim 2048, conv_kernel 9, conv_norm_type `batch_norm`, subsampling ÷8
(channels 256), `xscaling=false`. Preprocessor: 80 mels, n_fft 512, win 400,
hop 160, preemph 0.97, mag_power 2.0, per-feature norm. Transducer: pred_hidden
640 / 1 LSTM layer, joint_hidden 640 / relu. TDT durations `[0,1,2,3,4]`. Vocab
1024, blank id 1024, SentencePiece pieces stored in KV.

## Baseline intermediates — `baseline.gguf`

`scripts/gen_nemo_baseline.py` dumps NeMo's intermediate tensors for the fixed
fixture clip (`tests/fixtures/clip.wav`, 2 s 16 kHz mono) so each Phase 1 C++
stage can be diffed against ground truth. Determinism is enforced by forcing
`m.preprocessor.featurizer.dither = 0.0` and running under `torch.no_grad()` /
`eval()`. The CTC logits come from an **explicit** forward
(`m.preprocessor → m.encoder → m.ctc_decoder`), not `transcribe` — the hybrid's
default transcribe uses the TDT head and never runs the CTC head.

All tensors are squeezed (batch dim removed) and stored f32 except the int32
ids. **Axis order matters** — note that the per-layer / subsampling captures are
time-major `[T', d_model]` (NeMo's internal conformer orientation) whereas
`encoder_out` is feature-major `[d_model, T']` (the encoder transposes on the way
out). Shapes below are for the committed fixture on `parakeet-tdt_ctc-110m`.

| Tensor | Source module | Axis order | Example shape |
|---|---|---|---|
| `mel` | `m.preprocessor` (element 0) | `[n_mels, T]` | `[80, 201]` |
| `subsampling_out` | `m.encoder.pre_encode` (element 0) | `[T', d_model]` | `[26, 512]` |
| `enc_layer_0` | `m.encoder.layers[0]` | `[T', d_model]` | `[26, 512]` |
| `enc_layer_mid` | `m.encoder.layers[n//2]` | `[T', d_model]` | `[26, 512]` |
| `enc_layer_last` | `m.encoder.layers[n-1]` | `[T', d_model]` | `[26, 512]` |
| `encoder_out` | `m.encoder` (element 0, post-transpose) | `[d_model, T']` | `[512, 26]` |
| `ctc_logits` | `m.ctc_decoder` (log-softmax) | `[T', V+1]` | `[26, 1025]` |
| `ctc_argmax_ids` | argmax of `ctc_logits` over vocab axis | `[T']` int32 | `[26]` |

### Transducer-core intermediates (Phase 2)

Ground truth for the RNN-Transducer prediction net (`m.decoder`,
`RNNTDecoder`) and joint net (`m.joint`, `RNNTJoint` TDT joint), so the C++
`pk::PredictionNet` and `pk::Joint` can be diffed against NeMo. Dumped from the
same `eval()` / `torch.no_grad()` run as above.

| Tensor | Source | Axis order | Example shape |
|---|---|---|---|
| `pred_input_ids` | fixed label seq `[120, 7, 300, 42]` (all non-blank) | `[U]` int32 | `[4]` |
| `pred_out` | `m.decoder.predict(y, add_sos=True)` output `g` | `[U+1, pred_hidden]` | `[5, 640]` |
| `joint_out` | `m.joint.joint(enc_slice, pred_out)` **raw logits** | `[N, U+1, V+1+durations]` | `[4, 5, 1030]` |
| `joint_enc_frames` | N = leading encoder frames used for `joint_out` | `[1]` int32 | `[4]` |

**`add_sos` (prediction net).** `m.decoder.predict(y, state=None, add_sos=True,
batch_size=None)` returns `(g, hidden)`. `add_sos=True` (the `predict()` default,
and what NeMo's RNNT/TDT decoders use to prime the network) **prepends a zero
"start-of-sequence" embedding step**, so for `U=4` input ids the output `g` has
length `U+1 = 5`. The SOS step uses the zero embedding row
`decoder.prediction.embed.weight[1024]` (`padding_idx = blank = 1024`), which is
verified all-zero at dump time (a warning is printed if it ever is not). **The
C++ prediction net must also prepend the SOS step** to match `pred_out`'s length
and row 0.

**Raw logits vs log_softmax (joint net).** `m.joint.joint(f, g)` takes
`f = [B, T, enc_hidden]` and `g = [B, U, pred_hidden]` and returns
`[B, T, U, 1030]` where `1030 = 1024 vocab + 1 blank + 5 TDT durations`. On CPU
the joint's `log_softmax` config is `None`, so by default `joint.joint` would
apply `log_softmax` over **all 1030** entries. That is **not** what the TDT
greedy decoder consumes: it uses the **raw logits** and splits them into token
logits `[..., :1025]` (1024 vocab + blank) and duration logits `[..., 1025:]`
(the 5 durations `[0,1,2,3,4]`), applying **separate** log_softmaxes to each
split. The C++ joint emits plain (ReLU + linear) raw logits, so the dumper forces
`m.joint.log_softmax = False` and stores **raw logits** in `joint_out`. The
token/duration split point is `vocab+1 = 1025`.

**Encoder slice.** The real encoder output `enc` (`[B, d_model, T']`) is
transposed to `[B, T', d_model]` and sliced to the first `N = joint_enc_frames =
4` frames before being fed to the joint, keeping `joint_out` small. The C++ joint
test feeds `encoder_out[:, :N]` to match.
