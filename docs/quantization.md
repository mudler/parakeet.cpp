# parakeet.cpp — Quantization

This document records **which** model weights parakeet.cpp quantizes, **why** the
rest must stay F32, the supported converter `--dtype` values, and the measured
GGUF size + WER-vs-NeMo for each type.

The guiding rule: a weight may be stored in a reduced precision (`f16`/`q8_0`)
**only when the C++ engine feeds it directly into `ggml_mul_mat`** — ggml
dequantizes an `f16`/`q8_0` `src0` weight on the fly inside the compute graph.
Every weight the hand-rolled C++ reads as a raw `float*` (or reshapes/transposes
before the matmul in a way that does not survive block-quantized storage) **must
stay F32**, or the engine produces garbage.

---

## Supported `--dtype`

`scripts/convert_parakeet_to_gguf.py --dtype {f32,f16,q8_0}` (default `f32`).

* `f32` — every tensor F32 (unchanged from earlier phases; the default).
* `f16` — allowlisted linear weights stored as IEEE half; everything else F32.
* `q8_0` — allowlisted linear weights stored as ggml `Q8_0` (8-bit, 32-element
  blocks with one F16 scale per block); everything else F32.

A weight is quantized only if it is on the allowlist below **and** it is at least
2-D with both dims ≥ 32 **and** (for `q8_0`) its leading/contraction dim (ggml
`ne[0]`) is divisible by the 32-element block size. Otherwise it stays F32. See
`should_quantize()` in the converter.

## CLI K-quants (`parakeet-cli quantize`)

The Python `gguf` writer can emit `f16`/`q8_0`/`q4_0`/`q5_0` but **not** the
K-quants (`Q4_K`/`Q5_K`/`Q6_K`). For those, re-quantize an existing F32 GGUF
with the CLI:

```bash
parakeet-cli quantize <in.gguf> <out.gguf> <q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>
# e.g.
parakeet-cli quantize model_f32.gguf model_q4k.gguf q4_k
parakeet-cli quantize model_f32.gguf model_q6k.gguf q6_k
```

The type string is case-insensitive. The CLI reads the source GGUF (gguf C API
+ ggml context, same as `ModelLoader`), re-quantizes **exactly the same
allowlist** (the linear `mul_mat` `src0` weights below) via
`ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, /*imatrix*/ NULL)`
(none of these types require an importance matrix), copies every other tensor
verbatim in its stored type, and copies **all** KV metadata unchanged
(`gguf_init_empty` + `gguf_set_kv` + `gguf_add_tensor` + `gguf_write_to_file`).

A tensor is quantized only if it is on the allowlist, F32, 2-D with both dims
≥ 32, **and** its leading ggml dim (`ne[0]`, the contraction axis) is divisible
by the target type's block size. The `q*_0` blocks are 32 elements; the
**K-quants use a 256-element superblock**. Any allowlisted tensor whose row is
not divisible by the superblock is kept F32 (a warning is printed) — see the
K-quant note below for `joint.pred.weight`.

This section's allowlist and policy are identical to the converter's
`f16`/`q8_0` path documented below.

---

## Audit — quantizable vs must-stay-F32

Each loaded tensor was traced through its C++ consumer (`src/encoder.cpp`,
`conformer.cpp`, `relpos_attention.cpp`, `joint.cpp`, `ctc_decoder.cpp`,
`prediction.cpp`, `subsampling.cpp`, `mel.cpp`).

### Quantizable (fed directly to `ggml_mul_mat`, dequantized on the fly)

These are the only weights the converter quantizes. Each is cloned into the
compute context preserving its stored type (the per-component `clone_weight` /
`clone_w` helpers) and passed straight to `ggml_mul_mat` with **no** reshape or
transpose on the weight itself.

| Tensor (verbatim NeMo name) | ggml `ne[0]` (contraction) | Consumer |
| --- | --- | --- |
| `encoder.layers.N.feed_forward1.linear1.weight` | d_model | `conformer.cpp` FFN `linear()` → `ggml_mul_mat` |
| `encoder.layers.N.feed_forward1.linear2.weight` | ff_dim | same |
| `encoder.layers.N.feed_forward2.linear1.weight` | d_model | same |
| `encoder.layers.N.feed_forward2.linear2.weight` | ff_dim | same |
| `encoder.layers.N.self_attn.linear_q.weight` | d_model | `relpos_attention.cpp` `linear()` → `ggml_mul_mat` |
| `encoder.layers.N.self_attn.linear_k.weight` | d_model | same |
| `encoder.layers.N.self_attn.linear_v.weight` | d_model | same |
| `encoder.layers.N.self_attn.linear_out.weight` | d_model | same |
| `encoder.layers.N.self_attn.linear_pos.weight` | d_model | same |
| `encoder.pre_encode.out.weight` | C·F′ | `subsampling.cpp` `ggml_mul_mat(ow, flat)` (no reshape) |
| `joint.enc.weight` | enc_hidden | `joint.cpp` `ggml_mul_mat(W, x)` |
| `joint.pred.weight` | pred_hidden | `joint.cpp` `ggml_mul_mat(W, x)` |

Allowlist patterns (see `_QUANTIZABLE_PATTERNS` in the converter):

```
^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$
^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$
^encoder\.pre_encode\.out\.weight$
^joint\.enc\.weight$
^joint\.pred\.weight$
```

For the 110m anchor this is **156 tensors**; for `parakeet-tdt-0.6b-v2`,
**219 tensors**.

### Must stay F32 (and why)

| Tensor(s) | Why it cannot be quantized |
| --- | --- |
| `preprocessor.featurizer.fb`, `preprocessor.featurizer.window` | Read as raw `float*` in `mel.cpp` (mel filterbank / Hann window); never touch a ggml graph. |
| `decoder.prediction.embed.weight`, `decoder.prediction.dec_rnn.lstm.weight_ih_lN`, `weight_hh_lN`, `bias_ih_lN`, `bias_hh_lN` | The prediction net is a **hand-rolled LSTM** in `prediction.cpp`; all weights/embeddings are dereferenced as raw `float*`. |
| `joint.joint_net.2.weight`, `joint.joint_net.2.bias` | The joint **output** projection is computed with a hand-written `T·U` C++ loop over a raw `float*` (`joint.cpp`), not `ggml_mul_mat`. (The enc/pred projections that *are* `ggml_mul_mat`'d **are** quantized.) |
| `ctc_decoder.decoder_layers.0.weight` / `decoder.decoder_layers.0.weight` | The CTC head is stored `[1, d_model, V]` and squeezed in-graph via `ggml_reshape_2d` (`ctc_decoder.cpp`). Its leading ggml dim is 1, so block-quantization is impossible without a convert-time transpose; kept F32 (safest). |
| `encoder.pre_encode.conv.{0,2,3,5,6}.weight`/`.bias` | Subsampling conv kernels fed to `ggml_conv_2d` / `ggml_conv_2d_dw_direct`, which have no quantized path. |
| `encoder.layers.N.conv.pointwise_conv{1,2}.weight`/`.bias`, `conv.depthwise_conv.weight`/`.bias` | Conformer conv module weights are `ggml_reshape_2d`'d (pointwise) or used by the F32 depthwise conv (`conformer.cpp`: one SSM_CONV on a GPU backend, im2col + mul_mat on the CPU). |
| `encoder.layers.N.conv.batch_norm.{weight,bias,running_mean,running_var}` | Batch-norm running stats folded into per-channel scale/shift via raw `float*` in `conformer.cpp`. |
| `encoder.layers.N.self_attn.pos_bias_u`, `pos_bias_v` | Reshaped to `[dk,1,H]` before broadcasting; not a matmul weight. |
| all `*norm*.weight` / `*norm*.bias` (LayerNorm gain/bias) | Small per-channel vectors; consumed by `ggml_norm`/`ggml_mul`/`ggml_add`, not `ggml_mul_mat`. |
| all `*.bias` | Bias vectors are added after the matmul, never the `ggml_mul_mat` `src0`. |

**Rule of thumb:** conv / LSTM / featurizer / batch_norm / norm / bias / pos_bias
/ embedding tensors **always stay F32**. Only the large encoder FFN + attention
projections, the subsampling output projection, and the joint enc/pred
projections are quantized.

#### Implementation note

The per-component `clone_weight` / `clone_w` helpers were relaxed to clone a
weight in its **stored** type (preserving the raw bytes) instead of asserting
F32, so a quantized `src0` flows straight into `ggml_mul_mat`. For F32 tensors
this is byte-for-byte identical to the previous behaviour, so the default
(`--dtype f32`) path and all existing tests are unchanged. The
`joint.joint_net.2.weight` and CTC-head `assert(type == GGML_TYPE_F32)` guards
remain in place as a safety net — those tensors are never quantized.

---

### Ternary tensors

Packed ternary GGUFs (`--ternary keep`, see `docs/ternary.md`) store their
linears as `.qweight` (I8) and `.scales` (F16). `parakeet-cli quantize` never
re-quantizes them; it copies both tensors verbatim.

## Measured size + WER

WER is word-level vs NeMo (`scripts/validate_vs_nemo.py` on
`tests/fixtures/speech.wav`; 0.0 = byte-for-byte identical transcript). CPU,
batch 1, deterministic greedy.

### `nvidia/parakeet-tdt_ctc-110m` (anchor, 156 tensors quantized)

| dtype | GGUF size | vs F32 | WER (TDT) | WER (CTC) |
| --- | --- | --- | --- | --- |
| `f32` | 437.5 MB (458,719,328 B) | 1.00× | 0.0 | 0.0 |
| `f16` | 255.1 MB (267,452,512 B) | 0.58× | **0.0** | — |
| `q8_0` | 169.6 MB (177,796,192 B) | 0.39× | **0.0** | **0.0** |
| `q6_k`¹ | 148.7 MB (155,937,344 B) | 0.34× | **0.0** | — |
| `q4_k`¹ | 125.3 MB (131,387,456 B) | 0.29× | **0.0** | — |

¹ K-quants are produced by `parakeet-cli quantize` (not the Python converter).
They quantize **155** of the 156 allowlisted tensors: `joint.pred.weight`
(ggml `ne[0]` = 640, **not** divisible by the 256-element K-quant superblock) is
kept F32. `joint.enc.weight` (`ne[0]` = 512) and `encoder.pre_encode.out.weight`
(`ne[0]` = 2560) **are** quantized. The `q*_0` variants (block 32) quantize all
156. Even `q4_k` reproduces the NeMo transcript byte-for-byte on the speech
clip (WER 0.0).

### `nvidia/parakeet-tdt-0.6b-v2` (spot-check, 219 tensors quantized)

| dtype | GGUF size | vs F32 | WER (TDT) |
| --- | --- | --- | --- |
| `f32` | 2357.2 MB (2,471,701,760 B) | 1.00× | 0.0 |
| `q8_0` | 862.0 MB (903,835,904 B) | 0.37× | **0.0** |

No tensor had to be pulled from the allowlist after testing — every allowlisted
weight quantized cleanly and both models held WER 0 at `f16` and `q8_0`.

---

## Usage

```bash
# F16 (default linear weights → half precision; everything else F32)
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m --dtype f16 --output model_f16.gguf

# Q8_0
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m --dtype q8_0 --output model_q8.gguf

# Parity gate (WER vs NeMo)
.venv/bin/python scripts/validate_vs_nemo.py \
    --model nvidia/parakeet-tdt_ctc-110m --gguf model_q8.gguf \
    --audio tests/fixtures/speech.wav --head rnnt
```
