# Nemotron 3.5 Transcribe: language model weights

Status: converter and loader only. There is no graph and no forward pass yet.

`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` pairs an audio encoder with a 52-layer
language model. This page covers the language model (`llm.*` tensors of the checkpoint).
The audio side is a separate file (`docs/speechlm-audio.md`).

## Model

Each layer is one pre-norm block with one mixer. The layer pattern is stored in the GGUF
(23 Mamba-2, 23 MoE, 6 attention for the released model). Hidden size 2688, vocabulary
131072, untied embeddings and `lm_head`, no RoPE. MoE layers have 128 routed experts
(top 6, sigmoid router with a score bias, ReLU squared, no gate projection) and one shared
expert.

## GGUF layout

Architecture `nemotron-h-moe-llm`. Keys, all under the prefix `nemotron-h-moe-llm.`:
`block_count`, `layer_pattern` (string array of `mamba`, `moe`, `attention`),
`embedding_length`, `vocab_size`, `context_length`, `rms_norm_eps`, `tie_word_embeddings`,
`ssm.{head_count,head_dim,inner_size,state_size,group_count,conv_kernel,chunk_size,time_step_min}`,
`attention.{head_count,head_count_kv,head_dim,use_rope}`,
`expert_{count,used_count,shared_count,feed_forward_length,shared_feed_forward_length}`,
`expert_weights_scale`, `expert_weights_norm`, `expert_gating_func`, `ffn_activation`,
`token.{audio_id,eos_id,bos_id,pad_id}`.

Tensors (ggml `ne` order, `ne[0]` first). The numpy shape is the reverse.

| GGUF name | ne | Type |
|---|---|---|
| `token_embd.weight`, `output.weight` | [2688, 131072] | linear |
| `output_norm.weight`, `blk.i.attn_norm.weight` | [2688] | F32 |
| `blk.i.ssm_in.weight` | [2688, 10304] | linear |
| `blk.i.ssm_conv1d.weight` / `.bias` | [4, 6144] / [6144] | F32 |
| `blk.i.ssm_a` | [64] | F32, stored as `-exp(A_log)` |
| `blk.i.ssm_d`, `blk.i.ssm_dt.bias` | [64] | F32 |
| `blk.i.ssm_norm.weight` | [4096] | F32 |
| `blk.i.ssm_out.weight` | [4096, 2688] | linear |
| `blk.i.attn_q.weight` / `attn_k` / `attn_v` | [2688, 4096] / [2688, 256] / [2688, 256] | linear |
| `blk.i.attn_output.weight` | [4096, 2688] | linear |
| `blk.i.ffn_gate_inp.weight` | [2688, 128] | F32 |
| `blk.i.ffn_exp_probs_b` | [128] | F32 |
| `blk.i.ffn_up_exps.weight` | [2688, 1856, 128] | expert |
| `blk.i.ffn_down_exps.weight` | [1856, 2688, 128] | expert |
| `blk.i.ffn_up_shexp.weight` / `ffn_down_shexp.weight` | [2688, 3712] / [3712, 2688] | linear |

"linear" and "expert" tensors are F32, F16 or Q8_0. Q8_0 needs `ne[0]` divisible by 32;
the converter falls back to F16 otherwise. The checkpoint stores the experts as
`[expert, in, out]`; the converter transposes each to `[expert, out, in]`. A GGUF of the
released model has 401 tensors and 31.58 billion parameters.

Skipped: `perception.*`, `ctc_timestamp.*`, `llm.mtp.*` (the MTP block) and all
`_extra_state` tensors.

## Convert

```
python scripts/convert_nemotron_speechlm_llm.py --hf --output llm-q8_0.gguf --dtype q8_0
python scripts/convert_nemotron_speechlm_llm.py --safetensors model.safetensors \
    --config config.json --output llm-f16.gguf --dtype f16 [--expert-dtype q8_0]
```

Output size: about 63 GB (F16), 33.6 GB (Q8_0), 126 GB (F32). Needs `numpy` and `gguf`.

Memory: the converter never loads the model. It reads each tensor as byte ranges (an expert
at a time, other tensors in pieces of at most `--chunk-mb`, default 256 MB of float32) and
writes the GGUF in order, so peak RAM is a few times the chunk size. The checkpoint can be
read over HTTP range requests, but the weights are then downloaded in full: plan for the
network time and the output disk space. Nothing is written to a temporary file.

`--dry-run` prints the tensor table. `--layout-only` writes a sparse GGUF with the real
header and tensor table and no data. Check a GGUF against the checkpoint header (a few
hundred KB, no weights):

```
python scripts/check_llm_gguf_against_safetensors.py --gguf llm.gguf --hf
```

## Load

`pk::LlmLoader::open(path, &err)` (`src/llm_loader.hpp`) reads the header and checks every
key, and every tensor name, shape and type against the layout the layer pattern implies.
It refuses missing or extra tensors, wrong shapes, F32-only tensors in another type, and
truncated files, with a message that names the tensor. `realize(backend, &err)` allocates the
weights on a backend and copies them from the file through a 64 MB staging buffer. Before
`realize`, `tensor(name)` returns descriptors only (type and shape).

The loader is standalone, like `SileroVad::load`, and does not use `ModelLoader` (which
reads the whole file into memory and expects an encoder).

## Not implemented

Tokenizer and prompt, the audio splice, the Mamba-2, MoE and attention graphs, prefill and
decode, KV and state caches, K-quants, the MTP block, bundle components, GPU parity checks.
