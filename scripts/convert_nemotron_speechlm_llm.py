#!/usr/bin/env python3
"""Convert the language model of nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B to GGUF.

The checkpoint is one 67 GB safetensors file. Only the `llm.*` tensors are read (the
decoder without its MTP block, about 63 GB in BF16). The converter streams them: each
tensor is read as byte ranges (a local file, or HTTP range requests from the Hub),
converted in pieces of at most --chunk-mb of source data, and written at once to the
output, so peak memory stays near a few times the chunk size. Nothing else is
downloaded and no torch / NeMo / transformers is needed.

Skipped: `perception.*` and `ctc_timestamp.*` (audio side, see convert_nemotron_speechlm_audio.py),
`llm.mtp.*` (speculative decoding block) and every `_extra_state` tensor.

The GGUF has architecture `nemotron-h-moe-llm`. Layout: docs/speechlm-llm.md.

  convert_nemotron_speechlm_llm.py --safetensors model.safetensors --config config.json --output out.gguf
  convert_nemotron_speechlm_llm.py --hf nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --output out.gguf

--dtype selects the type of the large linear weights and the embeddings (f32, f16 or
q8_0; q8_0 needs an inner dimension divisible by 32, otherwise the tensor stays F16).
--expert-dtype does the same for the routed expert tensors (default: --dtype). Router,
router bias, norms, conv, A, D and dt_bias are always F32.

--dry-run prints the tensor table as JSON and writes nothing. --layout-only writes the
real header and tensor table but skips the data (a sparse file of the full apparent
size); it exists to check a layout against a checkpoint header without downloading
the weights.
"""
import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.request

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    print("pip install gguf", file=sys.stderr)
    raise

HF_REPO = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
ARCH = "nemotron-h-moe-llm"
ALIGN = 32
LAYER_TYPES = {"mamba": "mamba", "moe": "moe", "attention": "attention"}
F32, F16, Q8 = gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.Q8_0


class Source:
    """Reads tensors of a safetensors file, locally or over HTTP ranges."""

    def __init__(self, path=None, url=None):
        self.path, self.url = path, url
        (n,) = struct.unpack("<Q", self._read(0, 8))
        self.base = 8 + n
        self.header = json.loads(self._read(8, n))
        self.header.pop("__metadata__", None)

    def _read(self, off, n):
        if self.path:
            with open(self.path, "rb") as f:
                f.seek(off)
                data = f.read(n)
        else:
            err = None
            for attempt in range(5):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + n - 1}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    break
                except Exception as e:  # network hiccup: retry the same range
                    err = e
                    time.sleep(2 * (attempt + 1))
            else:
                raise SystemExit(f"range read failed at {off}+{n}: {err}")
        if len(data) != n:
            raise SystemExit(f"short read at {off}: wanted {n} bytes, got {len(data)}")
        return data

    def read_rows(self, name, r0, r1):
        """Rows [r0, r1) of the leading dimension as float32, shape [r1 - r0, ...]."""
        m = self.header[name]
        shape = m["shape"]
        row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        item = {"BF16": 2, "F16": 2, "F32": 4}.get(m["dtype"])
        if item is None:
            raise SystemExit(f"unsupported dtype {m['dtype']} for {name}")
        s, e = m["data_offsets"]
        if e - s != item * row * shape[0]:
            raise SystemExit(f"{name}: data_offsets size does not match shape {shape}")
        raw = self._read(self.base + s + r0 * row * item, (r1 - r0) * row * item)
        if m["dtype"] == "BF16":
            a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif m["dtype"] == "F32":
            a = np.frombuffer(raw, dtype="<f4")
        else:
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        return a.reshape([r1 - r0] + list(shape[1:]))


class Plan:
    """One output tensor: where it comes from, its numpy shape (ggml ne reversed) and type."""

    def __init__(self, src, dst, src_shape, dst_shape, kind, qtype):
        self.src, self.dst = src, dst
        self.src_shape, self.dst_shape = list(src_shape), list(dst_shape)
        self.kind, self.qtype = kind, qtype

    @property
    def nbytes(self):
        n = int(np.prod(self.dst_shape, dtype=np.int64))
        if self.qtype == F32:
            return n * 4
        if self.qtype == F16:
            return n * 2
        bs, ts = gguf.GGML_QUANT_SIZES[self.qtype]
        return n // bs * ts

    def as_json(self):
        return {"name": self.dst, "shape": self.dst_shape, "type": self.qtype.name, "src": self.src,
                "kind": self.kind}


def pick_type(dtype, shape):
    """Type of a large 2-D weight: Q8_0 needs the inner (last) dimension divisible by 32."""
    if dtype == "f32":
        return F32
    if dtype == "q8_0" and shape[-1] % 32 == 0:
        return Q8
    return F16


def layer_kinds(cfg):
    out = []
    for t in cfg["llm_config"]["layers_block_type"]:
        if t not in LAYER_TYPES:
            raise SystemExit(f"unknown layer type {t!r} in llm_config.layers_block_type")
        out.append(t)
    return out


def plan_tensors(header, cfg, dtype="f32", expert_dtype=None):
    """The ordered output tensor list, checked against the checkpoint header."""
    expert_dtype = expert_dtype or dtype
    kinds = layer_kinds(cfg)
    lc = cfg["llm_config"]
    plans, used = [], set()

    def shape_of(name):
        if name not in header:
            raise SystemExit(f"checkpoint has no tensor {name}")
        used.add(name)
        return header[name]["shape"]

    def add(src, dst, kind, dst_shape=None, expert=False):
        sh = shape_of(src)
        ds = dst_shape if dst_shape is not None else sh
        if kind in ("f32", "a_log"):
            q = F32
        else:
            q = pick_type(expert_dtype if expert else dtype, ds)
        plans.append(Plan(src, dst, sh, ds, kind, q))

    add("llm.model.embed_tokens.weight", "token_embd.weight", "linear")
    add("llm.model.norm.weight", "output_norm.weight", "f32")
    add("llm.lm_head.weight", "output.weight", "linear")
    for i, k in enumerate(kinds):
        p = f"llm.model.layers.{i}."
        b = f"blk.{i}."
        mx = p + "mixer."
        add(p + "norm.weight", b + "attn_norm.weight", "f32")
        if k == "mamba":
            add(mx + "in_proj.weight", b + "ssm_in.weight", "linear")
            cw = shape_of(mx + "conv1d.weight")
            add(mx + "conv1d.weight", b + "ssm_conv1d.weight", "f32", dst_shape=[cw[0], cw[2]])
            add(mx + "conv1d.bias", b + "ssm_conv1d.bias", "f32")
            add(mx + "_fp32_params.A_log", b + "ssm_a", "a_log")
            add(mx + "_fp32_params.D", b + "ssm_d", "f32")
            add(mx + "_fp32_params.dt_bias", b + "ssm_dt.bias", "f32")
            add(mx + "norm.weight", b + "ssm_norm.weight", "f32")
            add(mx + "out_proj.weight", b + "ssm_out.weight", "linear")
        elif k == "attention":
            for s, d in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                add(mx + s + ".weight", b + d + ".weight", "linear")
        else:
            add(mx + "gate.weight", b + "ffn_gate_inp.weight", "f32")
            add(mx + "gate.e_score_correction_bias", b + "ffn_exp_probs_b", "f32")
            for s, d in (("gate_and_up_projs", "ffn_up_exps"), ("down_projs", "ffn_down_exps")):
                sh = shape_of(mx + "experts." + s)
                add(mx + "experts." + s, b + d + ".weight", "expert", dst_shape=[sh[0], sh[2], sh[1]], expert=True)
            add(mx + "shared_experts.up_proj.weight", b + "ffn_up_shexp.weight", "linear")
            add(mx + "shared_experts.down_proj.weight", b + "ffn_down_shexp.weight", "linear")

    # Every llm.* tensor must be either converted or on the skip list.
    skip = re.compile(r"(^llm\.mtp\.)|(_extra_state$)")
    extra = [k for k in header if k.startswith("llm.") and k not in used and not skip.search(k)]
    if extra:
        raise SystemExit(f"checkpoint has llm tensors this converter does not know: {extra[:5]}")
    # Cross-check the shapes against the config.
    h, v = int(lc["hidden_size"]), int(lc["vocab_size"])
    if header["llm.model.embed_tokens.weight"]["shape"] != [v, h]:
        raise SystemExit("embed_tokens shape does not match hidden_size / vocab_size in the config")
    return plans


def write_kvs(w, cfg):
    lc = cfg["llm_config"]
    kinds = layer_kinds(cfg)
    eos = cfg.get("eos_token_id", lc.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    heads, hd = int(lc["mamba_num_heads"]), int(lc["mamba_head_dim"])
    a = ARCH + "."
    w.add_string("general.name", "nemotron-3.5-transcribe-30b-a4b-llm")
    w.add_string("general.license", "OpenMDW-1.1")
    w.add_uint32(a + "block_count", len(kinds))
    w.add_array(a + "layer_pattern", kinds)
    w.add_uint32(a + "embedding_length", int(lc["hidden_size"]))
    w.add_uint32(a + "vocab_size", int(lc["vocab_size"]))
    w.add_uint32(a + "context_length", int(lc["max_position_embeddings"]))
    w.add_float32(a + "rms_norm_eps", float(lc["layer_norm_epsilon"]))
    w.add_bool(a + "tie_word_embeddings", bool(lc["tie_word_embeddings"]))
    w.add_uint32(a + "ssm.head_count", heads)
    w.add_uint32(a + "ssm.head_dim", hd)
    w.add_uint32(a + "ssm.inner_size", heads * hd)
    w.add_uint32(a + "ssm.state_size", int(lc["ssm_state_size"]))
    w.add_uint32(a + "ssm.group_count", int(lc["n_groups"]))
    w.add_uint32(a + "ssm.conv_kernel", int(lc["conv_kernel"]))
    w.add_uint32(a + "ssm.chunk_size", int(lc["chunk_size"]))
    w.add_float32(a + "ssm.time_step_min", float(lc["time_step_min"]))
    w.add_uint32(a + "attention.head_count", int(lc["num_attention_heads"]))
    w.add_uint32(a + "attention.head_count_kv", int(lc["num_key_value_heads"]))
    w.add_uint32(a + "attention.head_dim", int(lc["head_dim"]))
    w.add_bool(a + "attention.use_rope", False)
    w.add_uint32(a + "expert_count", int(lc["n_routed_experts"]))
    w.add_uint32(a + "expert_used_count", int(lc["num_experts_per_tok"]))
    w.add_uint32(a + "expert_shared_count", int(lc["n_shared_experts"]))
    w.add_uint32(a + "expert_feed_forward_length", int(lc["moe_intermediate_size"]))
    w.add_uint32(a + "expert_shared_feed_forward_length", int(lc["moe_shared_expert_intermediate_size"]))
    w.add_float32(a + "expert_weights_scale", float(lc["routed_scaling_factor"]))
    w.add_bool(a + "expert_weights_norm", bool(lc["norm_topk_prob"]))
    w.add_string(a + "expert_gating_func", "sigmoid")
    w.add_string(a + "ffn_activation", str(lc["mlp_hidden_act"]))
    w.add_uint32(a + "token.audio_id", int(cfg["audio_token_id"]))
    w.add_uint32(a + "token.eos_id", int(eos))
    w.add_uint32(a + "token.bos_id", int(lc["bos_token_id"]))
    w.add_uint32(a + "token.pad_id", int(cfg.get("pad_token_id", lc["pad_token_id"])))


def pad_to(f, n_written):
    pad = -n_written % ALIGN
    if pad:
        f.write(bytes(pad))


def encode(arr, qtype):
    """float32 array (rows, ..., inner) -> bytes of the target type."""
    if qtype == F32:
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if qtype == F16:
        return np.ascontiguousarray(arr, dtype="<f2").tobytes()
    return gguf.quantize(np.ascontiguousarray(arr, dtype=np.float32), qtype).tobytes()


def stream_tensor(src, p, chunk_bytes):
    """Yield the output bytes of one tensor, piece by piece."""
    n0 = p.src_shape[0]
    row_bytes = int(np.prod(p.src_shape[1:], dtype=np.int64) or 1) * 4  # float32 working copy
    step = max(1, min(n0, chunk_bytes // max(row_bytes, 1)))
    if p.kind == "expert":
        step = 1  # one expert at a time: 10 MB of BF16
    for r0 in range(0, n0, step):
        a = src.read_rows(p.src, r0, min(n0, r0 + step))
        if p.kind == "expert":
            a = np.swapaxes(a, 1, 2)  # [E, in, out] -> [E, out, in]
        elif p.kind == "a_log":
            a = -np.exp(a)
        elif p.kind == "f32" and list(a.shape[1:]) != p.dst_shape[1:]:
            a = a.reshape([a.shape[0]] + p.dst_shape[1:])  # conv1d [C, 1, K] -> [C, K]
        yield encode(a, p.qtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--safetensors", help="local model.safetensors")
    src.add_argument("--hf", nargs="?", const=HF_REPO, help="read the ranges from the Hub repo")
    ap.add_argument("--config", help="config.json (default: fetched from --hf; required with --safetensors)")
    ap.add_argument("--output", help="output .gguf path")
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f32")
    ap.add_argument("--expert-dtype", choices=["f32", "f16", "q8_0"], default=None)
    ap.add_argument("--chunk-mb", type=int, default=256, help="largest piece of float32 data held at once")
    ap.add_argument("--dry-run", action="store_true", help="print the tensor table as JSON, write nothing")
    ap.add_argument("--layout-only", action="store_true", help="write header and tensor table, skip the data")
    args = ap.parse_args()
    if not args.dry_run and not args.output:
        ap.error("--output is required")

    if args.safetensors:
        if not args.config:
            ap.error("--config is required with --safetensors")
        st = Source(path=args.safetensors)
        cfg = json.load(open(args.config))
    else:
        base = f"https://huggingface.co/{args.hf}/resolve/main/"
        st = Source(url=base + "model.safetensors")
        cfg = json.load(open(args.config)) if args.config else json.loads(urllib.request.urlopen(base + "config.json").read())

    plans = plan_tensors(st.header, cfg, args.dtype, args.expert_dtype)
    total = sum(p.nbytes for p in plans)
    if args.dry_run:
        json.dump({"arch": ARCH, "tensors": [p.as_json() for p in plans]}, sys.stdout, indent=1)
        print()
        print(f"{len(plans)} tensors, {total} bytes", file=sys.stderr)
        return

    w = gguf.GGUFWriter(args.output, ARCH, use_temp_file=False)
    write_kvs(w, cfg)
    for p in plans:
        if p.qtype == Q8:
            w.add_tensor_info(p.dst, gguf.quant_shape_to_byte_shape(p.dst_shape, Q8), np.uint8, p.nbytes, raw_dtype=Q8)
        else:
            w.add_tensor_info(p.dst, p.dst_shape, np.float32 if p.qtype == F32 else np.float16, p.nbytes)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    f = w.fout[0]
    pad_to(f, f.tell())
    chunk = args.chunk_mb * 1024 * 1024
    t0 = time.time()
    for i, p in enumerate(plans):
        start = f.tell()
        if args.layout_only:
            f.seek(start + p.nbytes)
        else:
            written = 0
            for piece in stream_tensor(st, p, chunk):
                f.write(piece)
                written += len(piece)
            if written != p.nbytes:
                raise SystemExit(f"{p.dst}: wrote {written} bytes, expected {p.nbytes}")
        pad_to(f, p.nbytes)
        if i % 20 == 0 or i == len(plans) - 1:
            print(f"[{i + 1}/{len(plans)}] {p.dst} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    if args.layout_only:
        f.truncate(f.tell())
    w.close()
    print(f"wrote {args.output}: tensors={len(plans)} bytes={total} dtype={args.dtype}"
          f" expert_dtype={args.expert_dtype or args.dtype}{' (layout only)' if args.layout_only else ''}")


if __name__ == "__main__":
    main()
