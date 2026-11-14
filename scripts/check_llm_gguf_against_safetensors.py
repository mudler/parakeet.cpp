#!/usr/bin/env python3
"""Compare the tensor table of an LLM GGUF with the header of the checkpoint it comes from.

Only the safetensors header is read (a few hundred KB, locally or by one HTTP range
request); no weights are needed. The expected GGUF name, shape and type class of every
`llm.*` tensor (without `llm.mtp.*` and `_extra_state`) are derived here, independent of
the converter, and compared with the GGUF in both directions.

  check_llm_gguf_against_safetensors.py --gguf out.gguf --hf [REPO]
  check_llm_gguf_against_safetensors.py --gguf out.gguf --safetensors model.safetensors
  check_llm_gguf_against_safetensors.py --table table.json --hf   # `--dry-run` output of the converter

--layout-only GGUF files (sparse, no data) are fine: only the tensor table is read.
Exit code 0 when everything matches, 1 otherwise.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_nemotron_speechlm_llm import HF_REPO, Source  # noqa: E402  (only the safetensors reader)

SKIP = re.compile(r"(^llm\.mtp\.)|(_extra_state$)")
F32_ONLY = ("norm.weight", "conv1d", "_fp32_params", "mixer.gate.", "mixer.norm.weight")
RULES = [  # checkpoint pattern -> (gguf name, shape rule)
    (r"llm\.model\.embed_tokens\.weight", "token_embd.weight", "same"),
    (r"llm\.lm_head\.weight", "output.weight", "same"),
    (r"llm\.model\.norm\.weight", "output_norm.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.norm\.weight", r"blk.\1.attn_norm.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.in_proj\.weight", r"blk.\1.ssm_in.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.conv1d\.weight", r"blk.\1.ssm_conv1d.weight", "drop1"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.conv1d\.bias", r"blk.\1.ssm_conv1d.bias", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\._fp32_params\.A_log", r"blk.\1.ssm_a", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\._fp32_params\.D", r"blk.\1.ssm_d", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\._fp32_params\.dt_bias", r"blk.\1.ssm_dt.bias", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.norm\.weight", r"blk.\1.ssm_norm.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.out_proj\.weight", r"blk.\1.ssm_out.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.q_proj\.weight", r"blk.\1.attn_q.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.k_proj\.weight", r"blk.\1.attn_k.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.v_proj\.weight", r"blk.\1.attn_v.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.o_proj\.weight", r"blk.\1.attn_output.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.gate\.weight", r"blk.\1.ffn_gate_inp.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.gate\.e_score_correction_bias", r"blk.\1.ffn_exp_probs_b", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.experts\.gate_and_up_projs", r"blk.\1.ffn_up_exps.weight", "swap12"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.experts\.down_projs", r"blk.\1.ffn_down_exps.weight", "swap12"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.shared_experts\.up_proj\.weight", r"blk.\1.ffn_up_shexp.weight", "same"),
    (r"llm\.model\.layers\.(\d+)\.mixer\.shared_experts\.down_proj\.weight", r"blk.\1.ffn_down_shexp.weight", "same"),
]


def expected_table(header):
    out = {}
    for name, meta in header.items():
        if not name.startswith("llm.") or SKIP.search(name):
            continue
        for pat, repl, rule in RULES:
            if re.fullmatch(pat, name):
                sh = list(meta["shape"])
                if rule == "drop1":
                    sh = [sh[0], sh[2]]
                elif rule == "swap12":
                    sh = [sh[0], sh[2], sh[1]]
                out[re.sub(pat, repl, name)] = (name, sh)
                break
        else:
            print(f"UNMAPPED checkpoint tensor: {name}", file=sys.stderr)
            out[None, name] = (name, None)
    return out


def gguf_table(path):
    import gguf
    r = gguf.GGUFReader(path)
    return {t.name: (list(t.shape[::-1]), t.tensor_type.name, int(t.n_bytes)) for t in r.tensors}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gguf")
    g.add_argument("--table", help="JSON written by the converter's --dry-run")
    s = ap.add_mutually_exclusive_group(required=True)
    s.add_argument("--safetensors")
    s.add_argument("--hf", nargs="?", const=HF_REPO)
    a = ap.parse_args()

    st = Source(path=a.safetensors) if a.safetensors else Source(
        url=f"https://huggingface.co/{a.hf}/resolve/main/model.safetensors")
    exp = expected_table(st.header)
    if a.gguf:
        got = gguf_table(a.gguf)
    else:
        got = {t["name"]: (t["shape"], t["type"], None) for t in json.load(open(a.table))["tensors"]}

    errors = []
    for key in [k for k in exp if isinstance(k, tuple)]:
        errors.append(f"unmapped checkpoint tensor {key[1]}")
    for name, (src, shape) in exp.items():
        if isinstance(name, tuple):
            continue
        if name not in got:
            errors.append(f"missing in GGUF: {name} (from {src})")
            continue
        gshape, gtype, _ = got[name]
        if gshape != shape:
            errors.append(f"{name}: shape {gshape}, checkpoint {src} implies {shape}")
        f32_only = any(k in src for k in F32_ONLY) or src.endswith(".e_score_correction_bias")
        if f32_only and gtype != "F32":
            errors.append(f"{name}: type {gtype}, must be F32")
    for name in got:
        if name not in exp:
            errors.append(f"extra in GGUF, not in the checkpoint's LLM set: {name}")
    n_ckpt = sum(1 for k in st.header if k.startswith("llm."))
    n_mtp = sum(1 for k in st.header if k.startswith("llm.mtp."))
    n_extra = sum(1 for k in st.header if k.startswith("llm.") and not k.startswith("llm.mtp.") and k.endswith("_extra_state"))
    n_exp = sum(1 for k in exp if not isinstance(k, tuple))
    print(f"checkpoint tensors {len(st.header)}; llm.* {n_ckpt} = {n_exp} converted + {n_mtp} mtp + {n_extra} _extra_state"
          f" (+ {n_ckpt - n_exp - n_mtp - n_extra} unaccounted)")
    print(f"GGUF tensors {len(got)}; expected {n_exp}")
    n_params = sum(int(__import__('math').prod(sh)) for _, sh in exp.values() if sh)
    print(f"converted parameters {n_params}; BF16 bytes of the converted tensors "
          f"{sum(st.header[s]['data_offsets'][1] - st.header[s]['data_offsets'][0] for s, _ in exp.values())}")
    if errors:
        print(f"{len(errors)} problems:")
        for e in errors[:50]:
            print("  " + e)
        return 1
    print("OK: names, shapes and F32-only types match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
