#!/usr/bin/env python3
"""scripts/convert_nemotron_speechlm_llm.py on a tiny synthetic checkpoint (no download, no model).

Checks tensor names, shapes, types, the expert transpose, the A = -exp(A_log) transform, the
metadata, the skip list, F16 / Q8_0 output and --layout-only. Skips (77) without numpy or gguf.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import gguf
except ImportError:
    print("skip: numpy and gguf are needed")
    sys.exit(77)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import llm_synth as S  # noqa: E402

CONVERT = os.path.join(ROOT, "scripts", "convert_nemotron_speechlm_llm.py")
VERIFY = os.path.join(ROOT, "scripts", "check_llm_gguf_against_safetensors.py")
failures = 0


def check(cond, msg):
    global failures
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures += 1


def run(*args):
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def tensors_of(path):
    r = gguf.GGUFReader(path)
    return r, {t.name: t for t in r.tensors}


def kv(r, key):
    f = r.fields[key]
    return f.contents()


def main():
    base = os.environ.get("TMPDIR") or os.getcwd()
    work = tempfile.mkdtemp(prefix="llm_convert_", dir=base)
    try:
        st, cfg, src = S.make(work)
        out = os.path.join(work, "f32.gguf")
        rc, log = run(CONVERT, "--safetensors", st, "--config", cfg, "--output", out)
        check(rc == 0, "f32 conversion failed: " + log)
        r, t = tensors_of(out)

        want = {"token_embd.weight": [S.V, S.H], "output.weight": [S.V, S.H], "output_norm.weight": [S.H]}
        for i, k in enumerate(S.PATTERN):
            b = f"blk.{i}."
            want[b + "attn_norm.weight"] = [S.H]
            if k == "mamba":
                want.update({b + "ssm_in.weight": [S.INP, S.H], b + "ssm_conv1d.weight": [S.CONV, S.KER],
                             b + "ssm_conv1d.bias": [S.CONV], b + "ssm_a": [S.HEADS], b + "ssm_d": [S.HEADS],
                             b + "ssm_dt.bias": [S.HEADS], b + "ssm_norm.weight": [S.INNER],
                             b + "ssm_out.weight": [S.H, S.INNER]})
            elif k == "attention":
                want.update({b + "attn_q.weight": [S.AH * S.AHD, S.H], b + "attn_k.weight": [S.AKV * S.AHD, S.H],
                             b + "attn_v.weight": [S.AKV * S.AHD, S.H], b + "attn_output.weight": [S.H, S.AH * S.AHD]})
            else:
                want.update({b + "ffn_gate_inp.weight": [S.NE, S.H], b + "ffn_exp_probs_b": [S.NE],
                             b + "ffn_up_exps.weight": [S.NE, S.FF, S.H], b + "ffn_down_exps.weight": [S.NE, S.H, S.FF],
                             b + "ffn_up_shexp.weight": [S.SFF, S.H], b + "ffn_down_shexp.weight": [S.H, S.SFF]})
        check(set(t) == set(want), f"tensor set differs: extra {sorted(set(t) - set(want))} missing {sorted(set(want) - set(t))}")
        for n, shape in want.items():
            if n in t:
                # numpy order = the checkpoint-style order; ggml ne is its reverse
                check(list(t[n].shape[::-1]) == shape, f"{n}: shape {list(t[n].shape[::-1])}, want {shape}")
                check(t[n].tensor_type.name == "F32", f"{n}: type {t[n].tensor_type.name}, want F32")

        def data(n):
            return np.array(t[n].data, dtype=np.float32).reshape(want[n])

        ex = "llm.model.layers.1.mixer.experts."
        check(np.array_equal(data("blk.1.ffn_up_exps.weight"), np.swapaxes(src[ex + "gate_and_up_projs"][1], 1, 2)),
              "ffn_up_exps is not the [expert, out, in] transpose")
        check(np.array_equal(data("blk.1.ffn_down_exps.weight"), np.swapaxes(src[ex + "down_projs"][1], 1, 2)),
              "ffn_down_exps is not the [expert, out, in] transpose")
        check(not np.array_equal(src[ex + "gate_and_up_projs"][1].shape, data("blk.1.ffn_up_exps.weight").shape),
              "transpose test is vacuous (square expert)")
        check(np.allclose(data("blk.0.ssm_a"), -np.exp(src["llm.model.layers.0.mixer._fp32_params.A_log"][1])),
              "ssm_a is not -exp(A_log)")
        check(np.array_equal(data("blk.0.ssm_d"), src["llm.model.layers.0.mixer._fp32_params.D"][1]), "ssm_d differs")
        check(np.array_equal(data("blk.0.ssm_conv1d.weight"),
                             src["llm.model.layers.0.mixer.conv1d.weight"][1][:, 0, :]), "conv1d weight differs")
        check(np.array_equal(data("token_embd.weight"), src["llm.model.embed_tokens.weight"][1]), "embedding differs")
        check(np.array_equal(data("blk.4.attn_q.weight"), src["llm.model.layers.4.mixer.q_proj.weight"][1]), "q_proj differs")

        a = S.PATTERN
        check(kv(r, "general.architecture") == "nemotron-h-moe-llm", "architecture")
        p = "nemotron-h-moe-llm."
        check(kv(r, p + "layer_pattern") == a, "layer_pattern")
        for k, v in {"block_count": 5, "embedding_length": S.H, "vocab_size": S.V, "ssm.head_count": S.HEADS,
                     "ssm.head_dim": S.HD, "ssm.inner_size": S.INNER, "ssm.state_size": S.STATE,
                     "ssm.group_count": S.GROUPS, "ssm.conv_kernel": S.KER, "ssm.chunk_size": 8,
                     "attention.head_count": S.AH, "attention.head_count_kv": S.AKV, "attention.head_dim": S.AHD,
                     "expert_count": S.NE, "expert_used_count": 2, "expert_shared_count": 1,
                     "expert_feed_forward_length": S.FF, "expert_shared_feed_forward_length": S.SFF,
                     "token.audio_id": S.V, "token.eos_id": 11, "token.bos_id": 1, "token.pad_id": 0}.items():
            check(kv(r, p + k) == v, f"{k}: {kv(r, p + k)!r}, want {v!r}")
        check(abs(float(kv(r, p + "expert_weights_scale")) - 2.5) < 1e-6, "expert_weights_scale")
        check(bool(kv(r, p + "expert_weights_norm")) is True, "expert_weights_norm")

        rc, log = run(VERIFY, "--gguf", out, "--safetensors", st)
        check(rc == 0 and "OK" in log, "verifier rejects a good GGUF: " + log)

        # F16 and Q8_0: linears change type, small tensors stay F32, experts follow --expert-dtype
        for dt, ty in (("f16", "F16"), ("q8_0", "Q8_0")):
            o = os.path.join(work, dt + ".gguf")
            rc, log = run(CONVERT, "--safetensors", st, "--config", cfg, "--output", o, "--dtype", dt)
            check(rc == 0, dt + " conversion failed: " + log)
            _, t2 = tensors_of(o)
            for n in ("token_embd.weight", "output.weight", "blk.0.ssm_in.weight", "blk.4.attn_q.weight",
                      "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_shexp.weight"):
                check(t2[n].tensor_type.name == ty, f"{dt}: {n} is {t2[n].tensor_type.name}, want {ty}")
            for n in ("blk.1.ffn_gate_inp.weight", "blk.0.ssm_a", "blk.0.ssm_conv1d.weight", "output_norm.weight",
                      "blk.1.ffn_exp_probs_b", "blk.0.ssm_norm.weight"):
                check(t2[n].tensor_type.name == "F32", f"{dt}: {n} must stay F32")
            check(list(t2["blk.1.ffn_up_exps.weight"].shape[::-1]) == want["blk.1.ffn_up_exps.weight"], f"{dt}: expert shape")
            ref = np.swapaxes(src[ex + "gate_and_up_projs"][1], 1, 2)
            if dt == "f16":
                got = np.array(t2["blk.1.ffn_up_exps.weight"].data, dtype=np.float32).reshape(ref.shape)
            else:
                got = gguf.dequantize(t2["blk.1.ffn_up_exps.weight"].data, gguf.GGMLQuantizationType.Q8_0)
                got = got.reshape(ref.shape)
            check(np.abs(got - ref).max() < 0.02, f"{dt}: expert values off by {np.abs(got - ref).max()}")
            rc, log = run(VERIFY, "--gguf", o, "--safetensors", st)
            check(rc == 0, f"verifier rejects the {dt} GGUF: " + log)
        o = os.path.join(work, "mixed.gguf")
        rc, log = run(CONVERT, "--safetensors", st, "--config", cfg, "--output", o, "--dtype", "f16", "--expert-dtype", "q8_0")
        _, t3 = tensors_of(o)
        check(rc == 0 and t3["blk.1.ffn_up_exps.weight"].tensor_type.name == "Q8_0"
              and t3["output.weight"].tensor_type.name == "F16", "--expert-dtype")

        # --layout-only: same table, and the verifier accepts it
        o = os.path.join(work, "layout.gguf")
        rc, log = run(CONVERT, "--safetensors", st, "--config", cfg, "--output", o, "--layout-only")
        check(rc == 0, "layout-only failed: " + log)
        check(os.path.getsize(o) == os.path.getsize(out), "layout-only file size differs from the full file")
        rc, log = run(VERIFY, "--gguf", o, "--safetensors", st)
        check(rc == 0, "verifier rejects the layout-only GGUF: " + log)

        # --dry-run table and the verifier's failure paths
        rc, log = run(CONVERT, "--safetensors", st, "--config", cfg, "--dry-run")
        tab = json.loads(log[: log.rindex("}") + 1])
        check(len(tab["tensors"]) == len(want), "dry-run tensor count")
        tab["tensors"][3]["shape"] = [1, 1]
        tj = os.path.join(work, "bad.json")
        json.dump(tab, open(tj, "w"))
        rc, log = run(VERIFY, "--table", tj, "--safetensors", st)
        check(rc == 1 and "shape" in log, "verifier accepted a wrong shape")

        # unknown llm tensor in the checkpoint is an error, not silently dropped
        src2 = dict(src)
        src2["llm.model.layers.0.mixer.surprise.weight"] = ("BF16", np.zeros(4, np.float32))
        S.write_safetensors(os.path.join(work, "bad.safetensors"), src2)
        rc, log = run(CONVERT, "--safetensors", os.path.join(work, "bad.safetensors"), "--config", cfg,
                      "--output", os.path.join(work, "x.gguf"))
        check(rc != 0 and "does not know" in log, "unknown llm tensor was not rejected")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if failures:
        print(f"{failures} failure(s)", file=sys.stderr)
        return 1
    print("check_llm_convert OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
