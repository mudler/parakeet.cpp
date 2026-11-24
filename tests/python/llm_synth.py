"""Builds a tiny synthetic Nemotron-3.5-Transcribe checkpoint (safetensors + config) for the LLM
converter tests. 2 Mamba-2 + 2 MoE + 1 attention layers, tiny dimensions, BF16 weights with the
checkpoint's tensor names. Shared by check_llm_convert.py and test_llm_loader.cpp's fixture setup."""
import json
import os
import struct

import numpy as np

H, V = 64, 96                      # hidden size, vocab
HEADS, HD, STATE, GROUPS, KER = 4, 8, 16, 2, 4
INNER = HEADS * HD                 # 32
CONV = INNER + 2 * GROUPS * STATE  # 96
INP = INNER + CONV + HEADS         # 132
AH, AKV, AHD = 4, 2, 16
NE, FF, SFF = 4, 32, 64
PATTERN = ["mamba", "moe", "mamba", "moe", "attention"]


def config():
    return {
        "audio_token_id": V, "eos_token_id": [11], "pad_token_id": 0,
        "llm_config": {
            "hidden_size": H, "vocab_size": V, "layers_block_type": PATTERN,
            "layer_norm_epsilon": 1e-5, "tie_word_embeddings": False, "max_position_embeddings": 4096,
            "mamba_num_heads": HEADS, "mamba_head_dim": HD, "ssm_state_size": STATE, "n_groups": GROUPS,
            "conv_kernel": KER, "chunk_size": 8, "time_step_min": 0.001,
            "num_attention_heads": AH, "num_key_value_heads": AKV, "head_dim": AHD,
            "n_routed_experts": NE, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "moe_intermediate_size": FF, "moe_shared_expert_intermediate_size": SFF,
            "routed_scaling_factor": 2.5, "norm_topk_prob": True, "mlp_hidden_act": "relu2",
            "bos_token_id": 1, "pad_token_id": 0, "eos_token_id": 2,
        },
    }


def tensors(seed=0):
    """name -> (dtype, float32 array). BF16 values are rounded to bf16 so they convert exactly."""
    rng = np.random.default_rng(seed)
    t = {}

    def bf(name, *shape, scale=0.5):
        a = (rng.standard_normal(shape) * scale).astype(np.float32)
        t[name] = ("BF16", (a.view(np.uint32) & 0xFFFF0000).view(np.float32))

    def f32(name, *shape, scale=0.5):
        t[name] = ("F32", (rng.standard_normal(shape) * scale).astype(np.float32))

    bf("llm.model.embed_tokens.weight", V, H)
    bf("llm.lm_head.weight", V, H)
    bf("llm.model.norm.weight", H)
    t["llm.lm_head._extra_state"] = ("BF16", np.zeros(0, np.float32))
    for i, k in enumerate(PATTERN):
        p = f"llm.model.layers.{i}."
        bf(p + "norm.weight", H)
        m = p + "mixer."
        if k == "mamba":
            bf(m + "in_proj.weight", INP, H)
            bf(m + "conv1d.weight", CONV, 1, KER)
            bf(m + "conv1d.bias", CONV)
            f32(m + "_fp32_params.A_log", HEADS)
            f32(m + "_fp32_params.D", HEADS)
            f32(m + "_fp32_params.dt_bias", HEADS)
            bf(m + "norm.weight", INNER)
            bf(m + "out_proj.weight", H, INNER)
        elif k == "attention":
            bf(m + "q_proj.weight", AH * AHD, H)
            bf(m + "k_proj.weight", AKV * AHD, H)
            bf(m + "v_proj.weight", AKV * AHD, H)
            bf(m + "o_proj.weight", H, AH * AHD)
            for s in ("q_proj", "k_proj", "v_proj", "o_proj"):
                t[m + s + "._extra_state"] = ("BF16", np.zeros(0, np.float32))
            t[m + "attn_module._extra_state"] = ("BF16", np.zeros(0, np.float32))
        else:
            bf(m + "gate.weight", NE, H)
            f32(m + "gate.e_score_correction_bias", NE)
            bf(m + "experts.gate_and_up_projs", NE, H, FF)
            bf(m + "experts.down_projs", NE, FF, H)
            bf(m + "shared_experts.up_proj.weight", SFF, H)
            bf(m + "shared_experts.down_proj.weight", H, SFF)
    # tensors of other parts of the checkpoint: the converter must ignore them
    bf("llm.mtp.layers.0.eh_proj.weight", H, 2 * H)
    bf("perception.proj.weight", H, 32)
    bf("ctc_timestamp.head.weight", 8, 8)
    return t


def write_safetensors(path, t):
    header, off, blobs = {}, 0, []
    for name, (dt, a) in t.items():
        if dt == "BF16":
            raw = (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raw = np.ascontiguousarray(a, "<f4").tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def make(dirpath, seed=0):
    t = tensors(seed)
    st, cfg = os.path.join(dirpath, "model.safetensors"), os.path.join(dirpath, "config.json")
    write_safetensors(st, t)
    json.dump(config(), open(cfg, "w"))
    return st, cfg, t
