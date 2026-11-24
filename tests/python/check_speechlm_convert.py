#!/usr/bin/env python3
"""CPU-only check of scripts/convert_nemotron_speechlm_audio.py (no checkpoint, no torch).

Builds a tiny safetensors file with the checkpoint's `perception.*` tensor names (one-layer
encoders, BF16), converts it, and checks the two GGUF files: architecture and keys the loader
reads, tensor names and shapes, the BF16 -> F32 conversion, the fused QKV passed through, and
that the unused speaker head and the LLM tensors are left out. Exit 0 on success, 77 when numpy or
the gguf package is missing.
"""
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
try:
    import numpy as np  # noqa: E402
    import gguf  # noqa: E402
except ImportError:
    print("skip: numpy and gguf are needed")
    sys.exit(77)
import convert_nemotron_speechlm_audio as conv  # noqa: E402

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("FAIL:", what)


def bf16(a):
    return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype("<u2")


def main():
    rng = np.random.default_rng(1)
    D, H, L, FF = 64, 4, 1, 256            # ASR encoder
    d, h, ff = 32, 2, 64                   # speaker encoder
    tensors = {}

    def add(name, shape):
        tensors[name] = rng.standard_normal(shape).astype(np.float32).round(2)

    def encoder(prefix, dm, ffd, qk):
        add(prefix + "pre_encode.proj.weight", (dm, 1024))
        for n in ("embed_norm", "final_norm"):
            add(prefix + n + ".weight", (dm,)); add(prefix + n + ".bias", (dm,))
        for l in range(L):
            b = prefix + f"layers.{l}."
            for n in ("norm1", "norm2"):
                add(b + n + ".weight", (dm,)); add(b + n + ".bias", (dm,))
            add(b + "attn.w_qkv.weight", (3 * dm, dm))
            add(b + "attn.out_proj.weight", (dm, dm)); add(b + "attn.out_proj.bias", (dm,))
            if qk:
                for n in ("q_norm", "k_norm"):
                    add(b + f"attn.{n}.weight", (dm // (H if dm == D else h),))
                    add(b + f"attn.{n}.bias", (dm // (H if dm == D else h),))
            add(b + "ffn.net.0.weight", (ffd, dm)); add(b + "ffn.net.0.bias", (ffd,))
            add(b + "ffn.net.3.weight", (dm, ffd)); add(b + "ffn.net.3.bias", (dm,))

    encoder("perception.encoder.asr_encoder.", D, FF, True)
    for n, k in (("asr_norm", D), ("diar_norm", 8)):
        add(f"perception.encoder.{n}.weight", (k,)); add(f"perception.encoder.{n}.bias", (k,))
    add("perception.proj.weight", (48, D)); add("perception.proj.bias", (48,))
    add("perception.preprocessor.featurizer.fb", (1, 128, 257)); add("perception.preprocessor.featurizer.window", (400,))
    dp = "perception.encoder.diarization_model."
    encoder(dp + "encoder.", d, ff, False)
    add(dp + "preprocessor.featurizer.fb", (1, 128, 257)); add(dp + "preprocessor.featurizer.window", (400,))
    sm = dp + "sortformer_modules."
    add(sm + "encoder_proj.weight", (192, d)); add(sm + "encoder_proj.bias", (192,))
    add(sm + "subpixel_upsample.weight", (1536, 192, 3)); add(sm + "subpixel_upsample.bias", (1536,))
    add(sm + "first_hidden_to_hidden.weight", (192, 192)); add(sm + "first_hidden_to_hidden.bias", (192,))
    add(sm + "single_hidden_to_spks.weight", (8, 192)); add(sm + "single_hidden_to_spks.bias", (8,))
    add(sm + "hidden_to_spks.weight", (8, 384)); add(sm + "hidden_to_spks.bias", (8,))
    add(sm + "learnable_sil_emb", (d,))
    add("llm.model.embed_tokens.weight", (10, 4))      # must be ignored

    cfg = {
        "perception": {"output_dim": 48},
        "audio_config": {"speaker_activity_threshold": 0.5, "diarization_config": {
            "chunk_length": 500, "chunk_left_context": 50, "chunk_right_context": 50, "fifo_length": 40,
            "speaker_cache_update_period": 300, "head_config": {"hidden_size": 192},
            "streaming_config": {"speaker_cache_length": 188, "speaker_cache_silence_frames_per_speaker": 1,
                                 "prediction_score_threshold": 0.25, "latest_frames_score_boost": 0.05,
                                 "strong_boost_rate": 0.75, "weak_boost_rate": 1.5,
                                 "min_positive_scores_rate": 0.5}}},
        "pe_encoder_config": {
            "asr_encoder_cfg": {"d_model": D, "n_heads": H, "n_layers": L, "ff_expansion": 4, "qk_norm": True,
                                "subsampling_factor": 8, "rope_base": 10000.0, "rotary_fraction": 0.5},
            "diarization_model_cfg": {"max_num_of_spks": 8, "encoder": {
                "d_model": d, "n_heads": h, "n_layers": L, "ff_expansion": 2.0}}},
    }
    # FFN width must follow the config: ASR D*4 = FF, speaker d*2 = ff.
    assert D * 4 == FF and int(d * 2.0) == ff

    with tempfile.TemporaryDirectory() as tmp:
        header, blob, off = {}, bytearray(), 0
        for name, a in tensors.items():
            raw = bf16(a).tobytes()
            header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
            blob += raw; off += len(raw)
        hb = json.dumps(header).encode()
        st = os.path.join(tmp, "model.safetensors")
        with open(st, "wb") as f:
            f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(blob)
        cj = os.path.join(tmp, "config.json")
        json.dump(cfg, open(cj, "w"))
        out = os.path.join(tmp, "out")
        sys.argv = ["convert", "--safetensors", st, "--config", cj, "--output-dir", out, "--name", "t"]
        conv.main()

        def read(path):
            r = gguf.GGUFReader(path)
            kv = {}
            for k, f in r.fields.items():
                if f.types and f.types[0] == gguf.GGUFValueType.STRING and len(f.types) == 1:
                    kv[k] = bytes(f.parts[-1]).decode()
                elif f.types and len(f.types) == 1:
                    kv[k] = f.parts[-1][0].item() if hasattr(f.parts[-1][0], "item") else f.parts[-1][0]
            return kv, {t.name: t for t in r.tensors}

        kv, ts = read(os.path.join(out, "t-audio.gguf"))
        check(kv["parakeet.arch"] == "speechlm_audio", "audio arch")
        check(kv["parakeet.encoder.qk_norm"] == 1 and kv["parakeet.encoder.activation"] == "gelu_erf", "audio qk_norm, activation")
        check(abs(kv["parakeet.encoder.rotary_fraction"] - 0.5) < 1e-6, "rotary fraction 0.5")
        check(kv["parakeet.encoder.ff_dim"] == FF and kv["parakeet.encoder.d_model"] == D, "audio sizes")
        check(kv["parakeet.preprocessor.normalize"] == "NA", "raw mel")
        check(kv["parakeet.speechlm.llm_dim"] == 48 and kv["parakeet.speechlm.num_speakers"] == 8, "speechlm keys")
        check(kv["parakeet.speechlm.chunk_len"] == 500 and kv["parakeet.speechlm.chunk_left_context"] == 50, "chunking keys")
        want = {"encoder.pre_encode.proj.weight", "encoder.embed_norm.bias", "encoder.layers.0.attn.q_norm.weight",
                "encoder.layers.0.attn.w_qkv.weight", "encoder.layers.0.ffn.net.3.bias", "speechlm.asr_norm.weight",
                "speechlm.diar_norm.bias", "speechlm.proj.weight", "speechlm.proj.bias",
                "preprocessor.featurizer.fb", "preprocessor.featurizer.window"}
        check(want <= set(ts), f"audio tensors present: missing {sorted(want - set(ts))}")
        check(not any(n.startswith(("llm", "perception", "sortformer")) for n in ts), "no LLM or speaker tensors in audio GGUF")
        t = ts["encoder.layers.0.attn.w_qkv.weight"]
        check(list(t.shape) == [D, 3 * D], "w_qkv ggml shape")
        ref = bf16(tensors["perception.encoder.asr_encoder.layers.0.attn.w_qkv.weight"]).astype(np.uint32) << 16
        check(np.array_equal(np.asarray(t.data).reshape(-1), ref.view(np.float32).reshape(-1)), "BF16 -> F32 exact")

        kv, ts = read(os.path.join(out, "t-diar.gguf"))
        check(kv["parakeet.arch"] == "diarization" and kv["parakeet.diar.n_speakers"] == 8, "diar arch")
        check(kv["parakeet.encoder.qk_norm"] == 0 and abs(kv["parakeet.encoder.rotary_fraction"] - 1.0) < 1e-6, "diar: no qk_norm, full rotary")
        check(kv["parakeet.diar.chunk_len"] == 500 and kv["parakeet.diar.spkcache_len"] == 188 and
              kv["parakeet.diar.fifo_len"] == 40 and kv["parakeet.diar.spkcache_update_period"] == 300, "diar offline streaming config")
        check("sortformer_modules.learnable_sil_emb" in ts and "sortformer_modules.subpixel_upsample.weight" in ts, "diar head tensors")
        check("sortformer_modules.hidden_to_spks.weight" not in ts, "unused head dropped")
        check("encoder.layers.0.attn.q_norm.weight" not in ts, "diar has no qk norm tensors")

        # f16 keeps norms and biases in F32 and quantizes only the allowlisted linears
        sys.argv = ["convert", "--safetensors", st, "--config", cj, "--output-dir", out, "--name", "h", "--dtype", "f16"]
        conv.main()
        _, ts = read(os.path.join(out, "h-audio.gguf"))
        check(ts["encoder.layers.0.attn.w_qkv.weight"].tensor_type == gguf.GGMLQuantizationType.F16, "f16 linear")
        check(ts["encoder.layers.0.attn.q_norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32, "qk norm stays F32")
        check(ts["speechlm.proj.weight"].tensor_type == gguf.GGMLQuantizationType.F16, "adapter weight f16")
        check(ts["speechlm.proj.bias"].tensor_type == gguf.GGMLQuantizationType.F32, "adapter bias F32")
    print("FAILED" if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
