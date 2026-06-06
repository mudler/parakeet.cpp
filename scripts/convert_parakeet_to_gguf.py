#!/usr/bin/env python3
"""Convert a NeMo Parakeet checkpoint to GGUF (f32 / f16 / q8_0).

The GGUF is fully metadata-driven: all config lives in KV, and tensor names are
kept **verbatim** from the NeMo ``state_dict`` (no renaming) so the C++ port is a
1:1 mapping. The two featurizer buffers (``preprocessor.featurizer.fb`` and
``preprocessor.featurizer.window``) are lifted directly from the checkpoint so the
C++ side never re-derives the mel filterbank with librosa.

Quantization (``--dtype f16|q8_0``) is applied **only** to the large linear
weights that the C++ engine consumes directly via ``ggml_mul_mat`` (the encoder
FFN + attention projections, the subsampling output projection, and the joint
enc/pred projections). ggml dequantizes those on the fly inside the compute
graph. Everything the hand-rolled C++ reads as raw F32 (the mel filterbank /
window, the LSTM prediction net, the joint output projection, batch_norm running
stats, conv kernels, embeddings, all norms and biases, pos_bias) stays F32 -- see
``should_quantize`` and ``docs/quantization.md``.

See ``docs/conversion.md`` for the full schema.
"""
import argparse
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    from nemo.collections.asr.models import ASRModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'nemo_toolkit[asr]': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def _get(cfg, key, default=None):
    """Read ``key`` from an OmegaConf node or plain object, tolerating both."""
    try:
        return cfg[key]
    except Exception:
        return getattr(cfg, key, default)


def detect_arch(m):
    """Map a NeMo model to one of ctc/rnnt/tdt/hybrid_rnnt_ctc/hybrid_tdt_ctc."""
    cfg = m.cfg
    # An aux_ctc *config* block is necessary but not sufficient for a hybrid
    # model: prompt-conditioned RNNT checkpoints (nemotron) carry an unconfigured
    # aux_ctc stub (num_classes=-1, empty vocabulary) but NO ctc decoder and zero
    # ctc_decoder.* weights -- NeMo initializes them RNNT-only. Require an actual
    # ctc_decoder on the model (the same module the engine loads ctc_decoder.*
    # tensors from) before classifying as hybrid; otherwise fall through to the
    # rnnt/tdt detection below.
    has_ctc = getattr(m, "ctc_decoder", None) is not None
    if _get(cfg, "aux_ctc") is not None and has_ctc:
        loss = _get(_get(cfg, "loss", {}) or {}, "loss_name", "")
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        return "hybrid_tdt_ctc" if (loss == "tdt" or durs) else "hybrid_rnnt_ctc"
    if _get(cfg, "joint") is not None:
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        nxo = _get(_get(cfg, "joint", {}) or {}, "num_extra_outputs", 0)
        return "tdt" if (durs or (nxo and nxo > 0)) else "rnnt"
    return "ctc"


def prompt_config(cfg):
    """Return (present, num_prompts, dict_keys, dict_vals, default_lang) for a
    prompt-conditioned model, or (False, 0, [], [], "") otherwise. The prompt
    feature lives under cfg.model_defaults (initialize_prompt_feature +
    prompt_dictionary); the projection weights (prompt_kernel.*) are written
    verbatim by the generic tensor loop, so only the KV metadata is new here."""
    md = _get(cfg, "model_defaults", {}) or {}
    if not bool(_get(md, "initialize_prompt_feature", False)):
        return False, 0, [], [], ""
    pdict = _get(md, "prompt_dictionary", None)
    if not pdict:
        return False, 0, [], [], ""
    num = int(_get(md, "num_prompts", 128))
    keys = [str(k) for k in pdict.keys()]
    vals = [int(pdict[k]) for k in pdict.keys()]
    default_lang = "auto" if "auto" in pdict else keys[0]
    return True, num, keys, vals, default_lang


# ---------------------------------------------------------------------------
# Quantization policy.
#
# The C++ engine only tolerates a non-F32 weight when that weight is fed
# *directly* into ``ggml_mul_mat`` (ggml dequantizes f16/q8_0 src0 on the fly).
# Every other weight is read by hand-rolled C++ as a raw ``float*`` (mel
# filterbank/window, LSTM prediction net, joint output projection, batch_norm
# stats, embeddings), or is reshaped/transposed before the matmul in a way that
# does not survive block-quantized storage (the CTC head is stored [1, d, V] and
# squeezed in-graph; conv pointwise weights are reshaped from [1, in, out]).
# Those MUST stay F32 or the engine produces garbage.
#
# Allowlist of weights that are passed verbatim to ggml_mul_mat (see the audit in
# docs/quantization.md). Names are matched after the verbatim NeMo state_dict
# name; "N" is any layer index.
_QUANTIZABLE_PATTERNS = [
    # Conformer feed-forward modules: linear1 (d->ff) and linear2 (ff->d).
    r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$",
    # Conformer self-attention projections q/k/v/out/pos.
    r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$",
    # Subsampling output projection (Linear C*F' -> d_model), fed straight to
    # ggml_mul_mat in subsampling.cpp with no reshape.
    r"^encoder\.pre_encode\.out\.weight$",
    # Joint enc/pred projections (ggml_mul_mat in joint.cpp). NOTE: the joint
    # OUTPUT projection joint.joint_net.2.weight is read as a raw float* and
    # stays F32 -- it is intentionally NOT in this allowlist.
    r"^joint\.enc\.weight$",
    r"^joint\.pred\.weight$",
]
_QUANTIZABLE_RE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]


def should_quantize(name, shape, dtype):
    """Return the ggml quantization type for ``name`` given the requested dtype.

    ``shape`` is the ggml ``ne`` (reverse of the torch shape), so ``shape[0]`` is
    the contraction / leading dimension -- the axis q8_0 blocks along (block
    size 32). Returns ``None`` (keep F32) unless the tensor is on the linear-
    weight allowlist, is at least 2-D with both dims >= 32, and (for q8_0) has a
    leading dimension divisible by the 32-element block size.
    """
    if dtype == "f32":
        return None
    if not any(rx.match(name) for rx in _QUANTIZABLE_RE):
        return None
    if len(shape) < 2 or shape[0] < 32 or shape[1] < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0":
        if shape[0] % 32 != 0:
            return None  # leading dim not block-aligned -> keep F32
        return gguf.GGMLQuantizationType.Q8_0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dtype",
        choices=["f32", "f16", "q8_0"],
        default="f32",
        help="quantization for allowlisted linear weights (everything else f32)",
    )
    args = ap.parse_args()

    is_local = pathlib.Path(args.model).exists()
    try:
        if is_local:
            m = ASRModel.restore_from(args.model, map_location="cpu")
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    m.eval()

    arch = detect_arch(m)
    cfg = m.cfg
    enc = cfg.encoder
    feat = m.preprocessor.featurizer  # effective runtime values live here

    w = gguf.GGUFWriter(args.output, "parakeet")
    w.add_string("general.name", args.model)
    w.add_string("parakeet.arch", arch)

    # encoder
    w.add_uint32("parakeet.encoder.feat_in", int(_get(enc, "feat_in")))
    w.add_uint32("parakeet.encoder.d_model", int(_get(enc, "d_model")))
    w.add_uint32("parakeet.encoder.n_layers", int(_get(enc, "n_layers")))
    w.add_uint32("parakeet.encoder.n_heads", int(_get(enc, "n_heads")))
    ffx = int(_get(enc, "ff_expansion_factor", 4))
    w.add_uint32("parakeet.encoder.ff_dim", int(_get(enc, "d_model")) * ffx)
    w.add_uint32("parakeet.encoder.conv_kernel", int(_get(enc, "conv_kernel_size")))
    w.add_string("parakeet.encoder.conv_norm_type",
                 str(_get(enc, "conv_norm_type", "batch_norm")))
    w.add_uint32("parakeet.encoder.subsampling_factor",
                 int(_get(enc, "subsampling_factor")))
    w.add_uint32("parakeet.encoder.subsampling_conv_channels",
                 int(_get(enc, "subsampling_conv_channels")))
    w.add_bool("parakeet.encoder.xscaling", bool(_get(enc, "xscaling", True)))
    w.add_uint32("parakeet.encoder.pos_emb_max_len",
                 int(_get(enc, "pos_emb_max_len", 5000)))

    # encoder bias flag (use_bias=False checkpoints omit the linear biases; the
    # C++ loader reads them with clone_weight_opt and tolerates absence).
    w.add_bool("parakeet.encoder.use_bias", bool(_get(enc, "use_bias", True)))

    # --- Prompt conditioning (multilingual nemotron) ------------------------
    # Orthogonal capability flag (like streaming.present). When present, the C++
    # engine inserts the prompt_kernel (Linear->ReLU->Linear) on the encoder
    # output, selected by a one-hot language vector resolved from target_lang.
    p_present, p_num, p_keys, p_vals, p_default = prompt_config(cfg)
    if p_present:
        w.add_bool("parakeet.prompt.present", True)
        w.add_uint32("parakeet.prompt.num_prompts", p_num)
        w.add_array("parakeet.prompt.dictionary.keys", p_keys)
        w.add_array("parakeet.prompt.dictionary.values", p_vals)
        w.add_string("parakeet.prompt.default_lang", p_default)

    # --- Cache-aware streaming / causal config (Phase 5) ---------------------
    # These KVs describe the chunked-limited attention + causal conv that the
    # streaming FastConformer (e.g. parakeet_realtime_eou_120m-v1) uses. They are
    # emitted ONLY for streaming models (att_context_style != "regular") so that
    # offline checkpoints continue to convert byte-identically; the C++ loader
    # supplies offline-safe defaults (style "regular", causal flags false,
    # streaming block absent) when these keys are missing.
    att_style = str(_get(enc, "att_context_style", "regular"))
    is_streaming = att_style != "regular"
    if is_streaming:
        # att_context_size = [left, right]; streaming models use finite values
        # (e.g. [70, 1]) while offline models use [-1, -1]. Stored as signed
        # int32 so the -1 sentinel survives if a streaming model ever uses it;
        # the loader reads them as int32 and defaults to -1 when absent.
        att_ctx = _get(enc, "att_context_size", [-1, -1]) or [-1, -1]
        # Multi-context models store a LIST of [left,right] presets; the default
        # is the first (NeMo's default att_context_size index). A flat [l,r]
        # (older streaming models like the eou) is used as-is. The first element
        # being a non-scalar (list/tuple/OmegaConf ListConfig) marks the nested
        # form -- detect it by "not a plain number" rather than an exact type so
        # OmegaConf's ListConfig is handled too.
        if att_ctx and not isinstance(att_ctx[0], (int, float)):
            presets = [[int(x) for x in p] for p in att_ctx]
            att_left, att_right = presets[0][0], presets[0][1]
            # Record all presets so a future latency knob can pick another.
            w.add_array("parakeet.encoder.att_context_presets",
                        [int(v) for p in presets for v in p])  # flattened [l,r,l,r,...]
        else:
            att_ctx = [int(x) for x in att_ctx]
            att_left = att_ctx[0] if len(att_ctx) > 0 else -1
            att_right = att_ctx[1] if len(att_ctx) > 1 else -1
        w.add_int32("parakeet.encoder.att_context_left", int(att_left))
        w.add_int32("parakeet.encoder.att_context_right", int(att_right))
        w.add_string("parakeet.encoder.att_context_style", att_style)
        w.add_bool("parakeet.encoder.causal_downsampling",
                   bool(_get(enc, "causal_downsampling", False)))
        # conv_context_size == "causal" (a string) means the depthwise conv uses
        # left-only padding; a list of two ints means symmetric/explicit padding.
        conv_ctx = _get(enc, "conv_context_size", None)
        conv_causal = isinstance(conv_ctx, str) and conv_ctx == "causal"
        w.add_bool("parakeet.encoder.conv_causal", bool(conv_causal))

        # Streaming params read straight off the live encoder's streaming_cfg
        # (populated by setup_streaming_params() in __init__). List fields
        # (chunk_size/shift_size/pre_encode_cache_size) are emitted as int32
        # arrays; scalar fields as int32. Verified field names against
        # CacheAwareStreamingConfig in models/configs/asr_models_config.py.
        m.encoder.setup_streaming_params()
        sc = m.encoder.streaming_cfg

        def _int_list(v):
            return [int(x) for x in (v if isinstance(v, (list, tuple)) else [v])]

        w.add_array("parakeet.streaming.chunk_size", _int_list(sc.chunk_size))
        w.add_array("parakeet.streaming.shift_size", _int_list(sc.shift_size))
        w.add_int32("parakeet.streaming.cache_drop_size", int(sc.cache_drop_size))
        w.add_int32("parakeet.streaming.last_channel_cache_size",
                    int(sc.last_channel_cache_size))
        w.add_int32("parakeet.streaming.valid_out_len", int(sc.valid_out_len))
        w.add_array("parakeet.streaming.pre_encode_cache_size",
                    _int_list(sc.pre_encode_cache_size))
        w.add_int32("parakeet.streaming.drop_extra_pre_encoded",
                    int(sc.drop_extra_pre_encoded))

    # preprocessor (effective values off the featurizer object)
    w.add_uint32("parakeet.preprocessor.sample_rate",
                 int(getattr(feat, "sample_rate", 16000)))
    w.add_uint32("parakeet.preprocessor.n_mels", int(getattr(feat, "nfilt")))
    w.add_uint32("parakeet.preprocessor.n_fft", int(getattr(feat, "n_fft")))
    w.add_uint32("parakeet.preprocessor.win_length", int(getattr(feat, "win_length")))
    w.add_uint32("parakeet.preprocessor.hop_length", int(getattr(feat, "hop_length")))
    pre = getattr(feat, "preemph", None)
    w.add_float32("parakeet.preprocessor.preemph", float(pre) if pre is not None else 0.0)
    w.add_float32("parakeet.preprocessor.mag_power",
                  float(getattr(feat, "mag_power", 2.0)))
    w.add_string("parakeet.preprocessor.normalize",
                 str(getattr(feat, "normalize", "per_feature")))
    lzg = getattr(feat, "log_zero_guard_value", None)
    w.add_float32("parakeet.preprocessor.log_zero_guard",
                  float(lzg) if isinstance(lzg, (int, float)) else 2 ** -24)

    # vocab / tokenizer
    vocab = int(m.tokenizer.vocab_size)
    w.add_uint32("parakeet.vocab_size", vocab)
    w.add_uint32("parakeet.blank_id", vocab)  # blank always == vocab_size
    pieces = [m.tokenizer.ids_to_tokens([i])[0] for i in range(vocab)]
    w.add_array("parakeet.tokenizer.pieces", [str(p) for p in pieces])

    # transducer config
    if arch in ("rnnt", "tdt", "hybrid_rnnt_ctc", "hybrid_tdt_ctc"):
        prednet = _get(cfg.decoder, "prednet", {}) or {}
        w.add_uint32("parakeet.decoder.pred_hidden", int(_get(prednet, "pred_hidden")))
        w.add_uint32("parakeet.decoder.pred_rnn_layers",
                     int(_get(prednet, "pred_rnn_layers", 1)))
        jn = _get(cfg.joint, "jointnet", {}) or {}
        w.add_uint32("parakeet.joint.joint_hidden", int(_get(jn, "joint_hidden")))
        w.add_string("parakeet.joint.activation", str(_get(jn, "activation", "relu")))
        # Greedy max symbols emitted per frame (NeMo decoding.greedy.max_symbols;
        # default 10). Emitted so the C++ decoder honors a model's own value
        # instead of a hardcoded literal.
        greedy = _get(_get(cfg, "decoding", {}) or {}, "greedy", {}) or {}
        max_sym = _get(greedy, "max_symbols", _get(greedy, "max_symbols_per_step", 10))
        w.add_uint32("parakeet.decoding.max_symbols", int(max_sym) if max_sym is not None else 10)
    if arch in ("tdt", "hybrid_tdt_ctc"):
        durs = (_get(_get(cfg, "decoding", {}) or {}, "durations")
                or _get(_get(cfg, "model_defaults", {}) or {}, "tdt_durations"))
        if not durs:
            raise ValueError(
                f"arch={arch} requires TDT durations but none found in "
                "cfg.decoding.durations or cfg.model_defaults.tdt_durations"
            )
        w.add_array("parakeet.tdt.durations", [int(d) for d in durs])

    # tensors: verbatim names. Allowlisted linear weights are quantized per
    # --dtype (ggml dequantizes them on the fly inside ggml_mul_mat); everything
    # else stays f32. Include featurizer buffers explicitly.
    sd = m.state_dict()
    written = 0
    quantized = 0
    keep_buffers = {"preprocessor.featurizer.fb", "preprocessor.featurizer.window"}
    for name, t in sd.items():
        if name.startswith("preprocessor.") and name not in keep_buffers:
            continue  # skip preprocessor internals except fb/window
        if not hasattr(t, "detach"):
            continue
        arr = t.detach().cpu().float().numpy()
        if arr.ndim == 0:
            continue  # skip scalar bookkeeping (e.g. num_batches_tracked)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        # ggml ne is the reverse of the numpy/torch shape; ne[0] is the leading
        # (contraction) axis q8_0 blocks along.
        ggml_ne = list(arr.shape[::-1])
        qtype = should_quantize(name, ggml_ne, args.dtype)
        if qtype is None:
            w.add_tensor(name, arr)
        else:
            raw = gguf.quantize(arr, qtype)
            # gguf expects raw_shape to be the *byte* shape of the quantized
            # buffer; it derives the element shape from it via raw_dtype.
            w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qtype)
            quantized += 1
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: arch={arch} vocab={vocab} tensors={written} "
        f"dtype={args.dtype} quantized={quantized}"
    )


if __name__ == "__main__":
    main()
