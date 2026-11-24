#!/usr/bin/env python3
"""Convert the audio side of nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B to GGUF.

The checkpoint is one 67 GB safetensors file. Only the `perception.*` tensors (about
1.5 GB, one contiguous byte range at the end of the file) are needed, so this script
reads just those: from a local file, or by HTTP range requests from the Hub. Nothing
else is downloaded and no torch / NeMo / transformers is needed.

It writes two GGUF files (the LLM is not converted):

  <out>/<name>-audio.gguf   arch speechlm_audio: mel constants, 32-layer ASR encoder,
                            speaker fusion norms, adapter projection
  <out>/<name>-diar.gguf    arch diarization: the Sortformer speaker encoder and head

pk::SpeechLmAudio loads them together (docs/speechlm-audio.md). They can also be put
in one bundle GGUF (docs/bundle.md) as components of kind speechlm_audio and diar.

  convert_nemotron_speechlm_audio.py --safetensors model.safetensors --config config.json --output-dir out/
  convert_nemotron_speechlm_audio.py --hf nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --output-dir out/

--dtype f16 / q8_0 stores the large linear weights in that type; norms, biases, the QK
norms and the mel constants stay F32.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import urllib.request

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    print("pip install gguf", file=sys.stderr)
    raise

HF_REPO = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"

# Large linear weights that --dtype may quantize (ggml ne[0] must be a multiple of 32 for q8_0).
QUANTIZABLE = [re.compile(p) for p in (
    r"^encoder\.layers\.\d+\.attn\.(w_qkv|out_proj)\.weight$",
    r"^encoder\.layers\.\d+\.ffn\.net\.\d+\.weight$",
    r"^encoder\.pre_encode\.proj\.weight$",
    r"^speechlm\.proj\.weight$",
)]


class Source:
    """Reads tensors of a safetensors file, locally or over HTTP ranges."""

    def __init__(self, path=None, url=None):
        self.path, self.url = path, url
        head = self._read(0, 8)
        (n,) = struct.unpack("<Q", head)
        self.base = 8 + n
        self.header = json.loads(self._read(8, n))

    def _read(self, off, n):
        if self.path:
            with open(self.path, "rb") as f:
                f.seek(off)
                return f.read(n)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + n - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def names(self, prefix):
        return sorted(k for k in self.header if k != "__metadata__" and k.startswith(prefix))

    def read(self, name):
        m = self.header[name]
        s, e = m["data_offsets"]
        raw = self._read(self.base + s, e - s)
        if m["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            a = u.view(np.float32)
        elif m["dtype"] == "F32":
            a = np.frombuffer(raw, dtype="<f4")
        elif m["dtype"] == "F16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise SystemExit(f"unsupported dtype {m['dtype']} for {name}")
        return np.ascontiguousarray(a.reshape(m["shape"]), dtype=np.float32)


def quant_type(name, shape, dtype):
    if dtype == "f32" or not any(rx.match(name) for rx in QUANTIZABLE):
        return None
    ne = list(shape[::-1])
    if len(ne) < 2 or ne[0] < 32 or ne[1] < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0" and ne[0] % 32 == 0:
        return gguf.GGMLQuantizationType.Q8_0
    return None


def write_tensors(w, items, dtype):
    n = q = 0
    for name, arr in items:
        qt = quant_type(name, arr.shape, dtype)
        if qt is None:
            w.add_tensor(name, arr)
        else:
            raw = gguf.quantize(arr, qt)
            w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qt)
            q += 1
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n, q


def pre_kvs(w, n_mels, n_fft, win, hop):
    w.add_uint32("parakeet.preprocessor.sample_rate", 16000)
    w.add_uint32("parakeet.preprocessor.n_mels", n_mels)
    w.add_uint32("parakeet.preprocessor.n_fft", n_fft)
    w.add_uint32("parakeet.preprocessor.win_length", win)
    w.add_uint32("parakeet.preprocessor.hop_length", hop)
    w.add_float32("parakeet.preprocessor.preemph", 0.97)
    w.add_float32("parakeet.preprocessor.mag_power", 2.0)
    # The speaker branch needs the raw log-mel; the engine normalises the ASR branch itself.
    w.add_string("parakeet.preprocessor.normalize", "NA")
    w.add_float32("parakeet.preprocessor.log_zero_guard", 2 ** -24)


def encoder_kvs(w, d_model, n_layers, n_heads, ff_dim, sub, rope_base, rot, qk_norm):
    w.add_uint32("parakeet.encoder.feat_in", 128)
    w.add_uint32("parakeet.encoder.d_model", d_model)
    w.add_uint32("parakeet.encoder.n_layers", n_layers)
    w.add_uint32("parakeet.encoder.n_heads", n_heads)
    w.add_uint32("parakeet.encoder.ff_dim", ff_dim)
    w.add_uint32("parakeet.encoder.conv_kernel", 0)
    w.add_string("parakeet.encoder.conv_norm_type", "layer_norm")
    w.add_uint32("parakeet.encoder.subsampling_factor", sub)
    w.add_uint32("parakeet.encoder.subsampling_conv_channels", 0)
    w.add_bool("parakeet.encoder.xscaling", False)
    w.add_uint32("parakeet.encoder.pos_emb_max_len", 5000)
    w.add_bool("parakeet.encoder.use_bias", True)
    w.add_string("parakeet.encoder.self_attention_model", "rope")
    w.add_bool("parakeet.encoder.qkv_bias", False)
    w.add_bool("parakeet.encoder.pre_block_norm", True)
    w.add_float32("parakeet.encoder.rope_base", rope_base)
    w.add_float32("parakeet.encoder.rotary_fraction", rot)
    w.add_bool("parakeet.encoder.qk_norm", qk_norm)
    w.add_string("parakeet.encoder.activation", "gelu_erf")   # exact GELU, as in the checkpoint's code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--safetensors", help="local model.safetensors (only the perception.* byte range is read)")
    src.add_argument("--hf", nargs="?", const=HF_REPO, help="read the ranges from the Hub repo")
    ap.add_argument("--config", help="config.json (default: fetched from --hf, required with --safetensors)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--name", default="nemotron-3.5-transcribe-30b-a4b")
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f32")
    args = ap.parse_args()

    if args.safetensors:
        if not args.config:
            ap.error("--config is required with --safetensors")
        st = Source(path=args.safetensors)
        cfg = json.load(open(args.config))
    else:
        base = f"https://huggingface.co/{args.hf}/resolve/main/"
        st = Source(url=base + "model.safetensors")
        cfg = json.load(open(args.config)) if args.config else json.loads(urllib.request.urlopen(base + "config.json").read())

    pe = cfg["pe_encoder_config"]
    asr = pe["asr_encoder_cfg"]
    dcfg = pe["diarization_model_cfg"]
    dhf = cfg["audio_config"]["diarization_config"]
    stream = dhf["streaming_config"]
    n_spk = int(dcfg["max_num_of_spks"])
    sub = int(asr["subsampling_factor"])
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- audio GGUF ----
    path = os.path.join(args.output_dir, f"{args.name}-audio.gguf")
    w = gguf.GGUFWriter(path, "parakeet")
    w.add_string("general.name", args.name + "-audio")
    w.add_string("general.license", "OpenMDW-1.1")
    w.add_string("parakeet.arch", "speechlm_audio")
    d = int(asr["d_model"])
    encoder_kvs(w, d, int(asr["n_layers"]), int(asr["n_heads"]), d * int(asr["ff_expansion"]), sub,
                float(asr["rope_base"]), float(asr["rotary_fraction"]), bool(asr["qk_norm"]))
    pre_kvs(w, 128, 512, 400, 160)
    w.add_bool("parakeet.speechlm.present", True)
    w.add_uint32("parakeet.speechlm.llm_dim", int(cfg["perception"]["output_dim"]))
    w.add_uint32("parakeet.speechlm.num_speakers", n_spk)
    w.add_float32("parakeet.speechlm.speaker_activity_threshold",
                  float(cfg["audio_config"].get("speaker_activity_threshold", 0.5)))
    w.add_uint32("parakeet.speechlm.chunk_len", int(dhf["chunk_length"]))
    w.add_uint32("parakeet.speechlm.chunk_left_context", int(dhf["chunk_left_context"]))
    w.add_uint32("parakeet.speechlm.chunk_right_context", int(dhf["chunk_right_context"]))

    items = []
    for name in st.names("perception."):
        if name.startswith("perception.encoder.asr_encoder."):
            new = "encoder." + name[len("perception.encoder.asr_encoder."):]
        elif name.startswith("perception.encoder.asr_norm."):
            new = "speechlm.asr_norm." + name.rsplit(".", 1)[1]
        elif name.startswith("perception.encoder.diar_norm."):
            new = "speechlm.diar_norm." + name.rsplit(".", 1)[1]
        elif name.startswith("perception.proj."):
            new = "speechlm." + name[len("perception."):]
        elif name.startswith("perception.preprocessor.featurizer."):
            new = name[len("perception."):]
        else:
            continue
        items.append((new, st.read(name)))
    n, q = write_tensors(w, items, args.dtype)
    print(f"wrote {path}: tensors={n} quantized={q} dtype={args.dtype}")

    # ---- diarization GGUF ----
    path = os.path.join(args.output_dir, f"{args.name}-diar.gguf")
    enc = dcfg["encoder"]
    w = gguf.GGUFWriter(path, "parakeet")
    w.add_string("general.name", args.name + "-diar")
    w.add_string("general.license", "OpenMDW-1.1")
    w.add_string("parakeet.arch", "diarization")
    encoder_kvs(w, int(enc["d_model"]), int(enc["n_layers"]), int(enc["n_heads"]),
                int(enc["d_model"] * float(enc["ff_expansion"])), sub, 10000.0, 1.0, False)
    pre_kvs(w, 128, 512, 400, 160)
    w.add_uint32("parakeet.diar.n_speakers", n_spk)
    w.add_uint32("parakeet.diar.tf_d_model", int(dhf["head_config"]["hidden_size"]))
    w.add_uint32("parakeet.diar.upsample_factor", sub)
    w.add_float32("parakeet.diar.frame_resolution_sec", 0.01)
    w.add_float32("parakeet.diar.onset_threshold", 0.5)
    w.add_float32("parakeet.diar.offset_threshold", 0.5)
    w.add_bool("parakeet.diar.streaming_mode", True)
    # The checkpoint runs the speaker branch with the cache and FIFO of its offline mode
    # (chunk 500 frames, FIFO 40, cache update every 300 frames, cache 188 frames).
    w.add_uint32("parakeet.diar.chunk_len", int(dhf["chunk_length"]))
    w.add_uint32("parakeet.diar.spkcache_len", int(stream["speaker_cache_length"]))
    w.add_uint32("parakeet.diar.fifo_len", int(dhf["fifo_length"]))
    w.add_uint32("parakeet.diar.spkcache_update_period", int(dhf["speaker_cache_update_period"]))
    w.add_uint32("parakeet.diar.spkcache_sil_frames_per_spk", int(stream["speaker_cache_silence_frames_per_speaker"]))
    w.add_float32("parakeet.diar.pred_score_threshold", float(stream["prediction_score_threshold"]))
    w.add_float32("parakeet.diar.scores_boost_latest", float(stream["latest_frames_score_boost"]))
    w.add_float32("parakeet.diar.strong_boost_rate", float(stream["strong_boost_rate"]))
    w.add_float32("parakeet.diar.weak_boost_rate", float(stream["weak_boost_rate"]))
    w.add_float32("parakeet.diar.min_pos_scores_rate", float(stream["min_positive_scores_rate"]))
    w.add_bool("parakeet.diar.use_learnable_sil_emb", True)

    items = []
    dpre = "perception.encoder.diarization_model."
    for name in st.names(dpre):
        rest = name[len(dpre):]
        if rest.startswith("sortformer_modules.hidden_to_spks."):
            continue   # unused 384 -> 8 head
        if rest.startswith(("encoder.", "sortformer_modules.", "preprocessor.featurizer.")):
            items.append((rest, st.read(name)))
    n, q = write_tensors(w, items, args.dtype)
    print(f"wrote {path}: tensors={n} quantized={q} dtype={args.dtype}")


if __name__ == "__main__":
    main()
