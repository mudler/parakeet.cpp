#!/usr/bin/env python3
"""Reference tensors for the Nemotron 3.5 Transcribe audio side, from Hugging Face transformers.

Runs `Nemotron3_5TranscribeAudioModel` (fp32, CPU) and the adapter on one WAV and writes the
stage outputs as an F32 baseline GGUF (the format tests/parity.hpp reads):

  mel            [T_valid, 128]   raw log-mel (valid frames only)
  asr_hidden     [N, 1280]        chunked ASR encoder output (before asr_norm)
  speaker_probs  [N, 8]           speaker probabilities, mean over the 8 sub-frames
  fused          [N, 1280]        asr_norm(hidden) + speaker term
  adapter        [N, 2688]        proj(fused), what the LLM receives
  (N = ceil(T_valid / 8))

Needs: torch, transformers==5.22.0, librosa, soundfile, gguf, numpy; and the checkpoint's
perception.* tensors in a local safetensors file (the same file convert_nemotron_speechlm_audio.py
reads; only that byte range is touched) plus config.json and processor_config.json.

  gen_speechlm_audio_baseline.py --safetensors model.safetensors --config config.json \\
      --processor-config processor_config.json --wav tests/fixtures/clip.wav --output ref.gguf
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_nemotron_speechlm_audio import Source  # noqa: E402

from transformers import AutoConfig  # noqa: E402
from transformers.models.nemotron3_5_transcribe.modeling_nemotron3_5_transcribe import (  # noqa: E402
    Nemotron3_5TranscribeAudioModel, normalize_per_feature)
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (  # noqa: E402
    NemotronAsrStreamingFeatureExtractor)

DIAR = r"^perception\.encoder\.diarization_model\.sortformer_modules\."
# Checkpoint (NeMo) names -> transformers names; the same table as transformers' conversion mapping.
RULES = [
    (r"^perception\.encoder\.asr_encoder\.", "asr_encoder."),
    (r"^perception\.encoder\.diarization_model\.encoder\.", "diarization_model.model.audio_tower."),
    (DIAR + r"encoder_proj\.", "diarization_model.model.proj."),
    (DIAR + r"subpixel_upsample\.", "diarization_model.model.upsampler.conv."),
    (DIAR + r"first_hidden_to_hidden\.", "diarization_model.classifier.dense."),
    (DIAR + r"single_hidden_to_spks\.", "diarization_model.classifier.out_proj."),
    (DIAR + r"learnable_sil_emb", "diarization_model.silence_embeds"),
    (r"^perception\.encoder\.asr_norm\.", "asr_norm."),
    (r"^perception\.encoder\.diar_norm\.", "speaker_norm."),
    (r"\.pre_encode\.proj\.", ".embedder.projection."),
    (r"\.embed_norm\.", ".input_layer_norm."),
    (r"\.final_norm\.", ".layer_norm."),
    (r"\.norm1\.", ".layer_norm1."),
    (r"\.norm2\.", ".layer_norm2."),
    (r"\.attn\.out_proj\.", ".self_attn.o_proj."),
    (r"\.attn\.q_norm\.", ".self_attn.q_norm."),
    (r"\.attn\.k_norm\.", ".self_attn.k_norm."),
    (r"\.ffn\.net\.0\.", ".mlp.fc1."),
    (r"\.ffn\.net\.3\.", ".mlp.fc2."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--processor-config", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stages", default="mel,asr_hidden,speaker_probs,fused,adapter",
                    help="comma list of tensors to store")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = AutoConfig.from_pretrained(os.path.dirname(os.path.abspath(args.config)))
    acfg = cfg.audio_config
    model = Nemotron3_5TranscribeAudioModel(acfg).eval().float()
    proj = torch.nn.Linear(acfg.hidden_size, cfg.text_config.hidden_size)

    st = Source(path=args.safetensors)
    sd = {}
    for name in st.names("perception."):
        if ".preprocessor." in name or name.startswith("perception.preprocessor."):
            continue
        if name.endswith("sortformer_modules.hidden_to_spks.weight") or name.endswith("sortformer_modules.hidden_to_spks.bias"):
            continue   # unused head
        t = torch.from_numpy(st.read(name))
        if name.startswith("perception.proj."):
            continue
        k = name
        for pat, rep in RULES:
            k = re.sub(pat, rep, k)
        if k.endswith(".attn.w_qkv.weight"):
            b = k[:-len(".attn.w_qkv.weight")]
            q, kk, v = t.chunk(3, 0)
            sd[b + ".self_attn.q_proj.weight"], sd[b + ".self_attn.k_proj.weight"], sd[b + ".self_attn.v_proj.weight"] = q, kk, v
        else:
            sd[k] = t
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit(f"state dict mismatch: missing={res.missing_keys} unexpected={res.unexpected_keys}")
    proj.weight.data = torch.from_numpy(st.read("perception.proj.weight"))
    proj.bias.data = torch.from_numpy(st.read("perception.proj.bias"))

    pc = json.load(open(args.processor_config))["feature_extractor"]
    pc.pop("feature_extractor_type", None)
    fe = NemotronAsrStreamingFeatureExtractor(**pc)
    audio, sr = sf.read(args.wav, dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise SystemExit("the WAV must be 16 kHz mono")
    feat = fe(torch.from_numpy(audio), sampling_rate=16000, return_tensors="pt")
    x, m = feat["input_features"], feat["attention_mask"]
    pad = -x.shape[1] % 8
    x = torch.nn.functional.pad(x, (0, 0, 0, pad))     # what the processor does
    m = torch.nn.functional.pad(m, (0, pad))
    valid = int(m.sum())
    n_tok = (valid + 7) // 8
    with torch.no_grad():
        o = model(x, input_features_mask=m)
        adapter = proj(o.last_hidden_state)
    p80 = o.speaker_logits[0].sigmoid().unflatten(0, (-1, 8)).mean(1)
    out = {
        "mel": x[0, :valid].numpy(),
        "asr_hidden": o.asr_last_hidden_state[0, :n_tok].numpy(),
        "speaker_probs": p80[:n_tok].numpy(),
        "fused": o.last_hidden_state[0, :n_tok].numpy(),
        "adapter": adapter[0, :n_tok].numpy(),
    }
    w = gguf.GGUFWriter(args.output, "parakeet-baseline")
    w.add_string("general.name", "speechlm-audio-baseline")
    w.add_string("baseline.source", "transformers " + __import__("transformers").__version__ + " fp32 cpu")
    w.add_uint32("baseline.valid_mel_frames", valid)
    for k in args.stages.split(","):
        w.add_tensor(k, np.ascontiguousarray(out[k], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: valid_mel={valid} tokens={n_tok}")


if __name__ == "__main__":
    main()
