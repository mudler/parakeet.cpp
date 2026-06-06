#!/usr/bin/env python3
"""Benchmark the prompt-conditioned nemotron model: NeMo (PyTorch CPU) vs our
C++/ggml engine, on one clip at one target language.

Mirrors scripts/benchmark.py methodology but for a LOCAL .nemo + GGUF pair that
the HF-id-keyed benchmark.py does not cover:

  * ours  : ``parakeet-cli bench --decoder tdt --lang <L> --threads T`` over a
            manifest that repeats the clip ``--passes`` times. bench loads the
            model once, warms up once (untimed), then times each transcribe with
            ``transcribe_pcm`` only (load excluded). We take the MEDIAN per-pass
            proc time, the same as the median latency benchmark.py reports.
  * NeMo  : load the .nemo once, run the SAME prompt branch as NeMo's forward()
            (preprocessor -> encoder -> cat(one-hot prompt) -> prompt_kernel ->
            rnnt greedy via ``decoding.rnnt_decoder_predictions_tensor``). Warm
            up once (untimed), then time ``--passes`` forward+decode passes and
            take the median. This is the authoritative offline path from
            ``gen_nemo_baseline.compute_prompt_reference`` (the lhotse
            transcribe(target_lang=...) dataloader needs per-cut language
            metadata our bare wav fixtures lack).

RTFx = audio_sec / median_proc_sec (higher = faster). Speedup = ours / NeMo.
Run under the NeMo venv python so ``import nemo`` works; it shells out to the
C++ CLI for the ours pass.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from asr_metrics import wer  # noqa: E402
from gen_nemo_baseline import resolve_prompt_lang  # noqa: E402


def run_ours(cli: Path, gguf: Path, wav: Path, lang: str, threads: int,
             passes: int) -> dict:
    """parakeet-cli bench over a manifest that repeats ``wav`` ``passes`` times.

    Returns {median_proc_s, audio_sec, text, load_ms, all_proc_s}.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as mf:
        for _ in range(passes):
            mf.write(f"{wav}\n")
        manifest = Path(mf.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
        out_json = Path(jf.name)
    try:
        cmd = [
            str(cli), "bench",
            "--model", str(gguf),
            "--manifest", str(manifest),
            "--decoder", "tdt",
            "--lang", lang,
            "--threads", str(threads),
            "--json", str(out_json),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"ours pass failed (rc={res.returncode})\ncmd: {' '.join(cmd)}\n"
                f"stderr:\n{res.stderr[-2000:]}")
        doc = json.loads(out_json.read_text())
        files = doc["files"]
        proc_s = [f["proc_ms"] / 1000.0 for f in files]
        return {
            "median_proc_s": statistics.median(proc_s),
            "audio_sec": files[0]["audio_sec"],
            "text": files[0]["text"],
            "load_ms": doc.get("load_ms"),
            "all_proc_s": proc_s,
        }
    finally:
        manifest.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)


def run_ours_stream(cli: Path, gguf: Path, wav: Path, lang: str, threads: int,
                    load_s: float, passes: int) -> dict:
    """Time the cache-aware STREAMING path. It has no built-in proc timer, so we
    time the whole ``transcribe --stream`` invocation (median wall over a few
    runs after one warmup) and subtract the measured one-time model load to get a
    compute-only estimate. Streaming is latency-oriented (many small chunked
    forward passes), so this RTFx is far below the offline number by design.
    """
    def one_wall() -> float:
        t = time.perf_counter()
        res = subprocess.run(
            [str(cli), "transcribe", "--model", str(gguf), "--input", str(wav),
             "--lang", lang, "--stream", "--threads", str(threads)],
            capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"stream run failed:\n{res.stderr[-1500:]}")
        return time.perf_counter() - t

    one_wall()  # warmup
    walls = [one_wall() for _ in range(passes)]
    median_wall_s = statistics.median(walls)
    compute_s = max(median_wall_s - load_s, 1e-9)
    return {
        "median_wall_s": median_wall_s,
        "compute_s": compute_s,
        "load_s": load_s,
    }


def run_nemo(nemo_path: Path, wav: Path, lang: str, threads: int,
             passes: int) -> dict:
    """Load the .nemo once and time ``passes`` offline prompt forward+decode runs.

    Returns {median_proc_s, audio_sec, text, load_s, nemo_version, all_proc_s}.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import nemo
    from nemo.collections.asr.models import ASRModel

    torch.set_num_threads(threads)
    nemo_version = getattr(nemo, "__version__", "unknown")

    t0 = time.perf_counter()
    m = ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    load_s = time.perf_counter() - t0

    target_lang, pidx, num_prompts = resolve_prompt_lang(m, lang)

    wav_np, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"expected 16k mono, got sr={sr}")
    audio_sec = len(wav_np) / 16000.0
    wt = torch.from_numpy(np.ascontiguousarray(wav_np)).float().unsqueeze(0)
    lt = torch.tensor([wt.shape[1]], dtype=torch.int64)

    def forward_decode() -> str:
        with torch.no_grad():
            feats, flen = m.preprocessor(input_signal=wt, length=lt)
            enc, elen = m.encoder(audio_signal=feats, length=flen)   # [1, D, T]
            encoded = enc.transpose(1, 2)                            # [1, T, D]
            T = encoded.shape[1]
            onehot = torch.zeros(1, T, num_prompts, dtype=encoded.dtype)
            onehot[:, :, pidx] = 1.0
            concat = torch.cat([encoded, onehot], dim=-1)            # [1, T, D+P]
            pk_out = m.prompt_kernel(concat)                         # [1, T, D]
            pk_enc = pk_out.transpose(1, 2).contiguous()            # [1, D, T]
            hyps = m.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=pk_enc, encoded_lengths=elen,
                return_hypotheses=True)
        first = hyps[0] if isinstance(hyps, list) else hyps
        if isinstance(first, list):
            first = first[0]
        return first.text if hasattr(first, "text") else str(first)

    # Warm up once (untimed), then time `passes` forward+decode runs.
    text = forward_decode()
    proc_s = []
    for _ in range(passes):
        t = time.perf_counter()
        text = forward_decode()
        proc_s.append(time.perf_counter() - t)

    return {
        "median_proc_s": statistics.median(proc_s),
        "audio_sec": audio_sec,
        "text": text,
        "load_s": load_s,
        "nemo_version": nemo_version,
        "target_lang": target_lang,
        "all_proc_s": proc_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nemo", required=True, help="local .nemo checkpoint")
    ap.add_argument(
        "--gguf", action="append", required=True, metavar="DTYPE=PATH",
        help="dtype-labelled GGUF, e.g. f32=/tmp/nemotron.gguf (repeatable)")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--cli", default=str(
        REPO / "build" / "examples" / "cli" / "parakeet-cli"))
    ap.add_argument("--stream-dtype", default="",
                    help="dtype key to also measure on the streaming path "
                         "(e.g. f32); empty -> skip streaming")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cli = Path(args.cli)
    wav = Path(args.wav)
    ggufs: dict[str, Path] = {}
    for spec in args.gguf:
        dtype, _, path = spec.partition("=")
        ggufs[dtype] = Path(path)

    print(f"=== nemotron bench: lang={args.lang} threads={args.threads} "
          f"passes={args.passes} clip={wav.name} ===", flush=True)

    nemo = run_nemo(Path(args.nemo), wav, args.lang, args.threads, args.passes)
    nemo_rtfx = nemo["audio_sec"] / nemo["median_proc_s"]
    print(f"[nemo {nemo['nemo_version']}] lang={nemo['target_lang']} "
          f"median_proc={nemo['median_proc_s']*1000:.1f}ms RTFx={nemo_rtfx:.2f}",
          flush=True)

    ours: dict[str, dict] = {}
    for dtype, gguf in ggufs.items():
        o = run_ours(cli, gguf, wav, args.lang, args.threads, args.passes)
        rtfx = o["audio_sec"] / o["median_proc_s"]
        agree = wer(nemo["text"], o["text"])
        ours[dtype] = {**o, "rtfx": rtfx, "speedup": rtfx / nemo_rtfx,
                       "agreement_wer": agree}
        print(f"[ours {dtype}] median_proc={o['median_proc_s']*1000:.1f}ms "
              f"RTFx={rtfx:.2f} speedup={rtfx/nemo_rtfx:.2f}x "
              f"agreement_WER={agree:.4f}", flush=True)

    stream = None
    if args.stream_dtype and args.stream_dtype in ggufs:
        d = args.stream_dtype
        s = run_ours_stream(cli, ggufs[d], wav, args.lang, args.threads,
                            ours[d]["load_ms"] / 1000.0, args.passes)
        s_rtfx = nemo["audio_sec"] / s["compute_s"]
        s_wall_rtfx = nemo["audio_sec"] / s["median_wall_s"]
        stream = {"dtype": d, "compute_rtfx": s_rtfx, "wall_rtfx": s_wall_rtfx,
                  **s}
        print(f"[ours stream {d}] median_wall={s['median_wall_s']*1000:.1f}ms "
              f"compute_RTFx={s_rtfx:.2f} (wall_RTFx={s_wall_rtfx:.2f})",
              flush=True)

    doc = {
        "clip": wav.name,
        "audio_sec": nemo["audio_sec"],
        "lang": args.lang,
        "threads": args.threads,
        "passes": args.passes,
        "nemo": {"rtfx": nemo_rtfx, "median_proc_s": nemo["median_proc_s"],
                 "text": nemo["text"], "version": nemo["nemo_version"],
                 "load_s": nemo["load_s"]},
        "ours": {d: {"rtfx": v["rtfx"], "speedup": v["speedup"],
                     "median_proc_s": v["median_proc_s"],
                     "agreement_wer": v["agreement_wer"], "text": v["text"],
                     "load_ms": v["load_ms"]}
                 for d, v in ours.items()},
    }
    if stream is not None:
        doc["stream"] = stream
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2))
        print(f"-> wrote {args.out}", flush=True)
    else:
        print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
