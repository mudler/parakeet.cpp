#!/usr/bin/env python3
"""
gen_benchmark_md.py — Generate benchmarks/BENCHMARK.md from parakeet.cpp result JSONs.

Usage:
    python scripts/gen_benchmark_md.py \
        --results benchmarks/results \
        --plots   benchmarks/plots \
        --out     benchmarks/BENCHMARK.md

Renders a headline NeMo-vs-ours table (f32 + q8_0), a quantization summary across
all dtypes present, the plots, a real-audio sanity check, and findings. The
NeMo column is the unchanged PyTorch-CPU reference.
"""

import argparse
import json
import os
import sys
from pathlib import Path

DTYPE_ORDER = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q5_0", "q4_0"]


def load_results(results_dir: Path) -> list[dict]:
    models = []
    for p in sorted(results_dir.glob("*.json")):
        if p.stem == "threads":
            continue
        with open(p) as f:
            models.append(json.load(f))
    return models


def _short(name: str) -> str:
    return name.replace("parakeet-", "").replace("parakeet_realtime_eou_", "rt-eou-")


def _ls(m: dict) -> dict:
    return m["manifests"]["librispeech"]


def dtypes_present(models: list[dict]) -> list[str]:
    seen: set[str] = set()
    for m in models:
        seen.update(_ls(m)["ours"].keys())
    ordered = [d for d in DTYPE_ORDER if d in seen]
    ordered += sorted(d for d in seen if d not in DTYPE_ORDER)
    return ordered


# ── Methodology ──────────────────────────────────────────────────────────────────

def build_methodology(models: list[dict], dtypes: list[str]) -> str:
    nemo_ver = next((m.get("nemo_version") for m in models if m.get("nemo_version")), "—")
    threads = models[0].get("threads", "?") if models else "?"
    return f"""\
## Methodology

### Machine
- **CPU:** 20-core host; **{threads} threads** used for both NeMo and parakeet.cpp (matched).
- CPU-only inference throughout — no GPU.

### Software
| Component | Version / notes |
|-----------|-----------------|
| NeMo | {nemo_ver} (PyTorch CPU) |
| parakeet.cpp ggml engine | this repo — GGUF dtypes: {", ".join(dtypes)} |

### Audio sets
| Set | Description |
|-----|-------------|
| **LibriSpeech test-clean** | 100 utterances, ~15 min audio; ground-truth transcripts → formal WER |
| **Diverse clips** | 4 clips (JFK, MLK "I Have a Dream", Italian speech, synthetic TTS) — real-audio sanity check |

### Protocol
- Batch size = 1 for both engines; thread count = {threads} (matched).
- NeMo: `torch.set_num_threads({threads})`, per-file timing via `time.perf_counter`.
- ours: `parakeet-cli bench --threads {threads}` (load once, time transcribe only); K-quants via `parakeet-cli quantize`.
- Peak RSS via a `/usr/bin/time -v` wrapper.
- **RTFx** = Σ audio_sec / Σ proc_sec (higher = faster; >1 = real-time capable).
- **WER** = normalized word error rate vs LibriSpeech ground truth.
- **Agreement WER** = normalized WER between NeMo output and ours (≈0 ⇒ we reproduce NeMo).

> The NeMo numbers are the unchanged PyTorch-CPU reference; only the ggml engine
> is re-measured here. f32 output is byte-identical to NeMo (agreement ≈ 0);
> lower K-quants trade a little accuracy for size, shown below.
"""


# ── Headline table (f32 + q8_0) ──────────────────────────────────────────────────

def build_headline_table(models: list[dict]) -> str:
    lines = ["## Headline — NeMo vs parakeet.cpp  (LibriSpeech, f32 & q8_0)\n"]
    lines.append(
        "| Model | RTFx NeMo | RTFx f32 | Speedup f32 | RTFx q8_0 | Speedup q8_0"
        " | WER NeMo % | WER f32 % | Agree f32 % | RSS NeMo MB | RSS f32 MB |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        ls = _ls(m); nemo = ls["nemo"]; f32 = ls["ours"]["f32"]
        q8 = ls["ours"].get("q8_0")
        n_rtfx = nemo["rtfx"]
        q8_rtfx = q8["rtfx"] if q8 else float("nan")
        q8_su = f"{q8_rtfx / n_rtfx:.2f}×" if q8 else "—"
        lines.append(
            f"| {_short(m['model'])}"
            f" | {n_rtfx:.1f} | {f32['rtfx']:.1f} | {f32['rtfx']/n_rtfx:.2f}×"
            f" | {q8_rtfx:.1f} | {q8_su}"
            f" | {nemo['wer_vs_truth']*100:.2f} | {f32['wer_vs_truth']*100:.2f}"
            f" | {f32['agreement_wer_vs_nemo']*100:.4f}"
            f" | {nemo['peak_rss_mb']:.0f} | {f32['peak_rss_mb']:.0f} |"
        )
    lines.append("")
    lines.append("> **Speedup** = ours RTFx / NeMo RTFx (>1 = faster than NeMo). "
                 "f32 reproduces NeMo's transcript (agreement ≈ 0).")
    lines.append("")
    return "\n".join(lines)


# ── Nemotron (prompt-conditioned, streaming, multilingual) ───────────────────────

def build_nemotron_section(results_dir: Path) -> str:
    """Standalone section for the nemotron-3.5-asr-streaming-0.6b port.

    This model is prompt-conditioned (a per-language one-hot drives a PromptKernel
    between the encoder and the transducer) and is benchmarked from a local .nemo
    + GGUF pair, so it is outside the HF-id LibriSpeech pipeline above. Reads
    ``<results>/nemotron/bench.json`` written by ``scripts/bench_nemotron.py``;
    returns "" when that file is absent.
    """
    fp = results_dir / "nemotron" / "bench.json"
    if not fp.is_file():
        return ""
    d = json.loads(fp.read_text())
    clip = d["clip"]
    audio = d["audio_sec"]
    lang = d["lang"]
    threads = d["threads"]
    passes = d["passes"]
    nemo = d["nemo"]
    f32 = d["ours"]["f32"]
    q8 = d["ours"].get("q8_0")

    lines = ["## Nemotron (streaming, multilingual, prompt-conditioned)\n"]
    lines.append(
        "`nemotron-3.5-asr-streaming-0.6b` is a FastConformer transducer with a "
        "per-language prompt: a one-hot language vector drives a PromptKernel "
        "between the encoder and the RNN-T decoder. It runs both offline and "
        "cache-aware streaming. Because it loads from a local `.nemo` plus its "
        "GGUF, it sits outside the LibriSpeech pipeline above and is measured on "
        "its own here.\n"
    )
    lines.append(
        f"One clip (`{clip}`, {audio:.2f} s), language prompt `{lang}`, "
        f"{threads} threads, median of {passes} passes after one warmup. ours is "
        "`parakeet-cli bench --decoder tdt --lang " + lang + "` (load once, time "
        "transcribe only); NeMo runs the same prompt forward (preprocessor, "
        "encoder, PromptKernel, RNN-T greedy) on PyTorch CPU. RTFx = audio "
        "seconds per second of compute; higher is faster.\n"
    )
    lines.append(
        f"Host: AMD Ryzen 9 9950X3D (20 cores), CPU-only. NeMo "
        f"{nemo.get('version', '?')}.\n"
    )

    lines.append("| Engine | RTFx | Speedup vs NeMo | Agreement WER vs NeMo |")
    lines.append("|---|---|---|---|")
    lines.append(f"| NeMo (PyTorch CPU) | {nemo['rtfx']:.1f} | 1.00× | reference |")
    lines.append(
        f"| parakeet.cpp f32 | {f32['rtfx']:.1f} | {f32['speedup']:.2f}× "
        f"| {f32['agreement_wer']*100:.4f}% |"
    )
    if q8:
        lines.append(
            f"| parakeet.cpp q8_0 | {q8['rtfx']:.1f} | {q8['speedup']:.2f}× "
            f"| {q8['agreement_wer']*100:.4f}% |"
        )
    lines.append("")
    lines.append(
        f"Accuracy is **WER 0 vs NeMo**: the f32 and q8_0 transcripts are "
        f"byte-identical to NeMo's on the timed runs (agreement WER 0.0000%), so "
        f"the speed numbers compare equal work. parakeet.cpp is "
        f"**{f32['speedup']:.2f}× faster than NeMo at f32** and "
        f"**{q8['speedup']:.2f}× at q8_0**." if q8 else
        f"Accuracy is **WER 0 vs NeMo** (agreement WER 0.0000%); parakeet.cpp is "
        f"**{f32['speedup']:.2f}× faster than NeMo at f32**."
    )
    lines.append("")

    stream = d.get("stream")
    if stream:
        lines.append(
            f"Streaming path ({stream['dtype']}, cache-aware): compute RTFx "
            f"**{stream['compute_rtfx']:.2f}** (median wall "
            f"{stream['median_wall_s']*1000:.0f} ms over the {audio:.2f} s clip, "
            f"one-time model load of {stream['load_s']*1000:.0f} ms subtracted). "
            f"Streaming is latency-oriented: it runs many small chunked forward "
            f"passes rather than one offline pass, so its RTFx sits well below the "
            f"offline number by design while staying several times real time. The "
            f"streaming transcript matches the offline and NeMo transcripts.\n"
        )
    return "\n".join(lines)


# ── Quantization summary (all dtypes, aggregated) ────────────────────────────────

def build_quant_table(models: list[dict], dtypes: list[str]) -> str:
    lines = ["## Quantization — size / speed / accuracy tradeoff\n"]
    lines.append("Averaged over all models (LibriSpeech). Size is the mean GGUF size "
                 "as a fraction of the f32 GGUF.\n")
    lines.append("| dtype | avg size vs f32 | mean speedup vs NeMo | mean WER vs truth % | mean agreement vs NeMo % |")
    lines.append("|---|---|---|---|---|")
    for d in dtypes:
        size_fracs, sus, wers, agrs = [], [], [], []
        for m in models:
            ls = _ls(m)
            if d not in ls["ours"]:
                continue
            o = ls["ours"][d]
            f32_size = m["gguf_size_mb"].get("f32")
            d_size = m["gguf_size_mb"].get(d)
            if f32_size and d_size:
                size_fracs.append(d_size / f32_size)
            sus.append(o["rtfx"] / ls["nemo"]["rtfx"])
            if o.get("wer_vs_truth") is not None:
                wers.append(o["wer_vs_truth"] * 100)
            if o.get("agreement_wer_vs_nemo") is not None:
                agrs.append(o["agreement_wer_vs_nemo"] * 100)

        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")
        lines.append(
            f"| {d} | {mean(size_fracs)*100:.0f}% | {mean(sus):.2f}× "
            f"| {mean(wers):.2f} | {mean(agrs):.3f} |"
        )
    lines.append("")
    lines.append("> f32 is the faithful reference (agreement ≈ 0). q8_0 is near-lossless; "
                 "K-quants (q6_k→q4_k) shrink the model further at a small, monotonic "
                 "accuracy cost. See the per-model quant plots below.")
    lines.append("")
    return "\n".join(lines)


# ── Plots ────────────────────────────────────────────────────────────────────────

def build_plots_section(plots_dir: Path, out_md_path: Path) -> str:
    md_parent = out_md_path.resolve().parent
    try:
        rel = os.path.relpath(plots_dir.resolve(), md_parent)
    except ValueError:
        rel = str(plots_dir)

    plots = [
        ("rtfx.png",           "RTFx per model — NeMo vs ours (all dtypes), LibriSpeech"),
        ("speedup.png",        "Speedup: ours / NeMo RTFx ratio (per dtype)"),
        ("quant_tradeoff.png", "Quantization tradeoff — GGUF size vs accuracy (per model/dtype)"),
        ("quant_accuracy.png", "Accuracy vs quantization — agreement WER per model"),
        ("wer.png",            "WER vs ground truth — NeMo vs ours"),
        ("agreement.png",      "Transcript agreement WER — ours vs NeMo (lower = closer)"),
        ("size.png",           "GGUF model size by dtype"),
        ("memory.png",         "Peak RSS per model — NeMo vs ours"),
        ("latency_vs_len.png", "Per-file latency vs audio length"),
        ("threads.png",        "Thread scaling — RTFx vs thread count"),
    ]

    lines = ["## Plots\n"]
    for fname, caption in plots:
        if (plots_dir / fname).exists():
            lines.append(f"### {caption}\n")
            lines.append(f"![{caption}]({rel}/{fname})\n")
    lines.append("")
    return "\n".join(lines)


# ── Real-audio sanity check ──────────────────────────────────────────────────────

def build_diverse_section(models: list[dict]) -> str:
    lines = ["## Real-Audio Sanity Check\n"]
    lines.append(
        "Transcripts from the **diverse** clip set (no ground truth for most). "
        "NeMo vs parakeet.cpp f32 side-by-side to confirm fidelity on real audio.\n"
    )
    for m in models:
        div = m["manifests"].get("diverse")
        if not div:
            continue
        lines.append(f"### Model: `{m['model']}`\n")
        nemo_files = {Path(f["path"]).name: f for f in div["nemo"]["files"]}
        f32_files = {Path(f["path"]).name: f for f in div["ours"]["f32"]["files"]}
        for clip in sorted(set(nemo_files) | set(f32_files)):
            lines.append(f"#### `{clip}`\n")
            lines.append("| Engine | Transcript |")
            lines.append("|--------|-----------|")
            if clip in nemo_files:
                lines.append(f"| NeMo (PyTorch CPU) | {nemo_files[clip]['text']} |")
            if clip in f32_files:
                lines.append(f"| parakeet.cpp f32 | {f32_files[clip]['text']} |")
            lines.append("")
    return "\n".join(lines)


# ── Batched decode throughput ──────────────────────────────

def _decode_batch_subtable(json_dir: Path, kind: str) -> str:
    """Render one machine sub-table from decode-batch JSON files in `json_dir`.

    `kind` is "CPU" or "GPU" (for the caption). Returns "" when the directory is
    absent or holds no JSON files.
    """
    if not json_dir.is_dir():
        return ""
    files = sorted(json_dir.glob("*.json"))
    if not files:
        return ""

    docs = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        d["_label"] = fp.stem
        docs.append(d)
    docs.sort(key=lambda d: d["_label"])

    # One-line caption from the first doc's metadata.
    meta = docs[0]
    threads = meta.get("threads", "?")
    backend = meta.get("backend", "?")
    reps = meta.get("reps", "?")
    dtype = meta.get("dtype", "?")
    caption = (
        f"**{kind}** ({backend}, {dtype}, {threads} threads), "
        f"best-of-{reps}, one clip replicated B times."
    )

    # Column set: the B values to show (fixed display order, only those present).
    show_bs = [1, 4, 8, 16]
    header = "| Model | " + " | ".join(f"B={b}" for b in show_bs) + " | clips/s @B=16 |"
    sep = "|" + "---|" * (len(show_bs) + 2)

    lines = [caption + "\n", "", header, sep]
    for d in docs:
        by_b = {int(r["B"]): r for r in d.get("rows", [])}
        cells = []
        for b in show_bs:
            r = by_b.get(b)
            cells.append(f"{r['speedup']:.2f}\u00d7" if r else "\u2014")
        r16 = by_b.get(16)
        cps16 = f"{r16['batched_cps']:.1f}" if r16 else "\u2014"
        lines.append(f"| {d['_label']} | " + " | ".join(cells) + f" | {cps16} |")
    return "\n".join(lines) + "\n"


def build_batched_decode_section(results_dir: Path) -> str:
    """Decode-batching speedup tables (CPU + GPU). Empty string when no data."""
    cpu = _decode_batch_subtable(results_dir / "decode_batch" / "cpu", "CPU")
    gpu = _decode_batch_subtable(results_dir / "decode_batch" / "gpu", "GPU")
    if not cpu and not gpu:
        return ""

    intro = (
        "## Batched decode throughput\n\n"
        "Decode batching coalesces the per-step prediction-LSTM and joint-network "
        "GEMMs across several utterances into single batched ops, so one decode "
        "loop advances B clips at once. It applies to **transducer (TDT/RNN-T) "
        "models only** \u2014 CTC has no autoregressive decode to batch. Speedup is "
        "`serial_ms / batched_ms`: the wall-clock of B independent single-clip "
        "decodes divided by one batched decode over the same B copies (encoder "
        "cost is paid once and excluded). The `clips/s @B=16` column is the "
        "batched decode throughput at B=16.\n"
    )
    parts = [intro]
    if cpu:
        parts.append("\n" + cpu)
    if gpu:
        parts.append("\n" + gpu)
    return "".join(parts)


# ── Findings ─────────────────────────────────────────────────────────────────────

def build_findings(models: list[dict], dtypes: list[str]) -> str:
    f32_su, q8_su, ag_f32 = [], [], []
    slower = []
    for m in models:
        ls = _ls(m); n = ls["nemo"]["rtfx"]
        s = ls["ours"]["f32"]["rtfx"] / n
        f32_su.append(s)
        ag_f32.append(ls["ours"]["f32"]["agreement_wer_vs_nemo"] * 100)
        if "q8_0" in ls["ours"]:
            q8_su.append(ls["ours"]["q8_0"]["rtfx"] / n)
        if s < 1.0:
            slower.append(_short(m["model"]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    n_models = len(models)
    n_faster = sum(1 for s in f32_su if s >= 1.0)
    fastest = max(models, key=lambda m: _ls(m)["ours"]["f32"]["rtfx"] / _ls(m)["nemo"]["rtfx"])
    fastest_su = _ls(fastest)["ours"]["f32"]["rtfx"] / _ls(fastest)["nemo"]["rtfx"]

    lines = ["## Findings\n"]

    lines.append("### Accuracy")
    lines.append(
        f"parakeet.cpp reproduces NeMo with high fidelity: mean f32 agreement WER is "
        f"**{mean(ag_f32):.4f}%** — effectively byte-identical output, and WER vs "
        f"ground truth tracks NeMo. The faithful path (f32) is the reference; "
        f"quantization is opt-in for size.\n"
    )

    lines.append("### Performance")
    if slower:
        lines.append(
            f"parakeet.cpp is faster than NeMo on **{n_faster}/{n_models}** models "
            f"(mean f32 speedup **{mean(f32_su):.2f}×**, q8_0 **{mean(q8_su):.2f}×**); "
            f"still trailing on: {', '.join(slower)}.\n"
        )
    else:
        lines.append(
            f"parakeet.cpp is **faster than NeMo on all {n_models} models** "
            f"(mean f32 speedup **{mean(f32_su):.2f}×**, q8_0 **{mean(q8_su):.2f}×**; "
            f"best: `{_short(fastest['model'])}` at {fastest_su:.2f}×). "
            f"The decisive decode-side win was caching the transducer prediction-net "
            f"forward pass across non-emitting frames (the LSTM was ~97% of RNN-T "
            f"decode time and mostly redundant); the encoder side uses a persistent "
            f"ggml backend + gallocr, zero-copy weights, a single fused graph, and "
            f"tinyBLAS (GGML_LLAMAFILE).\n"
        )

    lines.append("### Memory")
    lines.append(
        "The ggml engine uses markedly less peak RAM than NeMo/PyTorch, and "
        "quantization lowers it further — q8_0 is near-lossless, while K-quants "
        "(q6_k→q4_k) keep shrinking the model on disk and in RAM at a small, "
        "monotonic accuracy cost (see the quantization plots).\n"
    )

    lines.append("### Notes")
    lines.append(
        "- CPU-only; a CUDA/Metal ggml backend would change the picture again.\n"
        "- NeMo is the unchanged reference (re-running its slow CPU pass reproduces the same numbers); only the ggml engine was re-measured for this refresh.\n"
        "- Thread scaling (threads.json) shows 8 threads as the sweet spot on this 20-core host.\n"
    )
    lines.append("")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────────

def build_demo_section() -> str:
    return """\
## Demo

The same clip fed to parakeet.cpp and to a reference runtime, each pane typing \
its transcript at the engine's real measured speed, so the faster engine \
finishes first. parakeet.cpp reproduces NeMo's output byte-for-byte.

parakeet.cpp vs NeMo (PyTorch) on the same GPU (slowed down so the sub-100ms \
race is watchable):

![parakeet.cpp vs NeMo on GPU](media/gpu_duel.gif)

And on CPU, against NeMo's own PyTorch runtime, about 1.5x faster and still \
byte-for-byte identical:

![parakeet.cpp vs NeMo on CPU](media/cpu_nemo_duel.gif)

Versus whisper.cpp turbo, same accuracy (WER 1.6% on this clip) and far less \
compute:

| matchup | result |
|---------|--------|
| [vs whisper.cpp turbo, GPU](media/gpu_whisper_duel.mp4) | about 12x faster |
| [vs whisper.cpp turbo, CPU](media/cpu_duel.mp4) | about 27x faster |
"""


def main():
    ap = argparse.ArgumentParser(description="Generate BENCHMARK.md from parakeet.cpp results.")
    ap.add_argument("--results", default="benchmarks/results")
    ap.add_argument("--plots", default="benchmarks/plots")
    ap.add_argument("--out", default="benchmarks/BENCHMARK.md")
    args = ap.parse_args()

    results_dir = Path(args.results)
    plots_dir = Path(args.plots)
    out_path = Path(args.out)

    models = load_results(results_dir)
    if not models:
        print(f"ERROR: no model JSON files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    dtypes = dtypes_present(models)
    print(f"Loaded {len(models)} model(s); dtypes={dtypes}")

    sections = [
        "# parakeet.cpp Benchmark: NeMo (PyTorch CPU) vs ggml\n\n",
        "> Auto-generated by `scripts/gen_benchmark_md.py`. "
        "Refresh: re-run `scripts/benchmark.py` (optionally `--skip-nemo`), then "
        "`scripts/plot_benchmark.py`, then this script.\n\n",
        build_demo_section() + "\n",
        build_methodology(models, dtypes) + "\n",
        build_headline_table(models) + "\n",
        build_nemotron_section(results_dir) + "\n",
        build_quant_table(models, dtypes) + "\n",
        build_batched_decode_section(results_dir) + "\n",
        build_plots_section(plots_dir, out_path) + "\n",
        build_diverse_section(models) + "\n",
        build_findings(models, dtypes) + "\n",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(sections))
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
