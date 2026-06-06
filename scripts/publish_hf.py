#!/usr/bin/env python3
"""Publish parakeet.cpp GGUF models to HuggingFace Hub.

For a given source NeMo checkpoint (e.g. ``nvidia/parakeet-tdt_ctc-110m``),
this script:

1. Converts the checkpoint to F16 and Q8_0 variants via
   ``scripts/convert_parakeet_to_gguf.py``.
2. Re-quantizes the F32 intermediate to Q4_K via
   ``build/examples/cli/parakeet-cli quantize``.
3. Generates a per-model README (model card) with architecture details,
   variant sizes, validated WER (from ``docs/parity.md``), and a usage snippet.
4. Uploads the GGUFs + README to ``mudler/parakeet.cpp-<variant>`` on HF Hub.

**DRY-RUN BY DEFAULT** — without ``--upload`` it prints what it *would* upload
(repo id, files, sizes) but does NOT contact HuggingFace at all. Pass
``--upload`` to perform the real push.

Usage:
    .venv/bin/python scripts/publish_hf.py \\
        --model nvidia/parakeet-tdt_ctc-110m      # dry-run (safe default)

    .venv/bin/python scripts/publish_hf.py \\
        --model nvidia/parakeet-tdt_ctc-110m \\
        --upload                                  # actually push to HF
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONVERTER = SCRIPTS_DIR / "convert_parakeet_to_gguf.py"
# Prefer build/ CLI; fall back to build-shared/ (or whatever exists).
_CLI_CANDIDATES = [
    REPO_ROOT / "build" / "examples" / "cli" / "parakeet-cli",
    REPO_ROOT / "build-shared" / "examples" / "cli" / "parakeet-cli",
]
PYTHON = Path(sys.executable)

# ---------------------------------------------------------------------------
# Publish settings
# ---------------------------------------------------------------------------
HF_USER = "mudler"
DEFAULT_REPO_PREFIX = f"{HF_USER}/parakeet-cpp-"
# Single collective repo (one repo, all models × all variants as flat GGUFs).
DEFAULT_COLLECTION_REPO = f"{HF_USER}/parakeet-cpp-gguf"
DEFAULT_VARIANTS = ["f16", "q8_0", "q6_k", "q5_k", "q4_k"]

# Canonical model set, smallest/validated-first so quick wins land before the
# large checkpoints (and a late failure leaves the important models published).
# `--model all` expands to this list.
ALL_MODELS = [
    "nvidia/parakeet-tdt_ctc-110m",
    "nvidia/parakeet_realtime_eou_120m-v1",
    "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "nvidia/parakeet-ctc-0.6b",
    "nvidia/parakeet-rnnt-0.6b",
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/parakeet-ctc-1.1b",
    "nvidia/parakeet-rnnt-1.1b",
    "nvidia/parakeet-tdt-1.1b",
    "nvidia/parakeet-tdt_ctc-1.1b",
]

# Validated WER data sourced from docs/parity.md and docs/quantization.md.
# Structure: {hf_model_id: {variant: {"wer": float|str, "size_mb": float|str}}}
KNOWN_WER: dict = {
    "nvidia/parakeet-tdt_ctc-110m": {
        "f16":  {"wer": 0.0, "size_mb": 255.1},
        "q8_0": {"wer": 0.0, "size_mb": 169.6},
        "q4_k": {"wer": 0.0, "size_mb": 125.3},
    },
    "nvidia/parakeet-tdt-0.6b-v2": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": 862.0},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-tdt-0.6b-v3": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-tdt-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-tdt_ctc-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-ctc-0.6b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-ctc-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-rnnt-0.6b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-rnnt-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    # Multilingual prompt-conditioned streaming model. WER measured offline
    # (en/de/auto) by scripts/e2e_nemo_compare.py vs NeMo; all five variants are
    # byte-for-byte (WER 0.0). prompt_kernel + LSTM + featurizer tensors stay F32.
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "f16":  {"wer": 0.0, "size_mb": 1484.3},
        "q8_0": {"wer": 0.0, "size_mb": 983.7},
        "q6_k": {"wer": 0.0, "size_mb": 855.7},
        "q5_k": {"wer": 0.0, "size_mb": 784.8},
        "q4_k": {"wer": 0.0, "size_mb": 718.1},
    },
}

# Per-model SPDX license id + human label, used in the generated card frontmatter
# and the License section. Defaults to CC-BY-4.0 (the NVIDIA NeMo Parakeet
# checkpoints); nemotron-3.5-asr-streaming is released under OpenMDW-1.1.
DEFAULT_LICENSE = ("cc-by-4.0", "CC-BY-4.0",
                   "https://creativecommons.org/licenses/by/4.0/")
LICENSES: dict = {
    "nvidia/nemotron-3.5-asr-streaming-0.6b": (
        "other", "OpenMDW-1.1", "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
    ),
}


def _license(model_id: str) -> tuple[str, str, str]:
    """(frontmatter license id, human label, upstream/license url) for *model_id*."""
    return LICENSES.get(model_id, DEFAULT_LICENSE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_cli() -> Path:
    for p in _CLI_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "parakeet-cli not found in build/ or build-shared/. "
        "Run: cmake -B build -DPARAKEET_BUILD_CLI=ON && cmake --build build -j"
    )


def _model_slug(model_id: str) -> str:
    """``nvidia/parakeet-tdt_ctc-110m`` → ``tdt_ctc-110m``.

    Strips the org and the redundant ``parakeet-``/``parakeet_`` prefix so the
    resulting HF repo (``<repo_prefix><slug>-<variant>``, default prefix
    ``mudler/parakeet-cpp-``) reads as ``mudler/parakeet-cpp-tdt_ctc-110m-q5_k``
    rather than ``mudler/parakeet-cpp-parakeet-tdt_ctc-110m-q5_k``.
    """
    name = model_id.split("/")[-1]
    for prefix in ("parakeet-", "parakeet_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _gguf_name(model_id: str, variant: str) -> str:
    return f"{_model_slug(model_id)}-{variant}.gguf"


def _run(cmd: List[str], *, label: str) -> None:
    """Run *cmd* and propagate failures with a clear message."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}"
        )


# ---------------------------------------------------------------------------
# Conversion + quantization
# ---------------------------------------------------------------------------

def convert_variant(model_id: str, variant: str, output_dir: Path, f32_path: Path,
                    reuse_existing: bool = False) -> Path:
    """Convert/quantize ``model_id`` to *variant* and return the output path.

    * ``f16`` / ``q8_0``: call the Python converter with ``--dtype``.
    * ``q4_k``: call ``parakeet-cli quantize`` on the F32 GGUF.
    * ``f32``: call the Python converter with ``--dtype f32`` (intermediate).

    With ``reuse_existing`` a non-empty output on disk is reused as-is and the
    (expensive) conversion is skipped — this makes an interrupted run resumable
    without reconverting everything.
    """
    out = output_dir / variant / _gguf_name(model_id, variant)
    out.parent.mkdir(parents=True, exist_ok=True)

    if reuse_existing and out.is_file() and out.stat().st_size > 0:
        print(f"  [reuse] {out} ({out.stat().st_size / 1e6:.1f} MB) — skipping conversion")
        return out

    if variant in ("f16", "q8_0"):
        _run(
            [
                str(PYTHON), str(CONVERTER),
                "--model", model_id,
                "--output", str(out),
                "--dtype", variant,
            ],
            label=f"convert {variant}",
        )
    elif variant == "f32":
        _run(
            [
                str(PYTHON), str(CONVERTER),
                "--model", model_id,
                "--output", str(out),
                "--dtype", "f32",
            ],
            label="convert f32",
        )
    elif variant.startswith("q") and "_k" in variant:
        # K-quant: re-quantize from f32
        if not f32_path.is_file():
            raise FileNotFoundError(
                f"F32 intermediate not found at {f32_path}; "
                "cannot produce K-quant without it"
            )
        cli = _find_cli()
        _run(
            [str(cli), "quantize", str(f32_path), str(out), variant],
            label=f"quantize {variant}",
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    if not out.is_file():
        raise RuntimeError(f"Expected output not produced: {out}")
    return out


def build_variants(
    model_id: str,
    variants: List[str],
    output_dir: Path,
    reuse_existing: bool = False,
) -> dict[str, Path]:
    """Build all requested variants; return {variant: path}."""
    slug = _model_slug(model_id)
    results: dict[str, Path] = {}

    def on_disk(variant: str) -> bool:
        p = output_dir / variant / _gguf_name(model_id, variant)
        return reuse_existing and p.is_file() and p.stat().st_size > 0

    # F32 is only an intermediate for K-quants — and only needed if at least one
    # K-quant still has to be built (when reusing, the quants may already exist).
    kquants = [v for v in variants if "_k" in v]
    needs_f32 = any(not on_disk(v) for v in kquants)
    f32_path = output_dir / "f32" / f"{slug}-f32.gguf"

    if needs_f32 and not (f32_path.is_file() and f32_path.stat().st_size > 0):
        print(f"\n--- Converting {model_id} → f32 (intermediate for K-quants) ---")
        f32_path = convert_variant(model_id, "f32", output_dir, f32_path, reuse_existing)

    for variant in variants:
        print(f"\n--- Converting {model_id} → {variant} ---")
        path = convert_variant(model_id, variant, output_dir, f32_path, reuse_existing)
        results[variant] = path
        size_mb = path.stat().st_size / 1e6
        print(f"  produced: {path} ({size_mb:.1f} MB)")

    return results


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def _wer_str(wer) -> str:
    if wer is None:
        return "not measured"
    return f"{wer:.4f}"


def _size_str(size_mb, path: Optional[Path] = None) -> str:
    if path is not None and path.is_file():
        return f"{path.stat().st_size / 1e6:.1f} MB"
    if size_mb is not None:
        return f"{size_mb:.1f} MB"
    return "—"


def _arch_info(model_id: str) -> tuple[str, str]:
    """Infer (arch_desc, decoder heads) from the model name."""
    name_lower = model_id.lower()
    if "nemotron" in name_lower:
        return (
            "Cache-aware streaming, multilingual, prompt-conditioned RNNT "
            "(FastConformer, 40+ locales, --lang <locale>)",
            "RNNT (streaming, prompt-conditioned)",
        )
    if "realtime_eou" in name_lower or "realtime-eou" in name_lower:
        return "Cache-aware streaming RNNT (FastConformer, EOU/EOB)", "RNNT (streaming)"
    if "tdt_ctc" in name_lower:
        return "Hybrid TDT+CTC (FastConformer)", "TDT + CTC"
    if "tdt" in name_lower:
        return "TDT transducer (FastConformer)", "TDT"
    if "rnnt" in name_lower:
        return "RNNT transducer (FastConformer)", "RNNT"
    if "ctc" in name_lower:
        return "CTC (FastConformer)", "CTC"
    return "Parakeet (FastConformer)", "default"


def build_model_card(
    model_id: str,
    variants: List[str],
    variant_paths: dict[str, Path],
    repo_prefix: str,
) -> str:
    """Generate a Markdown model card for one model × all its variants."""
    slug = _model_slug(model_id)
    wer_data = KNOWN_WER.get(model_id, {})

    arch_desc, heads = _arch_info(model_id)
    lic_id, lic_label, lic_url = _license(model_id)

    lines: List[str] = []

    # YAML frontmatter. For a non-SPDX license (license: other) HF wants an
    # explicit license_name + license_link.
    lines += ["---", f"license: {lic_id}"]
    if lic_id == "other":
        lines += [f"license_name: {lic_label}", f"license_link: {lic_url}"]
    lines += [
        "library_name: parakeet.cpp",
        "tags:",
        "  - automatic-speech-recognition",
        "  - asr",
        "  - parakeet",
        "  - gguf",
        "  - ggml",
        "  - cpp-inference",
        "  - nemo",
        "pipeline_tag: automatic-speech-recognition",
        f"base_model: {model_id}",
        "---",
        "",
    ]

    lines.append(f"# {slug} — GGUF for parakeet.cpp")
    lines.append("")
    lines.append(
        f"GGUF-format weights of [{model_id}](https://huggingface.co/{model_id}) "
        f"for use with [parakeet.cpp](https://github.com/mudler/parakeet.cpp), "
        f"a C++/ggml port of NVIDIA NeMo Parakeet that matches the upstream PyTorch "
        f"model on CPU."
    )
    lines.append("")
    lines.append(
        "This repo contains the quantized variants listed below. "
        "**F16 is the recommended default** — same accuracy as F32, ~1.7× smaller, "
        "and typically the fastest on modern CPUs via ggml's F32×F16 matmul fast path."
    )
    lines.append("")

    # Files table
    lines.append("## Available files")
    lines.append("")
    lines.append("| File | Variant | Size | WER vs NeMo |")
    lines.append("|---|---|---:|---:|")
    for v in variants:
        path = variant_paths.get(v)
        wd = wer_data.get(v, {})
        wer = _wer_str(wd.get("wer"))
        size = _size_str(wd.get("size_mb"), path)
        fname = _gguf_name(model_id, v)
        rec = " ← **recommended**" if v == "f16" else ""
        lines.append(f"| `{fname}`{rec} | {v.upper()} | {size} | {wer} |")
    lines.append("")

    lines.append(
        "> WER (word error rate) is computed against the upstream NeMo reference on "
        "`tests/fixtures/speech.wav` (LibriSpeech `2086-149220-0033`, ~7.4 s, English). "
        "0.0 = byte-for-byte identical transcript. "
        "See [parity.md](https://github.com/mudler/parakeet.cpp/blob/main/docs/parity.md) "
        "and [quantization.md](https://github.com/mudler/parakeet.cpp/blob/main/docs/quantization.md) "
        "for the full validation suite."
    )
    lines.append("")

    # Architecture
    lines.append("## Architecture")
    lines.append("")
    lines.append(f"- Source checkpoint: `{model_id}`")
    lines.append(f"- Architecture: {arch_desc}")
    lines.append(f"- Decoder head(s): {heads}")
    lines.append(f"- Upstream: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)")
    lines.append("")

    # Quantization notes
    lines.append("## Quantization notes")
    lines.append("")
    lines.append(
        "Quantization is applied **only** to the large linear weights fed directly "
        "into `ggml_mul_mat` (encoder FFN + attention projections, subsampling output "
        "projection, joint enc/pred projections). All other tensors (mel filterbank, "
        "LSTM prediction net, conv kernels, batch_norm stats, norms, biases, embeddings) "
        "stay F32."
    )
    lines.append("")
    lines.append("| Variant | What is quantized | Notes |")
    lines.append("|---|---|---|")
    lines.append(
        "| F16 | allowlisted linear weights → IEEE half | lossless; WER 0.0 on 110m |"
    )
    lines.append(
        "| Q8_0 | allowlisted linear weights → Q8_0 (8-bit, 32-element blocks) | WER 0.0 on 110m and 0.6B |"
    )
    lines.append(
        "| Q4_K | allowlisted linear weights → Q4_K (K-quant, 256-element superblocks) | "
        "`joint.pred.weight` stays F32 when ne[0] % 256 ≠ 0; WER 0.0 on 110m |"
    )
    lines.append("")

    # Usage
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Clone + build parakeet.cpp")
    lines.append("git clone https://github.com/mudler/parakeet.cpp")
    lines.append("cd parakeet.cpp")
    lines.append("cmake -B build -DPARAKEET_BUILD_CLI=ON && cmake --build build -j")
    lines.append("")
    lines.append("# 2. Download a quant (F16 recommended)")
    repo_id_example = f"{repo_prefix}{slug}-f16"
    lines.append(
        f"huggingface-cli download {repo_id_example} {_gguf_name(model_id, 'f16')} "
        f"--local-dir models/"
    )
    lines.append("")
    lines.append("# 3. Transcribe")
    lines.append("build/examples/cli/parakeet-cli transcribe \\")
    lines.append(f"    --model models/{_gguf_name(model_id, 'f16')} \\")
    lines.append("    --input audio.wav")
    lines.append("```")
    lines.append("")

    # License
    lines.append("## License")
    lines.append("")
    lines.append(
        f"The GGUF weights are derived from [{model_id}](https://huggingface.co/{model_id}), "
        f"released under the [{lic_label}]({lic_url}) license. "
        "The parakeet.cpp runtime is MIT-licensed."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _upload_with_retry(api, local_path: Path, remote_name: str, repo_id: str, max_retries: int = 3) -> None:
    """Upload one file with exponential backoff."""
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]

    size_mb = local_path.stat().st_size / 1e6
    print(f"  -> {remote_name} ({size_mb:.1f} MB)... ", end="", flush=True)
    delay = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model",
            )
            dt = time.time() - t0
            mbps = size_mb / dt if dt > 0 else 0.0
            print(f"ok ({dt:.1f}s, {mbps:.1f} MB/s)")
            return
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            last_err = e
            print(
                f"\n     attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(
        f"upload failed after {max_retries} attempts: {last_err}"
    ) from last_err


def publish_model(
    model_id: str,
    variants: List[str],
    variant_paths: dict[str, Path],
    *,
    repo_prefix: str,
    upload: bool,
) -> dict:
    """Publish one model to HF (or print what would happen in dry-run mode)."""
    slug = _model_slug(model_id)

    results = []
    for variant in variants:
        repo_id = f"{repo_prefix}{slug}-{variant}"
        path = variant_paths[variant]
        card = build_model_card(model_id, [variant], {variant: path}, repo_prefix)
        size_mb = path.stat().st_size / 1e6

        print(f"\n=== {repo_id} ===")
        print(f"  file: {path.name} ({size_mb:.1f} MB)")

        if not upload:
            print(f"  [dry-run] would create repo {repo_id} (public)")
            print(f"  [dry-run] would upload README.md ({len(card)} bytes)")
            print(f"  [dry-run] would upload {path.name} ({size_mb:.1f} MB)")
            results.append({
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "variant": variant,
                "file": path.name,
                "size_mb": size_mb,
                "dry_run": True,
            })
        else:
            from huggingface_hub import HfApi  # type: ignore[attr-defined]

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
            print(f"  uploading README.md ({len(card)} bytes)")
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add model card for {slug} {variant}",
            )
            _upload_with_retry(api, path, path.name, repo_id)
            results.append({
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "variant": variant,
                "file": path.name,
                "size_mb": size_mb,
                "dry_run": False,
            })

    return results


# ---------------------------------------------------------------------------
# Single-repo collection: one repo, all models × variants as flat GGUFs
# ---------------------------------------------------------------------------

def build_collection_card(
    models: List[str],
    variants: List[str],
    paths_by_model: dict[str, dict[str, Path]],
    repo_id: str,
) -> str:
    """One combined model card for the whole collection repo."""
    lines: List[str] = []

    # YAML frontmatter — base_model accepts a list. Licenses can differ per model
    # (most are CC-BY-4.0; nemotron is OpenMDW-1.1), so when the collection mixes
    # licenses we declare `other` here and spell each one out in the License
    # section / per-model rows below.
    licenses = {_license(m) for m in models}
    if len(licenses) == 1:
        only_id, only_label, only_url = next(iter(licenses))
        lines = ["---", f"license: {only_id}"]
        if only_id == "other":
            lines += [f"license_name: {only_label}", f"license_link: {only_url}"]
        lines += ["library_name: parakeet.cpp", "tags:"]
    else:
        lines = [
            "---", "license: other",
            "license_name: mixed (see per-model licenses below)",
            f"license_link: https://huggingface.co/{repo_id}",
            "library_name: parakeet.cpp", "tags:",
        ]
    lines += [f"  - {t}" for t in (
        "automatic-speech-recognition", "asr", "parakeet",
        "gguf", "ggml", "cpp-inference", "nemo",
    )]
    lines.append("pipeline_tag: automatic-speech-recognition")
    lines.append("base_model:")
    lines += [f"  - {m}" for m in models]
    lines += ["---", ""]

    lines.append("# Parakeet GGUF — models for parakeet.cpp")
    lines.append("")
    lines.append(
        "GGUF-format weights for [parakeet.cpp](https://github.com/mudler/parakeet.cpp), "
        "a C++/ggml port of NVIDIA NeMo Parakeet that matches the upstream PyTorch models "
        "on CPU. This single repo collects **every supported model × quantization** as a "
        "flat set of `.gguf` files — download just the one you need."
    )
    lines.append("")
    lines.append(
        "**F16 is the recommended default** — same accuracy as F32, ~1.7× smaller, and "
        "typically the fastest on modern CPUs via ggml's F32×F16 matmul fast path."
    )
    lines.append("")

    # Per-model sections with a variant table each.
    lines.append("## Models")
    lines.append("")
    for model_id in models:
        slug = _model_slug(model_id)
        arch_desc, heads = _arch_info(model_id)
        wer_data = KNOWN_WER.get(model_id, {})
        vp = paths_by_model.get(model_id, {})
        _lic_id, _lic_label, _lic_url = _license(model_id)
        lines.append(f"### {slug}")
        lines.append("")
        lines.append(
            f"Source: [{model_id}](https://huggingface.co/{model_id}) · "
            f"{arch_desc} · heads: {heads} · license: [{_lic_label}]({_lic_url})"
        )
        lines.append("")
        lines.append("| File | Variant | Size | WER vs NeMo |")
        lines.append("|---|---|---:|---:|")
        for v in variants:
            path = vp.get(v)
            wd = wer_data.get(v, {})
            wer = _wer_str(wd.get("wer"))
            size = _size_str(wd.get("size_mb"), path)
            fname = _gguf_name(model_id, v)
            rec = " ← **recommended**" if v == "f16" else ""
            lines.append(f"| `{fname}`{rec} | {v.upper()} | {size} | {wer} |")
        lines.append("")

    lines.append(
        "> WER (word error rate) is computed against the upstream NeMo reference on "
        "`tests/fixtures/speech.wav` (LibriSpeech `2086-149220-0033`, ~7.4 s, English). "
        "0.0 = byte-for-byte identical transcript. "
        "See [parity.md](https://github.com/mudler/parakeet.cpp/blob/main/docs/parity.md) "
        "and [quantization.md](https://github.com/mudler/parakeet.cpp/blob/main/docs/quantization.md)."
    )
    lines.append("")

    # Quantization notes (shared).
    lines.append("## Quantization notes")
    lines.append("")
    lines.append(
        "Quantization is applied **only** to the large linear weights fed directly into "
        "`ggml_mul_mat` (encoder FFN + attention projections, subsampling output projection, "
        "joint enc/pred projections). All other tensors (mel filterbank, LSTM prediction net, "
        "conv kernels, batch_norm stats, norms, biases, embeddings) stay F32."
    )
    lines.append("")

    # Usage.
    lines.append("## Usage")
    lines.append("")
    example_model = models[0]
    example_file = _gguf_name(example_model, "f16")
    lines.append("```bash")
    lines.append("# 1. Clone + build parakeet.cpp")
    lines.append("git clone https://github.com/mudler/parakeet.cpp")
    lines.append("cd parakeet.cpp")
    lines.append("cmake -B build -DPARAKEET_BUILD_CLI=ON && cmake --build build -j")
    lines.append("")
    lines.append("# 2. Download one quant (F16 recommended)")
    lines.append(f"huggingface-cli download {repo_id} {example_file} --local-dir models/")
    lines.append("")
    lines.append("# 3. Transcribe")
    lines.append("build/examples/cli/parakeet-cli transcribe \\")
    lines.append(f"    --model models/{example_file} \\")
    lines.append("    --input audio.wav")
    lines.append("```")
    lines.append("")

    # License. Each GGUF inherits the license of its source checkpoint; list them
    # explicitly because the collection can mix licenses (most CC-BY-4.0, nemotron
    # OpenMDW-1.1).
    lines.append("## License")
    lines.append("")
    lines.append(
        "Each GGUF is derived from its upstream NVIDIA NeMo checkpoint and inherits "
        "that checkpoint's license. The parakeet.cpp runtime itself is MIT-licensed."
    )
    lines.append("")
    lines.append("| Source checkpoint | License |")
    lines.append("|---|---|")
    for model_id in models:
        _lic_id, _lic_label, _lic_url = _license(model_id)
        lines.append(
            f"| [{model_id}](https://huggingface.co/{model_id}) | [{_lic_label}]({_lic_url}) |"
        )
    lines.append("")

    return "\n".join(lines)


def _f32_intermediate(model_id: str, output_dir: Path) -> Path:
    return output_dir / "f32" / f"{_model_slug(model_id)}-f32.gguf"


def plan_collection(models: List[str], variants: List[str], repo_id: str) -> None:
    """Print the intended repo layout — no conversion, no HF calls."""
    print("=" * 60)
    print("COLLECTION PLAN (no conversion, no upload)")
    print("=" * 60)
    print(f"repo: https://huggingface.co/{repo_id}  (single repo, flat layout)")
    print()
    for model_id in models:
        for v in variants:
            print(f"  {_gguf_name(model_id, v)}")
    print("  README.md  (combined model card)")
    print()
    print(
        f"total: {len(models) * len(variants)} GGUF files "
        f"({len(models)} models × {len(variants)} variants) + 1 README"
    )


def publish_collection(
    models: List[str],
    variants: List[str],
    output_dir: Path,
    repo_id: str,
    *,
    upload: bool,
    reuse_existing: bool = False,
) -> int:
    """Convert all models and publish them into one flat repo.

    Converts + uploads each model incrementally so a late failure leaves the
    earlier models already published; the combined README is (re)uploaded at
    the end from whatever succeeded. Returns a process exit code.

    The flow is resumable: with ``reuse_existing`` already-converted GGUFs on
    disk are not rebuilt, and files already present in the repo are not
    re-uploaded — so a watchdog can kill+restart a stalled upload and it
    picks up where it left off.
    """
    api = None
    remote_files: set = set()
    if upload:
        from huggingface_hub import HfApi  # type: ignore[attr-defined]

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        try:
            remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except Exception as e:
            print(f"  (could not list existing repo files: {e})", file=sys.stderr)
        print(f"repo ready: https://huggingface.co/{repo_id} "
              f"({len(remote_files)} files already present)\n")

    paths_by_model: dict[str, dict[str, Path]] = {}
    failures: List[tuple[str, str]] = []

    for model_id in models:
        print(f"\n########## {model_id} ##########")
        try:
            vp = build_variants(model_id, variants, output_dir, reuse_existing=reuse_existing)
        except Exception as e:  # isolate per-model failures
            print(f"  ERROR: conversion failed for {model_id}: {e}", file=sys.stderr)
            failures.append((model_id, str(e)))
            continue
        paths_by_model[model_id] = vp

        for v in variants:
            path = vp[v]
            size_mb = path.stat().st_size / 1e6
            if upload:
                if path.name in remote_files:
                    print(f"  [skip] {path.name} already in repo")
                    continue
                _upload_with_retry(api, path, path.name, repo_id)
                remote_files.add(path.name)
            else:
                print(f"  [dry-run] would upload {path.name} ({size_mb:.1f} MB)")

        # Drop the big F32 intermediate now that this model's quants are done.
        f32 = _f32_intermediate(model_id, output_dir)
        if f32.is_file():
            f32.unlink()

    # Combined README from whatever converted successfully.
    if paths_by_model:
        card = build_collection_card(
            list(paths_by_model.keys()), variants, paths_by_model, repo_id
        )
        if upload:
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update combined parakeet.cpp model card",
            )
            print("\nuploaded README.md")
        else:
            print(f"\n[dry-run] would upload README.md ({len(card)} bytes)")

    # Summary.
    print()
    print("=" * 60)
    print("COLLECTION SUMMARY" if upload else "COLLECTION DRY-RUN")
    print("=" * 60)
    ok_files = sum(len(v) for v in paths_by_model.values())
    print(f"repo: https://huggingface.co/{repo_id}")
    print(f"models converted: {len(paths_by_model)}/{len(models)}  ({ok_files} GGUF files)")
    if failures:
        print(f"failures: {len(failures)}")
        for m, err in failures:
            print(f"  - {m}: {err}")
    if not upload:
        print("\nNOTE: dry run. Re-run with --upload to push to HuggingFace.")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="HF_ID",
        help=(
            "Source HuggingFace model id, e.g. nvidia/parakeet-tdt_ctc-110m. "
            "Repeatable. Use the literal 'all' to expand to the full canonical set."
        ),
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        metavar="VAR,...",
        help=(
            "Comma-separated list of quantization variants to produce. "
            f"Supported: f16, q8_0, q6_k, q5_k, q4_k. Default: {','.join(DEFAULT_VARIANTS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "models"),
        metavar="DIR",
        help="Directory to write GGUFs under (sub-dirs per variant). Default: models/",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Actually push to HuggingFace Hub. "
            "Without this flag the script is a dry run: it converts/quantizes "
            "the GGUFs locally but does NOT contact HuggingFace."
        ),
    )
    parser.add_argument(
        "--repo",
        metavar="REPO_ID",
        nargs="?",
        const=DEFAULT_COLLECTION_REPO,
        help=(
            "Single-repo (collection) mode: publish every model × variant into one "
            "repo as flat <slug>-<variant>.gguf files with a combined README. "
            f"Bare --repo uses {DEFAULT_COLLECTION_REPO}. Without --repo, the legacy "
            "one-repo-per-variant layout is used."
        ),
    )
    parser.add_argument(
        "--repo-prefix",
        default=DEFAULT_REPO_PREFIX,
        metavar="PREFIX",
        help=(
            f"(legacy mode) HF repo-id prefix. Each variant is published to "
            f"<prefix><model-slug>-<variant>. Default: {DEFAULT_REPO_PREFIX}"
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Print the intended collection repo layout (filenames) and exit — "
            "no conversion, no HuggingFace calls. Implies --repo."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse GGUFs already on disk (skip reconversion) and skip files "
            "already present in the target repo. Makes an interrupted upload "
            "resumable — pair with a watchdog that restarts on stall."
        ),
    )
    args = parser.parse_args()

    # Resolve the model list ('all' → canonical set), de-duplicated, order-stable.
    raw_models = args.model or []
    models: List[str] = []
    for m in raw_models:
        for one in (ALL_MODELS if m == "all" else [m]):
            if one not in models:
                models.append(one)

    variants: List[str] = [v.strip() for v in args.variants.split(",") if v.strip()]
    output_dir = Path(args.output_dir)
    upload: bool = args.upload
    repo_prefix: str = args.repo_prefix

    # --plan is a pure preview and implies collection mode.
    if args.plan:
        if not models:
            print("error: --plan needs at least one --model (or --model all)", file=sys.stderr)
            return 2
        plan_collection(models, variants, args.repo or DEFAULT_COLLECTION_REPO)
        return 0

    if not models:
        print("error: --model is required (repeatable, or --model all)", file=sys.stderr)
        return 2

    # Validate variants
    supported = {"f16", "q8_0", "q4_k", "q5_k", "q6_k", "q4_0", "q5_0"}
    bad = [v for v in variants if v not in supported]
    if bad:
        print(f"error: unknown variants: {bad}. Supported: {sorted(supported)}", file=sys.stderr)
        return 2

    # Sanity check: converter must exist
    if not CONVERTER.is_file():
        print(f"error: converter not found: {CONVERTER}", file=sys.stderr)
        return 2

    # Auth check (only when uploading)
    if upload:
        try:
            from huggingface_hub import HfApi  # type: ignore[attr-defined]

            api = HfApi()
            me = api.whoami()
            print(f"authenticated as: {me['name']}")
        except Exception as e:
            print(f"error: HF auth failed: {e}", file=sys.stderr)
            print(
                "hint: run `huggingface-cli login` or place a token in "
                "~/.cache/huggingface/token",
                file=sys.stderr,
            )
            return 2
    else:
        print("[dry-run mode] No HuggingFace calls will be made. Pass --upload to push.")
        print()

    t0 = time.time()

    # Collection mode: one repo, all models × variants, flat layout.
    if args.repo:
        rc = publish_collection(models, variants, output_dir, args.repo, upload=upload,
                                reuse_existing=args.reuse_existing)
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        return rc

    # Legacy mode: one repo per model × variant.
    all_results: list = []
    for model_id in models:
        print(f"\n=== Building variants for {model_id} ===")
        try:
            variant_paths = build_variants(model_id, variants, output_dir)
        except Exception as e:
            print(f"error: conversion/quantization failed for {model_id}: {e}", file=sys.stderr)
            return 1
        try:
            all_results += publish_model(
                model_id, variants, variant_paths,
                repo_prefix=repo_prefix, upload=upload,
            )
        except Exception as e:
            print(f"error: publish failed for {model_id}: {e}", file=sys.stderr)
            return 1

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    mode = "UPLOAD PLAN (dry-run)" if not upload else "UPLOAD SUMMARY"
    print(mode)
    print("=" * 60)
    print(f"{'Repo':55s} {'File':35s} {'Size':>10s}")
    for r in all_results:
        marker = "  [dry-run]" if r["dry_run"] else ""
        print(f"{r['repo_id']:55s} {r['file']:35s} {r['size_mb']:>8.1f} MB{marker}")
        print(f"  {'(would be at)' if r['dry_run'] else 'published at'}: {r['url']}")
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    if not upload:
        print()
        print("NOTE: This was a dry run. Re-run with --upload to push to HuggingFace.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
