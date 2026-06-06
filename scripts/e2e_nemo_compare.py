#!/usr/bin/env python3
"""End-to-end: compare parakeet.cpp nemotron transcripts to NeMo's, per language,
offline and cache-aware streaming.

For each (clip, target_lang, mode):

  * NeMo reference  -- computed by ``gen_nemo_baseline.compute_prompt_reference``
    (the SAME prompt-conditioned path every Phase 2/3 baseline uses: encoder ->
    transpose -> cat(onehot) -> prompt_kernel -> transpose, then decode via
    ``m.decoding.rnnt_decoder_predictions_tensor``; for streaming it drives
    ``cache_aware_stream_step`` then applies the prompt and decodes). We do NOT
    use the lhotse ``transcribe(target_lang=...)`` path because it needs per-cut
    language metadata our bare wav fixtures lack.
  * Ours            -- the built ``parakeet-cli`` on the converted gguf with
    ``--lang <lang>`` (offline) and ``--lang <lang> --stream``.
  * WER             -- ``asr_metrics.wer(nemo_text, ours_text)``.

Prints a per-(clip,lang,mode) table and exits nonzero if any WER > 0.

Usage:
  .venv-nemotron/bin/python scripts/e2e_nemo_compare.py \\
    --nemo models/nemotron/model.nemo --gguf /tmp/nemotron.gguf \\
    --cli ./build/examples/cli/parakeet-cli \\
    --clips tests/fixtures/speech.wav,tests/fixtures/clip.wav \\
    --langs en,de,es,ja-JP,auto --mode both
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from asr_metrics import wer  # noqa: E402
from gen_nemo_baseline import compute_prompt_reference  # noqa: E402


def load_nemo(nemo_path: str):
    """Restore the NeMo model once (eval, dither off for determinism)."""
    from nemo.collections.asr.models import ASRModel

    if pathlib.Path(nemo_path).exists():
        m = ASRModel.restore_from(nemo_path, map_location="cpu")
    else:
        m = ASRModel.from_pretrained(nemo_path, map_location="cpu")
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    return m


def _run_cli(cli: str, gguf: str, wav: str, lang: str, stream: bool) -> str:
    cmd = [cli, "transcribe", "--model", gguf, "--input", wav]
    if lang:
        cmd += ["--lang", lang]
    if stream:
        cmd += ["--stream"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"parakeet-cli failed (exit {proc.returncode}) for "
            f"lang={lang} stream={stream}:\n{proc.stderr.strip()}"
        )
    out = proc.stdout
    if stream:
        # Take the authoritative final line: "[stream:final] <text>".
        final = ""
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("[stream:final]"):
                final = line[len("[stream:final]"):].strip()
        return final
    # Offline prints exactly the transcript; take the last non-empty line.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--nemo", required=True, help="NeMo .nemo checkpoint (or HF id)")
    ap.add_argument("--gguf", required=True, help="converted nemotron gguf")
    ap.add_argument("--cli", required=True, help="path to parakeet-cli")
    ap.add_argument("--clips", required=True, help="comma-separated wav paths")
    ap.add_argument("--langs", default="en,de,es,ja-JP,auto",
                    help="comma-separated target locales")
    ap.add_argument("--mode", default="both", choices=["offline", "stream", "both"])
    args = ap.parse_args()

    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    modes = ["offline", "stream"] if args.mode == "both" else [args.mode]

    print(f"Loading NeMo model from {args.nemo} ...", flush=True)
    m = load_nemo(args.nemo)

    rows = []      # (clip, lang, mode, wer, nemo_text, ours_text)
    any_fail = False

    for clip in clips:
        for lang in langs:
            ref = compute_prompt_reference(m, clip, lang)
            nemo_offline = ref["rnnt_text"]
            nemo_stream = ref["stream_text"]
            for mode in modes:
                if mode == "stream" and nemo_stream is None:
                    print(f"  SKIP stream {clip} {lang}: model not cache-aware")
                    continue
                nemo_text = nemo_offline if mode == "offline" else nemo_stream
                ours_text = _run_cli(
                    args.cli, args.gguf, clip, lang, stream=(mode == "stream")
                )
                w = wer(nemo_text, ours_text)
                if w > 0.0:
                    any_fail = True
                rows.append((clip, lang, mode, w, nemo_text, ours_text))

    # ---- Table ----
    print()
    print("=" * 78)
    print("E2E NeMo-vs-parakeet.cpp  (WER 0.0 = byte-for-byte parity)")
    print("=" * 78)
    print(f"{'clip':28s} {'lang':8s} {'mode':8s} {'WER':>8s}  status")
    print("-" * 78)
    for clip, lang, mode, w, nemo_text, ours_text in rows:
        name = pathlib.Path(clip).name
        status = "OK" if w == 0.0 else "MISMATCH"
        print(f"{name:28s} {lang:8s} {mode:8s} {w:8.4f}  {status}")
        if w > 0.0:
            print(f"    nemo: {nemo_text!r}")
            print(f"    ours: {ours_text!r}")
    print("-" * 78)
    print(f"{len(rows)} rows; {'ALL PARITY (WER 0.0)' if not any_fail else 'FAILURES PRESENT'}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
