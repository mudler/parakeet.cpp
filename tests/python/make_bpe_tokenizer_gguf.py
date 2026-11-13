#!/usr/bin/env python3
"""ctest setup: convert tokenizer.json (env PARAKEET_BPE_TOKENIZER_JSON) to a
tokenizer GGUF at argv[1]. Exits 77 (skip) when the file or the `gguf`
package is missing.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "scripts", "convert_bpe_tokenizer.py")


def main():
    src = os.environ.get("PARAKEET_BPE_TOKENIZER_JSON", "")
    if not src or not os.path.isfile(src):
        print("PARAKEET_BPE_TOKENIZER_JSON not set or missing; skip")
        return 77
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("gguf package missing; skip")
        return 77
    cmd = [sys.executable, SCRIPT, src, sys.argv[1]]
    cfg = os.path.join(os.path.dirname(src), "tokenizer_config.json")
    if os.path.isfile(cfg):
        cmd += ["--tokenizer-config", cfg]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
