#!/usr/bin/env python3
"""Check that tests/fixtures/bpe_tokenizer_parity.tsv still matches the HF
`tokenizers` encoding of the real tokenizer.json (env PARAKEET_BPE_TOKENIZER_JSON).
Exits 77 (skip) when the file or the `tokenizers` package is missing.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "fixtures", "bpe_tokenizer_parity.tsv")


def main():
    src = os.environ.get("PARAKEET_BPE_TOKENIZER_JSON", "")
    if not src or not os.path.isfile(src):
        print("PARAKEET_BPE_TOKENIZER_JSON not set or missing; skip")
        return 77
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("tokenizers package missing; skip")
        return 77
    tok = Tokenizer.from_file(src)
    total = bad = 0
    with open(FIXTURE, encoding="utf-8") as f:
        for line in f:
            text_json, ids = line.rstrip("\n").split("\t")
            text = json.loads(text_json)
            want = [int(x) for x in ids.split()]
            total += 1
            if tok.encode(text, add_special_tokens=False).ids != want:
                bad += 1
                print("MISMATCH:", text[:60])
    print("fixture vs HF tokenizers: %d/%d match" % (total - bad, total))
    return 1 if bad or total < 500 else 0


if __name__ == "__main__":
    sys.exit(main())
