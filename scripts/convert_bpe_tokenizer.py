#!/usr/bin/env python3
"""Convert a byte-level BPE tokenizer.json into a tokenizer-only GGUF file.

The metadata keys follow the llama.cpp BPE layout, so a model converter can
write the same keys into a full model GGUF:

    tokenizer.ggml.model            "gpt2"
    tokenizer.ggml.pre              "tekken"
    tokenizer.ggml.pre_pattern      the Split regex of tokenizer.json (informative)
    tokenizer.ggml.tokens           string[]  byte-level form; added tokens verbatim
    tokenizer.ggml.token_type       int32[]   1 normal, 3 special added, 4 other added
    tokenizer.ggml.merges           string[]  "left right", in rank order
    tokenizer.ggml.bos_token_id / eos_token_id / unknown_token_id / padding_token_id
    tokenizer.ggml.ignore_merges    bool      (not a llama.cpp key)

Ids at or above the vocabulary size (for example the Nemotron audio
placeholder) are appended to the token list, so the list has one entry per id.

Only the tekken pre-tokenizer is supported. The script stops when the Split
pattern, the ByteLevel settings or the model type differ from what
src/bpe_tokenizer.cpp implements.

    python scripts/convert_bpe_tokenizer.py tokenizer.json tokenizer.gguf \
        [--tokenizer-config tokenizer_config.json] [--eos-id 11]
"""
import argparse
import json
import sys

import numpy as np

try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(77)

TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

TYPE_NORMAL, TYPE_CONTROL, TYPE_USER_DEFINED = 1, 3, 4


def check_supported(tj):
    model = tj["model"]
    if model.get("type") != "BPE":
        sys.exit("model type is %r, only BPE is supported" % model.get("type"))
    if tj.get("normalizer") is not None:
        sys.exit("a normalizer is set; not supported")
    for key in ("unk_token", "continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(key):
            sys.exit("model.%s is set; not supported" % key)
    if model.get("byte_fallback") or model.get("dropout"):
        sys.exit("byte_fallback and dropout are not supported")
    pre = tj["pre_tokenizer"]
    seq = pre["pretokenizers"] if pre.get("type") == "Sequence" else [pre]
    if len(seq) != 2 or seq[0].get("type") != "Split" or seq[1].get("type") != "ByteLevel":
        sys.exit("pre_tokenizer must be Split then ByteLevel")
    sp, bl = seq
    if sp["pattern"].get("Regex") != TEKKEN_PATTERN:
        sys.exit("Split pattern is not the tekken pattern implemented in C++:\n%s" % sp["pattern"])
    if sp.get("behavior") != "Isolated" or sp.get("invert"):
        sys.exit("Split behavior must be Isolated, not inverted")
    if bl.get("use_regex") or bl.get("add_prefix_space"):
        sys.exit("ByteLevel must have use_regex false and add_prefix_space false")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer_json")
    ap.add_argument("out_gguf")
    ap.add_argument("--tokenizer-config", help="tokenizer_config.json, for bos/eos/unk/pad token names")
    ap.add_argument("--eos-id", type=int, help="override the EOS id (default: eos_token of the config)")
    args = ap.parse_args()

    with open(args.tokenizer_json, encoding="utf-8") as f:
        tj = json.load(f)
    check_supported(tj)
    model = tj["model"]
    vocab = model["vocab"]
    added = tj["added_tokens"]

    n = max(max(vocab.values()), max((a["id"] for a in added), default=-1)) + 1
    tokens = [None] * n
    types = [TYPE_NORMAL] * n
    for tok, i in vocab.items():
        tokens[i] = tok
    for a in added:
        i = a["id"]
        if tokens[i] is not None and tokens[i] != a["content"]:
            sys.exit("added token %d differs from the vocabulary entry" % i)
        tokens[i] = a["content"]
        types[i] = TYPE_CONTROL if a["special"] else TYPE_USER_DEFINED
        for flag in ("lstrip", "rstrip", "single_word", "normalized"):
            if a.get(flag):
                sys.exit("added token %r sets %s; not supported" % (a["content"], flag))
    holes = [i for i, t in enumerate(tokens) if t is None]
    if holes:
        sys.exit("ids without a token: %s" % holes[:10])

    merges = []
    for m in model["merges"]:
        merges.append(m if isinstance(m, str) else m[0] + " " + m[1])

    ids = {}
    if args.tokenizer_config:
        with open(args.tokenizer_config, encoding="utf-8") as f:
            cfg = json.load(f)
        for key, name in (("bos", "bos_token"), ("eos", "eos_token"), ("unk", "unk_token"), ("pad", "pad_token")):
            val = cfg.get(name)
            if isinstance(val, dict):
                val = val.get("content")
            if val is not None:
                if val not in tokens:
                    sys.exit("%s %r is not a token" % (name, val))
                ids[key] = tokens.index(val)
    if args.eos_id is not None:
        ids["eos"] = args.eos_id

    w = gguf.GGUFWriter(args.out_gguf, "bpe-tokenizer")
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "tekken")
    w.add_string("tokenizer.ggml.pre_pattern", TEKKEN_PATTERN)
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", types)
    w.add_array("tokenizer.ggml.merges", merges)
    w.add_bool("tokenizer.ggml.ignore_merges", bool(model.get("ignore_merges")))
    for key, kv in (("bos", "bos_token_id"), ("eos", "eos_token_id"), ("unk", "unknown_token_id"), ("pad", "padding_token_id")):
        if key in ids:
            w.add_uint32("tokenizer.ggml.%s" % kv, ids[key])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s: %d tokens, %d merges, %d added" % (args.out_gguf, n, len(merges), len(added)))


if __name__ == "__main__":
    main()
