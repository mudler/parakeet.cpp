# Byte-level BPE tokenizer and transcription prompt

`src/bpe_tokenizer.{hpp,cpp}` is a byte-level BPE tokenizer (encoder and
decoder) for LLM-decoder checkpoints, and the prompt builder for Nemotron 3.5
Transcribe. It sits beside `src/tokenizer.*`, which only detokenizes
SentencePiece pieces for the transducer models and is unchanged. No model
weights are needed.

## What it implements

The code does what the checkpoint's `tokenizer.json` specifies, and nothing more:

1. Added tokens (`<|im_start|>`, `<spk:0>`, `<|audio|>`, and so on) are matched as
   whole strings in the raw text, leftmost and longest first.
2. The text between them goes through the `Split` pre-tokenizer (the "tekken"
   regex, behavior `Isolated`). The regex is written out as a hand-coded matcher,
   with the same alternation order and backtracking as the regex engine.
3. `ByteLevel` with `use_regex` false maps each byte to a printable character
   (the GPT-2 table).
4. `ignore_merges` is true: a word that is one vocabulary entry is emitted as it
   is. Otherwise BPE merges run with a rank map and a priority queue (lowest
   rank first, then the leftmost pair), which is `O(n log n)` for a word of `n`
   bytes.
5. There is no normalizer, and no BOS or EOS is added.

Decoding maps the token characters back to bytes and repairs invalid UTF-8 with
U+FFFD (`decode_bytes` skips the repair).

`scripts/convert_bpe_tokenizer.py` refuses a `tokenizer.json` whose Split
pattern or settings differ from what the C++ code implements.

## GGUF keys

The converter writes the keys of the llama.cpp BPE layout, so a model converter
can write the same keys into a full model file.

| Key | Type | Meaning |
|---|---|---|
| `tokenizer.ggml.model` | string | `gpt2` |
| `tokenizer.ggml.pre` | string | `tekken` (the only supported value) |
| `tokenizer.ggml.pre_pattern` | string | the Split regex, informative only |
| `tokenizer.ggml.tokens` | string[] | byte-level form; added tokens verbatim; one entry per id |
| `tokenizer.ggml.token_type` | int32[] | 1 normal, 3 special added, 4 other added |
| `tokenizer.ggml.merges` | string[] | `left right`, in rank order |
| `tokenizer.ggml.bos_token_id`, `eos_token_id`, `unknown_token_id`, `padding_token_id` | uint32 | optional |
| `tokenizer.ggml.ignore_merges` | bool | not a llama.cpp key |

Token types 3 and 4 mark the added tokens. Tokens that start with `<spk:`,
`<ev:` or `<sty:` are tags.

```
python scripts/convert_bpe_tokenizer.py tokenizer.json tokenizer.gguf \
    --tokenizer-config tokenizer_config.json
```

```cpp
std::string err;
auto tok = pk::BpeTokenizer::load_gguf("tokenizer.gguf", &err);
std::vector<int32_t> ids = tok->encode("Hello world");
std::string text = tok->decode(ids);
```

## Prompt builder

`pk::build_transcription_prompt(tok, prompt, audio_rows)` renders the prompt as
`chat_template.jinja` does for one user turn with `enable_thinking` false:

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt} <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>
```

and returns its ids with the `<|audio|>` id (131072) repeated `audio_rows` times,
plus the index of the first audio row. With `audio_rows` 0 the audio part and
the space before it are left out. `pk::kDefaultTranscriptionPrompt` is
`default_transcription_prompt` from `processor_config.json`.

Generated ids are decoded with `tok->decode(ids, opt)`:

- It stops at the EOS id (11 for this model, read from the GGUF) and does not
  include it.
- `keep_tags` keeps the speaker, event and style tags in the text; by default
  they are removed. Do not use `strip_special_tokens` from `tokenizer.hpp` for
  this: it deletes every bracketed token, speaker tags included.
- Other special tokens are dropped unless `skip_special` is false.

## Unicode tables

The regex needs `\p{Lu}`, `\p{Ll}`, `\p{Lt}`, `\p{Lm}`, `\p{Lo}`, `\p{M}`,
`\p{N}` and `\s`. `src/unicode_props_data.hpp` is generated: about 3,200
`{first code point, class mask}` ranges, no ICU.

The tables are **Unicode 16.0.0**. Python's `unicodedata` (15.0) is not used,
because the HF `tokenizers` regex engine knows Unicode 16: comparing every code
point showed 4,975 characters added after 15.0 that it treats as letters or
marks and 80 digits. With the 16.0.0 tables the pre-tokenizer matched HF on
every code point in 13 different contexts.

To regenerate (needs network once, or a directory that holds the files):

```
python3 scripts/gen_unicode_tables.py --ucd-dir /path/to/ucd16 --fetch > src/unicode_props_data.hpp
```

The script reads `extracted/DerivedGeneralCategory.txt` and `PropList.txt` of
Unicode 16.0.0 and stops if the files are of another version. When a newer
`tokenizers` release moves to a newer Unicode version, bump `VERSION` in the
script and rerun the code point comparison (see below).

## Tests

- `test_bpe_tokenizer`: a tiny synthetic vocabulary (merges, `ignore_merges`,
  byte alphabet, added tokens, regex splitting with a non-ASCII letter and digit
  runs, decode, prompt, GGUF loader). Always runs.
- `test_bpe_tokenizer_real`: loads a GGUF made from the real `tokenizer.json`
  and checks exact id equality with `tests/fixtures/bpe_tokenizer_parity.tsv`,
  `decode(encode(x)) == x`, the prompt ids, tags, and timing of a 100 KB input.
- `check_bpe_tokenizer.py`: checks the fixture against HF `tokenizers`.

The last two skip (exit 77) unless `PARAKEET_BPE_TOKENIZER_JSON` points to the
checkpoint's `tokenizer.json` (and, for the Python check, `tokenizers` is
installed; for the GGUF setup test, `gguf`). Configure with
`-DPARAKEET_PY=/path/to/venv/bin/python` to use a venv that has them.

The fixture holds the rendered default prompt, all 1,001 added tokens and more
than 600 other strings (English, Italian, German, numerals, emoji, CJK, Hindi,
Arabic, code, whitespace runs, newlines, mixed scripts, a 20 KB paragraph). To
rebuild it:

```
python scripts/gen_bpe_tokenizer_fixture.py tokenizer.json tests/fixtures/bpe_tokenizer_parity.tsv
```
