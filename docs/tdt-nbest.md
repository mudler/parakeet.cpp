# TDT N-best decoding

`parakeet.cpp` has an opt-in offline beam decoder for TDT checkpoints. It
returns multiple complete transcript hypotheses without changing the existing
greedy decoder.

## Algorithm

`pk::tdt_beam_search` follows NVIDIA NeMo 2.7.3's
[`BeamTDTInfer.default_beam_search`](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.7.3/nemo/collections/asr/parts/submodules/tdt_beam_decoding.py#L346-L479):

1. Apply `log_softmax` independently to token and duration logits.
2. Expand the highest-scoring non-blank token-duration pairs.
3. Expand blank only with non-zero durations.
4. Merge paths with the same token sequence and final frame using `logaddexp`.
5. Sort by `score / (emitted_tokens + 1)` by default. The extra item is NeMo's
   leading blank/SOS sentinel.

The C++ loop additionally rejects non-finite log probabilities and a
zero-duration expansion that does not strictly reduce the accumulated score.
This prevents malformed logits or floating-point saturation from creating a
non-progress loop without truncating valid hypotheses.

There is one intentional edge-case difference from NeMo's reference loop.
When the duration beam contains only duration 0, blank cannot use that
duration. NeMo substitutes the smallest positive duration; this implementation
substitutes the highest-scoring positive duration so the legal blank expansion
preserves model score ordering. A model-independent regression covers this
case for `beam_size=1`.

The implementation is limited to the default sequence-level TDT beam search.
It does not add mAES/mALSD, an external language model, batched beam search, or
streaming beam search. [NeMo labels this strategy experimental and recommends
`malsd_batch`](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.7.3/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L513-L533);
this first implementation deliberately prioritizes a small, directly
comparable reference algorithm over a second decoder design.

## CLI

```sh
parakeet-cli transcribe \
  --model parakeet-tdt-0.6b-v3-q8_0.gguf \
  --input audio.wav \
  --decoder tdt \
  --beam-size 4 \
  --nbest 4
```

The command emits JSON:

```json
{
  "beam_size": 4,
  "score_norm": true,
  "frame_sec": 0.08,
  "hypotheses": [
    {
      "text": "A complete transcript.",
      "score": -0.941021,
      "normalized_score": -0.020912,
      "tokens": [
        {
          "id": 499,
          "frame": 4,
          "t": 0.32,
          "duration_frames": 1,
          "duration": 0.08
        }
      ]
    }
  ]
}
```

Use `--no-score-norm` to rank by raw accumulated score. `--nbest` defaults to
the beam size.

## C API

The additive entry points are:

```c
parakeet_capi_transcribe_path_nbest_json(...);
parakeet_capi_transcribe_pcm_nbest_json(...);
```

Both reuse a load-once `parakeet_ctx`, accept an optional language prompt, and
return the same JSON document as the CLI. Existing symbols and ABI version 5
are unchanged.

## Parity test

Generate an authoritative NeMo baseline:

```sh
.venv/bin/python scripts/gen_nemo_baseline.py \
  --model nvidia/parakeet-tdt_ctc-110m \
  --audio tests/fixtures/speech.wav \
  --tdt-beam-size 4 \
  --output /tmp/tdt_nbest_baseline.gguf
```

Then compare token IDs, token end frames, rank order, and raw scores:

```sh
PARAKEET_TEST_GGUF=/tmp/pk110m.gguf \
PARAKEET_TEST_TDT_NBEST_BASELINE=/tmp/tdt_nbest_baseline.gguf \
ctest --test-dir build --output-on-failure -R '^test_tdt_beam$'
```

The pull-request closed-loop job generates this baseline directly with NeMo
and runs the comparison.
