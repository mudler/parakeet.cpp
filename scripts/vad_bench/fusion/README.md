# Silero plus head fusion experiment

Offline experiment behind the section "Fusing Silero and the head (offline experiment)"
in [docs/vad-benchmarks.md](../../../docs/vad-benchmarks.md). It is not part of
parakeet.cpp: the fusion rules are a Python port of the segmenter, and no C++ code
implements them.

No audio, no probability files and no binary results are committed. The four small result
files are:

| File | What |
| --- | --- |
| `results.md` | All cross-validated tables for the Ultra head and the Redux head (precision, recall, F1 with bootstrap intervals, false alarms on noise-only clips, paired differences, noise-shift split, operating points at precision 99 percent or more) |
| `gap_results.md` | False alarms inside a 30 s noise stretch that sits in a file with speech. It is the printed output of `gapeval.py` with a longer heading |
| `pr_points.csv` | Points of the precision and recall frontiers (the data of the plot that `pr_curves.py` draws) |
| `ted_results.txt` | Output of `ted.py` on a 600 s talk: share of speech and agreement with Silero and with the head |

## How it works

Every script runs from one working directory (the "work directory") and reads and writes
files there with relative names: `data/`, `clips/`, `clips.json`, `gap.json`,
`probs.npz`, `probs_gap.npz`, `results.json`. All scripts need `numpy`, `soundfile`;
`run.py` and `gapeval.py` also need `scikit-learn`; `collect*.py` need `onnxruntime`;
`prep_libri.py` needs `datasets`; `pr_curves.py` needs `matplotlib`. Run them with
the work directory as the current directory, for example `python3 /path/to/fusion/run.py`
after copying or linking the scripts into it (the scripts import each other by name).

The models are given by environment variables, read by `collect.py`, `collect_gap.py`
and `ted.py`:

| Variable | Value |
| --- | --- |
| `SILERO_ONNX` | `silero_vad.onnx` of Silero VAD 6.2.3 |
| `PARAKEET_CLI` | built `parakeet-cli` (it must support `vad --probabilities`) |
| `ULTRA_GGUF` | ASR GGUF of `moondream/parakeet-ultra`, Q8_0 |
| `REDUX_GGUF` | ASR GGUF of `moondream/parakeet-redux`, packed (`keep`) |

## Steps

```
python3 prep_libri.py        # every 13th utterance of LibriSpeech test-clean -> data/libri.npy, data/spk.npy
python3 mkcorpus.py          # 342 speech clips (38 sets of 5 utterances x 9 conditions) + 96 noise-only 30 s clips
python3 mkgap.py             # 96 clips with a 30 s noise stretch between speech
python3 collect.py           # per-frame probabilities of Silero and both heads -> probs.npz
python3 collect_gap.py       # the same for the gap clips -> probs_gap.npz
python3 run.py               # all rules, thresholds and the learned models -> results.json
python3 report.py            # results.md (reads results.json)
python3 gapeval.py           # prints the gap tables
python3 pr_curves.py         # pr_curves.png and pr_points.csv
python3 detail.py            # boundary error and missed segments -> detail.json
python3 ted.py talk.wav      # speech share on one long talk (needs the clips and probs of the steps above)
```

The speaker folds are fixed by the seed in `mkcorpus.py`. `rules.py` holds the fusion
rules and the two-stage rule, `fl.py` the 10 ms grid, the post-processing and the
metrics, `analyze.py` the cross-validation and the bootstrap, `lib.py` the clip
builder and the noise.

## Missing inputs

- The TED talk used by `ted.py` is not part of this directory and is not fetched by
  these scripts; `../fetch_data.py` fetches one TED-LIUM talk. The word-time reference of
  the talk used in the main benchmark was lost, so the experiment could not score the
  fusion rules against a TED reference. `ted.py` only reports how much of the talk each
  rule calls speech and how well the rules agree with each other.
- `ted.py` fits its logistic regression on the synthetic clips, so it needs `clips.json`,
  `clips/` and `probs.npz` from the steps above.
- `analyze.py` and `report.py` read `results.json` (about 50 MB), which is not committed.
  Run `run.py` to make it. The random seeds are fixed, so a re-run should give the same tables
  on the same probabilities, but this was not checked on a second machine.
