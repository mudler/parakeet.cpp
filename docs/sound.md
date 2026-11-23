# Sound-event detection

parakeet.cpp can tag everyday sounds (dog bark, glass breaking, applause,
alarms, music, speech, and the rest of the 527-class AudioSet ontology) using
[CED](https://github.com/RicherMans/CED) (Consistent Ensemble Distillation),
run through [ced.cpp](https://github.com/localai-org/ced.cpp), the same LocalAI-team
ggml port used standalone. This is separate from ASR: a CED GGUF loads into
its own context (a "tagger"), and `pk::CedTagger` / the sound-stream C-API are
the only code in this repository that talks to it.

CED GGUFs (tiny, mini, small, base; f16 and q8_0, base also f32) are published
in one collection at
[huggingface.co/mudler/ced-gguf](https://huggingface.co/mudler/ced-gguf).
`ced-tiny` at q8_0 is about 6 MB; `ced-base` at q8_0 is about 88 MB.

## Loading a tagger

`parakeet_capi_load` detects the GGUF's architecture and returns a context
that holds either an ASR model, a diarization model, or (for a CED GGUF) a
tagger. Passing a tagger context to the ASR or diarization entry points fails
with a clear "context holds a CED sound model" error, and vice versa.

```c
parakeet_ctx* tagger = parakeet_capi_load("ced-tiny-q8_0.gguf");
```

On the C++ side this is `pk::CedTagger::load(path)`, which exposes
`n_classes()`, `label(index)`, and a `scorer()` (a `pk::SoundScorer`: scores
one PCM window and fills one probability per class, multi-label, not a
softmax).

### Device

ced.cpp picks its own device independently of the rest of parakeet.cpp: on a
GPU build it picks the first GPU it finds and falls back to CPU, and
`CED_DEVICE` overrides that choice the same way `PARAKEET_DEVICE` does for
Parakeet (`CED_DEVICE=cpu` forces CPU; a device name such as `CUDA0`,
`Vulkan0`, or `MTL0` selects that device). `CED_DEVICE` and `PARAKEET_DEVICE`
are read separately, so a tagger and an ASR/diarization context in the same
process can land on different devices.

## Windowing and timing rules

`pk::SoundStream` (`src/sound_stream.hpp`) runs a tagger over live PCM as a
sliding window: every `hop_sec` seconds, the most recent `window_sec` seconds
are scored, and a class's score is compared against two thresholds to decide
whether an event is open:

- A class **opens** when a scored window's score for it is `>= on_threshold`.
- An open class **closes** when a scored window's score drops `< off_threshold`.
- A closed segment shorter than `min_duration_sec` is dropped.

Because a score only tells you "this class was present somewhere in the last
`window_sec`," segment boundaries follow a **one-hop** grid: the code places
the open boundary at the start of the newest hop inside the triggering window
(the latest boundary consistent with the score), and the close boundary at
the end of the oldest hop inside the triggering window (the earliest boundary
consistent with the score). For a sharp event edge that puts the boundary
within about one `hop_sec` of it. A score that rises or falls slowly crosses
the thresholds later or earlier than the true edge, so it can move a boundary
by more than one hop.

At end of stream (`is_last`), a final tail window is scored if at least one
CED patch's worth of audio (0.16 s) remains unscored, and every class still
open is closed at the current stream time.

## Defaults

```cpp
struct SoundOpts {
    float window_sec = 3.0f;
    float hop_sec = 1.0f;
    float on_threshold = 0.4f;
    float off_threshold = 0.3f;
    float min_duration_sec = 0.3f;
    int   top_k = 5;
};
```

CED was trained and evaluated on clips up to about 10 s. A live stream cannot
wait 10 s to say what it is hearing, so the question is how much accuracy a
short window gives up. `examples/cli/sound_window_eval.cpp` (built as
`sound-window-eval`) measures this directly: for each WAV, it takes the
whole-clip (up to 10 s) top-1 class as the reference, then slides windows of
1, 2, 3, and 5 s (hop = window / 2) over the same clip and reports the share
of windows whose top-1 matches the reference, and the share whose top-5
contains it.

```
sound-window-eval <ced.gguf> <list.txt>   # one WAV path per line
```

### Measurement

Data: all 2000 clips of [ESC-50](https://github.com/karoldvl/ESC-50) (5 s,
44.1 kHz, resampled to 16 kHz mono) plus the three
`third_party/ced.cpp/benchmarks/demo/clips` demo clips (rooster, thunder,
guitar; 6 s), for `ced-tiny-q8_0`; ESC-50 folds 1 and 2 (800 clips) plus the
same three demo clips for `ced-base-q8_0`. ESC-50 clips are 5 s, so a 5 s
window is the reference clip itself; that row is trivially 100% and is listed
only for shape, not as a real data point.

`ced-tiny-q8_0`, 2003 clips:

| window | windows | top-1 = clip top-1 | clip top-1 in top-5 |
|---|--:|--:|--:|
| 1 s | 18033 | 33.1% | 55.0% |
| 2 s | 8015 | 57.8% | 80.4% |
| 3 s | 4009 | 74.1% | 91.5% |
| 5 s | 2003 | 100.0% | 100.0% |

`ced-base-q8_0`, 803 clips:

| window | windows | top-1 = clip top-1 | clip top-1 in top-5 |
|---|--:|--:|--:|
| 1 s | 7233 | 39.2% | 62.3% |
| 2 s | 3215 | 60.7% | 83.5% |
| 3 s | 1609 | 75.1% | 93.0% |
| 5 s | 803 | 100.0% | 100.0% |

### Decision

The rule was: keep window 3 s / hop 1 s unless ced-base's 3 s windows agree
with the clip top-1 on fewer than 70% of windows, in which case switch to 5 s
/ hop 1 s. ced-base's 3 s row is 75.1%, above the 70% line, so the defaults
stay at window 3 s / hop 1 s. A 1 s window is noticeably worse (39.2% on
ced-base) and a 2 s window is a real step down too (60.7%); 3 s is the
shortest window that stays reasonably close to what the whole clip would say,
which is the point of picking a default for a live stream instead of always
waiting for the whole clip.

## Sound-stream C-API

`include/parakeet_capi.h`, additive since ABI v8:

```c
typedef struct {
    int   size;                    // sizeof(parakeet_sound_opts)
    float window_sec, hop_sec;
    float on_threshold, off_threshold, min_duration_sec;
    int   top_k;
} parakeet_sound_opts;
void parakeet_capi_sound_opts_default(parakeet_sound_opts* o);

typedef struct {
    int         class_index;
    const char* label;             // borrowed from the tagger ctx
    float       start, end, peak;  // seconds from stream start
} parakeet_sound_segment;

typedef struct parakeet_sound_stream parakeet_sound_stream;

parakeet_sound_stream* parakeet_capi_sound_stream_begin(parakeet_ctx* tagger,
                                                        const parakeet_sound_opts* o);
int  parakeet_capi_sound_stream_feed(parakeet_sound_stream* s, const float* pcm, int n,
                                     int is_last, parakeet_sound_segment** out, int* n_out);
int  parakeet_capi_sound_stream_active(parakeet_sound_stream* s,
                                       parakeet_sound_segment** out, int* n_out);
char* parakeet_capi_sound_stream_drain_scores_json(parakeet_sound_stream* s);
void  parakeet_capi_free_sound_segments(parakeet_sound_segment* segs);
void  parakeet_capi_sound_stream_free(parakeet_sound_stream* s);

int         parakeet_capi_num_classes(const parakeet_ctx* ctx);
const char* parakeet_capi_class_label(const parakeet_ctx* ctx, int index);

#define PARAKEET_MODEL_KIND_NONE        0
#define PARAKEET_MODEL_KIND_ASR         1
#define PARAKEET_MODEL_KIND_DIARIZATION 2
#define PARAKEET_MODEL_KIND_SOUND       3
int parakeet_capi_model_kind(const parakeet_ctx* ctx);
```

`parakeet_capi_sound_stream_begin` with `o = NULL` uses the defaults above.
`_feed` accepts any chunk size and returns the segments that closed during
that call (free with `parakeet_capi_free_sound_segments`); pass `is_last = 1`
on the final chunk to flush and close everything still open. `_active` reads
the segments still open right now, with `end` set to the current stream time.
`_drain_scores_json` returns the raw per-window top-k scores scored since the
previous drain, as
`[{"start":..,"end":..,"tags":[{"index":..,"label":..,"score":..}]}]`.

`parakeet_capi_model_kind` reports which kind of model a `parakeet_capi_load`
context holds (`PARAKEET_MODEL_KIND_NONE` on `NULL`), so a caller loading ASR,
diarization and tagger GGUFs through the same entry point can dispatch without
probing individual functions.
The stream keeps one entry per hop until it is drained, so the queue grows
with the stream: drain regularly, or set `top_k = 0` to keep no scores.

### C example

```c
#include "parakeet_capi.h"
#include <stdio.h>

void tag_stream(const char* ced_gguf, const float* pcm, int n_samples) {
    parakeet_ctx* tagger = parakeet_capi_load(ced_gguf);
    if (!tagger) { fprintf(stderr, "load failed\n"); return; }

    parakeet_sound_stream* s = parakeet_capi_sound_stream_begin(tagger, NULL);
    if (!s) { fprintf(stderr, "%s\n", parakeet_capi_last_error(tagger)); parakeet_capi_free(tagger); return; }

    const int chunk = 8000; // 0.5 s at 16 kHz
    for (int i = 0; i < n_samples; i += chunk) {
        const int n = (i + chunk <= n_samples) ? chunk : n_samples - i;
        const int is_last = (i + n >= n_samples);

        parakeet_sound_segment* segs = NULL;
        int n_segs = 0;
        if (parakeet_capi_sound_stream_feed(s, pcm + i, n, is_last, &segs, &n_segs) != 0) {
            fprintf(stderr, "%s\n", parakeet_capi_last_error(tagger));
            break;
        }
        for (int k = 0; k < n_segs; ++k)
            printf("%-24s %6.2f - %6.2f  peak %.2f\n",
                   segs[k].label, segs[k].start, segs[k].end, segs[k].peak);
        parakeet_capi_free_sound_segments(segs);
    }

    parakeet_capi_sound_stream_free(s);
    parakeet_capi_free(tagger);
}
```

## `parakeet-cli scene`

`scene` combines any mix of an ASR model, a diarization model and a CED
tagger into one time-ordered feed. At least one of `--model`, `--diar`,
`--sound` is required, plus `--input`:

```
parakeet-cli scene --model <asr.gguf> --diar <diar.gguf> --sound <ced.gguf> \
    --input <audio.wav> [--latency model|low|very_low|ultra_low] [--chunk-ms N] \
    [--lang <locale>] [--show-speech] [--json]
```

`--latency` picks the diarization streaming mode (see `docs/diarization.md`);
it has no effect without `--diar`. `--chunk-ms` (default 200, capped at 60000)
sets how much PCM is fed to the stream per step. `--show-speech` keeps plain
speech labels (`Speech`, `Speech synthesizer`, `Conversation`, and similar)
in the sound output; by default they are filtered out since they are
redundant with the ASR transcript. `--json` prints one
`parakeet_capi_scene_stream_feed_json` document per step instead of the
rendered text. `--lang` sets the language prompt of a prompt-conditioned ASR
model (see [ASR models in the scene](#asr-models-in-the-scene)); other models
ignore it.

Real output, all three models, `--latency low`, on a demo clip (two
LibriSpeech speakers, a rooster clip, then a second LibriSpeech excerpt):

```
$ parakeet-cli scene --model asr.gguf --diar diar.gguf --sound ced-base-q8_0.gguf \
    --latency low --input scene_demo.wav
[00:00.4 - 00:03.2]  Speaker 0: mister Quilter is the apostle of the middle classes, and
[00:03.6 - 00:05.4]  Speaker 0: we're glad to welcome his gospel.
[00:06.6 - 00:06.7]  Speaker 1: Well,
[00:07.2 - 00:09.4]  Speaker 1: I don't wish to see it any more, observed Phoebe,
[00:09.7 - 00:10.8]  Speaker 1: turning away her eyes
[00:11.4 - 00:12.6]  Speaker 1: it is certainly very like
[00:12.9 - 00:13.6]  Speaker 1: old portrait.
[00:14.7 - 00:16.2]  Speaker 0: Nor is Mr Quilter's
[00:16.4 - 00:18.5]  Speaker 0: manner less interesting than his matter.
[00:20.0 - 00:21.5]  Speaker 1: Well, I don't wish to see it anymore,
[00:22.0 - 00:23.6]  Speaker 1: observed Phoebe, turning away her.
[00:24.0 - 00:30.0]  (Fowl 0.58)
[00:24.0 - 00:30.0]  (Chicken, rooster 0.86)
[00:25.0 - 00:27.0]  (Cluck 0.46)
[00:26.0 - 00:30.0]  (Crowing, cock-a-doodle-doo 0.65)
[00:30.0 - 00:32.6]  Speaker 1: Well, I don't wish to see it any more, observed Phoebe,
[00:33.0 - 00:36.7]  Speaker 1: turning away her eyes it is certainly very like the old portrait
```

Each line is `[start - end]  <speaker + text | (label peak)>` in stream
order. `Speech synthesizer` is one of the labels filtered out by default
(CED tags clean narration with it at moderate confidence; it is treated as a
plain-speech label alongside `Speech`, `Conversation`, and the rest, so it
does not show up twice next to the transcript). Pass `--show-speech` to see
it. With only `--sound`, the output is the sound lines alone; with only
`--model` (no `--diar`), the utterance lines drop the `Speaker N:` prefix.
With `--diar` but no `--model`, there is no transcript, so each closed
speaker segment prints as `[start - end]  Speaker N`, in time order with
the sound lines. A segment line waits while an earlier-starting segment is
still open.

### ASR models in the scene

The scene does not stream the ASR. It transcribes windows of at least 4 s
with the offline model and commits the words that end at least 1 s before the
end of the window (`src/asr_committer.cpp`). The next window starts after the
last committed word. Any offline ASR model works, with these differences:

| | TDT (`parakeet-tdt-0.6b-v3`) | RNN-T with no durations (`nemotron-labs-unified-asr-0.6b`) |
|---|---|---|
| Word times | from the predicted token durations | from the token emission frames, corrected for this checkpoint (below) |
| Window guards | none (unchanged) | on (below) |
| Language | implicit | `--lang` / `parakeet_capi_scene_stream_begin_lang`, default `auto` |

To use the unified model, pass its GGUF as `--model`:

```
parakeet-cli scene --model nemotron-labs-unified-asr-0.6b-q8_0.gguf --diar diar.gguf \
    --sound ced.gguf --input <audio.wav> [--lang en]
```

A model with no duration head (`pk::Model::word_times_from_emission()`) gets
the window guards below in the scene, and the unified checkpoint also gets
corrected word times. TDT and CTC models get neither, and their scene output
does not change. Other RNN-T models without durations (for example
`parakeet-rnnt-0.6b`) get the guards, but they were not measured with them.

* **Word times (unified checkpoint only).** RNN-T tokens only have an
  emission frame, and the emission comes late. Against TDT word times on the same clips, the raw unified word
  starts are a median 0.24 s late, and most words are one 80 ms frame long.
  The scene moves each word start back 3 frames and each word end back 2
  frames, and extends a word end towards the next word start, at most 6 frames
  after the word start (`pk::refine_emission_word_times`). The speaker of a
  word comes from these times. The commit rule still uses the raw times.
  `transcribe --timestamps` is not changed: it keeps the raw times that match
  NeMo.
* **Resume in a pause.** When the next word heard starts at least 0.5 s after
  the last committed word, the next window starts at the quietest 40 ms of
  that pause (at most 1 s into it), not on the tail of the last word.
* **Retry an empty window.** The unified model can return no word at all for
  a window cut out of running speech, also in NeMo on the same window. When a
  window that is not silent gives no word, the scene transcribes it again with
  0.16 s of silence before it, and if it is still empty, splits it once at its
  quietest point in the middle third (parts of at least 2 s) and transcribes
  the two parts. That is at most 4 decodes and 3 times the audio of that
  window. When a window stayed empty after these retries, the next window over
  the same audio is not retried, so a long stretch of loud non-speech (music,
  noise) pays for the retries once.

Measured on CPU with the q8_0 unified model and the q8_0 Nemotron-3
diarization model: word error rate of the scene words. The reference is the
TDT v3 transcript of the source clips for the first two inputs (the TDT scene
output at `model` latency for the first, the offline TDT transcript of the two
source files for the second), and the source transcripts for the last two.

| Input | `--latency` | Unified before | Unified after | TDT v3 f16 (unchanged) |
|---|---|---|---|---|
| `two_speakers.wav` + 0.6 s + 4 s tones + 0.6 s + `speech.wav` (36.2 s) | model | 42.5% | 0.0% | 0.0% |
| same | low | 3.4% | 3.4% | 24.1% |
| `two_speakers.wav` + 0.6 s silence + `speech.wav` (31.6 s) | model | 43.2% | 1.1% | 1.1% |
| same | low | 6.8% | 3.4% | 30.7% |
| 3 FLEURS en + 2 FLEURS de test clips, 0.6 s apart (58.9 s) | model | 7.1% | 3.1% | 10.2% |
| same | low | 10.2% | 10.2% | 28.6% |
| English and German read speech (85.1 s), not used to choose the guards | model | 4.7% | 5.5% | |
| same | low | 9.4% | 7.0% | |

Word times in the scene against TDT on the same words (36.2 s input, `model`
latency): start error median 0.24 s, p90 0.48 s before; median 0.08 s, p90
0.23 s after. The median word length is 0.24 s after, 0.08 s before.

Limits of the unified model in the scene:

* The guards change which windows the model sees, so they can also change a
  correct word: on the 85.1 s held-out input at `model` latency one correct
  word became wrong (4.7% to 5.5%), while `low` latency improved (9.4% to
  7.0%).
* Loud non-speech costs more CPU: on a 58.4 s clip of noise and synthetic chords around
  speech, the scene ASR user CPU went from 12.2 s to 24.5 s at `model`
  latency and from 62.0 s to 65.2 s at `low` latency (whole run, diarization
  included). The transcript did not change.
* Words can still be lost where a window opens in running speech: the model
  drops the first words of such a window (also in NeMo). The `low` latency
  modes commit more often and open more windows in speech.
* The word times are an estimate. On 8 clips (5 English, and 3 German that
  were not used to choose the correction) the corrected starts are within
  0.08 s of TDT for half of the words, and within 0.24 s for 90%.
* `--lang` did not change the transcript on any of the 9 English and German
  clips we checked (offline), nor on `speech.wav` in the scene: the model
  found the language itself. The model card says the prompt matters most in
  low-latency streaming, where the first chunks hold little speech. Set it
  when you know the language; `auto` is the default.
* There is no end-of-utterance event: the scene text commits on the window
  rule only.

## Scene stream C-API

`include/parakeet_capi.h`, additive since ABI v8. One stream carries any mix
of an ASR context, a diarization context and a tagger context (at least one
is required); each `_feed_json` call returns everything the stream finalized
in that call, as one JSON document:

```c
typedef struct {
    int size;                    // sizeof(parakeet_scene_opts)
    int diar_latency;            // PARAKEET_DIAR_LATENCY_*, used only with a diar ctx
    parakeet_sound_opts sound;   // used only with a tagger ctx
    int flags;                   // reserved, must be 0
    float speaker_accept_threshold;   // speaker fields: see docs/speaker.md
    float speaker_margin;
    float speaker_min_voice_sec;
    float speaker_refresh_sec;
    float speaker_max_voice_sec;
} parakeet_scene_opts;
void parakeet_capi_scene_opts_default(parakeet_scene_opts* o);

typedef struct parakeet_scene_stream parakeet_scene_stream;

parakeet_scene_stream* parakeet_capi_scene_stream_begin(parakeet_ctx* asr, parakeet_ctx* diar,
                                                         parakeet_ctx* tagger,
                                                         const parakeet_scene_opts* o);
parakeet_scene_stream* parakeet_capi_scene_stream_begin_lang(parakeet_ctx* asr, parakeet_ctx* diar,
                                                              parakeet_ctx* tagger, parakeet_ctx* speaker,
                                                              parakeet_speaker_registry* registry,
                                                              const parakeet_scene_opts* o,
                                                              const char* target_lang);
char* parakeet_capi_scene_stream_feed_json(parakeet_scene_stream* s, const float* pcm, int n,
                                           int is_last);
char* parakeet_capi_scene_stream_drain_scores_json(parakeet_scene_stream* s);
const char* parakeet_capi_scene_stream_last_error(parakeet_scene_stream* s);
void  parakeet_capi_scene_stream_free(parakeet_scene_stream* s);
```

The context arguments are borrowed (same lifetime rule as `sas_stream` and
`sound_stream`): free the scene stream before freeing any of the contexts it
was given. Passing `NULL` for a context leaves that part out of the stream;
`diar_latency` and `sound` in `parakeet_scene_opts` are ignored when the
matching context is `NULL`.

To set the language prompt of a prompt-conditioned ASR model, use
`parakeet_capi_scene_stream_begin_lang` (additive, ABI v11). It takes the
arguments of `parakeet_capi_scene_stream_begin_speaker` plus `target_lang`, for
example `"en"` or `"auto"`. `NULL` or `""` keeps the model default; models
without a prompt ignore it. An unknown value returns `NULL` with the message in
`parakeet_capi_last_error` of the ASR context. `parakeet_scene_opts` does not
change, so callers and bindings built for the earlier layout keep working.

Each `_feed_json` document has the shape:

```json
{"t":0.600,
 "utterances":[{"speaker":0,"text":"mister Quilter is","start":0.4,"end":1.6,"conf":0.98}],
 "words":[{"text":"mister","start":0.4,"end":0.6,"conf":0.99,"speaker":0}],
 "speakers":[{"speaker":0,"start":0.0,"end":0.6}],
 "sounds":[{"index":365,"label":"Chicken, rooster","start":24.0,"end":30.0,"peak":0.86}],
 "active":{"speakers":[{"speaker":0,"start":0.6}],
           "sounds":[{"index":365,"label":"Chicken, rooster","start":24.0,"end":26.0,"peak":0.7}]}}
```

`utterances`, `words` and `speakers` are the closed diarized ASR results for
this call (empty parts if the matching context is `NULL`); `sounds` are
sound-event segments that closed this call; `active` holds the speaker and
sound segments still open, with `end`/`peak` as of the current stream time.
`t` is the stream time consumed so far. `parakeet_capi_scene_stream_drain_scores_json`
returns the same shape `parakeet_capi_sound_stream_drain_scores_json` does
(`"[]"` without a tagger). The same rule applies: drain regularly, or set
`sound.top_k = 0` to keep no scores.

```c
#include "parakeet_capi.h"
#include <stdio.h>

void run_scene(const char* asr_gguf, const char* diar_gguf, const char* ced_gguf,
               const float* pcm, int n_samples) {
    parakeet_ctx* asr = parakeet_capi_load(asr_gguf);
    parakeet_ctx* diar = diar_gguf ? parakeet_capi_load(diar_gguf) : NULL;
    parakeet_ctx* tagger = ced_gguf ? parakeet_capi_load(ced_gguf) : NULL;

    parakeet_scene_opts o;
    parakeet_capi_scene_opts_default(&o);
    o.diar_latency = PARAKEET_DIAR_LATENCY_LOW;

    parakeet_scene_stream* s = parakeet_capi_scene_stream_begin(asr, diar, tagger, &o);
    if (!s) { fprintf(stderr, "scene_stream_begin failed\n"); return; }

    const int chunk = 3200; // 200 ms at 16 kHz
    for (int i = 0; i < n_samples; i += chunk) {
        const int n = (i + chunk <= n_samples) ? chunk : n_samples - i;
        const int is_last = (i + n >= n_samples);

        char* doc = parakeet_capi_scene_stream_feed_json(s, pcm + i, n, is_last);
        if (!doc) { fprintf(stderr, "%s\n", parakeet_capi_scene_stream_last_error(s)); break; }
        printf("%s\n", doc);
        parakeet_capi_free_string(doc);
    }

    parakeet_capi_scene_stream_free(s);
    if (tagger) parakeet_capi_free(tagger);
    if (diar) parakeet_capi_free(diar);
    parakeet_capi_free(asr);
}
```

## Server: `--sound-model`

`parakeet-server` accepts `--sound-model <ced.gguf>`, a local path to a CED
GGUF. With it set, a `verbose_json` transcription response gains a
`sound_events` array (`{"label","start","end","score"}` per event); `json`
and `text` responses are unchanged, and the sound pass only runs for
`verbose_json` requests. See `examples/server/README.md` for the full option
list.

## GPU

`test_ced_parity`, `test_sound_stream`, `test_sound_capi`, `test_scene_stream`,
`test_combined_offline`, `test_streaming_diarization`, `test_asr_committer`
and `parakeet-cli scene` (all three models, `--latency low`, on the demo clip
used above) were run on three GPU backends, staged and built through the `rc`
fleet (Vulkan on `strix:gpu0`, CUDA on `dgx:gpu0`) and over SSH (Metal on an
M4 Mac). All three matched the CPU transcript of the same build word for
word. Those runs predate the ASR changes that release non-speech audio,
which changed where the last transcript lines above break. Sound scores and
boundaries vary by low single hundredths and by hop-width ordering between
backends, as expected of independent floating-point runs.

| Device | Backend | `-LE model` | sound/scene ctest set | `test_ced_parity` | Scene demo wall time |
|---|---|---|---|---|---|
| strix:gpu0 (AMD Radeon 8060S, Vulkan0) | Vulkan | 21/22 (`server_e2e` not run) | 8/8 | pass | 2.3 s |
| dgx:gpu0 (NVIDIA GB10, CUDA0) | CUDA | 21/22 (`server_e2e` not run) | 5/8 (3 known teardown crashes, see below) | pass | 2.8 s |
| Apple M4 (MTL0) | Metal | 22/22 | 5/8 (3 known teardown crashes, see below) | pass | 9.5 s |

One known issue came out of these runs. It is not a regression in this
change:

- **`test_combined_offline`, `test_streaming_diarization` and
  `test_scene_stream` abort on process exit on CUDA and Metal, not Vulkan.**
  Each test holds more than one GPU-backed context in one process (ASR +
  diarization, or ASR + diarization + a CED tagger). All assertions print
  PASS first; the abort comes later, during static teardown. The
  process-global backend's allocator is freed by a static destructor after
  the GPU context it belongs to is already gone: on CUDA the backtrace is
  `ggml_gallocr_free -> ggml_backend_buffer_free -> cudaFree`, which fails
  with `CUDA error: driver shutting down`; on Metal,
  `ggml_metal_device_free` asserts `[rsets->data count] == 0`.
  `test_combined_offline` and `test_streaming_diarization` predate the
  sound-events work and abort the same way on the base branch, so this is
  not new with the scene tests. `parakeet-cli` does not hit it: it calls
  `pk::shutdown_backend()` before it returns from `main`, while the GPU
  context is still alive. No test tolerance or code was changed for it.
