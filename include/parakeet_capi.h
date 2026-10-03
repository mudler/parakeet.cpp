#ifndef PARAKEET_CAPI_H
#define PARAKEET_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for parakeet.cpp — designed for dlopen / cgo / purego (LocalAI).
//
// All functions are extern "C" and never let a C++ exception cross the
// boundary. The model is loaded ONCE into an opaque `parakeet_ctx` and reused
// across transcribe calls. Returned strings are malloc'd UTF-8 owned by the
// caller and must be released with parakeet_capi_free_string.

// Opaque transcription context (wraps a loaded model + last-error buffer).
typedef struct parakeet_ctx parakeet_ctx;

// ABI version of this header/implementation. Bump on any breaking change to the
// function signatures or semantics below.
//
// v3: added the target_lang variants (parakeet_capi_transcribe_path_lang,
//     parakeet_capi_transcribe_pcm_lang, parakeet_capi_stream_begin_lang,
//     parakeet_capi_transcribe_pcm_batch_json_lang,
//     parakeet_capi_transcribe_pcm_batch_lang) for multilingual
//     prompt-conditioned (nemotron) models. The original non-lang entry points
//     are unchanged and delegate with the model default language.
//
// v4: added the streaming JSON entry points (parakeet_capi_stream_feed_json,
//     parakeet_capi_stream_finalize_json) that surface per-word timestamps
//     (start/end/conf) plus frame_sec alongside the newly-finalized text, and
//     added "frame_sec" to the transcribe_*_json documents. The original entry
//     points are unchanged.
//
// v5: the <EOU> (end of utterance) vs <EOB> (end of backchannel) distinction is
//     now visible across the C boundary. BREAKING semantics on the streaming
//     surface: parakeet_capi_stream_feed's `*eou_out` is now a bitmask
//     (PARAKEET_EVENT_EOU | PARAKEET_EVENT_EOB) instead of an any-event 0/1,
//     and the JSON "eou" field now means "an <EOU> fired" only, with a new
//     "eob" field beside it (in v4 both meant "an <EOU> OR <EOB> fired").
//     Added parakeet_capi_stream_drain_events (typed per-event records with
//     is_eob + timestamps, freed with parakeet_capi_free_events) and an
//     "events" array in the stream_feed_json / stream_finalize_json documents.
//
// v6: added parakeet_capi_transcribe_pcm_logits, exposing the CTC head's
//     log-prob matrix (row-major [T, vocab+1], already log-softmaxed) instead
//     of decoded text — for external LM/decoder stacks (e.g. pyctcdecode +
//     KenLM) that need the raw distribution rather than this library's own
//     greedy/beam decode. Freed with the new parakeet_capi_free_logits. The
//     original entry points are unchanged.
// v7: added speaker diarization (parakeet_capi_diarize_*), speaker-attributed
//     ASR (parakeet_capi_transcribe_and_diarize*, parakeet_capi_sas_stream_*)
//     for nvidia/Nemotron-3-Diarization. A parakeet_ctx now holds either an
//     ASR or a diarization model; parakeet_capi_load detects which. No
//     existing signatures changed.
//     Additive, same ABI: parakeet_capi_diarize_stream_begin_latency /
//     _time / _active and parakeet_capi_sas_stream_begin_latency (the model
//     card's 1.04 / 0.64 / 0.32 s streaming modes).
// v8: sound-event detection (CED), sound_stream_*, scene_stream_*; additive.
//     A CED GGUF loads into a third parakeet_ctx kind (a "tagger"); no
//     existing signatures changed.
// v9: speaker identification (voice-detect.cpp). A voice-detect GGUF loads
//     into a fourth parakeet_ctx kind (a "speaker" encoder); a
//     parakeet_speaker_registry holds enrolled voices; the scene stream and
//     speaker-attributed ASR can name diarized speakers. Additive: no
//     existing signature changed.
// v10: raw-embedding enroll (parakeet_capi_speaker_registry_add_embedding)
//      and diarize-only naming (parakeet_capi_diarize_named_pcm_json), for
//      callers that keep speaker embeddings themselves. Additive: no
//      existing signature changed.
// Standalone VAD (parakeet_capi_vad_*, parakeet_capi_vad_stream_*,
//      parakeet_capi_transcribe_path_json_vad*) and Silero VAD contexts are
//      additive and keep ABI v10: a caller that needs them checks for the symbols
//      (dlsym) or for PARAKEET_MODEL_KIND_VAD.
// Bundle GGUF (parakeet_capi_load_component, parakeet_capi_bundle_components_json,
//      parakeet_capi_load_error; docs/bundle.md) is additive and keeps ABI v10.
int parakeet_capi_abi_version(void);

// Load a GGUF model. Returns an owning context, or NULL on failure.
// The returned context must be released with parakeet_capi_free.
//
// A bundle GGUF (several models in one file, general.architecture
// "parakeet-bundle"; format in docs/bundle.md) loads its ASR component: the only
// component of kind "asr", or, if it has none, its only loadable component. A
// bundle with several candidates is refused with a message that lists them (see
// parakeet_capi_load_error); open one with parakeet_capi_load_component. Only
// the chosen component's tensors are read from disk.
parakeet_ctx* parakeet_capi_load(const char* gguf_path);

// Additive (no ABI bump): open one named component of a bundle GGUF. The
// context is of the kind the component declares: "asr" gives an ASR context,
// "vad" a Silero VAD context (parakeet_capi_model_kind PARAKEET_MODEL_KIND_VAD).
// Returns NULL, with a reason in parakeet_capi_load_error, when `gguf_path` is
// not a bundle, has no such component, or has a component kind this build
// cannot load. A NULL or empty `component` behaves like parakeet_capi_load.
parakeet_ctx* parakeet_capi_load_component(const char* gguf_path, const char* component);

// Additive: the components of a bundle GGUF as a malloc'd JSON array (free with
// parakeet_capi_free_string), read from the header only. One object per
// component: {"name","kind","license","license_url","source","attribution",
// "changes","source_sha256","content_sha256","tensors","bytes"}. NULL (reason
// in parakeet_capi_load_error) when the file is not a bundle or its header is
// malformed or has a newer format version than this build reads.
char* parakeet_capi_bundle_components_json(const char* gguf_path);

// Additive: the reason the last parakeet_capi_load, parakeet_capi_load_component
// or parakeet_capi_bundle_components_json call on this thread failed. Valid
// until the next of those calls on the same thread; never NULL.
const char* parakeet_capi_load_error(void);

// Free a context obtained from parakeet_capi_load. Safe on NULL.
void parakeet_capi_free(parakeet_ctx* ctx);

// Transcribe a WAV file. `decoder` selects the head:
//   0 = default (by arch: transducer for tdt/rnnt/hybrid, CTC for ctc),
//   1 = ctc (force CTC head),
//   2 = tdt/rnnt (force the transducer head).
// On success returns a malloc'd, NUL-terminated UTF-8 transcript (free with
// parakeet_capi_free_string). On error returns NULL and sets the context's
// last error (see parakeet_capi_last_error).
char* parakeet_capi_transcribe_path(parakeet_ctx* ctx, const char* wav_path,
                                    int decoder);

// Transcribe in-memory mono float PCM (`samples`, length `n_samples`). If
// `sample_rate != 16000` the audio is linearly resampled to 16 kHz first.
// `decoder` is as in parakeet_capi_transcribe_path. On success returns a
// malloc'd UTF-8 transcript (free with parakeet_capi_free_string); on error
// returns NULL and sets the context's last error.
char* parakeet_capi_transcribe_pcm(parakeet_ctx* ctx, const float* samples,
                                   int n_samples, int sample_rate, int decoder);

// Like parakeet_capi_transcribe_path but selects the language prompt for
// multilingual (nemotron) models. `target_lang` is a locale string (e.g. "en",
// "de", "auto"); NULL or "" uses the model's default ("auto"). Ignored by
// non-prompt models. On an unknown locale (for a prompt model) returns NULL and
// sets the context's last error. parakeet_capi_transcribe_path delegates here
// with the model default.
char* parakeet_capi_transcribe_path_lang(parakeet_ctx* ctx, const char* wav_path,
                                         int decoder, const char* target_lang);

// Like parakeet_capi_transcribe_pcm but selects the language prompt (see
// parakeet_capi_transcribe_path_lang for `target_lang` semantics).
char* parakeet_capi_transcribe_pcm_lang(parakeet_ctx* ctx, const float* samples,
                                        int n_samples, int sample_rate, int decoder,
                                        const char* target_lang);

// Transcribe a batch of in-memory mono float PCM clips. `samples` is an array of
// `n_clips` pointers and `n_samples` an array of `n_clips` per-clip lengths; each
// clip is resampled to 16 kHz if `sample_rate != 16000`. `decoder` is as in
// parakeet_capi_transcribe_path (0=default,1=ctc,2=tdt/rnnt). On success returns
// 0 and fills `out` (a caller-allocated array of `n_clips` char*) with malloc'd
// NUL-terminated UTF-8 transcripts; release each with parakeet_capi_free_string.
// On error returns nonzero, sets the context's last error (see
// parakeet_capi_last_error), and leaves every out[] entry NULL: the caller owns
// nothing and has nothing to free.
int parakeet_capi_transcribe_pcm_batch(parakeet_ctx* ctx,
                                       const float* const* samples,
                                       const int* n_samples, int n_clips,
                                       int sample_rate, int decoder,
                                       char** out);

// Like parakeet_capi_transcribe_pcm_batch but selects the language prompt for
// multilingual (nemotron) models. ONE `target_lang` applies to the whole batch:
// a locale string (e.g. "en", "de", "auto"); NULL or "" uses the model's
// default ("auto"). Ignored by non-prompt models. On an unknown locale (for a
// prompt model) returns nonzero, sets the context's last error, and leaves
// every out[] entry NULL. parakeet_capi_transcribe_pcm_batch delegates here
// with the model default.
int parakeet_capi_transcribe_pcm_batch_lang(parakeet_ctx* ctx,
                                            const float* const* samples,
                                            const int* n_samples, int n_clips,
                                            int sample_rate, int decoder,
                                            const char* target_lang,
                                            char** out);

// Transcribe a WAV file returning a malloc'd UTF-8 JSON document with per-word
// and per-token timestamps + confidence (matching NeMo timestamps=True and the
// 'max_prob' confidence method). `decoder` is as in
// parakeet_capi_transcribe_path. The JSON shape is:
//
//   {"text":"...",
//    "frame_sec":0.080000,
//    "words":[{"w":"...","start":0.480,"end":0.640,"conf":0.9100}, ...],
//    "tokens":[{"id":123,"t":0.480,"conf":0.9100}, ...]}
//
// where "start"/"end"/"t" are seconds (3 decimals) and "conf" is the
// confidence in (0,1] (4 decimals). "frame_sec" is the encoder frame stride in
// seconds (hop_length * subsampling_factor / sample_rate); multiply a frame-unit
// segment gap threshold by it to get the seconds gap between words. The
// "w"/"text" strings are JSON-escaped
// (", \\, and control chars). On success returns the malloc'd string (free with
// parakeet_capi_free_string); on error returns NULL and sets the context's last
// error.
char* parakeet_capi_transcribe_path_json(parakeet_ctx* ctx, const char* wav_path,
                                         int decoder);

// Like parakeet_capi_transcribe_path_json, but long audio is cut at pauses found
// by the model's own VAD head into segments of at most 30 s, and the segments are
// transcribed one by one (word/token times are relative to the whole file). Audio
// of 30 s or less gives the same document as the plain function. Returns NULL and
// sets the context's last error to "model has no VAD head" when the model has no
// VAD head. It always uses the default segmenter options (30 s cap, threshold
// 0.5). Additive; no ABI bump.
char* parakeet_capi_transcribe_path_json_vad(parakeet_ctx* ctx, const char* wav_path,
                                             int decoder);

// Standalone voice-activity detection (additive; no ABI bump). Returns the
// speech segments as JSON, without transcribing. The context can hold either
// kind of detector:
//   * an ASR model with a VAD head (moondream parakeet-ultra / -redux): the head
//     gives one probability per 80 ms frame;
//   * a Silero VAD model (GGUF with general.architecture "silero_vad", loaded
//     with parakeet_capi_load, parakeet_capi_model_kind == PARAKEET_MODEL_KIND_VAD):
//     one probability per 32 ms chunk.
// Other models set the context's last error to "model has no VAD head" and
// return NULL. NULL is also returned for a bad argument or option (see
// last_error). Free the result with parakeet_capi_free_string.
//
//   parakeet_capi_vad_pcm_json: `samples` is n_samples mono float PCM at
//     `sample_rate` Hz; times in the result are on the original timeline.
//     n_samples == 0 gives no segments. A Silero context takes 16000 and 8000 Hz
//     as they are; any other rate (and every rate for an ASR context) is
//     resampled to 16 kHz with the library's linear resampler first.
//   parakeet_capi_vad_path_json: reads a WAV file instead (resampled to 16 kHz).
//
// `options_json` may be NULL or "" (all defaults) or a flat JSON object:
//   "threshold"     frame is speech when p >= threshold; (0, 1]; default 0.5
//   "min_pause"     seconds; a silence at least this long separates regions
//                   (shorter gaps are merged); default 0.2 (Silero: 0.1)
//   "min_speech"    seconds; shorter speech runs are dropped; default 0.1
//                   (Silero: 0.25)
//   "speech_pad"    seconds >= 0; "speech" mode: each region is widened by this
//                   on both sides; default 0 (Silero: 0.03)
//   "max_segment"   seconds; segment cap in "segments" mode; default 30
//   "mode"          "speech" (default) or "segments"
//   "probabilities" true to add the per-frame probabilities; default false
// Unknown keys and out-of-range values are errors. The Silero defaults are the
// values of Silero's own get_speech_timestamps (docs/vad.md).
//
// Result (same shape for both kinds):
//   {"mode":"speech","duration":12.340,"frame_sec":0.080,"backend":"cpu",
//    "segments":[{"start":0.480,"end":3.200},...],
//    "probabilities":[0.0123,...]}      // only with "probabilities":true
// Times are seconds (3 decimals). "frame_sec" is 0.080 for the VAD head and
// 0.032 for Silero. "probabilities" has one value per frame_sec frame, starting
// at time 0 (the last Silero chunk is zero padded). "backend" is the compute
// device the model ran on.
//
// Modes. "speech" returns the speech regions after smoothing (gaps shorter
// than 0.1 s bridged, runs shorter than min_speech dropped, regions closer than
// min_pause merged, then padded). They are ordered and disjoint, for audio of
// any length, and silence is never included: this is what a VAD consumer
// wants. "segments" returns the cuts the transcriber uses in
// parakeet_capi_transcribe_path_json_vad: pieces of at most max_segment seconds
// cut at pauses, pieces without speech dropped, and audio of at most
// max_segment seconds returned whole as one segment even when it holds no
// speech. speech_pad does not apply to "segments".
//
// Backend: the same rules as the transcribe functions. The VAD head runs on the
// context's compute backend (the pool of parakeet_capi_set_concurrency, when
// set); a packed (ternary) Redux model is CPU only. A Silero context runs on the
// process backend, which runs one graph at a time. Safe to call from several
// threads, on one context or on several.
char* parakeet_capi_vad_pcm_json(parakeet_ctx* ctx, const float* samples, int n_samples,
                                 int sample_rate, const char* options_json);
char* parakeet_capi_vad_path_json(parakeet_ctx* ctx, const char* wav_path,
                                  const char* options_json);

// Like parakeet_capi_transcribe_path_json_vad, but the segmenter can take its
// probabilities from a Silero VAD context, so any ASR model (also those without
// a VAD head) can cut long audio. `ctx` is the ASR context. `vad_ctx` is a
// Silero context, or NULL to use the ASR model's own head (then the result is
// as parakeet_capi_transcribe_path_json_vad, with the options below). Options
// are the JSON object of parakeet_capi_vad_pcm_json; only "threshold",
// "min_pause", "min_speech" and "max_segment" are used here (the other keys are
// accepted and ignored). NULL or "" gives the defaults of the VAD in use.
// Audio of at most max_segment seconds (30 by default) is transcribed whole,
// without running the VAD. Word and token times are relative to the whole file.
// Returns the same document as parakeet_capi_transcribe_path_json. Errors set
// ctx's last error.
char* parakeet_capi_transcribe_path_json_vad_with(parakeet_ctx* ctx, parakeet_ctx* vad_ctx,
                                                  const char* wav_path, int decoder,
                                                  const char* options_json);

// Streaming Silero VAD (additive; no ABI bump). One stream per audio stream;
// the Silero context must outlive it and stay loaded. Use one stream from one
// thread at a time; different streams may run on different threads.
typedef struct parakeet_vad_stream parakeet_vad_stream;

// `vad` is a Silero context; `sample_rate` is 16000 or 8000 (no resampling in a
// stream). `options_json` as in parakeet_capi_vad_pcm_json, but "mode" must be
// "speech" (the default). NULL on error (see vad's last error).
parakeet_vad_stream* parakeet_capi_vad_stream_begin(parakeet_ctx* vad, int sample_rate,
                                                    const char* options_json);

// Feeds n_samples mono float PCM in [-1, 1] (any n_samples >= 0, also 0).
// is_last != 0 zero-pads the buffered partial chunk and closes any open region;
// after it, feed again only after parakeet_capi_vad_stream_reset. Returns a
// malloc'd JSON document (free with parakeet_capi_free_string), or NULL on error:
//   {"frame_sec":0.032,"first_frame":120,
//    "events":[{"type":"start","time":3.456},{"type":"end","time":5.120}],
//    "probabilities":[0.01,...]}        // only when begun with "probabilities":true
// "events" lists the speech starts and ends that became known in this call, in
// order. Times are seconds from the start of the stream (after the last reset),
// padded by speech_pad. An event is known late: a start once the speech has
// lasted min_speech, an end once the silence has lasted min_pause, so it can
// describe a time before the audio of this call. "probabilities" holds the
// chunks completed in this call; "first_frame" is the index of the first one
// (frame i covers [i * 0.032, (i + 1) * 0.032) seconds). The events equal the
// regions of mode "speech" for the whole audio (when speech_pad <= min_pause / 2).
// Splitting the audio into calls of any size gives the same probabilities.
char* parakeet_capi_vad_stream_feed_json(parakeet_vad_stream* s, const float* pcm,
                                         int n_samples, int is_last);
// Back to the initial state (empty buffers, time 0). 0 on success.
int   parakeet_capi_vad_stream_reset(parakeet_vad_stream* s);
void  parakeet_capi_vad_stream_free(parakeet_vad_stream* s);

// Batched transcription with timestamps, returning ONE malloc'd JSON string that
// is a JSON ARRAY of n_clips objects, each identical in shape to
// parakeet_capi_transcribe_path_json's document ({"text","words","tokens"}).
// samples_concat holds all clips' 16 kHz mono float samples concatenated;
// n_samples gives each clip's sample count; n_clips is the array length.
// decoder: 0=default,1=ctc,2=tdt. PRECONDITION (caller MUST uphold, not
// validated here): the sum of n_samples[0..n_clips) equals the number of floats
// in samples_concat. A larger sum reads out of bounds.
// Returns the JSON string on success (free with parakeet_capi_free_string), or
// NULL on error (see parakeet_capi_last_error).
char* parakeet_capi_transcribe_pcm_batch_json(parakeet_ctx* ctx,
                                              const float* samples_concat,
                                              const int* n_samples, int n_clips,
                                              int sample_rate, int decoder);

// Like parakeet_capi_transcribe_pcm_batch_json but selects the language prompt
// for multilingual (nemotron) models. ONE `target_lang` applies to the whole
// batch: a locale string (e.g. "en", "de", "auto"); NULL or "" uses the model's
// default ("auto"). Ignored by non-prompt models. On an unknown locale (for a
// prompt model) returns NULL and sets the context's last error.
// parakeet_capi_transcribe_pcm_batch_json delegates here with the model default.
char* parakeet_capi_transcribe_pcm_batch_json_lang(parakeet_ctx* ctx,
                                                   const float* samples_concat,
                                                   const int* n_samples, int n_clips,
                                                   int sample_rate, int decoder,
                                                   const char* target_lang);

// Offline TDT beam search returning ranked hypotheses as JSON. These are
// additive, TDT-only entry points; they do not change the existing greedy
// transcription functions or ABI version. `beam_size >= nbest >= 1`.
// `score_norm != 0` matches NeMo's default score/sequence-length ranking.
// `target_lang` has the same semantics as the other *_lang functions.
//
// JSON shape:
//   {"beam_size":4,"score_norm":true,"frame_sec":0.080000,
//    "hypotheses":[
//      {"text":"...","score":-12.3,"normalized_score":-0.45,
//       "tokens":[{"id":123,"frame":7,"t":0.560,
//                  "duration_frames":2,"duration":0.160}, ...]}
//    ]}
char* parakeet_capi_transcribe_path_nbest_json(
    parakeet_ctx* ctx, const char* wav_path,
    int beam_size, int nbest, int score_norm, const char* target_lang);

char* parakeet_capi_transcribe_pcm_nbest_json(
    parakeet_ctx* ctx, const float* samples, int n_samples, int sample_rate,
    int beam_size, int nbest, int score_norm, const char* target_lang);

// Run mel + encoder + CTC head on in-memory mono float PCM and return the
// log-prob matrix instead of decoded text, for callers that run their own
// external decoder (e.g. pyctcdecode + a KenLM n-gram LM + hotwords) on top of
// this library's CTC output rather than using parakeet.cpp's own greedy/beam
// decode. If `sample_rate != 16000` the audio is linearly resampled to 16 kHz
// first. Always runs the CTC head regardless of the model's preferred
// decoder — `decoder` is not a parameter here, unlike parakeet_capi_transcribe_pcm.
//
// On success returns 0, mallocs `*out_logits` to `(*out_T) * (*out_vocab_plus_1)`
// floats — row-major [T, vocab+1], i.e. out_logits[t*(*out_vocab_plus_1) + v],
// already log-softmaxed over the vocab axis — and sets `*out_T` /
// `*out_vocab_plus_1`. Free `*out_logits` with parakeet_capi_free_logits.
//
// On error returns nonzero. A NULL `ctx` or any NULL out-param pointer
// returns nonzero without writing through any pointer (nothing to zero
// safely). Otherwise (ctx and all three out-params valid, but e.g. no model,
// invalid samples buffer, the model has no CTC head, or OOM) sets the
// context's last error (see parakeet_capi_last_error) and leaves `*out_logits`
// NULL and `*out_T`/`*out_vocab_plus_1` 0 — the caller owns nothing and has
// nothing to free.
int parakeet_capi_transcribe_pcm_logits(parakeet_ctx* ctx, const float* samples,
                                        int n_samples, int sample_rate,
                                        float** out_logits, int* out_T,
                                        int* out_vocab_plus_1);

// Free a logits buffer previously returned by
// parakeet_capi_transcribe_pcm_logits. Safe on NULL.
void parakeet_capi_free_logits(float* logits);

// ---------------------------------------------------------------------------
// Streaming API (cache-aware streaming RNN-T, e.g. the EOU model
// nvidia/parakeet_realtime_eou_120m-v1). The stream session buffers incoming
// 16 kHz mono float PCM, runs the mel front end + cache-aware StreamingEncoder +
// carried RNN-T decoder, and surfaces newly-finalized text plus end-of-utterance
// (<EOU>) / backchannel (<EOB>) events. No C++ exception crosses the boundary.
// ---------------------------------------------------------------------------

// Opaque streaming session. Begun from a loaded context; the context (and its
// model) must outlive the stream. Free with parakeet_capi_stream_free.
typedef struct parakeet_stream parakeet_stream;

// Begin a streaming session over `ctx`'s model. Returns NULL on failure (e.g.
// the model is not a cache-aware streaming model) and sets the ctx last error.
parakeet_stream* parakeet_capi_stream_begin(parakeet_ctx* ctx);

// Begin a streaming session selecting the language prompt for multilingual
// (nemotron) prompt-conditioned models. `target_lang` is a locale string (e.g.
// "en", "de", "auto"); NULL or "" uses the model's default. Ignored by
// non-prompt models. Returns NULL on failure (not a streaming model, or an
// unknown locale) and sets the ctx last error. parakeet_capi_stream_begin
// delegates here with the model default.
parakeet_stream* parakeet_capi_stream_begin_lang(parakeet_ctx* ctx,
                                                 const char* target_lang);

// Bits for parakeet_capi_stream_feed's *eou_out mask. <EOU> = the user
// finished a complete utterance (a voice agent responds); <EOB> = the user
// finished a backchannel, a short acknowledgment like "uh-huh" while the other
// party speaks (a voice agent must NOT treat it as the user taking the turn).
#define PARAKEET_EVENT_EOU 1
#define PARAKEET_EVENT_EOB 2

// Feed a block of 16 kHz MONO float PCM (`pcm`, length `n_samples`). The session
// buffers the audio and decodes as full encoder chunks become available.
// Returns the newly-finalized text since the last call as a malloc'd UTF-8
// string (free with parakeet_capi_free_string) — "" (empty, non-NULL) if no new
// text was finalized this call, NULL only on error. <EOU>/<EOB> are stripped
// from the text and surfaced as events: if `eou_out` is non-NULL it is set to
// the bitwise OR of PARAKEET_EVENT_EOU / PARAKEET_EVENT_EOB for the event types
// that fired during this feed (0 if none). Per-event timestamps are available
// via parakeet_capi_stream_drain_events.
char* parakeet_capi_stream_feed(parakeet_stream* s, const float* pcm,
                                int n_samples, int* eou_out);

// Flush the end-of-stream tail: process any remaining buffered audio (the final
// chunk completes the streaming tail). Returns the final newly-finalized text
// (malloc'd; "" if none, NULL on error). After this the running transcript is
// complete. Does NOT fabricate an <EOU> NeMo's streaming would not emit.
char* parakeet_capi_stream_finalize(parakeet_stream* s);

// One <EOU>/<EOB> event emitted by the streaming decoder. <EOU> marks the end
// of a complete utterance (the user yielded the turn); <EOB> marks the end of a
// backchannel (a short acknowledgment like "uh-huh" while the other party
// speaks — a voice agent typically responds on <EOU> but must NOT treat <EOB>
// as the user taking the turn). time_sec is the absolute (stream-relative)
// emission time: encoder_frame * frame_sec.
typedef struct parakeet_stream_event {
    int   token;          // raw vocab id of the special token
    int   is_eob;         // 0 = <EOU> (end of utterance), 1 = <EOB> (backchannel)
    int   encoder_frame;  // absolute encoder-output frame index of the emission
    float time_sec;       // encoder_frame * frame_sec, seconds from stream start
} parakeet_stream_event;

// Drain the <EOU>/<EOB> events accumulated since the last drain. On success
// returns the event count (>= 0) and, when the count is nonzero, sets
// `*out_events` to a malloc'd array of that many records (release with
// parakeet_capi_free_events); `*out_events` is NULL when the count is 0.
// Returns -1 on error (NULL stream/out pointer) with `*out_events` NULL.
// The queue is shared with the JSON entry points: stream_feed_json /
// stream_finalize_json also drain it (into their "events" array), so use one
// style or the other per stream.
int parakeet_capi_stream_drain_events(parakeet_stream* s,
                                      parakeet_stream_event** out_events);

// Free an event array previously returned by parakeet_capi_stream_drain_events.
// Safe on NULL.
void parakeet_capi_free_events(parakeet_stream_event* events);

// Like parakeet_capi_stream_feed but returns a malloc'd UTF-8 JSON document
// instead of bare text:
//   {"text":"...","eou":0,"eob":0,"frame_sec":0.080000,
//    "events":[{"type":"eou","frame":31,"t":2.480}, ...],
//    "words":[{"w":"...","start":0.480,"end":0.640,"conf":0.9100}, ...]}
// "text" is the newly-finalized text since the last call ("" if none); "eou" is
// 1 iff an <EOU> fired during this feed and "eob" 1 iff an <EOB> fired (see
// parakeet_stream_event for the semantics — they are distinct turn-taking
// signals, not conflated); "frame_sec" is the encoder frame stride in seconds;
// "events" are the <EOU>/<EOB> events drained this call, each with "type"
// ("eou" = end of utterance, "eob" = backchannel), the absolute encoder frame
// and the emission time in seconds (frame * frame_sec); "words" are the words
// finalized this call with absolute (stream-relative) start/end seconds and
// 'min'-aggregate confidence (the same drain as the offline pk::group_words).
// Returns NULL only on error (see parakeet_capi_last_error). Free with
// parakeet_capi_free_string.
char* parakeet_capi_stream_feed_json(parakeet_stream* s, const float* pcm,
                                     int n_samples);

// Like parakeet_capi_stream_finalize but returns the same JSON document shape as
// parakeet_capi_stream_feed_json (flushing the end-of-stream tail; "eou" is
// typically 0 — finalize does not fabricate an <EOU>). Free with
// parakeet_capi_free_string; NULL only on error.
char* parakeet_capi_stream_finalize_json(parakeet_stream* s);

// Free a streaming session. Safe on NULL.
void parakeet_capi_stream_free(parakeet_stream* s);

// Free a string previously returned by parakeet_capi_transcribe_* /
// parakeet_capi_stream_* / parakeet_capi_diarize_* /
// parakeet_capi_transcribe_and_diarize_json. Safe on NULL.
void parakeet_capi_free_string(char* s);

// Human-readable description of the last error on `ctx`, or "" if none.
// The returned pointer is owned by the context and valid until the next call on
// it (or until parakeet_capi_free). Returns "" if `ctx` is NULL.
const char* parakeet_capi_last_error(parakeet_ctx* ctx);
// (See also "Concurrent requests" below for how last_error behaves when several
// threads use one context.)

// ---------------------------------------------------------------------------
// Concurrent requests (additive, ABI unchanged)
//
// By default a context runs one request at a time: calls from several threads
// are safe but queue behind one compute backend. parakeet_capi_set_concurrency
// turns on a pool of `backends` CPU backends for the context's ASR model, each
// with `threads_each` compute threads (<= 0: the default thread budget divided
// by `backends`, at least 1). After that, concurrent
// parakeet_capi_transcribe_* calls on the same context run in parallel, up to
// `backends` at a time; further callers wait for a free backend. The library
// starts no threads of its own: each call runs on its caller's thread and
// borrows one backend for its whole duration. Transcripts, token ids and
// timestamps are the same as with one backend.
//
// Choose backends * threads_each at most the number of physical cores: the
// compute thread teams spin, and oversubscribing the machine slows everything.
// Each backend also keeps its own graph buffers, so memory grows with
// `backends`.
//
// Returns the effective number of backends: 1 when `backends` <= 1 (the pool is
// removed and the process-wide thread count of the CLI applies, as before), and
// also 1 on a GPU device, which keeps a single backend. Returns 0 if `ctx` is
// NULL or has no ASR model (the context's last error is set in the second case).
// Call it while no request is running, for example right after load; calls
// already running finish on the old pool. Streaming sessions and diarization
// are not affected by the pool and keep their single-call semantics.
//
// last_error and threads: the message is written under a lock, so concurrent
// failures are safe. parakeet_capi_last_error(ctx) returns the message of the
// most recently finished call on that context, from any thread (a successful
// call clears it). The pointer stays valid until the next
// parakeet_capi_last_error call on that context or until the context is freed;
// copy the text if you need it longer. A caller that needs the exact error of
// its own call should use one context per thread. The transcribe functions keep
// returning NULL (or non-zero) on failure.
int parakeet_capi_set_concurrency(parakeet_ctx* ctx, int backends, int threads_each);

// ---------------------------------------------------------------------------
// Speaker diarization (nvidia/Nemotron-3-Diarization and compatible Sortformer
// models), ABI v7.
//
// A parakeet_ctx loaded from a diarization GGUF holds a diarization model
// instead of an ASR model (parakeet_capi_load detects the arch). The functions
// below are the only valid entry points for such a context; the transcribe_*
// and stream_* functions fail on it with a last_error message, and the
// diarize_* functions fail on an ASR context.
//
// Times are seconds from the start of the audio; speakers are 0-based indices
// in order of first appearance, up to the model's capacity (8).
// ---------------------------------------------------------------------------

// Offline diarization of a WAV file. Returns a malloc'd UTF-8 JSON document
// (free with parakeet_capi_free_string):
//   {"speakers":8,"segments":[{"speaker":0,"start":0.50,"end":5.52}, ...]}
// "speakers" is the model's capacity. Segments are sorted by start time then
// speaker, with times rounded to 10 ms. NULL on error (see last_error).
char* parakeet_capi_diarize_path(parakeet_ctx* ctx, const char* wav_path);

// Same for in-memory mono float PCM; resampled to 16 kHz when
// `sample_rate != 16000`.
char* parakeet_capi_diarize_pcm(parakeet_ctx* ctx, const float* samples,
                                int n_samples, int sample_rate);

// Speaker-attributed ASR ("who said what"): one utterance is a run of
// consecutive words from one speaker.
typedef struct parakeet_sas_result {
    int   speaker;   // 0-based speaker index, -1 = no diarized speaker overlaps
    char* text;      // utterance text (space-joined words), owned by the array
    float start;     // first word start (seconds)
    float end;       // last word end (seconds)
    float conf;      // min word confidence
} parakeet_sas_result;

// Run ASR (`asr_ctx`) and diarization (`diar_ctx`) on the same mono float PCM
// and assign each ASR word to the speaker whose segments overlap it most.
// On success returns 0 and sets *out (malloc'd array, free with
// parakeet_capi_free_sas_results) and *n_out; *out may be NULL when
// *n_out == 0. On error returns non-zero and sets last_error on the context
// that failed.
int parakeet_capi_transcribe_and_diarize(parakeet_ctx* asr_ctx, parakeet_ctx* diar_ctx,
                                         const float* samples, int n_samples,
                                         int sample_rate,
                                         parakeet_sas_result** out, int* n_out);

// Free an array from parakeet_capi_transcribe_and_diarize or
// parakeet_capi_sas_stream_feed, including every .text. Safe on NULL.
void parakeet_capi_free_sas_results(parakeet_sas_result* results, int n);

// JSON variant with per-utterance and per-word detail (free with
// parakeet_capi_free_string; NULL on error):
//   {"speakers":8,
//    "utterances":[{"speaker":0,"text":"hello world","start":0.12,"end":0.85,"conf":0.95}],
//    "words":[{"speaker":0,"text":"hello","start":0.12,"end":0.45,"conf":0.97}]}
char* parakeet_capi_transcribe_and_diarize_json(parakeet_ctx* asr_ctx, parakeet_ctx* diar_ctx,
                                                const float* samples, int n_samples,
                                                int sample_rate);

// --- Streaming diarization -------------------------------------------------
// NeMo cache-aware streaming (speaker cache + FIFO) over live 16 kHz mono
// float PCM. Speaker indices stay consistent across chunks. The stream
// borrows `diar_ctx`: free the stream first, and do not use one context from
// two threads at once.
//
// Latency modes (the Nemotron-3-Diarization model card presets). Latency is
// the audio buffered before a chunk runs: (chunk + look-ahead) x 80 ms.
#define PARAKEET_DIAR_LATENCY_MODEL     0  // checkpoint config: 21.12 s chunks
#define PARAKEET_DIAR_LATENCY_LOW       1  // 1.04 s
#define PARAKEET_DIAR_LATENCY_VERY_LOW  2  // 0.64 s
#define PARAKEET_DIAR_LATENCY_ULTRA_LOW 3  // 0.32 s

typedef struct parakeet_diar_segment {
    int   speaker;
    float start;   // seconds from stream start
    float end;
} parakeet_diar_segment;

typedef struct parakeet_diar_stream parakeet_diar_stream;

// Begin a stream in the checkpoint's own configuration
// (PARAKEET_DIAR_LATENCY_MODEL). NULL on error (last_error on diar_ctx).
parakeet_diar_stream* parakeet_capi_diarize_stream_begin(parakeet_ctx* diar_ctx);

// Begin a stream in one of the PARAKEET_DIAR_LATENCY_* modes.
parakeet_diar_stream* parakeet_capi_diarize_stream_begin_latency(parakeet_ctx* diar_ctx,
                                                                 int latency);

// Samples buffered before a chunk runs (the input latency). 0 on NULL.
int parakeet_capi_diarize_stream_chunk_samples(parakeet_diar_stream* s);

// Seconds of audio diarized so far (trails the audio fed by up to the latency).
float parakeet_capi_diarize_stream_time(parakeet_diar_stream* s);

// Segments still open at the diarized time ("who is speaking now"), with `end`
// at parakeet_capi_diarize_stream_time. Same ownership as
// parakeet_capi_diarize_stream_feed. Returns 0, or non-zero on error.
int parakeet_capi_diarize_stream_active(parakeet_diar_stream* s,
                                        parakeet_diar_segment** out, int* n_out);

// Feed PCM; `is_last` flushes the tail and closes open segments. Returns 0 and
// sets *out / *n_out to the segments that ENDED since the previous call
// (free with parakeet_capi_free_diar_segments; *out may be NULL when
// *n_out == 0). Non-zero on error (last_error on the stream's diar_ctx).
int parakeet_capi_diarize_stream_feed(parakeet_diar_stream* s, const float* pcm,
                                      int n_samples, int is_last,
                                      parakeet_diar_segment** out, int* n_out);

void parakeet_capi_free_diar_segments(parakeet_diar_segment* segs);
void parakeet_capi_diarize_stream_free(parakeet_diar_stream* s);

// --- Streaming speaker-attributed ASR ---------------------------------------
// Streaming diarization plus ASR over the same live 16 kHz PCM. Each time
// diarization advances, the not-yet-committed audio is transcribed;
// all words but the last (which may still be cut by the chunk edge) are
// committed with their speakers, and the rest is carried into the next
// chunk. `is_last` commits everything. Borrows both contexts.

typedef struct parakeet_sas_stream parakeet_sas_stream;

// NULL on error (last_error on the context that failed). _begin uses the
// checkpoint's diarization config; _begin_latency takes a
// PARAKEET_DIAR_LATENCY_* mode, which also sets how often words commit.
parakeet_sas_stream* parakeet_capi_sas_stream_begin(parakeet_ctx* asr_ctx,
                                                    parakeet_ctx* diar_ctx);
parakeet_sas_stream* parakeet_capi_sas_stream_begin_latency(parakeet_ctx* asr_ctx,
                                                            parakeet_ctx* diar_ctx,
                                                            int latency);

// Returns 0 and sets *out / *n_out to the utterances committed by this call
// (free with parakeet_capi_free_sas_results(*out, *n_out)). Consecutive calls
// can each return an utterance from the same speaker. Non-zero on error.
int parakeet_capi_sas_stream_feed(parakeet_sas_stream* s, const float* pcm,
                                  int n_samples, int is_last,
                                  parakeet_sas_result** out, int* n_out);

void parakeet_capi_sas_stream_free(parakeet_sas_stream* s);

// --- Sound events (ABI v8) --------------------------------------------------
// A CED GGUF (ced.cpp) loads with parakeet_capi_load into a "tagger" context.

typedef struct {
    int   size;                    // sizeof(parakeet_sound_opts), for versioning
    float window_sec, hop_sec;
    float on_threshold, off_threshold, min_duration_sec;
    int   top_k;                   // per-window scores kept for the drain
} parakeet_sound_opts;
void parakeet_capi_sound_opts_default(parakeet_sound_opts* o);

typedef struct {
    int         class_index;
    const char* label;             // borrowed from the tagger ctx
    float       start, end, peak;  // seconds from stream start
} parakeet_sound_segment;

typedef struct parakeet_sound_stream parakeet_sound_stream;

// NULL opts = defaults. NULL on error (last_error on tagger).
parakeet_sound_stream* parakeet_capi_sound_stream_begin(parakeet_ctx* tagger,
                                                        const parakeet_sound_opts* o);
// Segments that closed since the previous call; is_last closes all.
int   parakeet_capi_sound_stream_feed(parakeet_sound_stream* s, const float* pcm, int n,
                                      int is_last, parakeet_sound_segment** out, int* n_out);
// Still-open segments, end = current stream time.
int   parakeet_capi_sound_stream_active(parakeet_sound_stream* s,
                                        parakeet_sound_segment** out, int* n_out);
// [{"start":..,"end":..,"tags":[{"index":..,"label":..,"score":..}]}], windows
// since the previous drain. Free with parakeet_capi_free_string. The stream
// keeps one entry per hop until drained: drain regularly, or set top_k = 0
// to keep no scores.
char* parakeet_capi_sound_stream_drain_scores_json(parakeet_sound_stream* s);
void  parakeet_capi_free_sound_segments(parakeet_sound_segment* segs);
void  parakeet_capi_sound_stream_free(parakeet_sound_stream* s);

// Tagger introspection: -1 / NULL on a context that is not a tagger.
int         parakeet_capi_num_classes(const parakeet_ctx* ctx);
const char* parakeet_capi_class_label(const parakeet_ctx* ctx, int index);

// Which kind of model a context holds, so a caller loading through the same
// parakeet_capi_load can dispatch without probing individual entry points.
#define PARAKEET_MODEL_KIND_NONE        0
#define PARAKEET_MODEL_KIND_ASR         1
#define PARAKEET_MODEL_KIND_DIARIZATION 2
#define PARAKEET_MODEL_KIND_SOUND       3
#define PARAKEET_MODEL_KIND_SPEAKER     4
#define PARAKEET_MODEL_KIND_VAD         5   // Silero VAD (additive; no ABI bump)
int parakeet_capi_model_kind(const parakeet_ctx* ctx);

// --- Combined scene stream (ABI v8) -----------------------------------------
// ASR, diarization and sound-event tagging over one live 16 kHz mono PCM
// stream. Any of the three contexts may be NULL; at least one is required.
// Borrows the contexts it is given (same lifetime rule as sas_stream /
// sound_stream): free the scene stream first.

typedef struct {
    int size;                    // sizeof(parakeet_scene_opts), for versioning
    int diar_latency;            // PARAKEET_DIAR_LATENCY_*, used only with a diar ctx
    parakeet_sound_opts sound;   // used only with a tagger ctx
    int flags;                   // reserved, must be 0
    // Speaker identification (used only with a speaker ctx and a registry).
    // Read only when `size` covers them; 0 keeps the default of that field.
    float speaker_accept_threshold;   // default 0.5
    float speaker_margin;             // default 0.05
    float speaker_min_voice_sec;      // default 2.0
    float speaker_refresh_sec;        // default 3.0
    float speaker_max_voice_sec;      // default 10.0
} parakeet_scene_opts;
void parakeet_capi_scene_opts_default(parakeet_scene_opts* o);

typedef struct parakeet_scene_stream parakeet_scene_stream;

// Any context may be NULL; at least one must be given. NULL opts = defaults.
// NULL on error: with an all-NULL call there is no context to report on, so
// nothing is set; otherwise last_error is set on the context of the wrong
// kind, on the diar ctx for an unknown diar_latency ("unknown diarization
// latency mode"), or, on an internal failure, on the part that failed.
parakeet_scene_stream* parakeet_capi_scene_stream_begin(parakeet_ctx* asr, parakeet_ctx* diar,
                                                         parakeet_ctx* tagger,
                                                         const parakeet_scene_opts* o);

// Everything finalized by this call, as one JSON document (see docs/sound.md
// for the shape: "t", "utterances", "words", "speakers", "sounds", "active").
// NULL on error. Free with parakeet_capi_free_string. After an error, later
// timestamps may be misaligned (the parts that did not see the failed chunk
// lag behind), so end the stream instead of feeding it more.
char* parakeet_capi_scene_stream_feed_json(parakeet_scene_stream* s, const float* pcm, int n,
                                           int is_last);

// Same shape as parakeet_capi_sound_stream_drain_scores_json; "[]" without a
// tagger. Free with parakeet_capi_free_string. Scores are kept until drained:
// drain regularly, or set sound.top_k = 0 to keep no scores.
char* parakeet_capi_scene_stream_drain_scores_json(parakeet_scene_stream* s);

// Last error of this stream, "" if none. Borrowed.
const char* parakeet_capi_scene_stream_last_error(parakeet_scene_stream* s);
void  parakeet_capi_scene_stream_free(parakeet_scene_stream* s);

// --- Speaker identification (ABI v9) ----------------------------------------
// A voice-detect.cpp GGUF (WeSpeaker, CAM++, ECAPA, ERes2Net) loads through
// parakeet_capi_load into a "speaker" context. A registry holds enrolled
// voices; it is model specific (the embedding size must match the model that
// enrolled the voices). Errors are reported on the speaker ctx
// (parakeet_capi_last_error) unless stated otherwise.

// Embedding size of a speaker ctx; -1 for a context that is not a speaker model.
int parakeet_capi_speaker_dim(const parakeet_ctx* ctx);

typedef struct parakeet_speaker_registry parakeet_speaker_registry;

// New empty registry, or NULL on out of memory. Free with _free (safe on NULL).
parakeet_speaker_registry* parakeet_capi_speaker_registry_new(void);
void parakeet_capi_speaker_registry_free(parakeet_speaker_registry* reg);
// Number of enrolled speakers; 0 for NULL.
int parakeet_capi_speaker_registry_size(const parakeet_speaker_registry* reg);
// Last error of this registry (save/load/enroll bookkeeping), "" if none. Borrowed.
const char* parakeet_capi_speaker_registry_last_error(const parakeet_speaker_registry* reg);

// Embeds 16 kHz (or resampled) mono PCM with `speaker` and adds it under `name`.
// Enrolling a name again refines that voice. 0 on success, nonzero on error
// (empty name, no audio, ctx not a speaker model, embedding size differs from
// the registry's); the message is on the speaker ctx.
int parakeet_capi_speaker_enroll(parakeet_speaker_registry* reg, parakeet_ctx* speaker,
                                 const char* name, const float* pcm, int n, int sample_rate);

// Add one already-computed speaker embedding to `reg` under `name`, without a
// speaker model. `dim` must equal the registry's embedding size once it has
// one (the first successful call fixes it); the values must be finite and not
// all zero. Calling it again with the same name averages the vectors, like
// enrolling more clips. Returns 0 on success; nonzero on error, with the
// message on the registry (parakeet_capi_speaker_registry_last_error). A NULL
// `reg` returns nonzero with no message. ABI v10.
int parakeet_capi_speaker_registry_add_embedding(parakeet_speaker_registry* reg, const char* name,
                                                 const float* embedding, int dim);

// Binary file. 0 on success; nonzero on error (message on the registry).
int parakeet_capi_speaker_registry_save(const parakeet_speaker_registry* reg, const char* path);
// NULL when the file is missing or is not a valid registry. Free with _free.
parakeet_speaker_registry* parakeet_capi_speaker_registry_load(const char* path);

// One-shot identification of a clip: {"name":"alice","score":0.71}, with
// "name":"" when unknown (score is then the best cosine). NULL on error
// (message on the speaker ctx). Free with parakeet_capi_free_string.
char* parakeet_capi_speaker_identify_pcm_json(parakeet_speaker_registry* reg, parakeet_ctx* speaker,
                                              const float* pcm, int n, int sample_rate);

// Like parakeet_capi_scene_stream_begin, plus speaker naming. A speaker needs a registry and a diarization
// ctx (a registry without a speaker is ignored). The registry is borrowed:
// keep it alive and unchanged while the stream runs. Speaker option fields of `o` are honoured only when o->size covers
// them. With NULL speaker and registry this is exactly
// parakeet_capi_scene_stream_begin.
parakeet_scene_stream* parakeet_capi_scene_stream_begin_speaker(parakeet_ctx* asr, parakeet_ctx* diar,
                                                                 parakeet_ctx* tagger,
                                                                 parakeet_ctx* speaker,
                                                                 parakeet_speaker_registry* registry,
                                                                 const parakeet_scene_opts* o);

// Same document as parakeet_capi_transcribe_and_diarize_json, plus "name" and
// "name_score" on each utterance and word (empty name = unknown) and a
// top-level "names" map from slot to {"name","score"}. NULL on error. Free with
// parakeet_capi_free_string.
char* parakeet_capi_transcribe_and_diarize_named_json(parakeet_ctx* asr, parakeet_ctx* diar,
                                                      parakeet_ctx* speaker,
                                                      parakeet_speaker_registry* registry,
                                                      const float* samples, int n_samples,
                                                      int sample_rate);

// Diarization with speaker names and no ASR model: the parakeet_capi_diarize_pcm
// document plus a "names" key, {"0":{"name":"ada","score":0.93},...}, one entry
// per diarization slot that has a segment ("name":"" with score 0 means no
// voice matched, or too little clean speech: under the identifier's 2 s
// minimum, or overlapped by another speaker). `accept_threshold` is a cosine in
// [-1, 1] and `margin` how far the best match must beat the runner-up; it must
// be 0 or more. 0 for either keeps its default (0.5 and 0.05). Any sample rate
// (resampled to 16 kHz like the other PCM entry points). Returns NULL on error,
// with the message on `diar` (a wrong kind, bad samples) or on `speaker` (a
// wrong kind, a NULL or wrong-sized registry, invalid options); both messages
// are cleared on entry, so only the ctx the failure belongs to has one. NULL
// `diar` or `speaker` returns NULL with no message. An empty buffer
// (`n_samples == 0`) is an error here, unlike parakeet_capi_diarize_pcm which
// returns an empty document. An empty registry gives every slot an empty name.
// `reg` is only read; it may be shared by concurrent calls as long as nothing
// adds to it meanwhile. Free with parakeet_capi_free_string. ABI v10.
char* parakeet_capi_diarize_named_pcm_json(parakeet_ctx* diar, parakeet_ctx* speaker,
                                           parakeet_speaker_registry* reg, const float* samples,
                                           int n_samples, int sample_rate, float accept_threshold,
                                           float margin);

// Opt-in export: same segments/names plus speaker_profiles version 1 (see
// docs/diarization.md). NULL reg means an empty registry. One clean embedding
// per usable speaker, never per segment. Does not enroll or mutate reg.
// Same thresholds, ownership and error conventions as diarize_named_pcm_json.
char* parakeet_capi_diarize_profiles_pcm_json(parakeet_ctx* diar, parakeet_ctx* speaker,
    parakeet_speaker_registry* reg, const float* samples, int n_samples, int sample_rate,
    float accept_threshold, float margin);

// Borrowed "sha256:<64 lowercase hex>" of the loaded speaker GGUF bytes;
// valid until context free. NULL for NULL/non-speaker contexts. Compare with
// the trusted server model's identity, NEVER a client-selected model tag.
// Dimension remains available through parakeet_capi_speaker_dim.
const char* parakeet_capi_speaker_identity(const parakeet_ctx* speaker);



#ifdef __cplusplus
} // extern "C"
#endif

#endif // PARAKEET_CAPI_H
