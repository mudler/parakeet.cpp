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
//     parakeet_capi_transcribe_pcm_lang, parakeet_capi_stream_begin_lang) for
//     multilingual prompt-conditioned (nemotron) models. The original non-lang
//     entry points are unchanged and delegate with the model default language.
int parakeet_capi_abi_version(void);

// Load a GGUF model. Returns an owning context, or NULL on failure.
// The returned context must be released with parakeet_capi_free.
parakeet_ctx* parakeet_capi_load(const char* gguf_path);

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

// Transcribe a WAV file returning a malloc'd UTF-8 JSON document with per-word
// and per-token timestamps + confidence (matching NeMo timestamps=True and the
// 'max_prob' confidence method). `decoder` is as in
// parakeet_capi_transcribe_path. The JSON shape is:
//
//   {"text":"...",
//    "words":[{"w":"...","start":0.480,"end":0.640,"conf":0.9100}, ...],
//    "tokens":[{"id":123,"t":0.480,"conf":0.9100}, ...]}
//
// where "start"/"end"/"t" are seconds (3 decimals) and "conf" is the
// confidence in (0,1] (4 decimals). The "w"/"text" strings are JSON-escaped
// (", \\, and control chars). On success returns the malloc'd string (free with
// parakeet_capi_free_string); on error returns NULL and sets the context's last
// error.
char* parakeet_capi_transcribe_path_json(parakeet_ctx* ctx, const char* wav_path,
                                         int decoder);

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

// Feed a block of 16 kHz MONO float PCM (`pcm`, length `n_samples`). The session
// buffers the audio and decodes as full encoder chunks become available.
// Returns the newly-finalized text since the last call as a malloc'd UTF-8
// string (free with parakeet_capi_free_string) — "" (empty, non-NULL) if no new
// text was finalized this call, NULL only on error. <EOU>/<EOB> are stripped
// from the text and surfaced as events: if `eou_out` is non-NULL it is set to 1
// when an <EOU>/<EOB> event fired during this feed, else 0.
char* parakeet_capi_stream_feed(parakeet_stream* s, const float* pcm,
                                int n_samples, int* eou_out);

// Flush the end-of-stream tail: process any remaining buffered audio (the final
// chunk completes the streaming tail). Returns the final newly-finalized text
// (malloc'd; "" if none, NULL on error). After this the running transcript is
// complete. Does NOT fabricate an <EOU> NeMo's streaming would not emit.
char* parakeet_capi_stream_finalize(parakeet_stream* s);

// Free a streaming session. Safe on NULL.
void parakeet_capi_stream_free(parakeet_stream* s);

// Free a string previously returned by parakeet_capi_transcribe_* /
// parakeet_capi_stream_*. Safe on NULL.
void parakeet_capi_free_string(char* s);

// Human-readable description of the last error on `ctx`, or "" if none.
// The returned pointer is owned by the context and valid until the next call on
// it (or until parakeet_capi_free). Returns "" if `ctx` is NULL.
const char* parakeet_capi_last_error(parakeet_ctx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // PARAKEET_CAPI_H
