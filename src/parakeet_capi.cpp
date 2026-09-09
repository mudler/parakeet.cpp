#include "parakeet_capi.h"
#include "parakeet.h"     // pk::Decoder
#include "model.hpp"      // pk::Model
#include "streaming.hpp"  // pk::StreamingSession
#include "mel.hpp"        // pk::MelFrontend

#include "transcription.hpp"  // pk::Transcription, pk::Word
#include "transcription_json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

// ABI version. Bump on breaking changes.
// v3: target_lang variants (transcribe_path_lang / transcribe_pcm_lang /
//     stream_begin_lang / transcribe_pcm_batch_json_lang /
//     transcribe_pcm_batch_lang) for multilingual prompt-conditioned (nemotron)
//     models.
// v4: streaming JSON entry points (stream_feed_json / stream_finalize_json) that
//     surface per-word timestamps (start/end/conf) plus frame_sec alongside the
//     newly-finalized text + eou flag, and a "frame_sec" field added to the
//     transcribe_*_json documents. Original entry points unchanged.
// v5: <EOU> vs <EOB> distinction across the C boundary. BREAKING semantics:
//     stream_feed's *eou_out is now a bitmask (PARAKEET_EVENT_EOU |
//     PARAKEET_EVENT_EOB) instead of an any-event 0/1, and the JSON "eou"
//     field now means "an <EOU> fired" only, with a new "eob" field beside it.
//     Added stream_drain_events / free_events (typed per-event records) and
//     the "events" array in the stream_feed_json / stream_finalize_json
//     documents.
// v6: transcribe_pcm_logits, exposing the CTC head's log-prob matrix (row-major
//     [T, vocab+1], already log-softmaxed) instead of decoded text, freed with
//     the new free_logits. Original entry points unchanged.
#define PARAKEET_CAPI_ABI_VERSION 6

// The opaque context: a loaded model plus a buffer for the last error message.
struct parakeet_ctx {
    std::unique_ptr<pk::Model> model;
    std::string last_error;
};

// The opaque streaming session: a pk::StreamingSession over the ctx's model plus
// an INCREMENTAL log-mel front end (pk::StreamingMel).
//
// `feed` turns the just-arrived 16 kHz mono PCM into the newly-ready mel frames
// via StreamingMel (frame-local, NO full-buffer recompute — see mel.hpp), grows
// the accumulated mel-frame buffer, then incrementally decodes any encoder
// chunks for which enough mel frames (and right context) are now buffered,
// carrying the encoder/decoder caches across feeds — so a live consumer gets
// partial text as audio arrives. Committed chunks are not re-decoded.
//
// The streaming model uses normalize="NA" (frame-local mel, no whole-utterance
// stats), so the incremental mel is bit-identical to MelFrontend::compute on the
// full clip (see tests/test_streaming_mel.cpp); only the accumulated mel frames
// (NOT the raw PCM) are retained, and StreamingMel itself keeps only ~n_fft
// recent samples.
//
// `finalize` appends StreamingMel's end zero-pad tail frames, then flushes the
// streaming decoder tail: it decodes the final (partial) chunk with
// keep_all_outputs so the trailing encoder frames complete, then returns any
// remaining text. It does NOT fabricate an <EOU> NeMo's streaming would not emit.
struct parakeet_stream {
    parakeet_ctx* ctx = nullptr;             // borrowed (must outlive the stream)
    std::unique_ptr<pk::StreamingMel> mel;   // incremental log-mel front end
    // mel_buf holds ONLY the still-reachable window of mel history, feat-major
    // [n_mels, mel_T - mel_buf_origin]; column 0 is ABSOLUTE frame mel_buf_origin
    // (see append_mel_frames / feed_available's window()). Frames strictly
    // before mel_buffer_idx - pre_encode_cache_size() can never be read again
    // (mel_buffer_idx only advances -- see feed_available), so they are dropped
    // as they age out instead of being retained for the life of the stream.
    std::vector<float> mel_buf;
    int n_mels = 0;
    int mel_T = 0;                           // total mel frames accumulated so far (absolute)
    int mel_buf_origin = 0;                  // absolute frame index of mel_buf's column 0
    std::unique_ptr<pk::StreamingSession> sess;
    int mel_buffer_idx = 0;                  // next un-fed mel frame (chunk schedule)
    bool first_chunk = true;                 // chunk 0 has no pre-encode overlap
    bool finalized = false;
};

namespace {
// Append `n_new` feat-major mel frames `[n_mels, n_new]` to the stream's
// accumulated feat-major mel buffer, growing mel_T, and drop the prefix that
// feed_available's window() can provably never read again.
//
// ROOT CAUSE of mudler/parakeet.cpp#63 (the streaming RSS leak): the mel buffer
// used to retain the FULL stream history from mel_buffer_idx==0 forever, so
// this function rebuilt a `[n_mels, T]` array from scratch on EVERY stream_feed
// call with T monotonically growing for the life of the stream -- not just O(T)
// CPU per call (the reason this rebuild exists at all, per the original
// comment), but a continuous sequence of UNIQUELY, EVER-LARGER-sized heap
// allocations. A general-purpose allocator can never satisfy a strictly-larger
// request from a smaller freed block, so the freed (now too-small) blocks from
// every earlier call accumulate as unusable, unreturned pages instead of being
// recycled -- the process's live working set stays tiny (a few MB) but the
// CUMULATIVE allocator debt grows with the stream's length, which is exactly
// why the upstream report finds it reclaimed by neither stream_free nor a
// fresh stream_begin: it is allocator-level fragmentation from the discarded
// buffers' sizes, not stream-owned memory.
//
// feed_available's window(lo, hi) only ever reads lo >= mel_buffer_idx -
// pre_cache, and mel_buffer_idx is monotonically non-decreasing (it only moves
// forward by chunk_size). So at the START of any call here (mel_buffer_idx
// unchanged since the previous feed_available pass), every mel frame before
// mel_buffer_idx - pre_cache is provably dead and safe to drop; mel_buf then
// stays bounded to O(pre_cache + chunk_size) -- a small constant -- instead of
// O(stream length), for the life of the stream.
void append_mel_frames(parakeet_stream* s, const std::vector<float>& frames, int n_new) {
    if (n_new <= 0) return;
    const int n_mels = s->n_mels;
    const int old_T = s->mel_T;                 // absolute frame count so far
    const int new_T = old_T + n_new;

    const int pre_cache = s->sess ? s->sess->pre_encode_cache_size() : 0;
    int keep_from = s->mel_buffer_idx - pre_cache;
    if (keep_from < s->mel_buf_origin) keep_from = s->mel_buf_origin;  // never move backward
    if (keep_from > old_T) keep_from = old_T;                          // never past what exists

    const int old_local_T = old_T - s->mel_buf_origin;   // mel_buf's current column count
    const int drop_local   = keep_from - s->mel_buf_origin;  // columns to drop from the front
    const int kept         = old_local_T - drop_local;       // columns carried forward
    const int new_local_T  = kept + n_new;

    std::vector<float> out((size_t)n_mels * new_local_T);
    for (int m = 0; m < n_mels; ++m) {
        // carry forward the still-reachable tail [drop_local, old_local_T)
        for (int t = 0; t < kept; ++t)
            out[(size_t)m * new_local_T + t] =
                s->mel_buf[(size_t)m * old_local_T + (drop_local + t)];
        // append the newly-arrived frames
        for (int t = 0; t < n_new; ++t)
            out[(size_t)m * new_local_T + (kept + t)] = frames[(size_t)m * n_new + t];
    }
    s->mel_buf.swap(out);
    s->mel_T = new_T;
    s->mel_buf_origin = keep_from;
}
} // namespace

namespace {

// Map the C decoder int to pk::Decoder. Unknown values fall back to default.
pk::Decoder to_decoder(int decoder) {
    switch (decoder) {
        case 1:  return pk::Decoder::kCTC;
        case 2:  return pk::Decoder::kTDT;
        case 0:
        default: return pk::Decoder::kDefault;
    }
}

// malloc a NUL-terminated copy of `s` so a C consumer frees it with free()
// (matching parakeet_capi_free_string). Returns NULL on OOM.
char* dup_to_c(const std::string& s) {
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
}

} // namespace

extern "C" int parakeet_capi_abi_version(void) {
    return PARAKEET_CAPI_ABI_VERSION;
}

extern "C" parakeet_ctx* parakeet_capi_load(const char* gguf_path) {
    if (!gguf_path) return nullptr;
    try {
        std::unique_ptr<pk::Model> model = pk::Model::load(gguf_path);
        if (!model) return nullptr;  // load failure (bad/missing GGUF)
        auto* ctx = new (std::nothrow) parakeet_ctx();
        if (!ctx) return nullptr;
        ctx->model = std::move(model);
        return ctx;
    } catch (...) {
        // Never let an exception cross the boundary.
        return nullptr;
    }
}

extern "C" void parakeet_capi_free(parakeet_ctx* ctx) {
    delete ctx;  // safe on nullptr; ~unique_ptr releases the model.
}

extern "C" char* parakeet_capi_transcribe_path_lang(parakeet_ctx* ctx,
                                                    const char* wav_path, int decoder,
                                                    const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!wav_path)   { ctx->last_error = "wav_path is NULL"; return nullptr; }
    // NULL / "" -> model default language (ignored by non-prompt models).
    const std::string lang = target_lang ? target_lang : "";
    try {
        std::string text = ctx->model->transcribe_path(wav_path, to_decoder(decoder), lang);
        ctx->last_error.clear();
        char* out = dup_to_c(text);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_transcribe_path(parakeet_ctx* ctx,
                                               const char* wav_path, int decoder) {
    // Delegate with the model default language.
    return parakeet_capi_transcribe_path_lang(ctx, wav_path, decoder, nullptr);
}

extern "C" char* parakeet_capi_transcribe_pcm_lang(parakeet_ctx* ctx,
                                                   const float* samples, int n_samples,
                                                   int sample_rate, int decoder,
                                                   const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!samples || n_samples < 0) { ctx->last_error = "invalid samples buffer"; return nullptr; }
    // NULL / "" -> model default language (ignored by non-prompt models).
    const std::string lang = target_lang ? target_lang : "";
    try {
        std::vector<float> pcm(samples, samples + n_samples);
        std::string text = ctx->model->transcribe_pcm(pcm, sample_rate, to_decoder(decoder), lang);
        ctx->last_error.clear();
        char* out = dup_to_c(text);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_transcribe_pcm(parakeet_ctx* ctx, const float* samples,
                                              int n_samples, int sample_rate,
                                              int decoder) {
    // Delegate with the model default language.
    return parakeet_capi_transcribe_pcm_lang(ctx, samples, n_samples, sample_rate,
                                             decoder, nullptr);
}

extern "C" int parakeet_capi_transcribe_pcm_logits(parakeet_ctx* ctx,
                                                    const float* samples, int n_samples,
                                                    int sample_rate, float** out_logits,
                                                    int* out_T, int* out_vocab_plus_1) {
    if (!ctx) return 1;
    if (!out_logits || !out_T || !out_vocab_plus_1) {
        ctx->last_error = "invalid output pointer(s)";
        return 1;
    }
    *out_logits = nullptr;
    *out_T = 0;
    *out_vocab_plus_1 = 0;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return 1; }
    if (!samples || n_samples < 0) { ctx->last_error = "invalid samples buffer"; return 1; }
    try {
        std::vector<float> pcm(samples, samples + n_samples);
        std::vector<float> logits;
        int T = 0, vocab_plus_1 = 0;
        ctx->model->transcribe_pcm_ctc_logits(pcm, sample_rate, logits, T, vocab_plus_1);

        float* buf = static_cast<float*>(std::malloc(logits.size() * sizeof(float)));
        if (!buf) { ctx->last_error = "out of memory"; return 1; }
        std::memcpy(buf, logits.data(), logits.size() * sizeof(float));

        ctx->last_error.clear();
        *out_logits = buf;
        *out_T = T;
        *out_vocab_plus_1 = vocab_plus_1;
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return 1;
    } catch (...) {
        ctx->last_error = "unknown error";
        return 1;
    }
}

extern "C" void parakeet_capi_free_logits(float* logits) {
    std::free(logits);
}

extern "C" int parakeet_capi_transcribe_pcm_batch_lang(parakeet_ctx* ctx,
                                                       const float* const* samples,
                                                       const int* n_samples, int n_clips,
                                                       int sample_rate, int decoder,
                                                       const char* target_lang,
                                                       char** out) {
    if (!ctx) return 1;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return 1; }
    if (!samples || !n_samples || !out || n_clips < 0) {
        ctx->last_error = "invalid batch arguments";
        return 1;
    }
    // NULL / "" -> model default language (ignored by non-prompt models).
    const std::string lang = target_lang ? target_lang : "";
    // Contract: on any error path (validation, exception, OOM) every out[]
    // entry is left NULL, so the caller owns nothing and frees nothing.
    for (int i = 0; i < n_clips; ++i) out[i] = nullptr;
    try {
        std::vector<std::vector<float>> pcms(n_clips);
        for (int i = 0; i < n_clips; ++i) {
            if (!samples[i] || n_samples[i] < 0) {
                ctx->last_error = "invalid samples buffer in batch";
                return 1;
            }
            pcms[i].assign(samples[i], samples[i] + n_samples[i]);
        }
        std::vector<std::string> texts =
            ctx->model->transcribe_pcm_batch(pcms, sample_rate, to_decoder(decoder), lang);
        ctx->last_error.clear();
        for (int i = 0; i < n_clips; ++i) {
            char* s = dup_to_c(texts[i]);
            if (!s) {
                // Roll back the strings already allocated this call so every
                // out[] entry is NULL on return (out[i..] are already NULL).
                for (int j = 0; j < i; ++j) { std::free(out[j]); out[j] = nullptr; }
                ctx->last_error = "out of memory";
                return 2;
            }
            out[i] = s;
        }
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return 3;
    } catch (...) {
        ctx->last_error = "unknown error";
        return 3;
    }
}

extern "C" int parakeet_capi_transcribe_pcm_batch(parakeet_ctx* ctx,
                                                  const float* const* samples,
                                                  const int* n_samples, int n_clips,
                                                  int sample_rate, int decoder,
                                                  char** out) {
    // Delegate with the model default language.
    return parakeet_capi_transcribe_pcm_batch_lang(ctx, samples, n_samples, n_clips,
                                                   sample_rate, decoder, nullptr, out);
}

extern "C" char* parakeet_capi_transcribe_path_json(parakeet_ctx* ctx,
                                                    const char* wav_path,
                                                    int decoder) {
    if (!ctx) return nullptr;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!wav_path)   { ctx->last_error = "wav_path is NULL"; return nullptr; }
    try {
        pk::Transcription tr =
            ctx->model->transcribe_path_with_timestamps(wav_path, to_decoder(decoder));
        // frame_sec = hop_length * subsampling_factor / sample_rate (token "t").
        const pk::ParakeetConfig& cfg = ctx->model->config();
        const float frame_sec =
            (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
        std::string json = pk::transcription_to_json(tr, frame_sec);
        ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_transcribe_pcm_batch_json_lang(parakeet_ctx* ctx,
        const float* samples_concat, const int* n_samples, int n_clips,
        int sample_rate, int decoder, const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!samples_concat || !n_samples || n_clips < 0) {
        ctx->last_error = "invalid batch arguments"; return nullptr;
    }
    // NULL / "" -> model default language (ignored by non-prompt models).
    const std::string lang = target_lang ? target_lang : "";
    try {
        std::vector<std::vector<float>> pcms(n_clips);
        size_t off = 0;
        for (int i = 0; i < n_clips; ++i) {
            if (n_samples[i] < 0) { ctx->last_error = "invalid clip length"; return nullptr; }
            pcms[i].assign(samples_concat + off, samples_concat + off + n_samples[i]);
            off += (size_t)n_samples[i];
        }
        std::vector<pk::Transcription> trs =
            ctx->model->transcribe_pcm_batch_with_timestamps(pcms, sample_rate,
                                                             to_decoder(decoder), lang);
        const pk::ParakeetConfig& cfg = ctx->model->config();
        const float frame_sec =
            (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
        std::string json = "[";
        for (size_t i = 0; i < trs.size(); ++i) {
            if (i) json += ',';
            json += pk::transcription_to_json(trs[i], frame_sec);
        }
        json += "]";
        ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what(); return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error"; return nullptr;
    }
}

extern "C" char* parakeet_capi_transcribe_pcm_batch_json(parakeet_ctx* ctx,
        const float* samples_concat, const int* n_samples, int n_clips,
        int sample_rate, int decoder) {
    // Delegate with the model default language.
    return parakeet_capi_transcribe_pcm_batch_json_lang(ctx, samples_concat, n_samples,
                                                        n_clips, sample_rate, decoder,
                                                        nullptr);
}

extern "C" char* parakeet_capi_transcribe_path_nbest_json(
        parakeet_ctx* ctx, const char* wav_path,
        int beam_size, int nbest, int score_norm, const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) {
        ctx->last_error = "context has no loaded model";
        return nullptr;
    }
    if (!wav_path) {
        ctx->last_error = "wav_path is NULL";
        return nullptr;
    }
    try {
        const bool normalize = score_norm != 0;
        const std::string lang = target_lang ? target_lang : "";
        std::vector<pk::NBestTranscription> hypotheses =
            ctx->model->transcribe_path_nbest(
                wav_path, beam_size, nbest, normalize, lang);
        const pk::ParakeetConfig& cfg = ctx->model->config();
        const float frame_sec =
            (float)cfg.hop_length * (float)cfg.subsampling_factor /
            (float)cfg.sample_rate;
        std::string json = pk::nbest_transcriptions_to_json(
            hypotheses, beam_size, normalize, frame_sec);
        ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) {
            ctx->last_error = "out of memory";
            return nullptr;
        }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_transcribe_pcm_nbest_json(
        parakeet_ctx* ctx, const float* samples, int n_samples, int sample_rate,
        int beam_size, int nbest, int score_norm, const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) {
        ctx->last_error = "context has no loaded model";
        return nullptr;
    }
    if (!samples || n_samples < 0) {
        ctx->last_error = "invalid samples buffer";
        return nullptr;
    }
    try {
        const bool normalize = score_norm != 0;
        const std::string lang = target_lang ? target_lang : "";
        const std::vector<float> pcm(samples, samples + n_samples);
        std::vector<pk::NBestTranscription> hypotheses =
            ctx->model->transcribe_pcm_nbest(
                pcm, sample_rate, beam_size, nbest, normalize, lang);
        const pk::ParakeetConfig& cfg = ctx->model->config();
        const float frame_sec =
            (float)cfg.hop_length * (float)cfg.subsampling_factor /
            (float)cfg.sample_rate;
        std::string json = pk::nbest_transcriptions_to_json(
            hypotheses, beam_size, normalize, frame_sec);
        ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) {
            ctx->last_error = "out of memory";
            return nullptr;
        }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

namespace {

// Feed any not-yet-fed encoder chunks for which enough mel frames are now
// buffered, carrying the StreamingSession caches. Operates over the stream's
// INCREMENTALLY-accumulated mel buffer (s->mel_buf / s->mel_T) — the mel for
// new PCM is produced frame-local by StreamingMel in the feed/finalize entry
// points, NOT recomputed over the whole buffer here. `flush` marks the final
// partial chunk is_last (keep_all_outputs), draining the remaining frames. Sets
// eou_flag / eob_flag to 1 if an <EOU> / <EOB> (respectively) fired in this
// pass — attributed by watermarking the session's un-drained event queue, so
// the queue itself is left intact for the caller to drain. Returns the newly-
// finalized text.
std::string feed_available(parakeet_stream* s, bool flush, int& eou_flag,
                           int& eob_flag) {
    eou_flag = 0;
    eob_flag = 0;
    pk::StreamingSession& sess = *s->sess;
    const size_t ev0 = sess.events().size();

    const int n_mels = s->n_mels;
    const int T = s->mel_T;              // absolute total frame count
    if (T <= 0) return std::string();
    const int origin = s->mel_buf_origin;   // absolute index of mel_buf's column 0
    const int local_T = T - origin;         // mel_buf's actual (bounded) column count
    const std::vector<float>& mel = s->mel_buf;  // [n_mels, local_T] feat-major, columns [origin, T)

    const int chunk0     = sess.chunk_size_first();
    const int chunk_main = sess.chunk_size();
    const int pre_cache  = sess.pre_encode_cache_size();

    // lo/hi are ABSOLUTE frame indices (as before); translate into mel_buf's
    // own local (origin-relative) columns. append_mel_frames guarantees
    // lo >= mel_buffer_idx - pre_cache is always still resident.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * local_T + (lo - origin + t)];
        return w;
    };

    std::string new_text;
    while (s->mel_buffer_idx < T) {
        const int chunk_size = s->first_chunk ? chunk0 : chunk_main;
        const int chunk_hi   = std::min(s->mel_buffer_idx + chunk_size, T);
        if (chunk_hi - s->mel_buffer_idx <= 0) break;
        const bool reaches_end = (chunk_hi >= T);
        // Mid-stream (not flushing): only feed a chunk if there is STRICTLY more
        // audio after it (chunk_hi < T) so it is definitely not the final chunk
        // (kept at valid_out_len). The chunk that reaches the end of the current
        // buffer is deferred to flush, where it is fed with keep_all_outputs
        // (is_last) so the streaming tail frames are retained — matching NeMo's
        // CacheAwareStreamingAudioBuffer last-chunk behaviour and the validated
        // run_stream_over_pcm / test_streaming_decode schedule.
        if (!flush && reaches_end) break;
        const int lo = s->first_chunk ? s->mel_buffer_idx
                                      : std::max(0, s->mel_buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = flush && reaches_end;

        sess.feed_mel_chunk(win, win_frames, is_last);
        new_text += sess.take_new_text();

        s->mel_buffer_idx += chunk_size;  // shift_size == chunk_size here
        s->first_chunk = false;
        if (is_last) break;               // flushed the end-of-stream tail
    }
    for (size_t i = ev0; i < sess.events().size(); ++i)
        (sess.events()[i].is_eob ? eob_flag : eou_flag) = 1;
    return new_text;
}

} // namespace

extern "C" parakeet_stream* parakeet_capi_stream_begin_lang(parakeet_ctx* ctx,
                                                           const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!ctx->model->config().streaming.present) {
        ctx->last_error = "model is not a cache-aware streaming model";
        return nullptr;
    }
    // NULL / "" -> model default language (ignored by non-prompt models).
    const std::string lang = target_lang ? target_lang : "";
    try {
        auto* s = new (std::nothrow) parakeet_stream();
        if (!s) { ctx->last_error = "out of memory"; return nullptr; }
        s->ctx = ctx;
        s->sess = std::make_unique<pk::StreamingSession>(ctx->model->loader(), lang);
        s->mel  = std::make_unique<pk::StreamingMel>(ctx->model->loader());
        s->n_mels = s->mel->n_mels();
        ctx->last_error.clear();
        return s;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" parakeet_stream* parakeet_capi_stream_begin(parakeet_ctx* ctx) {
    // Delegate with the model default language.
    return parakeet_capi_stream_begin_lang(ctx, nullptr);
}

extern "C" char* parakeet_capi_stream_feed(parakeet_stream* s, const float* pcm,
                                           int n_samples, int* eou_out) {
    if (eou_out) *eou_out = 0;
    if (!s) return nullptr;
    if (!s->ctx || !s->ctx->model) return nullptr;
    if (n_samples < 0 || (!pcm && n_samples > 0)) {
        s->ctx->last_error = "invalid PCM buffer";
        return nullptr;
    }
    try {
        // Incremental, frame-local mel for the just-arrived PCM (no full-buffer
        // recompute). StreamingMel carries the preemph history + partial frame
        // across feeds; the emitted frames are appended to the accumulated mel.
        if (n_samples > 0) {
            int n_new = 0;
            std::vector<float> frames = s->mel->feed(pcm, n_samples, n_new);
            append_mel_frames(s, frames, n_new);
        }
        int eou = 0, eob = 0;
        std::string delta = feed_available(s, /*flush=*/false, eou, eob);
        if (eou_out) *eou_out = (eou ? PARAKEET_EVENT_EOU : 0) |
                                (eob ? PARAKEET_EVENT_EOB : 0);
        s->ctx->last_error.clear();
        char* out = dup_to_c(delta);
        if (!out) { s->ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        s->ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_stream_finalize(parakeet_stream* s) {
    if (!s) return nullptr;
    if (!s->ctx || !s->ctx->model) return nullptr;
    try {
        // Emit the end zero-pad tail frames so the accumulated mel matches the
        // full-buffer MelFrontend::compute exactly, then flush the decoder tail.
        if (s->mel) {
            int n_tail = 0;
            std::vector<float> tail = s->mel->finalize(n_tail);
            append_mel_frames(s, tail, n_tail);
        }
        int eou = 0, eob = 0;
        std::string delta = feed_available(s, /*flush=*/true, eou, eob);
        // After the flush the session's finalize() is a no-op text-wise (no extra
        // audio) but documents the end-of-stream tail semantics.
        delta += s->sess->finalize();
        s->finalized = true;
        s->ctx->last_error.clear();
        char* out = dup_to_c(delta);
        if (!out) { s->ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        s->ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" int parakeet_capi_stream_drain_events(parakeet_stream* s,
                                                 parakeet_stream_event** out_events) {
    if (out_events) *out_events = nullptr;
    if (!s || !out_events) return -1;
    if (!s->ctx || !s->ctx->model || !s->sess) return -1;
    try {
        std::vector<pk::EouEvent> evs = s->sess->drain_events();
        s->ctx->last_error.clear();
        if (evs.empty()) return 0;
        auto* arr = static_cast<parakeet_stream_event*>(
            std::malloc(evs.size() * sizeof(parakeet_stream_event)));
        if (!arr) { s->ctx->last_error = "out of memory"; return -1; }
        for (size_t i = 0; i < evs.size(); ++i) {
            arr[i].token         = (int)evs[i].token;
            arr[i].is_eob        = evs[i].is_eob ? 1 : 0;
            arr[i].encoder_frame = evs[i].encoder_frame;
            arr[i].time_sec      = (float)evs[i].time_sec;
        }
        *out_events = arr;
        return (int)evs.size();
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
        return -1;
    } catch (...) {
        s->ctx->last_error = "unknown error";
        return -1;
    }
}

extern "C" void parakeet_capi_free_events(parakeet_stream_event* events) {
    std::free(events);
}

namespace {

// Serialize a streaming feed/finalize result to JSON: the newly-finalized text,
// the per-type eou/eob flags, frame_sec, the <EOU>/<EOB> events drained this
// call, and the words drained this call (absolute seconds). Shape matches the
// header doc on parakeet_capi_stream_feed_json. "eou" means an <EOU> fired and
// "eob" an <EOB> — they are NOT conflated (a voice agent responds on eou and
// must not treat eob as the user taking the turn); "events" carries the
// per-event timestamps.
std::string stream_json(const std::string& text, int eou, int eob,
                        float frame_sec,
                        const std::vector<pk::EouEvent>& events,
                        const std::vector<pk::Word>& words) {
    std::string out;
    out.reserve(80 + events.size() * 36 + words.size() * 48);
    out += "{\"text\":";
    pk::append_json_string(out, text);
    out += ",\"eou\":";
    out += (eou ? "1" : "0");
    out += ",\"eob\":";
    out += (eob ? "1" : "0");
    out += ",\"frame_sec\":";
    pk::append_json_float(out, "%.6f", frame_sec);
    out += ",\"events\":[";
    for (size_t i = 0; i < events.size(); ++i) {
        if (i) out += ',';
        out += "{\"type\":";
        out += events[i].is_eob ? "\"eob\"" : "\"eou\"";
        out += ",\"frame\":";
        pk::append_json_int(out, events[i].encoder_frame);
        out += ",\"t\":";
        pk::append_json_float(out, "%.3f", (float)events[i].time_sec);
        out += '}';
    }
    out += "],\"words\":[";
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out += ',';
        out += "{\"w\":";
        pk::append_json_string(out, words[i].text);
        out += ",\"start\":";
        pk::append_json_float(out, "%.3f", words[i].start);
        out += ",\"end\":";
        pk::append_json_float(out, "%.3f", words[i].end);
        out += ",\"conf\":";
        pk::append_json_float(out, "%.4f", words[i].conf);
        out += '}';
    }
    out += "]}";
    return out;
}

// frame_sec for the stream's model (encoder frame stride in seconds).
float stream_frame_sec(const parakeet_stream* s) {
    const pk::ParakeetConfig& cfg = s->ctx->model->config();
    return (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;
}

} // namespace

extern "C" char* parakeet_capi_stream_feed_json(parakeet_stream* s,
                                                const float* pcm, int n_samples) {
    if (!s) return nullptr;
    if (!s->ctx || !s->ctx->model) return nullptr;
    if (n_samples < 0 || (!pcm && n_samples > 0)) {
        s->ctx->last_error = "invalid PCM buffer";
        return nullptr;
    }
    try {
        if (n_samples > 0) {
            int n_new = 0;
            std::vector<float> frames = s->mel->feed(pcm, n_samples, n_new);
            append_mel_frames(s, frames, n_new);
        }
        int eou = 0, eob = 0;
        std::string delta = feed_available(s, /*flush=*/false, eou, eob);
        std::vector<pk::EouEvent> events = s->sess->drain_events();
        std::vector<pk::Word> words = s->sess->drain_words();
        std::string json = stream_json(delta, eou, eob, stream_frame_sec(s), events, words);
        s->ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) { s->ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        s->ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_stream_finalize_json(parakeet_stream* s) {
    if (!s) return nullptr;
    if (!s->ctx || !s->ctx->model) return nullptr;
    try {
        if (s->mel) {
            int n_tail = 0;
            std::vector<float> tail = s->mel->finalize(n_tail);
            append_mel_frames(s, tail, n_tail);
        }
        int eou = 0, eob = 0;
        std::string delta = feed_available(s, /*flush=*/true, eou, eob);
        delta += s->sess->finalize();
        std::vector<pk::EouEvent> events = s->sess->drain_events();
        std::vector<pk::Word> words = s->sess->drain_words();
        std::string json = stream_json(delta, eou, eob, stream_frame_sec(s), events, words);
        s->finalized = true;
        s->ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) { s->ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        s->ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" void parakeet_capi_stream_free(parakeet_stream* s) {
    delete s;  // safe on nullptr
}

extern "C" void parakeet_capi_free_string(char* s) {
    std::free(s);
}

extern "C" const char* parakeet_capi_last_error(parakeet_ctx* ctx) {
    if (!ctx) return "";
    return ctx->last_error.c_str();
}
