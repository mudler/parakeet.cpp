#include "parakeet_capi.h"
#include "parakeet.h"     // pk::Decoder
#include "model.hpp"      // pk::Model
#include "diarization.hpp" // pk::DiarizationModel
#include "diarization_streaming.hpp" // pk::StreamingDiarization
#include "streaming.hpp"  // pk::StreamingSession
#include "ced_tagger.hpp" // pk::CedTagger
#include "sound_stream.hpp" // pk::SoundStream
#include "mel.hpp"        // pk::MelFrontend
#include "sas_merge.hpp"  // pk::merge_asr_diarization, pk::group_speaker_words
#include "diar_pcm_stream.hpp" // pk::DiarPcmStream
#include "scene_stream.hpp" // pk::SceneStream
#include "speaker_model_identity.hpp"
#include "speaker_profiles_json.hpp"
#include "speaker_encoder.hpp"    // pk::SpeakerEncoder
#include "speaker_identifier.hpp" // pk::identify_offline
#include "speaker_registry.hpp"   // pk::SpeakerRegistry
#include "audio_io.hpp"   // pk::resample_linear
#include "common.hpp"     // pk::write_file_atomic

#include "transcription.hpp"  // pk::Transcription, pk::Word
#include "transcription_json.hpp"
#include "vad_json.hpp"
#include "silero_vad.hpp"
#include "bundle.hpp"
#include "audio_io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
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
// v7: speaker diarization (diarize_*), speaker-attributed ASR
//     (transcribe_and_diarize*, sas_stream_*) and streaming diarization
//     (diarize_stream_*). A context holds either an ASR or a diarization model.
// v8: sound-event detection (CED), sound_stream_*, scene_stream_*; additive.
// v9: speaker identification (voice-detect.cpp): a speaker ctx kind, a speaker
//     registry, scene_stream_begin_speaker, transcribe_and_diarize_named_json;
//     additive.
// v10: raw-embedding enroll (parakeet_capi_speaker_registry_add_embedding) for
//      callers that keep speaker embeddings themselves, and diarization with
//      speaker names and no ASR model (parakeet_capi_diarize_named_pcm_json);
//      additive.
#define PARAKEET_CAPI_ABI_VERSION 10

// The last error message of a context. Every write takes a lock, so several
// threads can fail on one context at once (concurrent requests on a pooled
// model). The slot holds the most recent message; read() copies it into a
// snapshot and returns the snapshot's pointer, which stays valid until the next
// read() or the context is freed. The assignment operators keep the syntax of
// the plain std::string this replaces.
class ErrorSlot {
public:
    ErrorSlot& operator=(const std::string& s) { std::lock_guard<std::mutex> l(mu_); msg_ = s; return *this; }
    ErrorSlot& operator=(const char* s)        { std::lock_guard<std::mutex> l(mu_); msg_ = s ? s : ""; return *this; }
    ErrorSlot& operator+=(const std::string& s){ std::lock_guard<std::mutex> l(mu_); msg_ += s; return *this; }
    void clear() { std::lock_guard<std::mutex> l(mu_); msg_.clear(); }
    const char* read() {
        std::lock_guard<std::mutex> l(mu_);
        snapshot_ = msg_;
        return snapshot_.c_str();
    }
private:
    std::mutex mu_;
    std::string msg_;
    std::string snapshot_;
};

// The opaque context: a loaded model plus a buffer for the last error message.
// Exactly one of `model` / `diar` / `tagger` / `speaker` / `silero` is non-null:
// ASR models use `model`, diarization models (Sortformer) use `diar`, CED
// sound-event taggers use `tagger`, voice-detect speaker encoders use `speaker`,
// Silero VAD models use `silero`.
struct parakeet_ctx {
    std::unique_ptr<pk::Model> model;
    std::unique_ptr<pk::DiarizationModel> diar;
    std::unique_ptr<pk::CedTagger> tagger;
    std::unique_ptr<pk::SpeakerEncoder> speaker;
    std::unique_ptr<pk::SileroVad> silero;   // Silero VAD model (architecture "silero_vad")
    std::string speaker_identity;
    ErrorSlot last_error;
};

// Enrolled voices plus a buffer for the last save/load error.
struct parakeet_speaker_registry {
    pk::SpeakerRegistry reg;
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

namespace {

// Reason for the last failed parakeet_capi_load / _load_component /
// _bundle_components_json on this thread (there is no context to hold it).
thread_local std::string g_load_error;

parakeet_ctx* load_plain(const char* gguf_path);

// Opens one component of a bundle GGUF, by the kind recorded in the header.
parakeet_ctx* load_bundle_component(const char* path, const std::string& name) {
    pk::BundleInfo info;
    std::string err;
    if (!pk::read_bundle_info(path, info, &err)) { g_load_error = err; return nullptr; }
    const pk::BundleComponent* c = info.find(name);
    if (!c) {
        g_load_error = std::string("bundle has no component \"") + name + "\"; components: " +
                       pk::bundle_component_names(info);
        return nullptr;
    }
    std::unique_ptr<parakeet_ctx> ctx(new (std::nothrow) parakeet_ctx());
    if (!ctx) { g_load_error = "out of memory"; return nullptr; }
    if (c->kind == pk::kBundleKindAsr) {
        ctx->model = pk::Model::load(path, name);
        if (!ctx->model) {
            g_load_error = "cannot load ASR component \"" + name + "\" (see the log for the reason)";
            return nullptr;
        }
    } else if (c->kind == pk::kBundleKindVad) {
        ctx->silero = pk::SileroVad::load(path, &err, name);
        if (!ctx->silero) { g_load_error = err; return nullptr; }
    } else {
        g_load_error = "component \"" + name + "\" has kind \"" + c->kind + "\", which this build cannot load";
        return nullptr;
    }
    return ctx.release();
}

parakeet_ctx* load_impl(const char* path, const char* component) {
    g_load_error.clear();
    if (!path) { g_load_error = "path is NULL"; return nullptr; }
    const bool bundle = pk::gguf_is_bundle(path);
    if (component && *component) {
        if (!bundle) { g_load_error = std::string(path) + " is not a bundle GGUF, so it has no components"; return nullptr; }
        return load_bundle_component(path, component);
    }
    if (bundle) {
        pk::BundleInfo info;
        std::string err, name;
        if (!pk::read_bundle_info(path, info, &err) || !pk::select_default_component(info, name, &err)) {
            g_load_error = err;
            return nullptr;
        }
        return load_bundle_component(path, name);
    }
    parakeet_ctx* c = load_plain(path);
    if (!c) g_load_error = std::string("cannot load ") + path + " (see the log for the reason)";
    return c;
}

parakeet_ctx* load_plain(const char* gguf_path) {
    if (!gguf_path) return nullptr;
    try {
        auto* ctx = new (std::nothrow) parakeet_ctx();
        if (!ctx) return nullptr;

        // A CED GGUF (general.architecture "ced") is a sound tagger. Check the
        // header first: the ASR loader would misread it.
        if (pk::gguf_is_ced(gguf_path)) {
            ctx->tagger = pk::CedTagger::load(gguf_path);
            if (ctx->tagger) return ctx;
            delete ctx;
            return nullptr;
        }

        // A Silero VAD GGUF (architecture "silero_vad") is a standalone detector.
        if (pk::gguf_is_silero(gguf_path)) {
            std::string err;
            ctx->silero = pk::SileroVad::load(gguf_path, &err);
            if (ctx->silero) return ctx;
            delete ctx;
            return nullptr;
        }

        // A voice-detect GGUF (architecture "voicedetect") is a speaker encoder.
        if (pk::gguf_is_voicedetect(gguf_path)) {
            try {
                ctx->speaker = pk::load_speaker_with_identity(gguf_path, ctx->speaker_identity,
                    [&] { return pk::SpeakerEncoder::load(gguf_path); });
            } catch (...) { delete ctx; return nullptr; }
            if (ctx->speaker) return ctx;
            delete ctx;
            return nullptr;
        }

        // Try ASR first. Model::load returns nullptr if the GGUF is not a
        // valid ASR model (bad/missing file, or arch=="diarization" which
        // Model::load rejects). Then try diarization.
        std::unique_ptr<pk::Model> model = pk::Model::load(gguf_path);
        if (model) {
            ctx->model = std::move(model);
            return ctx;
        }

        // Not an ASR model — try diarization.
        std::unique_ptr<pk::DiarizationModel> diar = pk::DiarizationModel::load(gguf_path);
        if (diar) {
            ctx->diar = std::move(diar);
            return ctx;
        }

        // Neither — load failed entirely.
        delete ctx;
        return nullptr;
    } catch (...) {
        // Never let an exception cross the boundary.
        return nullptr;
    }
}

} // namespace

extern "C" parakeet_ctx* parakeet_capi_load(const char* gguf_path) {
    try { return load_impl(gguf_path, nullptr); }
    catch (...) { g_load_error = "unknown error"; return nullptr; }
}

extern "C" parakeet_ctx* parakeet_capi_load_component(const char* gguf_path, const char* component) {
    try { return load_impl(gguf_path, component); }
    catch (...) { g_load_error = "unknown error"; return nullptr; }
}

extern "C" char* parakeet_capi_bundle_components_json(const char* gguf_path) {
    g_load_error.clear();
    if (!gguf_path) { g_load_error = "path is NULL"; return nullptr; }
    try {
        pk::BundleInfo info;
        std::string err;
        if (!pk::read_bundle_info(gguf_path, info, &err)) { g_load_error = err; return nullptr; }
        char* out = dup_to_c(pk::bundle_components_json(info));
        if (!out) g_load_error = "out of memory";
        return out;
    } catch (...) { g_load_error = "unknown error"; return nullptr; }
}

extern "C" const char* parakeet_capi_load_error(void) {
    return g_load_error.c_str();
}

extern "C" void parakeet_capi_free(parakeet_ctx* ctx) {
    delete ctx;  // safe on nullptr; ~unique_ptr releases the model.
}

extern "C" char* parakeet_capi_transcribe_path_lang(parakeet_ctx* ctx,
                                                    const char* wav_path, int decoder,
                                                    const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
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
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
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
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return 1;
    }
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
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
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

extern "C" char* parakeet_capi_transcribe_path_json_vad(parakeet_ctx* ctx,
                                                        const char* wav_path,
                                                        int decoder) {
    if (!ctx) return nullptr;
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
    if (!wav_path) { ctx->last_error = "wav_path is NULL"; return nullptr; }
    try {
        pk::Audio audio;
        if (!pk::load_audio_16k_mono(wav_path, audio)) {
            ctx->last_error = std::string("failed to load audio: ") + wav_path;
            return nullptr;
        }
        pk::Transcription tr = ctx->model->transcribe_pcm_vad_with_timestamps(
            audio.samples, audio.sample_rate, to_decoder(decoder));
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

namespace {
// Shared body of the two VAD entry points. `pcm` is mono at `rate`; an ASR
// context needs 16 kHz, a Silero context also takes 8 kHz.
char* vad_json_common(parakeet_ctx* ctx, const std::vector<float>& pcm, int rate, const char* options_json) {
    pk::VadRequest req;
    std::string err;
    const pk::VadKind kind = ctx->silero ? pk::VadKind::kSilero : pk::VadKind::kHead;
    if (!pk::parse_vad_options(options_json, req, err, kind)) { ctx->last_error = err; return nullptr; }
    std::string json = ctx->silero ? pk::silero_vad_to_json(*ctx->silero, pcm, rate, req)
                                   : pk::vad_to_json(*ctx->model, pcm, req);
    ctx->last_error.clear();
    char* out = dup_to_c(json);
    if (!out) { ctx->last_error = "out of memory"; return nullptr; }
    return out;
}
bool vad_ctx_ok(parakeet_ctx* ctx) {
    if (ctx->model || ctx->silero) return true;
    ctx->last_error = ctx->diar
        ? "context holds a diarization model; use parakeet_capi_diarize_*"
        : "context has no loaded model or VAD model";
    return false;
}
}  // namespace

extern "C" char* parakeet_capi_vad_pcm_json(parakeet_ctx* ctx, const float* samples,
                                            int n_samples, int sample_rate,
                                            const char* options_json) {
    if (!ctx) return nullptr;
    if (!vad_ctx_ok(ctx)) return nullptr;
    if (n_samples < 0 || (n_samples > 0 && !samples)) { ctx->last_error = "invalid samples buffer"; return nullptr; }
    if (sample_rate <= 0) { ctx->last_error = "invalid sample rate"; return nullptr; }
    try {
        std::vector<float> pcm(samples, samples + n_samples);
        int rate = sample_rate;
        const bool native = ctx->silero && ctx->silero->supports(sample_rate);
        if (!native && sample_rate != 16000) {
            if (!pcm.empty()) pcm = pk::resample_linear(pcm, sample_rate, 16000);
            rate = 16000;
        }
        return vad_json_common(ctx, pcm, rate, options_json);
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_vad_path_json(parakeet_ctx* ctx, const char* wav_path,
                                             const char* options_json) {
    if (!ctx) return nullptr;
    if (!vad_ctx_ok(ctx)) return nullptr;
    if (!wav_path) { ctx->last_error = "wav_path is NULL"; return nullptr; }
    try {
        pk::Audio audio;
        if (!pk::load_audio_16k_mono(wav_path, audio)) {
            ctx->last_error = std::string("failed to load audio: ") + wav_path;
            return nullptr;
        }
        return vad_json_common(ctx, audio.samples, 16000, options_json);
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_transcribe_path_json_vad_with(parakeet_ctx* ctx, parakeet_ctx* vad_ctx,
                                                             const char* wav_path, int decoder,
                                                             const char* options_json) {
    if (!ctx) return nullptr;
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
    if (!wav_path) { ctx->last_error = "wav_path is NULL"; return nullptr; }
    if (vad_ctx && !vad_ctx->silero) { ctx->last_error = "vad_ctx does not hold a Silero VAD model"; return nullptr; }
    try {
        pk::VadRequest req;
        std::string err;
        if (!pk::parse_vad_options(options_json, req, err,
                                   vad_ctx ? pk::VadKind::kSilero : pk::VadKind::kHead)) {
            ctx->last_error = err;
            return nullptr;
        }
        pk::Audio audio;
        if (!pk::load_audio_16k_mono(wav_path, audio)) {
            ctx->last_error = std::string("failed to load audio: ") + wav_path;
            return nullptr;
        }
        pk::Model::VadProbabilityFn fn;
        if (vad_ctx) {
            pk::SileroVad* sv = vad_ctx->silero.get();
            fn = [sv](const std::vector<float>& pcm16k) {
                std::vector<float> p = sv->probabilities(pcm16k.data(), pcm16k.size(), 16000);
                if (p.empty() && !pcm16k.empty()) throw std::runtime_error("Silero VAD failed");
                return p;
            };
        }
        pk::SegmenterOpts so = req.opts;
        pk::Transcription tr = ctx->model->transcribe_pcm_vad_with_timestamps(
            audio.samples, audio.sample_rate, to_decoder(decoder), "", so, vad_ctx ? &fn : nullptr);
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

// Streaming Silero VAD: a SileroVad::Stream plus the event tracker that turns
// its probabilities into speech start / end events.
struct parakeet_vad_stream {
    parakeet_ctx* ctx = nullptr;   // borrowed; must outlive the stream
    pk::SileroVad::Stream stream;
    std::unique_ptr<pk::VadEventTracker> tracker;
    int sample_rate = 16000;
    bool probabilities = false;
    bool finished = false;
    unsigned long long samples = 0;
};

extern "C" parakeet_vad_stream* parakeet_capi_vad_stream_begin(parakeet_ctx* vad, int sample_rate,
                                                               const char* options_json) {
    if (!vad) return nullptr;
    if (!vad->silero) { vad->last_error = "context does not hold a Silero VAD model"; return nullptr; }
    if (!vad->silero->supports(sample_rate)) { vad->last_error = "Silero VAD streams need 16000 or 8000 Hz audio"; return nullptr; }
    try {
        pk::VadRequest req;
        std::string err;
        if (!pk::parse_vad_options(options_json, req, err, pk::VadKind::kSilero)) { vad->last_error = err; return nullptr; }
        if (req.mode != pk::VadRequest::Mode::kSpeech) { vad->last_error = "VAD streams support mode \"speech\" only"; return nullptr; }
        req.opts.frame_sec = pk::kSileroFrameSec;
        auto* s = new (std::nothrow) parakeet_vad_stream();
        if (!s) { vad->last_error = "out of memory"; return nullptr; }
        s->ctx = vad;
        s->stream = vad->silero->new_stream(sample_rate);
        s->tracker.reset(new pk::VadEventTracker(req.opts));
        s->sample_rate = sample_rate;
        s->probabilities = req.probabilities;
        if (!s->stream.valid()) { delete s; vad->last_error = "failed to create the Silero stream"; return nullptr; }
        vad->last_error.clear();
        return s;
    } catch (const std::exception& e) {
        vad->last_error = e.what();
        return nullptr;
    } catch (...) {
        vad->last_error = "unknown error";
        return nullptr;
    }
}

extern "C" char* parakeet_capi_vad_stream_feed_json(parakeet_vad_stream* s, const float* pcm,
                                                    int n_samples, int is_last) {
    if (!s) return nullptr;
    parakeet_ctx* ctx = s->ctx;
    if (n_samples < 0 || (n_samples > 0 && !pcm)) { ctx->last_error = "invalid samples buffer"; return nullptr; }
    if (s->finished) { ctx->last_error = "stream is finished; call parakeet_capi_vad_stream_reset"; return nullptr; }
    try {
        const long long first = s->tracker->frames();
        std::vector<float> probs;
        if (!s->stream.process_chunk(pcm, (size_t)n_samples, &probs)) {
            ctx->last_error = "Silero VAD failed";
            return nullptr;
        }
        s->samples += (unsigned long long)n_samples;
        if (is_last && !s->stream.flush(&probs)) { ctx->last_error = "Silero VAD failed"; return nullptr; }
        std::vector<pk::VadEvent> ev;
        for (float p : probs) s->tracker->push(p, &ev);
        if (is_last) {
            s->tracker->finish((double)s->samples / (double)s->sample_rate, &ev);
            s->finished = true;
        }
        char buf[96];
        std::snprintf(buf, sizeof(buf), "{\"frame_sec\":%.3f,\"first_frame\":%lld,\"events\":[",
                      pk::kSileroFrameSec, first);
        std::string j = buf;
        for (size_t i = 0; i < ev.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "%s{\"type\":\"%s\",\"time\":%.3f}", i ? "," : "",
                          ev[i].start ? "start" : "end", ev[i].time);
            j += buf;
        }
        j += ']';
        if (s->probabilities) {
            j += ",\"probabilities\":[";
            for (size_t i = 0; i < probs.size(); ++i) {
                std::snprintf(buf, sizeof(buf), "%s%.4f", i ? "," : "", (double)probs[i]);
                j += buf;
            }
            j += ']';
        }
        j += '}';
        ctx->last_error.clear();
        char* out = dup_to_c(j);
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

extern "C" int parakeet_capi_vad_stream_reset(parakeet_vad_stream* s) {
    if (!s) return -1;
    s->stream.reset();
    s->tracker->reset();
    s->samples = 0;
    s->finished = false;
    return 0;
}

extern "C" void parakeet_capi_vad_stream_free(parakeet_vad_stream* s) {
    delete s;
}

extern "C" char* parakeet_capi_transcribe_pcm_batch_json_lang(parakeet_ctx* ctx,
        const float* samples_concat, const int* n_samples, int n_clips,
        int sample_rate, int decoder, const char* target_lang) {
    if (!ctx) return nullptr;
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
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
    if (!ctx->model) {
        ctx->last_error = ctx->diar
            ? "context holds a diarization model; use parakeet_capi_diarize_*"
            : "context has no loaded model";
        return nullptr;
    }
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
    return ctx->last_error.read();
}

extern "C" int parakeet_capi_set_concurrency(parakeet_ctx* ctx, int backends,
                                             int threads_each) {
    if (!ctx) return 0;
    if (!ctx->model) {
        ctx->last_error = "set_concurrency needs a context with a loaded ASR model";
        return 0;
    }
    try {
        return ctx->model->set_concurrency(backends, threads_each);
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return 0;
    } catch (...) {
        ctx->last_error = "unknown error";
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Speaker diarization + speaker-attributed ASR (ABI v7)
// ---------------------------------------------------------------------------

namespace {

bool require_diar(parakeet_ctx* ctx) {
    if (!ctx) return false;
    if (!ctx->diar) {
        ctx->last_error = ctx->model  ? "context holds an ASR model; diarize_* needs a diarization model"
                         : ctx->tagger ? "context holds a CED sound model; diarize_* needs a diarization model"
                         : ctx->speaker ? "context holds a speaker model; diarize_* needs a diarization model"
                                       : "context has no loaded model";
        return false;
    }
    return true;
}

bool require_asr(parakeet_ctx* ctx) {
    if (!ctx) return false;
    if (!ctx->model) {
        ctx->last_error = ctx->diar   ? "context holds a diarization model; an ASR model is needed here"
                         : ctx->tagger ? "context holds a CED sound model; an ASR model is needed here"
                         : ctx->speaker ? "context holds a speaker model; an ASR model is needed here"
                                       : "context has no loaded model";
        return false;
    }
    return true;
}

constexpr const char* kNoCed = "built without sound tagging (PARAKEET_WITH_CED=OFF)";

bool require_tagger(parakeet_ctx* ctx) {
    if (!ctx) return false;
    if (!pk::CedTagger::available()) { ctx->last_error = kNoCed; return false; }
    if (!ctx->tagger) {
        ctx->last_error = ctx->model ? "context holds an ASR model; a CED sound model is needed here"
                        : ctx->diar  ? "context holds a diarization model; a CED sound model is needed here"
                        : ctx->speaker ? "context holds a speaker model; a CED sound model is needed here"
                                     : "context has no loaded model";
        return false;
    }
    return true;
}

constexpr const char* kNoSpeaker = "built without speaker identification (PARAKEET_WITH_VOICEDETECT=OFF)";

bool require_speaker(parakeet_ctx* ctx) {
    if (!ctx) return false;
    if (!pk::SpeakerEncoder::available()) { ctx->last_error = kNoSpeaker; return false; }
    if (!ctx->speaker) {
        ctx->last_error = ctx->model  ? "context holds an ASR model; a speaker model is needed here"
                        : ctx->diar   ? "context holds a diarization model; a speaker model is needed here"
                        : ctx->tagger ? "context holds a CED sound model; a speaker model is needed here"
                                      : "context has no loaded model";
        return false;
    }
    return true;
}

std::string diar_result_json_string(const pk::DiarizationResult& r) {
    std::string json = "{\"speakers\":";
    pk::append_json_int(json, r.n_speakers);
    json += ",\"segments\":[";
    for (size_t i = 0; i < r.segments.size(); ++i) {
        if (i) json += ',';
        json += "{\"speaker\":";
        pk::append_json_int(json, r.segments[i].speaker);
        json += ",\"start\":";
        pk::append_json_float(json, "%.2f", r.segments[i].start);
        json += ",\"end\":";
        pk::append_json_float(json, "%.2f", r.segments[i].end);
        json += '}';
    }
    json += "]}";
    return json;
}

char* diar_result_to_json(const pk::DiarizationResult& r) { return dup_to_c(diar_result_json_string(r)); }

// {"0":{"name":"ada","score":0.9312},"1":{...}} for the slots the identifier has seen.
std::string names_to_json(const std::map<int, pk::SlotName>& names) {
    std::string s = "{";
    bool first = true;
    for (const auto& kv : names) {
        if (!first) s += ',';
        first = false;
        s += "\"" + std::to_string(kv.first) + "\":{\"name\":";
        pk::append_json_string(s, kv.second.name);
        s += ",\"score\":";
        pk::append_json_float(s, "%.4f", kv.second.score);
        s += '}';
    }
    return s + "}";
}

template <typename T>
void append_speaker_item(std::string& s, const T& x, const char* time_fmt,
                         const std::map<int, pk::SlotName>* names = nullptr) {
    s += "{\"speaker\":";
    pk::append_json_int(s, x.speaker);
    if (names) {
        s += ",\"name\":";
        pk::append_json_string(s, x.name);
        s += ",\"name_score\":";
        pk::append_json_float(s, "%.4f", x.name_score);
    }
    s += ",\"text\":";
    pk::append_json_string(s, x.text);
    s += ",\"start\":";
    pk::append_json_float(s, time_fmt, x.start);
    s += ",\"end\":";
    pk::append_json_float(s, time_fmt, x.end);
    s += ",\"conf\":";
    pk::append_json_float(s, "%.3f", x.conf);
    s += '}';
}

// Copy utterances into a malloc'd C array. Returns false on allocation failure.
bool to_c_results(const std::vector<pk::SpeakerUtterance>& utts,
                  parakeet_sas_result** out, int* n_out) {
    *out = nullptr;
    *n_out = 0;
    if (utts.empty()) return true;
    auto* r = static_cast<parakeet_sas_result*>(std::calloc(utts.size(), sizeof(parakeet_sas_result)));
    if (!r) return false;
    for (size_t i = 0; i < utts.size(); ++i) {
        r[i].speaker = utts[i].speaker;
        r[i].text    = dup_to_c(utts[i].text);
        r[i].start   = utts[i].start;
        r[i].end     = utts[i].end;
        r[i].conf    = utts[i].conf;
        if (!r[i].text) { parakeet_capi_free_sas_results(r, (int)i); return false; }
    }
    *out = r;
    *n_out = (int)utts.size();
    return true;
}

// ASR + diarization on the same audio, merged per word.
bool run_sas(parakeet_ctx* asr_ctx, parakeet_ctx* diar_ctx,
             const float* samples, int n_samples, int sample_rate,
             std::vector<pk::SpeakerWord>& words, int& n_speakers,
             std::vector<pk::SpeakerSegment>* segs_out = nullptr,
             std::vector<float>* pcm16k_out = nullptr) {
    if (!require_asr(asr_ctx) || !require_diar(diar_ctx)) return false;
    if (!samples || n_samples < 0) {
        asr_ctx->last_error = "invalid samples buffer";
        return false;
    }
    const std::vector<float> pcm(samples, samples + n_samples);
    pk::Transcription tr;
    try {
        tr = asr_ctx->model->transcribe_with_timestamps(pcm, sample_rate);
    } catch (const std::exception& e) {
        asr_ctx->last_error = e.what();
        return false;
    }
    pk::DiarizationResult dr;
    try {
        dr = diar_ctx->diar->diarize_pcm(pcm, sample_rate);
    } catch (const std::exception& e) {
        diar_ctx->last_error = e.what();
        return false;
    }
    n_speakers = dr.n_speakers;
    words = pk::merge_asr_diarization(tr.words, dr.segments);
    if (segs_out) *segs_out = dr.segments;
    if (pcm16k_out) *pcm16k_out = sample_rate == 16000 ? pcm : pk::resample_linear(pcm, sample_rate, 16000);
    asr_ctx->last_error.clear();
    diar_ctx->last_error.clear();
    return true;
}

}  // namespace

extern "C" char* parakeet_capi_diarize_path(parakeet_ctx* ctx, const char* wav_path) {
    if (!require_diar(ctx)) return nullptr;
    if (!wav_path) { ctx->last_error = "wav_path is NULL"; return nullptr; }
    try {
        char* out = diar_result_to_json(ctx->diar->diarize_path(wav_path));
        ctx->last_error.clear();
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    } catch (...) {
        ctx->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" char* parakeet_capi_diarize_pcm(parakeet_ctx* ctx, const float* samples,
                                           int n_samples, int sample_rate) {
    if (!require_diar(ctx)) return nullptr;
    if (!samples || n_samples < 0) { ctx->last_error = "invalid samples buffer"; return nullptr; }
    try {
        const std::vector<float> pcm(samples, samples + n_samples);
        char* out = diar_result_to_json(ctx->diar->diarize_pcm(pcm, sample_rate));
        ctx->last_error.clear();
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    } catch (...) {
        ctx->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" int parakeet_capi_transcribe_and_diarize(parakeet_ctx* asr_ctx, parakeet_ctx* diar_ctx,
                                                    const float* samples, int n_samples,
                                                    int sample_rate,
                                                    parakeet_sas_result** out, int* n_out) {
    if (!out || !n_out) return 1;
    *out = nullptr;
    *n_out = 0;
    try {
        std::vector<pk::SpeakerWord> words;
        int n_speakers = 0;
        if (!run_sas(asr_ctx, diar_ctx, samples, n_samples, sample_rate, words, n_speakers))
            return 1;
        if (!to_c_results(pk::group_speaker_words(words), out, n_out)) {
            asr_ctx->last_error = "out of memory";
            return 1;
        }
        return 0;
    } catch (...) {
        if (asr_ctx) asr_ctx->last_error = "unknown error";
        return 1;
    }
}

extern "C" void parakeet_capi_free_sas_results(parakeet_sas_result* results, int n) {
    if (!results) return;
    for (int i = 0; i < n; ++i) std::free(results[i].text);
    std::free(results);
}

extern "C" char* parakeet_capi_transcribe_and_diarize_json(parakeet_ctx* asr_ctx,
                                                           parakeet_ctx* diar_ctx,
                                                           const float* samples, int n_samples,
                                                           int sample_rate) {
    try {
        std::vector<pk::SpeakerWord> words;
        int n_speakers = 0;
        if (!run_sas(asr_ctx, diar_ctx, samples, n_samples, sample_rate, words, n_speakers))
            return nullptr;
        const std::vector<pk::SpeakerUtterance> utts = pk::group_speaker_words(words);
        std::string s = "{\"speakers\":";
        pk::append_json_int(s, n_speakers);
        s += ",\"utterances\":[";
        for (size_t i = 0; i < utts.size(); ++i) {
            if (i) s += ',';
            append_speaker_item(s, utts[i], "%.2f");
        }
        s += "],\"words\":[";
        for (size_t i = 0; i < words.size(); ++i) {
            if (i) s += ',';
            append_speaker_item(s, words[i], "%.3f");
        }
        s += "]}";
        return dup_to_c(s);
    } catch (...) {
        if (asr_ctx) asr_ctx->last_error = "unknown error";
        return nullptr;
    }
}

// --- Streaming diarization -------------------------------------------------

struct parakeet_diar_stream {
    parakeet_ctx* ctx = nullptr;
    std::unique_ptr<pk::DiarPcmStream> ds;
};

namespace {

pk::DiarLatency latency_from_int(int latency) {
    switch (latency) {
        case PARAKEET_DIAR_LATENCY_LOW: return pk::DiarLatency::Low;
        case PARAKEET_DIAR_LATENCY_VERY_LOW: return pk::DiarLatency::VeryLow;
        case PARAKEET_DIAR_LATENCY_ULTRA_LOW: return pk::DiarLatency::UltraLow;
        default: return pk::DiarLatency::Model;
    }
}

// Copy segments into a malloc'd C array (NULL when empty). False on OOM.
bool to_c_segments(const std::vector<pk::StreamingSpeakerSegment>& segs,
                   parakeet_diar_segment** out, int* n_out) {
    *out = nullptr;
    *n_out = 0;
    if (segs.empty()) return true;
    auto* r = static_cast<parakeet_diar_segment*>(std::malloc(segs.size() * sizeof(parakeet_diar_segment)));
    if (!r) return false;
    for (size_t i = 0; i < segs.size(); ++i) r[i] = {segs[i].speaker, segs[i].start, segs[i].end};
    *out = r;
    *n_out = (int)segs.size();
    return true;
}

}  // namespace

extern "C" parakeet_diar_stream* parakeet_capi_diarize_stream_begin_latency(parakeet_ctx* diar_ctx,
                                                                            int latency) {
    if (!require_diar(diar_ctx)) return nullptr;
    if (latency < PARAKEET_DIAR_LATENCY_MODEL || latency > PARAKEET_DIAR_LATENCY_ULTRA_LOW) {
        diar_ctx->last_error = "unknown diarization latency mode";
        return nullptr;
    }
    try {
        auto ds = std::make_unique<pk::DiarPcmStream>(*diar_ctx->diar, latency_from_int(latency));
        auto* s = new parakeet_diar_stream();
        s->ctx = diar_ctx;
        s->ds = std::move(ds);
        diar_ctx->last_error.clear();
        return s;
    } catch (const std::exception& e) {
        diar_ctx->last_error = e.what();
    } catch (...) {
        diar_ctx->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" parakeet_diar_stream* parakeet_capi_diarize_stream_begin(parakeet_ctx* diar_ctx) {
    return parakeet_capi_diarize_stream_begin_latency(diar_ctx, PARAKEET_DIAR_LATENCY_MODEL);
}

extern "C" int parakeet_capi_diarize_stream_chunk_samples(parakeet_diar_stream* s) {
    if (!s) return 0;
    return s->ds->chunk_samples();
}

extern "C" float parakeet_capi_diarize_stream_time(parakeet_diar_stream* s) {
    if (!s) return 0.0f;
    return (float)(s->ds->sd().frames_done() * s->ds->sd().frame_sec());
}

extern "C" int parakeet_capi_diarize_stream_active(parakeet_diar_stream* s,
                                                   parakeet_diar_segment** out, int* n_out) {
    if (!s || !out || !n_out) return 1;
    if (!to_c_segments(s->ds->open_segments(), out, n_out)) {
        s->ctx->last_error = "out of memory";
        return 1;
    }
    return 0;
}

extern "C" int parakeet_capi_diarize_stream_feed(parakeet_diar_stream* s, const float* pcm,
                                                 int n_samples, int is_last,
                                                 parakeet_diar_segment** out, int* n_out) {
    if (!s || !out || !n_out) return 1;
    *out = nullptr;
    *n_out = 0;
    if ((!pcm && n_samples > 0) || n_samples < 0) { s->ctx->last_error = "invalid samples buffer"; return 1; }
    if (s->ds->finished()) { s->ctx->last_error = "stream already finished"; return 1; }
    try {
        std::vector<pk::StreamingSpeakerSegment> segs;
        s->ds->feed(pcm, n_samples, is_last != 0, segs);
        if (!to_c_segments(segs, out, n_out)) { s->ctx->last_error = "out of memory"; return 1; }
        s->ctx->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
    } catch (...) {
        s->ctx->last_error = "unknown error";
    }
    return 1;
}

extern "C" void parakeet_capi_free_diar_segments(parakeet_diar_segment* segs) {
    std::free(segs);
}

extern "C" void parakeet_capi_diarize_stream_free(parakeet_diar_stream* s) {
    delete s;
}

// --- Streaming speaker-attributed ASR ---------------------------------------

// A pk::SceneStream over an ASR and a diarization context (no sound part).
struct parakeet_sas_stream {
    parakeet_ctx* asr = nullptr;
    parakeet_ctx* diar = nullptr;
    std::unique_ptr<pk::SceneStream> scene;
};

extern "C" parakeet_sas_stream* parakeet_capi_sas_stream_begin_latency(parakeet_ctx* asr_ctx,
                                                                       parakeet_ctx* diar_ctx,
                                                                       int latency) {
    if (!require_asr(asr_ctx) || !require_diar(diar_ctx)) return nullptr;
    if (latency < PARAKEET_DIAR_LATENCY_MODEL || latency > PARAKEET_DIAR_LATENCY_ULTRA_LOW) {
        diar_ctx->last_error = "unknown diarization latency mode";
        return nullptr;
    }
    try {
        pk::SceneParts parts;
        parts.asr = asr_ctx->model.get();
        parts.diar = diar_ctx->diar.get();
        parts.diar_latency = latency_from_int(latency);
        auto scene = std::make_unique<pk::SceneStream>(parts);
        auto* s = new parakeet_sas_stream();
        s->asr = asr_ctx;
        s->diar = diar_ctx;
        s->scene = std::move(scene);
        diar_ctx->last_error.clear();
        return s;
    } catch (const std::exception& e) {
        diar_ctx->last_error = e.what();
    } catch (...) {
        diar_ctx->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" parakeet_sas_stream* parakeet_capi_sas_stream_begin(parakeet_ctx* asr_ctx,
                                                               parakeet_ctx* diar_ctx) {
    return parakeet_capi_sas_stream_begin_latency(asr_ctx, diar_ctx, PARAKEET_DIAR_LATENCY_MODEL);
}

extern "C" int parakeet_capi_sas_stream_feed(parakeet_sas_stream* s, const float* pcm,
                                             int n_samples, int is_last,
                                             parakeet_sas_result** out, int* n_out) {
    if (!s || !out || !n_out) return 1;
    *out = nullptr;
    *n_out = 0;
    if ((!pcm && n_samples > 0) || n_samples < 0) { s->asr->last_error = "invalid samples buffer"; return 1; }
    if (s->scene->finished()) { s->asr->last_error = "stream already finished"; return 1; }
    try {
        const pk::SceneUpdate u = s->scene->feed(pcm, n_samples, is_last != 0);
        if (!to_c_results(u.utterances, out, n_out)) { s->asr->last_error = "out of memory"; return 1; }
        // Any successful feed clears the ASR ctx's last error, also one
        // that commits nothing.
        s->asr->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        (s->scene->failed_part() == pk::ScenePart::Asr ? s->asr : s->diar)->last_error = e.what();
    } catch (...) {
        (s->scene->failed_part() == pk::ScenePart::Asr ? s->asr : s->diar)->last_error = "unknown error";
    }
    return 1;
}

extern "C" void parakeet_capi_sas_stream_free(parakeet_sas_stream* s) {
    delete s;
}

// ---------------------------------------------------------------------------
// Sound events (ABI v8)
// ---------------------------------------------------------------------------

namespace {

pk::SoundOpts to_sound_opts(const parakeet_sound_opts* o) {
    pk::SoundOpts s;
    if (!o) return s;
    // Only read fields the caller's struct has (size versioning).
    auto has = [&](size_t end) { return o->size >= (int)end; };
    if (has(offsetof(parakeet_sound_opts, hop_sec) + sizeof(float))) {
        s.window_sec = o->window_sec;
        s.hop_sec = o->hop_sec;
    }
    if (has(offsetof(parakeet_sound_opts, min_duration_sec) + sizeof(float))) {
        s.on_threshold = o->on_threshold;
        s.off_threshold = o->off_threshold;
        s.min_duration_sec = o->min_duration_sec;
    }
    if (has(offsetof(parakeet_sound_opts, top_k) + sizeof(int))) s.top_k = o->top_k;
    return s;
}

bool to_c_sound_segments(const std::vector<pk::SoundSegment>& in, const pk::CedTagger& t,
                         parakeet_sound_segment** out, int* n_out) {
    *out = nullptr;
    *n_out = 0;
    if (in.empty()) return true;
    auto* a = static_cast<parakeet_sound_segment*>(std::malloc(in.size() * sizeof(parakeet_sound_segment)));
    if (!a) return false;
    for (size_t i = 0; i < in.size(); ++i) {
        const char* l = t.label(in[i].cls);
        a[i] = {in[i].cls, l ? l : "", in[i].start, in[i].end, in[i].peak};
    }
    *out = a;
    *n_out = (int)in.size();
    return true;
}

} // namespace

struct parakeet_sound_stream {
    parakeet_ctx* ctx = nullptr;
    std::unique_ptr<pk::SoundStream> ss;
};

extern "C" void parakeet_capi_sound_opts_default(parakeet_sound_opts* o) {
    if (!o) return;
    const pk::SoundOpts d;
    *o = {(int)sizeof(*o), d.window_sec, d.hop_sec, d.on_threshold, d.off_threshold,
          d.min_duration_sec, d.top_k};
}

extern "C" parakeet_sound_stream* parakeet_capi_sound_stream_begin(parakeet_ctx* tagger,
                                                                   const parakeet_sound_opts* o) {
    if (!require_tagger(tagger)) return nullptr;
    try {
        const pk::SoundOpts so = to_sound_opts(o);
        const std::string err = pk::validate_sound_opts(so, tagger->tagger->n_classes());
        if (!err.empty()) { tagger->last_error = "invalid sound options: " + err; return nullptr; }
        auto* s = new parakeet_sound_stream();
        s->ctx = tagger;
        s->ss = std::make_unique<pk::SoundStream>(tagger->tagger->scorer(),
                                                  tagger->tagger->n_classes(), so);
        tagger->last_error.clear();
        return s;
    } catch (const std::exception& e) {
        tagger->last_error = e.what();
    } catch (...) {
        tagger->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" int parakeet_capi_sound_stream_feed(parakeet_sound_stream* s, const float* pcm, int n,
                                               int is_last, parakeet_sound_segment** out, int* n_out) {
    if (!s || !out || !n_out) return 1;
    *out = nullptr;
    *n_out = 0;
    if ((!pcm && n > 0) || n < 0) { s->ctx->last_error = "invalid samples buffer"; return 1; }
    if (s->ss->finished()) { s->ctx->last_error = "stream already finished"; return 1; }
    try {
        auto closed = s->ss->feed(pcm, n, is_last != 0);
        if (!to_c_sound_segments(closed, *s->ctx->tagger, out, n_out)) {
            s->ctx->last_error = "out of memory";
            return 1;
        }
        s->ctx->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
        const std::string& detail = s->ctx->tagger->last_error();
        if (!detail.empty()) s->ctx->last_error += ": " + detail;
    } catch (...) {
        s->ctx->last_error = "unknown error";
    }
    return 1;
}

extern "C" int parakeet_capi_sound_stream_active(parakeet_sound_stream* s,
                                                 parakeet_sound_segment** out, int* n_out) {
    if (!s || !out || !n_out) return 1;
    *out = nullptr;
    *n_out = 0;
    try {
        if (!to_c_sound_segments(s->ss->open_segments(), *s->ctx->tagger, out, n_out)) {
            s->ctx->last_error = "out of memory";
            return 1;
        }
        s->ctx->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
    } catch (...) {
        s->ctx->last_error = "unknown error";
    }
    return 1;
}

extern "C" char* parakeet_capi_sound_stream_drain_scores_json(parakeet_sound_stream* s) {
    if (!s) return nullptr;
    try {
        const pk::CedTagger& t = *s->ctx->tagger;
        char* out = dup_to_c(pk::sound_windows_to_json(s->ss->drain_windows(),
                                                        [&](int i) { return t.label(i); }));
        s->ctx->last_error.clear();
        return out;
    } catch (const std::exception& e) {
        s->ctx->last_error = e.what();
    } catch (...) {
        s->ctx->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" void parakeet_capi_free_sound_segments(parakeet_sound_segment* segs) { std::free(segs); }

extern "C" void parakeet_capi_sound_stream_free(parakeet_sound_stream* s) { delete s; }

// ---------------------------------------------------------------------------
// Sound events (ABI v8): CED tagger introspection
// ---------------------------------------------------------------------------

extern "C" int parakeet_capi_num_classes(const parakeet_ctx* ctx) {
    return (ctx && ctx->tagger) ? ctx->tagger->n_classes() : -1;
}

extern "C" const char* parakeet_capi_class_label(const parakeet_ctx* ctx, int index) {
    return (ctx && ctx->tagger) ? ctx->tagger->label(index) : nullptr;
}

extern "C" int parakeet_capi_model_kind(const parakeet_ctx* ctx) {
    if (!ctx) return PARAKEET_MODEL_KIND_NONE;
    if (ctx->model) return PARAKEET_MODEL_KIND_ASR;
    if (ctx->diar) return PARAKEET_MODEL_KIND_DIARIZATION;
    if (ctx->tagger) return PARAKEET_MODEL_KIND_SOUND;
    if (ctx->speaker) return PARAKEET_MODEL_KIND_SPEAKER;
    if (ctx->silero) return PARAKEET_MODEL_KIND_VAD;
    return PARAKEET_MODEL_KIND_NONE;
}

// ---------------------------------------------------------------------------
// Combined scene stream (ABI v8)
// ---------------------------------------------------------------------------

// A pk::SceneStream over up to three contexts (ASR, diarization, tagger).
struct parakeet_scene_stream {
    parakeet_ctx* asr_ctx = nullptr;
    parakeet_ctx* diar_ctx = nullptr;
    parakeet_ctx* tagger_ctx = nullptr;
    parakeet_ctx* speaker_ctx = nullptr;
    std::unique_ptr<pk::SceneStream> scene;
    std::string last_error;
};

namespace {

// Which context was running when the scene stream last threw, mirroring the
// sas_stream wrapper's diar/asr attribution, extended with the tagger.
parakeet_ctx* scene_failed_ctx(parakeet_scene_stream* s) {
    switch (s->scene->failed_part()) {
        case pk::ScenePart::Diarization: return s->diar_ctx ? s->diar_ctx : s->asr_ctx;
        case pk::ScenePart::Asr:         return s->asr_ctx ? s->asr_ctx : s->diar_ctx;
        case pk::ScenePart::Sound:       return s->tagger_ctx;
        case pk::ScenePart::Speaker:     return s->speaker_ctx ? s->speaker_ctx : s->diar_ctx;
        default:                         return s->asr_ctx ? s->asr_ctx
                                                : s->diar_ctx ? s->diar_ctx : s->tagger_ctx;
    }
}

}  // namespace

extern "C" void parakeet_capi_scene_opts_default(parakeet_scene_opts* o) {
    if (!o) return;
    o->size = (int)sizeof(*o);
    o->diar_latency = PARAKEET_DIAR_LATENCY_MODEL;
    parakeet_capi_sound_opts_default(&o->sound);
    o->flags = 0;
    const pk::SpeakerIdOpts d;
    o->speaker_accept_threshold = d.accept_threshold;
    o->speaker_margin = d.margin;
    o->speaker_min_voice_sec = d.min_voice_sec;
    o->speaker_refresh_sec = d.refresh_sec;
    o->speaker_max_voice_sec = d.max_voice_sec;
}

namespace {

// Reads the float at byte offset `off` of `o` only when the caller's `size`
// covers it (a caller built against an older header owns a shorter struct, so
// nothing past `size` may be touched). A zero or uncovered field returns
// `dflt`.
float scene_float_field(const parakeet_scene_opts* o, size_t off, float dflt) {
    if (o->size < (int)(off + sizeof(float))) return dflt;
    float v;
    std::memcpy(&v, reinterpret_cast<const char*>(o) + off, sizeof(v));
    return v != 0.0f ? v : dflt;
}

}  // namespace

extern "C" parakeet_scene_stream* parakeet_capi_scene_stream_begin(parakeet_ctx* asr, parakeet_ctx* diar,
                                                                    parakeet_ctx* tagger,
                                                                    const parakeet_scene_opts* o) {
    return parakeet_capi_scene_stream_begin_speaker(asr, diar, tagger, nullptr, nullptr, o);
}

extern "C" parakeet_scene_stream* parakeet_capi_scene_stream_begin_speaker(
    parakeet_ctx* asr, parakeet_ctx* diar, parakeet_ctx* tagger, parakeet_ctx* speaker,
    parakeet_speaker_registry* registry, const parakeet_scene_opts* o) {
    if (speaker) {
        if (!require_speaker(speaker)) return nullptr;
        if (!diar) { speaker->last_error = "speaker identification needs a diarization model"; return nullptr; }
        if (!registry) { speaker->last_error = "speaker identification needs a registry"; return nullptr; }
    }
    if (!asr && !diar && !tagger) return nullptr;
    if ((asr && !require_asr(asr)) || (diar && !require_diar(diar)) || (tagger && !require_tagger(tagger)))
        return nullptr;
    parakeet_scene_opts def;
    parakeet_capi_scene_opts_default(&def);
    if (!o) o = &def;
    if (o->size >= (int)(offsetof(parakeet_scene_opts, flags) + sizeof(int)) && o->flags != 0) {
        (asr ? asr : diar ? diar : tagger)->last_error = "scene flags must be 0";
        return nullptr;
    }
    if (diar && (o->diar_latency < PARAKEET_DIAR_LATENCY_MODEL ||
                 o->diar_latency > PARAKEET_DIAR_LATENCY_ULTRA_LOW)) {
        diar->last_error = "unknown diarization latency mode";
        return nullptr;
    }
    pk::SpeakerIdOpts so;
    if (speaker) {
        so.accept_threshold = scene_float_field(o, offsetof(parakeet_scene_opts, speaker_accept_threshold), so.accept_threshold);
        so.margin = scene_float_field(o, offsetof(parakeet_scene_opts, speaker_margin), so.margin);
        so.min_voice_sec = scene_float_field(o, offsetof(parakeet_scene_opts, speaker_min_voice_sec), so.min_voice_sec);
        so.refresh_sec = scene_float_field(o, offsetof(parakeet_scene_opts, speaker_refresh_sec), so.refresh_sec);
        so.max_voice_sec = scene_float_field(o, offsetof(parakeet_scene_opts, speaker_max_voice_sec), so.max_voice_sec);
        const std::string err = pk::validate_speaker_opts(so);
        if (!err.empty()) { speaker->last_error = "invalid speaker options: " + err; return nullptr; }
        const int rd = registry->reg.dim();
        if (rd != 0 && rd != speaker->speaker->dim()) {
            speaker->last_error = "registry holds " + std::to_string(rd) +
                                  "-value embeddings, this model produces " +
                                  std::to_string(speaker->speaker->dim());
            return nullptr;
        }
    }
    try {
        pk::SceneParts p;
        if (speaker) {
            p.speaker_embed = speaker->speaker->embedder();
            p.registry = &registry->reg;
            p.speaker_opts = so;
        }
        p.asr = asr ? asr->model.get() : nullptr;
        p.diar = diar ? diar->diar.get() : nullptr;
        p.diar_latency = latency_from_int(o->diar_latency);
        p.tagger = tagger ? tagger->tagger.get() : nullptr;
        p.sound = to_sound_opts(&o->sound);
        if (tagger) {
            const std::string err = pk::validate_sound_opts(p.sound, tagger->tagger->n_classes());
            if (!err.empty()) { tagger->last_error = "invalid sound options: " + err; return nullptr; }
        }
        auto scene = std::make_unique<pk::SceneStream>(p);
        auto* s = new parakeet_scene_stream();
        s->scene = std::move(scene);
        s->asr_ctx = asr;
        s->diar_ctx = diar;
        s->tagger_ctx = tagger;
        s->speaker_ctx = speaker;
        if (asr) asr->last_error.clear();
        if (diar) diar->last_error.clear();
        if (tagger) tagger->last_error.clear();
        if (speaker) speaker->last_error.clear();
        return s;
    } catch (const std::exception& e) {
        (speaker ? speaker : asr ? asr : diar ? diar : tagger)->last_error = e.what();
    } catch (...) {
        (speaker ? speaker : asr ? asr : diar ? diar : tagger)->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" char* parakeet_capi_scene_stream_feed_json(parakeet_scene_stream* s, const float* pcm,
                                                       int n, int is_last) {
    if (!s) return nullptr;
    if ((!pcm && n > 0) || n < 0) { s->last_error = "invalid samples buffer"; return nullptr; }
    if (s->scene->finished()) { s->last_error = "stream already finished"; return nullptr; }
    try {
        const pk::SceneUpdate u = s->scene->feed(pcm, n, is_last != 0);
        const pk::CedTagger* t = s->tagger_ctx ? s->tagger_ctx->tagger.get() : nullptr;
        s->last_error.clear();
        return dup_to_c(pk::scene_update_to_json(u, [t](int i) { return t ? t->label(i) : nullptr; }));
    } catch (const std::exception& e) {
        s->last_error = e.what();
        if (parakeet_ctx* c = scene_failed_ctx(s)) c->last_error = e.what();
    } catch (...) {
        s->last_error = "unknown error";
        if (parakeet_ctx* c = scene_failed_ctx(s)) c->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" char* parakeet_capi_scene_stream_drain_scores_json(parakeet_scene_stream* s) {
    if (!s) return nullptr;
    try {
        const pk::CedTagger* t = s->tagger_ctx ? s->tagger_ctx->tagger.get() : nullptr;
        char* out = dup_to_c(pk::sound_windows_to_json(s->scene->drain_windows(),
                                                        [t](int i) { return t ? t->label(i) : nullptr; }));
        s->last_error.clear();
        return out;
    } catch (const std::exception& e) {
        s->last_error = e.what();
    } catch (...) {
        s->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" const char* parakeet_capi_scene_stream_last_error(parakeet_scene_stream* s) {
    return s ? s->last_error.c_str() : "";
}

extern "C" void parakeet_capi_scene_stream_free(parakeet_scene_stream* s) { delete s; }

// ---------------------------------------------------------------------------
// Speaker identification (ABI v9)
// ---------------------------------------------------------------------------

namespace {

// Embeds mono PCM at `sample_rate` with the speaker ctx. Errors go on the ctx.
bool speaker_embed_pcm(parakeet_ctx* speaker, const float* pcm, int n, int sample_rate,
                       std::vector<float>& emb) {
    if (!pcm || n <= 0) { speaker->last_error = "no audio"; return false; }
    if (sample_rate <= 0) { speaker->last_error = "invalid sample rate"; return false; }
    bool ok;
    if (sample_rate == 16000) {
        ok = speaker->speaker->embed(pcm, n, emb);
    } else {
        const std::vector<float> in(pcm, pcm + n);
        const std::vector<float> r = pk::resample_linear(in, sample_rate, 16000);
        if (r.empty()) { speaker->last_error = "no audio"; return false; }
        ok = speaker->speaker->embed(r.data(), (int)r.size(), emb);
    }
    if (!ok) {
        const std::string& e = speaker->speaker->last_error();
        speaker->last_error = e.empty() ? "speaker embedding failed" : e;
        return false;
    }
    return true;
}

}  // namespace

extern "C" int parakeet_capi_speaker_dim(const parakeet_ctx* ctx) {
    return (ctx && ctx->speaker) ? ctx->speaker->dim() : -1;
}

extern "C" parakeet_speaker_registry* parakeet_capi_speaker_registry_new(void) {
    return new (std::nothrow) parakeet_speaker_registry();
}

extern "C" void parakeet_capi_speaker_registry_free(parakeet_speaker_registry* reg) { delete reg; }

extern "C" int parakeet_capi_speaker_registry_size(const parakeet_speaker_registry* reg) {
    return reg ? (int)reg->reg.size() : 0;
}

extern "C" const char* parakeet_capi_speaker_registry_last_error(const parakeet_speaker_registry* reg) {
    return reg ? reg->last_error.c_str() : "";
}

extern "C" int parakeet_capi_speaker_enroll(parakeet_speaker_registry* reg, parakeet_ctx* speaker,
                                            const char* name, const float* pcm, int n, int sample_rate) {
    if (!speaker) return 1;
    try {
        if (!require_speaker(speaker)) return 1;
        if (!reg) { speaker->last_error = "registry is NULL"; return 1; }
        if (!name || !*name) { speaker->last_error = "speaker name is empty"; return 1; }
        std::vector<float> emb;
        if (!speaker_embed_pcm(speaker, pcm, n, sample_rate, emb)) return 1;
        reg->reg.enroll(name, emb);
        speaker->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        speaker->last_error = e.what();
    } catch (...) {
        speaker->last_error = "unknown error";
    }
    return 1;
}

extern "C" int parakeet_capi_speaker_registry_add_embedding(parakeet_speaker_registry* reg, const char* name,
                                                            const float* embedding, int dim) {
    if (!reg) return 1;
    try {
        if (!name || !*name) { reg->last_error = "speaker name is empty"; return 1; }
        if (!embedding || dim <= 0) { reg->last_error = "no embedding"; return 1; }
        reg->reg.enroll(name, std::vector<float>(embedding, embedding + dim));
        reg->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        reg->last_error = e.what();
    } catch (...) {
        reg->last_error = "unknown error";
    }
    return 1;
}

extern "C" int parakeet_capi_speaker_registry_save(const parakeet_speaker_registry* reg, const char* path) {
    if (!reg) return 1;
    auto* mreg = const_cast<parakeet_speaker_registry*>(reg);   // only last_error is written
    if (!path || !*path) { mreg->last_error = "path is empty"; return 1; }
    try {
        // Written to <path>.tmp and moved over the target, so a failed save
        // never costs the caller the registry file they already had.
        std::string err;
        if (!pk::write_file_atomic(path, reg->reg.serialize(), &err)) {
            mreg->last_error = err;
            return 1;
        }
        mreg->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        mreg->last_error = e.what();
    } catch (...) {
        mreg->last_error = "unknown error";
    }
    return 1;
}

extern "C" parakeet_speaker_registry* parakeet_capi_speaker_registry_load(const char* path) {
    if (!path) return nullptr;
    try {
        std::FILE* f = std::fopen(path, "rb");
        if (!f) return nullptr;
        std::string blob;
        char buf[4096];
        size_t got;
        while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) blob.append(buf, got);
        const bool err = std::ferror(f) != 0;
        std::fclose(f);
        if (err) return nullptr;
        auto* r = new (std::nothrow) parakeet_speaker_registry();
        if (!r) return nullptr;
        try {
            r->reg = pk::SpeakerRegistry::deserialize(blob);
        } catch (...) {
            delete r;
            return nullptr;
        }
        return r;
    } catch (...) {
        return nullptr;
    }
}

extern "C" char* parakeet_capi_speaker_identify_pcm_json(parakeet_speaker_registry* reg, parakeet_ctx* speaker,
                                                         const float* pcm, int n, int sample_rate) {
    if (!speaker) return nullptr;
    try {
        if (!require_speaker(speaker)) return nullptr;
        if (!reg) { speaker->last_error = "registry is NULL"; return nullptr; }
        std::vector<float> emb;
        if (!speaker_embed_pcm(speaker, pcm, n, sample_rate, emb)) return nullptr;
        const pk::SpeakerIdOpts d;
        const pk::SpeakerMatch m = reg->reg.identify(emb, d.accept_threshold, d.margin);
        std::string j = "{\"name\":";
        pk::append_json_string(j, m.name);
        j += ",\"score\":";
        pk::append_json_float(j, "%.4f", m.score);
        j += '}';
        speaker->last_error.clear();
        return dup_to_c(j);
    } catch (const std::exception& e) {
        speaker->last_error = e.what();
    } catch (...) {
        speaker->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" char* parakeet_capi_transcribe_and_diarize_named_json(
    parakeet_ctx* asr_ctx, parakeet_ctx* diar_ctx, parakeet_ctx* speaker,
    parakeet_speaker_registry* registry, const float* samples, int n_samples, int sample_rate) {
    try {
        if (!speaker) return nullptr;
        if (!require_speaker(speaker)) return nullptr;
        if (!registry) { speaker->last_error = "speaker identification needs a registry"; return nullptr; }
        std::vector<pk::SpeakerWord> words;
        std::vector<pk::SpeakerSegment> segs;
        std::vector<float> pcm16k;
        int n_speakers = 0;
        if (!run_sas(asr_ctx, diar_ctx, samples, n_samples, sample_rate, words, n_speakers, &segs, &pcm16k))
            return nullptr;
        const int rd = registry->reg.dim();
        if (rd != 0 && rd != speaker->speaker->dim()) {
            speaker->last_error = "registry holds " + std::to_string(rd) +
                                  "-value embeddings, this model produces " +
                                  std::to_string(speaker->speaker->dim());
            return nullptr;
        }
        std::map<int, pk::SlotName> names;
        try {
            names = pk::identify_offline(pcm16k, segs, speaker->speaker->embedder(), registry->reg,
                                         pk::SpeakerIdOpts{});
        } catch (const std::exception& e) {
            speaker->last_error = e.what();
            return nullptr;
        }
        for (pk::SpeakerWord& w : words) {
            auto it = names.find(w.speaker);
            if (w.speaker >= 0 && it != names.end()) {
                w.name = it->second.name;
                w.name_score = it->second.score;
            }
        }
        const std::vector<pk::SpeakerUtterance> utts = pk::group_speaker_words(words);
        std::string s = "{\"speakers\":";
        pk::append_json_int(s, n_speakers);
        s += ",\"names\":" + names_to_json(names);
        s += ",\"utterances\":[";
        for (size_t i = 0; i < utts.size(); ++i) {
            if (i) s += ',';
            append_speaker_item(s, utts[i], "%.2f", &names);
        }
        s += "],\"words\":[";
        for (size_t i = 0; i < words.size(); ++i) {
            if (i) s += ',';
            append_speaker_item(s, words[i], "%.3f", &names);
        }
        s += "]}";
        speaker->last_error.clear();
        return dup_to_c(s);
    } catch (const std::exception& e) {
        if (speaker) speaker->last_error = e.what();
        return nullptr;
    } catch (...) {
        if (speaker) speaker->last_error = "unknown error";
        return nullptr;
    }
}

static char* diarize_named_pcm_json(bool export_profiles, parakeet_ctx* diar_ctx, parakeet_ctx* speaker,
                                                      parakeet_speaker_registry* registry,
                                                      const float* samples, int n_samples, int sample_rate,
                                                      float accept_threshold, float margin) {
    try {
        // Each failure below sets the one message that applies; clear both so the other ctx
        // never shows an older, unrelated error.
        if (diar_ctx) diar_ctx->last_error.clear();
        if (speaker) speaker->last_error.clear();
        if (!speaker) return nullptr;
        if (!require_diar(diar_ctx)) return nullptr;
        if (!require_speaker(speaker)) return nullptr;
        parakeet_speaker_registry empty;
        if (!registry && export_profiles) registry = &empty;
        if (!registry) { speaker->last_error = "speaker identification needs a registry"; return nullptr; }
        if (!samples || n_samples <= 0) {
            diar_ctx->last_error = "invalid samples buffer";
            return nullptr;
        }
        if (sample_rate <= 0) {
            diar_ctx->last_error = "invalid sample rate";
            return nullptr;
        }
        const int rd = registry->reg.dim();
        if (rd != 0 && rd != speaker->speaker->dim()) {
            speaker->last_error = "registry holds " + std::to_string(rd) +
                                  "-value embeddings, this model produces " +
                                  std::to_string(speaker->speaker->dim());
            return nullptr;
        }
        pk::SpeakerIdOpts o;
        if (accept_threshold != 0.0f) o.accept_threshold = accept_threshold;
        if (margin != 0.0f) o.margin = margin;
        const std::string oerr = pk::validate_speaker_opts(o);
        if (!oerr.empty()) { speaker->last_error = "invalid speaker options: " + oerr; return nullptr; }

        // Resample once; diarize_pcm and the identifier both take the 16 kHz signal.
        std::vector<float> pcm16k(samples, samples + n_samples);
        if (sample_rate != 16000) pcm16k = pk::resample_linear(pcm16k, sample_rate, 16000);
        pk::DiarizationResult dr;
        try {
            dr = diar_ctx->diar->diarize_pcm(pcm16k, 16000);
        } catch (const std::exception& e) {
            diar_ctx->last_error = e.what();
            return nullptr;
        }
        std::map<int, pk::SpeakerProfile> profiles;
        std::map<int, pk::SlotName> names;
        try {
            names = pk::identify_offline(pcm16k, dr.segments, speaker->speaker->embedder(), registry->reg, o,
                export_profiles ? &profiles : nullptr, speaker->speaker->dim());
        } catch (const std::exception& e) {
            speaker->last_error = e.what();
            return nullptr;
        }
        std::string json = pk::named_profiles_json(diar_result_json_string(dr), names_to_json(names),
            export_profiles ? &profiles : nullptr, speaker->speaker_identity, speaker->speaker->dim());
        diar_ctx->last_error.clear();
        speaker->last_error.clear();
        return dup_to_c(json);
    } catch (const std::exception& e) {
        if (speaker) speaker->last_error = e.what();
    } catch (...) {
        if (speaker) speaker->last_error = "unknown error";
    }
    return nullptr;
}

extern "C" const char* parakeet_capi_speaker_identity(const parakeet_ctx* speaker) {
    return speaker && speaker->speaker ? speaker->speaker_identity.c_str() : nullptr;
}

extern "C" char* parakeet_capi_diarize_named_pcm_json(parakeet_ctx* diar, parakeet_ctx* speaker,
    parakeet_speaker_registry* reg, const float* samples, int n_samples, int sample_rate,
    float accept_threshold, float margin) {
    return diarize_named_pcm_json(false, diar, speaker, reg, samples, n_samples, sample_rate, accept_threshold, margin);
}

extern "C" char* parakeet_capi_diarize_profiles_pcm_json(parakeet_ctx* diar, parakeet_ctx* speaker,
    parakeet_speaker_registry* reg, const float* samples, int n_samples, int sample_rate,
    float accept_threshold, float margin) {
    return diarize_named_pcm_json(true, diar, speaker, reg, samples, n_samples, sample_rate, accept_threshold, margin);
}
