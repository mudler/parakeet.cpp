#include "parakeet_capi.h"
#include "parakeet_capi_test.hpp"  // per-stream mel-window diagnostics
#include "model.hpp"        // pk::Model
#include "streaming.hpp"    // pk::StreamingSession, pk::run_stream_over_pcm
#include "audio_io.hpp"     // pk::load_audio_16k_mono (test links the parakeet lib)
#include "stream_clips.hpp" // pktest::repeated_session_clip

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// MODEL: any cache-aware streaming model (e.g. nemotron-3.5-asr-streaming-0.6b
// or parakeet_realtime_eou_120m-v1).
// WORKING_DIRECTORY: the repo root (build/tests run from there).
//
// Regression for the SECOND O(N^2) long-session runaway, the one
// test_streaming_longrun_bounded does not see. That test drives
// pk::StreamingSession directly; this one drives the streaming C-API, which
// owns the accumulated mel buffer the live server feeds PCM into.
//
// The bug: parakeet_capi_stream_feed appended the newly-computed mel frames by
// allocating a whole new [n_mels, T] buffer and copying the ENTIRE history into
// it, every chunk (the feat-major row stride changes when T grows, so a naive
// append cannot be in place). Worse, the buffer was never pruned even though the
// chunk schedule only ever reads back as far as pre_encode_cache_size frames. So
// per-chunk work and RSS both grew with the whole session: measured on the live
// container, 12.6 ms/chunk at 2 min of audio and 23.2 ms/chunk at 10 min
// (ms ~ 10.0 + 0.0222 * audio_s), which crosses real time at ~1 h and never
// recovers -- the dictation silently dies with a core pinned at 99 %.
//
// We drive a long multi-utterance stream through the PUBLIC C-API in
// realtime-sized PCM chunks and assert:
//   (a) BOUNDED per-chunk cost: the hottest single append of an 8x longer
//       session copies no more floats than the short one (it grew ~8x before).
//   (b) BOUNDED memory: the retained mel window does not grow with the session.
//   (c) PARITY: the transcript is byte-identical to the non-windowed reference
//       path (pk::run_stream_over_pcm over the full-clip mel), so pruning has
//       not eaten any left context the encoder still needed.
//
// Skips (77) unless PARAKEET_TEST_GGUF_STREAM is set (the streaming model is a
// large download, not in CI). Optional:
//   PARAKEET_TEST_STREAM_WAV     clip to repeat (default tests/fixtures/speech.wav)
//   PARAKEET_TEST_STREAM_LANG    language prompt for prompt models (default "")
//   PARAKEET_TEST_STREAM_REPEATS repeats in the LONG session (default 24)

namespace {

// Take ownership of a char* the C-API malloc'd, as a std::string.
std::string take(char* p) {
    if (!p) return std::string();
    std::string s(p);
    parakeet_capi_free_string(p);
    return s;
}

struct RunStats {
    std::string text;
    unsigned long long max_copy = 0;
    unsigned long long max_held = 0;
    bool ok = false;
};

// Feed `pcm` through the C-API in 80 ms chunks (what the live server sends) and
// return the concatenated transcript plus the mel-window high-water marks.
RunStats run_capi(parakeet_ctx* ctx, const std::vector<float>& pcm,
                  const std::string& lang) {
    RunStats r;
    parakeet_stream* st = parakeet_capi_stream_begin_lang(ctx, lang.c_str());
    if (!st) {
        std::fprintf(stderr, "[melbound] stream_begin failed: %s\n",
                     parakeet_capi_last_error(ctx));
        return r;
    }

    const int chunk = 16000 * 80 / 1000;
    std::string text;
    for (size_t off = 0; off + (size_t)chunk <= pcm.size(); off += (size_t)chunk) {
        int eou = 0;
        char* d = parakeet_capi_stream_feed(st, pcm.data() + off, chunk, &eou);
        if (!d) {
            std::fprintf(stderr, "[melbound] feed failed at %zu: %s\n", off,
                         parakeet_capi_last_error(ctx));
            parakeet_capi_stream_free(st);
            return r;
        }
        text += take(d);
    }
    char* tail = parakeet_capi_stream_finalize(st);
    if (!tail) {
        std::fprintf(stderr, "[melbound] finalize failed: %s\n",
                     parakeet_capi_last_error(ctx));
        parakeet_capi_stream_free(st);
        return r;
    }
    text += take(tail);

    r.text     = text;
    r.max_copy = parakeet_capi_test_mel_max_copy(st);
    r.max_held = parakeet_capi_test_mel_max_frames_held(st);
    r.ok       = true;
    parakeet_capi_stream_free(st);
    return r;
}

} // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_STREAM");
    if (!gguf) {
        std::fprintf(stderr,
            "test_streaming_mel_bounded: PARAKEET_TEST_GGUF_STREAM not set; "
            "skip (streaming model is a large download, not in CI)\n");
        return 77;
    }
    const char* wav  = std::getenv("PARAKEET_TEST_STREAM_WAV");
    const char* lang = std::getenv("PARAKEET_TEST_STREAM_LANG");
    const char* rep  = std::getenv("PARAKEET_TEST_STREAM_REPEATS");
    const std::string wav_path    = wav ? wav : "tests/fixtures/speech.wav";
    const std::string target_lang = lang ? lang : "";
    int long_repeats = rep ? std::atoi(rep) : 24;
    if (long_repeats < 8) long_repeats = 8;
    const int short_repeats = long_repeats / 8;

    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav_path, a)) {
        std::fprintf(stderr, "[melbound] audio load failed %s\n", wav_path.c_str());
        return 1;
    }

    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) { std::fprintf(stderr, "[melbound] load failed %s\n", gguf); return 1; }

    const std::vector<float> pcm_short = pktest::repeated_session_clip(a.samples, short_repeats);
    const std::vector<float> pcm_long  = pktest::repeated_session_clip(a.samples, long_repeats);

    const RunStats s_short = run_capi(ctx, pcm_short, target_lang);
    const RunStats s_long  = run_capi(ctx, pcm_long,  target_lang);
    if (!s_short.ok || !s_long.ok) { parakeet_capi_free(ctx); return 1; }

    std::fprintf(stderr,
        "[melbound] short: %.0f s audio  max_copy=%llu floats  max_held=%llu frames\n"
        "[melbound] long : %.0f s audio  max_copy=%llu floats  max_held=%llu frames\n",
        pcm_short.size() / 16000.0, s_short.max_copy, s_short.max_held,
        pcm_long.size()  / 16000.0, s_long.max_copy,  s_long.max_held);

    // (a) BOUNDED per-chunk cost. The pre-fix append copied the whole history, so
    // the 8x longer session's hottest append copied ~8x more floats. A windowed
    // append copies its own new frames plus, now and then, the small live
    // window -- neither scales with the session, so allow only slack for the
    // capacity doubling that may land differently in the two runs.
    if (s_long.max_copy > s_short.max_copy * 2) {
        std::fprintf(stderr,
            "[melbound] FAIL (a): hottest append grew with the session "
            "(%llu -> %llu floats for 8x the audio)\n",
            s_short.max_copy, s_long.max_copy);
        parakeet_capi_free(ctx);
        return 1;
    }

    // (b) BOUNDED memory. The pre-fix buffer retained every frame ever seen
    // (~184 MB/h); the window retains only what the chunk schedule can reread.
    if (s_long.max_held > s_short.max_held * 2) {
        std::fprintf(stderr,
            "[melbound] FAIL (b): retained mel grew with the session "
            "(%llu -> %llu frames for 8x the audio)\n",
            s_short.max_held, s_long.max_held);
        parakeet_capi_free(ctx);
        return 1;
    }

    // (c) PARITY against the non-windowed reference path: pk::run_stream_over_pcm
    // computes the mel over the WHOLE clip once and feeds the identical chunk
    // schedule, with no window and no pruning. Byte-identical output is what
    // proves the pruning kept every frame the encoder's left context needs --
    // the real risk of this change is an off-by-one there.
    auto m = pk::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "[melbound] model load failed\n"); parakeet_capi_free(ctx); return 1; }
    pk::StreamingSession ref(m->loader(), target_lang);
    pk::run_stream_over_pcm(ref, m->loader(), pcm_long);
    if (s_long.text != ref.text()) {
        std::fprintf(stderr,
            "[melbound] FAIL (c): C-API transcript != full-mel reference\n"
            "  capi(%zu): %.160s\n  ref (%zu): %.160s\n",
            s_long.text.size(), s_long.text.c_str(),
            ref.text().size(), ref.text().c_str());
        parakeet_capi_free(ctx);
        return 1;
    }

    std::fprintf(stderr,
        "[melbound] PASS: per-chunk copy and retained mel bounded; transcript "
        "(%zu B) byte-identical to the full-mel reference\n", s_long.text.size());
    parakeet_capi_free(ctx);
    return 0;
}
