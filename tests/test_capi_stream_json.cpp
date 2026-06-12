#include "parakeet_capi.h"
#include "audio_io.hpp"     // pk::load_audio_16k_mono (test links the parakeet lib)
#include "stream_clips.hpp" // pktest::two_utterance_clip

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Streaming JSON C-API smoke test (segment-timestamp support).
//
// Drives parakeet_capi_stream_feed_json / parakeet_capi_stream_finalize_json on
// the cache-aware EOU streaming model and asserts the returned documents carry
// "frame_sec" (> 0) and a "words" array (the per-word start/end timestamps from
// the streaming session's drain_words) — the data LocalAI needs to build
// timestamped per-utterance segments.
//
// Skips (exit 77) unless PARAKEET_TEST_GGUF_EOU is set (the streaming EOU model
// is a ~480MB download, not in CI).
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    if (!gguf) {
        std::fprintf(stderr,
            "test_capi_stream_json: PARAKEET_TEST_GGUF_EOU not set; skip "
            "(streaming EOU model is a ~480MB download, not in CI)\n");
        return 77;
    }

    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
        std::fprintf(stderr, "test_capi_stream_json: failed to load speech.wav\n");
        return 1;
    }

    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) {
        std::fprintf(stderr, "test_capi_stream_json: load failed for %s\n", gguf);
        return 1;
    }
    parakeet_stream* s = parakeet_capi_stream_begin(ctx);
    if (!s) {
        std::fprintf(stderr, "test_capi_stream_json: stream_begin failed: %s\n",
                     parakeet_capi_last_error(ctx));
        parakeet_capi_free(ctx);
        return 1;
    }

    // Feed the PCM in real-time-sized chunks (~100 ms = 1600 samples).
    const int chunk = 1600;
    const int n = (int)audio.samples.size();
    std::string acc;
    for (int off = 0; off < n; off += chunk) {
        const int len = std::min(chunk, n - off);
        char* t = parakeet_capi_stream_feed_json(s, audio.samples.data() + off, len);
        if (!t) {
            std::fprintf(stderr, "test_capi_stream_json: feed_json NULL: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_stream_free(s);
            parakeet_capi_free(ctx);
            return 1;
        }
        acc += t;
        parakeet_capi_free_string(t);
    }

    char* fin = parakeet_capi_stream_finalize_json(s);
    if (!fin) {
        std::fprintf(stderr, "test_capi_stream_json: finalize_json NULL: %s\n",
                     parakeet_capi_last_error(ctx));
        parakeet_capi_stream_free(s);
        parakeet_capi_free(ctx);
        return 1;
    }
    acc += fin;
    parakeet_capi_free_string(fin);

    parakeet_capi_stream_free(s);

    std::fprintf(stderr, "test_capi_stream_json: concatenated docs:\n%s\n", acc.c_str());

    if (!contains(acc, "\"frame_sec\"")) {
        std::fprintf(stderr, "test_capi_stream_json: FAIL — no frame_sec in output\n");
        parakeet_capi_free(ctx);
        return 1;
    }
    if (!contains(acc, "\"words\"")) {
        std::fprintf(stderr, "test_capi_stream_json: FAIL — no words array in output\n");
        parakeet_capi_free(ctx);
        return 1;
    }
    // A real transcription must finalize at least one word with a non-zero end.
    if (!contains(acc, "\"end\":")) {
        std::fprintf(stderr, "test_capi_stream_json: FAIL — no word end timestamps\n");
        parakeet_capi_free(ctx);
        return 1;
    }
    // ABI v5: every streaming doc carries the (possibly empty) "events" array
    // and the per-type "eou"/"eob" flags.
    if (!contains(acc, "\"events\"")) {
        std::fprintf(stderr, "test_capi_stream_json: FAIL — no events array in output\n");
        parakeet_capi_free(ctx);
        return 1;
    }
    if (!contains(acc, "\"eou\":") || !contains(acc, "\"eob\":")) {
        std::fprintf(stderr, "test_capi_stream_json: FAIL — missing eou/eob flags\n");
        parakeet_capi_free(ctx);
        return 1;
    }

    // Two-utterance clip (see stream_clips.hpp): an <EOU> fires mid-stream and
    // must appear as a typed entry {"type":"eou","frame":...,"t":...} in some doc.
    {
        std::vector<float> clip = pktest::two_utterance_clip(audio.samples);

        parakeet_stream* s2 = parakeet_capi_stream_begin(ctx);
        if (!s2) {
            std::fprintf(stderr, "test_capi_stream_json: phase-2 stream_begin failed: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }
        std::string acc2;
        const int n2 = (int)clip.size();
        for (int off = 0; off < n2; off += chunk) {
            const int len = std::min(chunk, n2 - off);
            char* t = parakeet_capi_stream_feed_json(s2, clip.data() + off, len);
            if (!t) {
                std::fprintf(stderr, "test_capi_stream_json: phase-2 feed_json NULL: %s\n",
                             parakeet_capi_last_error(ctx));
                parakeet_capi_stream_free(s2);
                parakeet_capi_free(ctx);
                return 1;
            }
            acc2 += t;
            parakeet_capi_free_string(t);
        }
        char* fin2 = parakeet_capi_stream_finalize_json(s2);
        if (fin2) { acc2 += fin2; parakeet_capi_free_string(fin2); }
        parakeet_capi_stream_free(s2);

        if (!contains(acc2, "\"type\":\"eou\"")) {
            std::fprintf(stderr,
                "test_capi_stream_json: FAIL — no typed eou event on the "
                "two-utterance clip; docs:\n%s\n", acc2.c_str());
            parakeet_capi_free(ctx);
            return 1;
        }
        // The per-type flag fires with the event (and EOB stays 0 — this clip
        // has no backchannel).
        if (!contains(acc2, "\"eou\":1") || contains(acc2, "\"eob\":1")) {
            std::fprintf(stderr,
                "test_capi_stream_json: FAIL — eou/eob flags wrong on the "
                "two-utterance clip; docs:\n%s\n", acc2.c_str());
            parakeet_capi_free(ctx);
            return 1;
        }
    }

    parakeet_capi_free(ctx);

    std::fprintf(stderr, "test_capi_stream_json: PASS — streaming JSON carries "
                         "frame_sec + per-word timestamps + typed eou/eob events\n");
    return 0;
}
