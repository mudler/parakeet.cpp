#include "parakeet_capi.h"
#include "audio_io.hpp"  // pk::load_audio_16k_mono

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// MODEL: nvidia/parakeet_realtime_eou_120m-v1 (cache-aware streaming, <EOU>).
//
// Speech after a mid-stream <EOU> must still be transcribed. The session resets
// its decoder at an <EOU>; with a decoder-only reset (the StreamingEncoder
// caches kept) some clips decoded the first utterance and then nothing at all
// for the next one, so a realtime caller lost the whole next sentence. NeMo's
// reset_state also starts the encoder from fresh caches.
//
// The clip is PARAKEET_TEST_WAV_EOU_CONTINUE (16 kHz mono): two utterances with
// a pause between them, long enough for an <EOU> to fire in-stream. It is fed
// in 300 ms pieces, as a realtime caller does. The test asserts that an <EOU>
// fired and that at least PARAKEET_TEST_EOU_CONTINUE_MIN_WORDS words (default
// 3) were finalized after it.
//
// Skips (77) unless PARAKEET_TEST_GGUF_EOU and PARAKEET_TEST_WAV_EOU_CONTINUE
// are set.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* wav = std::getenv("PARAKEET_TEST_WAV_EOU_CONTINUE");
    if (!gguf || !wav) {
        std::fprintf(stderr,
            "test_streaming_eou_continue: PARAKEET_TEST_GGUF_EOU and/or "
            "PARAKEET_TEST_WAV_EOU_CONTINUE not set; skip\n");
        return 77;
    }
    int min_words = 3;
    if (const char* m = std::getenv("PARAKEET_TEST_EOU_CONTINUE_MIN_WORDS")) min_words = std::atoi(m);

    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio) || audio.samples.empty()) {
        std::fprintf(stderr, "[eou_continue] cannot read %s\n", wav);
        return 1;
    }
    const std::vector<float>& pcm = audio.samples;
    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) {
        std::fprintf(stderr, "[eou_continue] load failed %s\n", gguf);
        return 1;
    }
    parakeet_stream* s = parakeet_capi_stream_begin(ctx);
    if (!s) {
        std::fprintf(stderr, "[eou_continue] stream_begin: %s\n", parakeet_capi_last_error(ctx));
        parakeet_capi_free(ctx);
        return 1;
    }

    const int piece = 4800;  // 300 ms
    bool eou_seen = false;
    std::string before, after;
    for (size_t off = 0; off < pcm.size(); off += piece) {
        const int n = (int)std::min(pcm.size() - off, (size_t)piece);
        int events = 0;
        char* text = parakeet_capi_stream_feed(s, pcm.data() + off, n, &events);
        if (!text) {
            std::fprintf(stderr, "[eou_continue] feed: %s\n", parakeet_capi_last_error(ctx));
            return 1;
        }
        (eou_seen ? after : before) += text;
        parakeet_capi_free_string(text);
        if (events & PARAKEET_EVENT_EOU) eou_seen = true;
    }
    char* tail = parakeet_capi_stream_finalize(s);
    if (tail) {
        (eou_seen ? after : before) += tail;
        parakeet_capi_free_string(tail);
    }
    parakeet_capi_stream_free(s);
    parakeet_capi_free(ctx);

    std::istringstream words(after);
    int n_after = 0;
    for (std::string w; words >> w;) ++n_after;
    std::fprintf(stderr, "[eou_continue] before <EOU>: \"%s\"\n", before.c_str());
    std::fprintf(stderr, "[eou_continue] after  <EOU>: \"%s\" (%d words)\n", after.c_str(), n_after);
    if (!eou_seen) {
        std::fprintf(stderr, "[eou_continue] FAIL: no <EOU> fired in-stream; the clip does not "
                             "exercise the reset (use a longer pause)\n");
        return 1;
    }
    if (n_after < min_words) {
        std::fprintf(stderr, "[eou_continue] FAIL: %d words after the <EOU>, want >= %d\n",
                     n_after, min_words);
        return 1;
    }
    std::fprintf(stderr, "[eou_continue] PASS\n");
    return 0;
}
