// Scene stream with an ASR model that has no duration head (the unified
// RNN-T checkpoint), at the `model` diarization latency.
//  1. Two speakers, a short pause, then speech.wav: the window after the first
//     commit opens near the end of a word. Without the committer guards the
//     model returned no word for that window and the second half was lost.
//     Every word must come back, in order.
//  2. Word times are acoustic: the median word is longer than one 80 ms frame.
// Needs PARAKEET_TEST_GGUF_UNIFIED and PARAKEET_TEST_DIAR_GGUF.
#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL: " __VA_ARGS__); std::fprintf(stderr, "\n"); ++fails; } } while (0)

static bool load16k(const std::string& path, std::vector<float>& x) {
    pk::Audio a;
    if (!pk::load_audio_16k_mono(path, a)) return false;
    x = std::move(a.samples);
    return true;
}

struct W { std::string text; float start, end; };

// Words of every update ("words" arrays only; their items start with "text",
// the utterance items start with "speaker").
static std::vector<W> words_of(const std::string& jl) {
    std::vector<W> out;
    size_t p = 0;
    while ((p = jl.find("{\"text\":\"", p)) != std::string::npos) {
        p += 9;
        const size_t q = jl.find('"', p);
        W w;
        w.text = jl.substr(p, q - p);
        const size_t s = jl.find("\"start\":", q), e = jl.find("\"end\":", q);
        w.start = std::strtof(jl.c_str() + s + 8, nullptr);
        w.end = std::strtof(jl.c_str() + e + 6, nullptr);
        out.push_back(w);
        p = q;
    }
    return out;
}

static std::string lower_alpha(const std::string& s) {
    std::string o;
    for (unsigned char c : s) if (std::isalpha(c)) o += (char)std::tolower(c);
    return o;
}

int main() {
    const char* a = std::getenv("PARAKEET_TEST_GGUF_UNIFIED");
    const char* d = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    if (!a || !d) { std::fprintf(stderr, "SKIP: needs PARAKEET_TEST_GGUF_UNIFIED and PARAKEET_TEST_DIAR_GGUF\n"); return 77; }
    parakeet_ctx* asr = parakeet_capi_load(a);
    parakeet_ctx* diar = parakeet_capi_load(d);
    if (!asr || !diar) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

    std::vector<float> pcm, x;
    load16k(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", x);
    pcm.insert(pcm.end(), x.begin(), x.end());
    pcm.insert(pcm.end(), 16000 * 6 / 10, 0.0f);
    load16k(std::string(PK_SOURCE_DIR) + "/tests/fixtures/speech.wav", x);
    pcm.insert(pcm.end(), x.begin(), x.end());

    // 1 and 2.
    {
        parakeet_scene_opts o;
        parakeet_capi_scene_opts_default(&o);
        o.diar_latency = PARAKEET_DIAR_LATENCY_MODEL;
        parakeet_scene_stream* s = parakeet_capi_scene_stream_begin(asr, diar, nullptr, &o);
        CHECK(s, "scene begin: %s", parakeet_capi_last_error(asr));
        if (!s) return 1;
        std::string all;
        const int piece = 3200;   // 200 ms, as parakeet-cli scene
        for (size_t i = 0; i < pcm.size(); i += piece) {
            const int n = (int)std::min<size_t>(piece, pcm.size() - i);
            char* j = parakeet_capi_scene_stream_feed_json(s, pcm.data() + i, n, i + piece >= pcm.size());
            CHECK(j, "feed: %s", parakeet_capi_scene_stream_last_error(s));
            if (!j) break;
            all += j; all += "\n";
            parakeet_capi_free_string(j);
        }
        parakeet_capi_scene_stream_free(s);
        const std::vector<W> w = words_of(all);
        std::string text;
        for (const W& x : w) text += lower_alpha(x.text) + " ";
        std::fprintf(stderr, "%zu words: %s\n", w.size(), text.c_str());
        // The reference has 88 words; the unguarded committer gave 50.
        CHECK(w.size() >= 80, "expected at least 80 words, got %zu", w.size());
        // speech.wav (the end of the stream) must be there.
        CHECK(text.find("it is certainly very like the old portrait") != std::string::npos &&
              text.rfind("portrait") > text.size() / 2, "the second half is missing");
        for (size_t i = 1; i < w.size(); ++i)
            CHECK(w[i].start >= w[i - 1].start && w[i].end >= w[i].start, "word %zu out of order", i);
        std::vector<float> dur;
        for (const W& x : w) dur.push_back(x.end - x.start);
        std::sort(dur.begin(), dur.end());
        CHECK(!dur.empty() && dur[dur.size() / 2] > 0.081f, "median word length %.3f s is one frame",
              dur.empty() ? 0.0f : dur[dur.size() / 2]);
    }

    parakeet_capi_free(asr);
    parakeet_capi_free(diar);
    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
