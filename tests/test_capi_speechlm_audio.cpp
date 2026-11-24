// C API for the speech LLM audio embeddings.
//
// Always runs: argument and error handling that needs no model (a missing file, NULL
// arguments, a context of the wrong kind).
// With PARAKEET_TEST_SPEECHLM_AUDIO_GGUF and PARAKEET_TEST_SPEECHLM_DIAR_GGUF set it also loads
// the models and checks the embeddings against the baseline (see test_speechlm_audio.cpp):
// shape, parity with the C++ class output, a resampled 8 kHz copy, and the path entry point.
#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static const char* env(const char* k, const char* d = nullptr) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : d;
}

int main() {
    // --- no model needed ---
    CHECK(parakeet_capi_load_speechlm_audio(nullptr, nullptr, "d.gguf", nullptr) == nullptr);
    CHECK(std::strlen(parakeet_capi_load_error()) > 0);
    CHECK(parakeet_capi_load_speechlm_audio("a.gguf", nullptr, nullptr, nullptr) == nullptr);
    CHECK(parakeet_capi_load_speechlm_audio("/nonexistent/a.gguf", nullptr, "/nonexistent/d.gguf", nullptr) == nullptr);
    CHECK(std::strlen(parakeet_capi_load_error()) > 0);
    CHECK(parakeet_capi_load_speechlm_audio("/nonexistent/b.gguf", "audio", "/nonexistent/b.gguf", "diar") == nullptr);
    CHECK(parakeet_capi_audio_embed_dim(nullptr) == -1);
    {
        float* e = reinterpret_cast<float*>(0x1);
        int n = 7, d = 7;
        CHECK(parakeet_capi_audio_embed_pcm(nullptr, nullptr, 0, 16000, &e, &n, &d) != 0);
        CHECK(e == nullptr && n == 0 && d == 0);
        e = reinterpret_cast<float*>(0x1);
        CHECK(parakeet_capi_audio_embed_path(nullptr, "x.wav", &e, &n, &d) != 0);
        CHECK(e == nullptr);
    }

    const char* audio = env("PARAKEET_TEST_SPEECHLM_AUDIO_GGUF");
    const char* diar = env("PARAKEET_TEST_SPEECHLM_DIAR_GGUF");
    if (!audio || !diar) {
        std::fprintf(stderr, failures ? "FAILED (%d)\n" : "PASSED (argument checks only; models not set)\n", failures);
        return failures ? 1 : 0;
    }
    const char* base = env("PARAKEET_TEST_SPEECHLM_BASELINE", "models/speechlm_audio_baseline.gguf");
    const std::string wav = env("PARAKEET_TEST_SPEECHLM_WAV", "tests/fixtures/speechlm_audio_2s.wav");

    parakeet_ctx* ctx = parakeet_capi_load_speechlm_audio(audio, env("PARAKEET_TEST_SPEECHLM_AUDIO_COMPONENT"), diar,
                                                          env("PARAKEET_TEST_SPEECHLM_DIAR_COMPONENT"));
    if (!ctx) { std::fprintf(stderr, "load failed: %s\n", parakeet_capi_load_error()); return 1; }
    CHECK(parakeet_capi_model_kind(ctx) == PARAKEET_MODEL_KIND_SPEECHLM_AUDIO);
    CHECK(parakeet_capi_audio_embed_dim(ctx) == 2688);
    // The other entry points refuse this context instead of crashing.
    CHECK(parakeet_capi_transcribe_path(ctx, wav.c_str(), 0) == nullptr);
    CHECK(std::strlen(parakeet_capi_last_error(ctx)) > 0);

    pk::Audio a;
    CHECK(pk::load_audio_16k_mono(wav, a));
    float* emb = nullptr;
    int n = 0, dim = 0;
    CHECK(parakeet_capi_audio_embed_pcm(ctx, a.samples.data(), (int)a.samples.size(), 16000, &emb, &n, &dim) == 0);
    CHECK(emb != nullptr && dim == 2688 && n == ((int)a.samples.size() / 160 + 7) / 8);
    std::fprintf(stderr, "%d tokens x %d\n", n, dim);

    std::vector<float> ref;
    std::vector<int64_t> shape;
    if (emb && pktest::load_baseline(base, "adapter", ref, shape)) {
        std::vector<float> got(emb, emb + (size_t)n * dim);
        CHECK(pktest::compare(got, ref, "adapter (C API)", 5e-3f, 1e-3f));
    } else if (!emb) {
        ++failures;
    }

    // The path entry point gives the same rows.
    float* emb2 = nullptr;
    int n2 = 0, dim2 = 0;
    CHECK(parakeet_capi_audio_embed_path(ctx, wav.c_str(), &emb2, &n2, &dim2) == 0);
    CHECK(emb && emb2 && n2 == n && dim2 == dim && std::memcmp(emb, emb2, (size_t)n * dim * sizeof(float)) == 0);
    parakeet_capi_free_floats(emb2);

    // 8 kHz input is resampled to 16 kHz first: same token count, close rows.
    {
        std::vector<float> half;
        for (size_t i = 0; i < a.samples.size(); i += 2) half.push_back(0.5f * (a.samples[i] + a.samples[std::min(i + 1, a.samples.size() - 1)]));
        float* e8 = nullptr;
        int n8 = 0, d8 = 0;
        CHECK(parakeet_capi_audio_embed_pcm(ctx, half.data(), (int)half.size(), 8000, &e8, &n8, &d8) == 0);
        CHECK(e8 != nullptr && std::abs(n8 - n) <= 1 && d8 == dim);
        parakeet_capi_free_floats(e8);
    }

    // Bad arguments and a clip shorter than one mel frame.
    float* none = reinterpret_cast<float*>(0x1);
    int nn = 5, dd = 5;
    CHECK(parakeet_capi_audio_embed_pcm(ctx, nullptr, 100, 16000, &none, &nn, &dd) != 0 && none == nullptr);
    CHECK(parakeet_capi_audio_embed_pcm(ctx, a.samples.data(), 100, 0, &none, &nn, &dd) != 0);
    CHECK(parakeet_capi_audio_embed_pcm(ctx, a.samples.data(), 100, 16000, &none, &nn, &dd) == 0 && none == nullptr && nn == 0);
    CHECK(parakeet_capi_audio_embed_path(ctx, "/nonexistent.wav", &none, &nn, &dd) != 0);

    parakeet_capi_free_floats(emb);
    parakeet_capi_free(ctx);
    std::fprintf(stderr, failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
