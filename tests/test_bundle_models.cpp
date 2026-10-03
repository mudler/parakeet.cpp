// A real bundle against its single-model files: the transcript, the VAD head
// probabilities and the Silero probabilities are identical, through the C++ API
// and through the C-API, and loading one component does not read the others.
//
// LABEL model; run from the project root (fixtures are relative).
// Env (skip 77 without the first three):
//   PARAKEET_TEST_BUNDLE         the bundle GGUF (an "asr" and a "vad" component)
//   PARAKEET_TEST_BUNDLE_ASR     the single-model ASR GGUF the "asr" component was built from
//   PARAKEET_TEST_BUNDLE_SILERO  the Silero GGUF the "vad" component was built from
//   PARAKEET_TEST_BUNDLE_EXPECT  optional: the exact transcript of tests/fixtures/speech.wav
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "audio_io.hpp"
#include "bundle.hpp"
#include "model.hpp"
#include "parakeet_capi.h"
#include "silero_vad.hpp"
#include "vad_json.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static unsigned long long bytes_read() {
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (!f) return 0;
    char k[64];
    unsigned long long v = 0, r = 0;
    while (std::fscanf(f, "%63s %llu", k, &v) == 2)
        if (std::string(k) == "rchar:") r = v;
    std::fclose(f);
    return r;
}

static std::string take(char* p) {
    std::string s = p ? p : "";
    if (p) parakeet_capi_free_string(p);
    return s;
}

static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main() {
    const char* bundle = std::getenv("PARAKEET_TEST_BUNDLE");
    const char* asr_path = std::getenv("PARAKEET_TEST_BUNDLE_ASR");
    const char* sil_path = std::getenv("PARAKEET_TEST_BUNDLE_SILERO");
    if (!bundle || !asr_path || !sil_path) {
        std::puts("skip: PARAKEET_TEST_BUNDLE, _ASR and _SILERO must be set");
        return 77;
    }
    const char* expect = std::getenv("PARAKEET_TEST_BUNDLE_EXPECT");
    const std::string wav = "tests/fixtures/speech.wav";
    pk::Audio audio;
    CHECK(pk::load_audio_16k_mono(wav, audio));

    pk::BundleInfo info;
    std::string err;
    CHECK(pk::read_bundle_info(bundle, info, &err));
    const pk::BundleComponent* ac = info.find("asr");
    const pk::BundleComponent* vc = info.find("vad");
    CHECK(ac && vc);
    if (!ac || !vc) return 1;
    CHECK(!ac->license.empty() && !vc->license.empty() && ac->license != vc->license);
    CHECK(vc->license == "MIT");
    struct stat st{};
    CHECK(::stat(bundle, &st) == 0);

    // ---- ASR: transcript identical to the single-model file ----
    auto solo = pk::Model::load(asr_path);
    CHECK(solo != nullptr);
    CHECK(!pk::Model::load(bundle));   // a bundle without a component is refused
    const unsigned long long r0 = bytes_read();
    auto bun = pk::Model::load(bundle, "asr");
    const unsigned long long r1 = bytes_read();
    CHECK(bun != nullptr);
    if (!solo || !bun) return 1;
    const std::string t0 = solo->transcribe_path(wav), t1 = bun->transcribe_path(wav);
    std::printf("standalone: %s\nbundle    : %s\n", t0.c_str(), t1.c_str());
    CHECK(!t0.empty() && t0 == t1);
    if (expect) CHECK(t1 == expect);
    // Only the asr component's tensors are read (plus the header).
    if (r1 > r0) {
        std::printf("bytes read for asr: %.1f MB (component %.1f MB, file %.1f MB)\n", (double)(r1 - r0) / 1e6,
                    (double)ac->n_bytes / 1e6, (double)st.st_size / 1e6);
        CHECK(r1 - r0 < ac->n_bytes + (8u << 20));
        CHECK(r1 - r0 < (unsigned long long)st.st_size - vc->n_bytes / 2);
    }

    // ---- VAD head (Ultra/Redux) from the bundle's ASR component ----
    if (solo->config().vad.present) {
        CHECK(bun->config().vad.present && bun->config().vad.frame_sec == solo->config().vad.frame_sec);
        const std::vector<float> p0 = solo->vad_probabilities(audio.samples), p1 = bun->vad_probabilities(audio.samples);
        CHECK(!p0.empty() && same_bits(p0, p1));
        pk::VadRequest req;
        req.opts = pk::default_segmenter_opts(pk::VadKind::kHead);
        req.probabilities = true;
        CHECK(pk::vad_to_json(*solo, audio.samples, req) == pk::vad_to_json(*bun, audio.samples, req));
    } else {
        std::puts("note: the ASR component has no VAD head");
    }

    // ---- Silero from the bundle ----
    std::string e0, e1;
    auto s0 = pk::SileroVad::load(sil_path, &e0);
    const unsigned long long q0 = bytes_read();
    auto s1 = pk::SileroVad::load(bundle, &e1, "vad");
    const unsigned long long q1 = bytes_read();
    CHECK(s0 && s1);
    if (s0 && s1) {
        const std::vector<float> a = s0->probabilities(audio.samples.data(), audio.samples.size(), 16000);
        const std::vector<float> b = s1->probabilities(audio.samples.data(), audio.samples.size(), 16000);
        CHECK(!a.empty() && same_bits(a, b));
        pk::VadRequest req;
        req.kind = pk::VadKind::kSilero;
        req.opts = pk::default_segmenter_opts(pk::VadKind::kSilero);
        req.probabilities = true;
        CHECK(pk::silero_vad_to_json(*s0, audio.samples, 16000, req) == pk::silero_vad_to_json(*s1, audio.samples, 16000, req));
        // ASR transcription cut by the bundle's Silero equals the one cut by the single file's Silero.
        pk::SegmenterOpts opts = pk::default_segmenter_opts(pk::VadKind::kSilero);
        auto fn = [](const pk::SileroVad* sv) {
            return pk::Model::VadProbabilityFn([sv](const std::vector<float>& pcm) {
                return sv->probabilities(pcm.data(), pcm.size(), 16000);
            });
        };
        pk::Model::VadProbabilityFn f0 = fn(s0.get()), f1 = fn(s1.get());
        const std::string v0 = solo->transcribe_pcm_vad(audio.samples, audio.sample_rate, pk::Decoder::kDefault, "", opts, &f0);
        const std::string v1 = bun->transcribe_pcm_vad(audio.samples, audio.sample_rate, pk::Decoder::kDefault, "", opts, &f1);
        CHECK(!v0.empty() && v0 == v1);
        if (q1 > q0) {
            std::printf("bytes read for vad: %.2f MB (component %.2f MB)\n", (double)(q1 - q0) / 1e6, (double)vc->n_bytes / 1e6);
            CHECK(q1 - q0 < vc->n_bytes + (4u << 20));   // far below the ASR component
        }
    }

    // ---- C-API ----
    {
        parakeet_ctx* a0 = parakeet_capi_load(asr_path);
        parakeet_ctx* a1 = parakeet_capi_load(bundle);               // auto-selects the ASR component
        parakeet_ctx* a2 = parakeet_capi_load_component(bundle, "asr");
        parakeet_ctx* v0 = parakeet_capi_load(sil_path);
        parakeet_ctx* v1 = parakeet_capi_load_component(bundle, "vad");
        CHECK(a0 && a1 && a2 && v0 && v1);
        if (a0 && a1 && a2 && v0 && v1) {
            CHECK(parakeet_capi_model_kind(a1) == PARAKEET_MODEL_KIND_ASR);
            CHECK(parakeet_capi_model_kind(v1) == PARAKEET_MODEL_KIND_VAD);
            const std::string c0 = take(parakeet_capi_transcribe_path(a0, wav.c_str(), 0));
            const std::string c1 = take(parakeet_capi_transcribe_path(a1, wav.c_str(), 0));
            const std::string c2 = take(parakeet_capi_transcribe_path(a2, wav.c_str(), 0));
            CHECK(!c0.empty() && c0 == c1 && c0 == c2 && c0 == t0);
            const char* opt = "{\"probabilities\":true}";
            CHECK(take(parakeet_capi_vad_path_json(v0, wav.c_str(), opt)) == take(parakeet_capi_vad_path_json(v1, wav.c_str(), opt)));
            if (solo->config().vad.present)
                CHECK(take(parakeet_capi_vad_path_json(a0, wav.c_str(), opt)) == take(parakeet_capi_vad_path_json(a1, wav.c_str(), opt)));
            // segmented transcription cut by the bundle's Silero
            const std::string j0 = take(parakeet_capi_transcribe_path_json_vad_with(a0, v0, wav.c_str(), 0, nullptr));
            const std::string j1 = take(parakeet_capi_transcribe_path_json_vad_with(a1, v1, wav.c_str(), 0, nullptr));
            CHECK(!j0.empty() && j0 == j1);
        }
        parakeet_capi_free(a0); parakeet_capi_free(a1); parakeet_capi_free(a2);
        parakeet_capi_free(v0); parakeet_capi_free(v1);
        const std::string js = take(parakeet_capi_bundle_components_json(bundle));
        CHECK(js.find("\"name\":\"asr\"") != std::string::npos && js.find("\"name\":\"vad\"") != std::string::npos);
    }

    if (failures) return 1;
    std::printf("test_bundle_models OK\n");
    return 0;
}
