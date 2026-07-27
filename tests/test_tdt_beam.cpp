#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"
#include "parakeet_capi.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct BackendShutdown {
    ~BackendShutdown() { pk::shutdown_backend(); }
};

} // namespace

static bool read_i32_tensor(ggml_context* ctx, const char* name,
                            std::vector<int32_t>& out) {
    ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor || tensor->type != GGML_TYPE_I32)
        return false;
    const int n = (int)ggml_nelements(tensor);
    const int32_t* data = static_cast<const int32_t*>(tensor->data);
    out.assign(data, data + n);
    return true;
}

static bool read_f32_tensor(ggml_context* ctx, const char* name,
                            std::vector<float>& out) {
    ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor || tensor->type != GGML_TYPE_F32)
        return false;
    const int n = (int)ggml_nelements(tensor);
    const float* data = static_cast<const float*>(tensor->data);
    out.assign(data, data + n);
    return true;
}

static bool compare_nemo_baseline(
        const char* path, int beam_size,
        const std::vector<pk::NBestTranscription>& got) {
    ggml_context* ctx = nullptr;
    gguf_init_params params{/*no_alloc=*/false, &ctx};
    gguf_context* gguf = gguf_init_from_file(path, params);
    if (!gguf) {
        std::fprintf(stderr,
                     "test_tdt_beam: failed to open baseline %s\n", path);
        return false;
    }

    const int64_t beam_key =
        gguf_find_key(gguf, "baseline.tdt_nbest_beam_size");
    const int64_t count_key =
        gguf_find_key(gguf, "baseline.tdt_nbest_count");
    if (beam_key < 0 || count_key < 0) {
        std::fprintf(stderr,
                     "test_tdt_beam: baseline has no TDT N-best data\n");
        gguf_free(gguf);
        ggml_free(ctx);
        return false;
    }
    const int reference_beam = (int)gguf_get_val_u32(gguf, beam_key);
    const int reference_count = (int)gguf_get_val_u32(gguf, count_key);

    std::vector<int32_t> offsets;
    std::vector<int32_t> ids;
    std::vector<int32_t> end_frames;
    std::vector<float> scores;
    const bool tensors_ok =
        read_i32_tensor(ctx, "tdt_nbest_offsets", offsets) &&
        read_i32_tensor(ctx, "tdt_nbest_token_ids", ids) &&
        read_i32_tensor(ctx, "tdt_nbest_token_end_frames", end_frames) &&
        read_f32_tensor(ctx, "tdt_nbest_scores", scores);
    gguf_free(gguf);
    ggml_free(ctx);
    if (!tensors_ok || reference_beam != beam_size ||
        reference_count != (int)got.size() ||
        offsets.size() != got.size() + 1 ||
        scores.size() != got.size() ||
        ids.size() != end_frames.size()) {
        std::fprintf(stderr,
                     "test_tdt_beam: invalid N-best baseline shape\n");
        return false;
    }

    for (size_t rank = 0; rank < got.size(); ++rank) {
        const int begin = offsets[rank];
        const int end = offsets[rank + 1];
        if (begin < 0 || end < begin || end > (int)ids.size() ||
            end - begin != (int)got[rank].tokens.size()) {
            std::fprintf(stderr,
                         "test_tdt_beam: rank %zu token count mismatch\n",
                         rank);
            return false;
        }
        if (std::fabs(scores[rank] - got[rank].score) > 5e-3f) {
            std::fprintf(stderr,
                         "test_tdt_beam: rank %zu score mismatch "
                         "got=%.6f ref=%.6f\n",
                         rank, got[rank].score, scores[rank]);
            return false;
        }
        for (int i = begin; i < end; ++i) {
            const pk::TdtBeamToken& token =
                got[rank].tokens[(size_t)(i - begin)];
            if (token.id != ids[i] ||
                token.frame + token.duration != end_frames[i]) {
                std::fprintf(stderr,
                             "test_tdt_beam: rank %zu token %d mismatch "
                             "got=(%d,%d) ref=(%d,%d)\n",
                             rank, i - begin, token.id,
                             token.frame + token.duration,
                             ids[i], end_frames[i]);
                return false;
            }
        }
    }
    return true;
}

static bool same_hypothesis(const pk::NBestTranscription& a,
                            const pk::NBestTranscription& b) {
    if (a.text != b.text || a.tokens.size() != b.tokens.size())
        return false;
    if (std::fabs(a.score - b.score) > 1e-6f ||
        std::fabs(a.normalized_score - b.normalized_score) > 1e-6f)
        return false;
    for (size_t i = 0; i < a.tokens.size(); ++i) {
        if (a.tokens[i].id != b.tokens[i].id ||
            a.tokens[i].frame != b.tokens[i].frame ||
            a.tokens[i].duration != b.tokens[i].duration)
            return false;
    }
    return true;
}

int main() {
    BackendShutdown backend_shutdown;
    const char* model_path = std::getenv("PARAKEET_TEST_GGUF");
    if (!model_path) {
        std::fprintf(stderr,
                     "test_tdt_beam: PARAKEET_TEST_GGUF not set; skip\n");
        return 77;
    }

    std::unique_ptr<pk::Model> model = pk::Model::load(model_path);
    if (!model) {
        std::fprintf(stderr, "test_tdt_beam: failed to load model\n");
        return 1;
    }
    const pk::ParakeetConfig& cfg = model->config();
    if (cfg.tdt_durations.empty()) {
        std::fprintf(stderr, "test_tdt_beam: model is not TDT; skip\n");
        return 77;
    }

    const int beam_size = 4;
    std::vector<pk::NBestTranscription> first =
        model->transcribe_path_nbest(
            "tests/fixtures/speech.wav", beam_size, beam_size, true);
    if (first.empty() || first.size() > (size_t)beam_size) {
        std::fprintf(stderr,
                     "test_tdt_beam: invalid hypothesis count %zu\n",
                     first.size());
        return 1;
    }

    for (size_t i = 0; i < first.size(); ++i) {
        const pk::NBestTranscription& hyp = first[i];
        const float expected_norm =
            hyp.score / (float)(hyp.tokens.size() + 1);
        if (!std::isfinite(hyp.score) ||
            std::fabs(hyp.normalized_score - expected_norm) > 1e-6f) {
            std::fprintf(stderr,
                         "test_tdt_beam: invalid score at rank %zu\n", i);
            return 1;
        }
        if (i > 0 &&
            first[i - 1].normalized_score < hyp.normalized_score) {
            std::fprintf(stderr,
                         "test_tdt_beam: hypotheses are not score-sorted\n");
            return 1;
        }
        int previous_frame = -1;
        for (const pk::TdtBeamToken& token : hyp.tokens) {
            if (token.id < 0 || token.id >= (int)cfg.blank_id ||
                token.frame < previous_frame ||
                std::find(cfg.tdt_durations.begin(), cfg.tdt_durations.end(),
                          token.duration) == cfg.tdt_durations.end()) {
                std::fprintf(stderr,
                             "test_tdt_beam: invalid token metadata\n");
                return 1;
            }
            previous_frame = token.frame;
        }
    }

    std::vector<pk::NBestTranscription> second =
        model->transcribe_path_nbest(
            "tests/fixtures/speech.wav", beam_size, beam_size, true);
    if (first.size() != second.size()) {
        std::fprintf(stderr,
                     "test_tdt_beam: nondeterministic hypothesis count\n");
        return 1;
    }
    for (size_t i = 0; i < first.size(); ++i) {
        if (!same_hypothesis(first[i], second[i])) {
            std::fprintf(stderr,
                         "test_tdt_beam: nondeterministic rank %zu\n", i);
            return 1;
        }
    }

    std::vector<pk::NBestTranscription> raw_ranked =
        model->transcribe_path_nbest(
            "tests/fixtures/speech.wav", beam_size, 2, false);
    if (raw_ranked.empty() || raw_ranked.size() > 2) {
        std::fprintf(stderr,
                     "test_tdt_beam: invalid raw-ranked hypothesis count\n");
        return 1;
    }
    for (size_t i = 1; i < raw_ranked.size(); ++i) {
        if (raw_ranked[i - 1].score < raw_ranked[i].score) {
            std::fprintf(stderr,
                         "test_tdt_beam: hypotheses are not raw-score sorted\n");
            return 1;
        }
    }

    try {
        (void)model->transcribe_path_nbest(
            "tests/fixtures/does-not-exist.wav", 0, 1, true);
        std::fprintf(stderr,
                     "test_tdt_beam: invalid beam request unexpectedly succeeded\n");
        return 1;
    } catch (const std::invalid_argument& e) {
        if (!std::strstr(e.what(), "beam_size")) {
            std::fprintf(stderr,
                         "test_tdt_beam: unexpected validation error: %s\n",
                         e.what());
            return 1;
        }
    }

    const char* baseline_path =
        std::getenv("PARAKEET_TEST_TDT_NBEST_BASELINE");
    if (baseline_path &&
        !compare_nemo_baseline(baseline_path, beam_size, first)) {
        return 1;
    }

    model.reset();
    parakeet_ctx* ctx = parakeet_capi_load(model_path);
    if (!ctx) {
        std::fprintf(stderr, "test_tdt_beam: C API load failed\n");
        return 1;
    }
    char* json = parakeet_capi_transcribe_path_nbest_json(
        ctx, "tests/fixtures/speech.wav",
        beam_size, beam_size, 1, nullptr);
    if (!json || !std::strstr(json, "\"hypotheses\":[") ||
        !std::strstr(json, "\"normalized_score\":")) {
        std::fprintf(stderr, "test_tdt_beam: invalid C API JSON: %s\n",
                     json ? json : parakeet_capi_last_error(ctx));
        parakeet_capi_free_string(json);
        parakeet_capi_free(ctx);
        return 1;
    }
    parakeet_capi_free_string(json);

    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
        std::fprintf(stderr, "test_tdt_beam: failed to load PCM fixture\n");
        parakeet_capi_free(ctx);
        return 1;
    }
    char* pcm_json = parakeet_capi_transcribe_pcm_nbest_json(
        ctx, audio.samples.data(), (int)audio.samples.size(), audio.sample_rate,
        beam_size, 2, 0, nullptr);
    if (!pcm_json ||
        !std::strstr(pcm_json, "\"score_norm\":false") ||
        !std::strstr(pcm_json, "\"hypotheses\":[")) {
        std::fprintf(stderr, "test_tdt_beam: invalid PCM C API JSON: %s\n",
                     pcm_json ? pcm_json : parakeet_capi_last_error(ctx));
        parakeet_capi_free_string(pcm_json);
        parakeet_capi_free(ctx);
        return 1;
    }
    parakeet_capi_free_string(pcm_json);

    char* invalid = parakeet_capi_transcribe_path_nbest_json(
        ctx, "tests/fixtures/does-not-exist.wav", 0, 1, 1, nullptr);
    if (invalid || !std::strstr(parakeet_capi_last_error(ctx), "beam_size")) {
        std::fprintf(stderr,
                     "test_tdt_beam: C API validation did not fail early\n");
        parakeet_capi_free_string(invalid);
        parakeet_capi_free(ctx);
        return 1;
    }
    parakeet_capi_free(ctx);
    return 0;
}
