// Test error paths for invalid streaming diarization configurations.
//
// Verifies that StreamingDiarization checks configuration validity and throws
// std::runtime_error with message "parakeet: invalid diarization streaming config"
// when given invalid stream config parameters. No model file required.

#include "diarization_streaming.hpp"
#include "model_loader.hpp"
#include "gguf.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(c) do { \
    if (!(c)) { \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); \
        ++failures; \
    } \
} while (0)

const char* kTmpGguf = "test_streaming_diarization_config.tmp.gguf";

void create_dummy_diar_gguf(const char* path) {
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "parakeet");
    gguf_set_val_str(g, "parakeet.arch", "diarization");
    gguf_set_val_u32(g, "parakeet.encoder.d_model", 64);
    gguf_set_val_u32(g, "parakeet.encoder.n_layers", 1);
    gguf_set_val_u32(g, "parakeet.encoder.n_heads", 4);
    gguf_set_val_u32(g, "parakeet.encoder.subsampling_factor", 8);
    gguf_set_val_u32(g, "parakeet.preprocessor.n_mels", 80);
    gguf_set_val_str(g, "parakeet.encoder.self_attention_model", "rope");
    gguf_set_val_f32(g, "parakeet.encoder.rotary_fraction", 1.0f);
    gguf_set_val_bool(g, "parakeet.encoder.pre_block_norm", true);

    gguf_set_val_u32(g, "parakeet.diar.n_speakers", 4);
    gguf_set_val_u32(g, "parakeet.diar.tf_d_model", 64);
    gguf_set_val_u32(g, "parakeet.diar.upsample_factor", 8);
    gguf_set_val_u32(g, "parakeet.diar.chunk_len", 9);
    gguf_set_val_u32(g, "parakeet.diar.spkcache_len", 264);
    gguf_set_val_u32(g, "parakeet.diar.fifo_len", 264);
    gguf_set_val_u32(g, "parakeet.diar.spkcache_update_period", 222);
    gguf_set_val_u32(g, "parakeet.diar.spkcache_sil_frames_per_spk", 2);
    gguf_set_val_bool(g, "parakeet.diar.use_learnable_sil_emb", false);

    gguf_write_to_file(g, path, false);
    gguf_free(g);
}

}  // namespace

int main() {
    create_dummy_diar_gguf(kTmpGguf);

    pk::ModelLoader ml;
    if (!ml.load(kTmpGguf)) {
        std::fprintf(stderr, "FAIL: failed to load synthetic GGUF\n");
        std::remove(kTmpGguf);
        return 1;
    }

    const pk::ParakeetConfig& cfg = ml.config();

    // Create a base valid stream config derived from cfg.
    const pk::DiarStreamConfig valid_scfg = pk::DiarStreamConfig::from_model(cfg);

    struct TestCase {
        const char* description;
        pk::DiarStreamConfig scfg;
    };

    std::vector<TestCase> cases;

    {
        pk::DiarStreamConfig c = valid_scfg;
        c.chunk_len = 0;
        cases.push_back({"chunk_len <= 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.chunk_len = -1;
        cases.push_back({"chunk_len < 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.spkcache_len = 0;
        cases.push_back({"spkcache_len <= 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.fifo_len = -1;
        cases.push_back({"fifo_len < 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.left_context = -1;
        cases.push_back({"left_context < 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.right_context = -1;
        cases.push_back({"right_context < 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.update_period = 0;
        cases.push_back({"update_period <= 0", c});
    }
    {
        pk::DiarStreamConfig c = valid_scfg;
        c.update_period = -5;
        cases.push_back({"update_period < 0", c});
    }
    {
        // spkcache_len / n_spk_ - sil_frames_per_spk_ <= 0
        // n_spk_ = 4, sil_frames_per_spk_ = 2 => spkcache_len = 8 yields 8/4 - 2 = 0 <= 0
        pk::DiarStreamConfig c = valid_scfg;
        c.spkcache_len = 8;
        cases.push_back({"spkcache_len / n_spk - sil_frames_per_spk <= 0", c});
    }

    const std::string expected_msg = "parakeet: invalid diarization streaming config";

    for (const auto& tc : cases) {
        bool threw = false;
        std::string caught_msg;
        try {
            pk::StreamingDiarization sd(ml, tc.scfg);
        } catch (const std::runtime_error& e) {
            threw = true;
            caught_msg = e.what();
        } catch (...) {
            threw = true;
            caught_msg = "non-runtime_error exception";
        }

        if (!threw) {
            std::fprintf(stderr, "FAIL: %s did not throw exception\n", tc.description);
            ++failures;
        } else if (caught_msg != expected_msg) {
            std::fprintf(stderr, "FAIL: %s threw wrong message: got '%s', expected '%s'\n",
                         tc.description, caught_msg.c_str(), expected_msg.c_str());
            ++failures;
        }
    }

    std::remove(kTmpGguf);

    if (failures) return 1;
    std::puts("test_streaming_diarization_config: OK");
    return 0;
}
