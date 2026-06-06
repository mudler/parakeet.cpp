#include "model_loader.hpp"
#include "backend.hpp"
#include "prompt_kernel.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Parity: PromptKernel(encoder_out, prompt_index) must match NeMo's
// prompt_kernel(cat([encoded, onehot])). Skips (77) unless both the converted
// nemotron gguf and the prompt baseline are provided.
//   PARAKEET_TEST_GGUF_NEMOTRON      converted nemotron gguf
//   PARAKEET_TEST_BASELINE_NEMOTRON  prompt baseline gguf (encoder_out [D,T],
//                                    prompt_kernel_out [T,D], baseline.prompt_index)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_NEMOTRON");
    if (!gguf || !base) {
        std::fprintf(stderr, "test_prompt_kernel: fixtures not set; skip\n");
        return 77;
    }
    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    pk::ensure_weights_realized(ml);

    // encoder_out [D,T] (channels-first) + reference prompt_kernel_out [T,D].
    std::vector<float> enc; std::vector<int64_t> enc_shape;
    std::vector<float> ref; std::vector<int64_t> ref_shape;
    if (!pktest::load_baseline(base, "encoder_out", enc, enc_shape)) return 1;
    if (!pktest::load_baseline(base, "prompt_kernel_out", ref, ref_shape)) return 1;
    const int D = (int)enc_shape[0], T = (int)enc_shape[1];
    const int prompt_index = (int)pktest::pktest_read_u32(base, "baseline.prompt_index");
    std::fprintf(stderr, "[prompt_kernel] D=%d T=%d prompt_index=%d\n", D, T, prompt_index);

    pk::PromptKernel pkmod(ml);
    if (!pkmod.present()) { std::fprintf(stderr, "prompt not present in model\n"); return 1; }

    std::vector<float> got;  // channels-first [D, T]
    pkmod.apply(enc, D, T, prompt_index, got);

    // Transpose got [D,T] -> [T,D] to compare with ref [T,D].
    std::vector<float> got_td((size_t)T * D);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < D; ++c)
            got_td[(size_t)t * D + c] = got[(size_t)c * T + t];

    bool ok = pktest::compare(got_td, ref, "prompt_kernel", 2e-3f, 2e-3f);
    return ok ? 0 : 1;
}
