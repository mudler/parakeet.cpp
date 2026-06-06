#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char* env = std::getenv("PARAKEET_TEST_GGUF");
    const char* npath = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    if (!env && !npath) {
        std::fprintf(stderr, "PARAKEET_TEST_GGUF / PARAKEET_TEST_GGUF_NEMOTRON not set; skipping\n");
        return 77;
    }

    // Base model checks (only when PARAKEET_TEST_GGUF points at a fixture).
    if (env) {
        pk::ModelLoader ml;
        if (!ml.load(env)) { std::fprintf(stderr, "load failed\n"); return 1; }
        const pk::ParakeetConfig& c = ml.config();
        if (c.arch.empty())   { std::fprintf(stderr, "empty arch\n"); return 1; }
        if (c.d_model == 0 || c.n_layers == 0 || c.n_heads == 0) { std::fprintf(stderr, "bad encoder dims\n"); return 1; }
        if (c.vocab_size == 0) { std::fprintf(stderr, "bad vocab\n"); return 1; }
        if (c.blank_id != c.vocab_size) { std::fprintf(stderr, "blank!=vocab\n"); return 1; }
        // mel filterbank tensor must be present
        if (ml.tensor("preprocessor.featurizer.fb") == nullptr) { std::fprintf(stderr, "no fb\n"); return 1; }
        // first conformer layer norm must be present (verbatim name)
        if (ml.tensor("encoder.layers.0.norm_feed_forward1.weight") == nullptr) {
            std::fprintf(stderr, "no layer0 norm\n"); return 1;
        }
        std::printf("loader ok: arch=%s d_model=%u layers=%u heads=%u vocab=%u\n",
                    c.arch.c_str(), c.d_model, c.n_layers, c.n_heads, c.vocab_size);
    }

    // Prompt-conditioning config (nemotron). Runs whenever the fixture is set,
    // independently of PARAKEET_TEST_GGUF.
    if (npath) {
        pk::ModelLoader nl;
        if (!nl.load(npath)) { std::fprintf(stderr, "load nemotron failed\n"); return 1; }
        const pk::ParakeetConfig& nc = nl.config();
        if (!nc.prompt.present) { std::fprintf(stderr, "prompt.present false\n"); return 1; }
        if (nc.prompt.num_prompts != 128) { std::fprintf(stderr, "num_prompts!=128\n"); return 1; }
        if (nc.prompt.default_lang != "auto") { std::fprintf(stderr, "default_lang!=auto\n"); return 1; }
        if (nc.prompt.lang_to_index("de") != 9) { std::fprintf(stderr, "de!=9\n"); return 1; }
        if (nc.prompt.lang_to_index("auto") != 101) { std::fprintf(stderr, "auto!=101\n"); return 1; }
        if (nc.prompt.lang_to_index("zzz") != -1) { std::fprintf(stderr, "unknown!=-1\n"); return 1; }
        if (nc.use_bias) { std::fprintf(stderr, "use_bias should be false\n"); return 1; }
        std::fprintf(stderr, "nemotron prompt config OK\n");
    }
    return 0;
}
