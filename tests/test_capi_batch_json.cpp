#include "parakeet_capi.h"
#include "audio_io.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

// The two model env vars are independent: the offline block (PARAKEET_TEST_GGUF)
// exercises the non-lang batch JSON path and its language-aware delegate on a
// real offline model; the prompt block (PARAKEET_TEST_GGUF_NEMOTRON) exercises
// the batched target_lang error handling on a multilingual prompt model. Each
// block runs only when its env var is set; if neither is set we skip (77).
int main() {
    bool ran_any = false;

    // Build a 2-clip batch (speech.wav twice) reused by both blocks.
    pk::Audio a;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", a) || a.samples.empty()) {
        std::fprintf(stderr, "wav load failed\n"); return 1;
    }
    std::vector<float> concat;
    concat.insert(concat.end(), a.samples.begin(), a.samples.end());
    concat.insert(concat.end(), a.samples.begin(), a.samples.end());
    int n_samples[2] = { (int)a.samples.size(), (int)a.samples.size() };

    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (gguf) {
        ran_any = true;
        parakeet_ctx* ctx = parakeet_capi_load(gguf);
        if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }

        char* single = parakeet_capi_transcribe_path_json(ctx, "tests/fixtures/speech.wav", 0);
        if (!single) { std::fprintf(stderr, "single failed: %s\n", parakeet_capi_last_error(ctx)); parakeet_capi_free(ctx); return 1; }
        std::string single_doc(single);
        parakeet_capi_free_string(single);

        char* batch = parakeet_capi_transcribe_pcm_batch_json(ctx, concat.data(), n_samples, 2, 16000, 0);
        if (!batch) { std::fprintf(stderr, "batch failed: %s\n", parakeet_capi_last_error(ctx)); parakeet_capi_free(ctx); return 1; }
        std::string doc(batch);
        parakeet_capi_free_string(batch);

        auto text_field = [](const std::string& s) -> std::string {
            size_t p = s.find("\"text\":\"");
            if (p == std::string::npos) return "";
            p += 8; size_t q = s.find('"', p);
            return s.substr(p, q - p);
        };
        std::string t = text_field(single_doc);
        bool is_array = !doc.empty() && doc.front() == '[' && doc.back() == ']';
        size_t cnt = 0, pos = 0;
        std::string needle = "\"text\":\"" + t + "\"";
        while ((pos = doc.find(needle, pos)) != std::string::npos) { ++cnt; pos += needle.size(); }
        bool ok = is_array && !t.empty() && cnt == 2;
        std::fprintf(stderr, "array=%d text='%s' count=%zu -> %s\n", is_array, t.c_str(), cnt, ok?"OK":"FAIL");

        // The language-aware delegate with the model default (NULL) must match the
        // non-lang path byte-for-byte on an offline (non-prompt) model.
        char* batch_lang = parakeet_capi_transcribe_pcm_batch_json_lang(ctx, concat.data(), n_samples, 2, 16000, 0, nullptr);
        bool lang_ok = batch_lang && doc == std::string(batch_lang);
        std::fprintf(stderr, "batch_json_lang(NULL)==batch_json -> %s\n", lang_ok?"OK":"FAIL");
        parakeet_capi_free_string(batch_lang);

        parakeet_capi_free(ctx);
        if (!ok || !lang_ok) return 1;
    }

    // Prompt (multilingual / nemotron) model: exercise the batched target_lang
    // variant. This fixture is a CAUSAL streaming prompt model
    // (causal_downsampling=True). Batched causal subsampling is now supported
    // (byte-identical to per-item), so a valid-language 2-clip batch runs through
    // the batched encoder and returns a JSON array of length 2. We also assert
    // the catchable error path: an unknown locale is rejected by
    // resolve_prompt_index (which runs before the encoder) -> NULL + non-empty
    // last_error, proving target_lang is threaded through the batched C-API.
    // Skipped cleanly when the env var is unset.
    const char* nemotron = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    if (nemotron) {
        ran_any = true;
        parakeet_ctx* nctx = parakeet_capi_load(nemotron);
        if (!nctx) { std::fprintf(stderr, "nemotron load failed\n"); return 1; }

        // A valid locale ("de") must return a non-NULL JSON array of length 2.
        char* ngood = parakeet_capi_transcribe_pcm_batch_json_lang(
            nctx, concat.data(), n_samples, 2, 16000, 0, "de");
        if (!ngood) {
            std::fprintf(stderr, "nemotron batch_json_lang(de) returned NULL: %s\n",
                         parakeet_capi_last_error(nctx));
            parakeet_capi_free(nctx);
            return 1;
        }
        std::string ndoc(ngood);
        parakeet_capi_free_string(ngood);
        bool nis_array = !ndoc.empty() && ndoc.front() == '[' && ndoc.back() == ']';
        // Two JSON objects in the array (one per clip).
        size_t nobj = 0, npos = 0;
        while ((npos = ndoc.find("\"text\":", npos)) != std::string::npos) { ++nobj; npos += 7; }
        bool ngood_ok = nis_array && nobj == 2;
        std::fprintf(stderr, "nemotron batch_json_lang(de) array=%d objects=%zu -> %s\n",
                     nis_array, nobj, ngood_ok ? "OK" : "FAIL");
        if (!ngood_ok) { parakeet_capi_free(nctx); return 1; }

        // An unknown locale must fail cleanly: NULL + non-empty last_error.
        char* nbad = parakeet_capi_transcribe_pcm_batch_json_lang(
            nctx, concat.data(), n_samples, 2, 16000, 0, "zzz");
        if (nbad != nullptr) {
            std::fprintf(stderr, "nemotron batch_json_lang(zzz) returned non-NULL\n");
            parakeet_capi_free_string(nbad);
            parakeet_capi_free(nctx);
            return 1;
        }
        const char* nerr = parakeet_capi_last_error(nctx);
        if (!nerr || nerr[0] == '\0') {
            std::fprintf(stderr, "nemotron unknown-lang did not set last_error\n");
            parakeet_capi_free(nctx);
            return 1;
        }
        std::fprintf(stderr, "nemotron unknown-lang error = %s\n", nerr);
        parakeet_capi_free(nctx);
        std::fprintf(stderr, "PASS nemotron batch_json_lang error path\n");
    }

    if (!ran_any) {
        std::fprintf(stderr, "no model env var set (PARAKEET_TEST_GGUF / PARAKEET_TEST_GGUF_NEMOTRON); skip\n");
        return 77;
    }
    return 0;
}
