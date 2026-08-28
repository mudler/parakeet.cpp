#include "model.hpp"
#include "audio_io.hpp"
#include "search.hpp"
#include "tokenizer.hpp"
#include "parakeet_capi.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Coverage for parakeet_capi_transcribe_pcm_logits (the classroom-captions#63
// logits-exposure entry point), in two independent blocks (mirrors
// test_capi.cpp's two-optional-env-vars shape):
//
//   1. Self-consistency on a real standalone-CTC checkpoint, through the
//      actual C-API (parakeet_capi_load / parakeet_capi_transcribe_pcm_logits
//      / parakeet_capi_free_logits — exercising the C boundary and malloc/free
//      contract, not just the underlying C++ method): reconstructing text
//      from the exposed [T, vocab+1] log-prob matrix via the SAME ctc_greedy +
//      detokenize path decode_enc_out uses internally must reproduce
//      transcribe_pcm(..., kCTC)'s own greedy transcript byte-for-byte
//      (transcribe_pcm and the tokenizer/blank_id come from a separate
//      pk::Model load, used only for that reference text and metadata).
//      Exercises the ctc_head_tensor standalone-model fallback path
//      (decoder.* prefix, not the hybrid ctc_decoder.*).
//
//   2. Error path at the C-API boundary: a model with NO CTC head at all
//      (e.g. a pure RNNT/TDT streaming model) must make
//      parakeet_capi_transcribe_pcm_logits fail cleanly — nonzero return,
//      *out_logits left NULL, ctx last_error set — never crash or let the
//      underlying std::runtime_error (from ctc_head_tensor) cross the C
//      boundary.
//
// This is a self-consistency test (our own greedy decode vs. our own exposed
// logits, both computed here), not a NeMo parity check — that's already
// covered by test_transcribe_ctc.cpp / test_ctc.cpp.
//
// Env:
//   PARAKEET_TEST_GGUF_CTC      standalone CTC GGUF (block 1; skip if unset)
//   PARAKEET_TEST_GGUF_NO_CTC   a GGUF with no CTC head, e.g. a pure RNNT/TDT
//                               streaming model (block 2; skip if unset)
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; wav path is relative)
int main() {
    bool ran_any = false;

    const char* ctc_gguf = std::getenv("PARAKEET_TEST_GGUF_CTC");
    if (ctc_gguf) {
        ran_any = true;
        // pk::Model is used only for the reference text and tokenizer/blank_id
        // access below — the logits themselves come from the actual C-API
        // (parakeet_capi_load/parakeet_capi_transcribe_pcm_logits), so this
        // block exercises the C boundary and malloc/free contract, not just
        // the underlying C++ method.
        auto model = pk::Model::load(ctc_gguf);
        if (!model) {
            std::fprintf(stderr, "test_capi_ctc_logits: load failed for %s\n", ctc_gguf);
            return 1;
        }

        pk::Audio audio;
        if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio) || audio.samples.empty()) {
            std::fprintf(stderr, "test_capi_ctc_logits: wav load failed\n");
            return 1;
        }

        const std::string reference = model->transcribe_pcm(audio.samples, 16000, pk::Decoder::kCTC);
        const int blank_id = (int)model->config().blank_id;
        const std::vector<std::string> tokenizer_pieces = model->loader().tokenizer_pieces();
        // Release the pk::Model before loading a second full model via the C-API
        // below — keeping both resident at once nearly doubles peak RAM for a
        // 1.1B checkpoint. Only blank_id/tokenizer_pieces (cached above) and the
        // already-computed reference text are needed from here on.
        model.reset();

        parakeet_ctx* ctx = parakeet_capi_load(ctc_gguf);
        if (!ctx) {
            std::fprintf(stderr, "test_capi_ctc_logits: parakeet_capi_load failed for %s\n", ctc_gguf);
            return 1;
        }

        float* out_logits = nullptr;
        int T = 0, vocab_plus_1 = 0;
        int rc = parakeet_capi_transcribe_pcm_logits(
            ctx, audio.samples.data(), (int)audio.samples.size(), 16000,
            &out_logits, &T, &vocab_plus_1);

        if (rc != 0) {
            std::fprintf(stderr, "test_capi_ctc_logits: transcribe_pcm_logits failed: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }
        if (!out_logits || T <= 0 || vocab_plus_1 <= 0) {
            std::fprintf(stderr,
                "test_capi_ctc_logits: bad output out_logits=%p T=%d vocab_plus_1=%d\n",
                (void*)out_logits, T, vocab_plus_1);
            parakeet_capi_free_logits(out_logits);
            parakeet_capi_free(ctx);
            return 1;
        }

        std::vector<float> logits(out_logits, out_logits + (size_t)T * (size_t)vocab_plus_1);
        parakeet_capi_free_logits(out_logits);
        parakeet_capi_free(ctx);

        std::vector<int32_t> ids = pk::ctc_greedy(logits, T, vocab_plus_1, blank_id);
        const std::string reconstructed = pk::detokenize(
            tokenizer_pieces,
            pk::strip_special_tokens(tokenizer_pieces, ids));

        std::fprintf(stderr, "test_capi_ctc_logits: reference     = %s\n", reference.c_str());
        std::fprintf(stderr, "test_capi_ctc_logits: reconstructed = %s\n", reconstructed.c_str());
        std::fprintf(stderr, "test_capi_ctc_logits: T=%d vocab_plus_1=%d blank_id=%d\n",
                     T, vocab_plus_1, blank_id);

        if (reconstructed != reference) {
            std::fprintf(stderr, "test_capi_ctc_logits: MISMATCH\n");
            return 1;
        }

        std::fprintf(stderr,
            "test_capi_ctc_logits: PASS block 1 (argmax-greedy over the C-API's exposed "
            "logits reproduces the CLI's own greedy text)\n");
    } else {
        std::fprintf(stderr, "test_capi_ctc_logits: PARAKEET_TEST_GGUF_CTC not set; skip block 1\n");
    }

    const char* no_ctc_gguf = std::getenv("PARAKEET_TEST_GGUF_NO_CTC");
    if (no_ctc_gguf) {
        ran_any = true;
        parakeet_ctx* ctx = parakeet_capi_load(no_ctc_gguf);
        if (!ctx) {
            std::fprintf(stderr, "test_capi_ctc_logits: load failed for %s\n", no_ctc_gguf);
            return 1;
        }

        pk::Audio audio;
        if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio) || audio.samples.empty()) {
            std::fprintf(stderr, "test_capi_ctc_logits: wav load failed\n");
            parakeet_capi_free(ctx);
            return 1;
        }

        float* out_logits = nullptr;
        int out_T = 0, out_vocab_plus_1 = 0;
        int rc = parakeet_capi_transcribe_pcm_logits(
            ctx, audio.samples.data(), (int)audio.samples.size(), 16000,
            &out_logits, &out_T, &out_vocab_plus_1);

        if (rc == 0) {
            std::fprintf(stderr,
                "test_capi_ctc_logits: expected failure on a no-CTC-head model, got rc=0\n");
            parakeet_capi_free_logits(out_logits);
            parakeet_capi_free(ctx);
            return 1;
        }
        if (out_logits != nullptr) {
            std::fprintf(stderr,
                "test_capi_ctc_logits: rc!=0 but *out_logits is non-NULL (ownership contract violated)\n");
            parakeet_capi_free_logits(out_logits);
            parakeet_capi_free(ctx);
            return 1;
        }
        const char* err = parakeet_capi_last_error(ctx);
        if (!err || err[0] == '\0') {
            std::fprintf(stderr, "test_capi_ctc_logits: no-CTC-head failure did not set last_error\n");
            parakeet_capi_free(ctx);
            return 1;
        }
        std::fprintf(stderr, "test_capi_ctc_logits: no-CTC-head error (expected) = %s\n", err);

        parakeet_capi_free(ctx);
        std::fprintf(stderr,
            "test_capi_ctc_logits: PASS block 2 (no-CTC-head model fails cleanly, "
            "no crash, last_error set)\n");
    } else {
        std::fprintf(stderr, "test_capi_ctc_logits: PARAKEET_TEST_GGUF_NO_CTC not set; skip block 2\n");
    }

    if (!ran_any) {
        std::fprintf(stderr,
            "test_capi_ctc_logits: no model env var set (PARAKEET_TEST_GGUF_CTC / "
            "PARAKEET_TEST_GGUF_NO_CTC); skip\n");
        return 77;
    }
    return 0;
}
