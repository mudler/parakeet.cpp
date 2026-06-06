#include "parakeet_capi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Flat C-API end-to-end test: load -> transcribe -> free.
//
// Loads the 110m anchor GGUF, transcribes tests/fixtures/speech.wav with the
// TDT head (decoder == 2), and asserts the returned transcript equals the known
// NeMo TDT reference. Also checks that loading a nonexistent path returns NULL
// (no crash, exception contained at the boundary).
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; wav path is relative)
//
// Env:
//   PARAKEET_TEST_GGUF            model weights (skip 77 if unset)
//   PARAKEET_TEST_GGUF_NEMOTRON   prompt (multilingual) model; if set, also
//                                 exercises the target_lang C-API variants

static const char* kExpected =
    "Well, I don't wish to see it any more, observed Phoebe, turning away her "
    "eyes. It is certainly very like the old portrait.";

int main() {
    // ABI version must be a sane positive integer.
    if (parakeet_capi_abi_version() < 1) {
        std::fprintf(stderr, "test_capi: abi version < 1\n");
        return 1;
    }

    // Loading a nonexistent model must fail gracefully (NULL, no crash).
    parakeet_ctx* bad = parakeet_capi_load("/nonexistent/x.gguf");
    if (bad != nullptr) {
        std::fprintf(stderr, "test_capi: load of nonexistent path returned non-NULL\n");
        parakeet_capi_free(bad);
        return 1;
    }

    // The 110m anchor (PARAKEET_TEST_GGUF) and the prompt/multilingual model
    // (PARAKEET_TEST_GGUF_NEMOTRON) are independent: each block runs only when
    // its env var is set. If NEITHER is set the test skips (77).
    bool ran_any = false;

    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (gguf) {
        ran_any = true;
        parakeet_ctx* ctx = parakeet_capi_load(gguf);
        if (!ctx) {
            std::fprintf(stderr, "test_capi: parakeet_capi_load failed for %s\n", gguf);
            return 1;
        }

        // decoder == 2 -> TDT/transducer head.
        char* text = parakeet_capi_transcribe_path(ctx, "tests/fixtures/speech.wav", 2);
        if (!text) {
            std::fprintf(stderr, "test_capi: transcribe_path returned NULL: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }

        std::fprintf(stderr, "test_capi: got      = %s\n", text);
        std::fprintf(stderr, "test_capi: expected = %s\n", kExpected);

        const bool match = std::strcmp(text, kExpected) == 0;
        parakeet_capi_free_string(text);
        parakeet_capi_free(ctx);

        if (!match) {
            std::fprintf(stderr, "test_capi: MISMATCH vs NeMo TDT reference\n");
            return 1;
        }
        std::fprintf(stderr, "test_capi: PASS (word-for-word match with NeMo TDT)\n");
    } else {
        std::fprintf(stderr, "test_capi: PARAKEET_TEST_GGUF not set; skip anchor block\n");
    }

    // Prompt (multilingual) model: exercise the target_lang variants. Skipped
    // cleanly when PARAKEET_TEST_GGUF_NEMOTRON is unset.
    const char* nemotron = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    if (nemotron) {
        ran_any = true;
        parakeet_ctx* nctx = parakeet_capi_load(nemotron);
        if (!nctx) {
            std::fprintf(stderr, "test_capi: load failed for nemotron %s\n", nemotron);
            return 1;
        }

        // A known language prompt must transcribe (non-NULL).
        char* de = parakeet_capi_transcribe_path_lang(
            nctx, "tests/fixtures/speech.wav", 0, "de");
        if (!de) {
            std::fprintf(stderr, "test_capi: transcribe_path_lang(de) returned NULL: %s\n",
                         parakeet_capi_last_error(nctx));
            parakeet_capi_free(nctx);
            return 1;
        }
        std::fprintf(stderr, "test_capi: nemotron de = %s\n", de);
        parakeet_capi_free_string(de);

        // An unknown locale must fail cleanly: NULL + non-empty last_error.
        char* bad_lang = parakeet_capi_transcribe_path_lang(
            nctx, "tests/fixtures/speech.wav", 0, "zzz");
        if (bad_lang != nullptr) {
            std::fprintf(stderr, "test_capi: transcribe_path_lang(zzz) returned non-NULL\n");
            parakeet_capi_free_string(bad_lang);
            parakeet_capi_free(nctx);
            return 1;
        }
        const char* err = parakeet_capi_last_error(nctx);
        if (!err || err[0] == '\0') {
            std::fprintf(stderr, "test_capi: unknown-lang did not set last_error\n");
            parakeet_capi_free(nctx);
            return 1;
        }
        std::fprintf(stderr, "test_capi: nemotron unknown-lang error = %s\n", err);

        // Streaming path must reject an unknown locale exactly like the offline
        // path: NULL + non-empty last_error (no silent fallback to the default).
        parakeet_stream* bad_stream = parakeet_capi_stream_begin_lang(nctx, "zzz");
        if (bad_stream != nullptr) {
            std::fprintf(stderr,
                "test_capi: stream_begin_lang(zzz) returned non-NULL\n");
            parakeet_capi_stream_free(bad_stream);
            parakeet_capi_free(nctx);
            return 1;
        }
        const char* serr = parakeet_capi_last_error(nctx);
        if (!serr || serr[0] == '\0') {
            std::fprintf(stderr,
                "test_capi: stream unknown-lang did not set last_error\n");
            parakeet_capi_free(nctx);
            return 1;
        }
        std::fprintf(stderr, "test_capi: nemotron stream unknown-lang error = %s\n",
                     serr);

        // A known language prompt must begin a stream (non-NULL); free it.
        parakeet_stream* ok_stream = parakeet_capi_stream_begin_lang(nctx, "en");
        if (!ok_stream) {
            std::fprintf(stderr,
                "test_capi: stream_begin_lang(en) returned NULL: %s\n",
                parakeet_capi_last_error(nctx));
            parakeet_capi_free(nctx);
            return 1;
        }
        parakeet_capi_stream_free(ok_stream);

        parakeet_capi_free(nctx);
        std::fprintf(stderr, "test_capi: PASS nemotron target_lang variants\n");
    } else {
        std::fprintf(stderr,
            "test_capi: PARAKEET_TEST_GGUF_NEMOTRON not set; skip prompt block\n");
    }

    if (!ran_any) {
        std::fprintf(stderr,
            "test_capi: no model env var set (PARAKEET_TEST_GGUF / "
            "PARAKEET_TEST_GGUF_NEMOTRON); skip\n");
        return 77;
    }
    return 0;
}
