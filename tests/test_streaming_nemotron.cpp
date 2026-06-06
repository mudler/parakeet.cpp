#include "model.hpp"
#include "streaming.hpp"
#include "audio_io.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

// MODEL: nvidia nemotron-3.5-asr-streaming-0.6b (prompt-conditioned, cache-aware
// streaming FastConformer RNN-T).
// WORKING_DIRECTORY: the repo root (build/tests run from there).
//
// End-to-end STREAMING parity WITH the language prompt (Phase 3, Task 3.2). The
// C++ pk::StreamingSession drives the cache-aware streaming encoder, applies the
// PromptKernel per chunk for the baseline's target_lang, and decodes the RNN-T
// greedy carrying state across chunks. The running transcript (sess.text(), with
// <EOU>/<EOB> stripped) must equal NeMo's OWN cache-aware streaming transcript
// for the SAME language (baseline.stream_text), produced by gen_nemo_baseline's
// dump_prompt_baseline: NeMo streams the encoder, applies m.prompt_kernel to the
// concatenated streamed output, and RNN-T-greedy-decodes it.
//
// By the cache-aware equivalence property (test_streaming_decode) the per-chunk
// decode with carried state == whole-streamed-output decode, and the per-frame
// prompt one-hot is constant over time, so per-chunk prompt application == single
// application — hence the C++ streaming transcript must match NeMo's EXACTLY.
//
// Skips (77) unless set:
//   PARAKEET_TEST_GGUF_NEMOTRON       converted nemotron gguf
//   PARAKEET_TEST_BASELINE_NEMOTRON   prompt baseline (baseline.stream_text,
//                                     baseline.target_lang)
//   PARAKEET_TEST_NEMOTRON_WAV        the clip used for the baseline
//                                     (default tests/fixtures/speech.wav)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_NEMOTRON");
    if (!gguf || !base) {
        std::fprintf(stderr,
            "test_streaming_nemotron: PARAKEET_TEST_GGUF_NEMOTRON and/or "
            "PARAKEET_TEST_BASELINE_NEMOTRON not set; skip\n");
        return 77;
    }
    const char* wav = std::getenv("PARAKEET_TEST_NEMOTRON_WAV");
    std::string wav_path = wav ? wav : "tests/fixtures/speech.wav";

    std::string lang, ref;
    if (!pktest::load_kv_str(base, "baseline.target_lang", lang)) return 1;
    if (!pktest::load_kv_str(base, "baseline.stream_text", ref)) {
        std::fprintf(stderr,
            "[stream_nemotron] baseline.stream_text not found in %s "
            "(regenerate with gen_nemo_baseline.py --lang)\n", base);
        return 1;
    }

    auto m = pk::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "[stream_nemotron] load failed %s\n", gguf); return 1; }
    if (!m->config().prompt.present) {
        std::fprintf(stderr, "[stream_nemotron] model is not prompt-conditioned\n");
        return 1;
    }
    if (!m->config().streaming.present) {
        std::fprintf(stderr, "[stream_nemotron] model has no streaming config\n");
        return 1;
    }

    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav_path, a)) {
        std::fprintf(stderr, "[stream_nemotron] audio load failed %s\n", wav_path.c_str());
        return 1;
    }

    pk::StreamingSession sess(m->loader(), lang);
    pk::run_stream_over_pcm(sess, m->loader(), a.samples);
    std::string got = sess.text();

    std::fprintf(stderr, "lang=%s\n got=%s\n ref=%s\n",
                 lang.c_str(), got.c_str(), ref.c_str());
    if (got != ref) {
        std::fprintf(stderr, "[stream_nemotron] MISMATCH\n");
        return 1;
    }
    std::fprintf(stderr, "nemotron streaming parity OK\n");
    return 0;
}
