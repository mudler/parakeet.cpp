#include "model.hpp"
#include "audio_io.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

// End-to-end: the C++ nemotron transcript must equal NeMo's transcript for the
// SAME target_lang. Skips (77) unless set:
//   PARAKEET_TEST_GGUF_NEMOTRON       converted nemotron gguf
//   PARAKEET_TEST_BASELINE_NEMOTRON   prompt baseline (baseline.rnnt_text, baseline.target_lang)
//   PARAKEET_TEST_NEMOTRON_WAV        the clip used for the baseline (default tests/fixtures/speech.wav)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_NEMOTRON");
    if (!gguf || !base) { std::fprintf(stderr, "fixtures not set; skip\n"); return 77; }
    const char* wav = std::getenv("PARAKEET_TEST_NEMOTRON_WAV");
    std::string wav_path = wav ? wav : "tests/fixtures/speech.wav";

    std::string lang, ref;
    if (!pktest::load_kv_str(base, "baseline.target_lang", lang)) return 1;
    if (!pktest::load_kv_str(base, "baseline.rnnt_text", ref)) return 1;

    auto m = pk::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "load failed\n"); return 1; }
    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav_path, a)) { std::fprintf(stderr, "audio load failed\n"); return 1; }

    std::string got = m->transcribe_16k(a.samples, pk::Decoder::kDefault, lang);
    std::fprintf(stderr, "lang=%s\n got=%s\n ref=%s\n", lang.c_str(), got.c_str(), ref.c_str());
    if (got != ref) { std::fprintf(stderr, "MISMATCH\n"); return 1; }
    std::fprintf(stderr, "nemotron offline parity OK\n");
    return 0;
}
