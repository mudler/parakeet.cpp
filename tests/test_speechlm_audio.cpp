// Parity of the speech LLM audio side (mel -> ASR encoder -> speaker branch -> fusion ->
// adapter) against Hugging Face transformers fp32 (scripts/gen_speechlm_audio_baseline.py).
//
// Needs the converted models and a baseline, so it skips (77) unless these are set:
//   PARAKEET_TEST_SPEECHLM_AUDIO_GGUF   audio GGUF (arch speechlm_audio)
//   PARAKEET_TEST_SPEECHLM_DIAR_GGUF    diarization GGUF
// and the baseline exists. Make it from the fixture WAV (2 s of speech) with
//   scripts/gen_speechlm_audio_baseline.py --wav tests/fixtures/speechlm_audio_2s.wav \
//       --output models/speechlm_audio_baseline.gguf ...
// Optional, to check another clip (for example one long enough for several encoder chunks):
//   PARAKEET_TEST_SPEECHLM_BASELINE     baseline GGUF (default models/speechlm_audio_baseline.gguf)
//   PARAKEET_TEST_SPEECHLM_WAV          the WAV the baseline was made from
//                                       (default tests/fixtures/speechlm_audio_2s.wav)
// A bundle works too: set PARAKEET_TEST_SPEECHLM_AUDIO_COMPONENT / _DIAR_COMPONENT.
#include "audio_io.hpp"
#include "speechlm_audio.hpp"
#include "parity.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static const char* env(const char* k, const char* d = nullptr) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : d;
}

int main() {
    const char* audio = env("PARAKEET_TEST_SPEECHLM_AUDIO_GGUF");
    const char* diar = env("PARAKEET_TEST_SPEECHLM_DIAR_GGUF");
    if (!audio || !diar) {
        std::fprintf(stderr, "PARAKEET_TEST_SPEECHLM_{AUDIO_GGUF,DIAR_GGUF} not set; skipping\n");
        return 77;
    }
    const char* base = env("PARAKEET_TEST_SPEECHLM_BASELINE", "models/speechlm_audio_baseline.gguf");
    if (FILE* f = std::fopen(base, "rb")) std::fclose(f);
    else { std::fprintf(stderr, "baseline %s not found; skipping\n", base); return 77; }
    const std::string wav = env("PARAKEET_TEST_SPEECHLM_WAV", "tests/fixtures/speechlm_audio_2s.wav");
    std::string err;
    auto m = pk::SpeechLmAudio::load(audio, env("PARAKEET_TEST_SPEECHLM_AUDIO_COMPONENT", ""), diar,
                                     env("PARAKEET_TEST_SPEECHLM_DIAR_COMPONENT", ""), &err);
    if (!m) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }

    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav, a)) { std::fprintf(stderr, "cannot read %s\n", wav.c_str()); return 1; }

    std::vector<float> out;
    int n = 0;
    pk::SpeechLmAudioStages st;
    m->embed_pcm(a.samples, out, n, &st);
    std::fprintf(stderr, "%s: %.1f s, %d tokens x %d, %d valid mel frames\n", wav.c_str(),
                 a.samples.size() / 16000.0, n, m->dim(), st.n_valid_mel);

    int fail = 0;
    if (n != m->tokens_for_samples((long long)a.samples.size())) { std::fprintf(stderr, "token count mismatch\n"); ++fail; }
    struct Stage { const char* name; const std::vector<float>* got; float atol; float rtol; } stages[] = {
        {"asr_hidden", &st.asr_hidden, 2e-3f, 1e-3f},
        {"fused", &st.fused, 5e-3f, 1e-3f},
        {"adapter", &out, 5e-3f, 1e-3f},
    };
    for (const Stage& s : stages) {
        std::vector<float> ref;
        std::vector<int64_t> shape;
        if (!pktest::load_baseline(base, s.name, ref, shape)) continue;   // stage not stored
        if (!pktest::compare(*s.got, ref, s.name, s.atol, s.rtol)) ++fail;
    }
    {   // Speaker probabilities. The reference pads the spectrogram with one extra, fully
        // masked group of frames when the valid length is a multiple of 8 (the centred STFT
        // makes one frame more than the audio has), and the speaker head's convolution at the
        // last valid frame sees that group. This engine pads the edge with zeros, so the last
        // frame may differ a little; every other frame must match, and no speaker may flip.
        std::vector<float> ref;
        std::vector<int64_t> shape;
        if (pktest::load_baseline(base, "speaker_probs", ref, shape)) {
            const std::vector<float>& got = st.speaker_probs;
            const int ns = m->num_speakers();
            const size_t body = got.size() >= (size_t)ns ? got.size() - ns : 0;
            std::vector<float> g0(got.begin(), got.begin() + body), r0(ref.begin(), ref.begin() + body);
            if (!pktest::compare(g0, r0, "speaker_probs (all but last frame)", 5e-3f, 0.0f)) ++fail;
            std::vector<float> g1(got.begin() + body, got.end()), r1(ref.begin() + body, ref.end());
            if (!pktest::compare(g1, r1, "speaker_probs (last frame)", 5e-2f, 0.0f)) ++fail;
            size_t flips = 0;
            for (size_t i = 0; i < got.size() && i < ref.size(); ++i) flips += (got[i] > 0.5f) != (ref[i] > 0.5f);
            std::fprintf(stderr, "[speaker activity] threshold flips: %zu of %zu\n", flips, got.size());
            if (flips) ++fail;
        }
    }
    std::fprintf(stderr, fail ? "FAILED\n" : "PASSED\n");
    return fail ? 1 : 0;
}
