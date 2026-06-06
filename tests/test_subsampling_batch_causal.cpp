#include "model.hpp"
#include "audio_io.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

// Batched CAUSAL subsampling parity for the multilingual streaming nemotron
// model (causal_downsampling=True). Greedy decode is deterministic and the
// per-item causal path is NeMo-validated at WER 0, so the gold check is
// byte-identical equivalence: a clip transcribed inside a B>1 batch MUST equal
// the SAME clip transcribed standalone. Mixed-length batches exercise the
// per-item trailing-pad masking that interacts with the causal right pad of 1.
//
// Skips (77) unless PARAKEET_TEST_GGUF_NEMOTRON points at a causal nemotron gguf.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON");
    if (!gguf) { std::fprintf(stderr, "PARAKEET_TEST_GGUF_NEMOTRON not set; skip\n"); return 77; }

    auto model = pk::Model::load(gguf);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    pk::Audio speech, clip;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", speech) || speech.samples.empty()) {
        std::fprintf(stderr, "speech.wav load failed\n"); return 1;
    }
    if (!pk::load_audio_16k_mono("tests/fixtures/clip.wav", clip) || clip.samples.empty()) {
        std::fprintf(stderr, "clip.wav load failed\n"); return 1;
    }

    const std::string lang = "en";

    // A truncated speech slice: a SHORTER clip that still carries real content
    // (non-empty transcript), so when it is the padded/masked item in a batch a
    // per-item masking bug would corrupt its tokens. This is the strongest
    // mixed-length masking probe.
    std::vector<float> half(speech.samples.begin(),
                            speech.samples.begin() + (speech.samples.size() * 3) / 5);

    // Per-item standalone references (the ground truth for byte-identity).
    std::string ref_speech = model->transcribe_16k(speech.samples, pk::Decoder::kDefault, lang);
    std::string ref_clip   = model->transcribe_16k(clip.samples,   pk::Decoder::kDefault, lang);
    std::string ref_half   = model->transcribe_16k(half,           pk::Decoder::kDefault, lang);
    std::fprintf(stderr, "ref_speech='%s'\nref_clip='%s'\nref_half='%s'\n",
                 ref_speech.c_str(), ref_clip.c_str(), ref_half.c_str());

    bool ok = true;

    // (a) Uniform batch: both items identical to the standalone speech transcript.
    {
        auto out = model->transcribe_pcm_batch({speech.samples, speech.samples}, 16000,
                                               pk::Decoder::kDefault, lang);
        bool pass = out.size() == 2 && out[0] == ref_speech && out[1] == ref_speech;
        std::fprintf(stderr, "(a) uniform batch: size=%zu item0=%s item1=%s -> %s\n",
                     out.size(),
                     (out.size() > 0 && out[0] == ref_speech) ? "OK" : "DIFF",
                     (out.size() > 1 && out[1] == ref_speech) ? "OK" : "DIFF",
                     pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }

    // (b) MIXED-LENGTH batch (the real test): different lengths exercise the
    //     per-item causal masking; each item must be byte-identical to standalone.
    {
        auto out = model->transcribe_pcm_batch({speech.samples, clip.samples}, 16000,
                                               pk::Decoder::kDefault, lang);
        bool pass = out.size() == 2 && out[0] == ref_speech && out[1] == ref_clip;
        std::fprintf(stderr, "(b) mixed batch: size=%zu speech=%s clip=%s -> %s\n",
                     out.size(),
                     (out.size() > 0 && out[0] == ref_speech) ? "OK" : "DIFF",
                     (out.size() > 1 && out[1] == ref_clip) ? "OK" : "DIFF",
                     pass ? "PASS" : "FAIL");
        if (out.size() > 0 && out[0] != ref_speech)
            std::fprintf(stderr, "    batched speech='%s'\n", out[0].c_str());
        if (out.size() > 1 && out[1] != ref_clip)
            std::fprintf(stderr, "    batched clip='%s'\n", out[1].c_str());
        ok = ok && pass;
    }

    // Reversed order too (clip first): order must not perturb per-item identity.
    {
        auto out = model->transcribe_pcm_batch({clip.samples, speech.samples}, 16000,
                                               pk::Decoder::kDefault, lang);
        bool pass = out.size() == 2 && out[0] == ref_clip && out[1] == ref_speech;
        std::fprintf(stderr, "(b') reversed mixed batch: clip=%s speech=%s -> %s\n",
                     (out.size() > 0 && out[0] == ref_clip) ? "OK" : "DIFF",
                     (out.size() > 1 && out[1] == ref_speech) ? "OK" : "DIFF",
                     pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }

    // (b'') MIXED-LENGTH with a non-empty shorter item, both orderings. The half
    //       slice is the padded/masked item in one ordering; its tokens must be
    //       byte-identical to its standalone transcript.
    {
        auto out1 = model->transcribe_pcm_batch({speech.samples, half}, 16000,
                                                pk::Decoder::kDefault, lang);
        auto out2 = model->transcribe_pcm_batch({half, speech.samples}, 16000,
                                                pk::Decoder::kDefault, lang);
        bool pass = out1.size() == 2 && out1[0] == ref_speech && out1[1] == ref_half
                 && out2.size() == 2 && out2[0] == ref_half   && out2[1] == ref_speech;
        std::fprintf(stderr,
                     "(b'') [speech,half]: speech=%s half=%s ; [half,speech]: half=%s speech=%s -> %s\n",
                     (out1.size() > 0 && out1[0] == ref_speech) ? "OK" : "DIFF",
                     (out1.size() > 1 && out1[1] == ref_half) ? "OK" : "DIFF",
                     (out2.size() > 0 && out2[0] == ref_half) ? "OK" : "DIFF",
                     (out2.size() > 1 && out2[1] == ref_speech) ? "OK" : "DIFF",
                     pass ? "PASS" : "FAIL");
        if (out2.size() > 1 && out2[0] != ref_half)
            std::fprintf(stderr, "    batched half (padded item)='%s'\n", out2[0].c_str());
        ok = ok && pass;
    }

    // (c) Batched timestamps consistency: per-item text matches the single-clip
    //     timestamped text.
    {
        auto rs = model->transcribe_with_timestamps(speech.samples, 16000, pk::Decoder::kDefault, lang);
        auto rc = model->transcribe_with_timestamps(clip.samples,   16000, pk::Decoder::kDefault, lang);
        auto out = model->transcribe_pcm_batch_with_timestamps({speech.samples, clip.samples}, 16000,
                                                               pk::Decoder::kDefault, lang);
        bool pass = out.size() == 2 && out[0].text == rs.text && out[1].text == rc.text;
        std::fprintf(stderr, "(c) batched timestamps: speech=%s clip=%s -> %s\n",
                     (out.size() > 0 && out[0].text == rs.text) ? "OK" : "DIFF",
                     (out.size() > 1 && out[1].text == rc.text) ? "OK" : "DIFF",
                     pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }

    std::fprintf(stderr, "%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
