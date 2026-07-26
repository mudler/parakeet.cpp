#include "model.hpp"
#include "transcription_json.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    pk::NBestTranscription first;
    first.text = "hello";
    first.score = -2.0f;
    first.normalized_score = -1.0f;
    first.tokens.push_back(pk::TdtBeamToken{3, 2, 4});

    const std::string got = pk::nbest_transcriptions_to_json(
        std::vector<pk::NBestTranscription>{first}, 4, true, 0.08f);
    const std::string expected =
        "{\"beam_size\":4,\"score_norm\":true,\"frame_sec\":0.080000,"
        "\"hypotheses\":[{\"text\":\"hello\",\"score\":-2.000000,"
        "\"normalized_score\":-1.000000,\"tokens\":[{\"id\":3,\"frame\":2,"
        "\"t\":0.160,\"duration_frames\":4,\"duration\":0.320}]}]}";
    if (got != expected) {
        std::fprintf(stderr, "test_nbest_json: mismatch\n  got: %s\n  exp: %s\n",
                     got.c_str(), expected.c_str());
        return 1;
    }
    return 0;
}
