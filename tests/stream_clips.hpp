#pragma once
#include <vector>

namespace pktest {

// Build the two-utterance clip used by the streaming event tests: the input
// clip + 0.6 s of silence + the input clip again, at 16 kHz. This is the same
// construction (and the same default gap) as scripts/gen_stream_reset_baseline.py,
// so a mid-stream <EOU> fires between the two utterances.
inline std::vector<float> two_utterance_clip(const std::vector<float>& samples) {
    std::vector<float> clip(samples);
    clip.insert(clip.end(), (size_t)(0.6f * 16000.0f), 0.0f);
    clip.insert(clip.end(), samples.begin(), samples.end());
    return clip;
}

} // namespace pktest
