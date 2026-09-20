#pragma once
#include <vector>

namespace pktest {

// Build a continuous multi-utterance session at 16 kHz: the input clip repeated
// `repeats` times, each pass followed by `gap_s` of silence so an <EOU> fires
// between utterances. The default gap matches
// scripts/gen_stream_reset_baseline.py.
//
// The long-session regressions drive this shape because the accumulated
// streaming buffers are NOT reset on <EOU> -- which is exactly what made a long
// dictation degrade to O(N^2).
inline std::vector<float> repeated_session_clip(const std::vector<float>& samples,
                                                int repeats, float gap_s = 0.6f) {
    const std::vector<float> gap((size_t)(gap_s * 16000.0f), 0.0f);
    std::vector<float> pcm;
    pcm.reserve((samples.size() + gap.size()) * (size_t)(repeats > 0 ? repeats : 0));
    for (int i = 0; i < repeats; ++i) {
        pcm.insert(pcm.end(), samples.begin(), samples.end());
        pcm.insert(pcm.end(), gap.begin(), gap.end());
    }
    return pcm;
}

// The two-utterance clip used by the streaming event tests: clip + 0.6 s of
// silence + clip, with no trailing gap.
inline std::vector<float> two_utterance_clip(const std::vector<float>& samples) {
    std::vector<float> clip = repeated_session_clip(samples, 1);
    clip.insert(clip.end(), samples.begin(), samples.end());
    return clip;
}

} // namespace pktest
