#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
namespace pk {
struct Audio {
    std::vector<float> samples; // mono, [-1,1]
    int sample_rate = 0;
};
// Loads any WAV, downmixes to mono, linearly resamples to 16 kHz. Returns false on error.
bool load_audio_16k_mono(const std::string& path, Audio& out);
// Loads WAV bytes from memory, downmixes to mono, linearly resamples to 16 kHz.
bool load_audio_16k_mono_from_memory(const void* data, size_t size, Audio& out);
// Resample mono float PCM from in_sr to out_sr (linear). Exposed for reuse/testing.
std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr);
}
