#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "audio_io.hpp"
#include "common.hpp"
#include <cmath>

namespace pk {

std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    const double ratio = (double)out_sr / (double)in_sr;
    const size_t n_out = (size_t)std::floor(in.size() * ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double src = i / ratio;
        const size_t i0 = (size_t)src;
        const double frac = src - i0;
        const float a = in[i0];
        const float b = (i0 + 1 < in.size()) ? in[i0 + 1] : a;
        out[i] = (float)(a + (b - a) * frac);
    }
    return out;
}

namespace {

bool decode_pcm_16k_mono(float* pcm, unsigned int ch, unsigned int sr,
                         drwav_uint64 frames, Audio& out, const char* source) {
    if (!pcm) {
        PK_LOG("failed to open wav: %s", source);
        return false;
    }
    if (ch == 0 || sr == 0) {
        PK_LOG("invalid wav metadata: %s", source);
        return false;
    }
    std::vector<float> mono(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        double acc = 0; for (unsigned int c = 0; c < ch; ++c) acc += pcm[i*ch + c];
        mono[i] = (float)(acc / ch);
    }
    out.samples = resample_linear(mono, (int)sr, 16000);
    out.sample_rate = 16000;
    return true;
}

} // namespace

bool load_audio_16k_mono(const std::string& path, Audio& out) {
    unsigned int ch = 0, sr = 0; drwav_uint64 frames = 0;
    float* pcm = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &frames, nullptr);
    const bool ok = decode_pcm_16k_mono(pcm, ch, sr, frames, out, path.c_str());
    if (pcm) drwav_free(pcm, nullptr);
    return ok;
}

bool load_audio_16k_mono_from_memory(const void* data, size_t size, Audio& out) {
    if (!data || size == 0) {
        PK_LOG("empty wav input");
        return false;
    }
    unsigned int ch = 0, sr = 0; drwav_uint64 frames = 0;
    float* pcm = drwav_open_memory_and_read_pcm_frames_f32(data, size, &ch, &sr, &frames, nullptr);
    const bool ok = decode_pcm_16k_mono(pcm, ch, sr, frames, out, "memory");
    if (pcm) drwav_free(pcm, nullptr);
    return ok;
}

} // namespace pk
