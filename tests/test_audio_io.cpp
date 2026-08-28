#include "audio_io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>
#include <cstdint>

// dr_wav writer is only needed in the test; include without implementation
// (DR_WAV_IMPLEMENTATION lives in audio_io.cpp, linked via parakeet)
#include "dr_wav.h"

static void write_sine(const char* path, int sr, int n, float freq) {
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) s[i] = 0.25f * std::sin(2.0*M_PI*freq*i/sr);
    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = 1; fmt.sampleRate = (drwav_uint32)sr; fmt.bitsPerSample = 32;
    drwav w; drwav_init_file_write(&w, path, &fmt, nullptr);
    drwav_write_pcm_frames(&w, n, s.data());
    drwav_uninit(&w);
}

int main() {
    const char* path = "/tmp/pk_test_44k.wav";
    write_sine(path, 44100, 44100, 440.0f); // 1s @ 44.1k

    pk::Audio a;
    if (!pk::load_audio_16k_mono(path, a)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (a.sample_rate != 16000) { std::fprintf(stderr, "sr=%d\n", a.sample_rate); return 1; }
    // ~1s of audio resampled to 16k → ~16000 samples (allow small edge slack)
    if (a.samples.size() < 15800 || a.samples.size() > 16200) {
        std::fprintf(stderr, "n=%zu\n", a.samples.size()); return 1;
    }
    // not silent
    double e = 0; for (float v : a.samples) e += (double)v*v;
    if (e < 1.0) { std::fprintf(stderr, "energy too low %f\n", e); return 1; }

    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
    if (bytes.empty()) { std::fprintf(stderr, "readback failed\n"); return 1; }
    pk::Audio mem;
    if (!pk::load_audio_16k_mono_from_memory(bytes.data(), bytes.size(), mem)) {
        std::fprintf(stderr, "memory load failed\n"); return 1;
    }
    if (mem.sample_rate != a.sample_rate || mem.samples.size() != a.samples.size()) {
        std::fprintf(stderr, "memory mismatch: %zu/%d vs %zu/%d\n",
                     mem.samples.size(), mem.sample_rate, a.samples.size(), a.sample_rate);
        return 1;
    }
    double max_delta = 0.0;
    for (size_t i = 0; i < a.samples.size(); ++i)
        max_delta = std::max(max_delta, std::fabs((double)a.samples[i] - (double)mem.samples[i]));
    if (max_delta > 1e-6) {
        std::fprintf(stderr, "memory sample mismatch: max_delta=%g\n", max_delta);
        return 1;
    }
    std::printf("audio_io ok: %zu samples @ %d Hz\n", a.samples.size(), a.sample_rate);
    return 0;
}
