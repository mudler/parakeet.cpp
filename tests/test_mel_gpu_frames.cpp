// GpuMel's framing (pk::gpu_mel_frames) computes each preemphasized, zero-padded sample where it reads
// it. Its frame matrix must equal, bit for bit, the one GpuMel built before from two copies of the signal
// (preemphasized, then padded), for clips shorter than a hop, around hop and n_fft boundaries, and a
// second plus a few samples, with preemphasis on and off.
#include "mel_gpu.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

// The framing as GpuMel did it before: a double copy of the preemphasized signal, a padded copy of
// that, then the windowed frames.
static std::vector<float> two_copy_frames(const std::vector<float>& samples, float preemph,
                                          const std::vector<float>& window, int n_fft, int hop, int T) {
    const int S = (int)samples.size();
    std::vector<double> x((size_t)S);
    if (preemph > 0.0f && S > 0) {
        x[0] = samples[0];
        for (int t = 1; t < S; ++t) x[t] = (double)samples[t] - (double)preemph * (double)samples[t - 1];
    } else {
        for (int t = 0; t < S; ++t) x[t] = samples[t];
    }
    const int pad = n_fft / 2;
    std::vector<double> padded((size_t)S + 2 * pad, 0.0);
    for (int j = 0; j < S; ++j) padded[pad + j] = x[j];
    std::vector<float> Xw((size_t)T * n_fft);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n_fft; ++i)
            Xw[(size_t)t * n_fft + i] = (float)(padded[(size_t)t * hop + i] * (double)window[i]);
    return Xw;
}

static std::vector<float> noise(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = (float)((double)(seed >> 8) / (double)(1u << 24) - 0.5);
    }
    return v;
}

int main() {
    const int n_fft = 512, hop = 160;
    const std::vector<float> window = noise((size_t)n_fft, 99);
    int failures = 0, cases = 0;
    for (int S : {1, 2, 159, 160, 161, 255, 256, 257, 511, 512, 513, 16007})
        for (float preemph : {0.97f, 0.0f}) {
            const std::vector<float> samples = noise((size_t)S, (uint32_t)S);
            const int T = 1 + (S + 2 * (n_fft / 2) - n_fft) / hop;  // GpuMel::compute's frame count
            std::vector<float> got;
            pk::gpu_mel_frames(samples, preemph, window, n_fft, hop, T, got);
            ++cases;
            if (got != two_copy_frames(samples, preemph, window, n_fft, hop, T)) {
                std::fprintf(stderr, "FAIL: S=%d preemph=%.2f: frames differ\n", S, preemph);
                ++failures;
            }
        }
    if (failures) return 1;
    std::printf("ok: %d cases, frames equal the two-copy framing bit for bit\n", cases);
    return 0;
}
