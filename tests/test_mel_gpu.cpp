// Validates the ggml DFT-matmul GpuMel against the FFT-based MelFrontend. On
// this CPU box both run on the CPU backend — the point is to check the GpuMel
// graph math (DFT-as-matmul, power, filterbank, log, transpose, per-feature
// norm) reproduces MelFrontend's output to within a loose f32 tolerance.
//
// With --gpu the graph runs on the first GPU device instead, as production runs GpuMel, and the test
// exits 77 without one (ctest's test_mel_gpu_device).
#include "gpu_devices.hpp"
#include "ggml_graph.hpp"
#include "mel.hpp"
#include "mel_gpu.hpp"
#include "model_loader.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "PARAKEET_TEST_GGUF not set; skip\n"); return 77; }
    std::string want_device;
    if (argc > 1 && std::strcmp(argv[1], "--gpu") == 0) {
        const std::vector<std::string> devices = gputest::device_names();
        if (int rc = gputest::skip_without_gpu(devices, "test_mel_gpu --gpu")) return rc;
        want_device = devices[0];
        setenv("PARAKEET_DEVICE", want_device.c_str(), 1);
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    // As Model::load does. On a GPU, realizing the weights frees the host tensors, and MelKernel holds
    // one across the weight read that would realize them.
    pk::ensure_weights_realized(ml);

    // Deterministic 4 s test signal: a fixed mix of sines + seeded LCG noise.
    const int S = 16000 * 4;
    std::vector<float> samples((size_t)S);
    uint32_t seed = 0x12345678u;
    for (int i = 0; i < S; ++i) {
        const double t = (double)i / 16000.0;
        double v = 0.6 * std::sin(2.0 * M_PI * 220.0 * t)
                 + 0.3 * std::sin(2.0 * M_PI * 740.0 * t)
                 + 0.1 * std::sin(2.0 * M_PI * 3100.0 * t);
        seed = seed * 1664525u + 1013904223u;
        const double noise = ((double)(seed >> 9) / (double)(1u << 23)) - 0.5; // [-0.5,0.5)
        v += 0.05 * noise;
        samples[i] = (float)v;
    }

    pk::MelFrontend mel(ml);
    std::vector<float> feats_cpu; int n_mels_c = 0, T_c = 0;
    mel.compute(samples, feats_cpu, n_mels_c, T_c);

    pk::GpuMel gmel(ml);
    std::vector<float> feats_gpu; int n_mels_g = 0, T_g = 0;
    gmel.compute(samples, feats_gpu, n_mels_g, T_g);

    if (n_mels_c != n_mels_g || T_c != T_g) {
        std::fprintf(stderr, "shape mismatch: cpu (%d,%d) vs gpu (%d,%d)\n",
                     n_mels_c, T_c, n_mels_g, T_g);
        return 1;
    }
    if (feats_cpu.size() != feats_gpu.size()) {
        std::fprintf(stderr, "size mismatch: %zu vs %zu\n",
                     feats_cpu.size(), feats_gpu.size());
        return 1;
    }

    double max_abs = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < feats_cpu.size(); ++i) {
        const double d = std::fabs((double)feats_cpu[i] - (double)feats_gpu[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    const double mean_abs = feats_cpu.empty() ? 0.0 : sum_abs / (double)feats_cpu.size();

    std::fprintf(stderr, "GpuMel on %s vs MelFrontend: shape (%d,%d) max_abs_diff=%.6g mean_abs_diff=%.6g\n",
                 pk::global_backend().device_name(), n_mels_c, T_c, max_abs, mean_abs);

    if (!want_device.empty() && want_device != pk::global_backend().device_name()) {
        std::fprintf(stderr, "FAIL: GpuMel ran on %s, not %s\n", pk::global_backend().device_name(),
                     want_device.c_str());
        return 1;
    }
    if (max_abs >= 0.05) {
        std::fprintf(stderr, "FAIL: max_abs_diff %.6g >= 0.05\n", max_abs);
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
