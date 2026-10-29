// GpuMel's DFT basis is built once per size and shared (pk::dft_basis). For each size the shared basis
// must hold exactly what GpuMel's constructor computed before it was shared; the same size must return
// the same object, another size its own; and threads asking at once must all get one object.
#include "mel_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

// The basis as GpuMel's constructor built it per instance, the same double arithmetic.
static void per_instance_basis(int N, int B, std::vector<float>& cos_b, std::vector<float>& sin_b) {
    cos_b.assign((size_t)B * N, 0.0f);
    sin_b.assign((size_t)B * N, 0.0f);
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n) {
            const double ang = 2.0 * M_PI * (double)b * (double)n / (double)N;
            cos_b[(size_t)b * N + n] = (float)std::cos(ang);
            sin_b[(size_t)b * N + n] = (float)(-std::sin(ang));
        }
}

int main() {
    struct Size { int n_fft, n_bins; };
    // Three sizes; the first and the last share n_fft, so a cache keyed on n_fft alone would fail.
    const Size sizes[] = {{512, 257}, {400, 201}, {512, 129}};
    std::shared_ptr<const pk::DftBasis> first[3];
    for (int i = 0; i < 3; ++i) {
        first[i] = pk::dft_basis(sizes[i].n_fft, sizes[i].n_bins);
        std::vector<float> c, s;
        per_instance_basis(sizes[i].n_fft, sizes[i].n_bins, c, s);
        CHECK(first[i]->cos == c);
        CHECK(first[i]->sin == s);
    }
    for (int i = 0; i < 3; ++i) CHECK(pk::dft_basis(sizes[i].n_fft, sizes[i].n_bins) == first[i]);
    CHECK(first[0] != first[2]);

    std::vector<std::shared_ptr<const pk::DftBasis>> got(8);
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back([i, &got]() { got[i] = pk::dft_basis(256, 129); });
    for (std::thread& t : threads) t.join();
    for (int i = 1; i < 8; ++i) CHECK(got[i] == got[0]);

    if (failures) return 1;
    std::printf("ok: one basis per size, equal to the per-instance one\n");
    return 0;
}
