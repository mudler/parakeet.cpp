#include "fft.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

// Run rfft on a pure cosine at bin k of length N and check the magnitude peak
// lands at bin k. Returns 0 on success, else the (wrong) peak bin.
static int check_fft(int N, int k) {
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = std::cos(2.0 * M_PI * k * i / N);
    std::vector<float> re(N / 2 + 1), im(N / 2 + 1);
    pk::rfft(x, re, im);
    int peak = 0;
    double best = -1;
    for (int b = 0; b <= N / 2; ++b) {
        double m = (double)re[b] * re[b] + (double)im[b] * im[b];
        if (m > best) { best = m; peak = b; }
    }
    return peak == k ? 0 : peak;
}

int main() {
    // rfft caches a per-length FftPlan thread-locally. Call it with several
    // different power-of-two lengths on the SAME thread, in both orders, so a
    // plan keyed only to the first-seen length would produce a wrong peak (or
    // read past a shorter input) and fail here. Regression for the rfft plan
    // cache being keyed by n.
    if (int p = check_fft(512, 8))  { std::fprintf(stderr, "512@8 peak=%d\n", p); return 1; }
    if (int p = check_fft(256, 4))  { std::fprintf(stderr, "256@4 peak=%d\n", p); return 1; }
    if (int p = check_fft(512, 16)) { std::fprintf(stderr, "512@16 peak=%d\n", p); return 1; }
    if (int p = check_fft(128, 3))  { std::fprintf(stderr, "128@3 peak=%d\n", p); return 1; }
    if (int p = check_fft(1024, 20)){ std::fprintf(stderr, "1024@20 peak=%d\n", p); return 1; }
    if (int p = check_fft(256, 7))  { std::fprintf(stderr, "256@7 peak=%d\n", p); return 1; }

    std::printf("fft ok: multi-length, both orders\n");
    return 0;
}
