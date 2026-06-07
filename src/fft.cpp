#include "fft.hpp"
#include <cassert>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdint>
#include <vector>

namespace pk {

// Iterative radix-2 Cooley-Tukey FFT (in-place, complex double buffer).
// n must be a power of 2.
static void fft_inplace(std::vector<double>& re, std::vector<double>& im) {
    const int n = static_cast<int>(re.size());
    assert(n > 0 && (n & (n - 1)) == 0); // must be power of 2

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wr = std::cos(ang);
        double wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double cur_wr = 1.0, cur_wi = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                int u = i + k;
                int v = i + k + len / 2;
                double tr = cur_wr * re[v] - cur_wi * im[v];
                double ti = cur_wr * im[v] + cur_wi * re[v];
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] = re[u] + tr;
                im[u] = im[u] + ti;
                // advance twiddle factor
                double new_wr = cur_wr * wr - cur_wi * wi;
                double new_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = new_wr;
                cur_wi = new_wi;
            }
        }
    }
}

void rfft(const std::vector<float>& in, std::vector<float>& re, std::vector<float>& im) {
    const int n = static_cast<int>(in.size());
    assert(n > 0 && (n & (n - 1)) == 0);

    // Build complex buffer from real input (imag = 0)
    std::vector<double> buf_re(n), buf_im(n);
    for (int i = 0; i < n; ++i) {
        buf_re[i] = static_cast<double>(in[i]);
        buf_im[i] = 0.0;
    }

    fft_inplace(buf_re, buf_im);

    // Copy bins 0..n/2 into output (n/2 + 1 bins)
    const int n_bins = n / 2 + 1;
    re.resize(n_bins);
    im.resize(n_bins);
    for (int k = 0; k < n_bins; ++k) {
        re[k] = static_cast<float>(buf_re[k]);
        im[k] = static_cast<float>(buf_im[k]);
    }
}

} // namespace pk
