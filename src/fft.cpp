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

// Precomputed radix-2 Cooley-Tukey plan for a FIXED length n (a power of two).
// The twiddle sequences are generated with the exact iterative complex-multiply
// used by the original per-frame FFT, so the transform is bit-identical to it,
// but the cos/sin-per-stage and per-butterfly twiddle advance are paid once at
// plan construction instead of on every frame.
class FftPlan {
public:
    explicit FftPlan(int n) : n_(n) {
        assert(n > 0 && (n & (n - 1)) == 0 && "n must be a power of 2");
        // Bit-reversal permutation.
        rev_.resize((size_t)n);
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            rev_[i] = j;
        }
        // Per-stage twiddle sequences, generated bit-identically to the original
        // loop (start at (1,0) and advance by complex-multiplying cos/sin(-2π/len)).
        stage_off_.push_back(0);
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * M_PI / len;
            double wr = std::cos(ang);
            double wi = std::sin(ang);
            double cur_wr = 1.0, cur_wi = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                tw_r_.push_back(cur_wr);
                tw_i_.push_back(cur_wi);
                double new_wr = cur_wr * wr - cur_wi * wi;
                double new_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = new_wr;
                cur_wi = new_wi;
            }
            stage_off_.push_back((int)tw_r_.size());
        }
    }

    // Transform `in` into the full n complex bins, keeping the entire transform
    // in double (bit-identical to the original per-frame FFT); copies bins
    // 0..n_bins-1 into the float outputs.
    void apply(const float* in, std::vector<float>& re, std::vector<float>& im) const {
        const int n = n_;
        d_re_.resize((size_t)n);
        d_im_.resize((size_t)n);
        for (int i = 0; i < n; ++i) {
            d_re_[(size_t)rev_[i]] = (double)in[i];
            d_im_[(size_t)rev_[i]] = 0.0;
        }
        const double* tw_r = tw_r_.data();
        const double* tw_i = tw_i_.data();
        double* re0 = d_re_.data();
        double* im0 = d_im_.data();
        int st = 0;
        for (int len = 2; len <= n; len <<= 1, ++st) {
            const int half = len >> 1;
            const double* wbase_r = tw_r + stage_off_[st];
            const double* wbase_i = tw_i + stage_off_[st];
            for (int i = 0; i < n; i += len) {
                const double* wr = wbase_r;
                const double* wi = wbase_i;
                for (int k = 0; k < half; ++k) {
                    const int u = i + k;
                    const int v = u + half;
                    const double tr = wr[k] * re0[v] - wi[k] * im0[v];
                    const double ti = wr[k] * im0[v] + wi[k] * re0[v];
                    re0[v] = re0[u] - tr;
                    im0[v] = im0[u] - ti;
                    re0[u] = re0[u] + tr;
                    im0[u] = im0[u] + ti;
                }
            }
        }
        const int n_bins = n / 2 + 1;
        re.resize((size_t)n_bins);
        im.resize((size_t)n_bins);
        for (int b = 0; b < n_bins; ++b) {
            re[(size_t)b] = (float)d_re_[b];
            im[(size_t)b] = (float)d_im_[b];
        }
    }

    int n() const { return n_; }

private:
    int n_;
    std::vector<int> rev_;
    std::vector<double> tw_r_, tw_i_;
    std::vector<int> stage_off_;
    mutable std::vector<double> d_re_, d_im_;  // reusable scratch (no per-call alloc)
};

void rfft(const std::vector<float>& in, std::vector<float>& re, std::vector<float>& im) {
    const int n = static_cast<int>(in.size());
    assert(n > 0 && (n & (n - 1)) == 0);

    // Cache one plan per distinct n (n is fixed for a model; reuse across frames).
    // thread_local so concurrent rfft callers (e.g. a multi-request server) never
    // race on the reusable double scratch; the plan is rebuilt once per thread.
    static thread_local FftPlan plan(n);

    plan.apply(in.data(), re, im);
}

} // namespace pk
