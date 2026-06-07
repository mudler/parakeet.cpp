#include "mel_gpu.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>

namespace pk {

// CONSTANT epsilon added to std in NeMo normalize_batch (features.py: 1e-5).
// Kept identical to MelFrontend::compute so the per-feature norm matches.
static constexpr double kNormEps = 1e-5;

GpuMel::GpuMel(const ModelLoader& ml) : k_(ml) {
    // DFT basis matching pk::rfft's convention (no normalization), so the matmul
    // re/im equal the FFT's:  re[b] =  Σ_n x[n] cos(2π b n / N)
    //                         im[b] = -Σ_n x[n] sin(2π b n / N)
    // Row-major [n_bins, n_fft]: dft_cos_[b*n_fft + n], dft_sin_[b*n_fft + n].
    const int N = k_.n_fft_;
    const int B = k_.n_bins_;
    dft_cos_.assign((size_t)B * N, 0.0f);
    dft_sin_.assign((size_t)B * N, 0.0f);
    const double two_pi = 2.0 * M_PI;
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            const double ang = two_pi * (double)b * (double)n / (double)N;
            dft_cos_[(size_t)b * N + n] = (float)std::cos(ang);
            dft_sin_[(size_t)b * N + n] = (float)(-std::sin(ang));
        }
    }
}

void GpuMel::compute(const std::vector<float>& samples,
                     std::vector<float>& feats, int& n_mels, int& T) const {
    // Only mag_power == 2 is supported (power = re^2 + im^2) — true for all
    // supported models; the DFT-matmul graph squares re/im directly.
    assert(k_.mag_power_ == 2.0f && "GpuMel requires mag_power == 2");

    if (samples.empty()) { n_mels = k_.n_mels_; T = 0; feats.clear(); return; }

    const int S        = (int)samples.size();
    const int n_fft_   = k_.n_fft_;
    const int hop_     = k_.hop_;
    const int n_mels_  = k_.n_mels_;
    const int n_bins_  = k_.n_bins_;
    const float preemph_ = k_.preemph_;

    // ----- Preemphasis (double, exactly like MelFrontend) -----
    std::vector<double> x((size_t)S);
    if (preemph_ > 0.0f && S > 0) {
        x[0] = samples[0];
        for (int t = 1; t < S; ++t)
            x[t] = (double)samples[t] - (double)preemph_ * (double)samples[t - 1];
    } else {
        for (int t = 0; t < S; ++t) x[t] = samples[t];
    }

    const int seq_len = (S > 0) ? (S / hop_) : 0;

    // ----- Center zero-pad (constant), then frame count (identical to MelFrontend) -----
    const int pad = n_fft_ / 2;
    const int padded_len = S + 2 * pad;
    const int n_frames = (padded_len >= n_fft_) ? (1 + (padded_len - n_fft_) / hop_) : 0;
    T = n_frames;
    n_mels = n_mels_;

    std::vector<double> padded((size_t)padded_len, 0.0);
    for (int j = 0; j < S; ++j) padded[pad + j] = x[j];

    feats.assign((size_t)n_mels_ * T, 0.0f);
    if (T == 0) {
        // Mirror MelFrontend: per_feature norm over zero frames is a no-op.
        return;
    }

    // ----- Windowed frame matrix Xw [T, n_fft] row-major (row t = frame t) -----
    // Xw[t*n_fft + i] = (float)(padded[t*hop + i] * (double)window_[i]). Matches
    // MelFrontend's per-frame windowing exactly.
    std::vector<float> Xw((size_t)T * n_fft_);
    for (int t = 0; t < T; ++t) {
        const int start = t * hop_;
        float* row = &Xw[(size_t)t * n_fft_];
        for (int i = 0; i < n_fft_; ++i)
            row[i] = (float)(padded[(size_t)start + i] * (double)k_.window_[i]);
    }

    // ----- ggml graph: DFT-matmul -> power -> filterbank -> log -> feat-major -----
    const float guard = k_.log_guard_;
    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // Xw input: ne=[n_fft, T] (ne0=n_fft fastest, matches row-major Xw).
            int64_t xw_ne[2] = { n_fft_, T };
            ggml_tensor* xw = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xw_ne,
                                  Xw.data(), (size_t)T * n_fft_ * sizeof(float));
            // DFT basis: ne=[n_fft, n_bins] (row b over n).
            int64_t basis_ne[2] = { n_fft_, n_bins_ };
            ggml_tensor* cosb = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, basis_ne,
                                    dft_cos_.data(), (size_t)n_bins_ * n_fft_ * sizeof(float));
            ggml_tensor* sinb = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, basis_ne,
                                    dft_sin_.data(), (size_t)n_bins_ * n_fft_ * sizeof(float));

            // re[t][b] = Σ_n cos(b,n) * Xw[t][n]; im likewise. -> ne=[n_bins, T].
            ggml_tensor* re = ggml_mul_mat(ctx, cosb, xw);
            ggml_tensor* im = ggml_mul_mat(ctx, sinb, xw);

            // power = re^2 + im^2  (mag_power == 2). -> ne=[n_bins, T].
            ggml_tensor* power = ggml_add(ctx,
                                          ggml_mul(ctx, re, re),
                                          ggml_mul(ctx, im, im));

            // Filterbank: ne=[n_bins, n_mels] (row m = fb_[m*n_bins + b]).
            int64_t fb_ne[2] = { n_bins_, n_mels_ };
            ggml_tensor* fb = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, fb_ne,
                                  k_.fb_.data(), (size_t)n_mels_ * n_bins_ * sizeof(float));
            // mel[t][m] = Σ_b fb(m,b) * power[t][b]. -> ne=[n_mels, T].
            ggml_tensor* mel = ggml_mul_mat(ctx, fb, power);

            // log(mel + log_guard). add via a [1] broadcast tensor (ggml_add1 is
            // deprecated); ne all 1 -> ggml_can_repeat into [n_mels, T].
            int64_t g_ne[1] = { 1 };
            ggml_tensor* g = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, g_ne,
                                 &guard, sizeof(float));
            mel = ggml_add(ctx, mel, g);
            ggml_tensor* lm = ggml_log(ctx, mel);   // ne=[n_mels, T]

            // Transpose to feat-major: out (t,m) at m*T + t. -> ne=[T, n_mels].
            ggml_tensor* out_t = ggml_cont(ctx, ggml_transpose(ctx, lm));
            return out_t;
        }, feats);
    assert(ok && "GpuMel graph failed");
    (void)ok;

    // run_graph wrote n_mels_*T f32 in feat-major order (feats[m*T + t]).

    // ----- per-feature normalization (verbatim from MelFrontend::compute) -----
    if (k_.per_feature_ && T > 0) {
        const int valid = std::min(seq_len, T);
        for (int m = 0; m < n_mels_; ++m) {
            float* row = &feats[(size_t)m * T];
            double mean = 0.0;
            if (valid > 0) {
                for (int t = 0; t < valid; ++t) mean += row[t];
                mean /= (double)valid;
            }
            double var = 0.0;
            if (valid > 1) {
                for (int t = 0; t < valid; ++t) {
                    const double d = (double)row[t] - mean;
                    var += d * d;
                }
                var /= (double)(valid - 1); // unbiased (ddof=1)
            }
            double sd = std::sqrt(var);
            sd += kNormEps;
            for (int t = 0; t < T; ++t) {
                if (t < valid) row[t] = (float)(((double)row[t] - mean) / sd);
                else           row[t] = 0.0f; // masked beyond seq_len
            }
        }
    }
}

} // namespace pk
