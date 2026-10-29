#pragma once
// Shared by the depthwise conv tests: the inputs, NeMo's depthwise conv computed on the host, and one run
// of the production builder (pk::build_depthwise_conv) on a given Backend.
#include "backend.hpp"
#include "conformer.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dwtest {

struct Case {
    int K, T, C, B;
    bool causal;
};

// Deterministic values in [-1, 1) from a seeded LCG, so a failure reproduces.
inline std::vector<float> fill(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = (float)((double)(seed >> 8) / (double)(1u << 24) * 2.0 - 1.0);
    }
    return v;
}

// NeMo's Conv1d(groups = C) in double: y[b][c][t] = sum_k x[b][c][t + k - left] * w[c][k], with zeros
// outside [0, T) and left = K - 1 (causal) or (K - 1) / 2. Layouts are ggml's: x [T, C, B] with t
// fastest, w [K, 1, C], and y [C, T, B] with c fastest.
inline std::vector<float> reference(const Case& c, const std::vector<float>& x,
                                    const std::vector<float>& w) {
    const int left = c.causal ? c.K - 1 : (c.K - 1) / 2;
    std::vector<float> y((size_t)c.C * c.T * c.B);
    for (int b = 0; b < c.B; ++b)
        for (int ch = 0; ch < c.C; ++ch)
            for (int t = 0; t < c.T; ++t) {
                double s = 0.0;
                for (int k = 0; k < c.K; ++k) {
                    const int src = t + k - left;
                    if (src < 0 || src >= c.T) continue;
                    s += (double)x[((size_t)b * c.C + ch) * c.T + src] * (double)w[(size_t)ch * c.K + k];
                }
                y[((size_t)b * c.T + t) * c.C + ch] = (float)s;
            }
    return y;
}

// Builds the conv with `path` on `be` and computes it. `op` is the op of the tensor the builder
// returned: GGML_OP_SSM_CONV for the SSM_CONV path, GGML_OP_CONT (the final transpose) for im2col.
inline bool run(pk::Backend& be, const Case& c, const std::vector<float>& x,
                const std::vector<float>& w, pk::DepthwiseConvPath path,
                std::vector<float>& y, ggml_op& op) {
    op = GGML_OP_NONE;
    return be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t x_ne[3] = {c.T, c.C, c.B};
        const int64_t w_ne[3] = {c.K, 1, c.C};
        ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, x_ne, x.data(),
                                                 x.size() * sizeof(float));
        ggml_tensor* wt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, w_ne, w.data(),
                                                 w.size() * sizeof(float));
        ggml_tensor* out = pk::build_depthwise_conv(ctx, xt, wt, c.K, c.causal, path);
        op = out->op;
        return out;
    }, y);
}

// Largest |got - ref|, or infinity on a size mismatch.
inline double max_abs_diff(const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size()) return INFINITY;
    double m = 0.0;
    for (size_t i = 0; i < got.size(); ++i) m = std::fmax(m, std::fabs((double)got[i] - (double)ref[i]));
    return m;
}

inline const char* name(const Case& c) {
    static char buf[96];
    std::snprintf(buf, sizeof(buf), "%s K=%d T=%d C=%d B=%d", c.causal ? "causal" : "symmetric",
                  c.K, c.T, c.C, c.B);
    return buf;
}

} // namespace dwtest
