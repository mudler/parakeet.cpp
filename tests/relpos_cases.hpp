#pragma once
// Shared by test_rel_shift and test_relpos_attention_gpu: NeMo's rel_shift on the host, and the graph
// build_graph runs on its result (scores = ac + rel_shift(bd)).
#include "backend.hpp"
#include "relpos_attention.hpp"
#include "ggml.h"

#include <cstdint>
#include <vector>

namespace rstest {

inline std::vector<float> fill(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = (float)((double)(seed >> 8) / (double)(1u << 24) * 2.0 - 1.0);
    }
    return v;
}

// NeMo's RelPositionMultiHeadAttention.rel_shift, then the slice to T keys, on bd [P = 2T-1, T, H]
// (ggml layout, position fastest). Per head and query row: pad one zero in front of the row, view the
// [T, P+1] block as [P+1, T], drop its first row, view the rest as [T, P] and keep the first T columns.
// Returns [T(key), T(query), H], key fastest.
inline std::vector<float> nemo_rel_shift(const std::vector<float>& bd, int T, int H) {
    const int P = 2 * T - 1;
    std::vector<float> out((size_t)T * T * H);
    std::vector<float> padded((size_t)T * (P + 1));
    for (int h = 0; h < H; ++h) {
        for (int q = 0; q < T; ++q) {
            padded[(size_t)q * (P + 1)] = 0.0f;
            for (int p = 0; p < P; ++p)
                padded[(size_t)q * (P + 1) + 1 + p] = bd[((size_t)h * T + q) * P + p];
        }
        const float* shifted = padded.data() + T;  // drop the first row of the [P+1, T] view
        for (int q = 0; q < T; ++q)
            for (int j = 0; j < T; ++j) out[((size_t)h * T + q) * T + j] = shifted[(size_t)q * P + j];
    }
    return out;
}

// ac + rel_shift(bd) on `be`, as build_graph computes the attention scores.
inline bool scores(pk::Backend& be, const std::vector<float>& ac, const std::vector<float>& bd, int T,
                   int H, std::vector<float>& out) {
    return be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ac_ne[3] = {T, T, H}, bd_ne[3] = {2 * T - 1, T, H};
        ggml_tensor* a = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ac_ne, ac.data(), ac.size() * sizeof(float));
        ggml_tensor* b = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, bd_ne, bd.data(), bd.size() * sizeof(float));
        return ggml_add(ctx, a, pk::rel_shift(ctx, b));
    }, out);
}

} // namespace rstest
