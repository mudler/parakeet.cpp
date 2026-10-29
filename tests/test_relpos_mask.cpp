// The single-clip attention mask (pk::relpos_attention_mask) and the softmax that runs without it.
//   1. Over padded, unpadded, chunked-limited and sliding-window cases, the mask equals NeMo's mask
//      written out here, and it is skipped (null) only where that mask has no -inf entry. A full-context
//      clip without padding must skip it: that case builds T*T zeros per layer otherwise.
//   2. On the CPU backend, softmax with an all-zero mask and softmax without a mask give the same bits,
//      so skipping the mask leaves the attention unchanged. test_relpos_attention_gpu checks GPUs.
#include "relpos_attention.hpp"
#include "backend.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

// NeMo's mask (ConformerEncoder._create_masks and rel_pos_local_attn), as build_graph applies it.
static std::vector<float> nemo_mask(int T, int valid, int chunk, int left_chunks, int att_left, int att_right) {
    std::vector<float> m((size_t)T * T, 0.0f);
    for (int q = 0; q < valid && q < T; ++q)
        for (int k = 0; k < T; ++k) {
            bool attend = k < valid;
            if (chunk > 0) {
                const int back = q / chunk - k / chunk;
                attend = attend && back >= 0 && back <= left_chunks;
            }
            if (att_left >= 0) attend = attend && q - k <= att_left && k - q <= att_right;
            if (!attend) m[(size_t)q * T + k] = -INFINITY;
        }
    return m;
}

static void check_masks() {
    struct Chunk { int size, left; };
    struct Window { int left, right; };
    int cases = 0, skipped = 0;
    for (int T : {1, 2, 3, 8, 17})
        for (int valid : {T, T - 1, T / 2, 1}) {
            if (valid < 1) continue;
            for (Chunk ch : {Chunk{0, 0}, Chunk{1, 0}, Chunk{2, 1}, Chunk{4, 2}})
                for (Window w : {Window{-1, -1}, Window{0, 0}, Window{1, 2}, Window{3, 3}, Window{T, T}}) {
                    pk::GraphInputPool pool;
                    const std::vector<float>* got =
                        pk::relpos_attention_mask(T, valid, ch.size, ch.left, w.left, w.right, pool);
                    const std::vector<float> want = nemo_mask(T, valid, ch.size, ch.left, w.left, w.right);
                    bool any_inf = false;
                    for (float v : want) any_inf = any_inf || std::isinf(v);
                    ++cases;
                    if (!got) {
                        ++skipped;
                        if (any_inf)
                            std::fprintf(stderr, "FAIL: mask skipped though it masks keys: T=%d valid=%d "
                                         "chunk=%d/%d window=%d/%d\n", T, valid, ch.size, ch.left, w.left, w.right);
                        failures += any_inf;
                    } else if (*got != want) {
                        std::fprintf(stderr, "FAIL: mask differs from NeMo's: T=%d valid=%d chunk=%d/%d window=%d/%d\n",
                                     T, valid, ch.size, ch.left, w.left, w.right);
                        ++failures;
                    }
                    if (valid == T && ch.size == 0 && w.left < 0) CHECK(got == nullptr);
                }
        }
    std::printf("masks: %d cases, %d skipped\n", cases, skipped);
}

static void check_softmax() {
    const int T = 37, H = 3;
    std::vector<float> scores((size_t)T * T * H), zeros((size_t)T * T, 0.0f);
    uint32_t seed = 12345;
    for (float& x : scores) {
        seed = seed * 1664525u + 1013904223u;
        x = (float)((double)(seed >> 8) / (double)(1u << 24) * 20.0 - 10.0);
    }
    pk::Backend cpu(4, /*cpu_only=*/true);
    auto softmax = [&](bool with_mask, std::vector<float>& out) {
        return cpu.compute([&](ggml_context* ctx) -> ggml_tensor* {
            const int64_t s_ne[3] = {T, T, H}, m_ne[2] = {T, T};
            ggml_tensor* s = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, s_ne, scores.data(),
                                                    scores.size() * sizeof(float));
            ggml_tensor* m = with_mask ? pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, m_ne, zeros.data(),
                                                                zeros.size() * sizeof(float))
                                       : nullptr;
            return ggml_soft_max_ext(ctx, s, m, 0.125f, 0.0f);
        }, out);
    };
    std::vector<float> masked, unmasked;
    CHECK(softmax(true, masked));
    CHECK(softmax(false, unmasked));
    CHECK(masked.size() == unmasked.size() &&
          std::memcmp(masked.data(), unmasked.data(), masked.size() * sizeof(float)) == 0);
}

int main() {
    check_masks();
    check_softmax();
    if (failures) return 1;
    std::printf("ok: the mask matches NeMo's and is skipped only when it masks nothing\n");
    return 0;
}
