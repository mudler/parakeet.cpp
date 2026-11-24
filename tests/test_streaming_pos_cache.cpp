// CPU-only, no checkpoint. StreamingEncoder precomputes linear_pos(pos_emb) once per
// window length and feeds it to the chunk graph. This test checks, on a tiny synthetic
// streaming model that holds only the linear_pos weights:
//   1. the cached projection is bit-identical to the ggml ops the per-step graph used
//      before the cache (mul_mat, reshape to heads, permute) run on the same inputs,
//   2. it matches a double-precision host computation,
//   3. a second call returns the same data (cache hit) and the cache stays bounded with
//      least-recently-used eviction, and the entry of the evicted length is recomputed
//      to the same values.
#include "streaming_encoder.hpp"
#include "model_loader.hpp"
#include "pos_enc.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

namespace {

constexpr int kD = 16, kHeads = 2, kLayers = 3, kDk = kD / kHeads;

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float)((s >> 40) & 0xffffff) / 8388608.0f - 1.0f;
    }
};

bool write_model(const std::string& path, std::vector<std::vector<float>>& w) {
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "parakeet");
    gguf_set_val_str(g, "parakeet.arch", "rnnt");
    gguf_set_val_u32(g, "parakeet.encoder.feat_in", 8);
    gguf_set_val_u32(g, "parakeet.encoder.d_model", kD);
    gguf_set_val_u32(g, "parakeet.encoder.n_layers", kLayers);
    gguf_set_val_u32(g, "parakeet.encoder.n_heads", kHeads);
    gguf_set_val_u32(g, "parakeet.encoder.ff_dim", 32);
    gguf_set_val_u32(g, "parakeet.encoder.conv_kernel", 9);
    gguf_set_val_u32(g, "parakeet.encoder.subsampling_factor", 8);
    gguf_set_val_bool(g, "parakeet.encoder.causal_downsampling", true);
    gguf_set_val_bool(g, "parakeet.encoder.conv_causal", true);
    gguf_set_val_str(g, "parakeet.encoder.att_context_style", "chunked_limited");
    gguf_set_val_i32(g, "parakeet.encoder.att_context_left", 70);
    gguf_set_val_i32(g, "parakeet.encoder.att_context_right", 13);
    gguf_set_val_u32(g, "parakeet.preprocessor.n_mels", 8);
    gguf_set_val_u32(g, "parakeet.vocab_size", 8);
    const int32_t two[2] = {9, 16}, shift[2] = {9, 16}, pre[2] = {0, 9};
    gguf_set_arr_data(g, "parakeet.streaming.chunk_size", GGUF_TYPE_INT32, two, 2);
    gguf_set_arr_data(g, "parakeet.streaming.shift_size", GGUF_TYPE_INT32, shift, 2);
    gguf_set_arr_data(g, "parakeet.streaming.pre_encode_cache_size", GGUF_TYPE_INT32, pre, 2);
    gguf_set_val_i32(g, "parakeet.streaming.last_channel_cache_size", 70);
    gguf_set_val_i32(g, "parakeet.streaming.valid_out_len", 2);
    gguf_set_val_i32(g, "parakeet.streaming.drop_extra_pre_encoded", 2);

    ggml_init_params ip{(size_t)1 << 20, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    Rng r;
    w.assign(kLayers, {});
    for (int l = 0; l < kLayers; ++l) {
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kD, kD);   // [in, out]
        ggml_set_name(t, ("encoder.layers." + std::to_string(l) + ".self_attn.linear_pos.weight").c_str());
        float* d = (float*)t->data;
        for (int i = 0; i < kD * kD; ++i) d[i] = 0.5f * r.next();
        w[l].assign(d, d + kD * kD);
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// The ops the per-step graph ran before the cache existed, on the same inputs.
std::vector<float> old_per_step(const ModelLoader& ml, int layer, int window) {
    std::vector<float> pe;
    rel_pos_encoding(window, kD, pe);
    const int P = 2 * window - 1;
    std::vector<float> out;
    const std::string wn = "encoder.layers." + std::to_string(layer) + ".self_attn.linear_pos.weight";
    bool ok = run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ne[2] = {kD, P};
        ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne, pe.data(), pe.size() * sizeof(float));
        ggml_tensor* p = ggml_mul_mat(ctx, clone_weight(ctx, ml, wn.c_str()), x);
        p = ggml_reshape_3d(ctx, p, kDk, kHeads, P);
        return ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));
    }, out);
    CHECK(ok);
    return out;
}

// Double precision host reference, layout [dk, P, H] with dk fastest.
std::vector<double> host_ref(const std::vector<float>& w, int window) {
    std::vector<float> pe;
    rel_pos_encoding(window, kD, pe);
    const int P = 2 * window - 1;
    std::vector<double> out((size_t)kDk * P * kHeads);
    for (int p = 0; p < P; ++p)
        for (int o = 0; o < kD; ++o) {
            double acc = 0;
            for (int i = 0; i < kD; ++i) acc += (double)w[(size_t)o * kD + i] * pe[(size_t)p * kD + i];
            const int h = o / kDk, d = o % kDk;
            out[((size_t)h * P + p) * kDk + d] = acc;
        }
    return out;
}

} // namespace

int main() {
    const std::string path = "test_streaming_pos_cache.gguf";
    std::vector<std::vector<float>> w;
    if (!write_model(path, w)) { std::fprintf(stderr, "write failed\n"); return 1; }
    ModelLoader ml;
    if (!ml.load(path)) { std::fprintf(stderr, "load failed\n"); return 1; }
    StreamingEncoder enc(ml);

    const int win = 70 + 4;
    const auto& a = enc.pos_heads(win);
    CHECK((int)a.size() == kLayers);
    const size_t P = 2 * win - 1;
    for (int l = 0; l < kLayers; ++l) {
        CHECK(a[l].size() == (size_t)kDk * P * kHeads);
        const std::vector<float> ref = old_per_step(ml, l, win);
        CHECK(ref.size() == a[l].size());
        CHECK(ref.size() == a[l].size() && std::memcmp(ref.data(), a[l].data(), ref.size() * sizeof(float)) == 0);
        const std::vector<double> hr = host_ref(w[l], win);
        double worst = 0;
        for (size_t i = 0; i < hr.size() && i < a[l].size(); ++i)
            worst = std::fmax(worst, std::fabs(hr[i] - a[l][i]));
        std::fprintf(stderr, "layer %d: max |host - cached| = %.3g\n", l, worst);
        CHECK(worst < 1e-4);
    }

    // Cache hit: same storage, no growth.
    const float* first = enc.pos_heads(win)[0].data();
    CHECK(first == a[0].data());
    CHECK(enc.pos_cache_size() == 1);

    // Bounded, least recently used out first.
    std::vector<float> keep = a[1];
    for (int extra = 1; extra <= 6; ++extra) {
        enc.pos_heads(win + extra);
        enc.pos_heads(win);                 // keep `win` the most recently used
        CHECK(enc.pos_cache_size() <= 4);
    }
    CHECK(enc.pos_cache_size() == 4);
    // `win` survived every eviction; an evicted length is recomputed identically.
    CHECK(std::memcmp(enc.pos_heads(win)[1].data(), keep.data(), keep.size() * sizeof(float)) == 0);
    const std::vector<float> again = old_per_step(ml, 2, win + 1);
    CHECK(std::memcmp(enc.pos_heads(win + 1)[2].data(), again.data(), again.size() * sizeof(float)) == 0);

    std::remove(path.c_str());
    if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::fprintf(stderr, "ok\n");
    return 0;
}
