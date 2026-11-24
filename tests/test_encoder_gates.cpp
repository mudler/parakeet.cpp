// CPU-only, no checkpoint. The two switches of the transformer encoder that the speech LLM
// audio model needs, each checked against a double-precision host implementation on a tiny
// random encoder:
//   parakeet.encoder.qk_norm     per-head LayerNorm (weight and bias) on q and k before RoPE
//   parakeet.encoder.activation  "gelu_erf" (exact) against the default "gelu" (ggml tanh form)
// With both switches off the encoder must match the old behaviour (tanh GELU, no QK norm).
// Also: an unknown activation is refused at load, and normalize_per_feature matches a
// direct computation.
#include "diarization_encoder.hpp"
#include "mel.hpp"
#include "model_loader.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

namespace {

constexpr int kMels = 4, kSub = 2, kD = 16, kHeads = 2, kHd = 8, kLayers = 2, kFf = 32, kRot = 4;

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    float next() {   // uniform in [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float)((s >> 40) & 0xffffff) / 8388608.0f - 1.0f;
    }
};

struct Weights {
    std::vector<std::vector<float>> data;   // in tensor creation order
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> ne;
    const std::vector<float>& get(const std::string& n) const {
        for (size_t i = 0; i < names.size(); ++i) if (names[i] == n) return data[i];
        std::abort();
    }
};

void add(Weights& w, Rng& r, const std::string& name, std::vector<int64_t> ne, float scale, float offset = 0.0f) {
    size_t n = 1;
    for (int64_t e : ne) n *= (size_t)e;
    std::vector<float> v(n);
    for (float& x : v) x = offset + scale * r.next();
    w.names.push_back(name); w.ne.push_back(ne); w.data.push_back(std::move(v));
}

Weights make_weights(bool qk_norm) {
    Rng r;
    Weights w;
    add(w, r, "encoder.pre_encode.proj.weight", {kMels * kSub, kD}, 0.5f);
    add(w, r, "encoder.embed_norm.weight", {kD}, 0.2f, 1.0f);
    add(w, r, "encoder.embed_norm.bias", {kD}, 0.2f);
    for (int l = 0; l < kLayers; ++l) {
        const std::string b = "encoder.layers." + std::to_string(l) + ".";
        add(w, r, b + "norm1.weight", {kD}, 0.2f, 1.0f);
        add(w, r, b + "norm1.bias", {kD}, 0.2f);
        add(w, r, b + "attn.w_qkv.weight", {kD, 3 * kD}, 0.4f);
        if (qk_norm) {
            for (const char* n : {"attn.q_norm", "attn.k_norm"}) {
                add(w, r, b + n + std::string(".weight"), {kHd}, 0.5f, 1.0f);
                add(w, r, b + n + std::string(".bias"), {kHd}, 0.5f);
            }
        }
        add(w, r, b + "attn.out_proj.weight", {kD, kD}, 0.4f);
        add(w, r, b + "attn.out_proj.bias", {kD}, 0.2f);
        add(w, r, b + "norm2.weight", {kD}, 0.2f, 1.0f);
        add(w, r, b + "norm2.bias", {kD}, 0.2f);
        add(w, r, b + "ffn.net.0.weight", {kD, kFf}, 0.6f);
        add(w, r, b + "ffn.net.0.bias", {kFf}, 0.5f);
        add(w, r, b + "ffn.net.3.weight", {kFf, kD}, 0.4f);
        add(w, r, b + "ffn.net.3.bias", {kD}, 0.2f);
    }
    add(w, r, "encoder.final_norm.weight", {kD}, 0.2f, 1.0f);
    add(w, r, "encoder.final_norm.bias", {kD}, 0.2f);
    return w;
}

bool write_gguf(const std::string& path, const Weights& w, bool qk_norm, const char* activation) {
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "parakeet");
    gguf_set_val_str(g, "parakeet.arch", "diarization");
    gguf_set_val_u32(g, "parakeet.encoder.feat_in", kMels);
    gguf_set_val_u32(g, "parakeet.encoder.d_model", kD);
    gguf_set_val_u32(g, "parakeet.encoder.n_layers", kLayers);
    gguf_set_val_u32(g, "parakeet.encoder.n_heads", kHeads);
    gguf_set_val_u32(g, "parakeet.encoder.ff_dim", kFf);
    gguf_set_val_u32(g, "parakeet.encoder.subsampling_factor", kSub);
    gguf_set_val_str(g, "parakeet.encoder.self_attention_model", "rope");
    gguf_set_val_f32(g, "parakeet.encoder.rotary_fraction", (float)kRot / kHd);
    gguf_set_val_bool(g, "parakeet.encoder.pre_block_norm", true);
    gguf_set_val_u32(g, "parakeet.preprocessor.n_mels", kMels);
    if (qk_norm) gguf_set_val_bool(g, "parakeet.encoder.qk_norm", true);
    if (activation) gguf_set_val_str(g, "parakeet.encoder.activation", activation);
    gguf_set_val_u32(g, "parakeet.diar.n_speakers", 2);
    gguf_set_val_u32(g, "parakeet.diar.tf_d_model", 4);
    gguf_set_val_u32(g, "parakeet.diar.upsample_factor", kSub);

    size_t bytes = 1 << 16;
    for (const auto& d : w.data) bytes += d.size() * sizeof(float) + 512;
    ggml_init_params ip{bytes, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    for (size_t i = 0; i < w.names.size(); ++i) {
        ggml_tensor* t = w.ne[i].size() == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w.ne[i][0])
                                             : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ne[i][0], w.ne[i][1]);
        ggml_set_name(t, w.names[i].c_str());
        std::copy(w.data[i].begin(), w.data[i].end(), (float*)t->data);
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// ---- host reference, double precision ----
using V = std::vector<double>;

V layer_norm(const V& x, int rows, int n, const std::vector<float>& g, const std::vector<float>& b) {
    V y(x.size());
    for (int r = 0; r < rows; ++r) {
        double mean = 0, var = 0;
        for (int i = 0; i < n; ++i) mean += x[(size_t)r * n + i];
        mean /= n;
        for (int i = 0; i < n; ++i) { double d = x[(size_t)r * n + i] - mean; var += d * d; }
        var /= n;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < n; ++i)
            y[(size_t)r * n + i] = (x[(size_t)r * n + i] - mean) * inv * g[i] + b[i];
    }
    return y;
}

// y[t][n] = sum_k W[n][k] x[t][k] (+ bias)
V linear(const V& x, int T, int K, const std::vector<float>& W, int N, const std::vector<float>* bias) {
    V y((size_t)T * N);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n) {
            double a = bias ? (*bias)[n] : 0.0;
            for (int k = 0; k < K; ++k) a += (double)W[(size_t)n * K + k] * x[(size_t)t * K + k];
            y[(size_t)t * N + n] = a;
        }
    return y;
}

double gelu_exact(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }
double gelu_tanh(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

V reference(const Weights& w, const std::vector<float>& mel, int T, bool qk_norm, bool erf) {
    const int Te = T / kSub;
    V stacked((size_t)Te * kMels * kSub);
    for (int g = 0; g < Te; ++g)
        for (int j = 0; j < kSub; ++j)
            for (int m = 0; m < kMels; ++m)
                stacked[(size_t)g * kMels * kSub + j * kMels + m] = mel[(size_t)m * T + g * kSub + j];
    V x = linear(stacked, Te, kMels * kSub, w.get("encoder.pre_encode.proj.weight"), kD, nullptr);
    x = layer_norm(x, Te, kD, w.get("encoder.embed_norm.weight"), w.get("encoder.embed_norm.bias"));
    for (int l = 0; l < kLayers; ++l) {
        const std::string b = "encoder.layers." + std::to_string(l) + ".";
        V h = layer_norm(x, Te, kD, w.get(b + "norm1.weight"), w.get(b + "norm1.bias"));
        V qkv = linear(h, Te, kD, w.get(b + "attn.w_qkv.weight"), 3 * kD, nullptr);
        V q((size_t)Te * kD), k((size_t)Te * kD), v((size_t)Te * kD);
        for (int t = 0; t < Te; ++t)
            for (int c = 0; c < kD; ++c) {
                q[(size_t)t * kD + c] = qkv[(size_t)t * 3 * kD + c];
                k[(size_t)t * kD + c] = qkv[(size_t)t * 3 * kD + kD + c];
                v[(size_t)t * kD + c] = qkv[(size_t)t * 3 * kD + 2 * kD + c];
            }
        if (qk_norm) {   // per (token, head) LayerNorm over head_dim
            q = layer_norm(q, Te * kHeads, kHd, w.get(b + "attn.q_norm.weight"), w.get(b + "attn.q_norm.bias"));
            k = layer_norm(k, Te * kHeads, kHd, w.get(b + "attn.k_norm.weight"), w.get(b + "attn.k_norm.bias"));
        }
        auto rope = [&](V& a) {
            for (int t = 0; t < Te; ++t)
                for (int hh = 0; hh < kHeads; ++hh) {
                    double* p = &a[(size_t)t * kD + hh * kHd];
                    for (int i = 0; i < kRot / 2; ++i) {
                        const double th = t * std::pow(10000.0, -2.0 * i / kRot);
                        const double c = std::cos(th), s = std::sin(th);
                        const double x0 = p[i], x1 = p[i + kRot / 2];
                        p[i] = x0 * c - x1 * s;
                        p[i + kRot / 2] = x0 * s + x1 * c;
                    }
                }
        };
        rope(q); rope(k);
        V att((size_t)Te * kD);
        for (int hh = 0; hh < kHeads; ++hh)
            for (int t = 0; t < Te; ++t) {
                V sc(Te);
                double mx = -1e30;
                for (int u = 0; u < Te; ++u) {
                    double a = 0;
                    for (int c = 0; c < kHd; ++c) a += q[(size_t)t * kD + hh * kHd + c] * k[(size_t)u * kD + hh * kHd + c];
                    sc[u] = a / std::sqrt((double)kHd);
                    mx = std::max(mx, sc[u]);
                }
                double den = 0;
                for (double& s : sc) { s = std::exp(s - mx); den += s; }
                for (int c = 0; c < kHd; ++c) {
                    double a = 0;
                    for (int u = 0; u < Te; ++u) a += sc[u] / den * v[(size_t)u * kD + hh * kHd + c];
                    att[(size_t)t * kD + hh * kHd + c] = a;
                }
            }
        V o = linear(att, Te, kD, w.get(b + "attn.out_proj.weight"), kD, &w.get(b + "attn.out_proj.bias"));
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        h = layer_norm(x, Te, kD, w.get(b + "norm2.weight"), w.get(b + "norm2.bias"));
        V f = linear(h, Te, kD, w.get(b + "ffn.net.0.weight"), kFf, &w.get(b + "ffn.net.0.bias"));
        for (double& a : f) a = erf ? gelu_exact(a) : gelu_tanh(a);
        V f2 = linear(f, Te, kFf, w.get(b + "ffn.net.3.weight"), kD, &w.get(b + "ffn.net.3.bias"));
        for (size_t i = 0; i < x.size(); ++i) x[i] += f2[i];
    }
    return layer_norm(x, Te, kD, w.get("encoder.final_norm.weight"), w.get("encoder.final_norm.bias"));
}

double run_case(const char* label, bool qk_norm, const char* activation, double* out_vs_other = nullptr,
                std::vector<float>* keep = nullptr) {
    const std::string path = std::string("test_encoder_gates_") + label + ".gguf";
    const Weights w = make_weights(qk_norm);
    CHECK(write_gguf(path, w, qk_norm, activation));
    ModelLoader ml;
    CHECK(ml.load(path));
    CHECK(ml.config().qk_norm == qk_norm);
    CHECK(ml.config().activation == (activation ? activation : "gelu"));

    const int T = 12;   // 6 encoder frames
    std::vector<float> mel((size_t)kMels * T);
    Rng r; r.s ^= 77;
    for (float& x : mel) x = 2.0f * r.next();
    DiarizationEncoder enc(ml);
    std::vector<float> got;
    int Te = 0;
    enc.forward(mel, kMels, T, got, Te);
    CHECK(Te == T / kSub);
    const V ref = reference(w, mel, T, qk_norm, activation && std::string(activation) == "gelu_erf");
    double md = 0;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) md = std::max(md, std::fabs(ref[i] - got[i]));
    CHECK(got.size() == ref.size());
    std::fprintf(stderr, "[%s] max|got-ref| = %.3e\n", label, md);
    if (keep) *keep = got;
    std::remove(path.c_str());
    return md;
}

}  // namespace

int main() {
    // ggml's default GELU is the tanh form evaluated through a half-precision table on the CPU,
    // so it only matches the double reference to about 1e-3; the erf form is exact.
    const double kTanhTol = 3e-3, kExactTol = 2e-5;
    std::vector<float> tanh_out, erf_out;
    CHECK(run_case("default", false, nullptr, nullptr, &tanh_out) < kTanhTol);    // old behaviour
    CHECK(run_case("tanh", false, "gelu") < kTanhTol);
    CHECK(run_case("erf", false, "gelu_erf", nullptr, &erf_out) < kExactTol);
    CHECK(run_case("qk", true, "gelu") < kTanhTol);
    CHECK(run_case("qk_erf", true, "gelu_erf") < kExactTol);

    // The gate must change the result: same weights, tanh against erf.
    double d = 0;
    for (size_t i = 0; i < tanh_out.size() && i < erf_out.size(); ++i) d = std::max(d, (double)std::fabs(tanh_out[i] - erf_out[i]));
    std::fprintf(stderr, "tanh vs erf GELU differ by up to %.3e\n", d);
    CHECK(d > 2e-4);

    {   // an unknown activation is a load error, not a silent fallback
        const Weights w = make_weights(false);
        const std::string path = "test_encoder_gates_bad.gguf";
        CHECK(write_gguf(path, w, false, "relu"));
        ModelLoader ml;
        CHECK(!ml.load(path));
        std::remove(path.c_str());
    }

    {   // normalize_per_feature: mean/unbiased std of the first `valid` frames, + 1e-5, rest zeroed
        const int n_mels = 3, T = 7, valid = 5;
        std::vector<float> f((size_t)n_mels * T);
        Rng r; r.s ^= 5;
        for (float& x : f) x = 3.0f * r.next() - 1.0f;
        std::vector<float> g = f;
        normalize_per_feature(g, n_mels, T, valid);
        for (int m = 0; m < n_mels; ++m) {
            double mean = 0, var = 0;
            for (int t = 0; t < valid; ++t) mean += f[(size_t)m * T + t];
            mean /= valid;
            for (int t = 0; t < valid; ++t) var += (f[(size_t)m * T + t] - mean) * (f[(size_t)m * T + t] - mean);
            const double sd = std::sqrt(var / (valid - 1)) + 1e-5;
            for (int t = 0; t < T; ++t) {
                const double want = t < valid ? (f[(size_t)m * T + t] - mean) / sd : 0.0;
                CHECK(std::fabs(g[(size_t)m * T + t] - want) < 1e-5);
            }
        }
    }
    std::fprintf(stderr, failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
