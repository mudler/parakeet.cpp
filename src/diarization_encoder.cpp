#include "diarization_encoder.hpp"
#include "backend.hpp"
#include "graph_builder.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace pk {

DiarizationEncoder::DiarizationEncoder(const ModelLoader& ml)
    : ml_(ml) {
    const auto& cfg = ml.config();
    d_model_            = (int)cfg.d_model;
    n_layers_           = (int)cfg.n_layers;
    n_heads_            = (int)cfg.n_heads;
    subsampling_factor_ = (int)cfg.subsampling_factor;
    n_mels_             = (int)cfg.n_mels;
    pre_block_norm_     = cfg.pre_block_norm;
    rope_base_          = cfg.rope_base;
    ln_eps_             = 1e-5f;

    if (n_layers_ <= 0 || d_model_ <= 0 || n_heads_ <= 0 || subsampling_factor_ <= 0 ||
        d_model_ % n_heads_ != 0) {
        throw std::runtime_error("parakeet: invalid diarization encoder config");
    }
    if (!cfg.self_attention_model.empty() && cfg.self_attention_model != "rope") {
        throw std::runtime_error("parakeet: unsupported diarization self_attention_model '" +
                                 cfg.self_attention_model + "'");
    }
    head_dim_ = d_model_ / n_heads_;
    n_rot_    = (int)(head_dim_ * cfg.rotary_fraction);
}

static ggml_tensor* layer_norm(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                               const std::string& name, float eps) {
    ggml_tensor* g = pk::clone_weight(ctx, ml, (name + ".weight").c_str());
    ggml_tensor* b = pk::clone_weight(ctx, ml, (name + ".bias").c_str());
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), g), b);
}

static ggml_tensor* linear(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                           const std::string& name) {
    x = ggml_mul_mat(ctx, pk::clone_weight(ctx, ml, (name + ".weight").c_str()), x);
    ggml_tensor* b = pk::clone_weight_opt(ctx, ml, (name + ".bias").c_str());
    return b ? ggml_add(ctx, x, b) : x;
}

// mel: ne=[T_padded, n_mels] (the zero-padded [n_mels, T] frontend layout).
// Returns ne=[d_model, T_padded / factor].
ggml_tensor* DiarizationEncoder::build_pre_encode(ggml_context* ctx, ggml_tensor* mel,
                                                  int T_padded) const {
    const int factor = subsampling_factor_;
    // FeatureStacking: [C, T] -> [T, C] -> reshape [T/f, C*f] (f frames stacked).
    ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, mel));          // ne=[n_mels, T_padded]
    x = ggml_reshape_2d(ctx, x, (int64_t)n_mels_ * factor, T_padded / factor);
    return ggml_mul_mat(ctx, pk::clone_weight(ctx, ml_, "encoder.pre_encode.proj.weight"), x);
}

// x: pre-encoded ne=[d_model, T]; pos: I32 [T]. embed_norm -> blocks ->
// final_norm, returning ne=[d_model, T]. embed_norm lives here, not in
// pre_encode, because NeMo applies it after the (bypassable) pre-encoder:
// the streaming speaker cache holds pre-norm embeddings.
ggml_tensor* DiarizationEncoder::build_blocks(ggml_context* ctx, ggml_tensor* x,
                                              ggml_tensor* pos) const {
    const int d = d_model_, H = n_heads_, hd = head_dim_;
    const int64_t T = x->ne[1];
    const float scale = 1.0f / std::sqrt((float)hd);
    if (pre_block_norm_) x = layer_norm(ctx, ml_, x, "encoder.embed_norm", ln_eps_);

    for (int i = 0; i < n_layers_; ++i) {
        const std::string base = "encoder.layers." + std::to_string(i) + ".";

        // Attention: x = x + out_proj(attn(norm1(x)))
        ggml_tensor* h = layer_norm(ctx, ml_, x, base + "norm1", ln_eps_);
        ggml_tensor* qkv = linear(ctx, ml_, h, base + "attn.w_qkv");     // ne=[3d, T]
        const size_t row = qkv->nb[1];
        ggml_tensor* q = ggml_view_3d(ctx, qkv, hd, H, T, hd * sizeof(float), row, 0);
        ggml_tensor* k = ggml_view_3d(ctx, qkv, hd, H, T, hd * sizeof(float), row,
                                      (size_t)d * sizeof(float));
        ggml_tensor* v = ggml_view_3d(ctx, qkv, hd, H, T, hd * sizeof(float), row,
                                      (size_t)2 * d * sizeof(float));
        q = ggml_rope_ext(ctx, ggml_cont(ctx, q), pos, nullptr, n_rot_, GGML_ROPE_TYPE_NEOX,
                          0, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, ggml_cont(ctx, k), pos, nullptr, n_rot_, GGML_ROPE_TYPE_NEOX,
                          0, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // flash_attn_ext takes q/k/v as [hd, T, H] and returns [hd, H, T]
        // (heads already interleaved per frame), which reshapes to [d, T].
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, ggml_cont(ctx, v), 0, 2, 1, 3);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, ggml_cont(ctx, attn), d, T);
        x = ggml_add(ctx, x, linear(ctx, ml_, attn, base + "attn.out_proj"));

        // Feed-forward: x = x + W2 gelu(W1 norm2(x))
        h = layer_norm(ctx, ml_, x, base + "norm2", ln_eps_);
        h = ggml_gelu(ctx, linear(ctx, ml_, h, base + "ffn.net.0"));
        x = ggml_add(ctx, x, linear(ctx, ml_, h, base + "ffn.net.3"));
    }
    return layer_norm(ctx, ml_, x, "encoder.final_norm", ln_eps_);
}

// Copy mel [n_mels, T] into a zero-padded [n_mels, T_padded] graph input.
static ggml_tensor* mel_input(ggml_context* ctx, GraphInputPool& pool,
                              const std::vector<float>& mel, int n_mels, int T, int T_padded) {
    std::vector<float>& padded = pool.alloc_f32((size_t)n_mels * T_padded);
    for (int m = 0; m < n_mels; ++m)
        std::copy_n(mel.begin() + (size_t)m * T, T, padded.begin() + (size_t)m * T_padded);
    int64_t ne[2] = {T_padded, n_mels};
    return pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne, padded.data(),
                                  padded.size() * sizeof(float));
}

static std::vector<int32_t> positions(int T) {
    std::vector<int32_t> p(T);
    for (int i = 0; i < T; ++i) p[i] = i;
    return p;
}

void DiarizationEncoder::forward(const std::vector<float>& mel, int n_mels, int T,
                                 std::vector<float>& enc_out, int& T_enc) const {
    if (n_mels != n_mels_ || mel.size() != (size_t)n_mels * T || T <= 0)
        throw std::runtime_error("parakeet: diarization encoder got a bad mel shape");
    const int f = subsampling_factor_;
    const int T_padded = (T + f - 1) / f * f;
    T_enc = T_padded / f;
    const std::vector<int32_t> pos_data = positions(T_enc);

    pk::ensure_weights_realized(ml_);
    GraphInputPool pool;
    const bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = build_pre_encode(ctx, mel_input(ctx, pool, mel, n_mels, T, T_padded),
                                          T_padded);
        int64_t pos_ne[1] = {T_enc};
        ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                                  const_cast<int32_t*>(pos_data.data()),
                                                  pos_data.size() * sizeof(int32_t));
        return build_blocks(ctx, x, pos);
    }, enc_out);
    if (!ok) throw std::runtime_error("parakeet: diarization encoder graph failed");
}

void DiarizationEncoder::pre_encode(const std::vector<float>& mel, int n_mels, int T,
                                    std::vector<float>& emb, int& T_enc) const {
    if (n_mels != n_mels_ || mel.size() != (size_t)n_mels * T || T <= 0)
        throw std::runtime_error("parakeet: diarization pre_encode got a bad mel shape");
    const int f = subsampling_factor_;
    const int T_padded = (T + f - 1) / f * f;
    T_enc = T_padded / f;

    pk::ensure_weights_realized(ml_);
    GraphInputPool pool;
    const bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        return build_pre_encode(ctx, mel_input(ctx, pool, mel, n_mels, T, T_padded), T_padded);
    }, emb);
    if (!ok) throw std::runtime_error("parakeet: diarization pre_encode graph failed");
}

void DiarizationEncoder::transformer_forward(const std::vector<float>& emb, int T_enc,
                                             std::vector<float>& enc_out) const {
    if (emb.size() != (size_t)d_model_ * T_enc || T_enc <= 0)
        throw std::runtime_error("parakeet: diarization transformer got a bad input shape");
    const std::vector<int32_t> pos_data = positions(T_enc);

    pk::ensure_weights_realized(ml_);
    const bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ne[2] = {d_model_, T_enc};
        ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne,
                                                const_cast<float*>(emb.data()),
                                                emb.size() * sizeof(float));
        int64_t pos_ne[1] = {T_enc};
        ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                                  const_cast<int32_t*>(pos_data.data()),
                                                  pos_data.size() * sizeof(int32_t));
        return build_blocks(ctx, x, pos);
    }, enc_out);
    if (!ok) throw std::runtime_error("parakeet: diarization transformer graph failed");
}

} // namespace pk
