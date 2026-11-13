#pragma once
#include "model_loader.hpp"
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// DiarizationEncoder — pre-LN RoPE Transformer encoder for Nemotron-3-Diarization.
//
// This is NOT the FastConformer encoder used by ASR. Nemotron-3-Diarization uses
// a NeMo TransformerEncoder with:
//   - FeatureStacking subsampling (8x stack of mel frames -> Linear, no bias)
//   - embed_norm LayerNorm (pre_block_norm)
//   - N x TransformerBlock (pre-norm): x = x + attn(norm1(x)); x = x + ffn(norm2(x))
//   - attention: fused QKV (optional bias) -> optional per-head LayerNorm on q
//     and k (parakeet.encoder.qk_norm) -> RoPE (GPT-NeoX) -> softmax attention
//     -> out_proj (bias)
//   - FeedForward: Linear -> GELU -> Linear (both with bias). The GELU is ggml's
//     tanh form unless parakeet.encoder.activation is "gelu_erf" (exact).
//   - final_norm LayerNorm
//
// All sequences are TIME-MAJOR row-major [T, d_model] (x[t*d_model + c]), which
// is ggml's natural ne=[d_model, T] layout, so no transposes are needed between
// stages. The mel input keeps the frontend's [n_mels, T] layout.
class DiarizationEncoder {
public:
    explicit DiarizationEncoder(const ModelLoader& ml);

    // Full encoder: mel [n_mels, T] (mel[m*T + t]) -> enc_out [T_enc, d_model].
    // T_enc = ceil(T / subsampling).
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& enc_out, int& T_enc) const;

    // Streaming split. pre_encode = FeatureStacking + projection (no
    // embed_norm), i.e. what NeMo stores in the speaker cache / FIFO.
    //   mel [n_mels, T] -> emb [T_enc, d_model]
    void pre_encode(const std::vector<float>& mel, int n_mels, int T,
                    std::vector<float>& emb, int& T_enc) const;

    // embed_norm + transformer blocks + final_norm over pre-encoded embeddings
    // (NeMo frontend_encoder with bypass_pre_encode=True).
    //   emb [T_enc, d_model] -> enc_out [T_enc, d_model]
    void transformer_forward(const std::vector<float>& emb, int T_enc,
                             std::vector<float>& enc_out) const;

    int subsampling() const { return subsampling_factor_; }
    int n_mels() const { return n_mels_; }
    int d_model() const { return d_model_; }

private:
    // Graph builders shared by the entry points above.
    ggml_tensor* build_pre_encode(ggml_context* ctx, ggml_tensor* mel, int T_padded) const;
    ggml_tensor* build_blocks(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos) const;

    const ModelLoader& ml_;
    int d_model_;
    int n_layers_;
    int n_heads_;
    int head_dim_;
    int subsampling_factor_;
    int n_mels_;
    bool pre_block_norm_;
    float rope_base_;
    int n_rot_;          // rotated dims per head (head_dim * rotary_fraction)
    bool qk_norm_;       // per-head LayerNorm on q and k
    bool gelu_erf_;      // exact GELU in the FFN
    float ln_eps_;
};

} // namespace pk
