#pragma once
#include "model_loader.hpp"
#include "graph_builder.hpp"
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// NeMo's rel_shift of the position scores bd [P = 2T-1, T(query), H] for the single-clip attention, as a
// strided view [T(key), T(query), H] of bd: no copy. Its rows are not contiguous, so it suits an op that
// reads strided input, such as the ggml_add that consumes it in build_graph.
ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* bd);

// The additive attention mask of RelPosAttention::build_graph, row-major [T_q, T_k]: 0 where query qi
// may attend to key kj, -inf where it may not. Keys at or past valid_len are masked, and rows of padded
// queries stay all 0. With chunk_size > 0, keys outside the chunked-limited window (left_chunks chunks
// back) are masked too; with att_left >= 0, keys outside [qi - att_left, qi + att_right]. The mask is
// allocated in `pool`. Returns null, allocating nothing, when no entry would be -inf (valid_len >= T,
// chunk_size 0, att_left < 0): softmax then runs without a mask.
const std::vector<float>* relpos_attention_mask(int T, int valid_len, int chunk_size, int left_chunks,
                                                int att_left, int att_right, GraphInputPool& pool);

// Relative-position multi-head self-attention (Transformer-XL style) as used by
// the FastConformer encoder — NeMo RelPositionMultiHeadAttention.
//
// IMPORTANT (module boundary): this mirrors NeMo's `self_attn(...)` exactly. The
// input `x` is the ALREADY-normalized attention input, i.e. NeMo's
// `norm_self_att(residual)` (note: `residual` already includes FFN1, so it is
// NOT `norm_self_att(enc_pre_layers)`). The norm itself lives in the conformer
// layer (next task), NOT here. q, k and v are all projected from this same `x`.
//
// Layout convention (matches the rest of the port and the baseline GGUF):
//   x       row-major [T, d_model]      (d_model fastest)
//   pos_emb row-major [2T-1, d_model]   (d_model fastest)
//   out     row-major [T, d_model]      (d_model fastest)
//
// `valid_len` is the number of non-padding frames. Frames >= valid_len are
// padding: their key columns get -inf in the scores (zero attention weight) and
// their query rows are fully masked, so each padded output row reduces to
// linear_out.bias — exactly matching NeMo's att_mask handling. Pass valid_len ==
// T to disable masking.
class RelPosAttention {
public:
    RelPosAttention(const ModelLoader& ml, int layer_idx);

    // GRAPH-BUILDER: append MHSA ops to a SHARED graph `ctx`. `xt` is the
    // normalized attention input tensor [D, T] (ggml ne0=D fastest) and `pe` is
    // the positional-encoding tensor [D, pos_len], both ALREADY in the graph.
    // Returns the attention output [D, T]. Host-built additive masks are fed via
    // pk::graph_input_tensor and registered into `pool` (must outlive compute).
    // Reused by the fused conformer layer and the unit test.
    //
    // When att_left/att_right >= 0, an additional SYMMETRIC sliding-window mask
    // is applied: query qi may attend to key kj only if -att_left <= qi-kj <=
    // att_right (NeMo rel_pos_local_attn). Defaults (-1, -1) = full context.
    ggml_tensor* build_graph(ggml_context* ctx, ggml_tensor* xt, int T,
                             ggml_tensor* pe, int pos_len, int valid_len,
                             GraphInputPool& pool,
                             int att_left = -1, int att_right = -1) const;

    // LOCAL (banded / Longformer) GRAPH-BUILDER. Appends NeMo rel_pos_local_attn
    // ops to a SHARED graph. `xt` is [D, T] and `pe` is the LOCAL positional
    // encoding [D, att_left+att_right+1]. Each query attends only to keys within
    // [t-att_left, t+att_right] via pad-and-shift, so peak memory is
    // O(T * window) not O(T^2). Returns the attention output [D, T].
    ggml_tensor* build_graph_local(ggml_context* ctx, ggml_tensor* xt, int T,
                                   ggml_tensor* pe, int pos_len, int valid_len,
                                   int att_left, int att_right,
                                   GraphInputPool& pool) const;

    // Batched GRAPH-BUILDER. `xt` is [D, T, B]; `pe` is [D, pos_len] (shared
    // across the batch). `valid_len` is per item (size B). Returns [D, T, B].
    ggml_tensor* build_graph_batched(ggml_context* ctx, ggml_tensor* xt, int T,
                                     int B, ggml_tensor* pe, int pos_len,
                                     const std::vector<int>& valid_len,
                                     GraphInputPool& pool) const;

    // Batched LOCAL (banded / Longformer) GRAPH-BUILDER. Same as
    // build_graph_local but for B>1: `xt` is [D, T, B], `pe` is the LOCAL
    // positional encoding [D, att_left+att_right+1] (shared across the batch),
    // `valid_len` is per item. Returns [D, T, B]. Banded -> O(T*window) memory.
    ggml_tensor* build_graph_batched_local(ggml_context* ctx, ggml_tensor* xt, int T,
                                           int B, ggml_tensor* pe, int pos_len,
                                           const std::vector<int>& valid_len,
                                           int att_left, int att_right,
                                           GraphInputPool& pool) const;

    // CHUNK-MATMUL LOCAL GRAPH-BUILDER. Same math/output as build_graph_local
    // (NeMo rel_pos_local_attn, banded O(T*window)) but built with O(1) graph
    // nodes regardless of window: time is tiled into chunks of `chunk` frames,
    // K/V are gathered as OVER-PADDED overlapping chunks (so ggml's dense
    // ne-product view-bounds check passes), the per-chunk q.k blocks are one
    // batched ggml_mul_mat, and a diagonal "skew" view extracts the [P,T] band.
    // Lets the window go to NeMo's full [128,128] without the pad-and-shift
    // path's O(window) nodes + O(window^2) concat. `pe` is the LOCAL positional
    // encoding [D, att_left+att_right+1]. chunk<=0 picks a default.
    ggml_tensor* build_graph_local_chunked(ggml_context* ctx, ggml_tensor* xt,
                                           int T, ggml_tensor* pe, int pos_len,
                                           int valid_len, int att_left, int att_right,
                                           GraphInputPool& pool, int chunk = -1) const;

    // Thin wrapper over build_graph_local_chunked (test entry point). Same
    // signature/semantics as forward_local; output must match it.
    void forward_local_chunked(const std::vector<float>& x, int T,
                               const std::vector<float>& pos_emb, int pos_len,
                               int valid_len, int att_left, int att_right,
                               std::vector<float>& out, int chunk = -1) const;

    // Batched CHUNK-MATMUL local GRAPH-BUILDER. `xt` is [D, T, B]; `pe` is the
    // LOCAL positional encoding [D, att_left+att_right+1] (shared). Runs the O(1)
    // chunk-matmul construction per item (the 4D chunk kernel can't also carry a
    // batch dim - ggml is 4D), so this is O(B) nodes, still O(1) in the window.
    // `valid_len` is per item. Returns [D, T, B]. Same output as
    // build_graph_batched_local but window-cap-free.
    ggml_tensor* build_graph_batched_local_chunked(ggml_context* ctx, ggml_tensor* xt,
                                                   int T, int B, ggml_tensor* pe, int pos_len,
                                                   const std::vector<int>& valid_len,
                                                   int att_left, int att_right,
                                                   GraphInputPool& pool, int chunk = -1) const;

    // x: [T, d_model]; pos_emb: [2T-1, d_model]; out: [T, d_model].
    void forward(const std::vector<float>& x, int T,
                 const std::vector<float>& pos_emb, int pos_len,
                 int valid_len,
                 std::vector<float>& out) const;

    // Local (banded / Longformer) attention — NeMo rel_pos_local_attn. Each
    // query qi attends only to keys in [qi-att_left, qi+att_right]; pos_emb is
    // the LOCAL positional encoding [att_left+att_right+1, d_model] (i.e. 2W+1
    // for a symmetric [W,W] window), NOT the full 2T-1. Output matches NeMo's
    // Longformer self_attn while bounding memory to O(T * window) instead of
    // O(T^2). x: [T, d_model]; out: [T, d_model].
    void forward_local(const std::vector<float>& x, int T,
                       const std::vector<float>& pos_emb, int pos_len,
                       int valid_len, int att_left, int att_right,
                       std::vector<float>& out) const;
private:
    const ModelLoader& ml_;
    int layer_idx_;
    int d_model_;
    int n_heads_;
    int d_head_;
    // Chunked-limited attention (att_context_style=="chunked_limited"). When set,
    // an extra additive -inf window mask is applied to the scores (see forward()).
    bool chunked_limited_ = false;
    int att_left_  = -1;   // att_context_size[0] (left limit in frames)
    int att_right_ = -1;   // att_context_size[1] (right/lookahead in frames)
};

} // namespace pk
