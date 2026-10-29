#pragma once
#include "model_loader.hpp"
#include "graph_builder.hpp"
#include <functional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// How the conv module's depthwise Conv1d (groups = C) is computed. Production passes Auto; tests force a
// path to compare the two graphs on one backend.
enum class DepthwiseConvPath {
    Auto,    // SsmConv where depthwise_conv_takes_ssm says so, else Im2col
    Im2col,  // per-item im2col + mul_mat, then a transpose
    SsmConv, // one SSM_CONV over the padded input, batch on ne2
};

// Whether Auto takes SSM_CONV. Never on the CPU: it was no faster there, and the NeMo parity tests stay
// bit-identical. Only at the widths ggml-cuda has kernels for (3, 4, 5, 9, 15): its supports_op does not
// check the width, and it aborts on others. Only when `backend_runs_ops` says the backend runs the pad and
// the SSM_CONV, since an op it declines sends the whole graph through the scheduler's CPU fallback. That
// callback builds the two ops to ask about them, so it runs only after the first two checks pass.
bool depthwise_conv_takes_ssm(bool backend_is_cpu, int K,
                              const std::function<bool()>& backend_runs_ops);

// Depthwise Conv1d of x_tcb [T, C, B] (contiguous) with weight dww [K, 1, C], NeMo's cross-correlation
// with K - 1 zeros on the left when causal, else (K - 1) / 2 on each side (K odd). Returns [C, T, B].
// Must be called from inside a build lambda passed to Backend::compute.
ggml_tensor* build_depthwise_conv(ggml_context* ctx, ggml_tensor* x_tcb, ggml_tensor* dww,
                                  int K, bool causal,
                                  DepthwiseConvPath path = DepthwiseConvPath::Auto);

// A single FastConformer encoder layer (NeMo ConformerLayer), built per-layer
// from the GGUF weights. Mirrors NeMo's ConformerLayer.forward exactly:
//
//   r = x
//   r = r + 0.5 * feed_forward1(norm_feed_forward1(r))   # FFN1 (half-step)
//   r = r + self_attn(norm_self_att(r), pos_emb, mask)    # MHSA (RelPosAttention)
//   r = r + conv(norm_conv(r))                            # Conv module
//   r = r + 0.5 * feed_forward2(norm_feed_forward2(r))    # FFN2 (half-step)
//   out = norm_out(r)
//
// feed_forwardN  = linear2(silu(linear1(x))).
// conv (ConformerConvolution, operates on [d_model, T]):
//   pointwise_conv1 (d->2d, k=1) -> GLU(dim=channel) -> [zero padded time pos]
//   -> depthwise_conv (d->d, k=conv_kernel, groups=d)
//   -> norm -> SiLU -> pointwise_conv2 (d->d, k=1).
// Two conv-module variants, selected by config (byte-identical gating):
//   * conv_norm_type=batch_norm (offline models): inference affine fold from
//     running stats (eps 1e-5); symmetric depthwise pad (k-1)/2 each side.
//   * conv_norm_type=layer_norm (streaming models, e.g. the EOU model): LayerNorm
//     over the channel dim per time-frame (eps 1e-5), gamma/beta read from
//     conv.batch_norm.{weight,bias} (NeMo always names the attr `batch_norm`).
//     conv_causal=true -> causal depthwise pad (left k-1, right 0).
// All norm_* (outside conv) are LayerNorm (eps 1e-5). MHSA reuses pk::RelPosAttention.
//
// Layout convention (matches the rest of the port and the baseline GGUF):
//   x       row-major [T, d_model]      (d_model fastest)
//   pos_emb row-major [2T-1, d_model]   (d_model fastest)
//   out     row-major [T, d_model]      (d_model fastest)
//
// `valid_len` is the number of non-padding frames (frames >= valid_len are
// center-pad). It is threaded into RelPosAttention (key/query masking) and used
// to zero padded time positions before the depthwise conv, matching NeMo's
// pad_mask handling. Pass valid_len == T to disable masking.
class ConformerLayer {
public:
    ConformerLayer(const ModelLoader& ml, int layer_idx);

    // GRAPH-BUILDER: append the WHOLE conformer layer (FFN1 + MHSA + conv + FFN2
    // + norm_out) to a SHARED graph `ctx`. `xt` is the layer input tensor [D, T]
    // and `pe` is the positional-encoding tensor [D, pos_len], both ALREADY in
    // the graph. Returns the layer output [D, T] (ggml ne0=D fastest, row-major
    // [T, d_model]). Host-built masks / batch-norm fold constants are fed via
    // pk::graph_input_tensor and registered into `pool` (must outlive compute).
    // This is the unit reused by the fused encoder AND the unit test; computing
    // the entire layer as ONE sub-graph (vs the old 5 sub-graphs) is what lets
    // the fused encoder be a single graph.
    // When att_left/att_right >= 0, the self-attention uses NeMo
    // rel_pos_local_attn (banded, O(T*window)): `pe` must then be the LOCAL
    // positional encoding [d_model, att_left+att_right+1]. Defaults (-1, -1) keep
    // full attention with `pe` = [d_model, 2T-1].
    ggml_tensor* build_graph(ggml_context* ctx, ggml_tensor* xt, int T,
                             ggml_tensor* pe, int pos_len, int valid_len,
                             GraphInputPool& pool,
                             int att_left = -1, int att_right = -1) const;

    // Batched GRAPH-BUILDER. `xt` is [D, T, B]; `pe` is [D, pos_len] (shared
    // across the batch). `valid_len` is per item (size B). Returns [D, T, B].
    // att_left/att_right >= 0 routes self-attention to banded local attention
    // (pe = LOCAL [d_model, att_left+att_right+1]); defaults (-1,-1) = full.
    ggml_tensor* build_graph_batched(ggml_context* ctx, ggml_tensor* xt, int T,
                                     int B, ggml_tensor* pe, int pos_len,
                                     const std::vector<int>& valid_len,
                                     GraphInputPool& pool,
                                     int att_left = -1, int att_right = -1) const;

    // x: [T, d_model]; pos_emb: [pos_len=2T-1, d_model]; out: [T, d_model].
    void forward(const std::vector<float>& x, int T,
                 const std::vector<float>& pos_emb, int pos_len,
                 int valid_len,
                 std::vector<float>& out) const;

    // Same as forward(), but also returns the ConformerConvolution sub-module
    // output (NeMo `layers[i].conv` output, row-major [T, d_model]) for parity
    // localization against the baseline `l0_conv_out`.
    void forward_with_conv(const std::vector<float>& x, int T,
                           const std::vector<float>& pos_emb, int pos_len,
                           int valid_len,
                           std::vector<float>& out,
                           std::vector<float>& conv_out) const;

    // Run JUST the ConformerConvolution sub-module (everything AFTER norm_conv:
    // pointwise_conv1 -> GLU -> [pad_mask] -> depthwise_conv -> norm -> SiLU ->
    // pointwise_conv2) on `conv_in` (= norm_conv(residual), row-major [T,d_model]).
    // Test entry point for validating the conv module — including the layer_norm
    // and causal-conv variants — in ISOLATION from the surrounding attention.
    // `out` is row-major [T, d_model].
    void conv_module_forward(const std::vector<float>& conv_in, int T,
                             int valid_len, std::vector<float>& out) const;

private:
    const ModelLoader& ml_;
    int layer_idx_;
    int d_model_;
    int n_heads_;
    int ff_dim_;
    int conv_kernel_;
    std::string conv_norm_type_;  // "batch_norm" (offline) or "layer_norm" (streaming)
    bool conv_causal_ = false;    // causal depthwise conv pad (left k-1, right 0)
};

} // namespace pk
