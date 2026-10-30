#include "subsampling.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <vector>

namespace pk {

// Weights from the GGUF (loader context) are referenced DIRECTLY as graph
// leaves via the shared pk::clone_weight (backend.cpp) — they live in a CPU
// backend buffer (zero-copy). The conv kernels stay F32 (the converter never
// quantizes them); only out.weight is allowlisted and may be f16/q8_0, fed into
// ggml_mul_mat which dequantizes src0. GGUF ne is reverse of the torch shape ==
// ggml's [KW,KH,IC,OC] layout.

Subsampling::Subsampling(const ModelLoader& ml)
    : ml_(ml) {
    conv_channels_ = (int)ml.config().subsampling_conv_channels;
    d_model_       = (int)ml.config().d_model;
    causal_        = ml.config().causal_downsampling;
    trim_extra_frame_ = !(ml.config().unified.present && ml.config().unified.valid_length);
}

int Subsampling::valid_out_len(int T, int in_valid_frames) const {
    // The mel has T spatial frames, but the OFFLINE preprocessor reports a valid
    // length of T-1 (center-padding adds one extra trailing frame). Each of the
    // three stride-2, k=3 conv stages reduces the valid length via NeMo's
    // calc_length: out = floor((in + all_paddings - k)/s) + 1, all_paddings =
    // left+right.
    //
    // Non-causal (offline): symmetric pad (k-1)/2 each side -> all_paddings = 2.
    //   out = floor((in + 2 - 3)/2) + 1 = (in - 1)/2 + 1  (matches existing path).
    // Causal (causal_downsampling=True): left = k-1 = 2, right = stride-1 = 1 ->
    //   all_paddings = 3.  out = floor((in + 3 - 3)/2) + 1 = floor(in/2) + 1.
    // NeMo's calc_length runs in float; for these integer inputs floor matches
    // integer division, so we use integer arithmetic directly.
    //
    // Streaming (in_valid_frames >= 0): the chunk window is fully real audio, so
    // the entry valid length is the supplied count (typically T), NOT T-1.
    const int all_paddings = causal_ ? 3 : 2;
    int valid = (in_valid_frames >= 0) ? in_valid_frames : offline_entry_valid(T);
    for (int st = 0; st < 3; ++st)            // conv0, conv2, conv5
        valid = (valid + all_paddings - 3) / 2 + 1;
    return valid;
}

int Subsampling::subsample_len(int T) const {
    // Spatial output length after the three stride-2, k=3 conv stages, using
    // ggml conv2d's OH = floor((in + 2p - k)/s) + 1. Non-causal uses symmetric
    // pad p=1 (all_paddings=2); causal uses all_paddings=3. This mirrors the
    // valid_out_len recurrence but tracks the full (padded) spatial extent.
    const int all_paddings = causal_ ? 3 : 2;
    int x = T;
    for (int s = 0; s < 3; ++s) x = (x + all_paddings - 3) / 2 + 1;
    return x;
}

ggml_tensor* Subsampling::build_graph_batched(ggml_context* ctx,
                                      const float* mel,
                                      int n_mels, int T, int B, GraphInputPool& pool,
                                      int& out_Tp, std::vector<int>& out_valid,
                                      const std::vector<int>& valid_in) const {
    const int C = conv_channels_;
    const int F = n_mels;            // feature dim (80)
    const ModelLoader& ml = ml_;

    // Batched causal subsampling IS supported: the causal branch below applies
    // the leading ggml_pad_ext (lp1=2/rp1=1 on time) uniformly across the batch,
    // and the per-item trailing-pad time masking (mask_time on the batch axis)
    // plus the all_paddings=3 valid-length recurrence reproduce, per item, the
    // exact standalone causal boundary. A clip in a B>1 batch gets the same
    // transcript as the same clip transcribed standalone on the test clips (see
    // test_subsampling_batch_causal); float values are close but not guaranteed
    // bit-identical, since ggml picks different matmul kernels for B>1.

    // --- Input (host-side): ggml conv data layout is [W=feat, H=T, IC=1, N=B].
    // NeMo conv input is [B,1,T,feat] (H=T, W=feat). We must feed
    // x[(b*T + t)*F + f] = mel(item=b, feat=f, time=t). mel is per-item
    // feat-major [F,T] (mel[(b*F + f)*T + t]); transpose into time-major per
    // item in pool-owned storage (extra b*T block offset), feed as input.
    std::vector<float>& x_host = pool.alloc_f32((size_t)B * T * F);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < F; ++f)
                x_host[((size_t)b * T + t) * F + f] =
                    mel[((size_t)b * n_mels + f) * T + t];

    int64_t x_ne[4] = {F, T, 1, B};
    ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, x_ne,
                                            x_host.data(),
                                            x_host.size() * sizeof(float));

    // Subsampling conv padding. NeMo dw_striding uses k=3, s=2 on each stage; the
    // padding differs by model:
    //   non-causal (offline): symmetric (k-1)/2 = 1 on every side, applied
    //     directly via the conv's p0/p1 (byte-identical to the old path).
    //   causal (causal_downsampling=True, e.g. parakeet_realtime_eou_120m):
    //     NeMo CausalConv2D pads BOTH spatial axes (time H and feature W) with
    //     left = k-1 = 2, right = stride-1 = 1 (F.pad order (W_l,W_r,H_l,H_r)).
    //     ggml conv takes one symmetric p per axis, so for the causal case we pad
    //     explicitly with ggml_pad_ext (lp0/rp0 = W=feature, lp1/rp1 = H=time)
    //     and run the conv with p=0.
    const bool causal = causal_;
    auto pad_causal = [&](ggml_tensor* t) -> ggml_tensor* {
        return ggml_pad_ext(ctx, t, /*lp0*/2, /*rp0*/1, /*lp1*/2, /*rp1*/1,
                            0, 0, 0, 0);
    };

    // NeMo's MaskedConvSequential zeros the trailing (pad) time frames of the
    // conv input BEFORE every stage. We replicate this per-item, per-stage in
    // BOTH paths:
    //   - Causal: the right pad is +1, so the last valid output frame DOES read
    //     the trailing pad input frame; per-stage input masking is required for
    //     correctness even at B=1.
    //   - Non-causal (offline), B>1: a shorter clip is zero-padded to T_max, but
    //     after every conv stage bias+ReLU make the padded time region NON-ZERO,
    //     so the last valid output frame of a short item reads contaminated
    //     values instead of the clean conv zero-edge a standalone clip sees.
    //     Zeroing the trailing pad time frames before each stage reproduces the
    //     standalone boundary (the conv's own symmetric pad supplies clean zeros).
    // The mask is per-item: [1, H, 1, B], md[b*H + h] = (h < vt[b]) ? 1 : 0,
    // broadcasting over ne0 (W=feat) and ne2 (C).
    auto mask_time = [&](ggml_tensor* t, const std::vector<int>& vt) -> ggml_tensor* {
        const int H  = (int)t->ne[1];
        const int Bx = (int)t->ne[3];
        bool any = false;
        for (int b = 0; b < Bx; ++b) {
            int v = (b < (int)vt.size()) ? vt[b] : H;
            if (v < H) { any = true; break; }
        }
        if (!any) return t;
        std::vector<float>& md = pool.alloc_f32((size_t)Bx * H);
        for (int b = 0; b < Bx; ++b) {
            int v = (b < (int)vt.size()) ? vt[b] : H;
            for (int h = 0; h < H; ++h)
                md[(size_t)b * H + h] = (h < v) ? 1.0f : 0.0f;
        }
        int64_t m_ne[4] = {1, H, 1, Bx};
        ggml_tensor* tm = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, m_ne,
                              md.data(), md.size() * sizeof(float));
        return ggml_mul(ctx, t, tm); // broadcast over ne0(W), ne2(C)
    };
    // Per-item per-stage valid TIME lengths at the INPUT of each conv stage,
    // mirroring valid_out_len's recurrence (and the old single-item valid_t0/1/2).
    const int all_paddings = causal_ ? 3 : 2;
    std::vector<int> vt_stage0(B), vt_stage1(B), vt_stage2(B); // input of stage0/1/2
    for (int b = 0; b < B; ++b) {
        int vi = (b < (int)valid_in.size()) ? valid_in[b] : -1;
        int v0 = (vi >= 0) ? vi : offline_entry_valid(T);        // before stage 0
        int v1 = (v0 + all_paddings - 3) / 2 + 1;   // before stage 1 (after stage 0)
        int v2 = (v1 + all_paddings - 3) / 2 + 1;   // before stage 2 (after stage 1)
        vt_stage0[b] = v0;
        vt_stage1[b] = v1;
        vt_stage2[b] = v2;
    }

    // ---- Stage 1: full Conv2d(1 -> C, k=3, s=2) + ReLU ----
    // kernel conv.0.weight: torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,IC,OC].
    ggml_tensor* w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor* b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    x = mask_time(x, vt_stage0);      // zero trailing pad time frames (both paths)
    if (causal) {
        x = pad_causal(x);
        x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
    } else {
        x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
    }
    // x: ne [OW=F/2, OH=T/2, OC=C, 1]. Add bias broadcast over channels:
    // reshape bias to [1,1,C,1] so it broadcasts across W,H.
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // ---- Stages 2 & 3: depthwise(k=3,s=2,p=1,groups=C) + pointwise(k=1) + ReLU ----
    struct StageW { const char* dw_w; const char* dw_b; const char* pw_w; const char* pw_b; };
    const StageW stages[2] = {
        { "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
          "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias" },
        { "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
          "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias" },
    };
    const std::vector<int>* stage_valid_t[2] = {&vt_stage1, &vt_stage2};
    for (int si = 0; si < 2; ++si) {
        const StageW& s = stages[si];
        // Depthwise: weight torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,1,C].
        // ggml_conv_2d_dw_direct expects a:[KW,KH,1,C], b:[W,H,C,N].
        ggml_tensor* dww = clone_weight(ctx, ml, s.dw_w);
        ggml_tensor* dwb = clone_weight(ctx, ml, s.dw_b);
        x = mask_time(x, *stage_valid_t[si]); // zero trailing pad time frames (both paths)
        if (causal) {
            x = pad_causal(x);
            x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
        } else {
            x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
        }
        // x: ne [OW, OH, C, 1]. dw_direct keeps WHCN; make it contiguous so the
        // bias add and following ops see a standard layout.
        x = ggml_cont(ctx, x);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, dwb, 1, 1, C, 1));

        // Pointwise: weight torch [C,C,1,1] -> ggml ne [1,1,C,C] = [KW,KH,IC,OC].
        ggml_tensor* pww = clone_weight(ctx, ml, s.pw_w);
        ggml_tensor* pwb = clone_weight(ctx, ml, s.pw_b);
        x = ggml_conv_2d(ctx, pww, x, /*s0*/1, /*s1*/1, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, pwb, 1, 1, C, 1));
        x = ggml_relu(ctx, x);
    }

    // x: ne [F'=OW, T'=OH, C, B]. NeMo flatten (per item):
    //   [B,C,T',F'].transpose(1,2).reshape(B,T',C*F')
    // -> per time t, vector is channel-major: idx = c*F' + f.
    const int Fp = (int)x->ne[0]; // F'
    const int Tp = (int)x->ne[1]; // T'
    // Want contiguous [F', C, T', B] so flat[b] = t*(C*F') + c*F' + f.
    // current dims (0,1,2,3) = (F', T', C, B); permute to (F', C, T', B).
    ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    ggml_tensor* flat = ggml_reshape_3d(ctx, xp, (int64_t)C * Fp, Tp, B); // [C*F', T', B]

    // --- Length masking (faithful to NeMo MaskedConvSequential) ---
    // Valid output frames never read masked input frames (kernel reach stays
    // inside the valid region), so we can run the conv stack spatially and zero
    // the flattened conv output at frames >= valid_out[b] before the Linear.
    out_valid.assign(B, 0);
    bool any_masked = false;
    for (int b = 0; b < B; ++b) {
        int vi = (b < (int)valid_in.size()) ? valid_in[b] : -1;
        int vo = valid_out_len(T, vi);
        out_valid[b] = (vo > Tp) ? Tp : vo;
        if (vo < Tp) any_masked = true;
    }
    if (any_masked) {
        // [1, Tp, B] mask: md[b*Tp + t] = (t < valid_out[b]) ? 1 : 0; broadcasts
        // over ne0 (the C*F' feature axis).
        std::vector<float>& outmask = pool.alloc_f32((size_t)B * Tp);
        for (int b = 0; b < B; ++b) {
            // out_valid[b] == min(valid_out_len(T, vi), Tp); since this loop is
            // bounded by Tp, "t < out_valid[b]" matches the unclamped "t < vo".
            for (int t = 0; t < Tp; ++t)
                outmask[(size_t)b * Tp + t] = (t < out_valid[b]) ? 1.0f : 0.0f;
        }
        int64_t mk_ne[3] = {1, Tp, B};
        ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, mk_ne,
                                outmask.data(), outmask.size() * sizeof(float));
        flat = ggml_mul(ctx, flat, mask);
    }

    // ---- Linear out: torch [d_model, C*F'] -> ggml ne [C*F', d_model]. ----
    ggml_tensor* ow = clone_weight(ctx, ml, "encoder.pre_encode.out.weight");
    ggml_tensor* ob = clone_weight(ctx, ml, "encoder.pre_encode.out.bias");
    ggml_tensor* y = ggml_mul_mat(ctx, ow, flat); // [d_model, T', B]
    y = ggml_add(ctx, y, ob);                     // broadcast bias [d_model] over T',B

    out_Tp = Tp;
    return y; // ne [d_model, T', B] contiguous.
}

ggml_tensor* Subsampling::build_graph(ggml_context* ctx,
                                      const std::vector<float>& mel,
                                      int n_mels, int T, GraphInputPool& pool,
                                      int& out_Tp, int& out_valid,
                                      int in_valid_frames) const {
    const int C = conv_channels_;
    const int F = n_mels;            // feature dim (80)
    const ModelLoader& ml = ml_;

    // --- Input (host-side): ggml conv data layout is [W=feat, H=T, IC=1, N=1].
    // NeMo conv input is [B,1,T,feat] (H=T, W=feat). We must feed
    // x[t*F + f] = mel(feat=f, time=t). mel is feat-major [F,T] (mel[m*T + t])
    // so transpose into time-major in pool-owned storage, then feed as input.
    std::vector<float>& x_host = pool.alloc_f32((size_t)F * T);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < F; ++f)
            x_host[(size_t)t * F + f] = mel[(size_t)f * T + t];

    int64_t x_ne[4] = {F, T, 1, 1};
    ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, x_ne,
                                            x_host.data(),
                                            x_host.size() * sizeof(float));

    // Subsampling conv padding. NeMo dw_striding uses k=3, s=2 on each stage; the
    // padding differs by model:
    //   non-causal (offline): symmetric (k-1)/2 = 1 on every side, applied
    //     directly via the conv's p0/p1 (byte-identical to the old path).
    //   causal (causal_downsampling=True, e.g. parakeet_realtime_eou_120m):
    //     NeMo CausalConv2D pads BOTH spatial axes (time H and feature W) with
    //     left = k-1 = 2, right = stride-1 = 1 (F.pad order (W_l,W_r,H_l,H_r)).
    //     ggml conv takes one symmetric p per axis, so for the causal case we pad
    //     explicitly with ggml_pad_ext (lp0/rp0 = W=feature, lp1/rp1 = H=time)
    //     and run the conv with p=0.
    const bool causal = causal_;
    auto pad_causal = [&](ggml_tensor* t) -> ggml_tensor* {
        return ggml_pad_ext(ctx, t, /*lp0*/2, /*rp0*/1, /*lp1*/2, /*rp1*/1,
                            0, 0, 0, 0);
    };

    // NeMo's MaskedConvSequential zeros the trailing (pad) time frames of the
    // conv input BEFORE every stage. For the SYMMETRIC (offline) path a valid
    // output frame never reaches a masked input frame (centred kernel), so the
    // old code masks only the flattened output and stays byte-identical — keep
    // that. For the CAUSAL path the right pad is +1, so the last valid output
    // frame DOES read the trailing pad input frame; replicate the per-stage
    // input masking via a [1,H,1,1] mask broadcast over W (feat) and C.
    auto mask_time = [&](ggml_tensor* t, int valid_t) -> ggml_tensor* {
        const int H = (int)t->ne[1];
        if (valid_t >= H) return t;
        std::vector<float>& md = pool.alloc_f32(H);
        for (int h = 0; h < H; ++h) md[h] = (h < valid_t) ? 1.0f : 0.0f;
        int64_t m_ne[4] = {1, H, 1, 1};
        ggml_tensor* tm = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, m_ne,
                              md.data(), md.size() * sizeof(float));
        return ggml_mul(ctx, t, tm); // broadcast over ne0(W), ne2(C), ne3
    };
    int valid_t0 = (in_valid_frames >= 0) ? in_valid_frames : offline_entry_valid(T); // before stage 0
    int valid_t1 = (valid_t0 + 3 - 3) / 2 + 1;  // before stage 2 (after stage 0)
    int valid_t2 = (valid_t1 + 3 - 3) / 2 + 1;  // before stage 5 (after stage 2)

    // ---- Stage 1: full Conv2d(1 -> C, k=3, s=2) + ReLU ----
    // kernel conv.0.weight: torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,IC,OC].
    ggml_tensor* w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor* b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    if (causal) {
        x = mask_time(x, valid_t0);   // zero trailing pad mel frames
        x = pad_causal(x);
        x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
    } else {
        x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
    }
    // x: ne [OW=F/2, OH=T/2, OC=C, 1]. Add bias broadcast over channels:
    // reshape bias to [1,1,C,1] so it broadcasts across W,H.
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // ---- Stages 2 & 3: depthwise(k=3,s=2,p=1,groups=C) + pointwise(k=1) + ReLU ----
    struct StageW { const char* dw_w; const char* dw_b; const char* pw_w; const char* pw_b; };
    const StageW stages[2] = {
        { "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
          "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias" },
        { "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
          "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias" },
    };
    int stage_valid_t[2] = {valid_t1, valid_t2};
    for (int si = 0; si < 2; ++si) {
        const StageW& s = stages[si];
        // Depthwise: weight torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,1,C].
        // ggml_conv_2d_dw_direct expects a:[KW,KH,1,C], b:[W,H,C,N].
        ggml_tensor* dww = clone_weight(ctx, ml, s.dw_w);
        ggml_tensor* dwb = clone_weight(ctx, ml, s.dw_b);
        if (causal) {
            x = mask_time(x, stage_valid_t[si]); // zero trailing pad time frames
            x = pad_causal(x);
            x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
        } else {
            x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
        }
        // x: ne [OW, OH, C, 1]. dw_direct keeps WHCN; make it contiguous so the
        // bias add and following ops see a standard layout.
        x = ggml_cont(ctx, x);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, dwb, 1, 1, C, 1));

        // Pointwise: weight torch [C,C,1,1] -> ggml ne [1,1,C,C] = [KW,KH,IC,OC].
        ggml_tensor* pww = clone_weight(ctx, ml, s.pw_w);
        ggml_tensor* pwb = clone_weight(ctx, ml, s.pw_b);
        x = ggml_conv_2d(ctx, pww, x, /*s0*/1, /*s1*/1, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, pwb, 1, 1, C, 1));
        x = ggml_relu(ctx, x);
    }

    // x: ne [F'=OW, T'=OH, C, 1]. NeMo flatten:
    //   [B,C,T',F'].transpose(1,2).reshape(B,T',C*F')
    // -> per time t, vector is channel-major: idx = c*F' + f.
    const int Fp = (int)x->ne[0]; // F'
    const int Tp = (int)x->ne[1]; // T'
    // Want contiguous [F', C, T', 1] so flat = t*(C*F') + c*F' + f.
    // current dims (0,1,2,3) = (F', T', C, 1); permute to (F', C, T', 1).
    ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    ggml_tensor* flat = ggml_reshape_2d(ctx, xp, (int64_t)C * Fp, Tp); // [C*F', T']

    // --- Length masking (faithful to NeMo MaskedConvSequential) ---
    // Valid output frames never read masked input frames (kernel reach stays
    // inside the valid region), so we can run the conv stack spatially and zero
    // the flattened conv output at frames >= valid_out before the Linear.
    const int valid_out = valid_out_len(T, in_valid_frames);
    if (valid_out < Tp) {
        std::vector<float>& outmask = pool.alloc_f32(Tp);
        for (int t = 0; t < Tp; ++t) outmask[t] = (t < valid_out) ? 1.0f : 0.0f;
        int64_t mk_ne[2] = {1, Tp};
        ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mk_ne,
                                outmask.data(), outmask.size() * sizeof(float));
        flat = ggml_mul(ctx, flat, mask);
    }

    // ---- Linear out: torch [d_model, C*F'] -> ggml ne [C*F', d_model]. ----
    ggml_tensor* ow = clone_weight(ctx, ml, "encoder.pre_encode.out.weight");
    ggml_tensor* ob = clone_weight(ctx, ml, "encoder.pre_encode.out.bias");
    ggml_tensor* y = ggml_mul_mat(ctx, ow, flat); // [d_model, T']
    y = ggml_add(ctx, y, ob);                     // broadcast bias [d_model] over T'

    out_Tp = Tp;
    out_valid = (valid_out > Tp) ? Tp : valid_out;
    return y; // ne [d_model, T'] contiguous -> row-major [T', d_model].
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model) const {
    int valid_len_unused = 0;
    forward(mel, n_mels, T, out, Tout, d_model, valid_len_unused, -1);
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model,
                          int& valid_len) const {
    forward(mel, n_mels, T, out, Tout, d_model, valid_len, -1);
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model,
                          int& valid_len, int in_valid_frames) const {
    // Thin wrapper over the graph-builder: build JUST the subsampling sub-graph
    // and compute it on the persistent Backend. Used by the unit test and the
    // streaming path (the offline encoder uses build_graph directly, fused).
    int Tp = 0, valid = 0;
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            return build_graph(ctx, mel, n_mels, T, pool, Tp, valid, in_valid_frames);
        }, out);
    assert(ok && "subsampling graph failed");
    (void)ok;
    Tout = Tp;
    d_model = d_model_;
    valid_len = valid;
}

void Subsampling::forward_tiled(const std::vector<float>& mel, int n_mels, int T,
                                int tile_out_frames, std::vector<float>& out,
                                int& Tout, int& d_model, int& valid_len) const {
    const int Tp = subsample_len(T);
    d_model = d_model_;
    Tout = Tp;
    const int vo = valid_out_len(T, -1);
    valid_len = (vo > Tp) ? Tp : vo;

    // TODO: causal tiling needs the causal phase mapping; offline causal long-audio
    // is not a current target. Fall back to the single, untiled graph (== forward()).
    if (causal_ || tile_out_frames <= 0) {
        int t_unused = 0, dm_unused = 0, vl_unused = 0;
        forward(mel, n_mels, T, out, t_unused, dm_unused, vl_unused, -1);
        return;
    }

    // Non-causal symmetric-pad tiling. Receptive field is +-7 mel frames; output
    // frame o has mel center 8*o. Window start ws is a multiple of 8, so the
    // window-output frame j maps to global o = j + ws/8. A generous halo H=64 mel
    // frames (>> RF) keeps every emitted frame's RF inside the fed window (or, at
    // true utterance edges, inside build_graph's own pad which equals the boundary),
    // so every kept frame equals the full-utterance result.
    const int H = 64; // multiple of 8, >> receptive field (7)
    out.assign((size_t)Tp * d_model_, 0.0f);

    for (int os = 0; os < Tp; os += tile_out_frames) {
        const int oe = (os + tile_out_frames < Tp) ? (os + tile_out_frames) : Tp;
        int ws = 8 * os - H; if (ws < 0) ws = 0;       // multiple of 8 (clamp keeps it)
        int we = 8 * oe + H; if (we > T) we = T;
        const int Lw = we - ws;

        // Slice window mel, feat-major [n_mels, Lw].
        std::vector<float> win((size_t)n_mels * Lw);
        for (int f = 0; f < n_mels; ++f)
            for (int t = ws; t < we; ++t)
                win[(size_t)f * Lw + (t - ws)] = mel[(size_t)f * T + t];

        // Run the single-item graph on the window. The window is all-real, so pass
        // in_valid_frames = Lw (no trailing mask inside the tile).
        std::vector<float> win_out;
        int Tpw = 0, valid_w = 0;
        GraphInputPool pool;
        bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                return build_graph(ctx, win, n_mels, Lw, pool, Tpw, valid_w, Lw);
            }, win_out);
        assert(ok && "subsampling tile graph failed");
        (void)ok;

        // Window-output frame j has mel center ws+8*j -> global o = j + ws/8.
        const int j0 = ws / 8; // exact: ws is a multiple of 8
        for (int o = os; o < oe; ++o) {
            const int j = o - j0;
            assert(j >= 0 && j < Tpw && "tile frame out of window range");
            std::memcpy(&out[(size_t)o * d_model_],
                        &win_out[(size_t)j * d_model_],
                        (size_t)d_model_ * sizeof(float));
        }
    }
}

} // namespace pk
