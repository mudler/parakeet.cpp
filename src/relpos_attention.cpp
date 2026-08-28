#include "relpos_attention.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace pk {

// Weights from the loader are referenced DIRECTLY as graph leaves via the shared
// pk::clone_weight (backend.cpp; zero-copy via the loader's CPU backend buffer).
// Allowlisted attention linears (linear_q/k/v/out/pos.weight) may be f16/q8_0
// and are fed into ggml_mul_mat, which dequantizes src0 on the fly. pos_bias_u/v
// and every other weight stay F32. A std::string-name overload keeps the call
// sites (pre + suffix) unchanged.
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const std::string& name) {
    return pk::clone_weight(ctx, ml, name.c_str());
}

RelPosAttention::RelPosAttention(const ModelLoader& ml, int layer_idx)
    : ml_(ml), layer_idx_(layer_idx) {
    d_model_ = (int)ml.config().d_model;
    n_heads_ = (int)ml.config().n_heads;
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    d_head_ = d_model_ / n_heads_;
    // Chunked-limited attention (NeMo att_context_style=="chunked_limited",
    // e.g. parakeet_realtime_eou_120m-v1 with att_context_size=[70,1]). The
    // offline forward applies the SAME additive -inf mask NeMo builds in
    // ConformerEncoder._create_masks. Offline models use "regular" (full context).
    chunked_limited_ = (ml.config().att_context_style == "chunked_limited" &&
                        ml.config().att_context_right >= 0);
    att_left_  = ml.config().att_context_left;
    att_right_ = ml.config().att_context_right;
}

ggml_tensor* RelPosAttention::build_graph(ggml_context* ctx, ggml_tensor* xt,
                                          int T, ggml_tensor* pe, int pos_len,
                                          int valid_len,
                                          GraphInputPool& pool,
                                          int att_left, int att_right) const {
    // Scalar (B=1) builder: the verbatim v1 2-D/3D relative-position attention
    // graph. The single-clip conformer layer routes here so B=1 runs the lean
    // graph and is bit-exact with v1. build_graph_batched below serves B>1.
    const int D  = d_model_;
    const int H  = n_heads_;
    const int dk = d_head_;
    const float scale = 1.0f / std::sqrt((float)dk);
    assert(pos_len == 2 * T - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
    const ModelLoader& ml = ml_;

    // ---- linear projections (nn.Linear: ggml W ne=[in,out]) ----
    // The bias is added only when requested AND present: NeMo configures the
    // attention linears with bias=False in some checkpoints
    // (parakeet-tdt-0.6b-v2/-v3) and bias=True in others (110m).
    auto linear = [&](const char* w, const char* b, ggml_tensor* in) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + w);
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);  // [out, *]
        if (b && ml.tensor(pre + b)) {
            ggml_tensor* B = clone_weight(ctx, ml, pre + b);
            y = ggml_add(ctx, y, B);                // broadcast [out] over cols
        }
        return y;
    };
    ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt); // [D, T]
    ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt); // [D, T]
    ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt); // [D, T]
    ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);       // [D, P]

    // ---- split into heads: [D, *] -> [dk, H, *] -> [dk, *, H] ----
    auto to_heads = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_3d(ctx, t, dk, H, n);                 // [dk, H, n]
        t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));  // [dk, n, H]
        return t;
    };
    ggml_tensor* qh = to_heads(q, T);        // [dk, T, H]
    ggml_tensor* kh = to_heads(k, T);        // [dk, T, H]
    ggml_tensor* vh = to_heads(v, T);        // [dk, T, H]
    ggml_tensor* ph = to_heads(p, pos_len);  // [dk, P, H]

    // ---- pos_bias_u/v: ne [dk, H] -> [dk, 1, H] to broadcast over T ----
    ggml_tensor* bu = clone_weight(ctx, ml, pre + "pos_bias_u"); // [dk, H]
    ggml_tensor* bv = clone_weight(ctx, ml, pre + "pos_bias_v"); // [dk, H]
    bu = ggml_reshape_3d(ctx, bu, dk, 1, H);
    bv = ggml_reshape_3d(ctx, bv, dk, 1, H);
    ggml_tensor* qu = ggml_add(ctx, qh, bu); // [dk, T, H]
    ggml_tensor* qv = ggml_add(ctx, qh, bv); // [dk, T, H]

    // ---- ac = q_u @ k^T : ggml_mul_mat([dk,T,H],[dk,T,H]) -> [T_k, T_q, H] ----
    ggml_tensor* ac = ggml_mul_mat(ctx, kh, qu); // [T(key), T(query), H]

    // ---- bd = q_v @ p^T -> [P(pos), T(query), H], then rel_shift -> [T,T,H] ----
    ggml_tensor* bd = ggml_mul_mat(ctx, ph, qv); // [P, T, H]
    bd = ggml_pad_ext(ctx, bd, /*lp0*/1, /*rp0*/0, 0,0, 0,0, 0,0); // [P+1=2T, T, H]
    bd = ggml_reshape_3d(ctx, bd, T, 2 * T, H);                    // [T, 2T, H]
    bd = ggml_view_3d(ctx, bd, T, 2 * T - 1, H,
                      bd->nb[1], bd->nb[2], bd->nb[1]);            // [T, 2T-1, H]
    bd = ggml_cont(ctx, bd);
    bd = ggml_reshape_3d(ctx, bd, 2 * T - 1, T, H);               // [2T-1, T, H]
    bd = ggml_view_3d(ctx, bd, T, T, H, bd->nb[1], bd->nb[2], 0);
    bd = ggml_cont(ctx, bd);

    // ---- scores = ac + bd ; softmax(scores*scale + mask) ----
    ggml_tensor* scores = ggml_add(ctx, ac, bd); // [T_k, T_q, H]

    // Additive mask [T_k, T_q]: 0 where query qi may attend to key kj, -inf
    // otherwise. (1) pad mask: key kj valid iff kj < valid_len. (2) chunked-
    // limited window for streaming models. See header / NeMo _create_masks.
    // PADDED query rows (qi >= valid_len) are left fully unmasked: with a
    // limited window a deeply padded row can have its ENTIRE row masked, and
    // softmax over all -inf is NaN — which the post-softmax query mask does NOT
    // clear (NaN*0 == NaN) and which poisons every softmax row of the next
    // layer through the key path. Their finite garbage is zeroed below.
    const int chunk_size  = chunked_limited_ ? (att_right_ + 1) : 0;
    const int left_chunks = (chunked_limited_ && chunk_size > 0)
                            ? (att_left_ / chunk_size) : 0;
    std::vector<float>& mask_host = pool.alloc_f32((size_t)T * T);
    {
        float* md = mask_host.data();
        const float ninf = -INFINITY;
        for (int qi = 0; qi < T; ++qi) {
            if (qi >= valid_len) {  // dead (padded) query row: see above
                for (int kj = 0; kj < T; ++kj) md[(size_t)qi * T + kj] = 0.0f;
                continue;
            }
            const int cq = chunked_limited_ ? (qi / chunk_size) : 0;
            for (int kj = 0; kj < T; ++kj) {
                bool ok = (kj < valid_len);
                if (ok && chunked_limited_) {
                    const int ck = kj / chunk_size;
                    const int diff = cq - ck;
                    ok = (diff >= 0 && diff <= left_chunks);
                }
                // Symmetric sliding window (NeMo rel_pos_local_attn): keep only
                // keys within [qi-att_left, qi+att_right].
                if (ok && att_left >= 0) {
                    const int rel = qi - kj;
                    ok = (rel <= att_left) && (rel >= -att_right);
                }
                md[(size_t)qi * T + kj] = ok ? 0.0f : ninf;
            }
        }
    }
    int64_t mask_ne[2] = {T, T};
    ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mask_ne,
                            mask_host.data(), mask_host.size() * sizeof(float));
    ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // [T_k, T_q, H]

    // ---- context = attn @ v -> [dk, T_q, H] ----
    ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [T_k, dk, H]
    ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn); // [dk, T_q, H]

    // ---- concat heads: [dk, T, H] -> [dk, H, T] -> [D, T] ----
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3)); // [dk, H, T]
    merged = ggml_reshape_2d(ctx, merged, D, T); // [D, T]

    // Zero the context for PADDED query rows (NeMo masks padded query rows fully
    // -> output reduces to linear_out.bias). Apply a query-row mask [1, T].
    if (valid_len < T) {
        std::vector<float>& qmask_host = pool.alloc_f32(T);
        for (int qi = 0; qi < T; ++qi)
            qmask_host[qi] = (qi < valid_len) ? 1.0f : 0.0f;
        int64_t qm_ne[2] = {1, T};
        ggml_tensor* qmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, qm_ne,
                                 qmask_host.data(), qmask_host.size() * sizeof(float));
        merged = ggml_mul(ctx, merged, qmask); // broadcast over D
    }

    // ---- output projection ----
    return linear("linear_out.weight", "linear_out.bias", merged); // [D, T]
}

ggml_tensor* RelPosAttention::build_graph_batched(
        ggml_context* ctx, ggml_tensor* xt, int T, int B, ggml_tensor* pe,
        int pos_len, const std::vector<int>& valid_len,
        GraphInputPool& pool) const {
    const int D  = d_model_;
    const int H  = n_heads_;
    const int dk = d_head_;
    const float scale = 1.0f / std::sqrt((float)dk);
    assert(pos_len == 2 * T - 1);
    assert((int)valid_len.size() == B);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
    const ModelLoader& ml = ml_;

    // ---- linear projections (nn.Linear: ggml W ne=[in,out]) ----
    // The bias is added only when requested AND present: NeMo configures the
    // attention linears with bias=False in some checkpoints
    // (parakeet-tdt-0.6b-v2/-v3) and bias=True in others (110m).
    auto linear = [&](const char* w, const char* b, ggml_tensor* in) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + w);
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);  // [out, *]
        if (b && ml.tensor(pre + b)) {
            ggml_tensor* B = clone_weight(ctx, ml, pre + b);
            y = ggml_add(ctx, y, B);                // broadcast [out] over cols
        }
        return y;
    };
    // xt is [D, T, B]; mul_mat batches over ne2 -> q/k/v are [D, T, B]. pe is
    // shared [D, P] (NO batch) -> p is [D, P].
    ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt); // [D, T, B]
    ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt); // [D, T, B]
    ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt); // [D, T, B]
    ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);       // [D, P]

    // ---- split into heads (batched): [D, n, B] -> [dk, H, n, B] -> [dk, n, H, B] ----
    auto to_heads_b = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_4d(ctx, t, dk, H, n, B);              // [dk, H, n, B]
        t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));  // [dk, n, H, B]
        return t;
    };
    // p is shared (no batch) -> keep the 3D head-split: [dk, P, H].
    auto to_heads = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_3d(ctx, t, dk, H, n);                 // [dk, H, n]
        t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));  // [dk, n, H]
        return t;
    };
    ggml_tensor* qh = to_heads_b(q, T);      // [dk, T, H, B]
    ggml_tensor* kh = to_heads_b(k, T);      // [dk, T, H, B]
    ggml_tensor* vh = to_heads_b(v, T);      // [dk, T, H, B]
    ggml_tensor* ph = to_heads(p, pos_len);  // [dk, P, H]  (ne3=1, broadcast over B)

    // ---- pos_bias_u/v: ne [dk, H] -> [dk, 1, H, 1] to broadcast over T and B ----
    ggml_tensor* bu = clone_weight(ctx, ml, pre + "pos_bias_u"); // [dk, H]
    ggml_tensor* bv = clone_weight(ctx, ml, pre + "pos_bias_v"); // [dk, H]
    bu = ggml_reshape_4d(ctx, bu, dk, 1, H, 1);
    bv = ggml_reshape_4d(ctx, bv, dk, 1, H, 1);
    ggml_tensor* qu = ggml_add(ctx, qh, bu); // [dk, T, H, B]
    ggml_tensor* qv = ggml_add(ctx, qh, bv); // [dk, T, H, B]

    // ---- ac = q_u @ k^T : mul_mat([dk,T,H,B],[dk,T,H,B]) -> [T_k, T_q, H, B] ----
    ggml_tensor* ac = ggml_mul_mat(ctx, kh, qu); // [T(key), T(query), H, B]

    // ---- bd = q_v @ p^T -> [P, T_q, H, B], then rel_shift -> [T, T, H, B] ----
    // ph is [dk, P, H] (ne3=1) and broadcasts over the batch (ne3=B) of qv.
    ggml_tensor* bd = ggml_mul_mat(ctx, ph, qv); // [P, T, H, B]
    // 4D rel-shift: identical ne0/ne1 stride+offset arithmetic as the 3D path,
    // with the batch axis threaded through every reshape/view via ne3=B and
    // nb[3]. ne2/ne3 are passive (no offset on them).
    bd = ggml_pad_ext(ctx, bd, /*lp0*/1, /*rp0*/0, 0,0, 0,0, 0,0); // [2T, T, H, B]
    bd = ggml_reshape_4d(ctx, bd, T, 2 * T, H, B);                 // [T, 2T, H, B]
    bd = ggml_view_4d(ctx, bd, T, 2 * T - 1, H, B,
                      bd->nb[1], bd->nb[2], bd->nb[3], bd->nb[1]);  // [T, 2T-1, H, B]
    bd = ggml_cont(ctx, bd);
    bd = ggml_reshape_4d(ctx, bd, 2 * T - 1, T, H, B);            // [2T-1, T, H, B]
    bd = ggml_view_4d(ctx, bd, T, T, H, B,
                      bd->nb[1], bd->nb[2], bd->nb[3], 0);         // [T, T, H, B]
    bd = ggml_cont(ctx, bd);

    // ---- scores = ac + bd ; softmax(scores*scale + mask) ----
    ggml_tensor* scores = ggml_add(ctx, ac, bd); // [T_k, T_q, H, B]

    // Per-item additive mask [T_k, T_q, 1, B]: 0 where query qi may attend to key
    // kj, -inf otherwise. (1) pad mask: key kj valid iff kj < valid_len[b].
    // (2) chunked-limited window for streaming models. See NeMo _create_masks.
    // Mask shape is [T, T, 1, B] (ne2=1) so soft_max_ext broadcasts it over the
    // head axis (ne2=H) while indexing per item on ne3=B. Verified against the
    // ggml CPU kernel ggml_compute_forward_soft_max_f32: it reads the mask at
    // i12 = i02 % ne12 (head, here ne12=1 -> always 0) and i13 = i03 % ne13
    // (batch, here ne13=B -> exact per-item), and ggml_soft_max_impl asserts
    // a->ne[2] % mask->ne[2] == 0 and a->ne[3] % mask->ne[3] == 0.
    // PADDED query rows (qi >= valid_len[b]) are left fully unmasked — with the
    // chunked-limited window a deeply padded row (pad tail > att_left) has its
    // ENTIRE row masked, softmax over all -inf is NaN, the post-softmax query
    // mask keeps NaN (NaN*0), and the NaN key columns then poison EVERY softmax
    // row of the next layer. This corrupted whole items of a heterogeneous-
    // length batch on streaming models (clips differing by more than ~att_left
    // encoder frames decoded empty). Their finite garbage is zeroed below.
    const int chunk_size  = chunked_limited_ ? (att_right_ + 1) : 0;
    const int left_chunks = (chunked_limited_ && chunk_size > 0)
                            ? (att_left_ / chunk_size) : 0;
    std::vector<float>& mask_host = pool.alloc_f32((size_t)B * T * T);
    {
        float* md = mask_host.data();
        const float ninf = -INFINITY;
        for (int b = 0; b < B; ++b) {
            const int vl = valid_len[b];
            for (int qi = 0; qi < T; ++qi) {
                if (qi >= vl) {  // dead (padded) query row: see above
                    for (int kj = 0; kj < T; ++kj)
                        md[(size_t)b * T * T + (size_t)qi * T + kj] = 0.0f;
                    continue;
                }
                const int cq = chunked_limited_ ? (qi / chunk_size) : 0;
                for (int kj = 0; kj < T; ++kj) {
                    bool ok = (kj < vl);
                    if (ok && chunked_limited_) {
                        const int ck = kj / chunk_size;
                        const int diff = cq - ck;
                        ok = (diff >= 0 && diff <= left_chunks);
                    }
                    md[(size_t)b * T * T + (size_t)qi * T + kj] = ok ? 0.0f : ninf;
                }
            }
        }
    }
    int64_t mask_ne[4] = {T, T, 1, B};
    ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, mask_ne,
                            mask_host.data(), mask_host.size() * sizeof(float));
    ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // [T_k, T_q, H, B]

    // ---- context = attn @ v -> [dk, T_q, H, B] ----
    ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [T_k, dk, H, B]
    ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn); // [dk, T_q, H, B]

    // ---- concat heads: [dk, T, H, B] -> [dk, H, T, B] -> [D, T, B] ----
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3)); // [dk, H, T, B]
    merged = ggml_reshape_3d(ctx, merged, D, T, B); // [D, T, B]

    // Zero the context for PADDED query rows (NeMo masks padded query rows fully
    // -> output reduces to linear_out.bias). Apply a per-item query-row mask
    // [1, T, B] (broadcast over D). Emit only when some item has valid_len < T.
    bool any_pad = false;
    for (int b = 0; b < B; ++b) any_pad = any_pad || (valid_len[b] < T);
    if (any_pad) {
        std::vector<float>& qmask_host = pool.alloc_f32((size_t)B * T);
        for (int b = 0; b < B; ++b)
            for (int qi = 0; qi < T; ++qi)
                qmask_host[(size_t)b * T + qi] = (qi < valid_len[b]) ? 1.0f : 0.0f;
        int64_t qm_ne[3] = {1, T, B};
        ggml_tensor* qmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, qm_ne,
                                 qmask_host.data(), qmask_host.size() * sizeof(float));
        merged = ggml_mul(ctx, merged, qmask); // broadcast over D
    }

    // ---- output projection ----
    return linear("linear_out.weight", "linear_out.bias", merged); // [D, T, B]
}

ggml_tensor* RelPosAttention::build_graph_batched_local(
        ggml_context* ctx, ggml_tensor* xt, int T, int B, ggml_tensor* pe,
        int pos_len, const std::vector<int>& valid_len,
        int att_left, int att_right, GraphInputPool& pool) const {
    const int D = d_model_, H = n_heads_, dk = d_head_;
    const int P = pos_len;
    const float scale = 1.0f / std::sqrt((float)dk);
    assert(att_left >= 0 && att_right >= 0);
    assert(P == att_left + att_right + 1);
    assert((int)valid_len.size() == B);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
    const ModelLoader& ml = ml_;
    auto linear = [&](const char* wn, const char* bn, ggml_tensor* in) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + wn);
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bn && ml.tensor(pre + bn)) y = ggml_add(ctx, y, clone_weight(ctx, ml, pre + bn));
        return y;
    };
    ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt); // [D, T, B]
    ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt);
    ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt);
    ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);       // [D, P]
    auto to_heads_b = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_4d(ctx, t, dk, H, n, B);
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H, B]
    };
    auto to_heads = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_3d(ctx, t, dk, H, n);
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H]
    };
    ggml_tensor* qh = to_heads_b(q, T), *kh = to_heads_b(k, T);
    ggml_tensor* vh = to_heads_b(v, T), *php = to_heads(p, P);     // ph shared (ne3=1)
    ggml_tensor* bu = ggml_reshape_4d(ctx, clone_weight(ctx, ml, pre + "pos_bias_u"), dk, 1, H, 1);
    ggml_tensor* bv = ggml_reshape_4d(ctx, clone_weight(ctx, ml, pre + "pos_bias_v"), dk, 1, H, 1);
    ggml_tensor* qu = ggml_add(ctx, qh, bu);  // [dk, T, H, B]
    ggml_tensor* qv = ggml_add(ctx, qh, bv);  // [dk, T, H, B]

    // Pad K/V along time (ne1); view offset c -> key (t - att_left + c).
    ggml_tensor* kpad = ggml_pad_ext(ctx, kh, 0,0, att_left,att_right, 0,0, 0,0); // [dk, T+P-1, H, B]
    ggml_tensor* vpad = ggml_pad_ext(ctx, vh, 0,0, att_left,att_right, 0,0, 0,0);

    // Banded content scores ac[c, t, H, B]; stack on ne0=c.
    ggml_tensor* ac = nullptr;
    for (int c = 0; c < P; ++c) {
        ggml_tensor* kc = ggml_view_4d(ctx, kpad, dk, T, H, B,
                              kpad->nb[1], kpad->nb[2], kpad->nb[3], (size_t)c * kpad->nb[1]);
        ggml_tensor* acc = ggml_sum_rows(ctx, ggml_mul(ctx, qu, kc)); // [1, T, H, B]
        ac = ac ? ggml_concat(ctx, ac, acc, 0) : acc;
    }
    ggml_tensor* bd = ggml_mul_mat(ctx, php, qv);  // [P, T, H, B]  (php broadcasts over B)
    ggml_tensor* scores = ggml_add(ctx, ac, bd);   // [P, T, H, B]

    // Per-item band mask [P, T, 1, B]. Padded query rows whose entire band lies
    // beyond vl are left unmasked (all -inf -> softmax NaN -> NaN*0 stays NaN
    // and poisons later layers); the query mask below zeroes their output.
    std::vector<float>& mh = pool.alloc_f32((size_t)B * T * P);
    for (int b = 0; b < B; ++b) {
        const int vl = valid_len[b];
        for (int t = 0; t < T; ++t) {
            const bool dead_row = (t - att_left >= vl);
            for (int c = 0; c < P; ++c) {
                const int key = t - att_left + c;
                mh[(size_t)b * T * P + (size_t)t * P + c] =
                    (dead_row || (key >= 0 && key < vl)) ? 0.0f : -INFINITY;
            }
        }
    }
    int64_t mne[4] = {P, T, 1, B};
    ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, mne,
                            mh.data(), mh.size() * sizeof(float));
    ggml_tensor* prob = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // softmax over c

    // context[dk, t, H, B] = sum_c prob[c, t] * v[t-att_left+c].
    ggml_tensor* context = nullptr;
    for (int c = 0; c < P; ++c) {
        ggml_tensor* vc = ggml_view_4d(ctx, vpad, dk, T, H, B,
                              vpad->nb[1], vpad->nb[2], vpad->nb[3], (size_t)c * vpad->nb[1]);
        ggml_tensor* pc = ggml_view_4d(ctx, prob, 1, T, H, B,
                              prob->nb[1], prob->nb[2], prob->nb[3], (size_t)c * prob->nb[0]); // [1,T,H,B]
        ggml_tensor* term = ggml_mul(ctx, vc, pc);
        context = context ? ggml_add(ctx, context, term) : term;
    }
    // Merge heads [dk, T, H, B] -> [dk, H, T, B] -> [D, T, B].
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));
    merged = ggml_reshape_3d(ctx, merged, D, T, B);
    bool any_pad = false;
    for (int b = 0; b < B; ++b) any_pad = any_pad || (valid_len[b] < T);
    if (any_pad) {
        std::vector<float>& qm = pool.alloc_f32((size_t)B * T);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) qm[(size_t)b * T + t] = (t < valid_len[b]) ? 1.0f : 0.0f;
        int64_t qne[3] = {1, T, B};
        ggml_tensor* qmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, qne,
                                 qm.data(), qm.size() * sizeof(float));
        merged = ggml_mul(ctx, merged, qmask);
    }
    return linear("linear_out.weight", "linear_out.bias", merged); // [D, T, B]
}

void RelPosAttention::forward(const std::vector<float>& x, int T,
                              const std::vector<float>& pos_emb, int pos_len,
                              int valid_len,
                              std::vector<float>& out) const {
    const int D = d_model_;
    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);
    assert(pos_len == 2 * T - 1);

    // Thin wrapper over the graph-builder: build JUST the attention sub-graph
    // with x/pos_emb fed as inputs and compute it on the persistent Backend.
    // Used by the unit test (the fused conformer layer uses build_graph).
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {D, T};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  x.data(), (size_t)T * D * sizeof(float));
            int64_t pe_ne[2] = {D, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pos_emb.data(), (size_t)pos_len * D * sizeof(float));
            return build_graph(ctx, xt, T, pe, pos_len, valid_len, pool);
        }, out);
    assert(ok && "relpos attention graph failed");
    (void)ok;
}

ggml_tensor* RelPosAttention::build_graph_local(ggml_context* ctx, ggml_tensor* xt,
                                                int T, ggml_tensor* pe, int pos_len,
                                                int valid_len, int att_left, int att_right,
                                                GraphInputPool& pool) const {
    const int D = d_model_, H = n_heads_, dk = d_head_;
    const int P = pos_len;                 // window width = att_left+att_right+1
    const float scale = 1.0f / std::sqrt((float)dk);
    assert(att_left >= 0 && att_right >= 0);
    assert(P == att_left + att_right + 1);

    // Exact NeMo rel_pos_local_attn (RelPositionMultiHeadAttentionLongformer),
    // computed in O(T*window) via pad-and-shift instead of NeMo's skew/chunk
    // tricks. For query t and window column c in [0, P), the key is
    // (t - att_left + c). NeMo's local pos is ordered index0 = +att_left .. last
    // = -att_right, so column c uses pos row c directly (matrix_bd = q_v . p^T,
    // added 1:1 to the banded content scores).
    {
        const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
        const ModelLoader& ml = ml_;
        auto linear = [&](const char* wn, const char* bn, ggml_tensor* in) {
            ggml_tensor* W = clone_weight(ctx, ml, pre + wn);
            ggml_tensor* y = ggml_mul_mat(ctx, W, in);
            if (bn && ml.tensor(pre + bn)) y = ggml_add(ctx, y, clone_weight(ctx, ml, pre + bn));
            return y;
        };
        ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt);
        ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt);
        ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt);
        ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);
        auto to_heads = [&](ggml_tensor* t, int n) {
            t = ggml_reshape_3d(ctx, t, dk, H, n);
            return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H]
        };
        ggml_tensor* qh = to_heads(q, T), *kh = to_heads(k, T);
        ggml_tensor* vh = to_heads(v, T), *php = to_heads(p, P);
        ggml_tensor* bu = ggml_reshape_3d(ctx, clone_weight(ctx, ml, pre + "pos_bias_u"), dk, 1, H);
        ggml_tensor* bv = ggml_reshape_3d(ctx, clone_weight(ctx, ml, pre + "pos_bias_v"), dk, 1, H);
        ggml_tensor* qu = ggml_add(ctx, qh, bu);  // [dk, T, H]
        ggml_tensor* qv = ggml_add(ctx, qh, bv);  // [dk, T, H]

        // Pad K/V along time (ne1): att_left on the left, att_right on the right,
        // so view offset c yields key (t - att_left + c).
        ggml_tensor* kpad = ggml_pad_ext(ctx, kh, 0,0, att_left,att_right, 0,0, 0,0); // [dk, T+P-1, H]
        ggml_tensor* vpad = ggml_pad_ext(ctx, vh, 0,0, att_left,att_right, 0,0, 0,0);

        // Banded content scores ac[c, t, H] = q_u[t] . k[t-att_left+c]; stack on ne0=c.
        ggml_tensor* ac = nullptr;
        for (int c = 0; c < P; ++c) {
            ggml_tensor* kc = ggml_view_3d(ctx, kpad, dk, T, H, kpad->nb[1], kpad->nb[2],
                                           (size_t)c * kpad->nb[1]);
            ggml_tensor* acc = ggml_sum_rows(ctx, ggml_mul(ctx, qu, kc)); // [1, T, H]
            ac = ac ? ggml_concat(ctx, ac, acc, 0) : acc;
        }
        // Positional scores bd[c, t, H] = q_v[t] . p[c]  (direct, no rel-shift).
        ggml_tensor* bd = ggml_mul_mat(ctx, php, qv);   // [P, T, H]
        ggml_tensor* scores = ggml_add(ctx, ac, bd);    // [P, T, H]

        // Band mask [P, T]: 0 if key in [0, valid_len), else -inf (covers the
        // out-of-sequence window corners and pad frames). A padded query row
        // whose ENTIRE band lies beyond valid_len (pad tail > att_left, i.e. a
        // short item in a heavily padded batch) must NOT be all -inf: softmax
        // would yield NaN, the post-softmax query mask keeps NaN (NaN*0), and
        // the NaN keys poison later layers' valid frames. Leave such rows fully
        // unmasked instead — their finite garbage output is zeroed by the query
        // mask below.
        std::vector<float>& mh = pool.alloc_f32((size_t)P * T);
        for (int t = 0; t < T; ++t) {
            const bool dead_row = (t - att_left >= valid_len);
            for (int c = 0; c < P; ++c) {
                const int key = t - att_left + c;
                mh[(size_t)t * P + c] =
                    (dead_row || (key >= 0 && key < valid_len)) ? 0.0f : -INFINITY;
            }
        }
        int64_t mne[2] = {P, T};
        ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                                mh.data(), mh.size() * sizeof(float));
        ggml_tensor* prob = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // softmax over c

        // context[dk, t, H] = sum_c prob[c, t] * v[t-att_left+c].
        ggml_tensor* context = nullptr;
        for (int c = 0; c < P; ++c) {
            ggml_tensor* vc = ggml_view_3d(ctx, vpad, dk, T, H, vpad->nb[1], vpad->nb[2],
                                           (size_t)c * vpad->nb[1]);
            ggml_tensor* pc = ggml_view_3d(ctx, prob, 1, T, H, prob->nb[1], prob->nb[2],
                                           (size_t)c * prob->nb[0]); // [1, T, H]
            ggml_tensor* term = ggml_mul(ctx, vc, pc);               // broadcast pc over dk
            context = context ? ggml_add(ctx, context, term) : term;
        }
        // Merge heads [dk, T, H] -> [dk, H, T] -> [D, T].
        ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));
        merged = ggml_reshape_2d(ctx, merged, D, T);
        if (valid_len < T) { // zero padded query rows -> output = linear_out.bias
            std::vector<float>& qm = pool.alloc_f32(T);
            for (int t = 0; t < T; ++t) qm[t] = (t < valid_len) ? 1.0f : 0.0f;
            int64_t qne[2] = {1, T};
            ggml_tensor* qmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, qne,
                                     qm.data(), qm.size() * sizeof(float));
            merged = ggml_mul(ctx, merged, qmask);
        }
        ggml_tensor* Wo = clone_weight(ctx, ml, pre + "linear_out.weight");
        ggml_tensor* y = ggml_mul_mat(ctx, Wo, merged);
        if (ml.tensor(pre + "linear_out.bias"))
            y = ggml_add(ctx, y, clone_weight(ctx, ml, pre + "linear_out.bias"));
        return y; // [D, T]
    }
}

void RelPosAttention::forward_local(const std::vector<float>& x, int T,
                                    const std::vector<float>& pos_emb, int pos_len,
                                    int valid_len, int att_left, int att_right,
                                    std::vector<float>& out) const {
    const int D = d_model_;
    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);

    // Thin wrapper over build_graph_local: feed x/pos_emb as graph inputs and
    // compute the banded attention sub-graph on the persistent Backend. The
    // fused conformer encoder calls build_graph_local directly.
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {D, T};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  x.data(), (size_t)T * D * sizeof(float));
            int64_t pe_ne[2] = {D, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pos_emb.data(), (size_t)pos_len * D * sizeof(float));
            return build_graph_local(ctx, xt, T, pe, pos_len, valid_len,
                                     att_left, att_right, pool);
        }, out);
    assert(ok && "relpos local attention graph failed");
    (void)ok;
}

ggml_tensor* RelPosAttention::build_graph_local_chunked(
        ggml_context* ctx, ggml_tensor* xt, int T, ggml_tensor* pe, int pos_len,
        int valid_len, int att_left, int att_right, GraphInputPool& pool,
        int chunk) const {
    const int D = d_model_, H = n_heads_, dk = d_head_;
    const int P = pos_len;                       // window width = att_left+att_right+1
    const float scale = 1.0f / std::sqrt((float)dk);
    assert(att_left >= 0 && att_right >= 0);
    assert(P == att_left + att_right + 1);

    // Tile time into chunks of C frames (G chunks, Tp = G*C padded length). Each
    // chunk carries its own C+P-1 keys/values (the P-1 halo overlaps the next
    // chunk), so a query in chunk g only attends within g. Default C spans the
    // window so the halo is one chunk wide.
    int C = chunk > 0 ? chunk : (att_left + att_right);
    if (C < 1) C = 1;
    const int G  = (T + C - 1) / C;
    const int Tp = G * C;
    const int Lk = (C + P - 1) * G;              // dense length the chunk VIEW needs

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
    const ModelLoader& ml = ml_;
    auto linear = [&](const char* wn, const char* bn, ggml_tensor* in) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + wn);
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bn && ml.tensor(pre + bn)) y = ggml_add(ctx, y, clone_weight(ctx, ml, pre + bn));
        return y;
    };
    ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt);
    ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt);
    ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt);
    ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);
    auto to_heads = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_3d(ctx, t, dk, H, n);
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H]
    };
    ggml_tensor* qh = to_heads(q, T), *kh = to_heads(k, T);
    ggml_tensor* vh = to_heads(v, T), *php = to_heads(p, P);
    ggml_tensor* bu = ggml_reshape_3d(ctx, clone_weight(ctx, ml, pre + "pos_bias_u"), dk, 1, H);
    ggml_tensor* bv = ggml_reshape_3d(ctx, clone_weight(ctx, ml, pre + "pos_bias_v"), dk, 1, H);
    ggml_tensor* qu = ggml_add(ctx, qh, bu);  // [dk, T, H]
    ggml_tensor* qv = ggml_add(ctx, qh, bv);  // [dk, T, H]

    // ---- Content scores ac[c,t,H] via chunked matmul + diagonal skew-view ----
    // Pad queries to Tp and reshape into non-overlapping chunks [dk, C, G, H].
    ggml_tensor* qu_p = (Tp > T) ? ggml_pad_ext(ctx, qu, 0,0, 0,Tp-T, 0,0, 0,0) : qu;
    ggml_tensor* qu_c = ggml_reshape_4d(ctx, qu_p, dk, C, G, H);
    // Pad keys (left att_left, right att_right) then OVER-pad to Lk so the
    // overlapping chunk view's dense ne-product fits ggml's bounds check.
    ggml_tensor* kpad = ggml_pad_ext(ctx, kh, 0,0, att_left,att_right, 0,0, 0,0); // [dk,T+P-1,H]
    if (Lk > (int)kpad->ne[1]) kpad = ggml_pad_ext(ctx, kpad, 0,0, 0,Lk-(int)kpad->ne[1], 0,0, 0,0);
    // Overlapping key chunks [dk, C+P-1, G, H]: chunk g advances C along time.
    ggml_tensor* kchunk = ggml_view_4d(ctx, kpad, dk, C+P-1, G, H,
                              kpad->nb[1], (size_t)C*kpad->nb[1], kpad->nb[2], 0);
    kchunk = ggml_cont(ctx, kchunk);
    // Per-chunk q.k block [C+P-1, C, G, H]: sc[j,i,g] = k[gC+j] . qu[gC+i].
    ggml_tensor* sc = ggml_mul_mat(ctx, kchunk, qu_c);
    // Diagonal skew: ac_band[c,i,g] = sc[i+c, i, g] -> [P, C, G, H], nb1 walks (C+P).
    ggml_tensor* acb = ggml_view_4d(ctx, sc, P, C, G, H,
                           (size_t)(C+P)*sc->nb[0], sc->nb[2], sc->nb[3], 0);
    acb = ggml_cont(ctx, acb);
    acb = ggml_reshape_3d(ctx, acb, P, Tp, H);
    ggml_tensor* ac = (Tp > T) ? ggml_view_3d(ctx, acb, P, T, H, acb->nb[1], acb->nb[2], 0) : acb;

    // ---- Positional scores bd[c,t,H] = qv[t].p[c]  (same as build_graph_local) ----
    ggml_tensor* bd = ggml_mul_mat(ctx, php, qv);   // [P, T, H]
    ggml_tensor* scores = ggml_add(ctx, ac, bd);    // [P, T, H]

    // Band mask [P, T]: 0 if key in [0, valid_len), else -inf. As in
    // build_graph_local: a padded query row whose entire band lies beyond
    // valid_len must not be all -inf (softmax -> NaN, NaN*0 stays NaN and
    // poisons later layers through the key path); leave such rows fully
    // unmasked — the query mask below zeroes their output.
    std::vector<float>& mh = pool.alloc_f32((size_t)P * T);
    for (int t = 0; t < T; ++t) {
        const bool dead_row = (t - att_left >= valid_len);
        for (int c = 0; c < P; ++c) {
            const int key = t - att_left + c;
            mh[(size_t)t * P + c] =
                (dead_row || (key >= 0 && key < valid_len)) ? 0.0f : -INFINITY;
        }
    }
    int64_t mne[2] = {P, T};
    ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                            mh.data(), mh.size() * sizeof(float));
    ggml_tensor* prob = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // softmax over c

    // ---- Context[dk,t,H] = sum_c prob[c,t] v[t-att_left+c] via inverse-skew + matmul ----
    // Pad prob to Tp, chunk [P, C, G, H], inverse-skew to a banded [C+P-1, C, G, H].
    ggml_tensor* prob_p = (Tp > T) ? ggml_pad_ext(ctx, prob, 0,0, 0,Tp-T, 0,0, 0,0) : prob;
    ggml_tensor* prob_c = ggml_reshape_4d(ctx, prob_p, P, C, G, H);
    ggml_tensor* probpad = ggml_pad_ext(ctx, prob_c, 0,C, 0,0, 0,0, 0,0); // ne0 P->C+P
    // Pfull[j,i,g] = prob_c[j-i, i, g] (skew view; upper off-band already zero
    // from the pad, lower off-band masked below).
    ggml_tensor* pfull = ggml_view_4d(ctx, probpad, C+P-1, C, G, H,
                             (size_t)(C+P-1)*probpad->nb[0], probpad->nb[2], probpad->nb[3], 0);
    pfull = ggml_cont(ctx, pfull);
    std::vector<float>& b01 = pool.alloc_f32((size_t)(C+P-1) * C);
    for (int i = 0; i < C; ++i)
        for (int j = 0; j < C+P-1; ++j) {
            const int rel = j - i;
            b01[(size_t)i * (C+P-1) + j] = (rel >= 0 && rel < P) ? 1.0f : 0.0f;
        }
    int64_t bne[2] = {C+P-1, C};
    ggml_tensor* band01 = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, bne,
                              b01.data(), b01.size() * sizeof(float));
    pfull = ggml_mul(ctx, pfull, band01); // zero the lower off-band (broadcast over G,H)
    // Over-padded transposed V chunks [C+P-1, dk, G, H]: Vchunk[j,d,g]=v[gC+j].
    ggml_tensor* vpad = ggml_pad_ext(ctx, vh, 0,0, att_left,att_right, 0,0, 0,0); // [dk,T+P-1,H]
    if (Lk > (int)vpad->ne[1]) vpad = ggml_pad_ext(ctx, vpad, 0,0, 0,Lk-(int)vpad->ne[1], 0,0, 0,0);
    ggml_tensor* vpt = ggml_cont(ctx, ggml_permute(ctx, vpad, 1, 0, 2, 3)); // [Lk, dk, H]
    ggml_tensor* vchunk = ggml_view_4d(ctx, vpt, C+P-1, dk, G, H,
                              vpt->nb[1], (size_t)C*vpt->nb[0], vpt->nb[2], 0);
    vchunk = ggml_cont(ctx, vchunk);
    // context_g[d,i] = sum_j Vchunk[j,d] Pfull[j,i] -> [dk, C, G, H].
    ggml_tensor* cc = ggml_mul_mat(ctx, vchunk, pfull);
    cc = ggml_reshape_3d(ctx, cc, dk, Tp, H);
    ggml_tensor* context = (Tp > T) ? ggml_view_3d(ctx, cc, dk, T, H, cc->nb[1], cc->nb[2], 0) : cc;

    // Merge heads [dk,T,H] -> [dk,H,T] -> [D,T]; mask padded query rows; linear_out.
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));
    merged = ggml_reshape_2d(ctx, merged, D, T);
    if (valid_len < T) {
        std::vector<float>& qm = pool.alloc_f32(T);
        for (int t = 0; t < T; ++t) qm[t] = (t < valid_len) ? 1.0f : 0.0f;
        int64_t qne[2] = {1, T};
        ggml_tensor* qmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, qne,
                                 qm.data(), qm.size() * sizeof(float));
        merged = ggml_mul(ctx, merged, qmask);
    }
    ggml_tensor* Wo = clone_weight(ctx, ml, pre + "linear_out.weight");
    ggml_tensor* y = ggml_mul_mat(ctx, Wo, merged);
    if (ml.tensor(pre + "linear_out.bias"))
        y = ggml_add(ctx, y, clone_weight(ctx, ml, pre + "linear_out.bias"));
    return y; // [D, T]
}

ggml_tensor* RelPosAttention::build_graph_batched_local_chunked(
        ggml_context* ctx, ggml_tensor* xt, int T, int B, ggml_tensor* pe,
        int pos_len, const std::vector<int>& valid_len, int att_left, int att_right,
        GraphInputPool& pool, int chunk) const {
    const int D = d_model_;
    assert((int)valid_len.size() == B);
    // Run the O(1) chunk kernel per item (the 4D chunk graph can't also carry a
    // batch dim), then stack the per-item [D,T] outputs back into [D,T,B].
    ggml_tensor* out = nullptr;
    for (int b = 0; b < B; ++b) {
        ggml_tensor* xb = ggml_view_2d(ctx, xt, D, T, xt->nb[1], (size_t)b * xt->nb[2]);
        xb = ggml_cont(ctx, xb); // linear() mul_mat wants a dense [D,T] item
        ggml_tensor* yb = build_graph_local_chunked(ctx, xb, T, pe, pos_len,
                              valid_len[b], att_left, att_right, pool, chunk); // [D,T]
        yb = ggml_reshape_3d(ctx, yb, D, T, 1);
        out = out ? ggml_concat(ctx, out, yb, 2) : yb;
    }
    return out; // [D, T, B]
}

void RelPosAttention::forward_local_chunked(const std::vector<float>& x, int T,
                                            const std::vector<float>& pos_emb, int pos_len,
                                            int valid_len, int att_left, int att_right,
                                            std::vector<float>& out, int chunk) const {
    const int D = d_model_;
    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {D, T};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  x.data(), (size_t)T * D * sizeof(float));
            int64_t pe_ne[2] = {D, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pos_emb.data(), (size_t)pos_len * D * sizeof(float));
            return build_graph_local_chunked(ctx, xt, T, pe, pos_len, valid_len,
                                             att_left, att_right, pool, chunk);
        }, out);
    assert(ok && "relpos local chunked attention graph failed");
    (void)ok;
}

} // namespace pk
