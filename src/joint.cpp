#include "joint.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "exact_matvec.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <vector>

namespace pk {

Joint::Joint(const ModelLoader& ml) : ml_(ml) {
    // Read joint_hidden from the enc weight shape: ne[1] (ggml) = joint_hidden.
    ggml_tensor* ew = ml.tensor("joint.enc.weight");
    assert(ew && "missing joint.enc.weight");
    joint_hidden_ = (int)ew->ne[1];
    enc_hidden_   = (int)ew->ne[0];

    ggml_tensor* pw = ml.tensor("joint.pred.weight");
    assert(pw && "missing joint.pred.weight");
    pred_hidden_ = (int)pw->ne[0];
    assert((int)pw->ne[1] == joint_hidden_ && "pred/enc joint_hidden mismatch");

    vocab_size_    = (int)ml.config().vocab_size;
    num_durations_ = (int)ml.config().tdt_durations.size();
    V_plus_        = vocab_size_ + 1 + num_durations_;

    // Sanity-check the output projection shape (joint_net.2 is kept f32 by the
    // converter; enc/pred projections are quantization-allowlisted and may be
    // f16/q8_0 — ggml_mul_mat dequantizes those on the fly in the graphs below).
    ggml_tensor* wout = ml.tensor("joint.joint_net.2.weight");
    assert(wout && "missing joint.joint_net.2.weight");
    assert((int)wout->ne[0] == joint_hidden_ && (int)wout->ne[1] == V_plus_ &&
           "joint_net.2 weight shape mismatch");
    (void)wout;
}

void Joint::precompute_enc_proj(const std::vector<float>& enc, int T, int enc_hidden,
                                std::vector<float>& enc_proj) const {
    assert((int)enc.size() == T * enc_hidden);
    assert(enc_hidden == enc_hidden_ && "enc_hidden mismatch");
    const int H = joint_hidden_;

    // enc_proj[t][h] = enc.weight·enc[t] + enc.bias, over ALL T frames in ONE
    // multithreaded ggml matmul on the persistent backend (vs the old per-step
    // 1xH matmul that dispatched a graph for every emitted token). Done once per
    // utterance, reused by step_logits for every (t,u). ggml_mul_mat dequantizes
    // a quantized enc.weight on the fly, so quantized models work here too.
    bool ok = pk::run_graph(/*mem ignored*/0, /*n_threads ignored*/0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // Input: row-major [T, E] -> ggml ne[0]=E (fastest), ne[1]=T.
            int64_t x_ne[2] = { enc_hidden, T };
            ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, x_ne,
                                 enc.data(), (size_t)T * enc_hidden * sizeof(float));
            // Weight (zero-copy loader leaf): ne[0]=E, ne[1]=H -> mul_mat -> [H, T].
            ggml_tensor* W = pk::clone_weight(ctx, ml_, "joint.enc.weight");
            ggml_tensor* y = ggml_mul_mat(ctx, W, x);
            ggml_tensor* b = pk::clone_weight(ctx, ml_, "joint.enc.bias");
            y = ggml_add(ctx, y, b);
            return y;   // [H, T] ggml memory -> enc_proj[t*H + h]
        }, enc_proj);
    assert(ok && "enc_proj graph failed");
    (void)H;
}

ggml_tensor* Joint::build_step(ggml_context* ctx, const float* enc_proj_t, ggml_tensor* g) const {
    const int H = joint_hidden_;
    // enc_proj row for frame t: [H].
    int64_t ep_ne[1] = { H };
    ggml_tensor* ep = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ep_ne,
                          enc_proj_t, (size_t)H * sizeof(float));
    // pred_proj = pred.weight·g + pred.bias  (P->H). Weight ne=[P,H].
    ggml_tensor* Wp = pk::clone_weight(ctx, ml_, "joint.pred.weight");
    ggml_tensor* pp = ggml_mul_mat(ctx, Wp, g);             // [H]
    ggml_tensor* bp = pk::clone_weight(ctx, ml_, "joint.pred.bias");
    pp = ggml_add(ctx, pp, bp);
    // f = ReLU(enc_proj + pred_proj)
    ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, ep, pp)); // [H]
    // logits = joint_net.2.weight·f + joint_net.2.bias (H->V). Weight ne=[H,V].
    ggml_tensor* Wo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.weight");
    ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);             // [V]
    ggml_tensor* bo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.bias");
    return ggml_add(ctx, y, bo);                            // [V_plus]
}

void Joint::step_logits(const float* enc_proj_t,
                        const float* g, int pred_hidden,
                        std::vector<float>& logits) const {
    assert(pred_hidden == pred_hidden_ && "pred_hidden mismatch");

    // Per-step joint on the PERSISTENT backend (one reused graph; no per-call
    // ggml_init/gallocr churn — Backend::compute reuses the backend + gallocr).
    // The two matmuls (pred.weight: P->H and joint_net.2.weight: H->V) are the
    // hot cost and are memory-bandwidth bound (the H->V weight is ~2.6 MB read
    // every step); ggml's matmul parallelizes the output dimension across the
    // worker threads, hitting much higher aggregate memory bandwidth than a
    // single-threaded C++ matvec — measured ~15x faster per step (26us vs 389us)
    // on the 110m. The enc_proj input is the precomputed projection row for t.
    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // pred-net output g: [P].
            int64_t g_ne[1] = { pred_hidden_ };
            ggml_tensor* gv = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, g_ne,
                                  g, (size_t)pred_hidden_ * sizeof(float));
            return build_step(ctx, enc_proj_t, gv);
        }, logits);
    assert(ok && "step_logits graph failed");
}

void Joint::step_logits_batch(const float* enc_proj_gathered,
                              const float* g, int pred_hidden, int n,
                              std::vector<float>& logits) const {
    assert(pred_hidden == pred_hidden_ && "pred_hidden mismatch");
    const int H = joint_hidden_;

    // Batched per-step joint over N items on the PERSISTENT backend. Mirrors
    // step_logits with a batch axis (ggml ne1 = N): each of the two matmuls is
    // applied across all N columns at once, and the biases broadcast over N.
    // N=1 reduces exactly to step_logits. On the CPU backend both matmuls use
    // mul_mat_cols_exact, so every column is bitwise equal to step_logits
    // (test_exact_batch). The gathered enc_proj input holds one
    // joint_hidden row per item (item k at offset k*H), and g holds one
    // pred_hidden vector per item (item k at offset k*pred_hidden).
    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // Gathered enc_proj rows: [H, N].
            int64_t ep_ne[2] = { H, n };
            ggml_tensor* ep = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ep_ne,
                                  enc_proj_gathered, (size_t)H * n * sizeof(float));
            // Batched pred-net output g: [P, N].
            int64_t g_ne[2] = { pred_hidden_, n };
            ggml_tensor* gv = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, g_ne,
                                  g, (size_t)pred_hidden_ * n * sizeof(float));
            // pred_proj = pred.weight·g + pred.bias  (P->H). Weight ne=[P,H].
            ggml_tensor* Wp = pk::clone_weight(ctx, ml_, "joint.pred.weight");
            ggml_tensor* pp = pk::mul_mat_cols_exact(ctx, Wp, gv);            // [H, N]
            ggml_tensor* bp = pk::clone_weight(ctx, ml_, "joint.pred.bias");
            pp = ggml_add(ctx, pp, bp);                             // bp [H] broadcasts over N
            // f = ReLU(enc_proj + pred_proj)
            ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, ep, pp)); // [H, N]
            // logits = joint_net.2.weight·f + joint_net.2.bias (H->V). Weight ne=[H,V].
            ggml_tensor* Wo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.weight");
            ggml_tensor* y  = pk::mul_mat_cols_exact(ctx, Wo, f);             // [V, N]
            ggml_tensor* bo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.bias");
            y = ggml_add(ctx, y, bo);                               // bo [V] broadcasts over N
            return y;                                               // [V_plus, N]
        }, logits);
    assert(ok && "step_logits_batch graph failed");
}

void Joint::forward(const std::vector<float>& enc,  int T, int enc_hidden,
                    const std::vector<float>& pred, int U, int pred_hidden,
                    std::vector<float>& logits, int& V_plus_out) const {
    assert((int)enc.size()  == T * enc_hidden);
    assert((int)pred.size() == U * pred_hidden);

    const int V = V_plus_;
    V_plus_out = V;

    // Compute the full [T, U, V_plus] grid by reusing the SAME kernels the greedy
    // fast path uses: precompute enc_proj over all T frames once, then emit the
    // per-(t,u) logits via step_logits. This keeps the batch path (test_joint /
    // test_transducer_core parity vs NeMo) numerically identical to the per-step
    // decode path. (Used only by the parity tests; the real decode loops call
    // precompute_enc_proj + step_logits directly.)
    std::vector<float> enc_proj;
    precompute_enc_proj(enc, T, enc_hidden, enc_proj);

    const int H = joint_hidden_;
    logits.resize((size_t)T * U * V);
    std::vector<float> step;
    for (int t = 0; t < T; ++t) {
        const float* ep_t = enc_proj.data() + (size_t)t * H;
        for (int u = 0; u < U; ++u) {
            step_logits(ep_t, pred.data() + (size_t)u * pred_hidden, pred_hidden, step);
            std::memcpy(logits.data() + ((size_t)t * U + u) * V,
                        step.data(), (size_t)V * sizeof(float));
        }
    }
}

} // namespace pk
