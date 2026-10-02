#include "joint.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <vector>

namespace pk {

// Replayable per-step joint graph. Built once (first step_logits), replayed on
// every subsequent step. Holds the persistent ReplayGraph + the two input
// tensor handles so step_logits can feed fresh enc_proj/g rows each step
// without rebuilding the graph. Keeping the underlying ggml context alive is
// what lets ggml-cuda capture+replay the per-token joint (megapar's idea).
struct Joint::StepReplay {
    std::unique_ptr<ReplayGraph> rg;
    ggml_tensor* ep = nullptr;  // view into in_buf: enc_proj row for frame t [H]
    ggml_tensor* g  = nullptr;  // view into in_buf: pred-net output g [P]
    // Coalesced per-step host input [enc_proj_t | g], uploaded in ONE set_input.
    std::vector<float> in_buf;
};

// Per-N replayable BATCHED joint graph. Built once per batch size N, replayed
// every round of that batch (enc_proj [H,N] + g [P,N] re-fed each round).
struct Joint::StepReplayBatch {
    std::unique_ptr<ReplayGraph> rg;
    ggml_tensor* ep = nullptr;   // input #0: enc_proj rows [H, N]
    ggml_tensor* g  = nullptr;   // input #1: pred output g   [P, N]
    int H = 0, P = 0, N = 0;
};

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

// Out-of-line so the header's unique_ptr<StepReplay> can stay incomplete there.
Joint::~Joint() = default;

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

void Joint::step_logits(const float* enc_proj_t,
                        const float* g, int pred_hidden,
                        std::vector<float>& logits) const {
    assert(pred_hidden == pred_hidden_ && "pred_hidden mismatch");
    const int H = joint_hidden_;

    // Per-step joint. On a GPU backend the work is launch-overhead bound, so we
    // build the graph ONCE and REPLAY it every step: keeping the ggml context
    // alive makes cgraph->nodes[0] a stable pointer, so ggml-cuda captures +
    // replays the per-token joint instead of launching every op directly (the
    // megapar win; ~290ms -> tens of ms on the 0.6b GPU decode). On CPU the
    // per-step work is already cheap (multithreaded matmul; the launch overhead
    // is negligible), and replay's set_input + readback overhead would be a net
    // regression, so CPU keeps the original per-call run_graph path. The GPU
    // compute is byte-identical to the CPU path: same ops, order, weights.
    if (!global_backend().is_gpu()) {
        bool ok = pk::run_graph(0, 0,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t ep_ne[1] = { H };
                ggml_tensor* ep = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ep_ne,
                                      enc_proj_t, (size_t)H * sizeof(float));
                int64_t g_ne[1] = { pred_hidden_ };
                ggml_tensor* gv = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, g_ne,
                                      g, (size_t)pred_hidden_ * sizeof(float));
                ggml_tensor* Wp = pk::clone_weight(ctx, ml_, "joint.pred.weight");
                ggml_tensor* pp = ggml_mul_mat(ctx, Wp, gv);            // [H]
                ggml_tensor* bp = pk::clone_weight(ctx, ml_, "joint.pred.bias");
                pp = ggml_add(ctx, pp, bp);
                ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, ep, pp)); // [H]
                ggml_tensor* Wo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.weight");
                ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);             // [V]
                ggml_tensor* bo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.bias");
                y = ggml_add(ctx, y, bo);
                return y;                                               // [V_plus]
            }, logits);
        assert(ok && "step_logits graph failed");
        return;
    }

    // GPU replay path. Per-step inputs are COALESCED into ONE host buffer
    // [enc_proj_t | g] and uploaded with a single set_input (one
    // cudaStreamSynchronize) instead of two separate set_input calls.
    if (!replay_) {
        replay_ = std::unique_ptr<StepReplay>(new StepReplay());
        StepReplay* r = replay_.get();
        const int P = pred_hidden_;
        r->in_buf.assign((size_t)(H + P), 0.0f);
        r->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
            global_backend(),
            [&](ggml_context* ctx) -> ggml_tensor* {
                // ONE input tensor holding [enc_proj_t (H) | g (P)].
                int64_t in_ne[1] = { (int64_t)(H + P) };
                ggml_tensor* in_all = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, in_ne,
                                      r->in_buf.data(), (size_t)(H + P) * sizeof(float));
                r->ep = ggml_view_1d(ctx, in_all, H, 0);
                r->g  = ggml_view_1d(ctx, in_all, P, (size_t)H * sizeof(float));
                ggml_tensor* Wp = pk::clone_weight(ctx, ml_, "joint.pred.weight");
                ggml_tensor* pp = ggml_mul_mat(ctx, Wp, r->g);     // [H]
                ggml_tensor* bp = pk::clone_weight(ctx, ml_, "joint.pred.bias");
                pp = ggml_add(ctx, pp, bp);
                ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, r->ep, pp)); // [H]
                ggml_tensor* Wo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.weight");
                ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);             // [V]
                ggml_tensor* bo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.bias");
                y = ggml_add(ctx, y, bo);
                return y;                                             // [V_plus]
            }));
        assert(r->rg->n_inputs() == 1 && "joint step graph must have 1 coalesced input");
    }

    // Pack [enc_proj_t | g] into the coalesced buffer (host memcpy) and upload once.
    const int P = pred_hidden_;
    std::memcpy(replay_->in_buf.data(), enc_proj_t, (size_t)H * sizeof(float));
    std::memcpy(replay_->in_buf.data() + H, g, (size_t)P * sizeof(float));
    replay_->rg->set_input(0, replay_->in_buf.data(), (size_t)(H + P) * sizeof(float));
    bool ok = replay_->rg->compute(logits);
    assert(ok && "step_logits replay failed");
}

void Joint::step_logits_batch(const float* enc_proj_gathered,
                              const float* g, int pred_hidden, int n,
                              std::vector<float>& logits) const {
    assert(pred_hidden == pred_hidden_ && "pred_hidden mismatch");
    const int H = joint_hidden_;

    // CPU path: per-call run_graph (unchanged). On GPU the batched joint is, like
    // the single-step joint, launch-overhead bound; capture one replay graph per
    // batch size N and replay it every round, coalescing the two per-round inputs
    // (enc_proj [H,N] + g [P,N]) into ONE host buffer uploaded with a single
    // set_input. Byte-identical to the run_graph path.
    if (!global_backend().is_gpu()) {
        bool ok = pk::run_graph(0, 0,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t ep_ne[2] = { H, n };
                ggml_tensor* ep = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ep_ne,
                                      enc_proj_gathered, (size_t)H * n * sizeof(float));
                int64_t g_ne[2] = { pred_hidden_, n };
                ggml_tensor* gv = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, g_ne,
                                      g, (size_t)pred_hidden_ * n * sizeof(float));
                ggml_tensor* Wp = pk::clone_weight(ctx, ml_, "joint.pred.weight");
                ggml_tensor* pp = ggml_mul_mat(ctx, Wp, gv);            // [H, N]
                ggml_tensor* bp = pk::clone_weight(ctx, ml_, "joint.pred.bias");
                pp = ggml_add(ctx, pp, bp);
                ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, ep, pp)); // [H, N]
                ggml_tensor* Wo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.weight");
                ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);             // [V, N]
                ggml_tensor* bo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.bias");
                y = ggml_add(ctx, y, bo);
                return y;                                               // [V_plus, N]
            }, logits);
        assert(ok && "step_logits_batch graph failed");
        return;
    }

    // GPU replay path: one captured graph per N. (Inputs stay as two separate
    // tensors -- enc_proj [H,N] and g [P,N] are each contiguous per-item, and
    // packing them into one (H+P)xN buffer would need a costly host repack. Two
    // set_inputs is still a large win over the per-call run_graph path.)
    auto it = replay_batch_.find(n);
    if (it == replay_batch_.end()) {
        auto rb = std::unique_ptr<StepReplayBatch>(new StepReplayBatch());
        rb->H = H; rb->P = pred_hidden_; rb->N = n;
        StepReplayBatch* r = rb.get();
        r->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
            global_backend(),
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t ep_ne[2] = { H, n };
                r->ep = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ep_ne,
                                      enc_proj_gathered, (size_t)H * n * sizeof(float));
                int64_t g_ne[2] = { pred_hidden_, n };
                r->g  = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, g_ne,
                                      g, (size_t)pred_hidden_ * n * sizeof(float));
                ggml_tensor* Wp = pk::clone_weight(ctx, ml_, "joint.pred.weight");
                ggml_tensor* pp = ggml_mul_mat(ctx, Wp, r->g);            // [H, N]
                ggml_tensor* bp = pk::clone_weight(ctx, ml_, "joint.pred.bias");
                pp = ggml_add(ctx, pp, bp);
                ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, r->ep, pp)); // [H, N]
                ggml_tensor* Wo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.weight");
                ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);             // [V, N]
                ggml_tensor* bo = pk::clone_weight(ctx, ml_, "joint.joint_net.2.bias");
                y = ggml_add(ctx, y, bo);
                return y;                                               // [V_plus, N]
            }));
        it = replay_batch_.emplace(n, std::move(rb)).first;
    }
    StepReplayBatch* r = it->second.get();
    r->rg->set_input(0, enc_proj_gathered, (size_t)H * n * sizeof(float));
    r->rg->set_input(1, g, (size_t)pred_hidden_ * n * sizeof(float));
    bool ok = r->rg->compute(logits);
    assert(ok && "step_logits_batch replay failed");
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
