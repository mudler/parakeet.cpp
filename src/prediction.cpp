#include "prediction.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace pk {

PredictionNet::PredictionNet(const ModelLoader& ml) : ml_(ml) {
    H_        = (int)ml.config().pred_hidden;
    vocab_p1_ = (int)ml.config().vocab_size + 1;
    n_layers_ = (int)ml.config().pred_rnn_layers;
    if (n_layers_ <= 0) n_layers_ = 1; // default to a single LSTM layer
    assert(H_ > 0 && "pred_hidden not set");
}

// Out-of-line so the header's unique_ptr<StepReplay> can stay incomplete there.
PredictionNet::~PredictionNet() = default;

// Replayable per-step LSTM graph. Inputs (registration order): x0 (layer-0
// input), then per layer [h_in, c_in]. Captures: per layer [c_out, h_out]
// (written into the stable internal cap_* buffers). Built once, replayed every
// step so ggml-cuda captures + replays the per-token prediction net.
struct PredictionNet::StepReplay {
    std::unique_ptr<ReplayGraph> rg;
    ggml_tensor* x0 = nullptr;               // view into in_buf (layer-0 input)
    std::vector<ggml_tensor*> h_in;          // views into in_buf (input 1 + 2*l)
    std::vector<ggml_tensor*> c_in;          // views into in_buf (input 2 + 2*l)
    // Coalesced per-step host input buffer [x0 | h0 | c0 | h1 | c1 ...], uploaded
    // in ONE set_input (one cudaStreamSynchronize) instead of 5.
    std::vector<float> in_buf;
    std::vector<std::vector<float>> cap_c;   // stable capture dsts
    std::vector<std::vector<float>> cap_h;
};

// Per-N replayable BATCHED LSTM graph for step_batch. Built once per batch size
// N, replayed every round. Inputs x0/h_in/c_in are re-fed each round (1+2L
// set_inputs); the 2L per-layer (h',c') captures land in STABLE internal buffers
// (the caller's out_state is reassigned every round, so it can't be the capture
// dst) and are copied out once after compute.
struct PredictionNet::StepReplayBatch {
    std::unique_ptr<ReplayGraph> rg;
    ggml_tensor* x0 = nullptr;                  // input #0 [H, N]
    std::vector<ggml_tensor*> h_in;             // input 1 + 2*l [H, N]
    std::vector<ggml_tensor*> c_in;             // input 2 + 2*l [H, N]
    std::vector<std::vector<float>> cap_h;      // stable per-layer capture dsts
    std::vector<std::vector<float>> cap_c;
};


// ---------------------------------------------------------------------------
// Stateful helpers.
// ---------------------------------------------------------------------------
PredState PredictionNet::zero_state() const {
    PredState s;
    s.h.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    s.c.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    return s;
}

// ---------------------------------------------------------------------------
// Advance the stacked LSTM by one token, as a single ggml graph that runs on
// whatever backend pk::Backend selected (CPU or device). The embedding table is
// fetched to the host once (ggml_backend_tensor_get, works for device tensors)
// so the SOS/lookup row can seed the layer-0 input; the LSTM weights are
// referenced as zero-copy loader leaves via clone_weight. Each layer's new
// (h', c') is captured for out_state; the top layer's h' is the output g.
//
// LSTM math (must match the prior C++ exactly):
//   z = W_ih·x + b_ih + W_hh·h_in + b_hh                          [4H]
//   i=sigmoid(z[0:H]); f=sigmoid(z[H:2H]); g=tanh(z[2H:3H]); o=sigmoid(z[3H:4H])
//   c' = f*c_in + i*g ;  h' = o*tanh(c')
// ---------------------------------------------------------------------------
void PredictionNet::step(int32_t token_id, bool is_sos,
                          const PredState& in,
                          std::vector<float>& g,
                          PredState& out_state) const {
    const int H = H_;
    const int L = n_layers_;

    // Lazily fetch the embedding table to the host (device-safe). Ensure the
    // loader's weights have a backend buffer first (idempotent) so the tensor
    // is readable via ggml_backend_tensor_get even when step()/forward() is
    // exercised before the encoder graph has realized the weights.
    if (embed_host_.empty()) {
        pk::ensure_weights_realized(ml_);
        ggml_tensor* emb = ml_.tensor("decoder.prediction.embed.weight");
        assert(emb && "missing decoder.prediction.embed.weight");
        embed_host_.resize((size_t)vocab_p1_ * H);
        ggml_backend_tensor_get(emb, embed_host_.data(), 0, ggml_nbytes(emb));
    }

    // Layer-0 input: zeros for SOS, else the embedding row for token_id.
    std::vector<float> x0((size_t)H, 0.0f);
    if (!is_sos) {
        assert(token_id >= 0 && token_id < vocab_p1_ && "embedding id out of range");
        std::memcpy(x0.data(), &embed_host_[(size_t)token_id * H],
                    (size_t)H * sizeof(float));
    }

    out_state.h.assign((size_t)L, std::vector<float>((size_t)H));
    out_state.c.assign((size_t)L, std::vector<float>((size_t)H));

    // Per-step LSTM. On a GPU backend the work is launch-overhead bound, so the
    // graph is built ONCE and REPLAYED every step: keeping the ggml context alive
    // makes cgraph->nodes[0] a stable pointer, so ggml-cuda captures + replays
    // the per-token prediction net instead of launching every op directly (the
    // megapar win). On CPU the per-step work is already cheap and replay's
    // set_input + capture-readback overhead would be a net regression, so CPU
    // keeps the original per-call run_graph path. Both are byte-identical.
    //
    // The replay capture dsts live INSIDE StepReplay (stable across steps); the
    // caller's out_state is .assign()'d every step, which can move its data
    // pointer, so we capture internally and copy out after compute.
    if (!global_backend().is_gpu()) {
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t ne1[1] = { H };
            ggml_tensor* layer_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                        x0.data(), (size_t)H * sizeof(float));
            ggml_tensor* top_h = nullptr;
            for (int l = 0; l < L; ++l) {
                const std::string s = "_l" + std::to_string(l);
                ggml_tensor* Wih = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
                ggml_tensor* Whh = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
                ggml_tensor* bih = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
                ggml_tensor* bhh = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
                ggml_tensor* h_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                        in.h[l].data(), (size_t)H * sizeof(float));
                ggml_tensor* c_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                        in.c[l].data(), (size_t)H * sizeof(float));
                ggml_tensor* z = ggml_add(ctx,
                    ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                    ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_in),     bhh));
                ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, 0)));
                ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)H * sizeof(float))));
                ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)2 * H * sizeof(float))));
                ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)3 * H * sizeof(float))));
                ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, c_in), ggml_mul(ctx, i, gg));
                ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
                pk::capture_graph_output(c_out, &out_state.c[l]);
                pk::capture_graph_output(h_out, &out_state.h[l]);
                layer_in = h_out;
                top_h    = h_out;
            }
            return top_h;
        }, g);
        assert(ok && "pred-net step graph failed");
        return;
    }

    // GPU replay path. Per-step inputs are COALESCED into ONE host buffer and
    // uploaded with a single set_input (one cudaStreamSynchronize) instead of
    // 5 separate set_input calls (x0 + 2*L h/c), since each set/get is a
    // host-stalling sync on the CUDA backend. The graph slices the single input
    // tensor into x0 / h_in[l] / c_in[l] views. Captures are still read per
    // layer (see below); coalescing them is a follow-up.
    if (!replay_) {
        replay_ = std::unique_ptr<StepReplay>(new StepReplay());
        replay_->h_in.assign(L, nullptr);
        replay_->c_in.assign(L, nullptr);
        replay_->cap_c.assign(L, std::vector<float>((size_t)H));
        replay_->cap_h.assign(L, std::vector<float>((size_t)H));
        StepReplay* r = replay_.get();  // captures must read this stable addr
        const int n_in_blocks = 1 + 2 * L;     // x0 + per-layer h,c
        r->in_buf.assign((size_t)n_in_blocks * H, 0.0f);
        r->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
            global_backend(),
            [&](ggml_context* ctx) -> ggml_tensor* {
                // ONE input tensor holding [x0 | h0 | c0 | h1 | c1 ...].
                int64_t in_ne[1] = { (int64_t)n_in_blocks * H };
                ggml_tensor* in_all = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, in_ne,
                            r->in_buf.data(), (size_t)n_in_blocks * H * sizeof(float));
                // Slice views (offset in bytes).
                r->x0 = ggml_view_1d(ctx, in_all, H, 0);
                ggml_tensor* layer_in = r->x0;
                ggml_tensor* top_h = nullptr;
                for (int l = 0; l < L; ++l) {
                    const std::string s = "_l" + std::to_string(l);
                    ggml_tensor* Wih = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
                    ggml_tensor* Whh = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
                    ggml_tensor* bih = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
                    ggml_tensor* bhh = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
                    size_t h_off = (size_t)(1 + 2 * l)     * H * sizeof(float);
                    size_t c_off = (size_t)(1 + 2 * l + 1) * H * sizeof(float);
                    r->h_in[l] = ggml_view_1d(ctx, in_all, H, h_off);
                    r->c_in[l] = ggml_view_1d(ctx, in_all, H, c_off);
                    ggml_tensor* z = ggml_add(ctx,
                        ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                        ggml_add(ctx, ggml_mul_mat(ctx, Whh, r->h_in[l]), bhh));
                    ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, 0)));
                    ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)H * sizeof(float))));
                    ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)2 * H * sizeof(float))));
                    ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)3 * H * sizeof(float))));
                    ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, r->c_in[l]),
                                                  ggml_mul(ctx, i, gg));
                    ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
                    pk::capture_graph_output(c_out, &r->cap_c[l]);
                    pk::capture_graph_output(h_out, &r->cap_h[l]);
                    layer_in = h_out;
                    top_h    = h_out;
                }
                return top_h;
            }));
        assert(r->rg->n_inputs() == 1 && "pred step graph must have 1 coalesced input");
    }

    // Pack this step's inputs into the single coalesced buffer (host memcpy, no
    // sync) and upload once. Layout matches the build: [x0 | h0 | c0 | h1 | c1].
    std::vector<float>& inb = replay_->in_buf;
    std::memcpy(inb.data(), x0.data(), (size_t)H * sizeof(float));
    for (int l = 0; l < L; ++l) {
        std::memcpy(inb.data() + (size_t)(1 + 2*l)     * H, in.h[l].data(), (size_t)H * sizeof(float));
        std::memcpy(inb.data() + (size_t)(1 + 2*l + 1) * H, in.c[l].data(), (size_t)H * sizeof(float));
    }
    replay_->rg->set_input(0, inb.data(), (size_t)(1 + 2 * L) * H * sizeof(float));
    bool ok = replay_->rg->compute_with_captures(g);
    assert(ok && "pred-net step replay failed");
    for (int l = 0; l < L; ++l) {
        std::memcpy(out_state.h[l].data(), replay_->cap_h[l].data(), (size_t)H * sizeof(float));
        std::memcpy(out_state.c[l].data(), replay_->cap_c[l].data(), (size_t)H * sizeof(float));
    }
}


// ---------------------------------------------------------------------------
// Batched single-step advance: the same LSTM math as step(), but with a batch
// axis N. Inputs and state are laid out [H, N] in ggml (item n is column n,
// offset n*H in the flat host buffer). Gate slices become [H, N] views into the
// [4H, N] z, using z->nb[1] as the column stride. N=1 reduces to step().
// ---------------------------------------------------------------------------
void PredictionNet::step_batch(const std::vector<int32_t>& token_ids,
                               const std::vector<uint8_t>& is_sos,
                               const BatchedPredState& in,
                               std::vector<float>& g,
                               BatchedPredState& out_state) const {
    const int H = H_;
    const int L = n_layers_;
    const int N = (int)token_ids.size();
    assert(N > 0 && (int)is_sos.size() == N && "batch size mismatch");

    // Lazily fetch the embedding table to the host (device-safe), exactly as
    // step() does.
    if (embed_host_.empty()) {
        pk::ensure_weights_realized(ml_);
        ggml_tensor* emb = ml_.tensor("decoder.prediction.embed.weight");
        assert(emb && "missing decoder.prediction.embed.weight");
        embed_host_.resize((size_t)vocab_p1_ * H);
        ggml_backend_tensor_get(emb, embed_host_.data(), 0, ggml_nbytes(emb));
    }

    // Layer-0 input [H*N]: zeros for SOS items, else the embedding row.
    std::vector<float> x0((size_t)H * N, 0.0f);
    for (int n = 0; n < N; ++n) {
        if (!is_sos[n]) {
            assert(token_ids[n] >= 0 && token_ids[n] < vocab_p1_ && "embedding id out of range");
            std::memcpy(&x0[(size_t)n * H], &embed_host_[(size_t)token_ids[n] * H],
                        (size_t)H * sizeof(float));
        }
    }

    out_state.h.assign((size_t)L, std::vector<float>((size_t)H * N));
    out_state.c.assign((size_t)L, std::vector<float>((size_t)H * N));

    // CPU path: per-call run_graph (unchanged).
    if (!global_backend().is_gpu()) {
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t ne2[2] = { H, N };
            ggml_tensor* layer_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne2,
                                        x0.data(), (size_t)H * N * sizeof(float));
            ggml_tensor* top_h = nullptr;
            for (int l = 0; l < L; ++l) {
                const std::string s = "_l" + std::to_string(l);
                ggml_tensor* Wih = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
                ggml_tensor* Whh = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
                ggml_tensor* bih = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
                ggml_tensor* bhh = pk::clone_weight(ctx, ml_,
                    ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
                ggml_tensor* h_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne2,
                                        in.h[l].data(), (size_t)H * N * sizeof(float));
                ggml_tensor* c_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne2,
                                        in.c[l].data(), (size_t)H * N * sizeof(float));
                ggml_tensor* z = ggml_add(ctx,
                    ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                    ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_in),     bhh));
                ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], 0)));
                ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)H * sizeof(float))));
                ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)2 * H * sizeof(float))));
                ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)3 * H * sizeof(float))));
                ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, c_in), ggml_mul(ctx, i, gg));
                ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
                pk::capture_graph_output(c_out, &out_state.c[l]);
                pk::capture_graph_output(h_out, &out_state.h[l]);
                layer_in = h_out;
                top_h    = h_out;
            }
            return top_h;
        }, g);
        assert(ok && "pred-net step_batch graph failed");
        return;
    }

    // GPU replay path: one captured graph per N. Captures land in STABLE internal
    // buffers (out_state is reassigned each round) and are copied out after compute.
    auto it = replay_batch_.find(N);
    if (it == replay_batch_.end()) {
        auto rb = std::unique_ptr<StepReplayBatch>(new StepReplayBatch());
        rb->h_in.assign(L, nullptr);
        rb->c_in.assign(L, nullptr);
        rb->cap_h.assign(L, std::vector<float>((size_t)H * N));
        rb->cap_c.assign(L, std::vector<float>((size_t)H * N));
        StepReplayBatch* r = rb.get();
        int64_t ne2[2] = { H, N };
        r->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
            global_backend(),
            [&](ggml_context* ctx) -> ggml_tensor* {
                r->x0 = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne2,
                            x0.data(), (size_t)H * N * sizeof(float));
                ggml_tensor* layer_in = r->x0;
                ggml_tensor* top_h = nullptr;
                for (int l = 0; l < L; ++l) {
                    const std::string s = "_l" + std::to_string(l);
                    ggml_tensor* Wih = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
                    ggml_tensor* Whh = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
                    ggml_tensor* bih = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
                    ggml_tensor* bhh = pk::clone_weight(ctx, ml_,
                        ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
                    r->h_in[l] = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne2,
                                        in.h[l].data(), (size_t)H * N * sizeof(float));
                    r->c_in[l] = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne2,
                                        in.c[l].data(), (size_t)H * N * sizeof(float));
                    ggml_tensor* z = ggml_add(ctx,
                        ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                        ggml_add(ctx, ggml_mul_mat(ctx, Whh, r->h_in[l]), bhh));
                    ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], 0)));
                    ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)H * sizeof(float))));
                    ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)2 * H * sizeof(float))));
                    ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)3 * H * sizeof(float))));
                    ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, r->c_in[l]),
                                                  ggml_mul(ctx, i, gg));
                    ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
                    pk::capture_graph_output(c_out, &r->cap_c[l]);
                    pk::capture_graph_output(h_out, &r->cap_h[l]);
                    layer_in = h_out;
                    top_h    = h_out;
                }
                return top_h;
            }));
        it = replay_batch_.emplace(N, std::move(rb)).first;
    }
    StepReplayBatch* r = it->second.get();
    r->rg->set_input(0, x0.data(), (size_t)H * N * sizeof(float));
    for (int l = 0; l < L; ++l) {
        r->rg->set_input(1 + 2 * l, in.h[l].data(), (size_t)H * N * sizeof(float));
        r->rg->set_input(2 + 2 * l, in.c[l].data(), (size_t)H * N * sizeof(float));
    }
    bool ok = r->rg->compute_with_captures(g);
    assert(ok && "pred-net step_batch replay failed");
    for (int l = 0; l < L; ++l) {
        std::memcpy(out_state.h[l].data(), r->cap_h[l].data(), (size_t)H * N * sizeof(float));
        std::memcpy(out_state.c[l].data(), r->cap_c[l].data(), (size_t)H * N * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Full-sequence forward pass (unchanged API; now driven by step() so there is a
// single LSTM implementation). Carries (h, c) state across timesteps; the
// output at step t is the top layer's h'. add_sos prepends a zero SOS step.
// ---------------------------------------------------------------------------
void PredictionNet::forward(const std::vector<int32_t>& ids, bool add_sos,
                            std::vector<float>& out, int& U_out, int& hidden) const {
    const int H   = H_;
    const int seq = add_sos ? (int)ids.size() + 1 : (int)ids.size();
    U_out  = seq;
    hidden = H;
    out.assign((size_t)seq * H, 0.0f);

    PredState st = zero_state();
    std::vector<float> g;
    PredState nxt;
    for (int t = 0; t < seq; ++t) {
        const bool is_sos = add_sos && t == 0;
        const int32_t tok = is_sos ? -1 : ids[add_sos ? t - 1 : t];
        step(tok, is_sos, st, g, nxt);
        std::memcpy(&out[(size_t)t * H], g.data(), (size_t)H * sizeof(float));
        st = nxt;
    }
}

} // namespace pk
