#include "prediction.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "exact_matvec.hpp"
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

// ---------------------------------------------------------------------------
// Stateful helpers.
// ---------------------------------------------------------------------------
PredState PredictionNet::zero_state() const {
    PredState s;
    s.h.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    s.c.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    return s;
}

// Fetch the embedding table to the host once (device-safe), thread-safe so a
// PredictionNet can be shared by concurrent decodes. Ensure the loader's
// weights have a backend buffer first (idempotent) so the tensor is readable
// via ggml_backend_tensor_get even when step()/forward() runs before the
// encoder graph has realized the weights. After the call_once the table is
// read-only.
const float* PredictionNet::embed_table() const {
    std::call_once(embed_once_, [this]() {
        pk::ensure_weights_realized(ml_);
        ggml_tensor* emb = ml_.tensor("decoder.prediction.embed.weight");
        assert(emb && "missing decoder.prediction.embed.weight");
        embed_host_.resize((size_t)vocab_p1_ * H_);
        ggml_backend_tensor_get(emb, embed_host_.data(), 0, ggml_nbytes(emb));
        embed_fetched_.store(true, std::memory_order_release);
    });
    return embed_host_.data();
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
ggml_tensor* PredictionNet::build_step(ggml_context* ctx, int32_t token_id, bool is_sos,
                                       const PredState& in, std::vector<float>& x0,
                                       PredState& out_state) const {
    const int H = H_;
    const int L = n_layers_;

    // Fetched even for SOS: a caller's warm-up on silence emits no token, and must still pay this 21 MB read.
    const float* embed_host = embed_table();

    // Layer-0 input: zeros for SOS, else the embedding row for token_id.
    x0.assign((size_t)H, 0.0f);
    if (!is_sos) {
        assert(token_id >= 0 && token_id < vocab_p1_ && "embedding id out of range");
        std::memcpy(x0.data(), embed_host + (size_t)token_id * H,
                    (size_t)H * sizeof(float));
    }

    out_state.h.assign((size_t)L, std::vector<float>((size_t)H));
    out_state.c.assign((size_t)L, std::vector<float>((size_t)H));

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
        // z = W_ih·x + b_ih + W_hh·h_in + b_hh                       [4H]
        ggml_tensor* z = ggml_add(ctx,
            ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
            ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_in),     bhh));
        // Gate slices (i, f, g, o), each [H]. A 1-D view is already contiguous, so the gates need no copies.
        ggml_tensor* i  = ggml_sigmoid(ctx, ggml_view_1d(ctx, z, H, 0));
        ggml_tensor* f  = ggml_sigmoid(ctx, ggml_view_1d(ctx, z, H, (size_t)H * sizeof(float)));
        ggml_tensor* gg = ggml_tanh   (ctx, ggml_view_1d(ctx, z, H, (size_t)2 * H * sizeof(float)));
        ggml_tensor* o  = ggml_sigmoid(ctx, ggml_view_1d(ctx, z, H, (size_t)3 * H * sizeof(float)));
        // c' = f*c_in + i*g ;  h' = o*tanh(c')
        ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, c_in), ggml_mul(ctx, i, gg));
        ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
        pk::capture_graph_output(c_out, &out_state.c[l]);
        pk::capture_graph_output(h_out, &out_state.h[l]);
        layer_in = h_out;
        top_h    = h_out;
    }
    return top_h;
}

void PredictionNet::step(int32_t token_id, bool is_sos,
                          const PredState& in,
                          std::vector<float>& g,
                          PredState& out_state) const {
    std::vector<float> x0;
    bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        return build_step(ctx, token_id, is_sos, in, x0, out_state);
    }, g);
    assert(ok && "pred-net step graph failed");
}

// ---------------------------------------------------------------------------
// Batched single-step advance: the same LSTM math as step(), but with a batch
// axis N. Inputs and state are laid out [H, N] in ggml (item n is column n,
// offset n*H in the flat host buffer). Gate slices become [H, N] views into the
// [4H, N] z, using z->nb[1] as the column stride. N=1 reduces to step().
// On the CPU backend the two matmuls use mul_mat_cols_exact, so every column is
// bitwise equal to step() (test_exact_batch).
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

    const float* embed_host = embed_table();

    // Layer-0 input [H*N]: zeros for SOS items, else the embedding row.
    std::vector<float> x0((size_t)H * N, 0.0f);
    for (int n = 0; n < N; ++n) {
        if (!is_sos[n]) {
            assert(token_ids[n] >= 0 && token_ids[n] < vocab_p1_ && "embedding id out of range");
            std::memcpy(&x0[(size_t)n * H], embed_host + (size_t)token_ids[n] * H,
                        (size_t)H * sizeof(float));
        }
    }

    out_state.h.assign((size_t)L, std::vector<float>((size_t)H * N));
    out_state.c.assign((size_t)L, std::vector<float>((size_t)H * N));

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
            // z = W_ih·x + b_ih + W_hh·h_in + b_hh                  [4H, N]
            // (bias [4H] broadcasts over the N columns).
            ggml_tensor* z = ggml_add(ctx,
                ggml_add(ctx, pk::mul_mat_cols_exact(ctx, Wih, layer_in), bih),
                ggml_add(ctx, pk::mul_mat_cols_exact(ctx, Whh, h_in),     bhh));
            // Gate slices (i, f, g, o), each [H, N]. The view keeps z's FULL
            // column stride (z->nb[1] = 4H elems), reading only H contiguous
            // elements per column, so consecutive columns skip the other three
            // gate blocks. (Do NOT change the stride to H*sizeof(float).)
            ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], 0)));
            ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)H * sizeof(float))));
            ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)2 * H * sizeof(float))));
            ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z, H, N, z->nb[1], (size_t)3 * H * sizeof(float))));
            // c' = f*c_in + i*g ;  h' = o*tanh(c')
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
