#pragma once
#include "model_loader.hpp"
#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace pk {

// Carries the LSTM hidden and cell state for stateful single-step decoding.
// One (h, c) pair PER stacked LSTM layer (PyTorch nn.LSTM with num_layers>1).
// h[l] / c[l] are each [hidden]. For a single-layer LSTM the outer vectors hold
// exactly one element.
struct PredState {
    std::vector<std::vector<float>> h; // h[layer] = [hidden]
    std::vector<std::vector<float>> c; // c[layer] = [hidden]
};

// Batched LSTM state: one (h,c) per layer, each holding N items' columns laid
// out [H*N] (item n at offset n*H). Generalizes PredState to a batch.
struct BatchedPredState {
    std::vector<std::vector<float>> h; // h[layer] size H*N
    std::vector<std::vector<float>> c; // c[layer] size H*N
};

// RNN-Transducer prediction network — NeMo RNNTDecoder prediction net.
//
// Architecture:
//   Embedding(vocab+1, pred_hidden, padding_idx=blank) lookup
//   Stacked LSTM with `pred_rnn_layers` layers (PyTorch convention). The 110m
//   anchor has 1 layer; parakeet-tdt-0.6b-v2/-v3 have 2. The number of layers
//   is read from the GGUF (parakeet.decoder.pred_rnn_layers); each layer l uses
//   the weight_ih_l{l}/weight_hh_l{l}/bias_*_l{l} tensors, and layer l>0 takes
//   the previous layer's hidden output h as its input.
//
// LSTM math (per step, input x_t [H], prev h,c [H]; h0=c0=0):
//   z = W_ih · x_t + b_ih + W_hh · h + b_hh        # [4H]
//   i = sigmoid(z[0:H]); f = sigmoid(z[H:2H]); g = tanh(z[2H:3H]); o = sigmoid(z[3H:4H])
//   c' = f * c + i * g
//   h' = o * tanh(c')
// Gate order is PyTorch [input, forget, cell, output] stacked in the 4H dim.
//
// add_sos=true prepends a literal zero [H] "start of sequence" vector to the
// embedded sequence (matching NeMo predict(add_sos=True)), so the output has
// U+1 hidden states. The output is the sequence of h' states.
//
// Tensors (verbatim NeMo names; l = LSTM layer index 0..pred_rnn_layers-1):
//   decoder.prediction.embed.weight               ggml ne=[H, vocab+1]
//   decoder.prediction.dec_rnn.lstm.weight_ih_l{l} ggml ne=[H, 4H]
//   decoder.prediction.dec_rnn.lstm.weight_hh_l{l} ggml ne=[H, 4H]
//   decoder.prediction.dec_rnn.lstm.bias_ih_l{l}   ggml ne=[4H]
//   decoder.prediction.dec_rnn.lstm.bias_hh_l{l}   ggml ne=[4H]
class PredictionNet {
public:
    explicit PredictionNet(const ModelLoader& ml);
    ~PredictionNet();  // defined in the .cpp (StepReplay must be complete)

    // ids:    input label ids (length U). Each id indexes embed.weight.
    // add_sos: prepend a zero SOS step (output length becomes U+1).
    // out:    row-major [U_out, hidden] — out[u*hidden + h]
    // U_out:  number of output steps (U, or U+1 if add_sos)
    // hidden: pred_hidden (H)
    void forward(const std::vector<int32_t>& ids, bool add_sos,
                 std::vector<float>& out, int& U_out, int& hidden) const;

    // Returns a zero-initialised LSTM state (h and c each [hidden] zeros).
    PredState zero_state() const;

    // Advance the LSTM by one token.
    // token_id: embedding index (ignored when is_sos=true).
    // is_sos:   use the zero SOS input vector instead of the embedding.
    // in:       previous (h, c) state (use zero_state() for the first call).
    // g:        output — the new h' vector [hidden].
    // out_state: the new (h', c') state to carry to the next step.
    void step(int32_t token_id, bool is_sos,
              const PredState& in,
              std::vector<float>& g,
              PredState& out_state) const;

    // Advance the LSTM one token for N items at once.
    // token_ids[n]: embedding index for item n (ignored where is_sos[n]).
    // is_sos[n]:    use the zero SOS input for item n (1=true).
    // in:           batched prior state (h[L],c[L] each [H*N]).
    // g:            OUT, top-layer h' for all items [H*N] (item n at n*H).
    // out_state:    OUT, new batched (h',c').
    void step_batch(const std::vector<int32_t>& token_ids,
                    const std::vector<uint8_t>& is_sos,
                    const BatchedPredState& in,
                    std::vector<float>& g,
                    BatchedPredState& out_state) const;

    int hidden_size() const { return H_; }

    int num_layers() const { return n_layers_; }

private:
    const ModelLoader& ml_;
    int H_;       // pred_hidden
    int vocab_p1_; // vocab + 1 (embedding rows)
    int n_layers_; // pred_rnn_layers (stacked LSTM layers)

    // Host-side copy of the embedding table, lazily fetched on the first step()
    // via ggml_backend_tensor_get (works for both CPU and device-resident
    // weights). [vocab_p1_ * H_], row-major: embed_host_[id*H_ + h].
    mutable std::vector<float> embed_host_;

    // Replayable per-step LSTM graph (kept alive so ggml-cuda captures + replays
    // the per-token prediction net). Lazily built on the first step() call.
    struct StepReplay;
    mutable std::unique_ptr<StepReplay> replay_;

    // Per-batch-size replayable batched LSTM graph for step_batch (the batched
    // decode loop calls with a FIXED N per batch). Keyed on N, lazily built.
    struct StepReplayBatch;
    mutable std::unordered_map<int, std::unique_ptr<StepReplayBatch>> replay_batch_;
};

} // namespace pk
