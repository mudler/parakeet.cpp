#pragma once
#include "model_loader.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace pk {

// RNN-Transducer joint network — NeMo RNNTJoint.
//
// Architecture (NeMo RNNTJoint.joint):
//   enc_proj[t]  = joint.enc.weight  · enc[t]  + joint.enc.bias   (enc_hidden→joint_hidden)
//   pred_proj[u] = joint.pred.weight · pred[u] + joint.pred.bias  (pred_hidden→joint_hidden)
//   f[t,u]       = ReLU(enc_proj[t] + pred_proj[u])               (broadcast sum, ReLU on sum)
//   logits[t,u]  = joint_net.2.weight · f[t,u] + joint_net.2.bias (joint_hidden→V_plus)
//
// Output logits are RAW (no log_softmax), shape [T, U, V_plus] row-major.
// V_plus = vocab + 1 + num_durations  (full TDT output vector).
//
// enc input convention: row-major [T, enc_hidden], i.e. enc[t*enc_hidden + c].
//   (The Encoder outputs channels-first [enc_hidden, T]; the caller must
//    transpose before calling this function.)
//
// Tensors (verbatim NeMo names, ggml ne = reverse of torch):
//   joint.enc.weight          ggml ne=[enc_hidden=512, joint_hidden=640]
//   joint.enc.bias            ggml ne=[640]
//   joint.pred.weight         ggml ne=[pred_hidden=640, joint_hidden=640]
//   joint.pred.bias           ggml ne=[640]
//   joint.joint_net.2.weight  ggml ne=[joint_hidden=640, V_plus=1030]
//   joint.joint_net.2.bias    ggml ne=[V_plus=1030]
class Joint {
public:
    explicit Joint(const ModelLoader& ml);
    ~Joint();  // defined in the .cpp (StepReplay must be complete)

    // enc:  row-major [T, enc_hidden],  enc[t*enc_hidden + c]
    // pred: row-major [U, pred_hidden], pred[u*pred_hidden + h]
    // logits out: row-major [T, U, V_plus], logits[t*U*V_plus + u*V_plus + v]
    // V_plus = vocab + 1 + num_durations (written to the out param)
    void forward(const std::vector<float>& enc,  int T, int enc_hidden,
                 const std::vector<float>& pred, int U, int pred_hidden,
                 std::vector<float>& logits, int& V_plus) const;

    // --- Churn-free greedy-decode fast path (used by rnnt/tdt greedy loops) ---
    //
    // The greedy decoders call the joint once per emitted (t,u) step. The old
    // `forward(T=1,U=1,...)` rebuilt the FULL encoder projection per step (two
    // graphs/step, ~187 dispatches/utterance). Instead, precompute the encoder
    // projection over ALL T frames ONCE (one matmul, reused for every u), then
    // run a tight per-step joint (pred projection + ReLU + output projection) on
    // the PERSISTENT backend. Backend::compute reuses the backend + gallocr, so
    // the per-step graphs are churn-free; ggml's multithreaded matmul wins here
    // because the per-step joint is memory-bandwidth bound on the ~2.6 MB output
    // weight (measured ~15x faster/step than a single-threaded C++ matvec).
    // Quantized enc/pred weights are dequantized on the fly by ggml_mul_mat.

    // enc_proj[t][h] = enc.weight·enc[t] + enc.bias, over all T frames.
    // enc: row-major [T, enc_hidden]; out: row-major [T, joint_hidden].
    void precompute_enc_proj(const std::vector<float>& enc, int T, int enc_hidden,
                             std::vector<float>& enc_proj) const;

    // Per-step joint logits for a single (t,u):
    //   pred_proj[h] = pred.weight·g + pred.bias
    //   f[h]         = ReLU(enc_proj_t[h] + pred_proj[h])
    //   logits[v]    = joint_net.2.weight·f + joint_net.2.bias    → [V_plus]
    // enc_proj_t: pointer to enc_proj row for frame t ([joint_hidden]).
    // g:          prediction-net output for step u ([pred_hidden]).
    void step_logits(const float* enc_proj_t,
                     const float* g, int pred_hidden,
                     std::vector<float>& logits) const;

    // Batched per-step joint for N items.
    // enc_proj_gathered: [joint_hidden * N], each item's enc_proj row for its
    //   current frame (item n at offset n*joint_hidden).
    // g:                 [pred_hidden * N], batched pred output (item n at n*pred_hidden).
    // logits out:        [V_plus * N], item n at offset n*V_plus.
    void step_logits_batch(const float* enc_proj_gathered,
                           const float* g, int pred_hidden, int n,
                           std::vector<float>& logits) const;

    int joint_hidden() const { return joint_hidden_; }

    // V_plus = vocab_size + 1 + num_durations
    int V_plus()       const { return V_plus_; }
    int vocab_size()   const { return vocab_size_; }
    int num_durations() const { return num_durations_; }

private:
    const ModelLoader& ml_;
    int joint_hidden_;
    int vocab_size_;
    int num_durations_;
    int V_plus_;
    // Cached for shape asserts on the projection inputs.
    int enc_hidden_  = 0;   // ggml ne[0] of joint.enc.weight
    int pred_hidden_ = 0;   // ggml ne[0] of joint.pred.weight

    // Replayable per-step joint graph (kept alive so ggml-cuda's CUDA-graph
    // capture warms up and replays the per-token joint instead of launching
    // every op directly). Lazily built on the first step_logits call; lives in
    // the .cpp so the header need not include ReplayGraph.
    struct StepReplay;
    mutable std::unique_ptr<StepReplay> replay_;

    // Per-batch-size replayable joint graph for step_logits_batch (the batched
    // decode loop calls with a FIXED N per batch, so one captured graph per N is
    // reused for every round of that batch). Keyed on N, lazily built.
    struct StepReplayBatch;
    mutable std::unordered_map<int, std::unique_ptr<StepReplayBatch>> replay_batch_;
};

} // namespace pk
