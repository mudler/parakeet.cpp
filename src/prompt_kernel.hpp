#pragma once
#include "model_loader.hpp"
#include <vector>

namespace pk {

// Post-encoder prompt conditioning for multilingual nemotron models.
//
// Mirrors NeMo EncDecRNNTBPEModelWithPrompt.forward():
//   encoded[T, D] = transpose(encoder_out[D, T])
//   onehot[T, P]  = one_hot(prompt_index, num_prompts) broadcast over T
//   out[T, D]     = prompt_kernel(cat([encoded, onehot]))  // Linear->ReLU->Linear
// then transposed back to channels-first [D, T].
//
// The one-hot is constant over time (one language per utterance), so this is a
// concat + two matmuls + ReLU. Weights: prompt_kernel.0.{weight,bias} (D+P->2D),
// prompt_kernel.2.{weight,bias} (2D->D). present() is false for non-prompt
// models (callers skip apply()).
class PromptKernel {
public:
    explicit PromptKernel(const ModelLoader& ml);
    bool present() const { return present_; }

    // Apply the prompt projection to a channels-first encoder output
    // enc_out[d_model, T] for the given prompt_index, writing channels-first
    // out[d_model, T]. If present()==false this is a no-op copy (out = enc_out).
    void apply(const std::vector<float>& enc_out, int d_model, int T,
               int prompt_index, std::vector<float>& out) const;

private:
    const ModelLoader& ml_;
    bool present_ = false;
    int  num_prompts_ = 0;
    int  d_model_ = 0;
};

} // namespace pk
