#include "prompt_kernel.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#include <vector>

namespace pk {

PromptKernel::PromptKernel(const ModelLoader& ml) : ml_(ml) {
    const ParakeetConfig& c = ml_.config();
    present_     = c.prompt.present;
    num_prompts_ = (int)c.prompt.num_prompts;
    d_model_     = (int)c.d_model;
}

void PromptKernel::apply(const std::vector<float>& enc_out, int d_model, int T,
                         int prompt_index, std::vector<float>& out) const {
    if (!present_) { out = enc_out; return; }
    assert((int)enc_out.size() == d_model * T && "enc_out size mismatch");
    assert(prompt_index >= 0 && prompt_index < num_prompts_ && "prompt_index out of range");

    // Build the input in row-major [T, IN] (ggml ne0 = IN, the fastest axis):
    // rows [0, d_model) hold the encoder features (transposed from the
    // channels-first [d_model, T] input); rows [d_model, d_model+P) hold the
    // constant one-hot language vector. So the NeMo cat([encoded, onehot]) is a
    // fill, and the two Linear layers are plain matmuls.
    const int P  = num_prompts_;
    const int IN = d_model + P;
    std::vector<float> xbuf((size_t)IN * T, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < d_model; ++c)
            xbuf[(size_t)t * IN + c] = enc_out[(size_t)c * T + t];
        xbuf[(size_t)t * IN + d_model + prompt_index] = 1.0f;
    }

    // ggml graph: y = W2 · ReLU(W0 · x + b0) + b2, all on the persistent backend.
    //   prompt_kernel.0.weight ggml ne=[IN, 2D]   prompt_kernel.0.bias ne=[2D]
    //   prompt_kernel.2.weight ggml ne=[2D, D]    prompt_kernel.2.bias ne=[D]
    std::vector<float> y_td;  // run_graph fills row-major [T, D] (ggml ne=[D, T])
    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t x_ne[2] = { IN, T };
            ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, x_ne,
                                 xbuf.data(), (size_t)IN * T * sizeof(float));
            ggml_tensor* W0 = pk::clone_weight(ctx, ml_, "prompt_kernel.0.weight");
            ggml_tensor* b0 = pk::clone_weight(ctx, ml_, "prompt_kernel.0.bias");
            ggml_tensor* W2 = pk::clone_weight(ctx, ml_, "prompt_kernel.2.weight");
            ggml_tensor* b2 = pk::clone_weight(ctx, ml_, "prompt_kernel.2.bias");
            ggml_tensor* h = ggml_mul_mat(ctx, W0, x);          // [2D, T]
            h = ggml_add(ctx, h, b0);                           // bias broadcasts over T
            h = ggml_relu(ctx, h);
            ggml_tensor* yv = ggml_mul_mat(ctx, W2, h);         // [D, T]
            yv = ggml_add(ctx, yv, b2);                         // bias broadcasts over T
            return yv;                                          // ne=[D, T] -> row-major [T, D]
        }, y_td);
    assert(ok && "prompt_kernel graph failed");
    (void)ok;

    // Transpose row-major [T, D] back to channels-first [d_model, T] so the
    // result drops in for the raw encoder output the decoder consumes.
    out.resize((size_t)d_model * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < d_model; ++c)
            out[(size_t)c * T + t] = y_td[(size_t)t * d_model + c];
}

} // namespace pk
