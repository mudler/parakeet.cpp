#pragma once
// Weights of the Nemotron-H style MoE language model (arch "nemotron-h-moe-llm"), as written by
// scripts/convert_nemotron_speechlm_llm.py. Layout and key list: docs/speechlm-llm.md.
//
// This is the loader only: it opens the GGUF, checks every key and every tensor (name, shape,
// type) against the layout the layer pattern implies, and gives access to the weights. There is
// no graph and no forward pass.
//
// open() reads only the header and the tensor table. The weights are not read until realize(),
// which allocates one backend buffer and copies the data tensor by tensor from the file through
// a bounded staging buffer, so the file is never held in memory twice.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace pk {

enum class LlmLayer { Mamba, Moe, Attention };

struct LlmConfig {
    std::vector<LlmLayer> layers;  // one mixer per layer, in order
    uint32_t hidden = 0, vocab = 0, context = 0;
    float rms_eps = 0.0f;
    // Mamba-2
    uint32_t ssm_heads = 0, ssm_head_dim = 0, ssm_inner = 0, ssm_state = 0, ssm_groups = 0, ssm_conv = 0,
             ssm_chunk = 0;
    float ssm_dt_min = 0.0f;
    // attention (no RoPE)
    uint32_t attn_heads = 0, attn_kv_heads = 0, attn_head_dim = 0;
    // MoE
    uint32_t n_experts = 0, n_experts_used = 0, n_shared = 0, expert_ff = 0, shared_ff = 0;
    float routed_scale = 1.0f;
    bool norm_topk = true;
    // special tokens
    uint32_t audio_id = 0, eos_id = 0, bos_id = 0, pad_id = 0;

    uint32_t n_layers() const { return (uint32_t)layers.size(); }
    uint32_t ssm_conv_dim() const { return ssm_inner + 2 * ssm_groups * ssm_state; }
    uint32_t ssm_in_dim() const { return ssm_inner + ssm_conv_dim() + ssm_heads; }
};

// One tensor the layout requires. ne is in ggml order (ne[0] is the contiguous dimension).
struct LlmTensorSpec {
    std::string name;
    int64_t ne[3] = {1, 1, 1};
    bool f32_only = false;  // otherwise F32, F16 or Q8_0
};

// The exact tensor list a config implies.
std::vector<LlmTensorSpec> llm_expected_tensors(const LlmConfig& cfg);

class LlmLoader {
public:
    ~LlmLoader();
    LlmLoader(const LlmLoader&) = delete;
    LlmLoader& operator=(const LlmLoader&) = delete;

    // Opens and validates. Returns nullptr and sets *err (when not null) on any mismatch.
    static std::unique_ptr<LlmLoader> open(const std::string& path, std::string* err);

    const LlmConfig& config() const { return cfg_; }
    // Tensor by GGUF name (descriptor: type, shape; data only after realize()). nullptr if absent.
    ggml_tensor* tensor(const std::string& name) const;
    // "blk.<layer>.<suffix>", for example layer_tensor(3, "ffn_up_exps.weight").
    ggml_tensor* layer_tensor(uint32_t layer, const std::string& suffix) const;
    size_t n_tensors() const { return n_tensors_; }
    size_t weights_bytes() const { return weights_bytes_; }

    // Allocates the weights on `backend` and copies them from the file. Idempotent.
    bool realize(ggml_backend_t backend, std::string* err);
    bool realized() const { return buf_ != nullptr; }

private:
    LlmLoader() = default;
    LlmConfig cfg_;
    std::string path_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    size_t n_tensors_ = 0, weights_bytes_ = 0;
    size_t data_offset_ = 0;
};

}  // namespace pk
