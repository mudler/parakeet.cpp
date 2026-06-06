#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
namespace pk {
// Cache-aware streaming params (Phase 5), populated only for streaming models
// (att_context_style != "regular"). Mirrors NeMo CacheAwareStreamingConfig.
// List fields are stored as int32 arrays in the GGUF (e.g. chunk_size=[9,16]).
struct StreamingCfg {
    std::vector<int32_t> chunk_size;              // per-step encoder chunk frames
    std::vector<int32_t> shift_size;              // per-step shift frames
    std::vector<int32_t> pre_encode_cache_size;   // pre-encode cache (mel frames)
    int32_t cache_drop_size=0;
    int32_t last_channel_cache_size=0;            // attention left-context cache
    int32_t valid_out_len=0;                      // valid encoder frames per step
    int32_t drop_extra_pre_encoded=0;
    bool present=false;                           // true only for streaming models
};
// Prompt-conditioning config (multilingual nemotron). present=false for all
// existing models, which then skip the prompt stage entirely.
struct PromptCfg {
    bool present = false;
    uint32_t num_prompts = 0;
    std::string default_lang;                       // e.g. "auto"
    std::vector<std::string> dict_keys;             // locale strings
    std::vector<int32_t>     dict_vals;             // parallel prompt indices
    // Resolve a locale to its prompt index; -1 if unknown.
    int lang_to_index(const std::string& lang) const {
        for (size_t i = 0; i < dict_keys.size(); ++i)
            if (dict_keys[i] == lang) return (int)dict_vals[i];
        return -1;
    }
    // Resolve target_lang to its prompt index, applying the model default for an
    // empty string. THROWS std::runtime_error on an unknown locale. Shared by the
    // offline (Model::resolve_prompt_index) and streaming (StreamingSession ctor)
    // paths so both reject typos identically (matches the C-API contract).
    int resolve_index_or_throw(const std::string& target_lang) const;
};
struct ParakeetConfig {
    std::string arch;
    // encoder
    uint32_t feat_in=0, d_model=0, n_layers=0, n_heads=0, ff_dim=0, conv_kernel=0;
    std::string conv_norm_type;
    uint32_t subsampling_factor=0, subsampling_conv_channels=0, pos_emb_max_len=5000;
    bool xscaling=true;
    // cache-aware streaming / causal config (Phase 5; offline-safe defaults)
    int32_t att_context_left=-1, att_context_right=-1; // [-1,-1] = full context
    std::string att_context_style="regular";            // or "chunked_limited"
    bool causal_downsampling=false;                     // causal subsampling pad
    bool conv_causal=false;                             // causal depthwise conv pad
    bool use_bias=true;     // false for nemotron (encoder linears have no bias)
    StreamingCfg streaming;
    PromptCfg prompt;       // prompt conditioning (present=false for non-prompt)
    // preprocessor
    uint32_t sample_rate=16000, n_mels=0, n_fft=0, win_length=0, hop_length=0;
    float preemph=0.0f, mag_power=2.0f, log_zero_guard=0.0f;
    std::string normalize;
    // transducer (optional)
    uint32_t pred_hidden=0, pred_rnn_layers=0, joint_hidden=0;
    std::string joint_activation;
    std::vector<int32_t> tdt_durations;
    uint32_t max_symbols=10;  // greedy max symbols per frame (NeMo default 10)
    // vocab
    uint32_t vocab_size=0, blank_id=0;
    std::vector<std::string> tokenizer_pieces;
};
class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    bool load(const std::string& path);
    const ParakeetConfig& config() const { return cfg_; }
    const std::vector<std::string>& tokenizer_pieces() const { return cfg_.tokenizer_pieces; }
    ggml_tensor* tensor(const std::string& name) const; // nullptr if absent
    ggml_context* ggml_ctx() const { return ctx_; }

    // Give every weight tensor a CPU backend buffer (ONCE), so graphs can
    // reference the loader's tensors DIRECTLY as leaves with zero per-call
    // copying. The GGUF is loaded with no_alloc=false, so all weight data
    // already lives in one contiguous ctx mem_buffer; this wraps that exact
    // memory via ggml_backend_cpu_buffer_from_ptr (no data movement) and points
    // every tensor's ->buffer at it. After this the weights are valid graph
    // leaves on the CPU backend and reshapes/views of them resolve their data
    // at build time. Idempotent; safe to call once at load. The backend must be
    // the same CPU backend the compute path uses. Returns false on failure.
    bool realize_weights(ggml_backend_t backend);
    bool weights_realized() const { return weights_buf_ != nullptr; }
private:
    ParakeetConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    // CPU backend: wraps ctx_ mem_buffer (zero-copy). Device backend: owns the
    // device buffer holding the uploaded weights (mirrored into device_ctx_).
    ggml_backend_buffer_t weights_buf_ = nullptr;
    ggml_context* device_ctx_ = nullptr;  // no_alloc mirror ctx for device weights
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};
}
