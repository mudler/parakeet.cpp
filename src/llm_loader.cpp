#include "llm_loader.hpp"

#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

namespace pk {
namespace {

const char* kArch = "nemotron-h-moe-llm";

void set_err(std::string* e, const std::string& m) {
    if (e) *e = m;
}

std::string shape_str(const int64_t* ne, int n = 3) {
    std::string s = "[";
    for (int i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(ne[i]);
    return s + "]";
}

// Typed metadata reads that report instead of abort.
struct Kv {
    gguf_context* g;
    std::string prefix;
    std::vector<std::string> problems;

    int64_t find(const std::string& k, gguf_type want) {
        const int64_t id = gguf_find_key(g, (prefix + k).c_str());
        if (id < 0) {
            problems.push_back("missing key " + prefix + k);
            return -1;
        }
        if (gguf_get_kv_type(g, id) != want) {
            problems.push_back("key " + prefix + k + " has the wrong type");
            return -1;
        }
        return id;
    }
    uint32_t u32(const std::string& k) {
        const int64_t id = find(k, GGUF_TYPE_UINT32);
        return id < 0 ? 0 : gguf_get_val_u32(g, id);
    }
    float f32(const std::string& k) {
        const int64_t id = find(k, GGUF_TYPE_FLOAT32);
        return id < 0 ? 0.0f : gguf_get_val_f32(g, id);
    }
    bool boolean(const std::string& k) {
        const int64_t id = find(k, GGUF_TYPE_BOOL);
        return id < 0 ? false : gguf_get_val_bool(g, id);
    }
    std::string str(const std::string& k) {
        const int64_t id = find(k, GGUF_TYPE_STRING);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id));
    }
    std::vector<std::string> str_array(const std::string& k) {
        std::vector<std::string> out;
        const int64_t id = find(k, GGUF_TYPE_ARRAY);
        if (id < 0) return out;
        if (gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
            problems.push_back("key " + prefix + k + " is not a string array");
            return out;
        }
        for (size_t i = 0, n = gguf_get_arr_n(g, id); i < n; ++i) out.push_back(gguf_get_arr_str(g, id, i));
        return out;
    }
};

std::string join(const std::vector<std::string>& v, size_t max = 8) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < max; ++i) s += (i ? "; " : "") + v[i];
    if (v.size() > max) s += "; ... (" + std::to_string(v.size() - max) + " more)";
    return s;
}

}  // namespace

std::vector<LlmTensorSpec> llm_expected_tensors(const LlmConfig& c) {
    std::vector<LlmTensorSpec> t;
    auto add = [&](const std::string& n, int64_t a, int64_t b = 1, int64_t d = 1, bool f32 = false) {
        LlmTensorSpec s;
        s.name = n;
        s.ne[0] = a;
        s.ne[1] = b;
        s.ne[2] = d;
        s.f32_only = f32;
        t.push_back(s);
    };
    add("token_embd.weight", c.hidden, c.vocab);
    add("output_norm.weight", c.hidden, 1, 1, true);
    add("output.weight", c.hidden, c.vocab);
    for (uint32_t i = 0; i < c.n_layers(); ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        add(b + "attn_norm.weight", c.hidden, 1, 1, true);
        switch (c.layers[i]) {
            case LlmLayer::Mamba:
                add(b + "ssm_in.weight", c.hidden, c.ssm_in_dim());
                add(b + "ssm_conv1d.weight", c.ssm_conv, c.ssm_conv_dim(), 1, true);
                add(b + "ssm_conv1d.bias", c.ssm_conv_dim(), 1, 1, true);
                add(b + "ssm_a", c.ssm_heads, 1, 1, true);
                add(b + "ssm_d", c.ssm_heads, 1, 1, true);
                add(b + "ssm_dt.bias", c.ssm_heads, 1, 1, true);
                add(b + "ssm_norm.weight", c.ssm_inner, 1, 1, true);
                add(b + "ssm_out.weight", c.ssm_inner, c.hidden);
                break;
            case LlmLayer::Attention:
                add(b + "attn_q.weight", c.hidden, (int64_t)c.attn_heads * c.attn_head_dim);
                add(b + "attn_k.weight", c.hidden, (int64_t)c.attn_kv_heads * c.attn_head_dim);
                add(b + "attn_v.weight", c.hidden, (int64_t)c.attn_kv_heads * c.attn_head_dim);
                add(b + "attn_output.weight", (int64_t)c.attn_heads * c.attn_head_dim, c.hidden);
                break;
            case LlmLayer::Moe:
                add(b + "ffn_gate_inp.weight", c.hidden, c.n_experts, 1, true);
                add(b + "ffn_exp_probs_b", c.n_experts, 1, 1, true);
                add(b + "ffn_up_exps.weight", c.hidden, c.expert_ff, c.n_experts);
                add(b + "ffn_down_exps.weight", c.expert_ff, c.hidden, c.n_experts);
                add(b + "ffn_up_shexp.weight", c.hidden, c.shared_ff);
                add(b + "ffn_down_shexp.weight", c.shared_ff, c.hidden);
                break;
        }
    }
    return t;
}

LlmLoader::~LlmLoader() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    if (gguf_) gguf_free(gguf_);
}

ggml_tensor* LlmLoader::tensor(const std::string& name) const {
    return ctx_ ? ggml_get_tensor(ctx_, name.c_str()) : nullptr;
}

ggml_tensor* LlmLoader::layer_tensor(uint32_t layer, const std::string& suffix) const {
    return tensor("blk." + std::to_string(layer) + "." + suffix);
}

std::unique_ptr<LlmLoader> LlmLoader::open(const std::string& path, std::string* err) {
    gguf_init_params gp{/*no_alloc*/ true, /*ctx*/ nullptr};
    std::unique_ptr<LlmLoader> m(new LlmLoader());
    gp.ctx = &m->ctx_;
    m->gguf_ = gguf_init_from_file(path.c_str(), gp);
    if (!m->gguf_) {
        set_err(err, "cannot read " + path + " as a GGUF file");
        return nullptr;
    }
    m->path_ = path;
    gguf_context* g = m->gguf_;
    auto fail = [&](const std::string& msg) -> std::unique_ptr<LlmLoader> {
        set_err(err, path + ": " + msg);
        return nullptr;
    };

    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(g, arch) != GGUF_TYPE_STRING)
        return fail("no general.architecture; this is not an LLM GGUF of this model");
    if (std::strcmp(gguf_get_val_str(g, arch), kArch) != 0)
        return fail(std::string("general.architecture is \"") + gguf_get_val_str(g, arch) + "\", expected \"" + kArch +
                    "\"");

    // ---- metadata ----
    Kv kv{g, std::string(kArch) + ".", {}};
    LlmConfig& c = m->cfg_;
    const uint32_t block_count = kv.u32("block_count");
    for (const std::string& s : kv.str_array("layer_pattern")) {
        if (s == "mamba") c.layers.push_back(LlmLayer::Mamba);
        else if (s == "moe") c.layers.push_back(LlmLayer::Moe);
        else if (s == "attention") c.layers.push_back(LlmLayer::Attention);
        else kv.problems.push_back("layer_pattern has the unknown entry \"" + s + "\"");
    }
    c.hidden = kv.u32("embedding_length");
    c.vocab = kv.u32("vocab_size");
    c.context = kv.u32("context_length");
    c.rms_eps = kv.f32("rms_norm_eps");
    if (kv.boolean("tie_word_embeddings")) kv.problems.push_back("tie_word_embeddings is true; tied weights are not supported");
    c.ssm_heads = kv.u32("ssm.head_count");
    c.ssm_head_dim = kv.u32("ssm.head_dim");
    c.ssm_inner = kv.u32("ssm.inner_size");
    c.ssm_state = kv.u32("ssm.state_size");
    c.ssm_groups = kv.u32("ssm.group_count");
    c.ssm_conv = kv.u32("ssm.conv_kernel");
    c.ssm_chunk = kv.u32("ssm.chunk_size");
    c.ssm_dt_min = kv.f32("ssm.time_step_min");
    c.attn_heads = kv.u32("attention.head_count");
    c.attn_kv_heads = kv.u32("attention.head_count_kv");
    c.attn_head_dim = kv.u32("attention.head_dim");
    if (kv.boolean("attention.use_rope")) kv.problems.push_back("attention.use_rope is true; this model has no RoPE");
    c.n_experts = kv.u32("expert_count");
    c.n_experts_used = kv.u32("expert_used_count");
    c.n_shared = kv.u32("expert_shared_count");
    c.expert_ff = kv.u32("expert_feed_forward_length");
    c.shared_ff = kv.u32("expert_shared_feed_forward_length");
    c.routed_scale = kv.f32("expert_weights_scale");
    c.norm_topk = kv.boolean("expert_weights_norm");
    if (kv.str("expert_gating_func") != "sigmoid") kv.problems.push_back("expert_gating_func must be \"sigmoid\"");
    if (kv.str("ffn_activation") != "relu2") kv.problems.push_back("ffn_activation must be \"relu2\"");
    c.audio_id = kv.u32("token.audio_id");
    c.eos_id = kv.u32("token.eos_id");
    c.bos_id = kv.u32("token.bos_id");
    c.pad_id = kv.u32("token.pad_id");
    if (!kv.problems.empty()) return fail(join(kv.problems));

    // ---- consistency of the values ----
    std::vector<std::string> bad;
    if (c.layers.empty() || block_count != c.layers.size())
        bad.push_back("block_count " + std::to_string(block_count) + " does not match the layer_pattern length " +
                      std::to_string(c.layers.size()));
    for (auto [name, v] : {std::pair<const char*, uint32_t>{"embedding_length", c.hidden}, {"vocab_size", c.vocab},
                           {"ssm.head_count", c.ssm_heads}, {"ssm.head_dim", c.ssm_head_dim},
                           {"ssm.state_size", c.ssm_state}, {"ssm.group_count", c.ssm_groups},
                           {"ssm.conv_kernel", c.ssm_conv}, {"attention.head_count", c.attn_heads},
                           {"attention.head_count_kv", c.attn_kv_heads}, {"attention.head_dim", c.attn_head_dim},
                           {"expert_count", c.n_experts}, {"expert_used_count", c.n_experts_used},
                           {"expert_feed_forward_length", c.expert_ff},
                           {"expert_shared_feed_forward_length", c.shared_ff}})
        if (v == 0) bad.push_back(std::string(name) + " is 0");
    if (!bad.empty()) return fail(join(bad));
    if (c.ssm_inner != c.ssm_heads * c.ssm_head_dim)
        bad.push_back("ssm.inner_size " + std::to_string(c.ssm_inner) + " is not head_count * head_dim");
    if (c.ssm_heads % c.ssm_groups != 0) bad.push_back("ssm.head_count is not a multiple of ssm.group_count");
    if (c.attn_heads % c.attn_kv_heads != 0) bad.push_back("attention.head_count is not a multiple of head_count_kv");
    if (c.n_experts_used > c.n_experts) bad.push_back("expert_used_count is larger than expert_count");
    if (c.n_shared != 1) bad.push_back("expert_shared_count must be 1");
    if (c.audio_id < c.vocab) bad.push_back("token.audio_id must be outside the vocabulary (>= vocab_size)");
    if (c.eos_id >= c.vocab) bad.push_back("token.eos_id is outside the vocabulary");
    if (!bad.empty()) return fail(join(bad));

    // ---- tensors: the exact expected set, with shape and type ----
    const std::vector<LlmTensorSpec> want = llm_expected_tensors(c);
    std::set<std::string> seen;
    for (const LlmTensorSpec& s : want) {
        seen.insert(s.name);
        const ggml_tensor* t = ggml_get_tensor(m->ctx_, s.name.c_str());
        if (!t) {
            bad.push_back("missing tensor " + s.name);
            continue;
        }
        if (t->ne[0] != s.ne[0] || t->ne[1] != s.ne[1] || t->ne[2] != s.ne[2] || t->ne[3] != 1) {
            bad.push_back("tensor " + s.name + " has shape " + shape_str(t->ne, 4) + " (ggml order), expected " +
                          shape_str(s.ne));
            continue;
        }
        const bool ok = s.f32_only ? t->type == GGML_TYPE_F32
                                   : (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0);
        if (!ok)
            bad.push_back(std::string("tensor ") + s.name + " has type " + ggml_type_name(t->type) +
                          (s.f32_only ? ", expected F32" : ", expected F32, F16 or Q8_0"));
    }
    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; ++i) {
        const char* n = gguf_get_tensor_name(g, i);
        if (!seen.count(n)) bad.push_back(std::string("unexpected tensor ") + n);
    }
    if (!bad.empty()) return fail(join(bad));

    // ---- the file must hold all the data the table promises ----
    m->data_offset_ = gguf_get_data_offset(g);
    size_t end = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const ggml_tensor* t = ggml_get_tensor(m->ctx_, gguf_get_tensor_name(g, i));
        end = std::max(end, m->data_offset_ + gguf_get_tensor_offset(g, i) + ggml_nbytes(t));
        m->weights_bytes_ += ggml_nbytes(t);
    }
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long long size = (long long)std::ftell(f);
        std::fclose(f);
        if (size < (long long)end)
            return fail("the file is truncated: " + std::to_string(size) + " bytes, the tensor table needs " +
                        std::to_string(end));
    }
    m->n_tensors_ = (size_t)nt;
    return m;
}

bool LlmLoader::realize(ggml_backend_t backend, std::string* err) {
    if (buf_) return true;
    if (!backend || !ctx_) {
        set_err(err, "realize: no backend");
        return false;
    }
    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buf_) {
        set_err(err, "cannot allocate " + std::to_string(weights_bytes_) + " bytes of weights on the backend");
        return false;
    }
    FILE* f = std::fopen(path_.c_str(), "rb");
    if (!f) {
        set_err(err, "cannot open " + path_);
        return false;
    }
    const size_t kStage = 64u << 20;
    std::vector<char> stage;
    bool ok = true;
    for (int64_t i = 0, n = gguf_get_n_tensors(gguf_); i < n && ok; ++i) {
        ggml_tensor* t = ggml_get_tensor(ctx_, gguf_get_tensor_name(gguf_, i));
        const size_t size = ggml_nbytes(t);
        const size_t off = data_offset_ + gguf_get_tensor_offset(gguf_, i);
        stage.resize(std::min(size, kStage));
        for (size_t done = 0; done < size && ok; done += stage.size()) {
            const size_t len = std::min(stage.size(), size - done);
#ifdef _WIN32
            ok = _fseeki64(f, (long long)(off + done), SEEK_SET) == 0;
#else
            ok = fseeko(f, (off_t)(off + done), SEEK_SET) == 0;
#endif
            ok = ok && std::fread(stage.data(), 1, len, f) == len;
            if (ok) ggml_backend_tensor_set(t, stage.data(), done, len);
        }
        if (!ok) set_err(err, std::string("read error on tensor ") + ggml_get_name(t));
    }
    std::fclose(f);
    if (!ok) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    return ok;
}

}  // namespace pk
