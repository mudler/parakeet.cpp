// LlmLoader on a tiny synthetic GGUF (2 Mamba-2, 2 MoE and 1 attention layer): the happy path
// for F32 / F16 / Q8_0 weights, and a refusal with a clear message for every kind of mismatch.
// No checkpoint needed. When PARAKEET_TEST_LLM_GGUF is set (check_llm_convert.py does this), that
// file, written by the Python converter, is loaded as well.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llm_loader.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static const std::string kTmp = "test_llm_loader.tmp.gguf";

static pk::LlmConfig tiny_config() {
    pk::LlmConfig c;
    c.layers = {pk::LlmLayer::Mamba, pk::LlmLayer::Moe, pk::LlmLayer::Mamba, pk::LlmLayer::Moe,
                pk::LlmLayer::Attention};
    c.hidden = 64; c.vocab = 96; c.context = 4096; c.rms_eps = 1e-5f;
    c.ssm_heads = 4; c.ssm_head_dim = 8; c.ssm_inner = 32; c.ssm_state = 16; c.ssm_groups = 2;
    c.ssm_conv = 4; c.ssm_chunk = 8; c.ssm_dt_min = 0.001f;
    c.attn_heads = 4; c.attn_kv_heads = 2; c.attn_head_dim = 16;
    c.n_experts = 4; c.n_experts_used = 2; c.n_shared = 1; c.expert_ff = 32; c.shared_ff = 64;
    c.routed_scale = 2.5f; c.norm_topk = true;
    c.audio_id = 96; c.eos_id = 11; c.bos_id = 1; c.pad_id = 0;
    return c;
}

// What to break in the file.
struct Mut {
    ggml_type linear = GGML_TYPE_F32;     // type of the non-F32-only tensors
    std::string drop_key;                 // metadata key to leave out
    std::string wrong_type_key;           // metadata key written with the wrong type
    std::string arch = "nemotron-h-moe-llm";
    std::string drop_tensor;              // tensor to leave out
    std::string extra_tensor;             // tensor to add
    std::string reshape_tensor;           // tensor whose ne[1] is one too large
    std::string retype_tensor;            // tensor written as F16 although it must be F32
    std::vector<std::string> pattern = {"mamba", "moe", "mamba", "moe", "attention"};
    uint32_t block_count = 5;
    bool use_rope = false;
};

static void write_gguf(const Mut& m) {
    const pk::LlmConfig c = tiny_config();
    gguf_context* g = gguf_init_empty();
    const std::string p = "nemotron-h-moe-llm.";
    auto u32 = [&](const char* k, uint32_t v) { if (m.drop_key != k) { if (m.wrong_type_key == k) gguf_set_val_str(g, (p + k).c_str(), "x"); else gguf_set_val_u32(g, (p + k).c_str(), v); } };
    auto f32 = [&](const char* k, float v) { if (m.drop_key != k) gguf_set_val_f32(g, (p + k).c_str(), v); };
    auto bl = [&](const char* k, bool v) { if (m.drop_key != k) gguf_set_val_bool(g, (p + k).c_str(), v); };
    auto str = [&](const char* k, const char* v) { if (m.drop_key != k) gguf_set_val_str(g, (p + k).c_str(), v); };
    gguf_set_val_str(g, "general.architecture", m.arch.c_str());
    u32("block_count", m.block_count);
    {
        std::vector<const char*> pat;
        for (auto& s : m.pattern) pat.push_back(s.c_str());
        gguf_set_arr_str(g, (p + "layer_pattern").c_str(), pat.data(), pat.size());
    }
    u32("embedding_length", c.hidden); u32("vocab_size", c.vocab); u32("context_length", c.context);
    f32("rms_norm_eps", c.rms_eps); bl("tie_word_embeddings", false);
    u32("ssm.head_count", c.ssm_heads); u32("ssm.head_dim", c.ssm_head_dim); u32("ssm.inner_size", c.ssm_inner);
    u32("ssm.state_size", c.ssm_state); u32("ssm.group_count", c.ssm_groups); u32("ssm.conv_kernel", c.ssm_conv);
    u32("ssm.chunk_size", c.ssm_chunk); f32("ssm.time_step_min", c.ssm_dt_min);
    u32("attention.head_count", c.attn_heads); u32("attention.head_count_kv", c.attn_kv_heads);
    u32("attention.head_dim", c.attn_head_dim); bl("attention.use_rope", m.use_rope);
    u32("expert_count", c.n_experts); u32("expert_used_count", c.n_experts_used); u32("expert_shared_count", c.n_shared);
    u32("expert_feed_forward_length", c.expert_ff); u32("expert_shared_feed_forward_length", c.shared_ff);
    f32("expert_weights_scale", c.routed_scale); bl("expert_weights_norm", c.norm_topk);
    str("expert_gating_func", "sigmoid"); str("ffn_activation", "relu2");
    u32("token.audio_id", c.audio_id); u32("token.eos_id", c.eos_id); u32("token.bos_id", c.bos_id);
    u32("token.pad_id", c.pad_id);

    ggml_init_params ip{64u << 20, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    auto specs = pk::llm_expected_tensors(c);
    if (!m.extra_tensor.empty()) { pk::LlmTensorSpec s; s.name = m.extra_tensor; s.ne[0] = 8; specs.push_back(s); }
    for (auto& s : specs) {
        if (s.name == m.drop_tensor) continue;
        ggml_type ty = s.f32_only ? GGML_TYPE_F32 : m.linear;
        if (s.name == m.retype_tensor) ty = GGML_TYPE_F16;
        int64_t ne[3] = {s.ne[0], s.ne[1], s.ne[2]};
        if (s.name == m.reshape_tensor) ne[1] += 1;
        int nd = ne[2] > 1 ? 3 : (ne[1] > 1 ? 2 : 1);
        ggml_tensor* t = ggml_new_tensor(ctx, ty, nd, ne);
        ggml_set_name(t, s.name.c_str());
        std::memset(t->data, 0, ggml_nbytes(t));
        if (ty == GGML_TYPE_F32) {  // a recognisable pattern, so realize() can be checked
            float* d = (float*)t->data;
            for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = (float)(i % 251) * 0.01f;
        }
        gguf_add_tensor(g, t);
    }
    CHECK(gguf_write_to_file(g, kTmp.c_str(), false));
    gguf_free(g);
    ggml_free(ctx);
}

static void expect_ok(const Mut& m, int line) {
    write_gguf(m);
    std::string err;
    auto l = pk::LlmLoader::open(kTmp, &err);
    if (!l) { std::fprintf(stderr, "FAIL line %d: open failed: %s\n", line, err.c_str()); ++failures; return; }
    const pk::LlmConfig& c = l->config();
    CHECK(c.n_layers() == 5 && c.layers[4] == pk::LlmLayer::Attention && c.layers[1] == pk::LlmLayer::Moe);
    CHECK(c.hidden == 64 && c.vocab == 96 && c.ssm_in_dim() == 132 && c.ssm_conv_dim() == 96);
    CHECK(c.audio_id == 96 && c.eos_id == 11 && c.routed_scale == 2.5f && c.n_experts_used == 2);
    CHECK(l->n_tensors() == pk::llm_expected_tensors(c).size());
    ggml_tensor* up = l->layer_tensor(1, "ffn_up_exps.weight");
    CHECK(up && up->ne[0] == 64 && up->ne[1] == 32 && up->ne[2] == 4);
    CHECK(l->tensor("blk.0.ssm_a") && l->tensor("blk.0.ssm_a")->type == GGML_TYPE_F32);
    CHECK(!l->tensor("blk.1.ssm_a"));  // an MoE layer has no SSM tensors
}

static void expect_fail(const Mut& m, const char* needle, int line) {
    write_gguf(m);
    std::string err;
    auto l = pk::LlmLoader::open(kTmp, &err);
    if (l || err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "FAIL line %d: want an error with \"%s\", got %s \"%s\"\n", line, needle,
                     l ? "success" : "failure", err.c_str());
        ++failures;
    }
}

static void check_realize(const std::string& path, bool strict_values) {
    std::string err;
    auto l = pk::LlmLoader::open(path, &err);
    if (!l) { std::fprintf(stderr, "FAIL: open %s: %s\n", path.c_str(), err.c_str()); ++failures; return; }
    ggml_backend_t be = ggml_backend_cpu_init();
    CHECK(l->realize(be, &err));
    CHECK(l->realized() && l->realize(be, &err));
    ggml_tensor* t = l->tensor("blk.0.ssm_d");
    CHECK(t && t->buffer);
    if (t && t->buffer && strict_values) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        CHECK(v[0] == 0.0f && v[3] == 0.03f);
    }
    ggml_backend_free(be);
}

int main() {
    Mut ok;
    expect_ok(ok, __LINE__);
    check_realize(kTmp, true);
    Mut f16; f16.linear = GGML_TYPE_F16; expect_ok(f16, __LINE__);
    Mut q8; q8.linear = GGML_TYPE_Q8_0; expect_ok(q8, __LINE__);
    check_realize(kTmp, false);

    {   // not a GGUF
        std::string err;
        FILE* f = std::fopen(kTmp.c_str(), "wb"); std::fputs("not a gguf", f); std::fclose(f);
        CHECK(!pk::LlmLoader::open(kTmp, &err) && err.find("cannot read") != std::string::npos);
        CHECK(!pk::LlmLoader::open("does-not-exist.gguf", &err) && err.find("cannot read") != std::string::npos);
    }
    { Mut m; m.arch = "parakeet"; expect_fail(m, "expected \"nemotron-h-moe-llm\"", __LINE__); }
    { Mut m; m.drop_key = "expert_count"; expect_fail(m, "missing key nemotron-h-moe-llm.expert_count", __LINE__); }
    { Mut m; m.wrong_type_key = "ssm.state_size"; expect_fail(m, "ssm.state_size has the wrong type", __LINE__); }
    { Mut m; m.block_count = 6; expect_fail(m, "does not match the layer_pattern length", __LINE__); }
    { Mut m; m.pattern[2] = "lstm"; expect_fail(m, "unknown entry \"lstm\"", __LINE__); }
    { Mut m; m.use_rope = true; expect_fail(m, "no RoPE", __LINE__); }
    // A pattern change moves the tensors: layer 2 becomes attention, so its SSM tensors are unexpected.
    { Mut m; m.pattern[2] = "attention"; expect_fail(m, "unexpected tensor blk.2.ssm_in.weight", __LINE__); }
    { Mut m; m.drop_tensor = "blk.3.ffn_exp_probs_b"; expect_fail(m, "missing tensor blk.3.ffn_exp_probs_b", __LINE__); }
    { Mut m; m.extra_tensor = "blk.0.surprise.weight"; expect_fail(m, "unexpected tensor blk.0.surprise.weight", __LINE__); }
    // The corrupted shape: one expert tensor with a wrong dimension.
    { Mut m; m.reshape_tensor = "blk.1.ffn_up_exps.weight";
      expect_fail(m, "tensor blk.1.ffn_up_exps.weight has shape [64,33,4,1] (ggml order), expected [64,32,4]", __LINE__); }
    { Mut m; m.reshape_tensor = "token_embd.weight"; expect_fail(m, "token_embd.weight has shape", __LINE__); }
    { Mut m; m.retype_tensor = "blk.1.ffn_gate_inp.weight"; expect_fail(m, "blk.1.ffn_gate_inp.weight has type f16, expected F32", __LINE__); }
    { Mut m; m.retype_tensor = "blk.0.ssm_a"; expect_fail(m, "blk.0.ssm_a has type f16", __LINE__); }
    {   // a truncated file
        Mut m; write_gguf(m);
        FILE* f = std::fopen(kTmp.c_str(), "rb"); std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fclose(f);
        std::vector<char> buf(n - 1000);
        f = std::fopen(kTmp.c_str(), "rb"); CHECK(std::fread(buf.data(), 1, buf.size(), f) == buf.size()); std::fclose(f);
        f = std::fopen(kTmp.c_str(), "wb"); std::fwrite(buf.data(), 1, buf.size(), f); std::fclose(f);
        std::string err;
        CHECK(!pk::LlmLoader::open(kTmp, &err));  // gguf_init_from_file or the size check refuses it
    }
    std::remove(kTmp.c_str());

    if (const char* p = std::getenv("PARAKEET_TEST_LLM_GGUF")) {
        check_realize(p, false);
        std::string err;
        auto l = pk::LlmLoader::open(p, &err);
        CHECK(l && l->config().n_layers() == 5);
    }
    if (failures) return 1;
    std::printf("test_llm_loader OK\n");
    return 0;
}
