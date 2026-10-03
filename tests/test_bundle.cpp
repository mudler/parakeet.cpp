// Bundle GGUF (docs/bundle.md): header parsing, component selection, partial
// loading, the Silero component, malformed files and the C-API entry points.
// No model file is needed: the test writes small synthetic GGUF files itself.
// A synthetic "asr" component only has the keys and tensors that ModelLoader
// reads; the Silero component has the real tensor shapes with random weights.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bundle.hpp"
#include "ggml.h"
#include "gguf.h"
#include "model_loader.hpp"
#include "parakeet_capi.h"
#include "silero_vad.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

namespace {

// Everything one synthetic GGUF needs. Tensors live in `ctx`.
struct Builder {
    gguf_context* g = gguf_init_empty();
    ggml_context* ctx = nullptr;
    std::string pre;   // component prefix ("" for a plain file)
    Builder() {
        ggml_init_params ip{64u << 20, nullptr, false};
        ctx = ggml_init(ip);
    }
    ~Builder() {
        gguf_free(g);
        ggml_free(ctx);
    }
    void str(const std::string& k, const std::string& v) { gguf_set_val_str(g, (pre + k).c_str(), v.c_str()); }
    void u32(const std::string& k, uint32_t v) { gguf_set_val_u32(g, (pre + k).c_str(), v); }
    void i32s(const std::string& k, const std::vector<int32_t>& v) {
        gguf_set_arr_data(g, (pre + k).c_str(), GGUF_TYPE_INT32, v.data(), v.size());
    }
    // F32 or F16 tensor with seeded random values in [-s, s].
    void tensor(const std::string& name, std::vector<int64_t> ne, std::mt19937& rng, float s, bool f16 = false) {
        ggml_tensor* t = ggml_new_tensor(ctx, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, (int)ne.size(), ne.data());
        ggml_set_name(t, (pre + name).c_str());
        std::uniform_real_distribution<float> d(-s, s);
        const int64_t n = ggml_nelements(t);
        for (int64_t i = 0; i < n; ++i) {
            const float v = d(rng);
            if (f16) ((ggml_fp16_t*)t->data)[i] = ggml_fp32_to_fp16(v);
            else ((float*)t->data)[i] = v;
        }
        gguf_add_tensor(g, t);
    }
    bool write(const std::string& path) { return gguf_write_to_file(g, path.c_str(), false); }
};

// The keys and tensors of a Silero VAD file, under `b.pre`.
void silero_content(Builder& b, uint32_t seed) {
    std::mt19937 rng(seed);
    b.str("general.architecture", "silero_vad");
    b.str("general.license", "MIT");
    b.u32("silero_vad.lstm.hidden", 128);
    b.u32("silero_vad.encoder.n_layers", 4);
    b.u32("silero_vad.encoder.kernel", 3);
    b.i32s("silero_vad.encoder.strides", {1, 2, 2, 1});
    b.i32s("silero_vad.encoder.channels", {128, 64, 64, 128});
    b.i32s("silero_vad.sample_rates", {16000, 8000});
    struct R { int sr; int chunk, ctx, n_fft, hop, rpad; const char* p; } rates[] = {
        {16000, 512, 64, 256, 128, 64, "vad16k."}, {8000, 256, 32, 128, 64, 32, "vad8k."}};
    for (const R& r : rates) {
        const std::string k = "silero_vad." + std::to_string(r.sr);
        b.u32(k + ".chunk_samples", r.chunk);
        b.u32(k + ".context_samples", r.ctx);
        b.u32(k + ".stft.n_fft", r.n_fft);
        b.u32(k + ".stft.hop", r.hop);
        b.u32(k + ".stft.right_reflect_pad", r.rpad);
        const int64_t bins = r.n_fft / 2 + 1;
        const int64_t in_ch[4] = {bins, 128, 64, 64}, out_ch[4] = {128, 64, 64, 128};
        const std::string p = r.p;
        b.tensor(p + "stft.forward_basis_buffer", {r.n_fft, 1, 2 * bins}, rng, 0.2f);
        for (int i = 0; i < 4; ++i) {
            b.tensor(p + "encoder." + std::to_string(i) + ".reparam_conv.weight", {3, in_ch[i], out_ch[i]}, rng, 0.2f, true);
            b.tensor(p + "encoder." + std::to_string(i) + ".reparam_conv.bias", {out_ch[i]}, rng, 0.1f);
        }
        b.tensor(p + "decoder.rnn.weight_ih", {128, 512}, rng, 0.2f, true);
        b.tensor(p + "decoder.rnn.weight_hh", {128, 512}, rng, 0.2f, true);
        b.tensor(p + "decoder.rnn.bias_ih", {512}, rng, 0.1f);
        b.tensor(p + "decoder.rnn.bias_hh", {512}, rng, 0.1f);
        b.tensor(p + "decoder.decoder.2.weight", {1, 128, 1}, rng, 0.2f);
        b.tensor(p + "decoder.decoder.2.bias", {1}, rng, 0.1f);
    }
}

// A stand-in ASR component: the keys ModelLoader needs and two tensors, one big.
void asr_content(Builder& b, uint32_t seed, int64_t big_floats) {
    std::mt19937 rng(seed);
    b.str("general.architecture", "parakeet");
    b.str("parakeet.arch", "tdt");
    b.u32("parakeet.encoder.d_model", 16);
    b.u32("parakeet.encoder.n_layers", 2);
    b.u32("parakeet.encoder.n_heads", 2);
    b.u32("parakeet.vocab_size", 5);
    b.u32("parakeet.blank_id", 5);
    b.tensor("encoder.small", {16, 4}, rng, 1.0f);
    b.tensor("encoder.big", {big_floats}, rng, 1.0f);
}

struct Comp {
    std::string name, kind;
    bool omit_tensors = false;
};

// Writes a bundle. `defs` gives each component's content and kind.
bool write_bundle(const std::string& path, const std::vector<Comp>& defs, uint32_t version = 1,
                  bool with_components_key = true, int64_t big_floats = 1 << 20) {
    Builder b;
    b.str("general.architecture", "parakeet-bundle");
    b.str("general.name", "test-bundle");
    b.u32("parakeet.bundle.version", version);
    std::vector<const char*> names;
    for (const Comp& c : defs) names.push_back(c.name.c_str());
    if (with_components_key) gguf_set_arr_str(b.g, "parakeet.bundle.components", names.data(), names.size());
    uint32_t seed = 1;
    for (const Comp& c : defs) {
        const std::string pk = "parakeet.bundle." + c.name + ".";
        b.str(pk + "kind", c.kind);
        b.str(pk + "license", c.kind == "vad" ? "MIT" : "CC-BY-4.0");
        b.str(pk + "license_url", "https://example.org/license");
        b.str(pk + "source", "example/" + c.name);
        b.str(pk + "attribution", "Credit " + c.name);
        b.str(pk + "changes", "converted");
        if (c.omit_tensors) continue;
        b.pre = c.name + ".";
        if (c.kind == "vad") silero_content(b, 7);
        else asr_content(b, seed++, big_floats);
        b.pre.clear();
    }
    return b.write(path);
}

unsigned long long bytes_read() {
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (!f) return 0;
    char k[64];
    unsigned long long v = 0, r = 0;
    while (std::fscanf(f, "%63s %llu", k, &v) == 2)
        if (std::string(k) == "rchar:") r = v;
    std::fclose(f);
    return r;
}

std::string take(char* p) {
    std::string s = p ? p : "";
    if (p) parakeet_capi_free_string(p);
    return s;
}

std::vector<float> noise(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-0.3f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

int main() {
    const std::string dir = "test_bundle.tmp.";
    const std::string plain_sil = dir + "silero.gguf", plain_asr = dir + "asr.gguf";
    const std::string B = dir + "b.gguf", B2 = dir + "b2.gguf", Bv = dir + "bv.gguf", Bbad = dir + "bad.gguf";

    // Single-model files, written with the same content (no prefix).
    {
        Builder s; silero_content(s, 7); CHECK(s.write(plain_sil));
        Builder a; asr_content(a, 1, 1 << 20); CHECK(a.write(plain_asr));
    }
    CHECK(write_bundle(B, {{"asr", "asr"}, {"vad", "vad"}}));
    CHECK(write_bundle(B2, {{"a", "asr"}, {"b", "asr"}, {"vad", "vad"}}));
    CHECK(write_bundle(Bv, {{"vad", "vad"}}));

    // --- header ---
    CHECK(pk::gguf_is_bundle(B));
    CHECK(!pk::gguf_is_bundle(plain_sil));
    CHECK(!pk::gguf_is_bundle(plain_asr));
    CHECK(!pk::gguf_is_bundle("test_bundle.does-not-exist"));
    pk::BundleInfo info;
    std::string err;
    CHECK(pk::read_bundle_info(B, info, &err));
    CHECK(info.version == 1 && info.name == "test-bundle" && info.components.size() == 2);
    if (info.components.size() == 2) {
        const pk::BundleComponent* a = info.find("asr");
        const pk::BundleComponent* v = info.find("vad");
        CHECK(a && v && !info.find("nope"));
        if (a && v) {
            CHECK(a->kind == "asr" && a->license == "CC-BY-4.0" && a->source == "example/asr");
            CHECK(v->kind == "vad" && v->license == "MIT" && v->attribution == "Credit vad");
            CHECK(a->n_tensors == 2 && v->n_tensors == 30);
            CHECK(a->n_bytes > (1u << 22));
        }
    }
    CHECK(!pk::read_bundle_info(plain_sil, info, &err) && err.find("not a bundle") != std::string::npos);

    // --- selection ---
    std::string pick;
    CHECK(pk::read_bundle_info(B, info, &err) && pk::select_default_component(info, pick, &err) && pick == "asr");
    CHECK(pk::read_bundle_info(Bv, info, &err) && pk::select_default_component(info, pick, &err) && pick == "vad");
    CHECK(pk::read_bundle_info(B2, info, &err));
    CHECK(!pk::select_default_component(info, pick, &err));
    CHECK(err.find("several asr") != std::string::npos && err.find("a (asr)") != std::string::npos &&
          err.find("b (asr)") != std::string::npos);

    // --- ModelLoader: a bundle is refused without a component, plain files unchanged ---
    {
        pk::ModelLoader ml;
        CHECK(!ml.load(B));
        pk::ModelLoader pl;
        CHECK(pl.load(plain_asr) && pl.config().d_model == 16 && pl.config().vocab_size == 5);
    }
    // --- ModelLoader::load_component: same config and tensor bytes as the plain file ---
    {
        pk::ModelLoader plain, bun;
        CHECK(plain.load(plain_asr));
        const unsigned long long r0 = bytes_read();
        CHECK(bun.load_component(B, "asr"));
        const unsigned long long r1 = bytes_read();
        CHECK(bun.config().d_model == 16 && bun.config().n_layers == 2 && bun.config().vocab_size == 5);
        CHECK(bun.config().arch == "tdt");
        ggml_tensor* x = plain.tensor("encoder.big");
        ggml_tensor* y = bun.tensor("encoder.big");
        CHECK(x && y && x->type == y->type && ggml_nbytes(x) == ggml_nbytes(y));
        CHECK(x && y && std::memcmp(x->data, y->data, ggml_nbytes(x)) == 0);
        ggml_tensor* x2 = plain.tensor("encoder.small");
        ggml_tensor* y2 = bun.tensor("encoder.small");
        CHECK(x2 && y2 && std::memcmp(x2->data, y2->data, ggml_nbytes(x2)) == 0);
        CHECK(!bun.tensor("vad16k.stft.forward_basis_buffer") && !bun.tensor("asr.encoder.big"));
        // Partial read: the Silero component's bytes (about 1.2 MB) stay on disk, and
        // the other ASR component (4 MB) too. Loading "asr" reads about its own 4 MB.
        if (r1 > r0) {
            CHECK(r1 - r0 < 4300000ull);
            std::fprintf(stderr, "bytes read for asr of B: %llu\n", r1 - r0);
        }
        pk::ModelLoader none;
        CHECK(!none.load_component(B, "nope"));
        pk::ModelLoader onplain;
        CHECK(!onplain.load_component(plain_asr, "asr"));
    }
    {   // the second ASR component of a two-ASR bundle is its own file
        pk::ModelLoader bb, pa;
        CHECK(bb.load_component(B2, "b"));
        CHECK(bb.config().d_model == 16);
        ggml_tensor* t = bb.tensor("encoder.big");
        CHECK(t != nullptr);
        Builder a2; asr_content(a2, 2, 1 << 20);  // component b was built with seed 2
        const std::string p2 = dir + "asr2.gguf";
        CHECK(a2.write(p2));
        pk::ModelLoader pl2;
        CHECK(pl2.load(p2));
        ggml_tensor* t2 = pl2.tensor("encoder.big");
        CHECK(t && t2 && std::memcmp(t->data, t2->data, ggml_nbytes(t)) == 0);
        std::remove(p2.c_str());
    }

    // --- Silero from a component equals Silero from the plain file ---
    {
        std::string e1, e2;
        auto plain = pk::SileroVad::load(plain_sil, &e1);
        const unsigned long long r0 = bytes_read();
        auto bun = pk::SileroVad::load(B, &e2, "vad");
        const unsigned long long r1 = bytes_read();
        CHECK(plain && bun);
        if (plain && bun) {
            for (int sr : {16000, 8000}) {
                const std::vector<float> pcm = noise((size_t)sr * 2 + 123, 99);
                const std::vector<float> a = plain->probabilities(pcm.data(), pcm.size(), sr);
                const std::vector<float> b = bun->probabilities(pcm.data(), pcm.size(), sr);
                CHECK(!a.empty() && a.size() == b.size());
                CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
            }
        }
        if (r1 > r0) {
            CHECK(r1 - r0 < 2500000ull);   // the 4 MB ASR component is not read
            std::fprintf(stderr, "bytes read for vad of B: %llu\n", r1 - r0);
        }
        std::string e;
        CHECK(!pk::SileroVad::load(B, &e) && e.find("bundle") != std::string::npos);
        CHECK(!pk::SileroVad::load(B, &e, "asr") && e.find("kind") != std::string::npos);
        CHECK(!pk::SileroVad::load(B, &e, "nope") && e.find("components:") != std::string::npos);
        CHECK(!pk::SileroVad::load(plain_sil, &e, "vad") && e.find("not a bundle") != std::string::npos);
    }

    // --- an old reader cannot mistake a bundle for a model ---
    {
        gguf_init_params ip{true, nullptr};
        gguf_context* g = gguf_init_from_file(B.c_str(), ip);
        CHECK(g != nullptr);
        if (g) {
            for (const char* k : {"parakeet.encoder.d_model", "parakeet.vocab_size", "parakeet.arch",
                                  "silero_vad.lstm.hidden", "silero_vad.sample_rates"})
                CHECK(gguf_find_key(g, k) < 0);
            const int64_t a = gguf_find_key(g, "general.architecture");
            CHECK(a >= 0 && std::string(gguf_get_val_str(g, a)) == "parakeet-bundle");
            gguf_free(g);
        }
    }

    // --- malformed bundles ---
    {
        CHECK(write_bundle(Bbad, {{"asr", "asr"}}, /*version*/ 2));
        CHECK(!pk::read_bundle_info(Bbad, info, &err) && err.find("version 2") != std::string::npos);
        pk::ModelLoader ml;
        CHECK(!ml.load_component(Bbad, "asr"));
        CHECK(write_bundle(Bbad, {{"asr", "asr"}}, 1, /*components key*/ false));
        CHECK(!pk::read_bundle_info(Bbad, info, &err) && err.find("components") != std::string::npos);
        CHECK(write_bundle(Bbad, {{"asr", "asr"}, {"asr", "asr", true}}));
        CHECK(!pk::read_bundle_info(Bbad, info, &err) && err.find("twice") != std::string::npos);
        Comp notensors{"asr", "asr", true};
        CHECK(write_bundle(Bbad, {notensors, {"vad", "vad"}}));
        CHECK(!pk::read_bundle_info(Bbad, info, &err) && err.find("no tensors") != std::string::npos);
        CHECK(write_bundle(Bbad, {{"Bad.Name", "asr"}}));
        CHECK(!pk::read_bundle_info(Bbad, info, &err) && err.find("not valid") != std::string::npos);
        CHECK(write_bundle(Bbad, {{"parakeet", "asr"}}));   // reserved prefix
        CHECK(!pk::read_bundle_info(Bbad, info, &err));
        // a file cut short: the component whose data is missing is refused, the
        // earlier component still loads
        CHECK(write_bundle(Bbad, {{"vad", "vad"}, {"asr", "asr"}}));
        std::string all;
        {
            std::ifstream in(Bbad, std::ios::binary);
            all.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        }
        {
            std::ofstream out(Bbad, std::ios::binary);
            out.write(all.data(), (std::streamsize)(all.size() - 100));
        }
        pk::ModelLoader t1, t2;
        std::string e;
        CHECK(!t1.load_component(Bbad, "asr"));
        CHECK(pk::SileroVad::load(Bbad, &e, "vad") != nullptr);
        CHECK(write_bundle(Bbad, {{"asr", "asr"}, {"vad", "vad"}}));
        {
            std::ifstream in(Bbad, std::ios::binary);
            all.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::ofstream out(Bbad, std::ios::binary);
            out.write(all.data(), (std::streamsize)(all.size() - 100));
        }
        CHECK(!pk::SileroVad::load(Bbad, &e, "vad") && e.find("truncated") != std::string::npos);
        {   // a file that is not a GGUF at all
            std::ofstream out(Bbad, std::ios::binary);
            out << "not a gguf";
        }
        CHECK(!pk::gguf_is_bundle(Bbad));
        CHECK(!pk::read_bundle_info(Bbad, info, &err));
    }

    // --- C-API ---
    {
        // vad-only bundle: parakeet_capi_load picks the Silero component.
        parakeet_ctx* c = parakeet_capi_load(Bv.c_str());
        CHECK(c && parakeet_capi_model_kind(c) == PARAKEET_MODEL_KIND_VAD);
        parakeet_ctx* p = parakeet_capi_load(plain_sil.c_str());
        CHECK(p && parakeet_capi_model_kind(p) == PARAKEET_MODEL_KIND_VAD);
        parakeet_ctx* d = parakeet_capi_load_component(B.c_str(), "vad");
        CHECK(d && parakeet_capi_model_kind(d) == PARAKEET_MODEL_KIND_VAD);
        if (c && p && d) {
            const std::vector<float> pcm = noise(16000 * 3, 5);
            const char* opt = "{\"probabilities\":true}";
            const std::string jp = take(parakeet_capi_vad_pcm_json(p, pcm.data(), (int)pcm.size(), 16000, opt));
            const std::string jc = take(parakeet_capi_vad_pcm_json(c, pcm.data(), (int)pcm.size(), 16000, opt));
            const std::string jd = take(parakeet_capi_vad_pcm_json(d, pcm.data(), (int)pcm.size(), 16000, opt));
            CHECK(!jp.empty() && jp.find("\"probabilities\"") != std::string::npos);
            CHECK(jp == jc && jp == jd);
        }
        parakeet_capi_free(c); parakeet_capi_free(p); parakeet_capi_free(d);

        // a bundle with an ASR component and a Silero one: load selects the ASR
        // component (the stand-in has the keys the loader needs, so it opens).
        parakeet_ctx* a = parakeet_capi_load(B.c_str());
        CHECK(a && parakeet_capi_model_kind(a) == PARAKEET_MODEL_KIND_ASR);
        parakeet_capi_free(a);
        // ambiguous bundle
        CHECK(parakeet_capi_load(B2.c_str()) == nullptr);
        const std::string amb = parakeet_capi_load_error();
        CHECK(amb.find("several asr") != std::string::npos && amb.find("parakeet_capi_load_component") != std::string::npos);
        // component errors
        CHECK(parakeet_capi_load_component(B.c_str(), "nope") == nullptr);
        CHECK(std::string(parakeet_capi_load_error()).find("no component \"nope\"") != std::string::npos);
        CHECK(parakeet_capi_load_component(plain_sil.c_str(), "vad") == nullptr);
        CHECK(std::string(parakeet_capi_load_error()).find("not a bundle") != std::string::npos);
        CHECK(parakeet_capi_load_component(nullptr, "vad") == nullptr);
        // NULL or empty component is a plain load
        parakeet_ctx* e1 = parakeet_capi_load_component(plain_sil.c_str(), nullptr);
        parakeet_ctx* e2 = parakeet_capi_load_component(Bv.c_str(), "");
        CHECK(e1 && e2);
        parakeet_capi_free(e1); parakeet_capi_free(e2);
        // load error is cleared by a good call and set by a failure
        CHECK(parakeet_capi_load("test_bundle.does-not-exist") == nullptr);
        CHECK(std::string(parakeet_capi_load_error()).size() > 0);

        // components JSON
        const std::string js = take(parakeet_capi_bundle_components_json(B.c_str()));
        CHECK(js.find("\"name\":\"asr\"") != std::string::npos && js.find("\"kind\":\"vad\"") != std::string::npos);
        CHECK(js.find("\"license\":\"MIT\"") != std::string::npos && js.find("\"license\":\"CC-BY-4.0\"") != std::string::npos);
        CHECK(js.front() == '[' && js.back() == ']');
        CHECK(parakeet_capi_bundle_components_json(plain_sil.c_str()) == nullptr);
        CHECK(std::string(parakeet_capi_load_error()).find("not a bundle") != std::string::npos);
        CHECK(parakeet_capi_bundle_components_json(nullptr) == nullptr);
    }

    for (const std::string& f : {plain_sil, plain_asr, B, B2, Bv, Bbad}) std::remove(f.c_str());
    if (failures) return 1;
    std::printf("test_bundle OK\n");
    return 0;
}
