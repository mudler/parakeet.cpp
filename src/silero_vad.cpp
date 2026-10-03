#include "silero_vad.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <initializer_list>

#include "backend.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "bundle.hpp"
#include "ggml_graph.hpp"
#include "gguf.h"

namespace pk {

namespace {

void set_err(std::string* err, const std::string& m) {
    if (err) *err = m;
}

// conv1d over x [L, IC, 1] with kernel [K, IC, OC] (PyTorch weight (OC, IC, K)),
// zero padding p on both sides. Returns [OL, OC, 1]. F32 im2col, so the result
// does not lose precision (ggml_conv_1d would force an F16 im2col).
ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* x, int stride, int pad,
                    ggml_tensor* bias) {
    ggml_tensor* ic = ggml_im2col(ctx, kernel, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);  // [IC*K, OL, 1]
    ggml_tensor* y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, ic, ic->ne[0], ic->ne[1] * ic->ne[2]),
                                  ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    if (bias) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
    return ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], 1);
}

}  // namespace

namespace silero {

const RateParams* rate_params(int sample_rate) {
    static const RateParams k16{16000, 512, 64, 64, 256, 128};
    static const RateParams k8{8000, 256, 32, 32, 128, 64};
    if (sample_rate == 16000) return &k16;
    if (sample_rate == 8000) return &k8;
    return nullptr;
}

void reflect_pad_right(float* x, int base, int rpad) {
    for (int i = 0; i < rpad; ++i) x[base + i] = x[base - 2 - i];
}

Framer::Framer(const RateParams& p) : p_(p) { reset(); }

void Framer::reset() {
    ctx_.assign((size_t)p_.context, 0.0f);
    buf_.clear();
    off_ = 0;
}

void Framer::push(const float* pcm, size_t n) {
    if (!n) return;
    // Drop what was consumed before the buffer grows.
    if (off_ && off_ >= buf_.size() / 2) {
        buf_.erase(buf_.begin(), buf_.begin() + (std::ptrdiff_t)off_);
        off_ = 0;
    }
    buf_.insert(buf_.end(), pcm, pcm + n);
}

void Framer::emit(const float* chunk, float* input) {
    std::copy(ctx_.begin(), ctx_.end(), input);
    std::copy(chunk, chunk + p_.chunk, input + p_.context);
    reflect_pad_right(input, p_.context + p_.chunk, p_.rpad);
    // The next context is the tail of context | chunk.
    std::copy(input + p_.chunk, input + p_.chunk + p_.context, ctx_.begin());
}

bool Framer::next(float* input) {
    if (pending() < (size_t)p_.chunk) return false;
    emit(buf_.data() + off_, input);
    off_ += (size_t)p_.chunk;
    if (off_ == buf_.size()) buf_.clear(), off_ = 0;
    return true;
}

bool Framer::next_padded(float* input) {
    const size_t n = pending();
    if (n == 0) return false;
    if (n >= (size_t)p_.chunk) return next(input);
    std::vector<float> tmp((size_t)p_.chunk, 0.0f);
    std::copy(buf_.begin() + (std::ptrdiff_t)off_, buf_.end(), tmp.begin());
    emit(tmp.data(), input);
    buf_.clear();
    off_ = 0;
    return true;
}

}  // namespace silero

SileroVad::Stream::Stream(const SileroVad* m, const silero::RateParams* p)
    : model_(m), sr_(p->sample_rate), framer_(*p) {
    in_.resize((size_t)p->input_len());
    reset();
}

void SileroVad::Stream::reset() {
    if (!model_) return;
    framer_.reset();
    failed_ = false;
    h_.assign(128, 0.0f);
    c_.assign(128, 0.0f);
}

bool SileroVad::Stream::run(const float* input, std::vector<float>* probs) {
    const RateWeights* w = model_->find(sr_);
    float p = -1.0f;
    if (!w || !model_->run_chunk(*w, input, h_, c_, &p)) {
        failed_ = true;  // state is now unreliable
        return false;
    }
    if (probs) probs->push_back(p);
    return true;
}

bool SileroVad::Stream::process_chunk(const float* pcm, size_t n, std::vector<float>* probs) {
    if (!valid()) return false;
    framer_.push(pcm, n);
    while (framer_.next(in_.data()))
        if (!run(in_.data(), probs)) return false;
    return true;
}

bool SileroVad::Stream::flush(std::vector<float>* probs) {
    if (!valid()) return false;
    return !framer_.next_padded(in_.data()) || run(in_.data(), probs);
}

SileroVad::~SileroVad() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

const SileroVad::RateWeights* SileroVad::find(int sr) const {
    for (const auto& w : w_)
        if (w.p.sample_rate == sr) return &w;
    return nullptr;
}

bool SileroVad::supports(int sr) const { return find(sr) != nullptr; }
int SileroVad::chunk_samples(int sr) const {
    const RateWeights* w = find(sr);
    return w ? w->p.chunk : 0;
}
double SileroVad::chunk_sec(int sr) const {
    const RateWeights* w = find(sr);
    return w ? (double)w->p.chunk / (double)sr : 0.0;
}

SileroVad::Stream SileroVad::new_stream(int sample_rate) const {
    const RateWeights* w = find(sample_rate);
    return w ? Stream(this, &w->p) : Stream();
}

namespace {

bool shape_is(const ggml_tensor* t, std::initializer_list<int64_t> ne) {
    int d = 0;
    for (int64_t v : ne) {
        if (t->ne[d++] != v) return false;
    }
    for (; d < GGML_MAX_DIMS; ++d)
        if (t->ne[d] != 1) return false;
    return true;
}

std::string shape_str(const ggml_tensor* t) {
    std::string s = "[";
    for (int d = 0; d < GGML_MAX_DIMS && (d < 2 || t->ne[d] != 1); ++d) s += (d ? "," : "") + std::to_string(t->ne[d]);
    return s + "]";
}

}  // namespace

std::unique_ptr<SileroVad> SileroVad::load(const std::string& path, std::string* err, const std::string& component) {
    // A component of a bundle GGUF (docs/bundle.md): every key and tensor name has
    // the "<component>." prefix, and only those tensors are read.
    std::string pfx;
    if (!component.empty()) {
        BundleInfo info;
        std::string berr;
        if (!read_bundle_info(path, info, &berr)) {
            set_err(err, berr);
            return nullptr;
        }
        const BundleComponent* bc = info.find(component);
        if (!bc) {
            set_err(err, path + ": no component \"" + component + "\"; components: " + bundle_component_names(info));
            return nullptr;
        }
        if (bc->kind != kBundleKindVad) {
            set_err(err, path + ": component \"" + component + "\" has kind \"" + bc->kind + "\", not \"" +
                             kBundleKindVad + "\"");
            return nullptr;
        }
        pfx = component + ".";
    }
    ggml_context* meta = nullptr;
    gguf_init_params gp{/*no_alloc*/ true, /*ctx*/ &meta};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        set_err(err, "cannot read " + path + " as a GGUF file");
        return nullptr;
    }
    struct Guard {
        gguf_context* g;
        ggml_context* m;
        ~Guard() {
            if (g) gguf_free(g);
            if (m) ggml_free(m);
        }
    } guard{g, meta};
    auto fail = [&](const std::string& m) -> std::unique_ptr<SileroVad> {
        set_err(err, path + ": " + m);
        return nullptr;
    };

    // Typed metadata reads: a key of the wrong type is an error, not an abort.
    auto u32 = [&](const std::string& k, uint32_t* out) {
        const int64_t id = gguf_find_key(g, (pfx + k).c_str());
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return false;
        *out = gguf_get_val_u32(g, id);
        return true;
    };
    auto i32_array = [&](const std::string& k, std::vector<int32_t>* out) {
        const int64_t id = gguf_find_key(g, (pfx + k).c_str());
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_INT32)
            return false;
        const int32_t* d = (const int32_t*)gguf_get_arr_data(g, id);
        out->assign(d, d + gguf_get_arr_n(g, id));
        return true;
    };

    if (pfx.empty() && gguf_find_key(g, "general.architecture") >= 0 &&
        gguf_get_kv_type(g, gguf_find_key(g, "general.architecture")) == GGUF_TYPE_STRING &&
        std::string(gguf_get_val_str(g, gguf_find_key(g, "general.architecture"))) == kBundleArch)
        return fail("this is a bundle GGUF; load its Silero component by name");
    const int64_t arch = gguf_find_key(g, (pfx + "general.architecture").c_str());
    if (arch < 0 || gguf_get_kv_type(g, arch) != GGUF_TYPE_STRING)
        return fail("no general.architecture; this is not a Silero VAD GGUF");
    if (std::string(gguf_get_val_str(g, arch)) != "silero_vad")
        return fail(std::string("general.architecture is \"") + gguf_get_val_str(g, arch) +
                    "\", expected \"silero_vad\"");

    uint32_t hidden = 0, layers = 0, kernel = 0;
    std::vector<int32_t> strides, channels, rates;
    if (!u32("silero_vad.lstm.hidden", &hidden) || hidden != 128)
        return fail("silero_vad.lstm.hidden is missing or not 128");
    if (!u32("silero_vad.encoder.n_layers", &layers) || layers != 4 || !u32("silero_vad.encoder.kernel", &kernel) ||
        kernel != 3 || !i32_array("silero_vad.encoder.strides", &strides) ||
        strides != std::vector<int32_t>{1, 2, 2, 1} || !i32_array("silero_vad.encoder.channels", &channels) ||
        channels != std::vector<int32_t>{128, 64, 64, 128})
        return fail("encoder metadata is missing or differs from the supported 4-layer Silero encoder");
    if (!i32_array("silero_vad.sample_rates", &rates) || rates.empty())
        return fail("silero_vad.sample_rates is missing or not an int32 array");

    // Keep every tensor as F32 in a backend buffer (F16 files are widened here).
    const int64_t nt = gguf_get_n_tensors(g);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(nt + 4), nullptr, /*no_alloc*/ true};
    std::unique_ptr<SileroVad> m(new SileroVad());
    m->ctx_ = ggml_init(ip);
    if (!m->ctx_) return fail("ggml_init failed");
    std::vector<std::pair<ggml_tensor*, int64_t>> todo;  // dst, gguf tensor id
    for (int64_t i = 0; i < nt; ++i) {
        const char* full = gguf_get_tensor_name(g, i);
        if (std::strncmp(full, pfx.c_str(), pfx.size()) != 0) continue;  // another component
        const char* name = full + pfx.size();
        ggml_tensor* src = ggml_get_tensor(meta, full);
        if (!src) return fail(std::string("tensor ") + name + " has no metadata");
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)
            return fail(std::string("tensor ") + name + " has type " + ggml_type_name(src->type) +
                        "; only F32 and F16 are supported");
        ggml_tensor* dst = ggml_new_tensor(m->ctx_, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
        ggml_set_name(dst, name);
        todo.emplace_back(dst, i);
    }

    // Resolve and check every tensor before any weight memory is allocated.
    for (int32_t sr : rates) {
        const silero::RateParams* rp = silero::rate_params(sr);
        if (!rp) return fail("unsupported sample rate " + std::to_string(sr) + " in silero_vad.sample_rates");
        if (m->find(sr)) return fail("sample rate " + std::to_string(sr) + " listed twice");
        const std::string key = "silero_vad." + std::to_string(sr);
        const std::string rate_s = std::to_string(sr) + " Hz";
        const struct {
            const char* name;
            int want;
        } hp[] = {{".chunk_samples", rp->chunk},      {".context_samples", rp->context},
                  {".stft.n_fft", rp->n_fft},         {".stft.hop", rp->hop},
                  {".stft.right_reflect_pad", rp->rpad}};
        for (const auto& e : hp) {
            uint32_t v = 0;
            if (!u32(key + e.name, &v)) return fail("missing " + key + e.name);
            if ((int)v != e.want)
                return fail(key + e.name + " is " + std::to_string(v) + ", expected " + std::to_string(e.want));
        }
        const std::string pre = sr == 16000 ? "vad16k." : "vad8k.";
        RateWeights w;
        w.p = *rp;
        std::string bad;
        // Gets a tensor and checks its ggml shape (PyTorch shape reversed).
        auto get = [&](const std::string& name, std::initializer_list<int64_t> ne) -> ggml_tensor* {
            ggml_tensor* t = ggml_get_tensor(m->ctx_, (pre + name).c_str());
            if (!t) {
                if (bad.empty()) bad = "missing tensor " + pre + name;
            } else if (!shape_is(t, ne)) {
                if (bad.empty()) bad = "tensor " + pre + name + " has shape " + shape_str(t) + " (ggml order)";
                return nullptr;
            }
            return t;
        };
        const int64_t bins = rp->n_fft / 2 + 1;
        const int64_t in_ch[4] = {bins, 128, 64, 64}, out_ch[4] = {128, 64, 64, 128};
        w.stft = get("stft.forward_basis_buffer", {rp->n_fft, 1, 2 * bins});
        for (int i = 0; i < 4; ++i) {
            const std::string e = "encoder." + std::to_string(i) + ".reparam_conv.";
            w.enc_w[i] = get(e + "weight", {3, in_ch[i], out_ch[i]});
            w.enc_b[i] = get(e + "bias", {out_ch[i]});
        }
        w.w_ih = get("decoder.rnn.weight_ih", {128, 512});
        w.w_hh = get("decoder.rnn.weight_hh", {128, 512});
        w.b_ih = get("decoder.rnn.bias_ih", {512});
        w.b_hh = get("decoder.rnn.bias_hh", {512});
        w.w_out = get("decoder.decoder.2.weight", {1, 128, 1});
        w.b_out = get("decoder.decoder.2.bias", {1});
        if (!bad.empty()) return fail(bad + " for " + rate_s);
        m->rates_.push_back(sr);
        m->w_.push_back(w);
    }

    m->buf_ = ggml_backend_alloc_ctx_tensors(m->ctx_, global_backend().handle());
    if (!m->buf_) return fail("cannot allocate the weight buffer");
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail("cannot reopen the file to read the weights");
    const uint64_t base = gguf_get_data_offset(g);
    std::vector<uint8_t> raw;
    std::vector<float> wide;
    for (auto& pr : todo) {
        ggml_tensor* src = ggml_get_tensor(meta, (pfx + pr.first->name).c_str());
        const size_t n = (size_t)ggml_nelements(src);
        raw.resize(ggml_nbytes(src));
        f.seekg((std::streamoff)(base + gguf_get_tensor_offset(g, pr.second)));
        if (!f.read((char*)raw.data(), (std::streamsize)raw.size()))
            return fail(std::string("file is truncated (tensor ") + pr.first->name + ")");
        if (src->type == GGML_TYPE_F16) {
            wide.resize(n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)raw.data(), wide.data(), (int64_t)n);
            ggml_backend_tensor_set(pr.first, wide.data(), 0, n * sizeof(float));
        } else {
            ggml_backend_tensor_set(pr.first, raw.data(), 0, raw.size());
        }
    }
    return m;
}

bool SileroVad::run_chunk(const RateWeights& w, const float* input, std::vector<float>& h, std::vector<float>& c,
                          float* prob) const {
    const int L = w.p.input_len();
    const int H = 128;
    std::vector<float> h_new, c_new, out;
    const bool ok = run_graph(0, 1, [&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne_x[3] = {L, 1, 1};
        ggml_tensor* in = graph_input_tensor(ctx, GGML_TYPE_F32, 3, ne_x, input, (size_t)L * sizeof(float));
        const int64_t ne_h[1] = {H};
        ggml_tensor* hh = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne_h, h.data(), H * sizeof(float));
        ggml_tensor* cc = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne_h, c.data(), H * sizeof(float));

        // STFT as a strided conv with the windowed DFT basis: first half real, second imaginary.
        ggml_tensor* st = conv1d(ctx, w.stft, in, w.p.hop, 0, nullptr);  // [T, 2*bins, 1]
        const int64_t T = st->ne[0], bins = st->ne[1] / 2;
        ggml_tensor* re = ggml_view_2d(ctx, st, T, bins, st->nb[1], 0);
        ggml_tensor* im = ggml_view_2d(ctx, st, T, bins, st->nb[1], (size_t)bins * st->nb[1]);
        ggml_tensor* mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_mul(ctx, re, re), ggml_mul(ctx, im, im)));
        ggml_tensor* cur = ggml_reshape_3d(ctx, mag, T, bins, 1);

        static const int kStride[4] = {1, 2, 2, 1};
        for (int i = 0; i < 4; ++i) cur = ggml_relu(ctx, conv1d(ctx, w.enc_w[i], cur, kStride[i], 1, w.enc_b[i]));
        // The time length is 1 here: [1, 128, 1] is the LSTM input.
        ggml_tensor* xt = ggml_reshape_2d(ctx, cur, H, 1);
        ggml_tensor* g = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.w_ih, xt), ggml_reshape_2d(ctx, w.b_ih, 512, 1)),
                                  ggml_add(ctx, ggml_mul_mat(ctx, w.w_hh, ggml_reshape_2d(ctx, hh, H, 1)),
                                           ggml_reshape_2d(ctx, w.b_hh, 512, 1)));
        auto gate = [&](int k) { return ggml_view_1d(ctx, g, H, (size_t)k * H * sizeof(float)); };  // i, f, g, o
        ggml_tensor* c2 = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), cc),
                                   ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
        ggml_tensor* h2 = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, c2));
        capture_graph_output(h2, &h_new);
        capture_graph_output(c2, &c_new);
        ggml_tensor* z = ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w.w_out, H, 1), ggml_reshape_2d(ctx, ggml_relu(ctx, h2), H, 1)),
                                  ggml_reshape_2d(ctx, w.b_out, 1, 1));
        return ggml_sigmoid(ctx, z);
    }, out);
    if (!ok || out.size() != 1 || h_new.size() != (size_t)H || c_new.size() != (size_t)H) return false;
    h = std::move(h_new);
    c = std::move(c_new);
    *prob = out[0];
    return true;
}

std::vector<float> SileroVad::probabilities(const float* pcm, size_t n, int sample_rate) const {
    std::vector<float> p;
    Stream s = new_stream(sample_rate);
    if (!s.valid()) return p;
    p.reserve(n / (size_t)chunk_samples(sample_rate) + 1);
    if (!s.process_chunk(pcm, n, &p) || !s.flush(&p)) return {};
    return p;
}

}  // namespace pk
