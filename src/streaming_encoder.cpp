#include "streaming_encoder.hpp"
#include "subsampling.hpp"
#include "pos_enc.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "graph_builder.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pk {

// Weights from the loader are brought into the graph as zero-copy leaves via the
// shared pk::clone_weight / pk::clone_weight_opt (backend.cpp): allowlisted
// linears may be f16/q8_0 and are dequantized by ggml_mul_mat. std::string-name
// overloads keep the existing call sites unchanged.
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const std::string& name) {
    return pk::clone_weight(ctx, ml, name.c_str());
}
static ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                                     const std::string& name) {
    return pk::clone_weight_opt(ctx, ml, name.c_str());
}

StreamingEncoder::StreamingEncoder(const ModelLoader& ml) : ml_(ml) {
    if (ml.config().ternary.present || ml.has_tensor_with_suffix(".qweight"))
        throw std::runtime_error("streaming does not support packed ternary GGUFs; convert with --ternary dequant");
    const ParakeetConfig& c = ml.config();
    d_model_     = (int)c.d_model;
    n_layers_    = (int)c.n_layers;
    n_heads_     = (int)c.n_heads;
    conv_kernel_ = (int)c.conv_kernel;
    xscaling_    = c.xscaling;
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    d_head_ = d_model_ / n_heads_;
    left_pad_ = conv_kernel_ - 1;   // causal depthwise conv left-context (8)
    assert(c.streaming.present && "StreamingEncoder requires a streaming model");
    const StreamingCfg& s = c.streaming;
    // chunk_size/shift_size/pre_encode_cache_size are stored as the [a,b] arrays;
    // the cache-aware streaming buffer uses the [0] value for the FIRST chunk and
    // the [1] value thereafter (CacheAwareStreamingAudioBuffer.__iter__).
    auto pick = [](const std::vector<int32_t>& v, int idx, int dflt) {
        return (idx < (int)v.size()) ? (int)v[idx] : dflt;
    };
    chunk_first_       = pick(s.chunk_size, 0, 9);
    chunk_main_        = pick(s.chunk_size, 1, 16);
    pre_cache_         = pick(s.pre_encode_cache_size, 1, 9);
    drop_extra_        = s.drop_extra_pre_encoded;
    last_channel_cache_= s.last_channel_cache_size;
    valid_out_len_     = s.valid_out_len;
    att_left_  = c.att_context_left;
    att_right_ = c.att_context_right;
    assert(c.causal_downsampling && "streaming model expects causal subsampling");
    assert(c.conv_causal && "streaming model expects causal depthwise conv");
    assert(c.att_context_style == "chunked_limited");
    reset();
}

void StreamingEncoder::reset() {
    step_ = 0;
    clc_len_ = 0;
    cache_time_.assign(n_layers_,
                       std::vector<float>((size_t)left_pad_ * d_model_, 0.0f));
    cache_channel_.assign(n_layers_,
                          std::vector<float>((size_t)last_channel_cache_ * d_model_, 0.0f));
}

namespace {
// Per-layer streaming graph I/O: the cache reads are graph INPUTS (host data set
// after alloc by graph_input_tensor) and the next-step caches are graph OUTPUTS
// (captured via capture_graph_output and read back after compute). This moves the
// NeMo cache_last_time / cache_last_channel update IN-GRAPH (vs the old per-stage
// host shuffle), so the whole chunk forward is ONE ggml graph.
struct StreamLayerCapture {
    std::vector<float> next_channel;  // updated cache_last_channel [cache_len, D]
    std::vector<float> next_time;     // updated cache_last_time    [LP, D]
};
} // namespace

// Build ONE streaming conformer layer (NeMo ConformerLayer::forward) into the
// SHARED graph `ctx`, threading the per-layer caches in-graph:
//   * MHSA prepends the [cache_len, D] attention K/V cache (update_cache) and uses
//     the streaming chunked-limited + empty-cache mask over [cache_len+Tc] keys.
//   * Conv prepends the [D, LP] depthwise-conv cache before the depthwise conv.
//   * The NEXT caches are emitted as captured outputs (cap->next_channel/_time):
//       next_channel = cat([cache_channel[Tc:], attn_in[:Tc]])  (keep last cache_len)
//       next_time    = last LP cols of [cache_time(LP) ; glu(Tc)]
// `xt` is the chunk input [D, Tc]; `pos_heads` is this layer's linear_pos(pos_emb)
// split into heads, host [dk, pos_len, H] from StreamingEncoder::pos_heads, for
// pos_len = 2*(Tc+cache_len)-1. `cache_ch_t` is the K/V cache input [D, cache_len];
// `cache_t_t` is the conv cache input [D, LP]. Returns the layer output [D, Tc].
//
// This MIRRORS the proven (bit-equivalent, max|d| ~7e-6) per-stage layer_step op
// sequence exactly — same ggml ops in the same order — only fused into one graph
// with the cache reads/updates moved in-graph, so the numerics are preserved.
static ggml_tensor* build_stream_layer(
        ggml_context* ctx, const ModelLoader& ml, int layer_idx,
        ggml_tensor* xt, int Tc, const float* pos_heads, int pos_len,
        int cache_len, int clc_len, int n_heads, int d_model, int conv_kernel,
        int att_left, int att_right, const std::string& conv_norm_type,
        ggml_tensor* cache_ch_t, ggml_tensor* cache_t_t,
        GraphInputPool& pool, StreamLayerCapture* cap) {
    const int D  = d_model;
    const int H  = n_heads;
    const int dk = D / H;
    const int K  = conv_kernel;
    const int LP = K - 1;
    const float ln_eps = 1e-5f;
    const float scale = 1.0f / std::sqrt((float)dk);
    const int Tk = cache_len + Tc;            // attention key length
    assert(pos_len == 2 * (Tc + cache_len) - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx) + ".";

    auto layer_norm = [&](ggml_tensor* in, const std::string& nm) {
        ggml_tensor* g = clone_weight(ctx, ml, pre + nm + ".weight");
        ggml_tensor* b = clone_weight(ctx, ml, pre + nm + ".bias");
        ggml_tensor* y = ggml_norm(ctx, in, ln_eps);
        y = ggml_mul(ctx, y, g);
        y = ggml_add(ctx, y, b);
        return y;
    };
    auto linear = [&](ggml_tensor* in, const std::string& nm, bool bias) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + nm + ".weight");
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bias) { ggml_tensor* B = clone_weight_opt(ctx, ml, pre + nm + ".bias");
                    if (B) y = ggml_add(ctx, y, B); }
        return y;
    };
    auto feed_forward = [&](ggml_tensor* in, const std::string& ff) {
        ggml_tensor* h = linear(in, ff + ".linear1", true);
        h = ggml_silu(ctx, h);
        h = linear(h, ff + ".linear2", true);
        return h;
    };

    // === Stage A: r = x + 0.5*FFN1(norm_ff1(x)); attn_in = norm_self_att(r). ===
    ggml_tensor* h1 = feed_forward(layer_norm(xt, "norm_feed_forward1"),
                                   "feed_forward1");
    h1 = ggml_scale(ctx, h1, 0.5f);
    ggml_tensor* r = ggml_add(ctx, xt, h1);          // [D, Tc]
    ggml_tensor* attn_in = layer_norm(r, "norm_self_att"); // [D, Tc]

    // === Stage B: streaming MHSA (RelPosAttention with attention cache). ===
    // K/V = cat([cache_channel(cache_len), attn_in(Tc)]); Q = attn_in(Tc).
    ggml_tensor* attn_out;
    {
        const std::string ap = pre + "self_attn.";
        ggml_tensor* kvt = ggml_concat(ctx, cache_ch_t, attn_in, /*dim=*/1); // [D, Tk]

        auto lin = [&](const char* w, const char* b, ggml_tensor* in) {
            ggml_tensor* W = clone_weight(ctx, ml, ap + w);
            ggml_tensor* y = ggml_mul_mat(ctx, W, in);
            if (b && ml.tensor(ap + b)) y = ggml_add(ctx, y, clone_weight(ctx, ml, ap + b));
            return y;
        };
        ggml_tensor* q = lin("linear_q.weight", "linear_q.bias", attn_in); // [D, Tc]
        ggml_tensor* k = lin("linear_k.weight", "linear_k.bias", kvt);     // [D, Tk]
        ggml_tensor* v = lin("linear_v.weight", "linear_v.bias", kvt);     // [D, Tk]

        auto to_heads = [&](ggml_tensor* t, int n) {
            t = ggml_reshape_3d(ctx, t, dk, H, n);
            return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H]
        };
        ggml_tensor* qh = to_heads(q, Tc);
        ggml_tensor* kh = to_heads(k, Tk);
        // linear_pos(pos_emb) split into heads, precomputed per window length.
        int64_t ph_ne[3] = {dk, pos_len, H};
        ggml_tensor* ph = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ph_ne, pos_heads,
                                                 (size_t)dk * pos_len * H * sizeof(float)); // [dk, P, H]

        ggml_tensor* bu = clone_weight(ctx, ml, ap + "pos_bias_u");
        ggml_tensor* bv = clone_weight(ctx, ml, ap + "pos_bias_v");
        bu = ggml_reshape_3d(ctx, bu, dk, 1, H);
        bv = ggml_reshape_3d(ctx, bv, dk, 1, H);
        ggml_tensor* qu = ggml_add(ctx, qh, bu); // [dk, Tc, H]
        ggml_tensor* qv = ggml_add(ctx, qh, bv); // [dk, Tc, H]

        // ac = q_u @ k^T -> [Tk, Tc, H].
        ggml_tensor* ac = ggml_mul_mat(ctx, kh, qu);

        // bd = q_v @ p^T -> [P, Tc, H]; rel_shift; slice first Tk pos cols.
        ggml_tensor* bd = ggml_mul_mat(ctx, ph, qv); // [P, Tc, H]
        bd = ggml_pad_ext(ctx, bd, 1, 0, 0, 0, 0, 0, 0, 0);      // [P+1, Tc, H]
        bd = ggml_reshape_3d(ctx, bd, Tc, pos_len + 1, H);        // [Tc, P+1, H]
        bd = ggml_view_3d(ctx, bd, Tc, pos_len, H,
                          bd->nb[1], bd->nb[2], bd->nb[1]);       // drop first row
        bd = ggml_cont(ctx, bd);
        bd = ggml_reshape_3d(ctx, bd, pos_len, Tc, H);            // [P, Tc, H]
        bd = ggml_view_3d(ctx, bd, Tk, Tc, H, bd->nb[1], bd->nb[2], 0); // first Tk pos
        bd = ggml_cont(ctx, bd);

        ggml_tensor* scores = ggml_add(ctx, ac, bd); // [Tk, Tc, H]

        // Streaming attention mask [Tk(key), Tc(query)] (0 visible / -inf):
        //   global key index   gk = kj            in [0, Tk)
        //   global query index gq = cache_len + qi in [cache_len, cache_len+Tc)
        //   (1) empty-cache mask: cache cols [0, cache_len-clc_len) are not yet
        //       filled -> masked.
        //   (2) chunked_limited (chunk=att_right+1, left=att_left/chunk):
        //       visible iff 0 <= gq/chunk - gk/chunk <= left_chunks.
        const int chunk = att_right + 1;
        const int left_chunks = (chunk > 0) ? (att_left / chunk) : 0;
        const int empty_cache = cache_len - clc_len;
        std::vector<float>& mask_host = pool.alloc_f32((size_t)Tk * Tc);
        {
            float* md = mask_host.data();
            const float ninf = -INFINITY;
            for (int qi = 0; qi < Tc; ++qi) {
                const int gq = cache_len + qi;
                const int cq = gq / chunk;
                for (int kj = 0; kj < Tk; ++kj) {
                    bool vis = (kj >= empty_cache);
                    if (vis) {
                        const int ck = kj / chunk;
                        const int diff = cq - ck;
                        vis = (diff >= 0 && diff <= left_chunks);
                    }
                    md[(size_t)qi * Tk + kj] = vis ? 0.0f : ninf;
                }
            }
        }
        int64_t mask_ne[2] = {Tk, Tc};
        ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                mask_ne, mask_host.data(),
                                mask_host.size() * sizeof(float));
        ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

        ggml_tensor* vh = to_heads(v, Tk);           // [dk, Tk, H]
        ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [Tk, dk, H]
        ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn); // [dk, Tc, H]
        ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3));
        merged = ggml_reshape_2d(ctx, merged, D, Tc);
        attn_out = lin("linear_out.weight", "linear_out.bias", merged); // [D, Tc]
    }
    r = ggml_add(ctx, r, attn_out);   // residual after MHSA -> norm_conv -> conv

    // Update attention cache (NeMo update_cache, cache_drop_size=0):
    //   next = cat([cache[Tc:], attn_in[:Tc]]) keep last cache_len.
    // In-graph: drop the oldest Tc cache cols (view starting at col Tc), then
    // concat the Tc current frames; capture for readback into cache_channel_.
    if (cap) {
        const int keep = cache_len - Tc;
        ggml_tensor* next_ch;
        if (keep > 0) {
            ggml_tensor* kept = ggml_view_2d(ctx, cache_ch_t, D, keep,
                                    cache_ch_t->nb[1], (size_t)Tc * cache_ch_t->nb[1]);
            next_ch = ggml_concat(ctx, kept, attn_in, /*dim=*/1); // [D, cache_len]
        } else {
            // Tc >= cache_len: keep only the last cache_len cols of attn_in.
            next_ch = ggml_view_2d(ctx, attn_in, D, cache_len, attn_in->nb[1],
                                   (size_t)(Tc - cache_len) * attn_in->nb[1]);
        }
        next_ch = ggml_cont(ctx, next_ch);
        pk::capture_graph_output(next_ch, &cap->next_channel);
    }

    // === Stage C: streaming conv module (causal depthwise + conv cache). ===
    ggml_tensor* c = layer_norm(r, "norm_conv"); // [D, Tc]
    ggml_tensor* pw1w = clone_weight(ctx, ml, pre + "conv.pointwise_conv1.weight");
    pw1w = ggml_reshape_2d(ctx, pw1w, D, 2 * D);
    ggml_tensor* pw1b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv1.bias");
    ggml_tensor* y = ggml_mul_mat(ctx, pw1w, c); // [2D, Tc]
    if (pw1b) y = ggml_add(ctx, y, pw1b);
    ggml_tensor* a = ggml_view_2d(ctx, y, D, Tc, y->nb[1], 0);
    ggml_tensor* b = ggml_view_2d(ctx, y, D, Tc, y->nb[1], (size_t)D * y->nb[0]);
    ggml_tensor* glu = ggml_cont(ctx, ggml_mul(ctx, ggml_cont(ctx, a),
                        ggml_sigmoid(ctx, ggml_cont(ctx, b)))); // [D, Tc]

    // dw_in [D, LP+Tc] = [cache_time(LP) ; glu(Tc)].
    ggml_tensor* dw_in = ggml_concat(ctx, cache_t_t, glu, /*dim=*/1); // [D, LP+Tc]

    // Update conv cache: last LP cols of [cache_time(LP) ; glu(Tc)] (NeMo
    // new_x[:, :, -LP:]); capture for readback into cache_time_.
    if (cap) {
        ggml_tensor* next_t = ggml_view_2d(ctx, dw_in, D, LP, dw_in->nb[1],
                                  (size_t)Tc * dw_in->nb[1]);  // last LP cols
        next_t = ggml_cont(ctx, next_t);
        pk::capture_graph_output(next_t, &cap->next_time);
    }

    ggml_tensor* glu_tc = ggml_cont(ctx, ggml_transpose(ctx, dw_in)); // [LP+Tc, D]
    ggml_tensor* dww = clone_weight(ctx, ml, pre + "conv.depthwise_conv.weight"); // [K,1,C]
    ggml_tensor* dw;
    {
        ggml_tensor* nb = ggml_reshape_4d(ctx, glu_tc, glu_tc->ne[0], 1, glu_tc->ne[1], 1);
        ggml_tensor* ic = ggml_im2col(ctx, dww, nb, 1, 0, 0, 0, 1, 0,
                                      /*is_2D*/false, GGML_TYPE_F32);
        ggml_tensor* r2 = ggml_mul_mat(ctx, ic, dww);
        dw = ggml_reshape_3d(ctx, r2, r2->ne[0], r2->ne[2], 1); // [Tc, C, 1]
    }
    dw = ggml_reshape_2d(ctx, dw, Tc, D);                 // [Tc, C]
    ggml_tensor* dwb = clone_weight_opt(ctx, ml, pre + "conv.depthwise_conv.bias");
    ggml_tensor* dwt = ggml_cont(ctx, ggml_transpose(ctx, dw)); // [C, Tc]
    if (dwb) dwt = ggml_add(ctx, dwt, dwb);

    ggml_tensor* normed;
    if (conv_norm_type == "layer_norm") {
        ggml_tensor* g = clone_weight(ctx, ml, pre + "conv.batch_norm.weight");
        ggml_tensor* bb = clone_weight(ctx, ml, pre + "conv.batch_norm.bias");
        normed = ggml_norm(ctx, dwt, ln_eps);
        normed = ggml_mul(ctx, normed, g);
        normed = ggml_add(ctx, normed, bb);
    } else {
        std::vector<float>& sc_host = pool.alloc_f32(D);
        std::vector<float>& sh_host = pool.alloc_f32(D);
        std::vector<float> g, bb, mm, var;
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.weight").c_str(), g);
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.bias").c_str(), bb);
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.running_mean").c_str(), mm);
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.running_var").c_str(), var);
        for (int cc2 = 0; cc2 < D; ++cc2) {
            sc_host[cc2] = g[cc2] / std::sqrt(var[cc2] + 1e-5f);
            sh_host[cc2] = bb[cc2] - mm[cc2] * sc_host[cc2];
        }
        int64_t d_ne[1] = {D};
        ggml_tensor* sc_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                                d_ne, sc_host.data(), sc_host.size() * sizeof(float));
        ggml_tensor* sh_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                                d_ne, sh_host.data(), sh_host.size() * sizeof(float));
        normed = ggml_add(ctx, ggml_mul(ctx, dwt, sc_t), sh_t);
    }
    normed = ggml_silu(ctx, normed);
    ggml_tensor* pw2w = clone_weight(ctx, ml, pre + "conv.pointwise_conv2.weight");
    pw2w = ggml_reshape_2d(ctx, pw2w, D, D);
    ggml_tensor* pw2b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv2.bias");
    ggml_tensor* cout = ggml_mul_mat(ctx, pw2w, normed); // [D, Tc]
    if (pw2b) cout = ggml_add(ctx, cout, pw2b);
    r = ggml_add(ctx, r, cout);    // r = r + conv_out

    // === Stage D: r = r + 0.5*FFN2(norm_ff2(r)); out = norm_out(r). ===
    ggml_tensor* h2 = feed_forward(layer_norm(r, "norm_feed_forward2"),
                                   "feed_forward2");
    h2 = ggml_scale(ctx, h2, 0.5f);
    r = ggml_add(ctx, r, h2);
    r = layer_norm(r, "norm_out");
    return r; // [D, Tc]
}

// The relative-position projection linear_pos(pos_emb) depends only on the attention
// window length (cache_len + Tc), never on the audio, yet the streaming forward used to
// recompute it in every layer for every chunk: on a 24-layer, D=1024 model with ~145
// positions that is ~7 GFLOP per chunk. It is computed once per window length for the
// session, already split into heads, and fed to the chunk graph as an input. Same ggml
// ops on the same inputs, so the outputs are unchanged.
const std::vector<std::vector<float>>& StreamingEncoder::pos_heads(int window) {
    auto it = pos_heads_.find(window);
    if (it != pos_heads_.end()) return it->second;

    const int D = d_model_, H = n_heads_, dk = D / H;
    const int pos_len = 2 * window - 1;
    std::vector<float> pe_host;
    rel_pos_encoding(window, D, pe_host); // row-major [pos_len, D]
    std::vector<std::vector<float>> heads(n_layers_);
    for (int i = 0; i < n_layers_; ++i) {
        const std::string w = "encoder.layers." + std::to_string(i) + ".self_attn.linear_pos.weight";
        bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t pe_ne[2] = {D, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne, pe_host.data(),
                                                     pe_host.size() * sizeof(float));
            ggml_tensor* p = ggml_mul_mat(ctx, clone_weight(ctx, ml_, w), pe); // [D, P]
            p = ggml_reshape_3d(ctx, p, dk, H, pos_len);
            return ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));         // [dk, P, H]
        }, heads[i]);
        if (!ok || (int)heads[i].size() != dk * pos_len * H)
            throw std::runtime_error("streaming encoder: linear_pos precompute failed");
    }
    return pos_heads_.emplace(window, std::move(heads)).first->second;
}

std::vector<float> StreamingEncoder::step(const std::vector<float>& mel_chunk_frames,
                                          int n_mel_frames, bool is_last,
                                          int& n_valid_out) {
    const int D = d_model_;
    const int n_mels = (int)ml_.config().n_mels;
    assert((int)mel_chunk_frames.size() == n_mels * n_mel_frames);

    // ---- 1. Compute Tc / drop host-side (deterministic from the subsampling
    //         arithmetic) so the fused graph knows the shapes up front. The
    //         subsampling builder itself runs in-graph below; here we only
    //         re-derive its output time length (3 stride-2 k=3 causal stages),
    //         matching Subsampling's internal calc. ----
    Subsampling sub(ml_);
    int Tsub_calc = n_mel_frames;
    {
        const int all_paddings = 3; // causal
        int t = n_mel_frames;
        for (int st = 0; st < 3; ++st) t = (t + all_paddings - 3) / 2 + 1;
        Tsub_calc = t;
    }
    const int drop = (step_ != 0) ? drop_extra_ : 0;
    int Tc = Tsub_calc - drop;
    if (Tc < 0) Tc = 0;

    if (Tc == 0) {     // no output this step (e.g. a too-short tail chunk)
        n_valid_out = 0;
        step_ += 1;
        return {};
    }

    // ---- 2. Positional encoding. NeMo pos_enc(cache_len): input_len = Tc +
    //         cache_len; pos_emb spans 2*input_len-1 positions. cache_len = 70. ----
    const int cache_len = last_channel_cache_;
    const int Pn = Tc + cache_len;
    const int pos_len = 2 * Pn - 1;
    const int clc_len_now = clc_len_;   // snapshot for the in-graph mask

    const std::string cnt = ml_.config().conv_norm_type;
    const std::vector<std::vector<float>>& pos_heads = this->pos_heads(Pn);
    StreamLayerCapture caps[/*max layers*/64];
    assert(n_layers_ <= 64);

    // ---- 3. The WHOLE chunk forward is ONE ggml graph: subsampling (with the
    //         pre-encode-cache window) -> drop -> xscaling -> N streaming conformer
    //         layers (each reading its conv/attn cache as inputs and emitting the
    //         updated caches as captured outputs) -> valid output slice. Computed
    //         in a SINGLE Backend::compute (vs the old ~5 graphs/layer). ----
    std::vector<float> out_full;  // [Tc, D] row-major (the final layer output)
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // Subsampling on the full mel window (already includes pre-encode
            // overlap): the WHOLE window is real audio (no center-pad), so the
            // entry valid length is n_mel_frames. Output [D, Tsub].
            int Tp = 0, valid = 0;
            ggml_tensor* sx = sub.build_graph(ctx, mel_chunk_frames, n_mels,
                                              n_mel_frames, pool, Tp, valid,
                                              /*in_valid_frames*/n_mel_frames);
            assert(Tp == Tsub_calc);

            // Drop drop_extra_pre_encoded leading subsampled frames (NeMo
            // forward_internal; only when cache is present == every step after the
            // first). View columns [drop, Tp) of sx [D, Tp] -> [D, Tc].
            ggml_tensor* x = sx;
            if (drop > 0) {
                x = ggml_view_2d(ctx, sx, D, Tc, sx->nb[1],
                                 (size_t)drop * sx->nb[1]);
                x = ggml_cont(ctx, x);
            }

            // xscaling (gated; off for this model).
            if (xscaling_) {
                x = ggml_scale(ctx, x, std::sqrt((float)D));
            }

            // Conformer layer stack (conv + attention caches threaded in-graph).
            const int LP = left_pad_;
            for (int i = 0; i < n_layers_; ++i) {
                // Cache reads as graph inputs (host data set after alloc). The
                // host buffers ARE the live cache vectors, alive across compute.
                int64_t ch_ne[2] = {D, cache_len};
                ggml_tensor* cache_ch_t = pk::graph_input_tensor(ctx,
                        GGML_TYPE_F32, 2, ch_ne, cache_channel_[i].data(),
                        cache_channel_[i].size() * sizeof(float));
                int64_t t_ne[2] = {D, LP};
                ggml_tensor* cache_t_t = pk::graph_input_tensor(ctx,
                        GGML_TYPE_F32, 2, t_ne, cache_time_[i].data(),
                        cache_time_[i].size() * sizeof(float));

                x = build_stream_layer(ctx, ml_, i, x, Tc, pos_heads[i].data(), pos_len,
                                       cache_len, clc_len_now, n_heads_, D,
                                       conv_kernel_, att_left_, att_right_, cnt,
                                       cache_ch_t, cache_t_t, pool, &caps[i]);
            }
            return x; // [D, Tc] -> row-major [Tc, D]
        }, out_full);
    assert(ok && "streaming fused encoder graph failed"); (void)ok;

    // ---- 4. Commit the captured next-step caches (read back from the graph). ----
    for (int i = 0; i < n_layers_; ++i) {
        assert((int)caps[i].next_channel.size() == cache_len * D);
        assert((int)caps[i].next_time.size() == left_pad_ * D);
        cache_channel_[i].swap(caps[i].next_channel);
        cache_time_[i].swap(caps[i].next_time);
    }
    // After the layer stack the attention cache_last_channel_len has grown by Tc.
    clc_len_ = std::min(cache_len, clc_len_ + Tc);
    step_ += 1;

    // ---- 5. Slice valid output. keep_all_outputs == is_last: mid-stream keep
    //         only valid_out_len frames; final chunk keep all Tc frames. ----
    int valid = is_last ? Tc : std::min(valid_out_len_, Tc);
    n_valid_out = valid;
    std::vector<float> out((size_t)valid * D);
    std::memcpy(out.data(), out_full.data(), (size_t)valid * D * sizeof(float));
    return out;
}

} // namespace pk
