#include "encoder.hpp"
#include "subsampling.hpp"
#include "conformer.hpp"
#include "pos_enc.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "graph_builder.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace pk {

// Decide the self-attention window for an encoder of Tp frames. Returns W>0 to
// use NeMo rel_pos_local_attn [W,W] (banded, O(T*window)); -1 for full attention.
//
// The attention uses the chunk-matmul banded path (build_graph_local_chunked),
// which emits O(1) graph nodes regardless of window, so W can go to NeMo's full
// [128,128] without overflowing the metadata-context budget (backend.cpp
// kGraphSize). (The older pad-and-shift path emitted ~6*(2W+1) nodes/layer,
// which is why this was capped at 32.)
static int local_attn_window(int Tp) {
    constexpr int kMaxLocalWindow = 128;
    if (const char* e = std::getenv("PARAKEET_ATT_CONTEXT")) {
        const int w = std::atoi(e);
        if (w <= 0) return -1;                 // 0 / negative -> force full attention
        return w > kMaxLocalWindow ? kMaxLocalWindow : w;
    }
    // Auto: long audio (~>11 min at 8x subsampling) switches to local attention
    // so full O(T^2) attention can't OOM the device.
    constexpr int kLocalThreshold = 8192;
    return Tp > kLocalThreshold ? kMaxLocalWindow : -1;
}

Encoder::Encoder(const ModelLoader& ml)
    : ml_(ml) {
    d_model_  = (int)ml.config().d_model;
    n_layers_ = (int)ml.config().n_layers;
    xscaling_ = ml.config().xscaling;
    assert(n_layers_ > 0 && d_model_ > 0);
}

void Encoder::forward(const std::vector<float>& mel, int n_mels, int T,
                      std::vector<float>& enc_out, int& d_model, int& Tout) const {
    std::vector<int> none;
    std::vector<std::vector<float>> ignored;
    forward_capture(mel, n_mels, T, enc_out, d_model, Tout, none, ignored);
}

void Encoder::forward_capture(const std::vector<float>& mel, int n_mels, int T,
                              std::vector<float>& enc_out, int& d_model, int& Tout,
                              const std::vector<int>& capture_layers,
                              std::vector<std::vector<float>>& layer_outs) const {
    forward_impl(mel, n_mels, T, nullptr, enc_out, d_model, Tout, capture_layers, layer_outs);
}

void Encoder::forward_rc(const std::vector<float>& mel, int n_mels, int T, const RcMask& rc,
                         std::vector<float>& enc_out, int& d_model, int& Tout) const {
    std::vector<int> none;
    std::vector<std::vector<float>> ignored;
    forward_impl(mel, n_mels, T, &rc, enc_out, d_model, Tout, none, ignored);
}

void Encoder::forward_impl(const std::vector<float>& mel, int n_mels, int T, const RcMask* rc,
                           std::vector<float>& enc_out, int& d_model, int& Tout,
                           const std::vector<int>& capture_layers,
                           std::vector<std::vector<float>>& layer_outs) const {
    // The WHOLE encoder is ONE ggml graph: subsampling -> xscaling -> N conformer
    // layers (each FFN1+attn+conv+FFN2+norm_out in-graph) -> final transpose,
    // computed in a SINGLE Backend::compute. (Mel stays plain C++, transposed and
    // fed as the input inside the subsampling builder.) Versus the old ~85
    // per-utterance graphs, this lets the CPU threadpool parallelise across the
    // entire encoder. The per-component graph-BUILDERS (Subsampling/RelPos/
    // ConformerLayer::build_graph) are reused verbatim by the unit tests.
    layer_outs.assign(capture_layers.size(), {});

    // No frames (an empty clip, or a clip shorter than one hop with
    // parakeet.unified.valid_length): nothing to encode. Return an empty output,
    // and the decoders emit no tokens.
    if (T <= 0) {
        d_model = d_model_;
        Tout = 0;
        enc_out.clear();
        return;
    }

    // Pos_emb is computed host-side once T' is known (deterministic from T, but
    // we read it from the subsampling builder). Storage lives in the pool so it
    // outlives Backend::compute. capture_layers map: layer idx -> output slot.
    GraphInputPool pool;
    Subsampling sub(ml_);
    int Tp = 0, valid_len = 0, dm_out = 0;

    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- 1. Subsampling: mel [n_mels,T] -> x [d_model, T'] (+ valid). ----
            ggml_tensor* x = sub.build_graph(ctx, mel, n_mels, T, pool, Tp,
                                             valid_len);
            dm_out = d_model_;
            assert((int)x->ne[0] == d_model_);

            // ---- 2. xscaling (gated; off for this model). NeMo scales x. ----
            if (xscaling_) {
                x = ggml_scale(ctx, x, std::sqrt((float)d_model_));
            }

            // ---- 3. Positional encoding. Long audio uses NeMo
            //         rel_pos_local_attn (banded, O(T*window)) so attention can't
            //         OOM; short audio keeps full attention (NeMo-exact). The
            //         window-relative RC mask needs full attention, so it never
            //         takes the local path. ----
            const int att_w   = rc ? -1 : local_attn_window(Tp);
            const bool local  = att_w > 0;
            const int pos_len = local ? (2 * att_w + 1) : (2 * Tp - 1);
            std::vector<float>& pe_host = pool.alloc_f32();
            if (local) local_rel_pos_encoding(att_w, att_w, d_model_, pe_host);
            else       rel_pos_encoding(Tp, d_model_, pe_host); // [pos_len, d_model]
            int64_t pe_ne[2] = {d_model_, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pe_host.data(), pe_host.size() * sizeof(float));

            // ---- 4. Conformer layer stack (all in-graph). ----
            for (int i = 0; i < n_layers_; ++i) {
                ConformerLayer layer(ml_, i);
                x = layer.build_graph(ctx, x, Tp, pe, pos_len, valid_len, pool,
                                      local ? att_w : -1, local ? att_w : -1, rc);
                // Capture requested layer outputs from the SAME graph (row-major
                // [T', d_model], matching the layer output orientation).
                for (size_t c = 0; c < capture_layers.size(); ++c) {
                    if (capture_layers[c] == i)
                        pk::capture_graph_output(x, &layer_outs[c]);
                }
            }

            // ---- 5. Final transpose [d_model, T'] -> [T', d_model] (channels-
            //         first): ggml transpose+cont gives ne0=T', ne1=d_model ->
            //         row-major [d_model, T'] = enc_out[c*T' + t]. ----
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            return x; // ne [T', d_model] -> row-major [d_model, T']
        }, enc_out);

    assert(ok && "encoder graph failed");
    (void)ok;

    d_model = dm_out;
    Tout = Tp;
}

void Encoder::forward_batch(const MelBatch& mels,
                            std::vector<std::vector<float>>& enc_outs,
                            int& d_model, int& Tout,
                            std::vector<int>& valid_Tout) const {
    // Phase 5: the WHOLE batched encoder is ONE fused ggml graph, mirroring
    // forward_capture but at B>1: subsampling -> xscaling -> pos_emb -> N
    // conformer layers, all [d_model, T', B]. We return the raw [d_model, Tp, B]
    // tensor and do the channels-first transpose host-side while splitting per
    // item (see the index mapping below).
    GraphInputPool pool;
    Subsampling sub(ml_);
    int Tp = 0;
    std::vector<int> vout;
    // Per-item entry valid frames: T-1 per clip, or T for a mel already trimmed
    // to its valid length (see Subsampling::offline_entry_valid).
    std::vector<int> vin(mels.B);
    for (int b = 0; b < mels.B; ++b) vin[b] = sub.offline_entry_valid(mels.valid_T[b]);

    std::vector<float> flat; // receives [d_model, Tp, B] (ne0=d_model fastest)
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- 1. Subsampling (batched): mel -> x [d_model, T', B] (+ valid). ----
            ggml_tensor* x = sub.build_graph_batched(ctx, mels.data.data(),
                                mels.n_mels, mels.T_max, mels.B, pool, Tp, vout, vin);
            assert((int)x->ne[0] == d_model_);

            // ---- 2. xscaling (gated; off for this model). ----
            if (xscaling_) x = ggml_scale(ctx, x, std::sqrt((float)d_model_));

            // ---- 3. Positional encoding (local for long audio; see B=1 path). ----
            const int att_w   = local_attn_window(Tp);
            const bool local  = att_w > 0;
            const int pos_len = local ? (2 * att_w + 1) : (2 * Tp - 1);
            std::vector<float>& pe_host = pool.alloc_f32();
            if (local) local_rel_pos_encoding(att_w, att_w, d_model_, pe_host);
            else       rel_pos_encoding(Tp, d_model_, pe_host); // [pos_len, d_model]
            int64_t pe_ne[2] = {d_model_, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pe_host.data(), pe_host.size() * sizeof(float));

            // ---- 4. Conformer layer stack (all in-graph, shared pe). ----
            for (int i = 0; i < n_layers_; ++i) {
                ConformerLayer layer(ml_, i);
                x = layer.build_graph_batched(ctx, x, Tp, mels.B, pe, pos_len, vout, pool,
                                              local ? att_w : -1, local ? att_w : -1);
            }
            return x; // [d_model, Tp, B]
        }, flat);

    assert(ok && "batched encoder graph failed");
    (void)ok;

    // flat is the [d_model, Tp, B] tensor flattened in ggml order (ne0=d_model
    // fastest): element (c, t, b) sits at index ((size_t)b*Tp + t)*d_model_ + c.
    // Split per item and transpose to channels-first [d_model, Tp]
    // (enc_out[c*Tp + t]), matching what forward()/forward_capture return.
    d_model = d_model_;
    Tout = Tp;
    valid_Tout = vout;
    // Each enc_outs[b] is channels-first [d_model, valid_Tout[b]]: compact to the
    // per-item non-pad frame count so the row stride equals valid_Tout[b]. The
    // fused graph runs every item at the padded width Tp, but the trailing
    // (Tp - vout[b]) columns are pad-derived; emitting them would (a) make the
    // row stride differ from valid_Tout[b] (decoders index enc_out[c*Tout + t]
    // with Tout = valid_Tout[b]) and (b) feed pad frames into the decoder. Both
    // corrupt a padded (shorter) item's decode.
    enc_outs.assign(mels.B, std::vector<float>());
    for (int b = 0; b < mels.B; ++b) {
        const int tv = vout[b];
        enc_outs[b].resize((size_t)d_model_ * tv);
        for (int t = 0; t < tv; ++t)
            for (int c = 0; c < d_model_; ++c)
                enc_outs[b][(size_t)c * tv + t] =
                    flat[(((size_t)b * Tp) + t) * d_model_ + c];
    }
}

void Encoder::run_post_subsampling_batch(const std::vector<float>& x0_host,
            int Tp, int B, const std::vector<int>& vout,
            std::vector<std::vector<float>>& enc_outs, int& d_model, int& Tout,
            std::vector<int>& valid_Tout) const {
    // Mirror of forward_batch's post-subsampling body (steps 2-4 + per-item
    // split), but the subsampled features arrive pre-computed in x0_host and are
    // injected as a graph input instead of built via sub.build_graph_batched.
    GraphInputPool pool;
    std::vector<float> flat; // receives [d_model, Tp, B] (ne0=d_model fastest)
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- 1. Inject pre-subsampled features x [d_model, Tp, B]. ----
            int64_t x_ne[3] = { d_model_, Tp, B };
            ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, x_ne,
                                 x0_host.data(), x0_host.size() * sizeof(float));

            // ---- 2. xscaling (gated; off for this model). ----
            if (xscaling_) x = ggml_scale(ctx, x, std::sqrt((float)d_model_));

            // ---- 3. Positional encoding (local for long audio; see B=1 path). ----
            const int att_w   = local_attn_window(Tp);
            const bool local  = att_w > 0;
            const int pos_len = local ? (2 * att_w + 1) : (2 * Tp - 1);
            std::vector<float>& pe_host = pool.alloc_f32();
            if (local) local_rel_pos_encoding(att_w, att_w, d_model_, pe_host);
            else       rel_pos_encoding(Tp, d_model_, pe_host); // [pos_len, d_model]
            int64_t pe_ne[2] = {d_model_, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pe_host.data(), pe_host.size() * sizeof(float));

            // ---- 4. Conformer layer stack (all in-graph, shared pe). ----
            for (int i = 0; i < n_layers_; ++i) {
                ConformerLayer layer(ml_, i);
                x = layer.build_graph_batched(ctx, x, Tp, B, pe, pos_len, vout, pool,
                                              local ? att_w : -1, local ? att_w : -1);
            }
            return x; // [d_model, Tp, B]
        }, flat);

    assert(ok && "tiled post-subsampling encoder graph failed");
    (void)ok;

    d_model = d_model_;
    Tout = Tp;
    valid_Tout = vout;
    enc_outs.assign(B, std::vector<float>());
    for (int b = 0; b < B; ++b) {
        const int tv = vout[b];
        enc_outs[b].resize((size_t)d_model_ * tv);
        for (int t = 0; t < tv; ++t)
            for (int c = 0; c < d_model_; ++c)
                enc_outs[b][(size_t)c * tv + t] =
                    flat[(((size_t)b * Tp) + t) * d_model_ + c];
    }
}

void Encoder::forward_batch_tiled(const MelBatch& mels,
        std::vector<std::vector<float>>& enc_outs, int& d_model, int& Tout,
        std::vector<int>& valid_Tout, int tile_out_frames) const {
    Subsampling sub(ml_);
    const int B = mels.B;
    std::vector<std::vector<float>> sub_b(B);   // each [Tp_b, d_model] (t*dm+c)
    std::vector<int> Tp_b(B), vout(B);
    int Tp_max = 0, dm = 0;
    for (int b = 0; b < B; ++b) {
        // slice item b's real mel [n_mels, valid_T[b]] out of the padded batch buffer
        std::vector<float> mel((size_t)mels.n_mels * mels.valid_T[b]);
        for (int m = 0; m < mels.n_mels; ++m)
            for (int t = 0; t < mels.valid_T[b]; ++t)
                mel[(size_t)m*mels.valid_T[b]+t] =
                    mels.data[((size_t)b*mels.n_mels+m)*mels.T_max + t];
        int Tp=0, vl=0;
        sub.forward_tiled(mel, mels.n_mels, mels.valid_T[b], tile_out_frames,
                          sub_b[b], Tp, dm, vl);
        Tp_b[b]=Tp; vout[b]=vl; if (Tp>Tp_max) Tp_max=Tp;
    }
    // assemble x0 [d_model, Tp_max, B], zero-padded per item.
    // sub_b[b] is [Tp_b, d_model] (t*dm+c); x0 wants (c,t,b) at (b*Tp_max+t)*dm+c.
    std::vector<float> x0((size_t)dm*Tp_max*B, 0.0f);
    for (int b=0;b<B;++b) for (int t=0;t<Tp_b[b];++t) for (int c=0;c<dm;++c)
        x0[((size_t)b*Tp_max+t)*dm+c] = sub_b[b][(size_t)t*dm+c];
    run_post_subsampling_batch(x0, Tp_max, B, vout, enc_outs, d_model, Tout, valid_Tout);
}

} // namespace pk
