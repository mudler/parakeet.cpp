#include "mel.hpp"
#include "fft.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace pk {

// CONSTANT epsilon added to std in NeMo normalize_batch (features.py: CONSTANT = 1e-5).
static constexpr double kNormEps = 1e-5;

MelKernel::MelKernel(const ModelLoader& ml) {
    const ParakeetConfig& c = ml.config();
    n_fft_     = (int)c.n_fft;
    hop_       = (int)c.hop_length;
    assert((n_fft_ & (n_fft_ - 1)) == 0 && "n_fft must be a power of two for rfft");
    n_mels_    = (int)c.n_mels;
    n_bins_    = n_fft_ / 2 + 1;
    preemph_   = c.preemph;
    mag_power_ = c.mag_power;
    log_guard_ = c.log_zero_guard;
    per_feature_ = (c.normalize == "per_feature");

    // --- Window: lift the Hann window from the GGUF and center-pad to n_fft. ---
    // NeMo builds torch.hann_window(win_length, periodic=False) and passes it to
    // torch.stft with win_length < n_fft, so torch zero-pads the window to n_fft,
    // centered: pad (n_fft - win_length) // 2 zeros on each side.
    window_.assign(n_fft_, 0.0f);
    ggml_tensor* w = ml.tensor("preprocessor.featurizer.window");
    if (w) {
        std::vector<float> wbuf;
        pk::weight_to_host_f32(ml, "preprocessor.featurizer.window", wbuf);
        const int wlen = (int)w->ne[0];
        const float* wd = wbuf.data();
        if (wlen == n_fft_) {
            std::memcpy(window_.data(), wd, sizeof(float) * n_fft_);
        } else {
            assert(wlen <= n_fft_ && "window tensor wider than n_fft — wrong model?");
            const int left = (n_fft_ - wlen) / 2;
            for (int i = 0; i < wlen && (left + i) < n_fft_; ++i) {
                if (left + i < 0) continue;
                window_[left + i] = wd[i];
            }
        }
    }

    // --- Filterbank: preprocessor.featurizer.fb has ggml ne=[n_bins, n_mels, 1]
    // i.e. numpy [1, n_mels, n_bins] row-major -> fb_[m*n_bins + b]. ---
    fb_.assign((size_t)n_mels_ * n_bins_, 0.0f);
    ggml_tensor* fb = ml.tensor("preprocessor.featurizer.fb");
    if (fb) {
        std::vector<float> fbuf;
        pk::weight_to_host_f32(ml, "preprocessor.featurizer.fb", fbuf);
        const float* fd = fbuf.data();
        std::memcpy(fb_.data(), fd, sizeof(float) * (size_t)n_mels_ * n_bins_);
        // Build the sparse (bin, weight) index: skip exact-zero weights. This is
        // bit-identical to the dense accumulation (0.0*power == 0.0 for finite
        // power) while removing ~98% of the per-frame filterbank MACs.
        fb_nz_off_.assign((size_t)n_mels_ + 1, 0);
        for (int m = 0; m < n_mels_; ++m) {
            const float* fbm = &fb_[(size_t)m * n_bins_];
            for (int b = 0; b < n_bins_; ++b)
                if (fbm[b] != 0.0f) { fb_nz_idx_.push_back(b); fb_nz_val_.push_back(fbm[b]); }
            fb_nz_off_[m + 1] = (int32_t)fb_nz_idx_.size();
        }
    }
}

void MelKernel::frame_logmel(const double* frame_in, float* out_col, int out_stride,
                             std::vector<float>& frame,
                             std::vector<float>& re, std::vector<float>& im,
                             std::vector<double>& power) const {
    // Window the (already-preemphasized, frame-major, double) samples exactly
    // like MelFrontend: wframe[i] = (float)(padded[i] * (double)window[i]).
    frame.resize((size_t)n_fft_);
    for (int i = 0; i < n_fft_; ++i)
        frame[i] = (float)(frame_in[i] * (double)window_[i]);

    pk::rfft(frame, re, im);

    // ----- power spectrum -----
    // NeMo: x = sqrt(re^2 + im^2) (guard=0 at inference), then x = x^mag_power.
    power.resize((size_t)n_bins_);
    for (int b = 0; b < n_bins_; ++b) {
        const double mag = std::sqrt((double)re[b] * re[b] + (double)im[b] * im[b]);
        power[b] = (mag_power_ == 1.0f) ? mag : std::pow(mag, (double)mag_power_);
    }

    // ----- mel projection + log(mel + guard) -----
    // Iterate only the nonzero filterbank weights (fb is ~98% zeros); bit-identical
    // to the dense sum since an exact-zero weight contributes exactly 0.0.
    const int32_t* off = fb_nz_off_.data();
    const int32_t* idx = fb_nz_idx_.data();
    const float*   val = fb_nz_val_.data();
    for (int m = 0; m < n_mels_; ++m) {
        double acc = 0.0;
        for (int32_t k = off[m]; k < off[m + 1]; ++k)
            acc += (double)val[k] * power[idx[k]];
        out_col[(size_t)m * out_stride] = (float)std::log(acc + (double)log_guard_);
    }
}

MelFrontend::MelFrontend(const ModelLoader& ml) : k_(ml) {}

void MelFrontend::compute(const std::vector<float>& samples,
                          std::vector<float>& feats, int& n_mels, int& T) const {
    if (samples.empty()) { n_mels = k_.n_mels_; T = 0; feats.clear(); return; }
    const int S = (int)samples.size();
    const int n_fft_ = k_.n_fft_, hop_ = k_.hop_, n_mels_ = k_.n_mels_;
    const float preemph_ = k_.preemph_;

    // ----- Step 2: Preemphasis -----
    // NeMo: x = cat((x[:,0], x[:,1:] - preemph * x[:,:-1])); first sample unchanged.
    std::vector<double> x((size_t)S);
    if (preemph_ > 0.0f && S > 0) {
        x[0] = samples[0];
        for (int t = 1; t < S; ++t)
            x[t] = (double)samples[t] - (double)preemph_ * (double)samples[t - 1];
    } else {
        for (int t = 0; t < S; ++t) x[t] = samples[t];
    }

    // seq_len = floor(S / hop) (NeMo get_seq_len for center=True:
    //   floor((S + n_fft//2*2 - n_fft) / hop) = floor(S / hop)).
    const int seq_len = (S > 0) ? (S / hop_) : 0;

    // ----- Step 3: STFT with center=True -----
    // Reflect-pad n_fft//2 on each side, then frame with hop, apply the window,
    // FFT -> keep bins 0..n_fft/2. Frame count = 1 + floor((S + 2*pad - n_fft)/hop).
    const int pad = n_fft_ / 2;
    const int padded_len = S + 2 * pad;
    const int n_frames = (padded_len >= n_fft_) ? (1 + (padded_len - n_fft_) / hop_) : 0;
    T = n_frames;
    n_mels = n_mels_;

    // Zero-padded signal. NeMo's stft uses pad_mode="constant" (features.py),
    // so center=True pads n_fft//2 ZEROS on each side (not reflection).
    // The original signal sits at [pad, pad+S); everything else is 0.
    std::vector<double> padded((size_t)padded_len, 0.0);
    for (int j = 0; j < S; ++j) padded[pad + j] = x[j];

    // Power spectrum per frame: power[b][t] = (re^2 + im^2) (mag then ^mag_power).
    // We store frame-major scratch; mel is computed directly into feats.
    feats.assign((size_t)n_mels_ * T, 0.0f);

    std::vector<float> frame, re, im;
    std::vector<double> power;

    // Each frame's mel column shares the exact per-frame math with the streaming
    // front end via MelKernel::frame_logmel (one source of truth). out stride T
    // writes feats[m*T + t] (feat-major, inner = time).
    for (int t = 0; t < T; ++t) {
        const int start = t * hop_;
        k_.frame_logmel(&padded[start], &feats[t], T, frame, re, im, power);
    }

    // ----- Step 7: per-feature normalization (normalize_batch, per_feature) -----
    // Per mel-bin (row): mean over the first `seq_len` frames, unbiased std
    // (ddof=1), std += CONSTANT, then (x - mean) / std. Frames >= seq_len are
    // zeroed (valid_mask). B=1.
    if (k_.per_feature_ && T > 0) {
        const int valid = std::min(seq_len, T);
        for (int m = 0; m < n_mels_; ++m) {
            float* row = &feats[(size_t)m * T];
            double mean = 0.0;
            if (valid > 0) {
                for (int t = 0; t < valid; ++t) mean += row[t];
                mean /= (double)valid;
            }
            double var = 0.0;
            if (valid > 1) {
                for (int t = 0; t < valid; ++t) {
                    const double d = (double)row[t] - mean;
                    var += d * d;
                }
                var /= (double)(valid - 1); // unbiased (ddof=1)
            }
            double sd = std::sqrt(var); // NaN-guard not needed: valid>1 here
            sd += kNormEps;
            for (int t = 0; t < T; ++t) {
                if (t < valid) row[t] = (float)(((double)row[t] - mean) / sd);
                else           row[t] = 0.0f; // masked beyond seq_len
            }
        }
    }
}

// ===========================================================================
// StreamingMel — incremental, frame-local log-mel for the streaming front end.
// ===========================================================================

StreamingMel::StreamingMel(const ModelLoader& ml) : k_(ml), pad_(k_.n_fft_ / 2) {
    // Incremental causal mel is only equivalent to the full-buffer compute when
    // there is no whole-utterance normalization. The only streaming model
    // (nvidia/parakeet_realtime_eou_120m-v1) uses normalize="NA"; per_feature
    // would need cross-frame stats and must use MelFrontend::compute instead.
    assert(!k_.per_feature_ &&
           "StreamingMel is frame-local; per_feature needs whole-utterance "
           "stats — use MelFrontend::compute for per_feature models");
    reset();
}

void StreamingMel::reset() {
    samples_seen_ = 0;
    emitted_      = 0;
    have_prev_    = false;
    prev_raw_     = 0.0f;
    buf_.clear();
    buf_origin_   = 0;
    framebuf_.assign((size_t)k_.n_fft_, 0.0);
}

// Build every frame whose right edge index (t*hop + pad) is <= `avail_x` real
// preemph samples, appending each frame's mel column to `out`. `avail_x` is the
// number of real preemph samples conceptually available for THIS pass: during a
// feed it is samples_seen_; during finalize the padding lets all T frames go.
static inline void build_frames(const MelKernel& k, int pad,
                                const std::vector<double>& buf, long long buf_origin,
                                long long S /* total real preemph samples */,
                                int& emitted, int last_frame_excl,
                                std::vector<double>& framebuf,
                                std::vector<float>& wframe,
                                std::vector<float>& re, std::vector<float>& im,
                                std::vector<double>& power,
                                std::vector<float>& out, int& n_new) {
    const int hop = k.hop_, n_fft = k.n_fft_, n_mels = k.n_mels_;
    // Highest frame index whose right edge is real audio: t*hop + pad <= S.
    // (For finalize we pass last_frame_excl = T directly.)
    int start_frame = emitted;
    for (int t = start_frame; t < last_frame_excl; ++t) {
        const long long lo = (long long)t * hop - pad;   // preemph index of frame start
        // Gather n_fft preemph samples [lo, lo+n_fft), zero where out of [0, S).
        for (int i = 0; i < n_fft; ++i) {
            const long long idx = lo + i;
            double v = 0.0;
            if (idx >= 0 && idx < S) {
                const long long k_off = idx - buf_origin;
                // idx must be retained (we never drop a sample a pending frame needs)
                v = buf[(size_t)k_off];
            }
            framebuf[(size_t)i] = v;
        }
        // Append one mel column (feat-major: we grow `out` by n_mels, one frame
        // at a time; the caller transposes/concatenates frame-major below).
        const size_t base = out.size();
        out.resize(base + (size_t)n_mels);
        // Write column with stride 1 here (per-frame contiguous), the wrapper
        // reorders into [n_mels, n_new] feat-major after all frames are built.
        k.frame_logmel(framebuf.data(), out.data() + base, 1, wframe, re, im, power);
        ++emitted;
        ++n_new;
    }
}

std::vector<float> StreamingMel::feed(const float* pcm, int n, int& n_new_frames) {
    n_new_frames = 0;
    // ----- Preemphasis (carried across feeds) -----
    // y[0] (very first sample of the stream) = x[0]; otherwise
    // y[t] = x[t] - preemph * x[t-1], where x[t-1] may live in the prior feed.
    const float preemph = k_.preemph_;
    for (int i = 0; i < n; ++i) {
        const float raw = pcm[i];
        double y;
        if (preemph > 0.0f) {
            if (!have_prev_) y = raw;                                // first ever sample
            else             y = (double)raw - (double)preemph * (double)prev_raw_;
        } else {
            y = raw;
        }
        buf_.push_back(y);
        prev_raw_ = raw;
        have_prev_ = true;
    }
    samples_seen_ += n;
    const long long S = samples_seen_;  // real preemph samples available now

    // Frames ready mid-stream: t*hop + pad <= S  =>  t <= (S - pad)/hop.
    int last_frame_excl = 0;
    if (S >= pad_) last_frame_excl = (int)((S - pad_) / k_.hop_) + 1;

    std::vector<float> col_major;  // [n_new frames] x [n_mels], frame-contiguous
    build_frames(k_, pad_, buf_, buf_origin_, S, emitted_, last_frame_excl,
                 framebuf_, wframe_, re_, im_, power_, col_major, n_new_frames);

    // Drop retained preemph samples no longer needed by any future frame. The
    // next frame to emit is `emitted_`; its lowest needed preemph index is
    // emitted_*hop - pad. Keep from there on. This bounds the buffer to
    // ~n_fft + hop samples (NOT the stream length).
    const long long keep_from = std::max<long long>(0, (long long)emitted_ * k_.hop_ - pad_);
    if (keep_from > buf_origin_) {
        const long long drop = keep_from - buf_origin_;
        if (drop >= (long long)buf_.size()) { buf_.clear(); }
        else buf_.erase(buf_.begin(), buf_.begin() + (size_t)drop);
        buf_origin_ = keep_from;
    }

    return to_feat_major(col_major, n_new_frames);
}

std::vector<float> StreamingMel::finalize(int& n_tail_frames) {
    n_tail_frames = 0;
    const long long S = samples_seen_;
    // Total frames for the full padded clip: T = 1 + floor(S / hop) (even n_fft),
    // 0 if no audio at all.
    const int T = (S > 0) ? (int)(S / k_.hop_) + 1 : 0;

    std::vector<float> col_major;
    build_frames(k_, pad_, buf_, buf_origin_, S, emitted_, T,
                 framebuf_, wframe_, re_, im_, power_, col_major, n_tail_frames);

    // All frames emitted; the retained buffer is no longer needed.
    buf_.clear();
    buf_origin_ = (long long)emitted_ * k_.hop_;  // (consistent; not reused)

    return to_feat_major(col_major, n_tail_frames);
}

// Reorder the frame-contiguous columns produced by build_frames (n frames each
// of length n_mels, frame-major) into the [n_mels, n] feat-major layout the
// encoder/test expect (out[m*n + t]).
std::vector<float> StreamingMel::to_feat_major(const std::vector<float>& col_major,
                                               int n) const {
    const int n_mels = k_.n_mels_;
    std::vector<float> out((size_t)n_mels * n);
    for (int t = 0; t < n; ++t)
        for (int m = 0; m < n_mels; ++m)
            out[(size_t)m * n + t] = col_major[(size_t)t * n_mels + m];
    return out;
}

} // namespace pk
