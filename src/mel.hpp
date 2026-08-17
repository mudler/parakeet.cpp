#pragma once
#include "model_loader.hpp"
#include <cassert>
#include <vector>
namespace pk {

// Shared mel/STFT config + per-frame log-mel math, lifted from the GGUF. Both
// the offline MelFrontend and the streaming StreamingMel build on this so the
// numerics are identical (one source of truth for preemph/STFT/power/mel/log).
struct MelKernel {
    explicit MelKernel(const ModelLoader& ml);

    // Compute one log-mel frame from n_fft (already-preemphasized, NOT-yet-
    // windowed) samples `frame_in` (length n_fft, frame-major time, kept in
    // double precision exactly like MelFrontend's `padded`). Writes n_mels
    // values into out_col with stride `out_stride` (out_col[m*out_stride]).
    // Uses the shared scratch buffers (frame/re/im/power) to avoid per-frame
    // allocation. Bit-identical to the inner loop of MelFrontend::compute.
    void frame_logmel(const double* frame_in, float* out_col, int out_stride,
                      std::vector<float>& frame,
                      std::vector<float>& re, std::vector<float>& im,
                      std::vector<double>& power) const;

    int n_fft_, hop_, n_mels_, n_bins_;
    float preemph_, mag_power_, log_guard_;
    bool per_feature_;
    std::vector<float> window_;  // [n_fft] (centered-padded Hann from GGUF)
    std::vector<float> fb_;      // [n_mels, n_bins] row-major (fb_[m*n_bins + b])

    // Sparse filterbank: the mel matrix is overwhelmingly zeros (each filter is
    // band-limited in frequency). Store only the nonzero (bin, weight) pairs so
    // the per-frame mel projection skips the ~98% zero entries. Bit-identical to
    // the dense accumulation (skipping an exact-zero weight contributes 0.0).
    // Flat arrays + per-mel offset/count (no per-frame allocation).
    std::vector<int32_t> fb_nz_idx_;   // bin index per nonzero entry
    std::vector<float>   fb_nz_val_;   // weight per nonzero entry
    std::vector<int32_t> fb_nz_off_;   // [n_mels+1] prefix offsets into the above
};

class MelFrontend {
public:
    explicit MelFrontend(const ModelLoader& ml);
    // samples: 16k mono. Out: feats row-major [n_mels, T] (frame-major inner = T).
    void compute(const std::vector<float>& samples, std::vector<float>& feats, int& n_mels, int& T) const;
private:
    MelKernel k_;
};

// Incremental (frame-local) log-mel for the streaming front end. Produces mel
// frames that are bit-identical (modulo float-add order, which is preserved
// here) to MelFrontend::compute run on the full concatenated PCM, but without
// recomputing the whole buffer on every feed and without retaining O(stream)
// audio.
//
// center=True STFT: frame i is centered at original sample i*hop, covering
// original indices [i*hop - n_fft/2, i*hop + n_fft/2). Frame i is "ready" once
// `real_samples_seen >= i*hop + n_fft/2` (its right edge is real audio). feed()
// emits exactly the frames that became ready with the just-arrived PCM; the
// n_fft/2 start zero-pad of the earliest frames is applied once (conceptually
// prepended). finalize() appends the n_fft/2 end zero-pad and emits the
// remaining tail frames, so the TOTAL frame count and values match
// MelFrontend::compute(full_pcm) exactly (T = 1 + floor(len/hop) for even n_fft).
//
// Preemphasis (y[t] = x[t] - 0.97*x[t-1], y[0] = x[0] for the very first sample
// of the stream) is carried across feeds via a 1-sample raw history.
//
// LIMITATION: this is causal/frame-local and is ONLY equivalent to the full
// buffer when normalize="NA" (no per-feature normalization), which is the case
// for the only streaming model (nvidia/parakeet_realtime_eou_120m-v1). For
// normalize="per_feature" the incremental causal mel is NOT equivalent (it needs
// whole-utterance mean/std), so the constructor asserts NA and callers must fall
// back to MelFrontend::compute for per_feature models.
class StreamingMel {
public:
    explicit StreamingMel(const ModelLoader& ml);

    // True iff this model's normalization allows incremental (frame-local) mel.
    // (false for per_feature, which needs whole-utterance stats.)
    bool incremental_ok() const { return !k_.per_feature_; }

    int n_mels() const { return k_.n_mels_; }

    // Feed `n` new 16 kHz mono samples. Returns the newly-ready mel frames as
    // row-major [n_mels, n_new_frames] (feat-major, inner = frame); n_new_frames
    // is also written to the out-param. Does NOT apply end zero-padding (that
    // would make near-end frames non-causal); call finalize() for the tail.
    std::vector<float> feed(const float* pcm, int n, int& n_new_frames);

    // Emit the end-padded tail frames (the right n_fft/2 zero-pad), completing
    // the stream so the concatenation of all feed() + finalize() frames equals
    // MelFrontend::compute on the full PCM. Returns [n_mels, n_tail_frames].
    std::vector<float> finalize(int& n_tail_frames);

    // Reset to a fresh stream (clears history + counters).
    void reset();

private:
    // Reorder the frame-contiguous columns from build_frames (n frames, each of
    // length n_mels) into the [n_mels, n] feat-major layout (out[m*n + t]).
    std::vector<float> to_feat_major(const std::vector<float>& col_major, int n) const;

    MelKernel k_;
    int pad_;                       // n_fft / 2

    long long samples_seen_ = 0;    // total RAW samples fed so far
    int emitted_ = 0;               // frames already emitted (start-pad aware)
    bool have_prev_ = false;        // is prev_raw_ valid (i.e. seen >= 1 sample)?
    float prev_raw_ = 0.0f;         // last RAW sample, for cross-feed preemph

    // Retained PREEMPHASIZED samples (double, exactly like MelFrontend's
    // `padded`). buf_[k] is the preemph sample at ORIGINAL index
    // (buf_origin_ + k). We only keep samples still needed by a future frame,
    // so this is bounded by ~n_fft + hop, NOT the stream length.
    std::vector<double> buf_;
    long long buf_origin_ = 0;      // original preemph index of buf_[0]

    // Per-frame scratch (reused across frames to avoid allocation).
    std::vector<double> framebuf_;  // [n_fft] preemph samples for one frame
    std::vector<float> wframe_;     // [n_fft] windowed (float) frame
    std::vector<float> re_, im_;
    std::vector<double> power_;
};

} // namespace pk
