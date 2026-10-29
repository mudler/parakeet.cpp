#pragma once
#include "mel.hpp"            // reuse MelKernel for config + window_ + fb_
#include <memory>
#include <vector>
namespace pk {

// GPU-capable log-mel front end. Reproduces MelFrontend::compute exactly, except
// the per-frame rfft is replaced by a DFT expressed as a matmul so the heavy
// STFT/power/filterbank/log runs as ONE ggml graph on the selected backend (GPU
// when one is active). The host-side preemphasis/framing/windowing and the
// per-feature normalization are kept bit-for-bit identical to MelFrontend (the
// CPU path is untouched; this is only used when the backend is non-CPU).
//
// Numerics differ from MelFrontend only by the FFT->DFT-matmul substitution
// (f32 matmul vs the float radix-2 rfft), which is a small per-element delta.
struct DftBasis {
    std::vector<float> cos, sin;
};

// The DFT basis for n_fft samples and n_bins bins, [n_bins, n_fft] row-major cos(2*pi*b*n/n_fft) and
// -sin(...). Built on the first call for a size; later calls for that size return the same object.
std::shared_ptr<const DftBasis> dft_basis(int n_fft, int n_bins);

// GpuMel's host-side framing: the windowed frame matrix Xw [T, n_fft] row-major (row t = frame t) of the
// preemphasized signal centred in n_fft/2 zeros, with MelFrontend's double arithmetic. T is the frame count.
void gpu_mel_frames(const std::vector<float>& samples, float preemph, const std::vector<float>& window,
                    int n_fft, int hop, int T, std::vector<float>& Xw);

class GpuMel {
public:
    explicit GpuMel(const ModelLoader& ml);
    // Same contract as MelFrontend::compute. feats row-major [n_mels, T]
    // (feat-major, feats[m*T+t]).
    void compute(const std::vector<float>& samples, std::vector<float>& feats,
                 int& n_mels, int& T) const;
private:
    MelKernel k_;                 // window_, fb_, n_fft_, hop_, n_mels_, n_bins_, ...
    // [n_bins, n_fft] row-major cos(2*pi*b*n/n_fft) and -sin(...), built once per size: a GpuMel is made per call.
    std::shared_ptr<const DftBasis> dft_;
};

} // namespace pk
