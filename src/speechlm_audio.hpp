#pragma once
#include "model_loader.hpp"
#include "mel.hpp"
#include "diarization_encoder.hpp"
#include "diarization_streaming.hpp"
#include <memory>
#include <string>
#include <vector>

namespace pk {

// Intermediate tensors of SpeechLmAudio::embed_pcm, for parity tests. All are
// time-major row-major ([n_frames, width], x[t*width + c]).
struct SpeechLmAudioStages {
    int n_frames = 0;                    // audio tokens = ceil(valid mel frames / 8)
    int n_valid_mel = 0;                 // floor(samples / hop) mel frames
    std::vector<float> mel;              // [n_valid_mel, n_mels] raw log-mel
    std::vector<float> asr_hidden;       // [n_frames, d_model] chunked encoder output
    std::vector<float> speaker_probs;    // [n_frames, n_speakers] mean over the 8 sub-frames
    std::vector<float> fused;            // [n_frames, d_model] asr_norm(hidden) + speaker term
};

// SpeechLmAudio - the audio side of a speech LLM (nvidia/NVIDIA-Nemotron-3.5-
// Transcribe-30B-A4B): mel -> 32-layer ASR transformer encoder (run in chunks with
// context) -> Sortformer speaker branch -> speaker fusion -> adapter projection into
// the LLM embedding space. The LLM itself is not part of this class: the result is
// one embedding row per 80 ms of audio.
//
// Two models are needed. The "audio" model (arch speechlm_audio) holds the mel
// constants, the ASR encoder, the fusion norms and the adapter. The "diar" model
// (arch diarization) holds the speaker encoder and head. Each is a standalone GGUF,
// or a component of a bundle GGUF (docs/bundle.md).
//
// Pipeline, as in the reference implementation:
//   1. log-mel of the 16 kHz PCM, trimmed to floor(samples / hop) frames (the
//      centred STFT makes one extra frame). The ASR branch normalises each mel bin
//      over those frames; the speaker branch uses the raw log-mel.
//   2. Both branches see the mel zero-padded to a multiple of 8 frames; a run
//      always has ceil(valid / 8) frames, so a trailing group with no valid frame
//      does not exist.
//   3. ASR encoder: windows of chunk_len frames, each with left and right context,
//      positions restarting at 0 in every window; only the window's own frames
//      are kept.
//   4. Speaker branch: StreamingDiarization in offline mode (speaker cache, FIFO,
//      chunks with context). Probabilities are averaged over the 8 sub-frames of
//      every encoder frame and thresholded.
//   5. fused = LayerNorm_asr(hidden) + LayerNorm_spk(activity) @ S, with S a fixed
//      sinusoidal table [n_speakers, d_model]; then out = proj(fused).
class SpeechLmAudio {
public:
    // `*_component` is empty for a standalone GGUF, else the component of the
    // bundle at the same path. Returns nullptr on failure and writes a reason to *err.
    static std::unique_ptr<SpeechLmAudio> load(const std::string& audio_path,
                                               const std::string& audio_component,
                                               const std::string& diar_path,
                                               const std::string& diar_component,
                                               std::string* err = nullptr);

    // 16 kHz mono PCM -> out [n_tokens, dim()]. n_tokens = ceil(floor(n/hop) / 8).
    // Throws std::runtime_error on a graph failure.
    void embed_pcm(const std::vector<float>& pcm16k, std::vector<float>& out, int& n_tokens,
                   SpeechLmAudioStages* stages = nullptr) const;

    int dim() const { return (int)audio_.config().speechlm.llm_dim; }          // adapter width
    int hidden() const { return (int)audio_.config().d_model; }                // encoder width
    int num_speakers() const { return (int)audio_.config().speechlm.num_speakers; }
    int tokens_for_samples(long long n_samples) const;
    const ParakeetConfig& config() const { return audio_.config(); }

private:
    SpeechLmAudio() = default;
    int hidden_dim() const { return (int)audio_.config().d_model; }
    void asr_chunks(const std::vector<float>& mel_norm, int T_padded, std::vector<float>& hidden) const;
    void speaker_activity(const std::vector<float>& mel_raw, int T_padded, int n_frames,
                          std::vector<float>& probs80) const;
    void fuse_and_project(const std::vector<float>& hidden, const std::vector<float>& activity,
                          int n_frames, std::vector<float>& fused, std::vector<float>& out) const;

    ModelLoader audio_;
    ModelLoader diar_;
    std::unique_ptr<MelFrontend> mel_;
    std::unique_ptr<DiarizationEncoder> encoder_;
    int subsampling_ = 8;
};

}  // namespace pk
