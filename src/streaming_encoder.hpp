#pragma once
#include "model_loader.hpp"
#include <map>
#include <vector>
namespace pk {

// Cache-aware streaming FastConformer encoder (NeMo ConformerEncoder cache-aware
// streaming path: forward_internal 617-779 + cache_aware_stream_step + the
// CacheAwareStreamingAudioBuffer chunk loop). Processes the mel spectrogram
// chunk-by-chunk, carrying per-layer convolution and attention caches across
// chunks so that the concatenated per-chunk valid output equals the OFFLINE
// encoder output (cache-aware equivalence), except the trailing valid_out_len
// frames of the final chunk (the streaming tail, whose right context is
// incomplete by design).
//
// Per-layer caches (NeMo get_initial_cache_state = zeros):
//   cache_last_time    [n_layers] each [d_model, left_pad=conv_kernel-1] — the
//                      causal depthwise conv left-context (prepended before the
//                      depthwise conv; next cache = last left_pad columns of the
//                      cache+current conv input).
//   cache_last_channel [n_layers] each [last_channel_cache_size=70, d_model] —
//                      the attention key/value left-context (prepended to K/V;
//                      next cache = drop oldest Tc, append the Tc current query
//                      frames, keep the last 70).
//   cache_last_channel_len   scalar — number of *valid* (non-empty) frames in
//                      cache_last_channel (grows 0,2,4,..,70); the not-yet-filled
//                      leading cache columns are masked out of attention.
//
// The pre-encode mel cache is realized by the caller-supplied chunk windows:
// each step (after the first) prepends pre_encode_cache_size mel frames of
// overlap from the previous window. StreamingEncoder::step takes the FULL mel
// window already including that overlap (matching NeMo's audio_chunk), runs the
// subsampling, then drops drop_extra_pre_encoded leading subsampled frames.
class StreamingEncoder {
public:
    explicit StreamingEncoder(const ModelLoader& ml);

    // Reset all caches to zeros and the step counter to 0 (a fresh stream).
    void reset();

    // Process one mel chunk window. `mel_chunk_frames` is row-major
    // [n_mels, n_mel_frames] (feat-major inner = time), i.e. mel[m*n + t] — the
    // same orientation as the offline encoder's mel input. The window must
    // already include any pre-encode-cache overlap (NeMo's audio_chunk). Returns
    // the chunk's VALID encoder frames as row-major [valid, d_model] (d_model
    // fastest), i.e. out[t*d_model + c]. `valid` is reported via n_valid_out.
    //
    // is_last selects keep_all_outputs: when false (mid-stream) the output is
    // sliced to valid_out_len frames; when true (final chunk) all produced
    // frames are kept (the trailing streaming tail). Mirrors NeMo
    // streaming_post_process + cache_aware_stream_step keep_all_outputs.
    std::vector<float> step(const std::vector<float>& mel_chunk_frames,
                            int n_mel_frames, bool is_last, int& n_valid_out);

    // Convenience overload (mid-stream, discards the valid count).
    std::vector<float> step(const std::vector<float>& mel_chunk_frames,
                            int n_mel_frames) {
        int v = 0;
        return step(mel_chunk_frames, n_mel_frames, /*is_last*/false, v);
    }

    // Streaming schedule, read from the GGUF streaming KV (see StreamingCfg).
    int chunk_size_first() const { return chunk_first_; }  // mel frames, step 0
    int chunk_size() const { return chunk_main_; }         // mel frames, step>0
    int pre_encode_cache_size() const { return pre_cache_; }
    int valid_out_len() const { return valid_out_len_; }
    int step_num() const { return step_; }

private:
    const ModelLoader& ml_;
    int d_model_;
    int n_layers_;
    int n_heads_;
    int d_head_;
    int conv_kernel_;
    int left_pad_;            // conv left-context = conv_kernel - 1 (causal)
    bool xscaling_;
    // streaming schedule
    int chunk_first_;
    int chunk_main_;
    int pre_cache_;
    int drop_extra_;
    int last_channel_cache_;  // 70
    int valid_out_len_;
    // chunked-limited attention
    int att_left_;
    int att_right_;
    // caches
    int step_ = 0;
    int clc_len_ = 0;         // cache_last_channel_len (valid frames in attn cache)
    // cache_last_time[layer]: row-major [left_pad, d_model] (time-major; t*d_model+c)
    std::vector<std::vector<float>> cache_time_;
    // cache_last_channel[layer]: row-major [last_channel_cache, d_model]
    std::vector<std::vector<float>> cache_channel_;
    // linear_pos(pos_emb) split into heads, [layer] -> [dk, pos_len, H], per window
    // length Tc + cache_len (see pos_heads()).
    std::map<int, std::vector<std::vector<float>>> pos_heads_;
    const std::vector<std::vector<float>>& pos_heads(int window);
};

} // namespace pk
