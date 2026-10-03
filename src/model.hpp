#pragma once
#include "parakeet.h"          // pk::Decoder
#include "joint.hpp"
#include "model_loader.hpp"
#include "prediction.hpp"
#include "tdt.hpp"             // pk::TdtBeamToken
#include "transcription.hpp"   // pk::Transcription
#include "vad_head.hpp"
#include "vad_segmenter.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pk {

class BackendPool;

// One text hypothesis returned by the opt-in offline TDT N-best path.
struct NBestTranscription {
    std::string text;
    std::vector<TdtBeamToken> tokens;
    float score = 0.0f;
    float normalized_score = 0.0f;
};

// Load-once transcription context.
//
// Loads a GGUF model ONCE (owns the ModelLoader) and reuses it across many
// transcribe calls — in contrast to the free function pk::transcribe(), which
// reloads the model on every call. This is what the flat C-API holds.
//
// The component objects (MelFrontend, Encoder, PredictionNet, Joint, ...) are
// lightweight views over the ModelLoader (they hold `const ModelLoader&`), so
// they are constructed per call, except the transducer decoder objects
// (PredictionNet, Joint), which are built once on first use and shared by every
// decode (they are immutable after construction; see decoder_objects()). The
// expensive part — parsing the GGUF and
// mapping every weight tensor — happens exactly once, in load().
class Model {
public:
    // Loads the GGUF at `gguf_path`. Returns nullptr on failure (no throw).
    // With a non-empty `component`, `gguf_path` is a bundle GGUF (docs/bundle.md)
    // and only that component is read; the model is then the same as the one
    // loaded from the component's single-model file. A plain path without a
    // component is the unchanged single-model load; a bundle without a
    // component is refused.
    static std::unique_ptr<Model> load(const std::string& gguf_path, const std::string& component = "");

    // Transcribe raw mono float PCM. If `sample_rate != 16000` the audio is
    // linearly resampled to 16 kHz (via pk::resample_linear) before inference.
    // `target_lang` selects the language prompt for multilingual (nemotron)
    // models (e.g. "en", "de", "auto"); empty -> the model default. It is
    // ignored by non-prompt models. Throws std::runtime_error on failure (e.g.
    // unsupported arch, or an unknown target_lang for a prompt model).
    std::string transcribe_pcm(const std::vector<float>& pcm, int sample_rate,
                               Decoder decoder = Decoder::kDefault,
                               const std::string& target_lang = "") const;

    // Transcribe a WAV file (loaded + resampled to 16 kHz mono via
    // pk::load_audio_16k_mono). `target_lang` as in transcribe_pcm. Throws
    // std::runtime_error on failure.
    std::string transcribe_path(const std::string& wav_path,
                                Decoder decoder = Decoder::kDefault,
                                const std::string& target_lang = "") const;

    // Core orchestration: 16 kHz mono PCM -> transcript. Public so language-aware
    // callers/tests can drive it directly with a resolved target_lang.
    std::string transcribe_16k(const std::vector<float>& pcm16k,
                               Decoder decoder = Decoder::kDefault,
                               const std::string& target_lang = "") const;

    // Resolve a target_lang (locale string) to a prompt index using the model's
    // dictionary. Empty string -> the model's default_lang. Returns -1 and is
    // ignored when the model is not prompt-conditioned. Throws std::runtime_error
    // on an unknown locale for a prompt model (message lists a few valid keys).
    int resolve_prompt_index(const std::string& target_lang) const;

    // Transcribe a batch of mono float PCM clips. Each is resampled to 16 kHz if
    // needed, then all run through the batched encoder; decode is per item.
    // Returns one transcript per input, in order.
    std::vector<std::string> transcribe_pcm_batch(
        const std::vector<std::vector<float>>& pcms, int sample_rate,
        Decoder decoder = Decoder::kDefault,
        const std::string& target_lang = "") const;

    // Run mel + encoder + CTC head only, returning the log-prob matrix
    // (row-major [T, vocab+1], already log-softmaxed) instead of decoded text —
    // the seam external decoder stacks (e.g. pyctcdecode + KenLM) need. If
    // `sample_rate != 16000` the audio is linearly resampled to 16 kHz first.
    // `target_lang` as in transcribe_pcm (ignored by non-prompt models). Always
    // runs the CTC head regardless of the model's preferred decoder; throws
    // std::runtime_error if the model has no CTC head (e.g. a TDT/RNNT-only
    // streaming model).
    void transcribe_pcm_ctc_logits(const std::vector<float>& pcm, int sample_rate,
                                   std::vector<float>& logits, int& T,
                                   int& vocab_plus_1,
                                   const std::string& target_lang = "") const;

    // Transcribe raw mono float PCM, returning the flat text plus per-word and
    // per-token timestamps + confidence (matching NeMo timestamps=True +
    // 'max_prob' confidence). If `sample_rate != 16000` the audio is linearly
    // resampled to 16 kHz first. Throws std::runtime_error on failure.
    Transcription transcribe_with_timestamps(
        const std::vector<float>& pcm, int sample_rate,
        Decoder decoder = Decoder::kDefault,
        const std::string& target_lang = "") const;

    // Convenience: transcribe a WAV file with timestamps + confidence.
    Transcription transcribe_path_with_timestamps(
        const std::string& wav_path,
        Decoder decoder = Decoder::kDefault,
        const std::string& target_lang = "") const;

    // Offline TDT beam search. These methods are intentionally separate from
    // the greedy Decoder selector: they require a TDT duration table and return
    // up to `nbest` ranked hypotheses with token emission frames/durations.
    std::vector<NBestTranscription> transcribe_pcm_nbest(
        const std::vector<float>& pcm, int sample_rate,
        int beam_size, int nbest, bool score_norm = true,
        const std::string& target_lang = "") const;

    std::vector<NBestTranscription> transcribe_path_nbest(
        const std::string& wav_path,
        int beam_size, int nbest, bool score_norm = true,
        const std::string& target_lang = "") const;

    // Batched timestamped transcription. Each clip is resampled to 16 kHz if
    // needed, all run through the batched encoder; decode + timestamp extraction
    // are per item. Returns one Transcription per input, in order.
    std::vector<Transcription> transcribe_pcm_batch_with_timestamps(
        const std::vector<std::vector<float>>& pcms, int sample_rate,
        Decoder decoder = Decoder::kDefault,
        const std::string& target_lang = "") const;

    const ParakeetConfig& config() const { return loader_.config(); }
    // Per-frame speech probability from the model's own VAD head (80 ms frames,
    // cfg.vad.frame_sec). `v` overrides the head wiring for experiments; nullptr
    // uses the defaults. Throws std::runtime_error if the model has no VAD head.
    std::vector<float> vad_probabilities(const std::vector<float>& pcm16k,
                                         const VadVariant* v = nullptr) const;

    // Transcribe long audio in VAD-cut segments (see vad_segmenter.hpp). Audio no
    // longer than opts.max_seg_sec takes the plain path. Longer audio is cut at
    // pauses and segments without speech are dropped, so audio with no speech
    // gives an empty transcript. Requires a model with a
    // VAD head (throws std::runtime_error("model has no VAD head") otherwise).
    //
    // `external_vad` (optional) replaces the model's own VAD head, so a model
    // without a head can use another detector such as Silero: it maps 16 kHz mono
    // PCM to one speech probability per opts.frame_sec seconds, and opts.frame_sec
    // must then be set to that period. The segmenter, the 30 s cap and the decode
    // of each segment are the same as with the head.
    using VadProbabilityFn = std::function<std::vector<float>(const std::vector<float>&)>;
    std::string transcribe_pcm_vad(const std::vector<float>& pcm, int sample_rate,
                                   Decoder decoder = Decoder::kDefault,
                                   const std::string& target_lang = "",
                                   const SegmenterOpts& opts = SegmenterOpts(),
                                   const VadProbabilityFn* external_vad = nullptr) const;
    Transcription transcribe_pcm_vad_with_timestamps(
        const std::vector<float>& pcm, int sample_rate,
        Decoder decoder = Decoder::kDefault, const std::string& target_lang = "",
        const SegmenterOpts& opts = SegmenterOpts(),
        const VadProbabilityFn* external_vad = nullptr) const;

    // The underlying loaded GGUF. Exposed so the streaming C-API can build a
    // pk::StreamingSession (and a MelFrontend) over the same load-once model.
    const ModelLoader& loader() const { return loader_; }

    // Concurrent requests (opt-in). With `backends` > 1 the model keeps a pool
    // of that many CPU backends, each with `threads_each` ggml threads
    // (<= 0: the default thread budget divided by `backends`, at least 1). Every
    // public transcribe method then borrows one backend for the whole call, so
    // up to `backends` calls from different threads run in parallel and the
    // results are the same as with one backend. The library starts no threads
    // of its own. `backends` x `threads_each` should not exceed the physical
    // cores. With `backends` <= 1 the pool is removed and the process-global
    // backend (and `pk::set_num_threads`) is used, which is the default.
    // A GPU device keeps one backend. Returns the effective backend count.
    // Safe to call at any time; calls already running finish on the old pool.
    int set_concurrency(int backends, int threads_each = 0);
    int concurrency() const;            // effective backends (1 = no pool)
    int threads_per_backend() const;    // 0 when there is no pool
    // Graph allocator memory held by the pool's backends, in bytes (0 without
    // a pool). Grows to the largest graph each backend has run.
    size_t pool_working_set_bytes() const;

    // Non-copyable (owns the GGUF mapping).
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    Model() = default;

    // Core batched orchestration: N 16 kHz clips -> N transcripts. Stacks mels,
    // runs forward_batch, decodes each item with the existing greedy decoders.
    std::vector<std::string> transcribe_16k_batch(
        const std::vector<std::vector<float>>& pcms16k, Decoder decoder,
        const std::string& target_lang = "") const;

    // Core batched timestamped orchestration: N 16 kHz clips -> N Transcriptions.
    std::vector<Transcription> transcribe_16k_batch_with_timestamps(
        const std::vector<std::vector<float>>& pcms16k, Decoder decoder,
        const std::string& target_lang = "") const;

    // Core orchestration for the timestamps path: 16 kHz mono PCM -> full
    // Transcription (text + per-token TokenInfo + grouped words). Shared by the
    // two timestamp entry points.
    Transcription transcribe_16k_with_timestamps(
        const std::vector<float>& pcm16k, Decoder decoder,
        const std::string& target_lang = "") const;

    // VAD path: encode each 16 kHz clip on its own, decode them in groups of up
    // to kVadDecodeGroup with the exact batched decode (TDT and RNNT; CTC decodes
    // per item). Each result equals transcribe_16k / transcribe_16k_with_timestamps
    // of that clip alone, bit for bit on CPU. `with_timestamps` false fills only
    // `text`. Frame and time offsets are the caller's job.
    std::vector<Transcription> transcribe_16k_grouped(
        const std::vector<const std::vector<float>*>& pcms16k, Decoder decoder,
        const std::string& target_lang, bool with_timestamps) const;

    std::vector<NBestTranscription> transcribe_16k_nbest(
        const std::vector<float>& pcm16k, int beam_size, int nbest,
        bool score_norm, const std::string& target_lang) const;

    // Core orchestration for transcribe_pcm_ctc_logits: 16 kHz mono PCM -> CTC
    // log-prob matrix. Mirrors transcribe_16k through the encoder, then runs
    // the CTC head directly instead of decode_enc_out.
    void transcribe_16k_ctc_logits(const std::vector<float>& pcm16k,
                                   std::vector<float>& logits, int& T,
                                   int& vocab_plus_1,
                                   const std::string& target_lang = "") const;

    // The current backend pool, or null for the process-global backend.
    // Requests copy the pointer so a replaced pool outlives them.
    std::shared_ptr<BackendPool> pool_snapshot() const;

    ModelLoader loader_;

    // The transducer decoder objects, built once per model on first use
    // (thread-safe) so the 21 MB embedding table is not re-copied per
    // utterance. PredictionNet and Joint are read-only after construction
    // (the embedding table is filled once under a std::once_flag), so
    // concurrent decodes may share them. Only valid for transducer models.
    struct DecoderObjects {
        PredictionNet pred;
        Joint         joint;
        explicit DecoderObjects(const ModelLoader& ml) : pred(ml), joint(ml) {}
    };
    const DecoderObjects& decoder_objects() const;

    mutable std::once_flag decoder_once_;
    mutable std::unique_ptr<DecoderObjects> decoder_;

    // Declared last so the pool is destroyed before the loader and decoder
    // objects it computes against.
    mutable std::mutex pool_mu_;
    std::shared_ptr<BackendPool> pool_;
};

} // namespace pk
