#pragma once
#include "model_loader.hpp"
#include "streaming_encoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "rnnt.hpp"
#include "prompt_kernel.hpp"
#include "decode_types.hpp"
#include "transcription.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace pk {

// End-of-utterance / backchannel event emitted by the streaming decoder.
//
// The model nvidia/parakeet_realtime_eou_120m-v1 emits <EOU> (id 1024) and
// <EOB> (id 1025) as regular vocab tokens to mark end-of-utterance /
// backchannel. We surface them as events (stripped from the running text) with
// the encoder frame on which they were emitted and a wall-clock time:
//   time_sec = encoder_frame * hop_length * subsampling_factor / sample_rate
// (the subsampled encoder frame stride in seconds), matching NeMo's
// _get_eou_predictions_from_hypotheses timestamp scaling.
struct EouEvent {
    int32_t token = 0;             // the special token id (<EOU> or <EOB>)
    bool    is_eob = false;        // true for <EOB>, false for <EOU>
    int     encoder_frame = 0;     // encoder-output frame index of the emission
    double  time_sec = 0.0;        // encoder_frame * hop * subsampling / sample_rate
};

// Cache-aware streaming RNN-T session for the pure-RNNT streaming model
// nvidia/parakeet_realtime_eou_120m-v1.
//
// Owns a StreamingEncoder (carries per-layer conv + attention caches across
// chunks) AND the RNN-T greedy decoder state (prediction-net LSTM state, last
// emitted token, accumulated hypothesis). Per chunk:
//   1. StreamingEncoder::step(mel_chunk) -> the chunk's VALID encoder frames
//      [valid, d_model] row-major (the cache-aware-equivalent leading frames).
//   2. rnnt_decode_frames over those new frames, carrying the decoder state
//      (NOT reset between chunks). Newly emitted token ids are appended to the
//      session hypothesis and returned.
//
// Mirrors NeMo's rnnt_decoder_predictions_tensor(..., partial_hypotheses=...):
// the decoder state persists, so feeding the encoder frames in chunks produces
// the identical token sequence as decoding the whole offline encoder output at
// once.
//
// This is the decoder-facing surface only — PCM buffering / the C-API / EOU
// events are Task 7. The caller drives the encoder chunk schedule (mel windows
// already including pre-encode-cache overlap, matching test_streaming_encoder).
class StreamingSession {
public:
    // `target_lang` selects the language prompt for multilingual (nemotron)
    // prompt-conditioned models (e.g. "en", "de", "auto"); empty -> the model's
    // default_lang. It is ignored by non-prompt models (prompt_.present()==false).
    // For a prompt model an unknown locale THROWS std::runtime_error (matching the
    // offline Model::resolve_prompt_index and the C-API stream_begin_lang
    // contract), so a typo fails loudly rather than silently mis-transcribing.
    explicit StreamingSession(const ModelLoader& ml, const std::string& target_lang = "");

    // Reset the encoder caches AND the decoder state to a fresh stream.
    void reset();

    // Feed one mel chunk window (row-major [n_mels, n_frames], feat-major
    // inner=time — the same orientation StreamingEncoder::step expects, already
    // including any pre-encode-cache overlap). Runs the encoder step then the
    // RNN-T decode over the new encoder frames. Returns the token ids emitted in
    // THIS chunk (and accumulates them into the session hypothesis tokens()).
    // Also updates the running stripped text() and the EOU event queue.
    //
    // is_last selects the encoder keep_all_outputs path (final chunk keeps the
    // streaming tail). Defaults to false for mid-stream chunks.
    //
    // NOTE: this returns token ids (test_streaming_decode relies on that). The
    // C-API exposes the newly-finalized *text* via take_new_text() below; the
    // text/events surfaces are kept separate so the token-level parity test and
    // the text-level streaming API can both be served from one feed call.
    std::vector<int32_t> feed_mel_chunk(const std::vector<float>& mel_chunk,
                                        int n_frames, bool is_last = false);

    // Flush the end-of-stream tail. Mirrors NeMo's final-chunk keep_all_outputs:
    // re-feeds the LAST buffered mel chunk (if the caller did not already mark it
    // is_last) so the trailing encoder frames complete, decoding any remaining
    // tokens. Returns the newly-finalized text produced by the flush ("" if
    // none). It does NOT fabricate an <EOU> NeMo's streaming wouldn't emit: for
    // a clip whose final-chunk tail has incomplete right context (parakeet
    // streaming drops the trailing <EOU>), finalize emits no extra event — it
    // only commits whatever the carried decoder state already produced.
    std::string finalize();

    // The full accumulated emitted token-id sequence across all fed chunks
    // (includes <EOU>/<EOB> ids if any were emitted).
    const std::vector<int32_t>& tokens() const { return state_.hyp; }

    // Running transcript with <EOU>/<EOB> STRIPPED (specials surface as events,
    // not text), detokenized from the non-special emitted tokens.
    const std::string& text() const { return text_; }

    // The text appended since the previous take_new_text()/feed call — the
    // "newly-finalized text". Resets the delta marker. Returns "" if none.
    std::string take_new_text();

    // True iff the most recent feed_mel_chunk emitted an <EOU>/<EOB> event.
    bool last_chunk_had_eou() const { return last_chunk_had_eou_; }

    // Move out all EOU/EOB events collected so far (drains the queue).
    std::vector<EouEvent> drain_events();

    // Move out the WORDS finalized since the previous drain_words() call, with
    // per-word start/end (seconds) and 'min'-aggregate confidence (matching the
    // offline pk::group_words / NeMo timestamps=True convention). A word is
    // finalized when the NEXT `▁`-token (word-start marker) arrives; the final
    // open word is flushed by finalize(). EOU/EOB specials are NOT words (they
    // surface via drain_events()). Drains the returned-words queue.
    std::vector<Word> drain_words();

    // Streaming chunk schedule (delegated from the encoder), so the caller can
    // window the mel exactly like test_streaming_encoder.
    int chunk_size_first() const { return enc_.chunk_size_first(); }
    int chunk_size() const { return enc_.chunk_size(); }
    int pre_encode_cache_size() const { return enc_.pre_encode_cache_size(); }
    int valid_out_len() const { return enc_.valid_out_len(); }

private:
    // Recompute text_ from the non-special tokens in state_.hyp and append any
    // brand-new EOU/EOB events found beyond events_seen_tokens_.
    void process_emitted(const std::vector<int32_t>& emitted);

    const ModelLoader& ml_;
    StreamingEncoder enc_;
    PredictionNet pred_;
    Joint joint_;
    PromptKernel prompt_;          // post-encoder language conditioning (nemotron)
    int prompt_index_ = -1;        // resolved language prompt index (-1 if absent)
    int d_model_;
    int blank_id_;
    int max_symbols_;
    RnntDecodeState state_;

    // EOU/EOB token ids (resolved from the tokenizer pieces; -1 if absent).
    int eou_id_ = -1;
    int eob_id_ = -1;
    double frame_sec_ = 0.0;       // hop * subsampling / sample_rate (per enc frame)
    int enc_frame_ = 0;            // running encoder-output frame counter

    // Running stripped transcript + delta marker.
    std::vector<int32_t> non_special_;  // non-special tokens, for detokenize
    std::string text_;
    size_t text_taken_ = 0;        // byte offset of text_ already returned

    bool last_chunk_had_eou_ = false;
    std::vector<EouEvent> events_;

    // Per-word timestamp accumulation. `word_tokens_` holds the per-token
    // TokenInfo (absolute frame, conf, span) for the NON-SPECIAL tokens emitted
    // so far, in emission order — the same input pk::group_words consumes
    // offline. We regroup the whole accumulated sequence after each chunk and
    // surface words that are now FINAL (followed by a later word-start, so their
    // text + end-offset can't change). `words_finalized_` counts how many of the
    // regrouped words are already considered final; `words_taken_` how many have
    // been handed out by drain_words(). finalize() flushes the trailing word.
    std::vector<TokenInfo> word_tokens_;
    std::vector<Word> words_;       // last regrouping of word_tokens_
    size_t words_finalized_ = 0;    // # of words_ that are final (safe to emit)
    size_t words_taken_ = 0;        // # of words already returned by drain_words()
    float frame_sec_f_ = 0.0f;      // frame_sec as float (group_words uses float)

    // Regroup word_tokens_ into words_ and advance words_finalized_ to all but
    // the last (still-open) word — flush_all=true (finalize) makes every word
    // final, including the trailing one.
    void regroup_words(bool flush_all);
};

// Drive a StreamingSession over a whole 16 kHz mono PCM clip in the model's
// exact cache-aware chunk schedule (the same windowing as test_streaming_decode:
// chunk 0 = chunk_size_first mel frames with no overlap; each later chunk =
// pre_encode_cache_size overlap + chunk_size mel frames; keep_all_outputs only
// on the final chunk). The full-clip mel is computed once (matching NeMo's
// online_normalization=False reference), then fed chunk-by-chunk so the carried
// encoder/decoder caches reproduce NeMo's streaming output EXACTLY.
//
// `on_chunk` (optional) is invoked after each chunk with the newly-finalized
// text (<EOU>/<EOB> stripped), the events emitted in that chunk, and the WORDS
// finalized in that chunk (start/end/conf) — for the CLI's incremental printing.
// The final stripped transcript is sess.text().
//
// This is the authoritative full-clip driver the streaming C-API uses on
// finalize and the CLI uses for --stream. It does not buffer PCM incrementally;
// it processes the supplied clip in one pass (the encoder/decoder are still
// driven chunk-by-chunk with carried state — the streaming numerics, not a
// real-time PCM feeder).
// `target_lang` is currently unused by the driver itself (the session already
// owns its resolved prompt index from construction); it is accepted so callers
// (C-API/CLI, Phase 4) can pass a language through one entry point. Defaults to
// "" (the session's configured language).
void run_stream_over_pcm(
    StreamingSession& sess, const ModelLoader& ml,
    const std::vector<float>& pcm16k,
    const std::function<void(const std::string& new_text,
                             const std::vector<EouEvent>& chunk_events,
                             const std::vector<Word>& chunk_words)>& on_chunk
        = nullptr,
    const std::string& target_lang = "");

} // namespace pk
