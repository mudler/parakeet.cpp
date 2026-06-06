#include "streaming.hpp"
#include "tokenizer.hpp"
#include "mel.hpp"
#include <algorithm>
#include <cassert>

namespace pk {

StreamingSession::StreamingSession(const ModelLoader& ml, const std::string& target_lang)
    : ml_(ml), enc_(ml), pred_(ml), joint_(ml), prompt_(ml) {
    const ParakeetConfig& cfg = ml.config();
    d_model_  = (int)cfg.d_model;
    blank_id_ = (int)cfg.blank_id;

    // Resolve the language prompt index for multilingual (nemotron) models. The
    // one-hot is constant over time (one language per utterance), so the prompt
    // projection is applied per chunk in feed_mel_chunk using this fixed index.
    // Non-prompt models leave prompt_index_ = -1 and skip prompt_.apply().
    if (cfg.prompt.present) {
        // Empty target_lang -> the model default; an unknown locale THROWS
        // std::runtime_error (same message as Model::resolve_prompt_index), so a
        // typo (e.g. --lang xx) fails loudly instead of silently mis-transcribing.
        // Matches the offline path and the parakeet_capi_stream_begin_lang
        // contract (NULL + ctx last_error on an unknown locale).
        prompt_index_ = cfg.prompt.resolve_index_or_throw(target_lang);
    }
    // Greedy max symbols per frame, from model metadata (NeMo default 10);
    // matches the offline pk::transcribe path in model.cpp.
    max_symbols_ = (int)cfg.max_symbols;
    assert(joint_.num_durations() == 0 && "StreamingSession is RNN-T only (no TDT durations)");

    // Resolve the <EOU>/<EOB> special token ids from the tokenizer pieces (do
    // NOT hardcode 1024/1025; read them from the loaded vocab).
    const auto& pieces = cfg.tokenizer_pieces;
    for (int i = 0; i < (int)pieces.size(); ++i) {
        if (pieces[i] == "<EOU>") eou_id_ = i;
        else if (pieces[i] == "<EOB>") eob_id_ = i;
    }

    // Per-encoder-frame stride in seconds: each encoder output frame spans
    // hop_length * subsampling_factor input samples (the subsampling downsamples
    // the mel-frame time axis by subsampling_factor). time = frame * stride.
    const double hop = (double)cfg.hop_length;
    const double sub = (double)(cfg.subsampling_factor ? cfg.subsampling_factor : 1);
    const double sr  = (double)(cfg.sample_rate ? cfg.sample_rate : 16000);
    frame_sec_   = (hop * sub) / sr;
    frame_sec_f_ = (float)frame_sec_;

    reset();
}

void StreamingSession::reset() {
    enc_.reset();
    state_ = rnnt_decode_init(pred_);
    // eou_id_/eob_id_/frame_sec_ are resolved once in the constructor.
    enc_frame_ = 0;
    non_special_.clear();
    text_.clear();
    text_taken_ = 0;
    last_chunk_had_eou_ = false;
    events_.clear();
    word_tokens_.clear();
    words_.clear();
    words_finalized_ = 0;
    words_taken_ = 0;
}

void StreamingSession::process_emitted(const std::vector<int32_t>& emitted) {
    last_chunk_had_eou_ = false;
    bool text_changed = false;
    for (int32_t tok : emitted) {
        if (tok == eou_id_ || tok == eob_id_) {
            // EOU/EOB: surface as an event, do NOT add to the text.
            EouEvent ev;
            ev.token = tok;
            ev.is_eob = (tok == eob_id_);
            // The frame index is recorded into events_ by feed/finalize (which
            // knows the per-token frame); we patch it there. Here we only know
            // the token, so push a placeholder the caller will overwrite.
            ev.encoder_frame = enc_frame_;  // refined by caller via per-token frame
            ev.time_sec = ev.encoder_frame * frame_sec_;
            events_.push_back(ev);
            last_chunk_had_eou_ = true;
        } else {
            non_special_.push_back(tok);
            text_changed = true;
        }
    }
    if (text_changed) {
        text_ = detokenize(ml_.config().tokenizer_pieces, non_special_);
    }
}

std::vector<int32_t> StreamingSession::feed_mel_chunk(const std::vector<float>& mel_chunk,
                                                      int n_frames, bool is_last) {
    // 1. Encoder step: the chunk's valid encoder frames, row-major [valid, d_model]
    //    (d_model fastest) — exactly the orientation rnnt_decode_frames expects.
    int n_valid = 0;
    std::vector<float> enc_frames = enc_.step(mel_chunk, n_frames, is_last, n_valid);

    if (n_valid <= 0) {
        last_chunk_had_eou_ = false;
        return {};
    }

    // 1b. Prompt conditioning (nemotron multilingual): project the chunk's
    //     encoder frames through prompt_kernel for the resolved language before
    //     the RNN-T decode. The one-hot is constant over time, so applying it
    //     per chunk is exact (== the offline forward's single application).
    //     prompt_.apply() wants channels-first [d_model, valid]; enc_.step gives
    //     time-major [valid, d_model], so transpose in, apply, transpose back.
    //     No-op for non-prompt models (prompt_.present()==false): enc_frames is
    //     left byte-identical.
    if (prompt_.present()) {
        std::vector<float> chunk_cf((size_t)d_model_ * n_valid);  // [d_model, valid]
        for (int t = 0; t < n_valid; ++t)
            for (int c = 0; c < d_model_; ++c)
                chunk_cf[(size_t)c * n_valid + t] = enc_frames[(size_t)t * d_model_ + c];
        std::vector<float> projected;
        prompt_.apply(chunk_cf, d_model_, n_valid, prompt_index_, projected);  // [d_model, valid]
        for (int t = 0; t < n_valid; ++t)
            for (int c = 0; c < d_model_; ++c)
                enc_frames[(size_t)t * d_model_ + c] = projected[(size_t)c * n_valid + t];
    }

    // 2. RNN-T greedy over the new encoder frames, carrying the decoder state
    //    across chunks (do NOT reset). Appends to state_.hyp and returns the ids
    //    emitted in this chunk, with their LOCAL frame index in [0, n_valid) and
    //    per-token TokenInfo (LOCAL frame, max_prob conf, span==1).
    const int base_frame = enc_frame_;
    std::vector<int32_t> local_frames;
    std::vector<TokenInfo> chunk_tokens;
    std::vector<int32_t> emitted =
        rnnt_decode_frames(pred_, joint_, enc_frames, n_valid, d_model_,
                           state_, blank_id_, max_symbols_, &local_frames,
                           &chunk_tokens);
    enc_frame_ += n_valid;

    // 3. Update text + EOU events; refine each new event's absolute frame index
    //    from the per-token local frame the decoder reported.
    const size_t prev_events = events_.size();
    process_emitted(emitted);
    // Re-walk emitted to assign the correct absolute frame to each new event,
    // and accumulate NON-special tokens (absolute frame) for word grouping.
    size_t evi = prev_events;
    for (size_t i = 0; i < emitted.size(); ++i) {
        if (emitted[i] == eou_id_ || emitted[i] == eob_id_) {
            const int abs_frame = base_frame + (int)local_frames[i];
            events_[evi].encoder_frame = abs_frame;
            events_[evi].time_sec = abs_frame * frame_sec_;
            ++evi;
        } else {
            TokenInfo ti = chunk_tokens[i];
            ti.frame += base_frame;  // local -> absolute encoder frame
            word_tokens_.push_back(ti);
        }
    }
    // Regroup the accumulated tokens; words before the last (still-open) one are
    // final and become available to drain_words().
    regroup_words(/*flush_all=*/false);
    return emitted;
}

std::string StreamingSession::finalize() {
    // The end-of-stream tail is flushed by the caller feeding the final buffered
    // mel chunk with is_last=true (which keeps the streaming tail frames). At the
    // session level there is no further audio to process here, so finalize just
    // returns whatever newly-finalized text remains since the last take. NeMo's
    // cache-aware streaming behaves identically: the final chunk's incomplete
    // right context means a trailing <EOU> is NOT recovered, so we never
    // fabricate one — finalize emits only the carried-state tokens already
    // decoded.
    //
    // Word side: the end-of-stream has no further word-start markers, so the
    // trailing open word is now final too — regroup with flush_all so it becomes
    // available to drain_words().
    regroup_words(/*flush_all=*/true);
    return take_new_text();
}

void StreamingSession::regroup_words(bool flush_all) {
    // Re-run the validated offline grouping over the whole accumulated
    // non-special token sequence (it does the punctuation lookahead / refinement
    // exactly like the offline transcribe_with_timestamps path). The last word is
    // still "open" (its text/end can change when more tokens arrive), so only
    // words BEFORE it are considered final mid-stream; flush_all makes every word
    // final at end-of-stream.
    words_ = group_words(word_tokens_, ml_.config().tokenizer_pieces, frame_sec_f_);
    if (words_.empty()) {
        words_finalized_ = 0;
    } else {
        words_finalized_ = flush_all ? words_.size() : (words_.size() - 1);
    }
    // Never "un-finalize" a word we've already handed out.
    if (words_finalized_ < words_taken_) words_finalized_ = words_taken_;
}

std::vector<Word> StreamingSession::drain_words() {
    std::vector<Word> out;
    for (size_t i = words_taken_; i < words_finalized_ && i < words_.size(); ++i)
        out.push_back(words_[i]);
    words_taken_ = words_finalized_;
    return out;
}

std::string StreamingSession::take_new_text() {
    if (text_taken_ >= text_.size()) return std::string();
    std::string delta = text_.substr(text_taken_);
    text_taken_ = text_.size();
    return delta;
}

std::vector<EouEvent> StreamingSession::drain_events() {
    std::vector<EouEvent> out;
    out.swap(events_);
    return out;
}

void run_stream_over_pcm(
    StreamingSession& sess, const ModelLoader& ml,
    const std::vector<float>& pcm16k,
    const std::function<void(const std::string&,
                             const std::vector<EouEvent>&,
                             const std::vector<Word>&)>& on_chunk,
    const std::string& target_lang) {
    // target_lang is intentionally unused here: the session already carries its
    // resolved prompt index from construction. The parameter exists so callers
    // can route a language through a single entry point (Phase 4 C-API/CLI).
    (void)target_lang;
    // 1. Full-clip mel [n_mels, T] (feat-major inner=T), matching the offline /
    //    NeMo online_normalization=False reference (normalization over the whole
    //    clip). The streaming numerics come from the carried encoder/decoder
    //    caches, not from chunking the front end.
    MelFrontend mel_fe(ml);
    std::vector<float> mel;
    int n_mels = 0, T = 0;
    mel_fe.compute(pcm16k, mel, n_mels, T);
    if (T <= 0) return;

    const int chunk0     = sess.chunk_size_first();      // 9
    const int chunk_main = sess.chunk_size();            // 16
    const int pre_cache  = sess.pre_encode_cache_size(); // 9

    // mel[:, lo:hi] in feat-major layout.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    int buffer_idx = 0;
    bool first = true;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;  // shift_size == chunk_size here
        const int chunk_hi   = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        const int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        sess.feed_mel_chunk(win, win_frames, is_last);

        if (on_chunk) {
            std::string nt = sess.take_new_text();
            std::vector<EouEvent> ev = sess.drain_events();
            std::vector<Word> wd = sess.drain_words();
            if (!nt.empty() || !ev.empty() || !wd.empty()) on_chunk(nt, ev, wd);
        }

        buffer_idx += shift;
        first = false;
    }
}

} // namespace pk
