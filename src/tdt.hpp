#pragma once
#include "prediction.hpp"
#include "joint.hpp"
#include "decode_types.hpp"
#include <vector>
#include <cstdint>

namespace pk {

// One non-blank token emitted by TDT beam search.
//
// `frame` is the encoder frame at which the token was emitted. `duration` is
// the selected TDT duration, so the token's end frame is frame + duration.
struct TdtBeamToken {
    int32_t id;
    int32_t frame;
    int32_t duration;
};

// One completed TDT beam-search hypothesis.
//
// `score` is the accumulated token+duration log probability. NeMo's default
// score normalization divides by the sequence length INCLUDING its leading
// blank/SOS sentinel, hence tokens.size() + 1 here.
struct TdtBeamHypothesis {
    std::vector<TdtBeamToken> tokens;
    float score = 0.0f;
    float normalized_score = 0.0f;
};

// TDT (Token-and-Duration Transducer) duration-aware greedy decoding.
//
// Ports NeMo GreedyTDTInfer._greedy_decode (rnnt_greedy_decoding.py). Drives the
// prediction net + joint frame-by-frame, advancing the time index by the
// predicted duration each inner step.
//
// Algorithm (per the Phase 3 plan / NeMo):
//   t = 0; hyp = []; committed_state = zeros; last_token = SOS
//   while t < T:
//     symbols_added = 0; need_loop = true
//     while need_loop and symbols_added < max_symbols:
//       g, out_state = pred.step(last_label, committed_state)   # SOS on first emit
//       logits = joint(enc[t], g)                               # raw [V_plus]
//       k   = argmax(logits[:vocab+1])        # token (incl. blank)
//       d_k = argmax(logits[vocab+1:])        # duration index
//       skip = durations[d_k]
//       if k != blank:                        # commit ONLY on non-blank
//         hyp.append(k); last_token = k; committed_state = out_state
//       symbols_added += 1; t += skip; need_loop = (skip == 0)
//     if skip == 0: skip = 1                  # infinite-loop guard
//     if symbols_added == max_symbols: t += 1
//
// Argmax is taken over the RAW joint logits — NeMo log_softmaxes the token and
// duration slices separately only for confidence; argmax is invariant under a
// monotonic log_softmax, so greedy needs no softmax.
//
// pred:       stateful prediction net (carries LSTM h,c across steps).
// joint:      RNN-T joint network (called per (t, u) with T=U=1).
// enc:        encoder output, row-major [T, enc_hidden] — enc[t*enc_hidden + c].
// T:          number of encoder time frames.
// enc_hidden: encoder feature dimension (= d_model).
// durations:  the TDT durations table (e.g. [0,1,2,3,4]); skip = durations[d_k].
// blank_id:   blank token id (= vocab_size); token argmax range is [0, vocab+1).
// max_symbols: max symbols emitted per time frame (NeMo default 10).
//
// Returns the emitted token-id sequence (hyp). All emitted ids are < blank_id.
//
// If `tokens` is non-null it is filled (one entry per emitted id, in order) with
// per-token metadata matching NeMo's timestamps=True + 'max_prob' confidence
// (see TokenInfo): frame = the encoder frame t at emission, conf = max_prob over
// the token slice logits[0:vocab+1] (excluding the TDT duration logits), span =
// durations[d_k] (the predicted duration applied to the token). The id-only path
// (tokens == nullptr) is unchanged.
std::vector<int32_t> tdt_greedy(const PredictionNet& pred, const Joint& joint,
                                const std::vector<float>& enc, int T, int enc_hidden,
                                const std::vector<int32_t>& durations,
                                int blank_id, int max_symbols,
                                std::vector<TokenInfo>* tokens = nullptr);

// Sequence-level beam search for TDT models, matching NeMo BeamTDTInfer's
// `default_beam_search`:
//
//   * token and duration slices are log-softmaxed independently;
//   * the best non-blank token-duration pairs are expanded jointly;
//   * blank is expanded only with non-zero durations;
//   * duplicate (token sequence, last frame) paths are merged with logaddexp;
//   * final hypotheses are sorted by normalized score by default.
//
// A zero-duration expansion must strictly reduce the accumulated log score.
// Rejecting non-finite or non-decreasing scores prevents a malformed model from
// creating a non-progress loop without truncating valid hypotheses.
//
// This is an opt-in offline decoder. The existing tdt_greedy path is not used
// or modified. `beam_size` and `nbest` must be positive, with nbest <=
// beam_size. At most `nbest` hypotheses are returned.
std::vector<TdtBeamHypothesis> tdt_beam_search(
    const PredictionNet& pred, const Joint& joint,
    const std::vector<float>& enc, int T, int enc_hidden,
    const std::vector<int32_t>& durations, int blank_id,
    int beam_size, int nbest, bool score_norm = true);

namespace detail {

// Returns the index of the highest-scoring positive duration, or -1 when no
// positive duration exists. Kept in the internal TDT header so the beam-size-1
// blank fallback can be tested without loading a model.
int best_positive_duration_index(
    const std::vector<float>& scores,
    const std::vector<int32_t>& durations);

} // namespace detail

} // namespace pk
