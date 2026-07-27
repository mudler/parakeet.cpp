#include "tdt.hpp"
#include "decode_common.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pk {

std::vector<int32_t> tdt_greedy(const PredictionNet& pred, const Joint& joint,
                                const std::vector<float>& enc, int T, int enc_hidden,
                                const std::vector<int32_t>& durations,
                                int blank_id, int max_symbols,
                                std::vector<TokenInfo>* tokens) {
    assert((int)enc.size() == (size_t)T * enc_hidden);
    assert(!durations.empty());

    const int V_plus       = joint.V_plus();
    const int num_dur      = (int)durations.size();
    const int token_count  = V_plus - num_dur;   // vocab + 1 (incl. blank) = 1025
    assert(token_count == joint.vocab_size() + 1);
    assert(num_dur == joint.num_durations());

    std::vector<int32_t> hyp;
    if (tokens) tokens->clear();

    // Committed (non-blank) decoding state and last emitted token.
    PredState committed = pred.zero_state();
    int32_t last_token = -1;      // -1 sentinel: nothing emitted yet -> SOS.
    bool emitted_any = false;

    // Precompute the encoder projection over ALL frames ONCE (one matmul on the
    // persistent backend), reused for every step. The per-step joint below is a
    // tight churn-free graph on the same backend. The old code rebuilt the full
    // joint per (t,u) step (a fresh graph per step — the bulk of the
    // per-utterance graph dispatches).
    std::vector<float> enc_proj;   // row-major [T, joint_hidden]
    joint.precompute_enc_proj(enc, T, enc_hidden, enc_proj);
    const int H = joint.joint_hidden();

    // Scratch reused across inner steps.
    std::vector<float> g;
    PredState out_state;
    std::vector<float> logits;

    // Prediction-net output cache: `g`/`out_state` depend only on the committed
    // (last_token, lstm_state), which change exclusively on an emit (k != blank).
    // Steps that don't emit reuse the cached forward pass instead of recomputing
    // the LSTM. See the RNN-T loop for the full rationale.
    bool g_valid = false;

    int t = 0;
    while (t < T) {
        int symbols_added = 0;
        bool need_loop = true;
        int skip = 0;

        while (need_loop && symbols_added < max_symbols) {
            // Prediction net step from the committed state — only when the cache
            // is stale (first step, or the previous step emitted). SOS until the
            // first emit; otherwise feed the last EMITTED token.
            if (!g_valid) {
                const bool is_sos = !emitted_any;
                const int32_t last_label = emitted_any ? last_token : blank_id;
                pred.step(last_label, is_sos, committed, g, out_state);
                g_valid = true;
            }

            // Joint for (t,u): precomputed enc_proj[t] x g -> raw logits [V_plus].
            // `t` is always in [0, T): the outer loop guards the first inner
            // iteration, and subsequent inner iterations only run when skip==0
            // (t unchanged). A positive skip exits the inner loop. This matches
            // the old per-step `enc[t*enc_hidden]` access exactly.
            assert(t < T && "enc_proj row out of range");
            joint.step_logits(enc_proj.data() + (size_t)t * H,
                              g.data(), (int)g.size(), logits);

            // Split: token logits [0, token_count), duration logits [token_count, V_plus).
            const int k   = decode_argmax(logits.data(), token_count);
            const int d_k = decode_argmax(logits.data() + token_count, num_dur);
            skip = durations[d_k];

            // Commit state + last_token ONLY when k != blank.
            if (k != blank_id) {
                hyp.push_back((int32_t)k);
                if (tokens) {
                    // NeMo per-token metadata (matches GreedyTDTInfer._greedy_decode
                    // + max_prob confidence):
                    //   frame = the encoder frame t at emission (hypothesis.timestamp).
                    //   conf  = max_prob over the TOKEN slice logits[0:vocab+1]
                    //           (NeMo log_softmaxes that slice; exclude the duration
                    //           logits). N = token_count = vocab + 1.
                    //   span  = durations[d_k] (the duration/skip applied to the token).
                    const float conf = decode_max_prob_conf(logits.data(), token_count, k);
                    tokens->push_back(TokenInfo{ (int32_t)k, (int32_t)t, conf,
                                                 (int32_t)skip });
                }
                last_token = (int32_t)k;
                committed = out_state;   // carry the step's new (h', c')
                emitted_any = true;
                g_valid = false;         // committed state advanced -> recompute g
            }
            // else: discard out_state; committed/last_token unchanged (g stays valid).

            symbols_added += 1;
            t += skip;
            need_loop = (skip == 0);
        }

        // Infinite-loop guard: if we exited with duration 0 (blank + dur 0), step
        // forward by one frame anyway.
        if (skip == 0) skip = 1;

        // If we stopped because max_symbols was hit (not because of a positive
        // duration), advance the frame by one to make progress.
        if (symbols_added == max_symbols) t += 1;
    }

    return hyp;
}

namespace detail {

int best_positive_duration_index(
    const std::vector<float>& scores,
    const std::vector<int32_t>& durations) {
    if (scores.size() != durations.size())
        throw std::invalid_argument(
            "best_positive_duration_index: size mismatch");

    int best = -1;
    for (size_t i = 0; i < durations.size(); ++i) {
        if (durations[i] > 0 &&
            (best < 0 || scores[i] > scores[(size_t)best])) {
            best = (int)i;
        }
    }
    return best;
}

} // namespace detail

namespace {

struct BeamState {
    TdtBeamHypothesis hyp;
    PredState dec_state;
    PredState next_state;
    std::vector<float> pred_out;
    int last_frame = 0;
    int32_t last_token = -1;
    bool emitted_any = false;
    bool pred_valid = false;
};

struct RankedIndex {
    float score;
    int index;
};

std::vector<float> log_softmax(const float* logits, int n) {
    float max_logit = logits[0];
    for (int i = 1; i < n; ++i)
        max_logit = std::max(max_logit, logits[i]);

    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += std::exp((double)logits[i] - (double)max_logit);
    const double log_denom = std::log(sum);

    std::vector<float> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = (float)((double)logits[i] - (double)max_logit - log_denom);
    return out;
}

std::vector<RankedIndex> top_k(const float* scores, int n, int k) {
    std::vector<RankedIndex> ranked;
    ranked.reserve(n);
    for (int i = 0; i < n; ++i)
        ranked.push_back(RankedIndex{scores[i], i});

    const auto better = [](const RankedIndex& a, const RankedIndex& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.index < b.index;
    };
    const int kept = std::min(n, k);
    std::partial_sort(ranked.begin(), ranked.begin() + kept, ranked.end(), better);
    ranked.resize(kept);
    return ranked;
}

float log_add_exp(float a, float b) {
    const float hi = std::max(a, b);
    const float lo = std::min(a, b);
    return hi + std::log1p(std::exp(lo - hi));
}

bool same_path(const BeamState& a, const BeamState& b) {
    if (a.last_frame != b.last_frame || a.hyp.tokens.size() != b.hyp.tokens.size())
        return false;
    for (size_t i = 0; i < a.hyp.tokens.size(); ++i)
        if (a.hyp.tokens[i].id != b.hyp.tokens[i].id)
            return false;
    return true;
}

void merge_duplicate_hypotheses(std::vector<BeamState>& hypotheses) {
    std::sort(hypotheses.begin(), hypotheses.end(),
              [](const BeamState& a, const BeamState& b) {
                  return a.hyp.score > b.hyp.score;
              });

    std::vector<BeamState> merged;
    merged.reserve(hypotheses.size());
    for (BeamState& candidate : hypotheses) {
        auto duplicate = std::find_if(
            merged.begin(), merged.end(),
            [&](const BeamState& kept) { return same_path(kept, candidate); });
        if (duplicate == merged.end()) {
            merged.push_back(std::move(candidate));
        } else {
            duplicate->hyp.score = log_add_exp(duplicate->hyp.score,
                                               candidate.hyp.score);
        }
    }
    hypotheses.swap(merged);
}

float sort_score(const BeamState& hyp, bool score_norm) {
    if (!score_norm) return hyp.hyp.score;
    return hyp.hyp.score / (float)(hyp.hyp.tokens.size() + 1);
}

} // namespace

std::vector<TdtBeamHypothesis> tdt_beam_search(
    const PredictionNet& pred, const Joint& joint,
    const std::vector<float>& enc, int T, int enc_hidden,
    const std::vector<int32_t>& durations, int blank_id,
    int beam_size, int nbest, bool score_norm) {
    if (T < 0 || enc_hidden <= 0 ||
        enc.size() != (size_t)T * (size_t)enc_hidden)
        throw std::invalid_argument("tdt_beam_search: invalid encoder shape");
    if (durations.empty())
        throw std::invalid_argument("tdt_beam_search: durations must not be empty");
    if (beam_size < 1 || nbest < 1 || nbest > beam_size)
        throw std::invalid_argument(
            "tdt_beam_search: require beam_size >= nbest >= 1");

    const int num_durations = (int)durations.size();
    const int token_count = joint.V_plus() - num_durations;
    if (token_count != joint.vocab_size() + 1 ||
        num_durations != joint.num_durations() ||
        blank_id != token_count - 1 || blank_id < 1) {
        throw std::invalid_argument("tdt_beam_search: incompatible TDT head");
    }

    int zero_duration_idx = -1;
    int zero_duration_count = 0;
    bool has_positive_duration = false;
    for (int i = 0; i < num_durations; ++i) {
        if (durations[i] < 0)
            throw std::invalid_argument(
                "tdt_beam_search: durations must be non-negative");
        if (durations[i] == 0) {
            zero_duration_idx = i;
            ++zero_duration_count;
        }
        if (durations[i] > 0)
            has_positive_duration = true;
    }
    if (!has_positive_duration)
        throw std::invalid_argument(
            "tdt_beam_search: at least one positive duration is required");
    if (zero_duration_count > 1)
        throw std::invalid_argument(
            "tdt_beam_search: at most one zero duration is allowed");
    if (T == 0)
        return {TdtBeamHypothesis{}};

    const int beam = std::min(beam_size, blank_id);
    const int token_beam = std::min(beam, blank_id - 1);
    const int duration_beam = std::min(beam, num_durations);

    std::vector<float> enc_proj;
    joint.precompute_enc_proj(enc, T, enc_hidden, enc_proj);
    const int joint_hidden = joint.joint_hidden();

    BeamState start;
    start.dec_state = pred.zero_state();
    std::vector<BeamState> kept_hyps;
    kept_hyps.push_back(std::move(start));

    std::vector<float> logits;
    for (int time_idx = 0; time_idx < T; ++time_idx) {
        std::vector<BeamState> current_hyps;
        std::vector<BeamState> future_hyps;
        current_hyps.reserve(kept_hyps.size());
        future_hyps.reserve(kept_hyps.size());
        for (BeamState& hyp : kept_hyps) {
            if (hyp.last_frame == time_idx)
                current_hyps.push_back(std::move(hyp));
            else if (hyp.last_frame > time_idx)
                future_hyps.push_back(std::move(hyp));
        }
        kept_hyps.swap(future_hyps);

        while (!current_hyps.empty()) {
            auto best_it = std::max_element(
                current_hyps.begin(), current_hyps.end(),
                [](const BeamState& a, const BeamState& b) {
                    return a.hyp.score < b.hyp.score;
                });
            BeamState best = std::move(*best_it);
            current_hyps.erase(best_it);

            if (!best.pred_valid) {
                const bool is_sos = !best.emitted_any;
                const int32_t label = best.emitted_any ? best.last_token : blank_id;
                pred.step(label, is_sos, best.dec_state,
                          best.pred_out, best.next_state);
                best.pred_valid = true;
            }

            joint.step_logits(
                enc_proj.data() + (size_t)time_idx * joint_hidden,
                best.pred_out.data(), (int)best.pred_out.size(), logits);
            std::vector<float> token_logp =
                log_softmax(logits.data(), token_count);
            std::vector<float> duration_logp =
                log_softmax(logits.data() + token_count, num_durations);
            if (!std::all_of(token_logp.begin(), token_logp.end(),
                             [](float value) { return std::isfinite(value); }) ||
                !std::all_of(duration_logp.begin(), duration_logp.end(),
                             [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error(
                    "tdt_beam_search: non-finite log probability");
            }

            const std::vector<RankedIndex> best_durations =
                top_k(duration_logp.data(), num_durations, duration_beam);

            struct Pair {
                float score;
                int token;
                int duration_idx;
            };
            std::vector<Pair> pairs;
            const std::vector<RankedIndex> best_tokens =
                top_k(token_logp.data(), blank_id, token_beam);
            pairs.reserve(best_tokens.size() * best_durations.size());
            for (const RankedIndex& duration : best_durations)
                for (const RankedIndex& token : best_tokens)
                    pairs.push_back(Pair{
                        duration.score + token.score,
                        token.index,
                        duration.index});
            std::partial_sort(
                pairs.begin(),
                pairs.begin() + std::min(token_beam, (int)pairs.size()),
                pairs.end(),
                [](const Pair& a, const Pair& b) {
                    if (a.score != b.score) return a.score > b.score;
                    if (a.duration_idx != b.duration_idx)
                        return a.duration_idx < b.duration_idx;
                    return a.token < b.token;
                });
            pairs.resize(std::min(token_beam, (int)pairs.size()));

            for (const Pair& pair : pairs) {
                const int duration = durations[pair.duration_idx];
                BeamState child = best;
                child.hyp.score += pair.score;
                if (duration == 0 &&
                    !(child.hyp.score < best.hyp.score)) {
                    throw std::runtime_error(
                        "tdt_beam_search: zero-duration expansion "
                        "did not reduce score");
                }
                child.hyp.tokens.push_back(TdtBeamToken{
                    (int32_t)pair.token, (int32_t)time_idx, (int32_t)duration});
                child.dec_state = best.next_state;
                child.last_token = (int32_t)pair.token;
                child.emitted_any = true;
                child.pred_valid = false;
                child.last_frame += duration;
                if (duration == 0)
                    current_hyps.push_back(std::move(child));
                else
                    kept_hyps.push_back(std::move(child));
            }

            for (const RankedIndex& ranked_duration : best_durations) {
                int duration_idx = ranked_duration.index;
                if (duration_idx == zero_duration_idx) {
                    if (best_durations.size() != 1)
                        continue;
                    duration_idx = detail::best_positive_duration_index(
                        duration_logp, durations);
                }
                BeamState child = best;
                child.hyp.score += token_logp[blank_id] +
                                   duration_logp[duration_idx];
                child.last_frame += durations[duration_idx];
                kept_hyps.push_back(std::move(child));
            }

            merge_duplicate_hypotheses(kept_hyps);
            if (!current_hyps.empty()) {
                const float next_score = std::max_element(
                    current_hyps.begin(), current_hyps.end(),
                    [](const BeamState& a, const BeamState& b) {
                        return a.hyp.score < b.hyp.score;
                    })->hyp.score;
                const int more_probable_count = (int)std::count_if(
                    kept_hyps.begin(), kept_hyps.end(),
                    [next_score](const BeamState& hyp) {
                        return hyp.hyp.score > next_score;
                    });
                if (more_probable_count >= beam) {
                    std::vector<BeamState> more_probable;
                    more_probable.reserve(more_probable_count);
                    for (BeamState& hyp : kept_hyps)
                        if (hyp.hyp.score > next_score)
                            more_probable.push_back(std::move(hyp));
                    kept_hyps.swap(more_probable);
                    break;
                }
            } else {
                std::sort(kept_hyps.begin(), kept_hyps.end(),
                          [](const BeamState& a, const BeamState& b) {
                              return a.hyp.score > b.hyp.score;
                          });
                if ((int)kept_hyps.size() > beam)
                    kept_hyps.resize(beam);
            }
        }
    }

    std::sort(kept_hyps.begin(), kept_hyps.end(),
              [score_norm](const BeamState& a, const BeamState& b) {
                  return sort_score(a, score_norm) > sort_score(b, score_norm);
              });
    const int count = std::min(nbest, (int)kept_hyps.size());
    std::vector<TdtBeamHypothesis> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
        kept_hyps[i].hyp.normalized_score =
            kept_hyps[i].hyp.score /
            (float)(kept_hyps[i].hyp.tokens.size() + 1);
        result.push_back(std::move(kept_hyps[i].hyp));
    }
    return result;
}

} // namespace pk
