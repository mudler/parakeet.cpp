#pragma once
#include "decode_types.hpp"

#include <string>
#include <vector>

namespace pk {

// One decoded word with its time span (seconds) and aggregate confidence.
//
//   text  : the word string (SentencePiece pieces detokenized — `▁`->space,
//           leading space stripped, punctuation attached).
//   start : word start time in seconds (= first token frame * frame_sec).
//   end   : word end time in seconds (= last token end_offset * frame_sec; see
//           group_words for the per-head end_offset convention).
//   conf  : NeMo's 'min' aggregate of the word's per-token confidences.
struct Word {
    std::string text;
    float       start = 0.0f;
    float       end   = 0.0f;
    float       conf  = 0.0f;
    // Indices of the word's first and last token in Transcription::tokens, or
    // -1 when unknown. Used to remove a word's tokens with it.
    int32_t     tok_first = -1;
    int32_t     tok_last  = -1;
};

// A full transcription result: the flat text, the per-word timestamps +
// confidence, and the raw per-token metadata the words were grouped from.
struct Transcription {
    std::string            text;
    std::vector<Word>      words;
    std::vector<TokenInfo> tokens;
    // Number of words an optional word filter removed (see WordFilter). -1 when
    // no filter ran, so a caller can tell "nothing dropped" from "no filter".
    int                    dropped_words = -1;
};

// Group a per-token decode (TokenInfo sequence, in emission order) into words,
// matching NeMo's word-offset + confidence convention (timestamps=True,
// confidence aggregation='min').
//
// `pieces` is the tokenizer piece table (index by TokenInfo.id). A token whose
// piece begins with `▁` (U+2581) starts a new word; the leading-`▁` of the
// utterance starts the first word too. Consecutive non-`▁` pieces extend the
// current word. Word text = detokenize(the word's piece ids). A supported
// punctuation mark (extracted from `pieces` like NeMo
// extract_punctuation_from_vocab) attaches to the preceding word and has its
// own start/end refined to the previous token's end (NeMo _refine_timestamps).
//
// Per-token offsets (frames): start_offset = TokenInfo.frame; end_offset =
// TokenInfo.frame + TokenInfo.span. For TDT, span is the predicted duration
// (NeMo _compute_offsets_tdt: end = start + duration). For CTC, NeMo's
// end_offset is the NEXT token's start frame (cumulative run lengths); the
// caller therefore sets each CTC token's span to (next_frame - frame) so the
// `frame + span` rule reproduces NeMo exactly (the id-only ctc_greedy keeps
// span == 1; transcribe_with_timestamps does the run-length rewrite).
//
//   word.start = first_token.frame                      * frame_sec
//   word.end   = (last_token.frame + last_token.span)   * frame_sec
//   word.conf  = min over the word's token confidences
std::vector<Word> group_words(const std::vector<TokenInfo>& tokens,
                              const std::vector<std::string>& pieces,
                              float frame_sec);

// Optional word filter, off by default. It removes words from a Transcription
// by their own confidence values, with no change to the decoding.
//
//   min_local_conf : 0 = off. A word is dropped when the mean confidence of the
//                    words that start within local_radius_sec of its start (the
//                    word included) is below this value. A low confidence word
//                    between confident words keeps a high mean and stays. Words
//                    that stand alone, or only among other low confidence words,
//                    are dropped. This is what noise tends to produce.
//   local_radius_sec: window for the mean, both sides, seconds (default 5).
//   drop_punct_only: also drop words that are only punctuation (a CTC model
//                    can emit a lone "." or "?" on noise).
//
// A "decode unit" is one call of the decoder: the whole clip, or one VAD
// segment. Apply the filter to each unit alone, so that the mean never spans
// two segments. The dropped words leave `text`, `words` and `tokens`, and the
// token indices of the kept words are updated. When nothing is dropped the
// transcription is not changed. Returns the number of words dropped and adds it
// to `dropped_words` (which becomes 0 or more).
struct WordFilter {
    float min_local_conf   = 0.0f;
    float local_radius_sec = 5.0f;
    bool  drop_punct_only  = false;
    bool active() const { return min_local_conf > 0.0f || drop_punct_only; }
};
int apply_word_filter(Transcription& t, const WordFilter& f);

// Relative index (into `tokens`) of the first token of the LAST word group_words
// would emit from `tokens` — i.e. where the still-open (trailing) word begins.
// Uses group_words' exact word-start rule: a token opens a word iff its raw piece
// differs from its decoded text (a `▁`-prefixed sub-word) AND that decoded text
// is not a punctuation mark. Returns 0 if no token qualifies (all
// continuation/punctuation).
//
// StreamingSession uses this to split its accumulated word tokens into an
// already-finalized prefix and the still-open tail without re-grouping the whole
// session each chunk. Because group_words never merges across a word-start
// boundary, group_words(tokens[0,idx)) ++ group_words(tokens[idx,end)) ==
// group_words(tokens) with the second part being exactly the one open word — the
// invariant the incremental grouping relies on. Matching group_words' rule here
// (not just "piece starts with `▁`") is required: a `▁`-prefixed punctuation
// piece such as `▁'` (space + opening quote) starts with `▁` yet does NOT open a
// word, so treating it as a boundary would drop the real preceding word.
size_t open_word_start(const std::vector<TokenInfo>& tokens,
                       const std::vector<std::string>& pieces);

} // namespace pk
