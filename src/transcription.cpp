#include "transcription.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <vector>

namespace pk {

namespace {

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece meta-space marker (3 bytes).
const char    META_SPACE[]  = "\xe2\x96\x81";
const size_t  META_SPACE_LEN = 3;

bool starts_with_meta(const std::string& piece) {
    return piece.size() >= META_SPACE_LEN &&
           (unsigned char)piece[0] == 0xE2 &&
           (unsigned char)piece[1] == 0x96 &&
           (unsigned char)piece[2] == 0x81;
}

// decode_ids_to_str([id]) for a single piece: replace `▁`->space, strip a single
// leading space (mirrors detokenize() on a 1-element id list).
std::string piece_to_text(const std::string& piece) {
    std::string out;
    out.reserve(piece.size());
    for (size_t i = 0; i < piece.size();) {
        if (i + META_SPACE_LEN <= piece.size() &&
            (unsigned char)piece[i]   == 0xE2 &&
            (unsigned char)piece[i+1] == 0x96 &&
            (unsigned char)piece[i+2] == 0x81) {
            out += ' ';
            i += META_SPACE_LEN;
        } else {
            out += piece[i++];
        }
    }
    if (!out.empty() && out[0] == ' ') out.erase(0, 1);
    return out;
}

// Mirror of NeMo extract_punctuation_from_vocab: a single-char string is a
// supported punctuation mark if it is Unicode category 'P*' AND its containing
// piece is NOT a special token ([...] / <...> / ## / ▁-prefixed / whitespace).
// For these subword vocabularies the punctuation marks are ASCII, so we classify
// the ASCII punctuation code points (Unicode category P) directly. The result is
// the set of single-character punctuation *strings* (NeMo iterates over each
// char of each non-special piece).
bool is_ascii_punct(unsigned char c) {
    // Unicode category starts with 'P' for these ASCII code points:
    //   ! " # % & ' ( ) * , - . / : ; ? @ [ \ ] _ { }
    switch (c) {
        case '!': case '"': case '#': case '%': case '&': case '\'':
        case '(': case ')': case '*': case ',': case '-': case '.':
        case '/': case ':': case ';': case '?': case '@': case '[':
        case '\\': case ']': case '_': case '{': case '}':
            return true;
        default:
            return false;
    }
}

bool is_special_token(const std::string& tok) {
    if (tok.empty()) return true;                      // whitespace-only -> special
    if (tok.front() == '[' && tok.back() == ']') return true;
    if (tok.front() == '<' && tok.back() == '>') return true;
    if (tok.size() >= 2 && tok[0] == '#' && tok[1] == '#') return true;
    if (starts_with_meta(tok)) return true;            // ▁-prefixed
    // whitespace-only
    if (std::all_of(tok.begin(), tok.end(),
                    [](char ch){ return std::isspace((unsigned char)ch); }))
        return true;
    return false;
}

std::set<std::string> extract_punctuation(const std::vector<std::string>& pieces) {
    std::set<std::string> punct;
    for (const std::string& tok : pieces) {
        if (is_special_token(tok)) continue;
        for (unsigned char c : tok) {
            if (is_ascii_punct(c)) punct.insert(std::string(1, (char)c));
        }
    }
    return punct;
}

} // namespace

std::vector<Word> group_words(const std::vector<TokenInfo>& tokens,
                              const std::vector<std::string>& pieces,
                              float frame_sec) {
    std::vector<Word> words;
    const int n = (int)tokens.size();
    if (n == 0) return words;

    const std::set<std::string> punct = extract_punctuation(pieces);
    const std::string DELIM = " "; // word delimiter (NeMo word_seperator default)

    auto piece_of = [&](int i) -> const std::string& {
        static const std::string empty;
        int id = tokens[i].id;
        if (id >= 0 && (size_t)id < pieces.size()) return pieces[(size_t)id];
        return empty;
    };
    auto is_punct = [&](const std::string& s) {
        return !s.empty() && s != DELIM && punct.count(s) > 0;
    };

    // Per-token char offsets (NeMo char_offsets): text = decode_ids_to_str([id]),
    // token = raw piece, start/end in encoder frames. end_offset = frame + span
    // (TDT duration; CTC run-length supplied by the caller via TokenInfo.span).
    std::vector<std::string> text(n), tok(n);
    std::vector<int32_t>      start(n), end(n);
    std::vector<float>        conf(n);
    for (int i = 0; i < n; ++i) {
        tok[i]   = piece_of(i);
        text[i]  = piece_to_text(tok[i]);
        start[i] = tokens[i].frame;
        end[i]   = tokens[i].frame + tokens[i].span;
        conf[i]  = tokens[i].conf;
    }

    // NeMo _refine_timestamps / _refine_timestamps_tdt: a punctuation token
    // (text[0] in supported_punctuation, i>0) is pinned to the previous token's
    // end (start = prev end; end = start). This is the TDT refinement; NeMo's CTC
    // refine only sets end = start, but here start is never used for a punct token
    // (it merges into the previous word and only its end matters), so the unified
    // rule reproduces both heads' dumped baseline word offsets exactly.
    for (int i = 0; i < n; ++i) {
        if (!text[i].empty() && i > 0) {
            std::string first(1, text[i][0]);
            if (is_punct(first)) {
                start[i] = end[i - 1];
                end[i]   = start[i];
            }
        }
    }

    // NeMo get_words_offsets (tokenizer_type='bpe', word_delimiter=' '):
    //   word-start condition: token != token_text  (i.e. the piece differs from
    //   its decoded text -> a `▁`-prefixed sub-word starts a new word), OR the
    //   token IS the delimiter and the next non-delimiter token is not punct.
    // built holds the indices of the current word's tokens; previous_token_index
    // marks the word's first token.
    std::vector<int> built;
    int prev = 0;

    auto detok_built = [&](const std::vector<int>& idxs) {
        std::vector<int32_t> ids;
        ids.reserve(idxs.size());
        for (int k : idxs) ids.push_back(tokens[k].id);
        return detokenize(pieces, ids);
    };
    auto min_conf = [&](const std::vector<int>& idxs) {
        float m = 1.0f;
        for (int k : idxs) m = std::min(m, conf[k]);
        return m;
    };

    for (int i = 0; i < n; ++i) {
        const std::string& ct = text[i];
        const std::string& tk = tok[i];
        const bool curr_punct = is_punct(ct);

        // next non-delimiter token text (NeMo lookahead).
        std::string next_non_delim;
        int j = i;
        while (next_non_delim.empty() && j < n - 1) {
            ++j;
            if (text[j] != DELIM) next_non_delim = text[j];
        }
        const bool next_is_punct = !next_non_delim.empty() && is_punct(next_non_delim);

        const bool word_start_cond =
            (tk != ct) || (ct == DELIM && !next_is_punct);

        if (word_start_cond && !curr_punct) {
            if (!built.empty()) {
                Word w;
                w.text  = detok_built(built);
                w.start = (float)start[prev]      * frame_sec;
                w.end   = (float)end[built.back()] * frame_sec;
                w.conf  = min_conf(built);
                words.push_back(std::move(w));
            }
            built.clear();
            if (ct != DELIM) {
                built.push_back(i);
                prev = i;
            }
        } else if (curr_punct && built.empty() && !words.empty()) {
            // Punctuation with no open word: attach to the previous word, extend
            // its end, drop a trailing space, append the punctuation char.
            Word& lw = words.back();
            lw.end = (float)end[i] * frame_sec;
            if (!lw.text.empty() && lw.text.back() == ' ') lw.text.pop_back();
            lw.text += ct;
            lw.conf = std::min(lw.conf, conf[i]);
        } else if (curr_punct && !built.empty()) {
            // Punctuation closing an open word: drop a trailing delimiter token,
            // then append this token.
            if (!built.empty()) {
                const std::string& last = tok[built.back()];
                if (last == " " || last == "_" || last == META_SPACE) built.pop_back();
            }
            built.push_back(i);
        } else {
            // Continuation sub-word: extend the current word.
            if (built.empty()) prev = i;
            built.push_back(i);
        }
    }

    // NeMo tail handling: force the first word's start to the first token's start,
    // and flush any remaining built tokens as the final word.
    if (!words.empty()) {
        words[0].start = (float)start[0] * frame_sec;
        if (!built.empty()) {
            Word w;
            w.text  = detok_built(built);
            w.start = (float)start[prev]      * frame_sec;
            w.end   = (float)end[built.back()] * frame_sec;
            w.conf  = min_conf(built);
            words.push_back(std::move(w));
        }
    } else if (!built.empty()) {
        Word w;
        w.text  = detok_built(built);
        w.start = (float)start[0]          * frame_sec;
        w.end   = (float)end[built.back()] * frame_sec;
        w.conf  = min_conf(built);
        words.push_back(std::move(w));
    }

    return words;
}

size_t open_word_start(const std::vector<TokenInfo>& tokens,
                       const std::vector<std::string>& pieces) {
    // Same word-start rule group_words applies to select a word's first token:
    //   (piece != decoded_text)  -> a `▁`-prefixed sub-word, i.e. word_start_cond
    //   && decoded_text is not punctuation   -> group_words' !curr_punct guard
    //   && decoded_text != the space delimiter -> group_words' `ct != DELIM`
    // A `▁`-prefixed punctuation piece like `▁'` (piece != text, but text is
    // punctuation) is therefore NOT a boundary — matching group_words, which
    // attaches it to the preceding word instead of starting a new one.
    const std::set<std::string> punct = extract_punctuation(pieces);
    const std::string DELIM = " ";
    auto is_punct = [&](const std::string& s) {
        return !s.empty() && s != DELIM && punct.count(s) > 0;
    };
    for (size_t i = tokens.size(); i-- > 0;) {
        int id = tokens[i].id;
        if (id < 0 || (size_t)id >= pieces.size()) continue;
        const std::string& piece = pieces[(size_t)id];
        const std::string text = piece_to_text(piece);
        if (piece != text && text != DELIM && !is_punct(text)) return i;
    }
    return 0;
}

} // namespace pk
