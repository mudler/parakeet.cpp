#include "transcription.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
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

std::vector<Word> group_words(const std::vector<TokenInfo>& tokens_in,
                              const std::vector<std::string>& pieces,
                              float frame_sec) {
    // Language tags such as <en-US> are not words. The model emits a bare `▁`
    // piece before each tag, which would otherwise become an empty word.
    auto piece_at = [&](const TokenInfo& t) -> const std::string* {
        return t.id >= 0 && (size_t)t.id < pieces.size() ? &pieces[(size_t)t.id] : nullptr;
    };
    std::vector<TokenInfo> tokens;
    tokens.reserve(tokens_in.size());
    for (const TokenInfo& t : tokens_in) {
        const std::string* pc = piece_at(t);
        if (pc && is_special_piece(*pc)) {
            const std::string* prev = tokens.empty() ? nullptr : piece_at(tokens.back());
            if (prev && *prev == META_SPACE) tokens.pop_back();
            continue;
        }
        tokens.push_back(t);
    }
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
                w.tok_first = built.front(); w.tok_last = built.back();
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
            lw.tok_last = i;
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
            w.tok_first = built.front(); w.tok_last = built.back();
            words.push_back(std::move(w));
        }
    } else if (!built.empty()) {
        Word w;
        w.text  = detok_built(built);
        w.start = (float)start[0]          * frame_sec;
        w.end   = (float)end[built.back()] * frame_sec;
        w.conf  = min_conf(built);
        w.tok_first = built.front(); w.tok_last = built.back();
        words.push_back(std::move(w));
    }

    return words;
}

namespace {

// True when the word has no letter or digit: only punctuation or spaces.
// ASCII is classified directly. Other code points count as content except the
// common punctuation blocks (General Punctuation, CJK symbols and punctuation,
// and the Latin-1 marks used in Spanish and French).
bool is_punct_only(const std::string& w) {
    size_t i = 0;
    while (i < w.size()) {
        const unsigned char c = (unsigned char)w[i];
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0xF0 && i + 3 < w.size()) { cp = ((c & 0x07u) << 18) | (((unsigned char)w[i + 1] & 0x3Fu) << 12) | (((unsigned char)w[i + 2] & 0x3Fu) << 6) | ((unsigned char)w[i + 3] & 0x3Fu); len = 4; }
        else if (c >= 0xE0 && i + 2 < w.size()) { cp = ((c & 0x0Fu) << 12) | (((unsigned char)w[i + 1] & 0x3Fu) << 6) | ((unsigned char)w[i + 2] & 0x3Fu); len = 3; }
        else if (c >= 0xC0 && i + 1 < w.size()) { cp = ((c & 0x1Fu) << 6) | ((unsigned char)w[i + 1] & 0x3Fu); len = 2; }
        i += len;
        if (cp < 0x80) {
            if (std::isalnum((int)cp)) return false;
            continue;
        }
        const bool punct = (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) ||
                           cp == 0xA1 || cp == 0xA7 || cp == 0xAB || cp == 0xB6 || cp == 0xB7 ||
                           cp == 0xBB || cp == 0xBF;
        if (!punct) return false;
    }
    return true;
}

}  // namespace

int apply_word_filter(Transcription& t, const WordFilter& f) {
    if (!f.active() || t.words.empty()) return 0;
    const size_t n = t.words.size();
    std::vector<char> drop(n, 0);
    size_t dropped = 0;
    if (f.min_local_conf > 0.0f) {
        // Mean confidence of the words that start within the radius of a word,
        // the word itself included. Words are in time order, so the window of
        // each word is a range that moves forward; fall back to a scan when a
        // caller passes words out of order.
        bool sorted = true;
        for (size_t i = 1; i < n && sorted; ++i) sorted = t.words[i - 1].start <= t.words[i].start;
        std::vector<double> cum(n + 1, 0.0);
        for (size_t i = 0; i < n; ++i) cum[i + 1] = cum[i] + (double)t.words[i].conf;
        const float r = f.local_radius_sec;
        size_t lo = 0, hi = 0;  // window [lo, hi)
        for (size_t i = 0; i < n; ++i) {
            const float s = t.words[i].start;
            double sum = 0.0;
            size_t cnt = 0;
            if (sorted) {
                while (lo < i && s - t.words[lo].start > r) ++lo;
                if (hi < i + 1) hi = i + 1;
                while (hi < n && t.words[hi].start - s <= r) ++hi;
                sum = cum[hi] - cum[lo];
                cnt = hi - lo;
            } else {
                for (size_t j = 0; j < n; ++j)
                    if (std::fabs(t.words[j].start - s) <= r) { sum += (double)t.words[j].conf; ++cnt; }
            }
            if (cnt > 0 && sum / (double)cnt < (double)f.min_local_conf) drop[i] = 1;
        }
    }
    if (f.drop_punct_only)
        for (size_t i = 0; i < n; ++i)
            if (is_punct_only(t.words[i].text)) drop[i] = 1;
    for (size_t i = 0; i < n; ++i) dropped += drop[i] ? 1u : 0u;
    t.dropped_words = (t.dropped_words < 0 ? 0 : t.dropped_words) + (int)dropped;
    if (dropped == 0) return 0;

    // Rebuild words, text and tokens without the dropped words.
    std::vector<char> tdrop(t.tokens.size(), 0);
    std::vector<Word> kept;
    std::string text;
    for (size_t i = 0; i < n; ++i) {
        const Word& w = t.words[i];
        if (drop[i]) {
            for (int32_t k = w.tok_first; k >= 0 && k <= w.tok_last && (size_t)k < tdrop.size(); ++k) tdrop[(size_t)k] = 1;
            continue;
        }
        if (!text.empty()) text += ' ';
        text += w.text;
        kept.push_back(w);
    }
    std::vector<int32_t> remap(t.tokens.size(), -1);
    std::vector<TokenInfo> toks;
    for (size_t k = 0; k < t.tokens.size(); ++k)
        if (!tdrop[k]) { remap[k] = (int32_t)toks.size(); toks.push_back(t.tokens[k]); }
    for (Word& w : kept) {
        int32_t a = -1, b = -1;
        for (int32_t k = w.tok_first; k >= 0 && k <= w.tok_last && (size_t)k < remap.size(); ++k)
            if (remap[(size_t)k] >= 0) { if (a < 0) a = remap[(size_t)k]; b = remap[(size_t)k]; }
        w.tok_first = a;
        w.tok_last = b;
    }
    t.words = std::move(kept);
    t.tokens = std::move(toks);
    t.text = std::move(text);
    return (int)dropped;
}

} // namespace pk
