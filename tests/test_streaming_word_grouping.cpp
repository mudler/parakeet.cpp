#include "transcription.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <string>
#include <vector>

// Model-free regression for the incremental word-grouping bug in
// StreamingSession::regroup_words (PR #39 review).
//
// regroup_words splits the accumulated word tokens into an already-finalized
// prefix and the still-open trailing word, using open_word_start() to find where
// that open word begins. The invariant it relies on is that group_words never
// merges across a word-start boundary, so
//
//   group_words(tokens[0, idx)) ++ group_words(tokens[idx, end))
//       == group_words(tokens)                         (idx = open_word_start)
//
// with the SECOND part being exactly the one open word. If open_word_start picks
// the boundary with the wrong rule ("piece starts with `▁`" instead of
// group_words' real word-start rule), a `▁`-prefixed punctuation piece like `▁'`
// (space + opening quote) is mistaken for a word start: the cursor lands past the
// real preceding word, which is then dropped from every future tail — silently
// losing a word from drain_words()/--timestamps/--json.
//
// These cases are pure functions of tokens + a synthetic pieces vocab, so no
// model is needed (runs in CI).

namespace {

// SentencePiece meta-space prefix (U+2581) used for `▁`-prefixed word-start
// pieces, as UTF-8.
const std::string M = "\xe2\x96\x81";

struct Scenario {
    const char*              name;
    std::vector<std::string> pieces;  // vocab; ids index into this
    std::vector<int>         ids;     // emitted token ids, in order
};

std::vector<pk::TokenInfo> make_tokens(const std::vector<int>& ids) {
    std::vector<pk::TokenInfo> toks;
    toks.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
        toks.push_back(pk::TokenInfo{ids[i], (int32_t)i, 1.0f, 1});
    return toks;
}

bool word_eq(const pk::Word& a, const pk::Word& b) {
    return a.text == b.text && a.start == b.start && a.end == b.end &&
           a.conf == b.conf;
}

int failures = 0;

void check(const Scenario& sc) {
    const float fs = 0.01f;
    std::vector<pk::TokenInfo> toks = make_tokens(sc.ids);

    std::vector<pk::Word> full = pk::group_words(toks, sc.pieces, fs);

    const size_t idx = pk::open_word_start(toks, sc.pieces);

    std::vector<pk::TokenInfo> prefix(toks.begin(), toks.begin() + (std::ptrdiff_t)idx);
    std::vector<pk::TokenInfo> suffix(toks.begin() + (std::ptrdiff_t)idx, toks.end());
    std::vector<pk::Word> gp = pk::group_words(prefix, sc.pieces, fs);
    std::vector<pk::Word> gs = pk::group_words(suffix, sc.pieces, fs);

    std::vector<pk::Word> concat = gp;
    concat.insert(concat.end(), gs.begin(), gs.end());

    bool ok = true;

    // (a) The open tail must be exactly ONE word.
    if (gs.size() != 1) {
        ok = false;
        std::fprintf(stderr,
            "[%s] FAIL: open tail is %zu words, expected 1 (idx=%zu)\n",
            sc.name, gs.size(), idx);
    }

    // (b) prefix-words ++ open-word == full grouping, byte-for-byte.
    if (concat.size() != full.size()) {
        ok = false;
        std::fprintf(stderr,
            "[%s] FAIL: incremental produced %zu words, group_words(full) has %zu\n",
            sc.name, concat.size(), full.size());
    } else {
        for (size_t i = 0; i < full.size(); ++i) {
            if (!word_eq(concat[i], full[i])) {
                ok = false;
                std::fprintf(stderr,
                    "[%s] FAIL: word %zu differs: incremental=\"%s\" full=\"%s\"\n",
                    sc.name, i, concat[i].text.c_str(), full[i].text.c_str());
            }
        }
    }

    if (ok) {
        std::fprintf(stderr, "[%s] OK: idx=%zu, %zu words match group_words\n",
                     sc.name, idx, full.size());
    } else {
        std::fprintf(stderr, "[%s]   full grouping:", sc.name);
        for (const pk::Word& w : full) std::fprintf(stderr, " |%s|", w.text.c_str());
        std::fprintf(stderr, "\n[%s]   incremental  :", sc.name);
        for (const pk::Word& w : concat) std::fprintf(stderr, " |%s|", w.text.c_str());
        std::fprintf(stderr, "\n");
        ++failures;
    }
}

} // namespace

int main() {
    // The reported bug: `▁'` (space + opening single quote) is the LAST
    // `▁`-prefixed piece in the tail but is punctuation, not a word start. The
    // open word actually starts at `▁and`; scanning for `▁` alone drops "and".
    //   the and'in  ->  ["the", "and 'in"]   (the quote attaches to "and ... in")
    check(Scenario{
        "meta_punct_quote",
        { M + "the", M + "and", M + "'", "in", "'" },
        { 0, 1, 2, 3 },   // ▁the ▁and ▁' in
    });

    // A trailing normal word after a quoted one: the open word is `▁means`, a
    // clean `▁`-word-start. Boundary must land there, not on the earlier `▁'`.
    check(Scenario{
        "quote_then_word",
        { M + "quote", M + "'", "s", M + "means", "'" },
        { 0, 1, 2, 3 },   // ▁quote ▁' s ▁means
    });

    // Plain multi-word tail (no punctuation) — the boundary is the last
    // `▁`-word-start and behaviour is unchanged from before.
    check(Scenario{
        "plain_words",
        { M + "hello", M + "world", "ish" },
        { 0, 1, 2 },      // ▁hello ▁world ish
    });

    if (failures) {
        std::fprintf(stderr, "test_streaming_word_grouping: %d scenario(s) FAILED\n",
                     failures);
        return 1;
    }
    std::fprintf(stderr, "test_streaming_word_grouping: all scenarios passed\n");
    return 0;
}
