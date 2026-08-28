#include "tokenizer.hpp"
#include <cstdio>
#include <string>
#include <vector>

// Regression for issue #40: offline decode emits raw special tokens (e.g. the
// language tag <en-US>) in the transcript text.
//
// Root cause: src/model.cpp's four offline decode paths (decode_enc_out(),
// transcribe_16k_batch(), decode_enc_out_with_timestamps(),
// transcribe_16k_batch_with_timestamps()) all pass the raw decoded id sequence
// straight to pk::detokenize(), with no filtering. When the decoder emits a
// special-token id (a language tag for prompt-conditioned models, <EOU>, etc.),
// it lands verbatim in the output text.
//
// This test reproduces that defect using pk::detokenize() itself -- the exact,
// unmodified function every one of those four call sites invokes -- fed a
// decoded id sequence shaped like the one in the bug report (content tokens
// followed by a trailing language-tag id). It then verifies the fix: wrapping
// the same call with pk::strip_special_tokens(), exactly as model.cpp now
// does at each of the four sites, removes the tag.
int main() {
    const std::vector<std::string> pieces = {
        "\xe2\x96\x81The",  // ▁The
        "\xe2\x96\x81sun",  // ▁sun
        "<en-US>",          // language-tag special token, as in the bug report
        "<EOU>",
        "[CLS]",
    };

    // Decoded id sequence a TDT/RNNT greedy decode would produce for a
    // prompt-conditioned model: real words, then the trailing language tag.
    const std::vector<int32_t> decoded_ids = { 0, 1, 2 };

    // 1. Reproduce the reported defect: this is *exactly* what every one of
    //    the four offline decode call sites did before the fix -- hand the raw
    //    decoded ids straight to detokenize(), unmodified, no filtering.
    std::string buggy = pk::detokenize(pieces, decoded_ids);
    if (buggy.find("<en-US>") == std::string::npos) {
        std::fprintf(stderr,
                     "test_special_token_filter: test setup does not reproduce the bug; "
                     "raw detokenize() unexpectedly omitted the tag (got=[%s])\n",
                     buggy.c_str());
        return 1;
    }

    // 2. Verify the fix: model.cpp's call sites now filter with
    //    strip_special_tokens() before detokenizing. Same ids, same pieces.
    std::string fixed = pk::detokenize(pieces, pk::strip_special_tokens(pieces, decoded_ids));
    const std::string expected = "The sun";
    if (fixed != expected) {
        std::fprintf(stderr,
                     "test_special_token_filter: MISMATCH got=[%s] expected=[%s]\n",
                     fixed.c_str(), expected.c_str());
        return 1;
    }

    // Ordinary content tokens (▁-prefixed subwords) must survive untouched --
    // the filter must not over-strip real transcript text.
    std::vector<int32_t> only_words = pk::strip_special_tokens(pieces, { 0, 1 });
    if (only_words.size() != 2) {
        std::fprintf(stderr,
                     "test_special_token_filter: normal tokens were incorrectly stripped (got %zu)\n",
                     only_words.size());
        return 1;
    }

    // <...> and [...] tokens are stripped wherever they appear in the sequence,
    // not just at the end.
    std::vector<int32_t> stripped = pk::strip_special_tokens(pieces, { 0, 3, 1, 4, 2 });
    if (stripped.size() != 2) {
        std::fprintf(stderr,
                     "test_special_token_filter: bracketed special tokens not stripped (got %zu)\n",
                     stripped.size());
        return 1;
    }

    std::fprintf(stderr, "test_special_token_filter: OK\n");
    return 0;
}
