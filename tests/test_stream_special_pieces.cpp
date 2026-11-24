#include "tokenizer.hpp"
#include "transcription.hpp"

#include <cstdio>
#include <string>
#include <vector>

// A prompt-conditioned model emits a language tag such as <en-US> after each
// sentence, with a lone `▁` before it. detokenize() must not double the space a
// stripped tag leaves, and group_words() must make no word of the tag or the `▁`.
int main() {
    const std::vector<std::string> pieces = {
        "\xe2\x96\x81The",  // 0 ▁The
        "\xe2\x96\x81sun",  // 1 ▁sun
        ".",                // 2
        "\xe2\x96\x81",     // 3 ▁ (before the tag)
        "<en-US>",          // 4 language tag
        "\xe2\x96\x81It",   // 5 ▁It
    };
    int failures = 0;

    // Streamed text once the tag id is dropped: 0 1 2 3 5.
    const std::string text = pk::detokenize(pieces, {0, 1, 2, 3, 5});
    if (text != "The sun. It") {
        std::fprintf(stderr, "test_stream_special_pieces: text '%s', expected 'The sun. It'\n",
                     text.c_str());
        ++failures;
    }

    // Word timestamps from tokens that still hold the tag: 0 1 2 3 4 5.
    std::vector<pk::TokenInfo> tokens;
    for (int32_t i = 0; i < 6; ++i) {
        tokens.push_back({i, i, 0.9f, 1});
    }
    const std::vector<pk::Word> words = pk::group_words(tokens, pieces, 0.08f);
    std::string joined;
    for (const pk::Word& w : words) {
        joined += (joined.empty() ? "" : "|") + w.text;
    }
    if (joined != "The|sun.|It") {
        std::fprintf(stderr, "test_stream_special_pieces: words '%s', expected 'The|sun.|It'\n",
                     joined.c_str());
        ++failures;
    }

    if (failures) return 1;
    std::puts("test_stream_special_pieces: OK");
    return 0;
}
