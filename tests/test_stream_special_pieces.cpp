#include "streaming.hpp"
#include "tokenizer.hpp"
#include "transcription.hpp"

#include <cstdio>
#include <string>
#include <vector>

// A prompt-conditioned model emits a language tag such as <en-US> after each
// sentence, with a lone `▁` before it. The tag and that `▁` must stay out of the
// text (offline and streamed) and out of the words, and text without tags must
// not change. No model needed: the streaming checks drive append_stream_chunk()
// and regroup_stream_words(), which StreamingSession::feed_mel_chunk() calls for
// the text, the word tokens and the word grouping of every chunk.
static int failures = 0;
static void expect(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "test_stream_special_pieces: %s: '%s', expected '%s'\n",
                     what.c_str(), got.c_str(), want.c_str());
        ++failures;
    }
}

static std::string join_words(const std::vector<pk::Word>& words) {
    std::string joined;
    for (const pk::Word& w : words) joined += (joined.empty() ? "" : "|") + w.text;
    return joined;
}

// Runs the streaming bookkeeping over `ids` split into chunks at `cuts`; returns
// the final text and the words drained over the whole run, and checks that the
// text only grows. An empty word shows up in the joined words as '||'.
struct Run { std::string text; std::string words; };
static Run stream(const std::vector<std::string>& pieces, const std::vector<int32_t>& ids,
                  const std::vector<size_t>& cuts, int eou_id) {
    pk::StreamTextState st;
    std::string text;
    std::vector<pk::TokenInfo> word_tokens;
    std::vector<pk::Word> final_words, words, drained;
    size_t cursor = 0, finalized = 0, taken = 0;
    size_t begin = 0;
    std::vector<size_t> ends = cuts;
    ends.push_back(ids.size());
    for (size_t k = 0; k < ends.size(); ++k) {
        const size_t end = ends[k];
        std::vector<int32_t> emitted(ids.begin() + begin, ids.begin() + end);
        std::vector<pk::TokenInfo> infos;
        for (size_t i = 0; i < emitted.size(); ++i)
            infos.push_back({emitted[i], (int32_t)i, 0.9f, 1});
        const std::string before = text;
        pk::append_stream_chunk(pieces, emitted, infos, (int)begin, eou_id, -1, st, text,
                                word_tokens);
        if (text.compare(0, before.size(), before) != 0) {
            std::fprintf(stderr, "test_stream_special_pieces: text is not append-only\n");
            ++failures;
        }
        const bool last = (k + 1 == ends.size());
        pk::regroup_stream_words(word_tokens, pieces, 0.08f, last, final_words, cursor, words,
                                 finalized);
        if (finalized < taken) finalized = taken;
        for (size_t i = taken; i < finalized && i < words.size(); ++i) drained.push_back(words[i]);
        taken = finalized;
        begin = end;
    }
    return {text, join_words(drained)};
}

int main() {
    const std::string SP = "\xe2\x96\x81";
    const std::vector<std::string> pieces = {
        SP + "The",   // 0
        SP + "sun",   // 1
        ".",          // 2
        SP,           // 3 lone ▁ (before the tag)
        "<en-US>",    // 4 language tag
        SP + "It",    // 5
        "<EOU>",      // 6
        SP + "a",     // 7
        SP + "b",     // 8
    };

    // Offline: strip_special_tokens drops the tag and the lone ▁ before it.
    expect("offline text",
           pk::detokenize(pieces, pk::strip_special_tokens(pieces, {0, 1, 2, 3, 4, 5})),
           "The sun. It");

    // Streaming over every way of cutting the ids into two or three chunks,
    // including a cut between the lone ▁ and the tag, and one at the very start.
    const std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 6};  // ... <EOU> is an event
    for (size_t c1 = 0; c1 <= ids.size(); ++c1) {
        for (size_t c2 = c1; c2 <= ids.size(); ++c2) {
            const Run r = stream(pieces, ids, {c1, c2}, /*eou*/ 6);
            const std::string at = " (cuts " + std::to_string(c1) + "," + std::to_string(c2) + ")";
            expect("stream text" + at, r.text, "The sun. It");
            expect("stream words" + at, r.words, "The|sun.|It");
        }
    }

    // Ordinary text is unchanged: repeated `▁` pieces keep every space.
    const std::vector<int32_t> plain = {7, 3, 3, 8};  // ▁a ▁ ▁ ▁b
    expect("plain detokenize", pk::detokenize(pieces, plain), "a   b");
    expect("plain strip+detokenize",
           pk::detokenize(pieces, pk::strip_special_tokens(pieces, plain)), "a   b");
    for (size_t c = 0; c <= plain.size(); ++c) {
        expect("plain stream text (cut " + std::to_string(c) + ")",
               stream(pieces, plain, {c}, 6).text, "a   b");
    }

    if (failures) return 1;
    std::puts("test_stream_special_pieces: OK");
    return 0;
}
