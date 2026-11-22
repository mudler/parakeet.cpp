// pk::refine_emission_word_times on scripted RNN-T style words (no model).
#include "transcription.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static constexpr float F = 0.08f;
static bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

static Word word(const char* t, float start, float end) {
    Word w;
    w.text = t;
    w.start = start;
    w.end = end;
    w.conf = 0.9f;
    w.tok_first = w.tok_last = 7;
    return w;
}

// Words close together: each start moves back 3 frames, and each end reaches
// the next (shifted) start, since that is within the maximum word length.
static void test_shift_and_fill() {
    std::vector<Word> w = {word("a", 1.20f, 1.28f), word("b", 1.60f, 1.68f), word("c", 2.00f, 2.08f)};
    refine_emission_word_times(w, F);
    CHECK(near(w[0].start, 0.96f)); CHECK(near(w[0].end, 1.36f));
    CHECK(near(w[1].start, 1.36f)); CHECK(near(w[1].end, 1.76f));
    // The last word has no next word: its end only moves back 2 frames.
    CHECK(near(w[2].start, 1.76f)); CHECK(near(w[2].end, 1.92f));
    CHECK(w[0].text == "a" && w[0].conf == 0.9f && w[0].tok_first == 7 && w[0].tok_last == 7);
}

// A pause after a word: the end grows at most kMaxWordFrames past the start.
static void test_max_length_in_a_pause() {
    std::vector<Word> w = {word("a", 1.00f, 1.08f), word("b", 4.00f, 4.08f)};
    refine_emission_word_times(w, F);
    CHECK(near(w[0].start, 0.76f));
    CHECK(near(w[0].end, 0.76f + kMaxWordFrames * F));
    CHECK(near(w[1].start, 3.76f));
}

// A multi-token word keeps a length of at least its own (shifted) span.
static void test_long_word_kept() {
    std::vector<Word> w = {word("long", 1.00f, 1.80f), word("x", 1.90f, 1.98f)};
    refine_emission_word_times(w, F);
    CHECK(near(w[0].start, 0.76f));
    CHECK(near(w[0].end, 1.64f));                  // 1.80 - 2 frames
    CHECK(near(w[1].start, 1.66f));                // 1.90 - 3 frames
}

// Starts clamp at 0, and words never overlap or go backwards.
static void test_clamp_and_order() {
    std::vector<Word> w = {word("a", 0.08f, 0.16f), word("b", 0.16f, 0.24f), word("c", 0.24f, 0.32f)};
    refine_emission_word_times(w, F);
    CHECK(near(w[0].start, 0.0f));
    for (size_t i = 0; i < w.size(); ++i) {
        CHECK(w[i].start >= 0.0f);
        CHECK(w[i].end >= w[i].start);
        if (i > 0) CHECK(w[i].start >= w[i - 1].end);
    }
}

static void test_empty_and_bad_frame() {
    std::vector<Word> none;
    refine_emission_word_times(none, F);
    CHECK(none.empty());
    std::vector<Word> w = {word("a", 1.0f, 1.08f)};
    refine_emission_word_times(w, 0.0f);           // no frame size: unchanged
    CHECK(near(w[0].start, 1.0f) && near(w[0].end, 1.08f));
}

int main() {
    test_shift_and_fill();
    test_max_length_in_a_pause();
    test_long_word_kept();
    test_clamp_and_order();
    test_empty_and_bad_frame();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
