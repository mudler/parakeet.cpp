// Unit test for pk::AsrCommitter with a fake transcriber (no model).
#include "asr_committer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace pk;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// The PCM value at each sample is its absolute time in seconds, so the fake
// knows where the buffer starts.
// Words start at `speech_from` seconds (silence before it).
static Transcriber fake(double speech_from = 0.0) {
    return [speech_from](const std::vector<float>& pcm) {
        std::vector<Word> w;
        if (pcm.empty()) return w;
        const double t0 = pcm[0];
        const double dur = pcm.size() / 16000.0;
        const int first = (int)std::ceil(std::max(t0, speech_from) / 0.5 - 1e-6);
        for (int i = first;; ++i) {
            const double s = i * 0.5 + 0.05 - t0, e = i * 0.5 + 0.45 - t0;
            if (e > dur) break;
            w.push_back({"w" + std::to_string(i), (float)s, (float)e, 0.9f});
        }
        return w;
    };
}
static std::vector<float> timeline(double a, double b) {
    std::vector<float> x;
    for (long i = std::lround(a * 16000); i < std::lround(b * 16000); ++i) x.push_back((float)(i / 16000.0));
    return x;
}

static void test_min_window() {
    AsrCommitter c(fake());
    auto x = timeline(0, 3);
    c.push(x.data(), (int)x.size());
    CHECK(c.commit(3.0, false).empty());       // 3 s < 4 s window
    CHECK(c.commit_sec() == 0.0);
}

static void test_right_context_and_resume() {
    AsrCommitter c(fake());
    auto x = timeline(0, 5);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(5.0, false);
    // words must end <= 5 - 1 = 4 s: w0..w7 (w7 ends 3.95)
    CHECK(w.size() == 8);
    if (w.size() == 8) { CHECK(w.front().text == "w0"); CHECK(w.back().text == "w7"); }
    CHECK(std::fabs(c.commit_sec() - 3.95) < 1e-3);   // resumes after the last word
    auto y = timeline(5, 10);
    c.push(y.data(), (int)y.size());
    auto w2 = c.commit(10.0, false);
    CHECK(!w2.empty() && w2.front().text == "w8");     // no repeat, no gap
    CHECK(std::fabs(w2.front().start - 4.05f) < 1e-3); // absolute time
}

static void test_is_last_commits_all() {
    AsrCommitter c(fake());
    auto x = timeline(0, 2);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(2.0, true);
    CHECK(w.size() == 4);                              // w0..w3, no right-context hold
}

static void test_duplicate_word_dropped() {
    // A transcriber that hears the last committed word again at the new start.
    int calls = 0;
    AsrCommitter c([&](const std::vector<float>& pcm) {
        ++calls;
        std::vector<Word> w;
        if (calls == 1) { w = {{"hello", 0.5f, 1.0f, 0.9f}, {"there", 1.1f, 1.5f, 0.9f}}; }
        else { w = {{"there", 0.0f, 0.2f, 0.9f}, {"friend", 0.4f, 0.9f, 0.9f}}; }
        (void)pcm;
        return w;
    });
    std::vector<float> x(16000 * 5, 0.0f);
    c.push(x.data(), (int)x.size());
    auto a = c.commit(5.0, false);
    CHECK(a.size() == 2);
    c.push(x.data(), (int)x.size());
    auto b = c.commit(10.0, true);
    CHECK(b.size() == 1 && b[0].text == "friend");
}

// A word whose reported span straddles the cut (its end lands past the right
// context, so it does not commit) must not have the cut land exactly on its
// start: the onset margin must back off from it, and the word must come back
// intact, at its correct absolute times, once more audio gives it context.
static void test_onset_margin_backoff() {
    AsrCommitter c([](const std::vector<float>& pcm) {
        std::vector<Word> w;
        if (pcm.empty()) return w;
        const double t0 = pcm[0];
        const double dur = pcm.size() / 16000.0;
        const double abs_start = 3.8, abs_end = 4.4;
        if (abs_start >= t0 && abs_end <= t0 + dur)
            w.push_back({"onset", (float)(abs_start - t0), (float)(abs_end - t0), 0.9f});
        return w;
    });
    auto x = timeline(0, 5);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(5.0, false);
    CHECK(w.empty());                                  // straddles limit (4.0), so keep == 0
    CHECK(std::fabs(c.commit_sec() - 3.5) < 1e-3);      // 3.8 - kOnsetMargin(0.3), not 3.8
    // More audio gives the word its right context; it must return intact.
    auto y = timeline(5, 10);
    c.push(y.data(), (int)y.size());
    auto w2 = c.commit(10.0, true);
    CHECK(w2.size() == 1);
    if (w2.size() == 1) {
        CHECK(w2[0].text == "onset");
        CHECK(std::fabs(w2[0].start - 3.8f) < 1e-3);
        CHECK(std::fabs(w2[0].end - 4.4f) < 1e-3);
    }
}

// Non-speech: no words at all. The commit point must still advance and the
// buffer must not grow with the stream.
static void test_silence_released() {
    AsrCommitter c([](const std::vector<float>&) { return std::vector<Word>{}; });
    size_t max_buffered = 0;
    for (int s = 0; s < 20; ++s) {
        auto x = timeline(s, s + 1);
        c.push(x.data(), (int)x.size());
        CHECK(c.commit(s + 1.0, false).empty());
        max_buffered = std::max(max_buffered, c.buffered_samples());
    }
    CHECK(c.commit_sec() >= 15.0);                     // only the right context is held back
    CHECK(max_buffered <= (size_t)(5 * 16000));        // min window + one piece
    CHECK(c.buffered_samples() <= (size_t)(5 * 16000));
}

// Speech after a silent stretch commits every word once, at absolute times.
static void test_speech_after_silence() {
    AsrCommitter c(fake(10.0));
    std::vector<Word> all;
    for (int s = 0; s < 20; ++s) {
        auto x = timeline(s, s + 1);
        c.push(x.data(), (int)x.size());
        auto w = c.commit(s + 1.0, s == 19);
        all.insert(all.end(), w.begin(), w.end());
        CHECK(c.buffered_samples() <= (size_t)(5 * 16000));
    }
    CHECK(all.size() == 20);                           // w20..w39
    for (size_t i = 0; i < all.size(); ++i) {
        const int k = 20 + (int)i;
        CHECK(all[i].text == "w" + std::to_string(k));
        CHECK(std::fabs(all[i].start - (k * 0.5 + 0.05)) < 1e-3);
        CHECK(std::fabs(all[i].end - (k * 0.5 + 0.45)) < 1e-3);
    }
}

// PCM that is loud everywhere except in [q0, q1) (absolute seconds), with the
// sample's absolute time recoverable from `times` for the fakes below.
static std::vector<float> loud_except(double a, double b, double q0, double q1) {
    std::vector<float> x;
    for (long i = std::lround(a * 16000); i < std::lround(b * 16000); ++i) {
        const double t = i / 16000.0;
        x.push_back(t >= q0 && t < q1 ? 0.0f : ((i & 1) ? 0.3f : -0.3f));
    }
    return x;
}

// quiet_resume: the next window opens in the pause after the last committed
// word, not at that word's (late) end.
static void test_quiet_resume() {
    double next_window_start = -1.0;
    int calls = 0;
    AsrCommitterOpts o;
    o.quiet_resume_min_gap_sec = 0.5;
    double base = 0.0;   // stream time of the window the fake sees
    AsrCommitter c([&](const std::vector<float>& pcm) {
        std::vector<Word> w;
        ++calls;
        if (calls == 1) {
            // "a" ends at 1.0 (late: the sound goes on to 1.2), pause
            // 1.2-2.0, "b" heard from 2.0 and still open at the cut.
            w = {{"a", 0.5f, 1.0f, 0.9f}, {"b", 2.0f, 4.6f, 0.9f}};
        } else {
            next_window_start = base;
        }
        (void)pcm;
        return w;
    }, 4.0, 1.0, o);
    auto x = loud_except(0, 5, 1.2, 2.0);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(5.0, false);
    CHECK(w.size() == 1 && w[0].text == "a");
    // Resumes inside the 1.2-2.0 pause, not at 1.0.
    CHECK(c.commit_sec() >= 1.2 - 1e-9 && c.commit_sec() <= 2.0 - 0.04 + 1e-9);
    base = c.commit_sec();
    auto y = loud_except(5, 6, 0, 0);
    c.push(y.data(), (int)y.size());
    c.commit(6.0, true);
    CHECK(calls == 2 && next_window_start >= 1.2 - 1e-9);
}

// Without the option the resume point is the last committed word end.
static void test_quiet_resume_off_by_default() {
    AsrCommitter c([](const std::vector<float>&) {
        return std::vector<Word>{{"a", 0.5f, 1.0f, 0.9f}, {"b", 2.0f, 4.6f, 0.9f}};
    });
    auto x = loud_except(0, 5, 1.2, 2.0);
    c.push(x.data(), (int)x.size());
    c.commit(5.0, false);
    CHECK(std::fabs(c.commit_sec() - 1.0) < 1e-6);
}

// empty_retry_pad_sec: a loud window that gives no word is transcribed again
// with silence before it, and the words come back at the real-audio times.
static void test_empty_retry() {
    int calls = 0;
    AsrCommitterOpts o;
    o.empty_retry_pad_sec = 0.16;
    AsrCommitter c([&](const std::vector<float>& pcm) {
        ++calls;
        std::vector<Word> w;
        // This fake only hears speech when the window starts with silence.
        if (!pcm.empty() && pcm[0] == 0.0f) w = {{"hi", 0.16f + 1.0f, 0.16f + 1.4f, 0.9f}};
        return w;
    }, 4.0, 1.0, o);
    auto x = loud_except(0, 3, 0, 0);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(3.0, true);
    CHECK(calls == 2);
    CHECK(w.size() == 1);
    if (w.size() == 1) {
        CHECK(std::fabs(w[0].start - 1.0f) < 1e-4);
        CHECK(std::fabs(w[0].end - 1.4f) < 1e-4);
    }
}

// The retry does not run on a silent window, nor when it is off.
static void test_empty_retry_skips_silence_and_default() {
    int calls = 0;
    AsrCommitterOpts o;
    o.empty_retry_pad_sec = 0.16;
    AsrCommitter c([&](const std::vector<float>&) { ++calls; return std::vector<Word>{}; }, 4.0, 1.0, o);
    std::vector<float> quiet(16000 * 3, 0.001f);
    c.push(quiet.data(), (int)quiet.size());
    c.commit(3.0, true);
    CHECK(calls == 1);
    int calls2 = 0;
    AsrCommitter d([&](const std::vector<float>&) { ++calls2; return std::vector<Word>{}; });
    auto x = loud_except(0, 3, 0, 0);
    d.push(x.data(), (int)x.size());
    d.commit(3.0, true);
    CHECK(calls2 == 1);
}

// No pause (the next word starts within the minimum gap): resume at the last
// committed word end, as without the option.
static void test_quiet_resume_needs_a_pause() {
    AsrCommitterOpts o;
    o.quiet_resume_min_gap_sec = 0.5;
    AsrCommitter c([](const std::vector<float>&) {
        return std::vector<Word>{{"a", 0.5f, 1.0f, 0.9f}, {"b", 1.3f, 4.6f, 0.9f}};
    }, 4.0, 1.0, o);
    auto x = loud_except(0, 5, 1.05, 1.3);
    c.push(x.data(), (int)x.size());
    c.commit(5.0, false);
    CHECK(std::fabs(c.commit_sec() - 1.0) < 1e-6);
}

// output_times changes the reported times only: the commit rule and the
// resume point use the transcriber's times.
static void test_output_times_only_reported() {
    AsrCommitterOpts o;
    o.output_times = [](std::vector<Word>& w) { for (Word& x : w) { x.start -= 0.2f; x.end -= 0.2f; } };
    AsrCommitter c([](const std::vector<float>&) {
        // "b" ends at 4.1, past the 4.0 limit: with the shifted times it would
        // commit, with the transcriber's times it must not.
        return std::vector<Word>{{"a", 0.5f, 1.0f, 0.9f}, {"b", 3.6f, 4.1f, 0.9f}};
    }, 4.0, 1.0, o);
    auto x = timeline(0, 5);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(5.0, false);
    CHECK(w.size() == 1);
    if (w.size() == 1) {
        CHECK(std::fabs(w[0].start - 0.3f) < 1e-4);
        CHECK(std::fabs(w[0].end - 0.8f) < 1e-4);
    }
    CHECK(std::fabs(c.commit_sec() - 1.0) < 1e-6);
}

// A loud window with a quiet stretch in [q0, q1) of its own time.
static std::vector<float> loud_window(double len, double q0, double q1) {
    std::vector<float> x;
    for (long i = 0; i < std::lround(len * 16000); ++i) {
        const double t = i / 16000.0;
        x.push_back(t >= q0 && t < q1 ? 0.0f : ((i & 1) ? 0.3f : -0.3f));
    }
    return x;
}

// Split: a fake that only decodes windows of 3.5 s or less (one word, 0.5 s
// into the window). A 6 s window: full, padded, then the two parts, cut in
// the quiet stretch; the tail word is shifted by the cut.
static void test_split_once() {
    int calls = 0;
    std::vector<size_t> lens;
    AsrCommitterOpts o;
    o.empty_retry_pad_sec = 0.16;
    o.empty_split_min_sec = 2.0;
    AsrCommitter c([&](const std::vector<float>& pcm) {
        ++calls;
        lens.push_back(pcm.size());
        std::vector<Word> w;
        if (pcm.size() <= 56000) w.push_back({"p" + std::to_string(calls), 0.5f, 0.9f, 0.9f});
        return w;
    }, 4.0, 1.0, o);
    auto x = loud_window(6.0, 2.9, 3.2);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(6.0, true);
    CHECK(calls == 4);
    CHECK(w.size() == 2);
    if (lens.size() == 4) {
        const double cut = lens[2] / 16000.0;
        CHECK(cut >= 2.9 - 1e-9 && cut <= 3.2 - 0.04 + 1e-9);   // in the quiet stretch
        CHECK(lens[2] + lens[3] == x.size());
        if (w.size() == 2) {
            CHECK(std::fabs(w[0].start - 0.5f) < 1e-4);
            CHECK(std::fabs(w[1].start - (float)(cut + 0.5)) < 1e-3);   // shifted by the cut
            CHECK(std::fabs(w[1].end - (float)(cut + 0.9)) < 1e-3);
        }
    }
}

// The split never recurses: a fake that never decodes costs 4 calls and at
// most 3 times the window audio (plus the pad), whatever the window length.
static void test_retry_bound() {
    for (double len : {4.0, 8.0, 17.6, 30.0}) {
        int calls = 0;
        size_t samples = 0;
        AsrCommitterOpts o;
        o.empty_retry_pad_sec = 0.16;
        o.empty_split_min_sec = 2.0;
        AsrCommitter c([&](const std::vector<float>& pcm) {
            ++calls; samples += pcm.size(); return std::vector<Word>{};
        }, 4.0, 1.0, o);
        auto x = loud_window(len, 0, 0);
        c.push(x.data(), (int)x.size());
        c.commit(len, true);
        CHECK(calls == 4);
        CHECK(samples <= 3 * x.size() + 2560);
    }
}

// A part shorter than the minimum is not made: a 3 s window with 2 s parts
// cannot be split, so only the plain and the padded calls run.
static void test_split_min_part() {
    int calls = 0;
    AsrCommitterOpts o;
    o.empty_retry_pad_sec = 0.16;
    o.empty_split_min_sec = 2.0;
    AsrCommitter c([&](const std::vector<float>&) { ++calls; return std::vector<Word>{}; }, 4.0, 1.0, o);
    auto x = loud_window(3.0, 0, 0);
    c.push(x.data(), (int)x.size());
    c.commit(3.0, true);
    CHECK(calls == 2);
}

// Budget: when a window stayed empty after its retries, the next window over
// the same audio gets one call only; a window with words resets it.
static void test_retry_budget() {
    int calls = 0;
    bool speech = false;
    AsrCommitterOpts o;
    o.empty_retry_pad_sec = 0.16;
    o.empty_split_min_sec = 2.0;
    AsrCommitter c([&](const std::vector<float>&) {
        ++calls;
        std::vector<Word> w;
        if (speech) w.push_back({"w", 0.5f, 0.9f, 0.9f});
        return w;
    }, 4.0, 1.0, o);
    auto x = loud_window(5.0, 0, 0);
    c.push(x.data(), (int)x.size());
    c.commit(5.0, false);
    CHECK(calls == 4);                     // retried: plain, pad, two parts
    c.push(x.data(), (int)x.size());
    calls = 0;
    c.commit(10.0, false);                 // overlaps the empty one
    CHECK(calls == 1);
    speech = true;
    c.push(x.data(), (int)x.size());
    calls = 0;
    c.commit(15.0, false);                 // words again: budget reset
    CHECK(calls == 1);
    speech = false;
    c.push(x.data(), (int)x.size());
    calls = 0;
    c.commit(20.0, false);
    CHECK(calls == 4);
}

// A silent window that gives no word ends the empty stretch: a loud window
// that follows gets its retries again, even when it overlaps the stretch.
static void test_silent_window_resets_budget() {
    int calls = 0;
    AsrCommitterOpts o;
    o.empty_retry_pad_sec = 0.16;
    o.empty_split_min_sec = 2.0;
    AsrCommitter c([&](const std::vector<float>&) {
        ++calls;
        return std::vector<Word>();
    }, 1.0, 1.0, o);
    auto x = loud_window(5.0, 2.0, 5.0);   // loud, then quiet
    c.push(x.data(), (int)x.size());
    c.commit(5.0, false);
    CHECK(calls == 4);                     // loud and empty: retried
    std::vector<float> quiet(24000, 0.0f);
    c.push(quiet.data(), (int)quiet.size());
    calls = 0;
    c.commit(6.0, false);                  // silent window: one call, no retry
    CHECK(calls == 1);
    auto loud = loud_window(3.0, 0, 0);
    c.push(loud.data(), (int)loud.size());
    calls = 0;
    c.commit(9.0, false);                  // loud again: a new stretch
    CHECK(calls == 4);
}

int main() {
    test_silent_window_resets_budget();
    test_split_once();
    test_retry_bound();
    test_split_min_part();
    test_retry_budget();
    test_quiet_resume();
    test_quiet_resume_needs_a_pause();
    test_output_times_only_reported();
    test_quiet_resume_off_by_default();
    test_empty_retry();
    test_empty_retry_skips_silence_and_default();
    test_silence_released();
    test_speech_after_silence();
    test_min_window();
    test_right_context_and_resume();
    test_is_last_commits_all();
    test_duplicate_word_dropped();
    test_onset_margin_backoff();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
